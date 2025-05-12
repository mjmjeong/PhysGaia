#!/usr/bin/env python3
"""
Render sampled particle trajectories from a sequence of JSON frames, using a static camera transform.

Each frame file should be named particles_frame_{:04d}.json and contain a list of
entries with:
  - "id": unique integer per particle
  - "position": [x, y, z] coordinates in world space

Usage:
    python traj_visualize.py \
        --input_dir /131_data/intern/gunhee/PhysTrack/New/Pyro/smoke_box \
        --object_name smoke \ 
        --camera_json camera_info_train.json \
        --camera_entry train/0_001 --sample_frame 1 \
        --output_image smoke_box.png \
        --start_frame 1 --end_frame 240 \
        --sample_num 10

Options:
  --cam_matrix    Manual 4×4 camera-to-world (c2w) matrix string; code will invert to world-to-camera
  --fovx          Vertical field-of-view in radians (based on image height); required with --cam_matrix
  --camera_json   NeRF-style JSON containing camera_angle_x and per-frame c2w transforms
  --camera_entry  Suffix in JSON"frames"[]."file_path" to select the desired c2w transform
"""

import os
import glob
import json
import argparse
import numpy as np
import cv2
import random


def parse_cam_matrix(s: str) -> np.ndarray:
    """
    Parse a semicolon-delimited string into a 4×4 NumPy array.
    Input is interpreted as camera-to-world (c2w) by default.
    Format: 'r1c1,r1c2,...;r2c1,...;...;r4c1,...'
    """
    rows = s.split(';')
    mat = [list(map(float, row.split(','))) for row in rows]
    if len(mat) != 4 or any(len(r) != 4 for r in mat):
        raise ValueError(
            "Expected 4×4 matrix, e.g. '1,0,0,0;0,1,0,0;0,0,1,0;0,0,0,1'"
        )
    return np.array(mat, dtype=np.float32)


def load_frame(path: str) -> dict:
    """
    Load particle positions from a JSON file.
    Returns a dict mapping particle_id -> 3D position array.
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return {int(item['id']): np.array(item['position'], dtype=np.float32)
            for item in data}


def project_points(world_pts: np.ndarray,
                   w2c: np.ndarray,
                   K: np.ndarray) -> tuple:
    """
    Transform and project 3D world points into 2D pixel coordinates.

    Args:
      world_pts: (N,3) array of world-space points
      w2c:       (4,4) world-to-camera extrinsic matrix
      K:         (3,3) camera intrinsics matrix

    Returns:
      pix_coords: (N,2) integer pixel positions
      mask:       (N,) boolean array, True if point is in front of camera
    """
    N = world_pts.shape[0]
    # convert to homogeneous coordinates
    homog = np.concatenate([world_pts, np.ones((N,1), dtype=np.float32)], axis=1)
    # camera-space points
    cam_pts = (w2c @ homog.T).T
    zs = cam_pts[:, 2]
    mask = zs > 1e-6
    pix = np.zeros((N,2), dtype=np.int32)
    if mask.any():
        pts_cam = cam_pts[mask, :3] / zs[mask,None]
        uvw = (K @ pts_cam.T).T
        pix[mask] = np.round(uvw[:, :2]).astype(np.int32)
    return pix, mask


def main():
    # --- Argument parsing ---
    parser = argparse.ArgumentParser(
        description='Render particle trajectories into a 2D image using OpenCV')
    parser.add_argument('--input_dir',  required=True,
                        help='Directory containing particles/<object_name>')
    parser.add_argument('--object_name', required=True,
                        help='Folder name under "particles" to load JSONs')
    parser.add_argument('--cam_matrix',
                        help='4×4 camera-to-world matrix string (semicolon-delimited)')
    parser.add_argument('--camera_json',
                        help='NeRF JSON with "camera_angle_x" and "frames" list')
    parser.add_argument('--camera_entry',
                        help='Frame suffix to select c2w transform in JSON')
    parser.add_argument('--width',  type=int, default=640,
                        help='Output image width (pixels)')
    parser.add_argument('--height', type=int, default=720,
                        help='Output image height (pixels)')
    parser.add_argument('--sample_num',  type=int, default=100,
                        help='Number of particles to randomly sample')
    parser.add_argument('--sample_frame', type=int,
                        help='Frame index to sample IDs from (default last frame)')
    parser.add_argument('--start_frame',  type=int,
                        help='Minimum frame index to include')
    parser.add_argument('--end_frame',    type=int,
                        help='Maximum frame index to include')
    parser.add_argument('--output_image', required=True,
                        help='Path to write the output PNG')
    parser.add_argument('--fovx', type=float,
                        help='Vertical FOV (radians) based on image height; required with --cam_matrix')
    args = parser.parse_args()

    # --- Load or parse camera transform ---
    if args.cam_matrix:
        # Input is camera-to-world (c2w); convert to world-to-camera (w2c)
        c2w = parse_cam_matrix(args.cam_matrix)
        # apply optional coordinate-system fix (flip Y/Z) if needed
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        if args.fovx is None:
            raise ValueError('--fovx is required when using --cam_matrix')
        fov_h = args.fovx
    elif args.camera_json and args.camera_entry:
        # JSON contains camera-to-world matrices for each frame
        json_path = os.path.join(args.input_dir, args.camera_json)
        with open(json_path, 'r') as f:
            cam_info = json.load(f)
        fov_h = float(cam_info.get('camera_angle_x'))
        w2c = None
        for frame in cam_info.get('frames', []):
            if frame.get('file_path', '').endswith(args.camera_entry):
                # JSON "transform_matrix" is c2w
                mat = np.array(frame['transform_matrix'], dtype=np.float32)
                # apply same handedness adjustment
                mat[:3, 1:3] *= -1
                w2c = np.linalg.inv(mat)
                break
        if w2c is None:
            raise KeyError(f'No matching camera_entry "{args.camera_entry}"')
        print(f'Loaded camera transform for entry: {args.camera_entry}')
    else:
        parser.error('Specify either --cam_matrix+--fovx or --camera_json+--camera_entry')

    # --- Build camera intrinsics ---
    w, h = args.width, args.height
    # Vertical FOV (based on image height)
    focal = (h / 2) / np.tan(fov_h / 2)
    K = np.array([[focal,     0, w / 2],
                  [    0, focal, h / 2],
                  [    0,     0,     1]], dtype=np.float32)

    # --- Gather frame JSONs ---
    json_folder = os.path.join(args.input_dir, 'particles', args.object_name)
    files = sorted(glob.glob(os.path.join(json_folder, 'particles_frame_*.json')))
    if not files:
        raise RuntimeError(f'No particle JSONs found in {json_folder}')
    # filter by frame range if specified
    def frame_index(path):
        return int(os.path.splitext(os.path.basename(path))[0].split('_')[-1])
    filtered = []
    for p in files:
        idx = frame_index(p)
        if args.start_frame and idx < args.start_frame: continue
        if args.end_frame   and idx > args.end_frame:   continue
        filtered.append(p)
    frames = filtered or files

    # --- Select sample frame and IDs ---
    if args.sample_frame:
        sample_path = next((p for p in frames if frame_index(p) == args.sample_frame), None)
        if sample_path is None:
            raise RuntimeError(f'Sample frame {args.sample_frame} not in selection')
    else:
        sample_path = frames[-1]
    all_ids = list(load_frame(sample_path).keys())
    sample_ids = (random.sample(all_ids, args.sample_num)
                  if args.sample_num < len(all_ids) else all_ids)
    trajectories = {pid: [] for pid in sample_ids}

    # --- Project each frame ---
    for i, path in enumerate(frames, start=1):
        data = load_frame(path)
        pts_world, hits = [], []
        for pid in sample_ids:
            if pid in data:
                pts_world.append(data[pid]); hits.append(pid)
        if pts_world:
            uv, mask = project_points(np.stack(pts_world), w2c, K)
            idx = 0
            for pid in sample_ids:
                if pid in hits:
                    trajectories[pid].append(tuple(uv[idx]) if mask[idx] else None)
                    idx += 1
                else:
                    trajectories[pid].append(None)
        else:
            for pid in sample_ids:
                trajectories[pid].append(None)
        if i % 50 == 0 or i == len(frames):
            print(f'Processed {i}/{len(frames)} frames')

    # --- Draw trajectories with OpenCV ---
    output = np.zeros((h, w, 3), dtype=np.uint8)
    for pid, traj in trajectories.items():
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        prev_pt = None
        for pt in traj:
            if pt is not None:
                if prev_pt is not None:
                    cv2.line(output, prev_pt, pt, color, 1)
                prev_pt = (int(pt[0]), int(pt[1]))
            else:
                prev_pt = None
    cv2.imwrite(args.output_image, output)
    print(f'Saved trajectory image to {args.output_image}')


if __name__ == '__main__':
    main()