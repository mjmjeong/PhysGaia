#!/usr/bin/env python3
"""
Render sampled particle trajectories from a sequence of JSON frames,
overlaid on a background render image, using a static camera transform.
Sample only particles whose 3D trajectory variance falls within specified
percentile range and whose camera-space depth and lateral distance from
the optical axis are within given thresholds. Optionally accelerate
projection with CUDA. Markers are drawn at the end of each trajectory.
"""

import os
import glob
import json
import argparse
import numpy as np
import cv2
import random
import torch


def parse_cam_matrix(s: str) -> np.ndarray:
    """Parse semicolon-delimited string into a 4×4 NumPy array."""
    rows = s.split(';')
    mat = [list(map(float, row.split(','))) for row in rows]
    if len(mat) != 4 or any(len(r) != 4 for r in mat):
        raise ValueError("Expected 4×4 matrix, e.g. '1,0,0,0;...;0,0,0,1'")
    return np.array(mat, dtype=np.float32)


def load_frame(path: str) -> dict:
    """Load JSON file and return a dict mapping id→(x,y,z)."""
    with open(path, 'r') as f:
        data = json.load(f)
    return {int(item['id']): np.array(item['position'], dtype=np.float32)
            for item in data}


def project_points_np(world_pts: np.ndarray, w2c: np.ndarray, K: np.ndarray):
    """
    Project points with NumPy: world_pts (N,3) → pix (N,2) and mask (N,).
    """
    N = world_pts.shape[0]
    homog = np.concatenate([world_pts, np.ones((N,1), dtype=np.float32)], axis=1)
    cam_pts = (w2c @ homog.T).T
    zs = cam_pts[:, 2]
    mask = zs > 1e-6
    pix = np.zeros((N,2), dtype=np.int32)
    if mask.any():
        pts_cam = cam_pts[mask, :3] / zs[mask, None]
        uvw = (K @ pts_cam.T).T
        pix[mask] = np.round(uvw[:, :2]).astype(np.int32)
    return pix, mask


def project_points_torch(world_pts: np.ndarray,
                         w2c_t: torch.Tensor,
                         K_t: torch.Tensor,
                         device: torch.device):
    """
    Project points with PyTorch: world_pts (N,3) → pix (N,2) and mask (N,).
    """
    pts = torch.from_numpy(world_pts).to(device)
    N = pts.shape[0]
    homog = torch.cat([pts, torch.ones((N,1), device=device)], dim=1)
    cam_pts = (w2c_t @ homog.T).T
    zs = cam_pts[:, 2]
    mask = zs > 1e-6
    pix = torch.zeros((N,2), dtype=torch.int32, device=device)
    if mask.any():
        pts_cam = cam_pts[mask, :3] / zs[mask].unsqueeze(1)
        uvw = (K_t @ pts_cam.T).T
        pix_vals = torch.round(uvw[:, :2]).to(torch.int32)
        pix[mask] = pix_vals
    return pix.cpu().numpy(), mask.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(
        description='Overlay sampled particle trajectories')
    parser.add_argument('--input_dir',       required=True,
                        help='Base directory containing data')
    parser.add_argument('--object_name',     required=True,
                        help='Name of the particle object')
    parser.add_argument('--cam_matrix',
                        help='Manual 4×4 camera-to-world matrix string')
    parser.add_argument('--camera_json',
                        help='JSON file with camera params')
    parser.add_argument('--camera_entry',    required=True,
                        help='Suffix of file_path for camera transform')
    parser.add_argument('--width',           type=int, default=640,
                        help='Output image width')
    parser.add_argument('--height',          type=int, default=720,
                        help='Output image height')
    parser.add_argument('--num_samples',     type=int, default=20,
                        help='Number of trajectories to sample')
    parser.add_argument('--var_min_percentile', type=float, default=0.01,
                        help='Lower percentile of trajectory variance')
    parser.add_argument('--var_max_percentile', type=float, default=0.05,
                        help='Upper percentile of trajectory variance')
    parser.add_argument('--start_frame',     type=int,
                        help='First frame index to include')
    parser.add_argument('--end_frame',       type=int,
                        help='Last frame index to include')
    parser.add_argument('--depth_min',       type=float, default=1e-6,
                        help='Minimum camera-space depth')
    parser.add_argument('--depth_max',       type=float, default=np.inf,
                        help='Maximum camera-space depth')
    parser.add_argument('--max_radial_dist', type=float, default=float('inf'),
                        help='Max lateral distance from optical axis')
    parser.add_argument('--use_cuda',        action='store_true',
                        help='Enable GPU acceleration if available')
    parser.add_argument('--output_image',    required=True,
                        help='Path to save the output image')
    parser.add_argument('--fov_horizontal',  type=float,
                        help='Horizontal FOV (radians) if using --cam_matrix')
    args = parser.parse_args()

    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    if args.use_cuda and device.type != 'cuda':
        print('CUDA requested but not available, using CPU instead')

    # Depth thresholds
    depth_min = args.depth_min
    depth_max = args.depth_max

    # Load camera extrinsics and intrinsics
    if args.cam_matrix:
        c2w = parse_cam_matrix(args.cam_matrix)
        c2w[:3, 1:3] *= -1  # adjust coordinate system
        w2c_np = np.linalg.inv(c2w)
        if args.fov_horizontal is None:
            parser.error('--fov_horizontal required with --cam_matrix')
        fov_x = args.fov_horizontal
    else:
        cam_path = os.path.join(args.input_dir, args.camera_json)
        with open(cam_path, 'r') as f:
            cam_info = json.load(f)
        fov_x = float(cam_info['camera_angle_x'])
        w2c_np = None
        for fr in cam_info['frames']:
            if fr['file_path'].endswith(args.camera_entry):
                mat = np.array(fr['transform_matrix'], dtype=np.float32)
                mat[:3, 1:3] *= -1
                w2c_np = np.linalg.inv(mat)
                break
        if w2c_np is None:
            raise KeyError('No matching camera_entry in JSON')

    # Compute intrinsics
    w, h = args.width, args.height
    focal = (h/2) / np.tan(fov_x/2)
    K_np = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]], dtype=np.float32)

    if device.type == 'cuda':
        w2c_t = torch.from_numpy(w2c_np).to(device)
        K_t   = torch.from_numpy(K_np).to(device)
    else:
        w2c_t = K_t = None

    # Load frames
    part_dir = os.path.join(args.input_dir, 'particles', args.object_name)
    files = sorted(glob.glob(os.path.join(part_dir, 'particles_frame_*.json')))
    def get_index(p): return int(os.path.basename(p).split('_')[-1].split('.')[0])
    frames = [p for p in files
              if (args.start_frame is None or get_index(p) >= args.start_frame)
              and (args.end_frame   is None or get_index(p) <= args.end_frame)]

    frame_data = [load_frame(p) for p in frames]
    all_pids = set(pid for fd in frame_data for pid in fd.keys())
    print(f'Loaded {len(all_pids)} total particles from {len(frames)} frames')

    # Filter by depth and radial distance
    first_fd = frame_data[0]
    front_pids = []
    for pid, pt in first_fd.items():
        if device.type == 'cuda':
            cam_pt = (w2c_t @ torch.tensor([*pt, 1.], device=device))[:3].cpu().numpy()
        else:
            cam_pt = (w2c_np @ np.append(pt, 1.))[:3]
        z = cam_pt[2]
        radial = np.linalg.norm(cam_pt[:2])
        if depth_min <= z <= depth_max and radial <= args.max_radial_dist:
            front_pids.append(pid)
    filtered_pids = set(front_pids)
    print(f'After filtering depth[{depth_min},{depth_max}] and radial<={args.max_radial_dist}: {len(filtered_pids)} particles')

    # Compute trajectory variance
    pid_variances = []
    for pid in filtered_pids:
        pts = [fd[pid] for fd in frame_data if pid in fd]
        if len(pts) >= 2:
            arr = np.stack(pts)
            var = float(np.var(arr, axis=0).sum())
            pid_variances.append((pid, var))
    print(f'Computed variance for {len(pid_variances)} candidates')

    # Select window by variance percentile
    pid_variances.sort(key=lambda x: x[1], reverse=True)
    n = len(pid_variances)
    low = int(n * args.var_min_percentile)
    high = int(n * args.var_max_percentile)
    low = max(0, min(low, n-1))
    high = max(low+1, min(high, n))
    window = [pid for pid, _ in pid_variances[low:high]]
    print(f'Window indices [{low}:{high}] -> {len(window)} particles')
    if not window:
        window = [pid_variances[0][0]]

    # Sample trajectories
    sample_ids = window if args.num_samples >= len(window) else random.sample(window, args.num_samples)
    print(f'Sampling {len(sample_ids)} of {len(window)} candidates')
    trajectories = {pid: [] for pid in sample_ids}

    # Project and collect
    for i, fd in enumerate(frame_data, 1):
        pts, hits = [], []
        for pid in sample_ids:
            if pid in fd:
                pts.append(fd[pid]); hits.append(pid)
        if pts:
            arr = np.stack(pts)
            if device.type == 'cuda':
                pix, mask = project_points_torch(arr, w2c_t, K_t, device)
            else:
                pix, mask = project_points_np(arr, w2c_np, K_np)
            idx = 0
            for pid in sample_ids:
                if pid in hits:
                    trajectories[pid].append(
                        (int(pix[idx][0]), int(pix[idx][1])) if mask[idx] else None
                    )
                    idx += 1
                else:
                    trajectories[pid].append(None)
        else:
            for pid in sample_ids:
                trajectories[pid].append(None)
        if i % 50 == 0 or i == len(frames):
            print(f'Processed {i}/{len(frames)} frames')

    # Render and save image
    bg_path = os.path.join(args.input_dir, 'render', 'test/00158.png')
    bg = cv2.imread(bg_path)
    if bg is None:
        raise RuntimeError(f'Could not load background at {bg_path}')
    output = cv2.resize(bg, (w, h))

    for pid, traj in trajectories.items():
        color = tuple(int(c) for c in np.random.randint(0, 255, 3))
        prev = None
        for pt in traj:
            if pt is not None:
                if prev is not None:
                    cv2.line(output, prev, pt, color, 2)
                prev = pt
            else:
                prev = None
        if prev is not None:
            cv2.drawMarker(output, prev, color,
                           markerType=cv2.MARKER_CROSS,
                           markerSize=10, thickness=5)

    cv2.imwrite(args.output_image, output)
    print(f'Saved trajectory image to {args.output_image}')


if __name__ == '__main__':
    main()
