#!/usr/bin/env python3
"""
Render sampled particle trajectories from a sequence of JSON frames,
overlaid on a background render image, using a static camera transform.
Sample only particles whose 2D pixel‐trajectory visibility and confidence segment is long enough.
Markers are drawn for the longest such segment of each sampled particle.
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
    """Parse semicolon‐delimited string into a 4×4 numpy array."""
    rows = s.split(';')
    mat = [list(map(float, row.split(','))) for row in rows]
    if len(mat) != 4 or any(len(r) != 4 for r in mat):
        raise ValueError("Expected 4×4 matrix, e.g. '1,0,0,0;...;0,0,0,1'")
    return np.array(mat, dtype=np.float32)


def load_frame(path: str) -> dict:
    """Load JSON file at `path` and return id→{'position', 'visible', 'confidence'}."""
    with open(path, 'r') as f:
        data = json.load(f)
    return {int(item['id']): {
                'position': np.array(item['position'], dtype=np.float32),
                'visible': bool(item.get('visible', False)),
                'confidence': float(item.get('confidence', 0.0))
            }
            for item in data}


def project_points_np(world_pts: np.ndarray, w2c: np.ndarray, K: np.ndarray):
    """
    Project with NumPy: world_pts (N,3) → pix (N,2) and mask (N,).
    """
    N = world_pts.shape[0]
    homog = np.concatenate([world_pts, np.ones((N,1), dtype=np.float32)], axis=1)
    cam_pts = (w2c @ homog.T).T
    zs = cam_pts[:,2]
    mask = zs > 1e-6
    pix = np.zeros((N,2), dtype=np.int32)
    if mask.any():
        pts_cam = cam_pts[mask,:3] / zs[mask,None]
        uvw = (K @ pts_cam.T).T
        pix[mask] = np.round(uvw[:,:2]).astype(np.int32)
    return pix, mask


def project_points_torch(world_pts: np.ndarray, w2c_t: torch.Tensor,
                         K_t: torch.Tensor, device: torch.device):
    """
    Project with PyTorch on `device`: world_pts (N,3) → pix (N,2) and mask.
    """
    pts = torch.from_numpy(world_pts).to(device)
    N = pts.shape[0]
    homog = torch.cat([pts, torch.ones((N,1), device=device)], dim=1)
    cam_pts = (w2c_t @ homog.T).T
    zs = cam_pts[:,2]
    mask = zs > 1e-6
    pix = torch.zeros((N,2), dtype=torch.int32, device=device)
    if mask.any():
        pts_cam = cam_pts[mask,:3] / zs[mask].unsqueeze(1)
        uvw = (K_t @ pts_cam.T).T
        pix_vals = torch.round(uvw[:,:2]).to(torch.int32)
        pix[mask] = pix_vals
    return pix.cpu().numpy(), mask.cpu().numpy()


def longest_visible_segment(vis_list):
    """Return (max_length, start_idx, end_idx) of longest True run in vis_list."""
    max_len = 0
    best_start = 0
    curr_len = 0
    curr_start = 0
    for i, v in enumerate(vis_list):
        if v:
            if curr_len == 0:
                curr_start = i
            curr_len += 1
            if curr_len > max_len:
                max_len = curr_len
                best_start = curr_start
        else:
            curr_len = 0
    return max_len, best_start, best_start + max_len


def main():
    parser = argparse.ArgumentParser(description='Overlay particle trajectories')
    parser.add_argument('--input_dir',          required=True)
    parser.add_argument('--object_name',        required=True)
    parser.add_argument('--cam_matrix')
    parser.add_argument('--camera_json')
    parser.add_argument('--camera_entry',       required=True)
    parser.add_argument('--width',               type=int, default=640)
    parser.add_argument('--height',              type=int, default=720)
    parser.add_argument('--sample_num',          type=int, default=20)
    parser.add_argument('--min_visible_frames',  type=int, default=5,
                        help='Minimum contiguous visible frames to include')
    parser.add_argument('--confidence_thresh',   type=float, default=0.5,
                        help='Minimum confidence to count as visible')
    parser.add_argument('--start_frame',         type=int)
    parser.add_argument('--end_frame',           type=int)
    parser.add_argument('--use_cuda',            action='store_true')
    parser.add_argument('--output_image',        required=True)
    parser.add_argument('--fovx',                type=float,
                        help='Vertical FOV (radians) if using --cam_matrix')
    args = parser.parse_args()

    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    if args.use_cuda and device.type != 'cuda':
        print('CUDA requested but not available – falling back to CPU')

    # load camera transform
    if args.cam_matrix:
        c2w = parse_cam_matrix(args.cam_matrix)
        c2w[:3,1:3] *= -1
        w2c_np = np.linalg.inv(c2w)
        if args.fovx is None:
            raise ValueError('--fovx required with --cam_matrix')
        fov_h = args.fovx
    else:
        cam_path = os.path.join(args.input_dir, args.camera_json)
        with open(cam_path, 'r') as f:
            cam_info = json.load(f)
        fov_h = float(cam_info['camera_angle_x'])
        w2c_np = None
        for fr in cam_info['frames']:
            if fr['file_path'].endswith(args.camera_entry):
                mat = np.array(fr['transform_matrix'], dtype=np.float32)
                mat[:3,1:3] *= -1
                w2c_np = np.linalg.inv(mat)
                break
        if w2c_np is None:
            raise KeyError('No matching camera_entry in JSON')

    # intrinsics
    w, h = args.width, args.height
    focal = (h/2) / np.tan(fov_h/2)
    K_np = np.array([[focal,0,w/2],[0,focal,h/2],[0,0,1]], dtype=np.float32)
    if device.type == 'cuda':
        w2c_t = torch.from_numpy(w2c_np).to(device)
        K_t   = torch.from_numpy(K_np).to(device)
    else:
        w2c_t = K_t = None

    # load frames
    part_dir = os.path.join(args.input_dir, 'particles', args.object_name)
    files = sorted(glob.glob(os.path.join(part_dir, 'particles_frame_*.json')))
    def idxf(p): return int(os.path.basename(p).split('_')[-1].split('.')[0])
    frames = [p for p in files
              if (not args.start_frame or idxf(p) >= args.start_frame)
              and (not args.end_frame   or idxf(p) <= args.end_frame)]

    # load all frames: each fd maps id->info dict
    frame_data_list = [load_frame(p) for p in frames]
    all_pids = set(pid for fd in frame_data_list for pid in fd.keys())

    # filter by camera-front (z>0) at first frame
    first_fd = frame_data_list[0]
    front_pids = []
    for pid, info in first_fd.items():
        pt = info['position']
        if device.type == 'cuda':
            cam_pt = (w2c_t @ torch.tensor([*pt,1.], device=device))[:3]
            z = cam_pt[2].item()
        else:
            cam_pt = w2c_np @ np.append(pt,1.)
            z = cam_pt[2]
        if z > 1e-6:
            front_pids.append(pid)
    all_pids = set(front_pids)

    # project & record uv and combined visibility per frame per pid
    trajectories = {pid: [] for pid in all_pids}
    vis_records  = {pid: [] for pid in all_pids}
    for i, fd in enumerate(frame_data_list):
        pts, pids, vis_flag = [], [], []
        for pid in all_pids:
            info = fd.get(pid)
            if info:
                pts.append(info['position'])
                pids.append(pid)
                # only visible if JSON visible, above confidence, and front-of-camera mask
                vis_flag.append(info['visible'] and info['confidence'] >= args.confidence_thresh)
        if pts:
            arr = np.stack(pts)
            if device.type == 'cuda':
                uv, mask = project_points_torch(arr, w2c_t, K_t, device)
            else:
                uv, mask = project_points_np(arr, w2c_np, K_np)
            for idx, pid in enumerate(pids):
                trajectories[pid].append(tuple(uv[idx]) if mask[idx] else None)
                vis_records[pid].append(vis_flag[idx] and mask[idx])
        else:
            for pid in all_pids:
                trajectories[pid].append(None)
                vis_records[pid].append(False)
        if (i+1) % 50 == 0 or (i+1) == len(frames):
            print(f'Processed {i+1}/{len(frames)} frames')

    # compute longest valid segment per pid
    pid_segs = []
    for pid, vis_list in vis_records.items():
        length, start, end = longest_visible_segment(vis_list)
        pid_segs.append((pid, length, start, end))
    # filter by threshold
    eligible = [pid for pid, length, *_ in pid_segs if length >= args.min_visible_frames]
    if not eligible:
        raise RuntimeError('No particles meet visibility & confidence threshold')
    # sample
    sample_ids = (eligible if args.sample_num >= len(eligible)
                  else random.sample(eligible, args.sample_num))
    print(f"Sampling {len(sample_ids)} particles from {len(eligible)} eligible")

    # render background
    bg_path = os.path.join(args.input_dir, 'render', f"{args.camera_entry}.png")
    bg = cv2.imread(bg_path)
    if bg is None:
        raise RuntimeError(f'Could not load background at {bg_path}')
    output = cv2.resize(bg, (w, h))

    # draw longest segment for each sampled pid
    for pid in sample_ids:
        _, length, start, end = next(x for x in pid_segs if x[0]==pid)
        traj = trajectories[pid][start:end]
        color = tuple(int(x) for x in np.random.randint(0,255,3))
        prev = None
        for pt in traj:
            if pt is not None:
                if prev is not None:
                    cv2.line(output, prev, pt, color, 2)
                prev = pt
            else:
                prev = None

    cv2.imwrite(args.output_image, output)
    print(f'Saved trajectory image to {args.output_image}')


if __name__ == '__main__':
    main()
