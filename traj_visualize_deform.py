#!/usr/bin/env python3
"""
Render sampled particle trajectories from a single .pt tensor,
overlaid on a background render image, using a static camera transform.
Sample only particles whose 3D trajectory variance falls within specified
percentile range and whose camera-space depth and lateral distance from
the optical axis are within given thresholds. Optionally accelerate
projection with CUDA. Markers are drawn at the end of each trajectory.
"""

import os
import argparse
import numpy as np
import cv2
import random
import torch
import json


def parse_cam_matrix(s: str) -> np.ndarray:
    """Parse semicolon-delimited string into a 4×4 NumPy array."""
    rows = s.split(';')
    mat = [list(map(float, row.split(','))) for row in rows]
    if len(mat) != 4 or any(len(r) != 4 for r in mat):
        raise ValueError("Expected 4×4 matrix, e.g. '1,0,0,0;...;0,0,0,1'")
    return np.array(mat, dtype=np.float32)


def project_points_np(world_pts: np.ndarray, w2c: np.ndarray, K: np.ndarray):
    """Project points with NumPy: world_pts (N,3) → pix (N,2) and mask (N,)."""
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


def project_points_torch(world_pts: np.ndarray, w2c_t: torch.Tensor, K_t: torch.Tensor, device: torch.device):
    """Project points with PyTorch: world_pts (N,3) → pix (N,2) and mask (N,)."""
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
    parser = argparse.ArgumentParser(description='Overlay sampled particle trajectories')
    parser.add_argument('--pt_file',         required=True, help='Path to .pt file containing tensor [frame*2, N, 4]')
    parser.add_argument('--cam_matrix',      help='Manual 4×4 camera-to-world matrix string')
    parser.add_argument('--camera_json',     help='JSON file with camera params')
    parser.add_argument('--camera_entry',    required=True, help='Suffix of file_path for camera transform')
    parser.add_argument('--width',           type=int, default=640, help='Output image width')
    parser.add_argument('--height',          type=int, default=720, help='Output image height')
    parser.add_argument('--num_samples',     type=int, default=20, help='Number of trajectories to sample')
    parser.add_argument('--var_min_percentile', type=float, default=0.01, help='Lower percentile of trajectory variance')
    parser.add_argument('--var_max_percentile', type=float, default=0.05, help='Upper percentile of trajectory variance')
    parser.add_argument('--depth_min',       type=float, default=1e-6, help='Minimum camera-space depth')
    parser.add_argument('--depth_max',       type=float, default=float('inf'), help='Maximum camera-space depth')
    parser.add_argument('--max_radial_dist', type=float, default=float('inf'), help='Max lateral distance from optical axis')
    parser.add_argument('--use_cuda',        action='store_true', help='Enable GPU acceleration if available')
    parser.add_argument('--output_image',    required=True, help='Path to save the output image')
    parser.add_argument('--fov_horizontal',  type=float, help='Horizontal FOV (radians) if using --cam_matrix')
    args = parser.parse_args()

    # Device setup
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    if args.use_cuda and device.type != 'cuda':
        print('CUDA requested but not available, using CPU instead')

    # Load tensor and parse frames
    pt_data = torch.load(args.pt_file, map_location='cpu')
    if not isinstance(pt_data, torch.Tensor):
        raise RuntimeError(f'Expected a Tensor in {args.pt_file}, got {type(pt_data)}')
    sampled = pt_data[::2, :, :]
    #mask = sampled[:, :, 3] > 0.9
    sampled = sampled[:, :, :3]  # keep only the first 3 coordinates
    #sampled = sampled[mask]      # apply mask after slicing
    np_data = sampled.numpy()
    num_frames, num_particles, _ = np_data.shape
    frame_data = [{pid: np_data[f, pid] for pid in range(num_particles)} for f in range(num_frames)]
    all_pids = set(range(num_particles))
    print(f'Loaded {len(all_pids)} particles from {num_frames} frames (from PT)')

    # Camera extrinsics/intrinsics
    if args.cam_matrix:
        c2w = parse_cam_matrix(args.cam_matrix)
        c2w[:3, 1:3] *= -1
        w2c_np = np.linalg.inv(c2w)
        if args.fov_horizontal is None:
            parser.error('--fov_horizontal required with --cam_matrix')
        fov_x = args.fov_horizontal
    else:
        with open(args.camera_json, 'r') as f:
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

    # Intrinsics
    w, h = args.width, args.height
    focal = (h/2) / np.tan(fov_x/2)
    K_np = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]], dtype=np.float32)
    if device.type == 'cuda':
        w2c_t = torch.from_numpy(w2c_np).to(device)
        K_t   = torch.from_numpy(K_np).to(device)
    else:
        w2c_t = K_t = None

    # Filter by depth & radial
    first_fd = frame_data[0]
    filtered = [pid for pid, pt in first_fd.items() if
                (lambda cam_pt: args.depth_min <= cam_pt[2] <= args.depth_max and np.linalg.norm(cam_pt[:2]) <= args.max_radial_dist)
                ((w2c_np @ np.append(pt,1.))[:3])]
    print(f'After filtering: {len(filtered)} particles')

    # Trajectory variance
    pid_vars = [(pid, float(np.var([fd[pid] for fd in frame_data], axis=0).sum()))
                for pid in filtered if pid in frame_data[0]]
    pid_vars.sort(key=lambda x: x[1], reverse=True)
    n = len(pid_vars)
    low = int(n * args.var_min_percentile)
    high = int(n * args.var_max_percentile)
    low, high = max(0, low), min(n, max(low+1, high))
    window = [pid for pid,_ in pid_vars[low:high]] or [pid_vars[0][0]]
    print(f'Variance window [{low}:{high}]: {len(window)} particles')

    # Sampling
    sample_ids = window if args.num_samples >= len(window) else random.sample(window, args.num_samples)
    print(f'Sampling {len(sample_ids)} trajectories')
    trajectories = {pid: [] for pid in sample_ids}

    # Project & collect
    for i, fd in enumerate(frame_data, 1):
        pts = np.stack([fd[pid] for pid in sample_ids])
        pix, mask = (project_points_torch(pts, w2c_t, K_t, device)
                     if device.type=='cuda' else project_points_np(pts, w2c_np, K_np))
        for idx, pid in enumerate(sample_ids):
            trajectories[pid].append(tuple(pix[idx]) if mask[idx] else None)
        if i % 50 == 0 or i==num_frames:
            print(f'Processed {i}/{num_frames} frames')

    # Render
    bg = cv2.imread(os.path.join('render','test','00158.png'))
    output = cv2.resize(bg, (w,h)) if bg is not None else np.zeros((h,w,3),dtype=np.uint8)
    for pid,traj in trajectories.items():
        color = tuple(map(int, np.random.randint(0,255,3)))
        prev=None
        for pt in traj:
            if pt:
                if prev: cv2.line(output,prev,pt,color,2)
                prev=pt
            else:
                prev=None
        #if prev: cv2.drawMarker(output,prev,color,cv2.MARKER_CROSS,20,10)

    cv2.imwrite(args.output_image, output)
    print(f'Saved trajectory image to {args.output_image}')

if __name__ == '__main__':
    main()
