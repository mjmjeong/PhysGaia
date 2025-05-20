#!/usr/bin/env python3
"""
Render sampled particle trajectories from multiple sequences of JSON frames,
overlayed on a background render image, using a static camera transform.
Sample only particles whose depth along the camera’s view direction
falls within the specified percentile range (--near_thresh, --far_thresh).
Optionally accelerate projection with CUDA.
Supports multiple `--object_name` arguments.
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
    """Parse a semicolon‐delimited string into a 4×4 NumPy array."""
    rows = s.split(';')
    mat = [list(map(float, row.split(','))) for row in rows]
    if len(mat) != 4 or any(len(r) != 4 for r in mat):
        raise ValueError("Expected 4×4 matrix, e.g. '1,0,0,0;...;0,0,0,1'")
    return np.array(mat, dtype=np.float32)


def load_frame(path: str) -> dict:
    """Load a JSON file of particle positions and return a dict id→(x,y,z)."""
    with open(path, 'r') as f:
        data = json.load(f)
    return {int(item['id']): np.array(item['position'], dtype=np.float32)
            for item in data}


def project_points_np(world_pts: np.ndarray, w2c: np.ndarray, K: np.ndarray):
    N = world_pts.shape[0]
    homog = np.concatenate([world_pts, np.ones((N,1), dtype=np.float32)], axis=1)
    cam_pts = (w2c @ homog.T).T
    zs = cam_pts[:,2]
    mask = zs > 1e-6
    pix = np.zeros((N,2), dtype=np.int32)
    if mask.any():
        pts_cam = cam_pts[mask, :3] / zs[mask,None]
        uvw = (K @ pts_cam.T).T
        pix[mask] = np.round(uvw[:,:2]).astype(np.int32)
    return pix, mask


def project_points_torch(world_pts: np.ndarray, w2c_t: torch.Tensor, K_t: torch.Tensor, device: torch.device):
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


def main():
    parser = argparse.ArgumentParser(description='Overlay particle trajectories')
    parser.add_argument('--input_dir',    required=True,
                        help='Root directory containing particles/ and render/')
    parser.add_argument('--object_name',  nargs='+', required=True,
                        help='One or more subfolders under particles/ to process')
    parser.add_argument('--cam_matrix',   help='Manual 4×4 camera-to-world matrix (semicolon‐delimited)')
    parser.add_argument('--camera_json',  help='NeRF‐style JSON file with camera_angle_x and frames list')
    parser.add_argument('--camera_entry', required=True,
                        help='Suffix to match frame entry in camera_json')
    parser.add_argument('--width',        type=int, default=640,
                        help='Output image width in pixels')
    parser.add_argument('--height',       type=int, default=720,
                        help='Output image height in pixels')
    parser.add_argument('--sample_num',   type=int, default=20,
                        help='Number of particles to randomly sample')
    parser.add_argument('--near_thresh',  type=float, default=0.45,
                        help='Lower percentile of depth to include (0–1)')
    parser.add_argument('--far_thresh',   type=float, default=0.5,
                        help='Upper percentile of depth to include (0–1)')
    parser.add_argument('--start_frame',  type=int, help='Minimum frame index to load')
    parser.add_argument('--end_frame',    type=int, help='Maximum frame index to load')
    parser.add_argument('--use_cuda',     action='store_true', help='Enable CUDA')
    parser.add_argument('--output_image', required=True, help='Path to save PNG')
    parser.add_argument('--fovx',         type=float, help='Vertical FOV when using --cam_matrix')

    args = parser.parse_args()

    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    if args.use_cuda and device.type!='cuda':
        print('[INFO] CUDA requested but not available; falling back to CPU')

    # load camera
    if args.cam_matrix:
        c2w = parse_cam_matrix(args.cam_matrix)
        c2w[:3,1:3] *= -1
        fov_h = args.fovx or (_ for _ in ()).throw(ValueError('--fovx required'))
    else:
        cam = json.load(open(os.path.join(args.input_dir, args.camera_json),'r'))
        fov_h = float(cam['camera_angle_x'])
        c2w = next((np.array(fr['transform_matrix'],dtype=np.float32) for fr in cam['frames'] if fr['file_path'].endswith(args.camera_entry)), None)
        if c2w is None:
            raise KeyError(f"No camera_entry '{args.camera_entry}' in JSON")
        c2w[:3,1:3] *= -1

    w2c_np = np.linalg.inv(c2w)
    w,h = args.width, args.height
    focal = (h/2)/np.tan(fov_h/2)
    K_np = np.array([[focal,0,w/2],[0,focal,h/2],[0,0,1]],dtype=np.float32)
    if device.type=='cuda':
        w2c_t = torch.from_numpy(w2c_np).to(device)
        K_t   = torch.from_numpy(K_np).to(device)
    else:
        w2c_t = K_t = None

    # camera origin & view dir
    cam_pos  = c2w[:3,3]
    view_dir = c2w[:3,2]

    all_trajs = {}
    def idxf(p): return int(os.path.basename(p).split('_')[-1].split('.')[0])

    for obj in args.object_name:
        part_dir = os.path.join(args.input_dir,'particles',obj)
        files = sorted(glob.glob(os.path.join(part_dir,'particles_frame_*.json')))
        if not files:
            print(f"[WARN] No JSON in {part_dir}, skip"); continue
        frames = [p for p in files if (not args.start_frame or idxf(p)>=args.start_frame)
                  and (not args.end_frame or idxf(p)<=args.end_frame)]
        data = [load_frame(p) for p in frames]

        # compute depths on first frame
        first = data[0]
        depths = np.array([np.dot(pos-cam_pos, view_dir) for pos in first.values()],dtype=np.float32)
        pids    = list(first.keys())
        # sort by depth ascending
        order   = np.argsort(depths)
        n       = len(order)
        if not (0<=args.near_thresh<args.far_thresh<=1):
            raise ValueError('thresholds must satisfy 0<=near<far<=1')
        low  = int(n*args.near_thresh)
        high = int(n*args.far_thresh)
        low  = max(0,min(low,n-1))
        high = max(low+1,min(high,n))
        window = [pids[i] for i in order[low:high]]
        if not window:
            window = pids
            print(f"[WARN] Empty window, using all {n} particles")

        # sampling
        if len(window)<=args.sample_num:
            sample_ids = window
        else:
            sample_ids = random.sample(window,args.sample_num)

        # collect trajectories
        trajs = {pid:[] for pid in sample_ids}
        for fd in data:
            pts, hit = [], []
            for pid in sample_ids:
                if pid in fd:
                    pts.append(fd[pid]); hit.append(pid)
            if pts:
                arr = np.stack(pts)
                if device.type=='cuda':
                    uv,mask = project_points_torch(arr,w2c_t,K_t,device)
                else:
                    uv,mask = project_points_np(arr,w2c_np,K_np)
                ii=0
                for pid in sample_ids:
                    if pid in hit:
                        trajs[pid].append((int(uv[ii][0]),int(uv[ii][1])) if mask[ii] else None)
                        ii+=1
                    else:
                        trajs[pid].append(None)
            else:
                for pid in sample_ids:
                    trajs[pid].append(None)
        all_trajs[obj]=trajs

    # render
    bg = cv2.imread(os.path.join(args.input_dir,'render',args.camera_entry+'.png'))
    out= cv2.resize(bg,(args.width,args.height))
    for trajs in all_trajs.values():
        for pid, traj in trajs.items():
            col=tuple(int(x) for x in np.random.randint(0,255,3))
            prev=None
            for pt in traj:
                if pt is not None:
                    if prev is not None:
                        cv2.line(out,prev,pt,col,2)
                    prev=pt
                else:
                    prev=None

    cv2.imwrite(args.output_image,out)
    print('[DONE]',args.output_image)

if __name__=='__main__':
    main()
