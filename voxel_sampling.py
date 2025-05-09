#!/usr/bin/env python3
import os
import argparse

import torch
import numpy as np
import orjson as json  # high-performance JSON library


def write_ply(filename: str, points: torch.Tensor):
    """
    Write particle positions to a PLY file in ASCII format.
    """
    pts = points.cpu().numpy()
    with open(filename, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {pts.shape[0]}\n')
        f.write('property float x\nproperty float y\nproperty float z\nend_header\n')
        for x, y, z in pts:
            f.write(f'{x:.6f} {y:.6f} {z:.6f}\n')
    print(f"Wrote PLY: {filename} with {pts.shape[0]} points")


def write_particles_json(filename: str, points: torch.Tensor):
    """
    Write particle positions to a JSON file using compact, fast serialization.
    """
    pts = points.cpu().tolist()
    data = [{"id": i, "position": p} for i, p in enumerate(pts)]
    b = json.dumps(data)
    with open(filename, 'wb') as f:
        f.write(b)
    print(f"Wrote JSON: {filename} with {len(data)} particles")


def load_velocity_field(path: str, device: torch.device):
    """
    Load a JSON velocity field and metadata, vectorizing the inner loop.
    """
    print(f"Loading velocity field: {path}")
    with open(path, 'rb') as f:
        data = json.loads(f.read())
    resolution = tuple(data['grid_info']['resolution'])
    voxel_size = torch.tensor(data['grid_info']['voxelSize'], device=device)
    voxels = data['voxels']
    idxs = np.array([[v['i'], v['j'], v['k']] for v in voxels], dtype=np.int64)
    vels = np.array([v['vel'] for v in voxels], dtype=np.float32)
    grid = torch.zeros((*resolution, 3), dtype=torch.float32, device=device)
    grid[idxs[:,0], idxs[:,1], idxs[:,2]] = torch.from_numpy(vels).to(device)
    origin = torch.tensor(voxels[0]['world_pos'], dtype=torch.float32, device=device)
    return grid, origin, voxel_size, resolution


def load_source_points(source_path: str, device: torch.device):
    """
    Load smoke source positions from a JSON file once at startup.
    """
    print(f"Loading smoke sources: {source_path}")
    with open(source_path, 'rb') as f:
        data = json.loads(f.read())
    positions = [entry['position'] for entry in data]
    return torch.tensor(positions, dtype=torch.float32, device=device)


def update_particles(pts: torch.Tensor, grid: torch.Tensor,
                     origin: torch.Tensor, voxel_size: torch.Tensor,
                     resolution: tuple, dt: float = 1.0):
    """
    Advance particle positions by one time step, but keep out-of-bounds particles static.
    """
    nx, ny, nz = resolution
    # Compute relative indices
    rel = (pts - origin.unsqueeze(0)) / voxel_size.unsqueeze(0)
    idxs = rel.round().long()
    # Determine in-bounds mask before clamping
    in_bounds = (
        (idxs[:,0] >= 0) & (idxs[:,0] < nx) &
        (idxs[:,1] >= 0) & (idxs[:,1] < ny) &
        (idxs[:,2] >= 0) & (idxs[:,2] < nz)
    )
    # Clamp for safe indexing
    idxs[:,0].clamp_(0, nx-1)
    idxs[:,1].clamp_(0, ny-1)
    idxs[:,2].clamp_(0, nz-1)
    # Gather velocities
    vels = grid[idxs[:,0], idxs[:,1], idxs[:,2]]
    # Compute new positions, but only advect in-bounds particles
    pts_next = pts.clone()
    pts_next[in_bounds] = pts[in_bounds] + vels[in_bounds] * dt
    return pts_next


def main(voxel_dir: str, source_path: str, output_dir: str,
         start_frame: int = 1, end_frame: int = 240,
         dt: float = 1/24, mode: str = 'ply'):
    print(f"Starting smoke simulation: frames {start_frame} to {end_frame}, mode={mode}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)

    source_pts = load_source_points(source_path, device)
    particles = None

    for frame in range(start_frame, end_frame+1):
        vel_path = os.path.join(voxel_dir, f'velVolume_{frame:04d}.json')
        grid, origin, voxel_size, resolution = load_velocity_field(vel_path, device)

        if frame == start_frame:
            particles = source_pts
        else:
            print(f"Frame {frame}: updating and sourcing new particles")
            particles = update_particles(particles, grid, origin, voxel_size, resolution, dt)
            particles = torch.cat([particles, source_pts], dim=0)

        if mode in ('ply','both'):
            write_ply(os.path.join(output_dir, f'sample_{frame:04d}.ply'), particles)
        if mode in ('json','both'):
            write_particles_json(os.path.join(output_dir, f'particles_frame_{frame:04d}.json'), particles)

    print("Smoke simulation completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Advect smoke particles with bounds-checking.')
    parser.add_argument('--voxel_dir', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=240)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--dt', type=float, default=1/24)
    parser.add_argument('--mode', choices=['ply','json','both'], default='ply')
    args = parser.parse_args()

    main(args.voxel_dir, args.source, args.out_dir,
         args.start, args.end, args.dt, args.mode)
