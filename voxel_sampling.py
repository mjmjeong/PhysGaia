import os
import json
import torch
import argparse


def write_ply(filename: str, points: torch.Tensor):
    """
    Write particle positions to a PLY file in ASCII format.

    Args:
        filename: Path to the output PLY file.
        points: (N, 3) tensor of point positions on CPU.
    """
    pts = points.cpu().numpy()
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {pts.shape[0]}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('end_header\n')
        for x, y, z in pts:
            f.write(f'{x:.6f} {y:.6f} {z:.6f}\n')
    print(f"Wrote PLY: {filename} with {pts.shape[0]} points")


def write_particles_json(filename: str, points: torch.Tensor):
    """
    Write particle positions to a JSON file.

    Args:
        filename: Path to the output JSON file.
        points: (N, 3) tensor of point positions on CPU.
    """
    pts = points.cpu().tolist()
    data = [{"id": i, "position": p} for i, p in enumerate(pts)]
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Wrote JSON: {filename} with {len(data)} particles")


def load_velocity_field(path: str, device: torch.device):
    """
    Load a JSON velocity field and metadata.

    Args:
        path: Path to a velVolume_XXXX.json file.
        device: torch device (CPU or CUDA).

    Returns:
        grid: (nx, ny, nz, 3) tensor of velocities.
        origin: (3,) tensor for the world position of voxel (0,0,0).
        voxel_size: (3,) tensor of voxel dimensions.
        resolution: Tuple of ints (nx, ny, nz).
    """
    print(f"Loading velocity field: {path}")
    with open(path, 'r') as f:
        data = json.load(f)

    resolution = tuple(data['grid_info']['resolution'])
    voxel_size = torch.tensor(data['grid_info']['voxelSize'], device=device)
    grid = torch.zeros((*resolution, 3), device=device)
    for v in data['voxels']:
        i, j, k = v['i'], v['j'], v['k']
        grid[i, j, k] = torch.tensor(v['vel'], device=device)
    origin = torch.tensor(data['voxels'][0]['world_pos'], device=device)
    return grid, origin, voxel_size, resolution


def load_source_points(source_path: str, device: torch.device):
    """
    Load smoke source positions from a JSON file.

    Args:
        source_path: Path to a smoke_source.json file.
        device: torch device.

    Returns:
        pts: (M,3) tensor of source positions.
    """
    print(f"Loading smoke sources: {source_path}")
    with open(source_path, 'r') as f:
        data = json.load(f)
    positions = [entry['position'] for entry in data]
    return torch.tensor(positions, dtype=torch.float32, device=device)


def update_particles(pts: torch.Tensor, grid: torch.Tensor,
                     origin: torch.Tensor, voxel_size: torch.Tensor,
                     resolution: tuple, dt: float = 1.0):
    """
    Advance particle positions by one time step using nearest-neighbor lookup.

    Args:
        pts: (N, 3) tensor of current particle positions.
        grid: (nx, ny, nz, 3) tensor of voxel velocities.
        origin: (3,) tensor of world position of voxel (0,0,0).
        voxel_size: (3,) tensor of voxel dimensions.
        resolution: Tuple (nx, ny, nz).
        dt: Time step for integration.

    Returns:
        pts_next: (N, 3) tensor of updated particle positions.
    """
    nx, ny, nz = resolution
    rel = (pts - origin.unsqueeze(0)) / voxel_size.unsqueeze(0)
    idxs = rel.round().long()
    idxs[:, 0].clamp_(0, nx - 1)
    idxs[:, 1].clamp_(0, ny - 1)
    idxs[:, 2].clamp_(0, nz - 1)
    vels = grid[idxs[:, 0], idxs[:, 1], idxs[:, 2]]
    return pts + vels * dt


def main(voxel_dir: str, source_path: str, output_dir: str,
         start_frame: int = 1, end_frame: int = 240,
         dt: float = 1/24, mode: str = 'ply'):
    """
    Sample and advect particles through voxel velocity fields,
    sourcing new particles each frame from a source JSON,
    and save output in PLY and/or JSON based on mode.

    Args:
        voxel_dir: Directory with velVolume_XXXX.json files.
        source_path: Path to smoke_source.json file.
        output_dir: Directory to save outputs.
        start_frame: First frame index.
        end_frame: Last frame index.
        dt: Time step interval between frames.
        mode: 'ply', 'json', or 'both'.
    """
    print(f"Starting smoke simulation: frames {start_frame} to {end_frame}, mode={mode}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)

    particles = None
    for frame in range(start_frame, end_frame + 1):
        vel_path = os.path.join(voxel_dir, f'velVolume_{frame:04d}.json')
        grid, origin, voxel_size, resolution = load_velocity_field(vel_path, device)

        if frame == start_frame:
            particles = load_source_points(source_path, device)
            if mode in ('ply', 'both'):
                write_ply(os.path.join(output_dir, 'sample_init.ply'), particles)
            if mode in ('json', 'both'):
                write_particles_json(os.path.join(output_dir, f'particles_frame_{frame:04d}.json'), particles)
        else:
            print(f"Frame {frame}: updating and sourcing new particles")
            particles = update_particles(particles, grid, origin,
                                         voxel_size, resolution, dt)
            new_pts = load_source_points(source_path, device)
            particles = torch.cat([particles, new_pts], dim=0)

            if mode in ('ply', 'both'):
                write_ply(os.path.join(output_dir, f'sample_{frame:04d}.ply'), particles)
            if mode in ('json', 'both'):
                write_particles_json(os.path.join(output_dir, f'particles_frame_{frame:04d}.json'), particles)

    print("Smoke simulation completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Advect and source smoke particles through voxel velocity fields.'
    )
    parser.add_argument('--voxel_dir', type=str, required=True,
                        help='Directory containing velVolume_XXXX.json files')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to smoke_source.json file')
    parser.add_argument('--start', type=int, default=1,
                        help='Start frame index')
    parser.add_argument('--end', type=int, default=240,
                        help='End frame index')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Directory to save output files')
    parser.add_argument('--dt', type=float, default=1/24,
                        help='Time step interval between frames')
    parser.add_argument('--mode', type=str, choices=['ply', 'json', 'both'],
                        default='ply', help="Output mode: 'ply', 'json', or 'both'")
    args = parser.parse_args()

    main(args.voxel_dir, args.source, args.out_dir,
         args.start, args.end, args.dt, args.mode)
