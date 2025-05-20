import numpy as np
import json
import os
from pathlib import Path

def read_npz_and_save_as_json(bases_file_path: str, coefs_file_path: str, out_dir: str) -> None:
    # Create output directory
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    bases      = np.load(bases_file_path)["transls"]
    coefs      = np.load(coefs_file_path)
    n_bases, n_frames, _ = bases.shape # (N_bases, N_frames, 3)
    n_pts, n_bases = coefs.shape # (N_particles, N_bases)
    print("Loaded", bases_file_path)
    print("Loaded", coefs_file_path)
    x = coefs @ bases[:, :, 0] 
    y = coefs @ bases[:, :, 1] 
    z = coefs @ bases[:, :, 2] 
    particles_data = np.stack([x, y, z], axis=-1) # (N_particles, N_frames, 3)
    
    print(f"  particles: {n_pts} · frames: {n_frames}")
    for f in range(n_frames):
        frame_records = [
            {"id": pid + 1, "position": particles_data[pid, f].tolist()}
            for pid in range(n_pts)
        ]
        if f == 0:
            print("  first frame:", frame_records[0])

        out_path = Path(out_dir) / f"particles_frame_{f+1:04d}.json"
        with out_path.open("w") as fp:
            json.dump(frame_records, fp, indent=2)   # all values are now Python types

        print("Wrote", out_path)


if __name__ == "__main__":
    bases_file   = "/131_data/intern/gunhee/PhysTrack/New/Pyro/smoke_box/test_smoke_box_results/motion_bases.npz"
    coefs_file  = "/131_data/intern/gunhee/PhysTrack/New/Pyro/smoke_box/test_smoke_box_results/motion_coefs.npy"
    output_dir = "./vis_track/particles/smoke"
    read_npz_and_save_as_json(bases_file, coefs_file, output_dir)