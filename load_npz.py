import numpy as np
import json
import os
from pathlib import Path

def read_npz_and_save_as_json(npz_file_path: str, out_dir: str) -> None:
    # Create output directory
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    data      = np.load(npz_file_path)
    for key in data.keys():
        print(f"  {key}: {data[key].shape} ({data[key].dtype})")
    xyz       = data["xyz"]                 # (N_particles, N_frames, 3)
    visibles = data["visibles"]           # (N_particles, N_frames)
    confidences = data["confidences"]       # (N_particles, N_frames)
    n_pts, n_frames, _ = xyz.shape

    print("Loaded", npz_file_path)
    print(f"  particles: {n_pts} · frames: {n_frames}")
    
    for f in range(n_frames):
        frame_records = [
            {"id": pid + 1, "position": xyz[pid, f].tolist(), "visible": bool(visibles[pid, f]), "confidence": float(confidences[pid, f])}
            for pid in range(n_pts)
        ]
        if f == 0:
            print("  first frame:", frame_records[0])

        out_path = Path(out_dir) / f"particles_frame_{f+1:04d}.json"
        with out_path.open("w") as fp:
            json.dump(frame_records, fp, indent=2)   # all values are now Python types

        print("Wrote", out_path)

if __name__ == "__main__":
    npz_file   = "/131_data/intern/gunhee/PhysTrack/New/Pyro/smoke_box/saved_tracks/tracks_3d_data.npz"
    output_dir = "./vis_track/particles/smoke"
    read_npz_and_save_as_json(npz_file, output_dir)