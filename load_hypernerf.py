#!/usr/bin/env python
import os
import glob
import open3d as o3d
import argparse

def load_geometries(load_dir):
    geometries = []
    # Load point cloud if it exists
    pcd_path = os.path.join(load_dir, "point_cloud.ply")
    if os.path.exists(pcd_path):
        pcd = o3d.io.read_point_cloud(pcd_path)
        geometries.append(pcd)
    # Load frustum files (assumed to be saved as frustum_*.ply)
    frustum_files = sorted(glob.glob(os.path.join(load_dir, "frustum_*.ply")))
    for f in frustum_files:
        frustum = o3d.io.read_line_set(f)
        geometries.append(frustum)
    # Load camera path if it exists
    path_file = os.path.join(load_dir, "camera_path.ply")
    if os.path.exists(path_file):
        camera_path = o3d.io.read_line_set(path_file)
        geometries.append(camera_path)
    return geometries

def main(args):
    geometries = load_geometries(args.load_dir)
    if not geometries:
        print(f"No geometries found in {args.load_dir}")
    else:
        o3d.visualization.draw_geometries(geometries,
                                          window_name="Loaded Geometries Visualization",
                                          width=800, height=600)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and Visualize Saved Geometries")
    parser.add_argument("--load_dir", type=str, required=True,
                        help="Directory from which to load the saved geometries")
    args = parser.parse_args()
    main(args)