import numpy as np
import matplotlib.pyplot as plt

def print_depth_stats(path_colmap, path_metric):
    colmap_depth = np.load(path_colmap)
    metric_depth = np.load(path_metric)
    print(colmap_depth)
    print(metric_depth)
    mask = (colmap_depth > 0) & (metric_depth > 0)

    print("Colmap-aligned:")
    print(f"  Min: {colmap_depth[mask].min():.3f}, Max: {colmap_depth[mask].max():.3f}, Mean: {colmap_depth[mask].mean():.3f}")

    print("Metric-aligned:")
    print(f"  Min: {metric_depth[mask].min():.3f}, Max: {metric_depth[mask].max():.3f}, Mean: {metric_depth[mask].mean():.3f}")

    diff = np.abs(colmap_depth[mask] - metric_depth[mask])
    print("Difference:")
    print(f"  Mean Abs Diff: {diff.mean():.3f}, Median Abs Diff: {np.median(diff):.3f}, Max Abs Diff: {diff.max():.3f}")

print_depth_stats("/131_data/intern/gunhee/PhysTrack/New/MPM/pancake/flow3d_preprocessed/aligned_depth_anything_v2_test/0_001.npy", "/131_data/intern/gunhee/PhysTrack/New/MPM/pancake/flow3d_preprocessed/aligned_depth_anything_v2/0_001.npy")