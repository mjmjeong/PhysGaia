#!/usr/bin/env python
import os
import shutil
import json
import numpy as np
import open3d as o3d
import random
import argparse

def create_camera_frustum(json_path, scale=0.5):
    with open(json_path, 'r') as f:
        data = json.load(f)
    position = np.array(data["position"])
    R = np.array(data["orientation"]).T
    focal_length = data["focal_length"]
    principal_point = np.array(data["principal_point"])
    image_size = data["image_size"]
    h, w = image_size
    corners_px = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    rays = (corners_px - principal_point) / focal_length
    rays = np.hstack([rays, np.ones((4, 1))])
    frustum_corners = scale * rays
    origin = np.array([[0, 0, 0]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = position
    def transform(points):
        pts_h = np.hstack([points, np.ones((points.shape[0], 1))])
        pts_world = (T @ pts_h.T).T
        return pts_world[:, :3]
    origin_world = transform(origin)[0]
    corners_world = transform(frustum_corners)
    vertices = np.vstack([origin_world, corners_world])
    lines = []
    for i in range(1, 5):
        lines.append([0, i])
    for i in range(1, 5):
        lines.append([i, 1 if i == 4 else i+1])
    colors = [[1, 0, 0] for _ in lines]  # Red for frustums
    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(vertices)
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector(colors)
    return frustum

def read_camera_frustums(json_dir, scale=0.5):
    frustums = []
    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])
    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        try:
            frustum = create_camera_frustum(json_path, scale)
            frustums.append(frustum)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    return frustums

def group_camera_centers_by_split(json_dir, dataset_path):
    """
    Group camera centers into two groups ("train" and "test") using dataset.json.
    dataset.json should contain "train_ids" and "val_ids" (used here as test_ids).
    Returns a dictionary: {"train": [center, ...], "test": [center, ...]}.
    """
    groups = {"train": [], "test": []}
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    train_ids = set(dataset.get("train_ids", []))
    test_ids = set(dataset.get("val_ids", []))
    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])
    for json_file in json_files:
        file_id = os.path.splitext(json_file)[0]
        full_path = os.path.join(json_dir, json_file)
        with open(full_path, 'r') as f:
            data = json.load(f)
        center = np.array(data["position"])
        if file_id in train_ids:
            groups["train"].append((json_file, center))
        elif file_id in test_ids:
            groups["test"].append((json_file, center))
    
    for key in groups:
        groups[key] = [center for (_, center) in sorted(groups[key], key=lambda x: x[0])]
    return groups

def create_camera_path(camera_centers, color):
    """
    Create a LineSet representing the camera path by connecting the camera centers in order.
    All lines are assigned the given color.
    """
    if len(camera_centers) < 2:
        return None
    points = camera_centers.tolist()
    lines = [[i, i+1] for i in range(len(points)-1)]
    colors = [color for _ in lines]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def main(args):
    base_path = args.path
    points_path = os.path.join(base_path, "points.npy")
    camera_json_dir = os.path.join(base_path, "camera")
    
    # Load point cloud
    points = np.load(points_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Create camera frustums and randomly sample if needed
    all_frustums = read_camera_frustums(camera_json_dir, scale=0.5)
    if len(all_frustums) > args.num_cameras:
        sampled_frustums = random.sample(all_frustums, args.num_cameras)
    else:
        sampled_frustums = all_frustums

    # If dataset.json exists, use it to split camera centers into train and test groups.
    dataset_path = os.path.join(base_path, "dataset.json")
    if os.path.exists(dataset_path):
        groups = group_camera_centers_by_split(camera_json_dir, dataset_path)
        train_centers = np.array(groups.get("train", []))
        test_centers = np.array(groups.get("test", []))
        camera_path_train = create_camera_path(train_centers, [0, 0, 1])  # Blue for train
        camera_path_test = create_camera_path(test_centers, [0, 1, 0])    # Green for test
    else:
        camera_centers = []
        json_files = sorted([f for f in os.listdir(camera_json_dir) if f.endswith('.json')])
        for json_file in json_files:
            full_path = os.path.join(camera_json_dir, json_file)
            with open(full_path, 'r') as f:
                data = json.load(f)
            camera_centers.append(np.array(data["position"]))
        camera_centers = np.array(camera_centers)
        camera_path = create_camera_path(camera_centers, [0, 0, 1])
        camera_path_train = camera_path
        camera_path_test = None

    # Build the list of geometries based on view_mode
    geometries = [pcd]
    if args.view_mode in ["frustums", "both"]:
        geometries += sampled_frustums
    if args.view_mode in ["path", "both"]:
        if camera_path_train is not None:
            geometries.append(camera_path_train)
        if camera_path_test is not None:
            geometries.append(camera_path_test)
    
    # If save_dir is provided, remove existing directory and create a new one, then save outputs.
    if args.save_dir:
        if os.path.exists(args.save_dir):
            shutil.rmtree(args.save_dir)
        os.makedirs(args.save_dir, exist_ok=True)
        
        pcd_file = os.path.join(args.save_dir, "point_cloud.ply")
        o3d.io.write_point_cloud(pcd_file, pcd)
        
        if args.view_mode in ["frustums", "both"]:
            for i, frustum in enumerate(sampled_frustums):
                frustum_file = os.path.join(args.save_dir, f"frustum_{i}.ply")
                o3d.io.write_line_set(frustum_file, frustum)
                
        if args.view_mode in ["path", "both"]:
            if camera_path_train is not None:
                train_path_file = os.path.join(args.save_dir, "camera_path_train.ply")
                o3d.io.write_line_set(train_path_file, camera_path_train)
            if camera_path_test is not None  and args.dataset == "hypernerf":
                test_path_file = os.path.join(args.save_dir, "camera_path_test.ply")
                o3d.io.write_line_set(test_path_file, camera_path_test)
    else:
        o3d.visualization.draw_geometries(geometries,
                                          window_name="Point Cloud with Camera Visualization",
                                          width=800, height=600)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera Visualization using Open3D")
    parser.add_argument("--view_mode", type=str, default="both",
                        choices=["frustums", "path", "both"],
                        help="Visualization mode: 'frustums', 'path', or 'both'")
    parser.add_argument("--path", type=str, default="aleks-teapot",
                        help="Base path containing points.npy, dataset.json, and the camera JSON folder")
    parser.add_argument("--num_cameras", type=int, default=10,
                        help="Number of cameras to randomly sample")
    parser.add_argument("--save_dir", type=str, default="",
                        help="Directory to save each geometry separately (PLY format)")
    parser.add_argument("--dataset", type=str, default="hypernerf",
                        choices=["hypernerf", "dycheck"],
                        help="Dataset type (used for camera path visualization)")
    args = parser.parse_args()
    main(args)