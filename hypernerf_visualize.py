import os
import shutil
import json
import numpy as np
import open3d as o3d
import random
import argparse

def create_camera_frustum(json_path, scale=0.5):
    """
    Create a camera frustum (as a LineSet) from the camera parameters in a given JSON file.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extrinsics: the camera position and orientation (camera-to-world matrix)
    position = np.array(data["position"])
    R = np.array(data["orientation"]).T  # Using the transpose for transformation
    # Intrinsics
    focal_length = data["focal_length"]
    principal_point = np.array(data["principal_point"])
    image_size = data["image_size"]
    h, w = image_size

    # Image plane corners in pixel coordinates
    corners_px = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32)
    
    # Compute rays in camera coordinate system: (u - cx, v - cy)/f, 1
    rays = (corners_px - principal_point) / focal_length
    rays = np.hstack([rays, np.ones((4, 1))])
    
    # Scale rays to obtain frustum corner coordinates
    frustum_corners = scale * rays
    origin = np.array([[0, 0, 0]])
    
    # Construct camera-to-world transformation matrix
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
    
    # Define lines: from the origin to each corner and connecting corners cyclically
    lines = []
    for i in range(1, 5):
        lines.append([0, i])
    for i in range(1, 5):
        lines.append([i, 1 if i == 4 else i+1])
    
    colors = [[1, 0, 0] for _ in lines]  # Red lines
    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(vertices)
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector(colors)
    return frustum

def read_camera_frustums(json_dir, scale=0.5):
    """
    Create a list of camera frustums from JSON files in a given directory.
    """
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

def read_camera_centers(json_dir):
    """
    Extract camera centers (positions) from JSON files in the given directory.
    """
    centers = []
    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])
    for json_file in json_files:
        path = os.path.join(json_dir, json_file)
        with open(path, 'r') as f:
            data = json.load(f)
            centers.append(np.array(data["position"]))
    return np.array(centers)

def create_camera_path(camera_centers):
    """
    Create a LineSet representing the camera path by connecting the camera centers in order.
    """
    if len(camera_centers) < 2:
        return None
    points = camera_centers.tolist()
    lines = [[i, i+1] for i in range(len(points)-1)]
    colors = [[0, 0, 1] for _ in lines]  # Blue lines
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def main(args):
    # Set base path for points.npy and the camera JSON folder
    base_path = args.path
    points_path = os.path.join(base_path, "points.npy")
    camera_json_dir = os.path.join(base_path, "camera")
    
    # Load point cloud from points.npy
    points = np.load(points_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Create camera frustums from JSON files and randomly sample if needed
    all_frustums = read_camera_frustums(camera_json_dir, scale=0.5)
    if len(all_frustums) > args.num_cameras:
        sampled_frustums = random.sample(all_frustums, args.num_cameras)
    else:
        sampled_frustums = all_frustums
    
    # Extract camera centers and create camera path LineSet
    camera_centers = read_camera_centers(camera_json_dir)
    camera_path = create_camera_path(camera_centers)
    
    # Build the list of geometries based on view_mode
    geometries = [pcd]
    if args.view_mode in ["frustums", "both"]:
        geometries += sampled_frustums
    if args.view_mode in ["path", "both"] and camera_path is not None:
        geometries.append(camera_path)
    
    # If save_dir is provided, save each geometry separately instead of visualizing immediately
    if args.save_dir:
        if os.path.exists(args.save_dir):
            shutil.rmtree(args.save_dir)
        os.makedirs(args.save_dir, exist_ok=True)
        # Save point cloud
        pcd_file = os.path.join(args.save_dir, "point_cloud.ply")
        o3d.io.write_point_cloud(pcd_file, pcd)
        # Save frustums if applicable
        if args.view_mode in ["frustums", "both"]:
            for i, frustum in enumerate(sampled_frustums):
                frustum_file = os.path.join(args.save_dir, f"frustum_{i}.ply")
                o3d.io.write_line_set(frustum_file, frustum)
        # Save camera path if applicable
        if args.view_mode in ["path", "both"] and camera_path is not None:
            path_file = os.path.join(args.save_dir, "camera_path.ply")
            o3d.io.write_line_set(path_file, camera_path)
    else:
        # Visualize the geometries
        o3d.visualization.draw_geometries(geometries,
                                          window_name="Point Cloud with Camera Visualization",
                                          width=800, height=600)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera Visualization using Open3D")
    parser.add_argument("--view_mode", type=str, default="both",
                        choices=["frustums", "path", "both"],
                        help="Visualization mode: 'frustums', 'path', or 'both'")
    parser.add_argument("--path", type=str, default="aleks-teapot",
                        help="Base path containing points.npy and the camera JSON folder")
    parser.add_argument("--num_cameras", type=int, default=10,
                        help="Number of cameras to randomly sample")
    parser.add_argument("--save_dir", type=str, default="",
                        help="Directory to save each geometry separately (Ply format)")
    args = parser.parse_args()
    main(args)