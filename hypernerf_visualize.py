import os
import json
import numpy as np
import open3d as o3d
import random
import argparse

def create_camera_frustum(json_path, scale=0.5):
    """
    Create a camera frustum (as a LineSet) from the camera parameters in a given JSON file.
    
    JSON example:
    {
      "orientation": [[...], [...], [...]],
      "position": [tx, ty, tz],
      "focal_length": f,
      "principal_point": [cx, cy],
      "image_size": [h, w],
      ...
    }
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extrinsics: the "position" field is the camera center,
    # and "orientation" is used as the camera-to-world transformation matrix (use transpose if needed)
    position = np.array(data["position"])
    R = np.array(data["orientation"]).T  # Use the transpose by default
    # Intrinsics
    focal_length = data["focal_length"]
    principal_point = np.array(data["principal_point"])
    image_size = data["image_size"]
    h, w = image_size

    # Image plane corners (pixel coordinates)
    corners_px = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32)
    
    # Rays in the camera coordinate system: (u - cx)/f, (v - cy)/f, 1
    rays = (corners_px - principal_point) / focal_length
    rays = np.hstack([rays, np.ones((4, 1))])
    
    # Multiply by the desired depth (scale) to compute the corner coordinates in the camera coordinate system
    frustum_corners = scale * rays
    origin = np.array([[0, 0, 0]])
    
    # Construct the camera-to-world transformation matrix
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
    
    # Define the connections (LineSet): from the origin to each corner and connecting the corners cyclically
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
    Extract only the camera centers (positions) from JSON files in the given directory.
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
    Create a LineSet representing the path by connecting the camera centers in order.
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

def create_xy_plane(size=1.0, num_lines=10):
    """
    Create a grid on the XY plane (at z=0) as a LineSet.
    
    Parameters:
        size (float): The half-length of the grid in both X and Y directions.
        num_lines (int): Number of grid divisions per axis.
    
    Returns:
        An Open3D LineSet representing the grid.
    """
    points = []
    lines = []
    # Horizontal lines (constant y)
    for i in range(num_lines + 1):
        y = -size + (2 * size * i) / num_lines
        points.append([-size, y, 0])
        points.append([size, y, 0])
        lines.append([len(points) - 2, len(points) - 1])
    # Vertical lines (constant x)
    for i in range(num_lines + 1):
        x = -size + (2 * size * i) / num_lines
        points.append([x, -size, 0])
        points.append([x, size, 0])
        lines.append([len(points) - 2, len(points) - 1])
    
    colors = [[0.5, 0.5, 0.5] for _ in lines]  # Gray grid lines
    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(points)
    grid.lines = o3d.utility.Vector2iVector(lines)
    grid.colors = o3d.utility.Vector3dVector(colors)
    return grid

def main(args):
    # Set the base path for points.npy and the camera JSON folder
    base_path = args.path
    points_path = os.path.join(base_path, "points.npy")
    camera_json_dir = os.path.join(base_path, "camera")
    
    # Load point cloud from points.npy
    points = np.load(points_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Create camera frustums from JSON files and randomly sample num_cameras of them
    all_frustums = read_camera_frustums(camera_json_dir, scale=0.5)
    if len(all_frustums) > args.num_cameras:
        sampled_frustums = random.sample(all_frustums, args.num_cameras)
    else:
        sampled_frustums = all_frustums
    
    # Extract camera centers and create a camera path LineSet
    camera_centers = read_camera_centers(camera_json_dir)
    camera_path = create_camera_path(camera_centers)
    
    # Determine which objects to visualize based on view_mode
    geometries = [pcd]
    if args.view_mode == "frustums":
        geometries += sampled_frustums
    elif args.view_mode == "path":
        if camera_path is not None:
            geometries.append(camera_path)
    elif args.view_mode == "both":
        geometries += sampled_frustums
        if camera_path is not None:
            geometries.append(camera_path)
    
    o3d.visualization.draw_geometries(geometries,
                                      window_name="Point Cloud with Camera Visualization",
                                      width=800, height=600,
                                      left=50, top=50,
                                      point_show_normal=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera Visualization using Open3D")
    parser.add_argument("--view_mode", type=str, default="both",
                        choices=["frustums", "path", "both"],
                        help="Visualization mode: 'frustums' (only camera frustums), 'path' (only camera path), or 'both'")
    parser.add_argument("--path", type=str, default="aleks-teapot",
                        help="Base path containing points.npy and the camera JSON folder")
    parser.add_argument("--num_cameras", type=int, default=10,
                        help="Number of cameras to randomly sample")
    args = parser.parse_args()
    main(args)