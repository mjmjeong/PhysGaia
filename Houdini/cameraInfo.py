import hou
import math
import json
import os
from pxr import Usd, UsdGeom

# Get the current node and its USD stage.
node = hou.pwd()
stage = node.editableStage()

frames_list = []
first_camera_angle_x = None

# Loop over camera numbers 1 to 11.
for num in range(1, 12):
    cam_name = f"camera{num}"
    camera_prim_path = f"/cameras/{cam_name}"
    camera_prim = stage.GetPrimAtPath(camera_prim_path)
    if not camera_prim:
        hou.ui.displayMessage("Camera prim not found: {}".format(camera_prim_path))
        continue

    # Create a USD Camera object.
    camera = UsdGeom.Camera(camera_prim)

    # Sample lens parameters at time=0 (static scene).
    focal_length = camera.GetFocalLengthAttr().Get(0)             # in millimeters
    vertical_aperture = camera.GetVerticalApertureAttr().Get(0)   # in millimeters

    # Compute horizontal field-of-view (in radians), then convert to degrees.
    fov_rad = 2 * math.atan((vertical_aperture / 2.0) / focal_length)
    first_camera_angle_x = fov_rad

    # Compute the camera's transformation matrix at time 0.
    xformable = UsdGeom.Xformable(camera_prim)
    matrix = xformable.ComputeLocalToWorldTransform(0)
    matrix = matrix.GetTranspose()

    # Convert the Gf.Matrix4d to a nested Python list.
    matrix_list = [list(matrix[i]) for i in range(4)]

    # Build the frame dictionary (without camera name).
    frame_dict = {
        "file_path": f"./train/r_{num:03d}",
        "rotation": 0.041887902047863905,
        "time": 0.0,
        "transform_matrix": matrix_list
    }
    frames_list.append(frame_dict)

# Build the final output dictionary with only camera_angle_x and frames.
output_data = {
    "camera_angle_x": first_camera_angle_x,
    "frames": frames_list
}

# Expand the $HIP environment variable and create the output file path.
hip_dir = hou.expandString("$HIP")
file_path = os.path.join(hip_dir, "camera_info.json")

# Write the JSON data to the file.
with open(file_path, "w") as f:
    json.dump(output_data, f, indent=4)

print("JSON file '{}' created successfully.".format(file_path))