import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def extract_camera_positions(data, prefix=None):
    """Extract camera positions from JSON data."""
    x_positions = []
    y_positions = []
    z_positions = []

    for frame in data.get("frames", []):
        file_path = frame.get("file_path", "")
        if prefix and not file_path.startswith(prefix):
            continue

        transform_matrix = frame.get("transform_matrix", [])
        if transform_matrix and len(transform_matrix) == 4:
            x_positions.append(transform_matrix[0][3])
            y_positions.append(transform_matrix[1][3])
            z_positions.append(transform_matrix[2][3])

    return x_positions, y_positions, z_positions

def visualize_camera_trajectories(train_json_path, test_json_path):
    # Load JSON files
    with open(train_json_path, 'r') as f:
        train_data = json.load(f)
    with open(test_json_path, 'r') as f:
        test_data = json.load(f)

    # Extract positions for train data
    train_x, train_y, train_z = extract_camera_positions(train_data)

    # Extract positions for test data (camera 1 and camera 2)
    test_x1, test_y1, test_z1 = extract_camera_positions(test_data, prefix="./test/1_")
    test_x2, test_y2, test_z2 = extract_camera_positions(test_data, prefix="./test/2_")

    # Create a larger 3D plot
    fig = plt.figure(figsize=(12, 8))  # Increase the figure size
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectories with different colors
    ax.plot(train_x, train_y, train_z, label='Train Camera', color='blue', marker='o')
    ax.plot(test_x1, test_y1, test_z1, label='Test Camera 1', color='red', marker='^')
    ax.plot(test_x2, test_y2, test_z2, label='Test Camera 2', color='green', marker='s')

    # Set labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('Camera Trajectories Visualization')
    ax.legend()

    # Save the figure
    plt.savefig('camera_trajectories_3d.png')
    plt.show()

# Paths to the JSON files
train_json_path = '/131_data/intern/gunhee/PhysTrack/MPM/bouncing_balls/camera_info_train.json'
test_json_path = '/131_data/intern/gunhee/PhysTrack/MPM/bouncing_balls/camera_info_test.json'

# Call the function
visualize_camera_trajectories(train_json_path, test_json_path)