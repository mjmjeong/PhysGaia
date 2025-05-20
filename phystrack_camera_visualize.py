import os
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

def extract_camera_positions(data, prefix=None):
    xs, ys, zs = [], [], []
    for frame in data.get("frames", []):
        fp = frame.get("file_path", "")
        if prefix and not fp.startswith(prefix):
            continue
        tm = frame.get("transform_matrix", [])
        if tm and len(tm) == 4 and all(len(row) == 4 for row in tm):
            xs.append(tm[0][3])
            ys.append(tm[1][3])
            zs.append(tm[2][3])
    return xs, ys, zs

def visualize_camera_trajectories(json_dir, train_json_name, outfile):
    train_path = os.path.join(json_dir, train_json_name)
    test_path  = os.path.join(json_dir, 'camera_info_test.json')

    if train_json_name == 'camera_info_train.json':
        train_cam_count = 2
    elif train_json_name == 'camera_info_train_multi.json':
        train_cam_count = 4
    else:
        raise ValueError(
            "train_json_name must be 'camera_info_train.json' or 'camera_info_train_multi.json'"
        )

    with open(train_path, 'r') as f:
        train_data = json.load(f)
    with open(test_path, 'r') as f:
        test_data  = json.load(f)

    train_prefixes = ["./train/0_", "./train/3_"]
    if train_cam_count == 4:
        train_prefixes += ["./train/4_", "./train/5_"]
    test_prefixes = ["./test/1_", "./test/2_"]

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Train cams
    colors  = ['blue', 'red', 'green', 'orange']
    markers = ['o', '^', 's', 'd']
    train_markersize = 2
    for i, pre in enumerate(train_prefixes):
        x, y, z = extract_camera_positions(train_data, pre)
        ax.plot(x, z, y,
                label=f"Train Cam {i}",
                color=colors[i],
                marker=markers[i],
                linestyle='-',
                markersize=train_markersize)

    # Test cams (크기 크게)
    test_colors     = ['magenta', 'cyan']
    test_markers    = ['*', 'P']
    test_markersize = 25
    for i, pre in enumerate(test_prefixes):
        x, y, z = extract_camera_positions(test_data, pre)
        ax.plot(x, z, y,
                label=f"Test Cam {i+1}",
                color=test_colors[i],
                marker=test_markers[i],
                linestyle='--',
                markersize=test_markersize)

    ax.set_xlabel('X Position')
    ax.set_ylabel('Z Position')
    ax.set_zlabel('Y Position')
    ax.set_title('Camera Trajectories (Y-axis Up)')
    ax.legend(
        title="Cameras",
        loc='upper right',
        bbox_to_anchor=(1.15, 1),   # 플롯 오른쪽 밖으로 약간 빼기
        ncol=2,                     # 2열로 나누기
        scatterpoints=1,            # 마커 하나만 표시
        markerscale=0.7,            # 범례 마커 크기
        handlelength=2,             # 선 길이
        handletextpad=0.5,          # 핸들-텍스트 간 패딩
        columnspacing=1.2,          # 칼럼 간 간격
        labelspacing=0.8            # 항목 간 행 간격
    )

    plt.tight_layout()
    plt.savefig(outfile)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize train/test camera trajectories in 3D (Y-axis Up)"
    )
    parser.add_argument(
        "--json_dir", "-jd",
        required=True,
        help="공통 JSON 파일 디렉토리"
    )
    parser.add_argument(
        "--train_json_name", "-tjn",
        required=True,
        choices=['camera_info_train.json', 'camera_info_train_multi.json'],
        help="Train JSON 파일 이름 (2cam: camera_info_train.json / 4cam: camera_info_train_multi.json)"
    )
    parser.add_argument(
        "--outfile", "-o",
        default="camera_trajectories_3d.png",
        help="저장할 출력 이미지 파일명 (예: trajs.png)"
    )

    args = parser.parse_args()
    visualize_camera_trajectories(
        args.json_dir,
        args.train_json_name,
        args.outfile
    )