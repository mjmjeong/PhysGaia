import argparse
import fnmatch
import json
import os
import os.path as osp
from glob import glob
from typing import Literal

import cv2
import imageio.v2 as iio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Pipeline, pipeline
from scipy.ndimage import map_coordinates

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UINT16_MAX = 65535


models = {
    "depth-anything": "LiheYoung/depth-anything-large-hf",
    "depth-anything-v2": "depth-anything/Depth-Anything-V2-Large-hf",
}


def get_pipeline(model_name: str):
    pipe = pipeline(task="depth-estimation", model=models[model_name], device=DEVICE)
    print(f"{model_name} model loaded.")
    return pipe


def to_uint16(disp: np.ndarray):
    disp_min = disp.min()
    disp_max = disp.max()

    if disp_max - disp_min > np.finfo("float").eps:
        disp_uint16 = UINT16_MAX * (disp - disp_min) / (disp_max - disp_min)
    else:
        disp_uint16 = np.zeros(disp.shape, dtype=disp.dtype)
    disp_uint16 = disp_uint16.astype(np.uint16)
    return disp_uint16


def get_depth_anything_disp(
    pipe: Pipeline,
    img_file: str,
    ret_type: Literal["uint16", "float"] = "float",
):

    image = Image.open(img_file)
    disp = pipe(image)["predicted_depth"]
    if disp.ndim == 2:  # (H, W)
        disp = disp.unsqueeze(0).unsqueeze(0)
    elif disp.ndim == 3:  # (B, H, W)
        disp = disp.unsqueeze(1)

    disp = torch.nn.functional.interpolate(
        disp, size=image.size[::-1], mode="bicubic", align_corners=False
    )
    disp = disp.squeeze().cpu().numpy()
    if ret_type == "uint16":
        return to_uint16(disp)
    elif ret_type == "float":
        return disp
    else:
        raise ValueError(f"Unknown return type {ret_type}")


def save_disp_from_dir(
    model_name: str,
    img_dir: str,
    out_dir: str,
    matching_pattern: str = "*",
):
    img_files = sorted(glob(osp.join(img_dir, "*.jpg"))) + sorted(
        glob(osp.join(img_dir, "*.png"))
    )
    img_files = [
        f for f in img_files if fnmatch.fnmatch(osp.basename(f), matching_pattern)
    ]
    if osp.exists(out_dir) and len(glob(osp.join(out_dir, "*.png"))) == len(img_files):
        print(f"Raw {model_name} depth maps already computed for {img_dir}")
        return

    pipe = get_pipeline(model_name)
    os.makedirs(out_dir, exist_ok=True)
    for img_file in tqdm(img_files, f"computing {model_name} depth maps"):
        disp = get_depth_anything_disp(pipe, img_file, ret_type="uint16")
        out_file = osp.join(out_dir, osp.splitext(osp.basename(img_file))[0] + ".png")
        iio.imwrite(out_file, disp)


def get_camera_params_from_json(json_path, img_dir, use_width_for_focal=False):
    with open(json_path, 'r') as f:
        meta = json.load(f)

    angle_x = meta["camera_angle_x"]
    K_dict = {}  
    extrinsics_dict = {}  

    for frame in meta["frames"]:
        file_path = frame["file_path"]  
        image_name = os.path.basename(file_path) + ".png"
        image_path = os.path.join(img_dir, image_name)
        if not os.path.exists(image_path):
            continue

        img = iio.imread(image_path)
        h, w = img.shape[:2]

        if use_width_for_focal:
            fx = fy = 0.5 * w / np.tan(angle_x / 2)
        else:
            fx = fy = 0.5 * h / np.tan(angle_x / 2)
            
        cx, cy = w / 2, h / 2
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ])
        K_dict[image_name] = K 
        
        T_c2w = np.array(frame["transform_matrix"])
        flip = np.diag([1, -1, -1, 1])
        T_c2w_cv = T_c2w @ flip
        T_w2c = np.linalg.inv(T_c2w_cv)
        extrinsics_dict[image_name] = T_w2c  
        
    return K_dict, extrinsics_dict  


def align_monodepth_with_metric_depth(
    metric_depth_dir: str,
    input_monodepth_dir: str,
    output_monodepth_dir: str,
    matching_pattern: str = "*",
):
    print(
        f"Aligning monodepth in {input_monodepth_dir} with metric depth in {metric_depth_dir}"
    )
    mono_paths = sorted(glob(f"{input_monodepth_dir}/{matching_pattern}"))
    img_files = [osp.basename(p) for p in mono_paths]
    os.makedirs(output_monodepth_dir, exist_ok=True)
    if len(os.listdir(output_monodepth_dir)) == len(img_files):
        print(f"Founds {len(img_files)} files in {output_monodepth_dir}, skipping")
        return

    for f in tqdm(img_files):
        imname = os.path.splitext(f)[0]
        metric_path = osp.join(metric_depth_dir, imname + ".npy")
        mono_path = osp.join(input_monodepth_dir, imname + ".png")

        mono_disp_map = iio.imread(mono_path) / UINT16_MAX
        metric_disp_map = np.load(metric_path)
        ms_colmap_disp = metric_disp_map - np.median(metric_disp_map) + 1e-8
        ms_mono_disp = mono_disp_map - np.median(mono_disp_map) + 1e-8

        scale = np.median(ms_colmap_disp / ms_mono_disp)
        shift = np.median(metric_disp_map - scale * mono_disp_map)

        aligned_disp = scale * mono_disp_map + shift

        min_thre = min(1e-6, np.quantile(aligned_disp, 0.01))
        # set depth values that are too small to invalid (0)
        aligned_disp[aligned_disp < min_thre] = 0.0
        out_file = osp.join(output_monodepth_dir, imname + ".npy")
        np.save(out_file, aligned_disp)


def align_monodepth_with_colmap(
    sparse_dir: str,
    input_monodepth_dir: str,
    output_monodepth_dir: str,
    img_dir: str,
    camera_json_path: str = None,
    matching_pattern: str = "*",
):
    from pycolmap import SceneManager

    manager = SceneManager(sparse_dir)
    manager.load()

    cameras = manager.cameras
    images = manager.images
    points3D = manager.points3D
    point3D_id_to_point3D_idx = manager.point3D_id_to_point3D_idx

    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
    os.makedirs(output_monodepth_dir, exist_ok=True)

    images = [
        image
        for _, image in images.items()
        if fnmatch.fnmatch(image.name, matching_pattern)
    ]

    if camera_json_path:
        K_dict, extrinsics_dict = get_camera_params_from_json(camera_json_path, img_dir)
    for image in tqdm(images, "Aligning monodepth with colmap point cloud"):

        point3D_ids = image.point3D_ids
        point3D_ids = point3D_ids[point3D_ids != manager.INVALID_POINT3D]
        pts3d_valid = points3D[[point3D_id_to_point3D_idx[id] for id in point3D_ids]]  # type: ignore
        
        if camera_json_path:
            image_name = os.path.basename(image.name)
            K = K_dict[image_name]
            extrinsics = extrinsics_dict[image_name]
        else:
            K = cameras[image.camera_id].get_camera_matrix()
            rot = image.R()
            trans = image.tvec.reshape(3, 1)
            extrinsics = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)

        pts3d_valid_homo = np.concatenate(
            [pts3d_valid, np.ones_like(pts3d_valid[..., :1])], axis=-1
        )
        pts3d_valid_cam_homo = extrinsics.dot(pts3d_valid_homo.T).T
        pts2d_valid_cam = K.dot(pts3d_valid_cam_homo[..., :3].T).T
        pts2d_valid_cam = pts2d_valid_cam[..., :2] / pts2d_valid_cam[..., 2:3]
        colmap_depth = pts3d_valid_cam_homo[..., 2]

        monodepth_path = osp.join(
            input_monodepth_dir, osp.splitext(image.name)[0] + ".png"
        )
        mono_disp_map = iio.imread(monodepth_path) / UINT16_MAX

        colmap_disp = 1.0 / np.clip(colmap_depth, a_min=1e-6, a_max=1e6)
        mono_disp = cv2.remap(
            mono_disp_map,  # type: ignore
            pts2d_valid_cam[None, ...].astype(np.float32),
            None,  # type: ignore
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )[0]
        ms_colmap_disp = colmap_disp - np.median(colmap_disp) + 1e-8
        ms_mono_disp = mono_disp - np.median(mono_disp) + 1e-8

        scale = np.median(ms_colmap_disp / ms_mono_disp)
        shift = np.median(colmap_disp - scale * mono_disp)

        mono_disp_aligned = scale * mono_disp_map + shift

        min_thre = min(1e-6, np.quantile(mono_disp_aligned, 0.01))
        # set depth values that are too small to invalid (0)
        mono_disp_aligned[mono_disp_aligned < min_thre] = 0.0
        np.save(
            osp.join(output_monodepth_dir, image.name.split(".")[0] + ".npy"),
            mono_disp_aligned,
        )


def align_monodepth_with_colmap_ply(
    ply_path: str,
    input_monodepth_dir: str,
    output_monodepth_dir: str,
    img_dir: str,
    camera_json_path: str,
    matching_pattern: str = "*",
):
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)

    os.makedirs(output_monodepth_dir, exist_ok=True)

    img_files = sorted(glob(osp.join(img_dir, "*.jpg"))) + sorted(
        glob(osp.join(img_dir, "*.png"))
    )
    img_files = [
        osp.basename(f) for f in img_files if fnmatch.fnmatch(osp.basename(f), matching_pattern)
    ]

    if camera_json_path:
        K_dict, extrinsics_dict = get_camera_params_from_json(camera_json_path, img_dir)

    for image_name in tqdm(img_files, desc="Aligning monodepth with PLY point cloud"):
        image_path = osp.join(img_dir, image_name)
        if not osp.exists(image_path):
            continue

        img = iio.imread(image_path)
        h, w = img.shape[:2]
        K = K_dict[image_name]
        T_w2c = extrinsics_dict[image_name]

        points_h = np.concatenate([points, np.ones_like(points[:, :1])], axis=1)
        cam_points = (T_w2c @ points_h.T).T  

        z_vals = cam_points[:, 2]
        valid = z_vals > 1e-6
        cam_points = cam_points[valid]

        if len(cam_points) == 0:
            print(f"[Warning] No visible PLY points for {image_name}")
            continue

        pts2d = (K @ cam_points[:, :3].T).T
        pts2d = pts2d[:, :2] / pts2d[:, 2:3]  

        in_bounds = ((pts2d[:, 0] >= 0) & (pts2d[:, 0] < w) & 
                    (pts2d[:, 1] >= 0) & (pts2d[:, 1] < h))

        pts2d = pts2d[in_bounds]
        colmap_depth = cam_points[in_bounds, 2]
        colmap_disp = 1.0 / np.clip(colmap_depth, a_min=1e-6, a_max=1e6)

        monodepth_path = osp.join(input_monodepth_dir, osp.splitext(image_name)[0] + ".png")
        mono_disp_map = iio.imread(monodepth_path).astype(np.float32) / UINT16_MAX

        coords = np.stack([pts2d[:, 1], pts2d[:, 0]], axis=0) 
        mono_disp = map_coordinates(mono_disp_map, coords, order=1, mode='constant', cval=0.0)

        ms_colmap_disp = colmap_disp - np.median(colmap_disp) + 1e-8
        ms_mono_disp = mono_disp - np.median(mono_disp) + 1e-8

        scale = np.median(ms_colmap_disp / ms_mono_disp)
        shift = np.median(colmap_disp - scale * mono_disp)
        mono_disp_aligned = scale * mono_disp_map + shift

        min_thre = min(1e-6, np.quantile(mono_disp_aligned, 0.01))
        # set depth values that are too small to invalid (0)
        mono_disp_aligned[mono_disp_aligned < min_thre] = 0.0
        np.save(
            osp.join(output_monodepth_dir, osp.splitext(image_name)[0] + ".npy"),
            mono_disp_aligned
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="depth-anything",
        help="depth model to use, one of [depth-anything, depth-anything-v2]",
    )
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--out_raw_dir", type=str, required=True)
    parser.add_argument("--out_aligned_dir", type=str, default=None)
    parser.add_argument("--sparse_dir", type=str, default=None)
    parser.add_argument("--metric_dir", type=str, default=None)
    parser.add_argument("--camera_json_path", type=str, default=None)
    parser.add_argument("--ply_path", type=str, default=None)
    parser.add_argument("--matching_pattern", type=str, default="*")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    assert args.model in [
        "depth-anything",
        "depth-anything-v2",
    ], f"Unknown model {args.model}"
    save_disp_from_dir(
        args.model, args.img_dir, args.out_raw_dir, args.matching_pattern
    )

    if args.ply_path is not None and args.out_aligned_dir is not None:
        align_monodepth_with_colmap_ply(
            args.ply_path,
            args.out_raw_dir,
            args.out_aligned_dir,
            args.img_dir,
            args.camera_json_path,
            args.matching_pattern,
        )

    elif args.sparse_dir is not None and args.out_aligned_dir is not None:
        align_monodepth_with_colmap(
            args.sparse_dir,
            args.out_raw_dir,
            args.out_aligned_dir,
            args.img_dir,
            args.camera_json_path,
            args.matching_pattern,
        )

    elif args.metric_dir is not None and args.out_aligned_dir is not None:
        align_monodepth_with_metric_depth(
            args.metric_dir,
            args.out_raw_dir,
            args.out_aligned_dir,
            args.matching_pattern,
        )


if __name__ == "__main__":
    """ example usage for iphone dataset:
    python compute_depth.py \
        --img_dir /home/qianqianwang_google_com/datasets/iphone/dycheck/paper-windmill/rgb/1x \
        --out_raw_dir /home/qianqianwang_google_com/datasets/iphone/dycheck/paper-windmill/flow3d_preprocessed/depth_anything_v2/1x \
        --out_aligned_dir /home/qianqianwang_google_com/datasets/iphone/dycheck/paper-windmill/flow3d_preprocessed/aligned_depth_anything_v2/1x \
        --sparse_dir /home/qianqianwang_google_com/datasets/iphone/dycheck/paper-windmill/flow3d_preprocessed/colmap/sparse \
        --matching_pattern "0_*"
    """
    main()
