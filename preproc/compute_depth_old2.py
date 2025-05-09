import argparse
import fnmatch
import os
import os.path as osp
from glob import glob
import json
from typing import Literal

import cv2
import imageio.v2 as iio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Pipeline, pipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UINT16_MAX = 65535

models = {
    "depth-anything": "LiheYoung/depth-anything-large-hf",
    "depth-anything-v2": "depth-anything/Depth-Anything-V2-Large-hf",
}

def load_intrinsics_and_extrinsics_from_camera_json(json_path, image_dir):
    import imageio.v3 as iio

    with open(json_path, 'r') as f:
        meta = json.load(f)

    angle_x = meta["camera_angle_x"]
    K_dict = {}
    extrinsics_dict = {}

    for frame in meta["frames"]:
        file_path = frame["file_path"]
        image_name = os.path.basename(file_path) + ".png"
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            continue

        img = iio.imread(image_path)
        h, w = img.shape[:2]

        fx = fy = 0.5 * w / np.tan(angle_x / 2)
        cx, cy = w / 2, h / 2
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ])
        K_dict[image_name] = K

        T_c2w = np.array(frame["transform_matrix"])
        T_w2c = np.linalg.inv(T_c2w)
        extrinsics_dict[image_name] = T_w2c

    return K_dict, extrinsics_dict

def parse_colmap_txt(sparse_dir):
    images = {}
    points3D = {}
    with open(osp.join(sparse_dir, "images.txt")) as f:
        lines = [l.strip() for l in f if l and not l.startswith("#")]
    for i in range(0, len(lines), 2):
        elems = lines[i].split()
        image_id, qvec, tvec, cam_id, name = int(elems[0]), np.array(list(map(float, elems[1:5]))), np.array(list(map(float, elems[5:8]))), int(elems[8]), elems[9]
        point3D_ids = np.array(list(map(int, lines[i+1].split()[2::3])))
        images[image_id] = {
            "qvec": qvec, "tvec": tvec, "camera_id": cam_id, "name": name, "point3D_ids": point3D_ids
        }
    with open(osp.join(sparse_dir, "points3D.txt")) as f:
        for l in f:
            if l.startswith("#") or not l.strip(): continue
            elems = l.strip().split()
            id, xyz = int(elems[0]), np.array(list(map(float, elems[1:4])))
            points3D[id] = xyz
    return images, points3D

def get_pipeline(model_name: str):
    pipe = pipeline(task="depth-estimation", model=models[model_name], device=DEVICE)
    print(f"{model_name} model loaded.")
    return pipe

def to_uint16(disp: np.ndarray):
    disp_min, disp_max = disp.min(), disp.max()
    if disp_max - disp_min > np.finfo("float").eps:
        disp_uint16 = UINT16_MAX * (disp - disp_min) / (disp_max - disp_min)
    else:
        disp_uint16 = np.zeros(disp.shape, dtype=disp.dtype)
    return disp_uint16.astype(np.uint16)

def get_depth_anything_disp(pipe: Pipeline, img_file: str, ret_type: Literal["uint16", "float"] = "float"):
    image = Image.open(img_file)
    disp = pipe(image)["predicted_depth"]
    if disp.ndim == 3: disp = disp.unsqueeze(1)
    elif disp.ndim == 2: disp = disp.unsqueeze(0).unsqueeze(0)
    disp = torch.nn.functional.interpolate(disp, size=image.size[::-1], mode="bicubic", align_corners=False).squeeze().cpu().numpy()
    return to_uint16(disp) if ret_type == "uint16" else disp

def save_disp_from_dir(model_name: str, img_dir: str, out_dir: str, matching_pattern: str = "*"):
    img_files = sorted(glob(osp.join(img_dir, "*.jpg"))) + sorted(glob(osp.join(img_dir, "*.png")))
    img_files = [f for f in img_files if fnmatch.fnmatch(osp.basename(f), matching_pattern)]
    if osp.exists(out_dir) and len(glob(osp.join(out_dir, "*.png"))) == len(img_files): return
    pipe = get_pipeline(model_name)
    os.makedirs(out_dir, exist_ok=True)
    for img_file in tqdm(img_files, f"computing {model_name} depth maps"):
        disp = get_depth_anything_disp(pipe, img_file, ret_type="uint16")
        iio.imwrite(osp.join(out_dir, osp.splitext(osp.basename(img_file))[0] + ".png"), disp)

def align_monodepth_with_colmap_txt(sparse_dir, input_dir, output_dir, img_dir, camera_json_path, matching_pattern="*"):
    images, points3D = parse_colmap_txt(sparse_dir)
    K_dict, ext_dict = load_intrinsics_and_extrinsics_from_camera_json(camera_json_path, img_dir)
    os.makedirs(output_dir, exist_ok=True)

    for image in tqdm(images.values(), desc="Aligning with COLMAP TXT"):
        if not fnmatch.fnmatch(image["name"], matching_pattern): continue
        valid_ids = image["point3D_ids"]
        valid_ids = valid_ids[valid_ids != -1]
        pts3d = np.array([points3D[i] for i in valid_ids if i in points3D])
        print(pts3d)
        if pts3d.shape[0] == 0: continue
        name = os.path.basename(image["name"])
        if name not in K_dict or name not in ext_dict: continue
        K, ext = K_dict[name], ext_dict[name]
        homo = np.concatenate([pts3d, np.ones((pts3d.shape[0], 1))], axis=-1).T
        cam = ext @ homo
        pts2d = (K @ cam[:3]).T
        pts2d = pts2d[:, :2] / pts2d[:, 2:3]
        depth = cam[2]
        mono_path = osp.join(input_dir, osp.splitext(name)[0] + ".png")
        if not osp.exists(mono_path): continue
        mono = iio.imread(mono_path) / UINT16_MAX
        mono_interp = cv2.remap(mono, pts2d[:, 0].reshape(-1, 1).astype(np.float32), pts2d[:, 1].reshape(-1, 1).astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).flatten()
        col_disp = 1.0 / np.clip(depth, 1e-6, 1e6)
        ms_col = col_disp - np.median(col_disp) + 1e-8
        ms_mono = mono_interp - np.median(mono_interp) + 1e-8
        scale = np.median(ms_col / ms_mono)
        shift = np.median(col_disp - scale * mono_interp)
        aligned = scale * mono + shift
        aligned[aligned < min(1e-6, np.quantile(aligned, 0.01))] = 0.0
        np.save(osp.join(output_dir, name.split(".")[0] + ".npy"), aligned)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="depth-anything")
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--out_raw_dir", type=str, required=True)
    parser.add_argument("--out_aligned_dir", type=str, default=None)
    parser.add_argument("--sparse_dir", type=str, default=None)
    parser.add_argument("--camera_json_path", type=str, default=None)
    parser.add_argument("--matching_pattern", type=str, default="*")
    args = parser.parse_args()

    save_disp_from_dir(args.model, args.img_dir, args.out_raw_dir, args.matching_pattern)

    if args.sparse_dir and args.out_aligned_dir:
        align_monodepth_with_colmap_txt(
            args.sparse_dir, args.out_raw_dir, args.out_aligned_dir,
            args.img_dir, args.camera_json_path, args.matching_pattern
        )

if __name__ == "__main__":
    main()
