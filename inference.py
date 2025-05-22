import os
import numpy as np
import imageio
import json
from dataclasses import dataclass
import tyro
from loguru import logger as guru
from flow3d.renderer import Renderer
import yaml
from pathlib import Path
from typing import Optional
import torch
import re

def extract_timestamp_from_path(path: str) -> str:
    import re
    timestamp_pattern = r'(\d{8}_\d{4})'
    match = re.search(timestamp_pattern, path)
    if match:
        timestamp = match.group(1)
        cache_res = f"train_{timestamp}"
        guru.info(f"Extracted timestamp from path: {timestamp} -> cache_res: {cache_res}")
        return cache_res
    else:
        guru.warning(f"Could not extract timestamp from path: {path}")
        return 
        
def load_scene_norm_dict(data_dir: str, cache_res: str) -> dict:
    if not cache_res:
        raise ValueError("cache_res is required to find scene_norm_dict.pth")
    
    scene_norm_path = os.path.join(data_dir, cache_res, "scene_norm_dict.pth")
    
    if os.path.exists(scene_norm_path):
        guru.info(f"Loading scene norm dict from: {scene_norm_path}")
        scene_norm_dict = torch.load(scene_norm_path, map_location='cpu')
        return scene_norm_dict
    else:
        raise FileNotFoundError(f"Scene norm dict not found at: {scene_norm_path}")


@dataclass
class JsonCameraInferenceConfig:
    image_dir: str  
    camera_json: str  
    output_dir: str  
    work_dir: str = ""  
    ckpt_path: str = ""  
    time_step: int = 0  
    start_frame: int = 0  
    end_frame: int = -1  
    image_ext: str = ".png"  
    cache_res: str = ""  


def extract_frame_name(path):
    return os.path.splitext(os.path.basename(path))[0]



def extract_ts_from_frame_name(name: str) -> int:
    match = re.match(r"\d+_(\d+)", name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Frame name {name} is not in the expected format (e.g., '1_123')")

def load_cameras_from_json_custom(
    json_path: str, 
    H: int, 
    W: int, 
    frame_names: list[str],
    scene_norm_dict: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    with open(json_path, 'r') as f:
        meta = json.load(f)

    angle_x = meta["camera_angle_x"]
    fx = fy = 0.5 * H / np.tan(angle_x / 2)
    cx, cy = W / 2, H / 2
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    json_frame_dict = {}
    for frame in meta["frames"]:
        full_frame_name = extract_frame_name(frame["file_path"])
        c2w = np.array(frame["transform_matrix"])
        flip = np.diag([1, -1, -1, 1])
        c2w = c2w @ flip
        w2c = np.linalg.inv(c2w)
        json_frame_dict[full_frame_name] = w2c

    # frame_names에 맞춰서 정렬
    w2cs = []
    Ks = []
    matched_count = 0
    for name in frame_names:
        if name in json_frame_dict:
            w2cs.append(json_frame_dict[name])
            Ks.append(K)
            matched_count += 1
        else:
            guru.warning(f"Frame {name} not found in JSON, skipping")

    if len(w2cs) == 0:
        available_frames = list(json_frame_dict.keys())[:10]
        raise ValueError(f"No frames found in JSON file. Available frames: {available_frames}...")

    w2cs = torch.from_numpy(np.stack(w2cs)).float()
    Ks = torch.from_numpy(np.stack(Ks)).float()
    
    scale = scene_norm_dict["scale"]
    transform = scene_norm_dict["transfm"]
    
    guru.info(f"Applying scene normalization: scale={scale}")
    guru.info(f"Transform matrix:\n{transform}")
    
    w2cs = torch.einsum("nij,jk->nik", w2cs, torch.linalg.inv(transform))
    w2cs[:, :3, 3] /= scale

    guru.info(f"Successfully loaded {len(w2cs)}/{len(frame_names)} camera poses with scene normalization")
    return w2cs, Ks


def load_images_from_directory(image_dir: str, image_ext: str = ".png") -> list[str]:
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith(image_ext)]

    def extract_numeric_prefix(f):
        match = re.match(r"(\d+)_", f)
        return int(match.group(1)) if match else float('inf')

    image_files.sort(key=lambda f: (extract_numeric_prefix(f), f))
    
    frame_names = [os.path.splitext(f)[0] for f in image_files]
    
    guru.info(f"Found {len(frame_names)} images in {image_dir}")
    return frame_names


def load_single_image(image_path: str) -> tuple[np.ndarray, tuple[int, int]]:
    img = imageio.imread(image_path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]  # RGBA -> RGB
    
    H, W = img.shape[:2]
    return img, (W, H)


def render_images_with_json_cameras(
    renderer: Renderer,
    image_dir: str,
    camera_json: str,
    output_dir: str,
    config: JsonCameraInferenceConfig,
    device: torch.device
):    
    frame_names = load_images_from_directory(image_dir, config.image_ext)
    
    if config.end_frame == -1:
        end_frame = len(frame_names)
    else:
        end_frame = min(config.end_frame, len(frame_names))
    
    frame_names = frame_names[config.start_frame:end_frame]
    guru.info(f"Processing frames {config.start_frame} to {end_frame-1}")
    
    if len(frame_names) == 0:
        raise ValueError("No frames to process")
    
    first_image_path = os.path.join(image_dir, f"{frame_names[0]}{config.image_ext}")
    _, img_wh = load_single_image(first_image_path)
    W, H = img_wh
    
    guru.info(f"Image resolution: {W}x{H}")
    
    data_dir = os.path.dirname(os.path.dirname(os.path.normpath(image_dir)))
    guru.info(f"Image dir: {image_dir}")
    guru.info(f"Data dir: {data_dir}")
    scene_norm_dict = load_scene_norm_dict(data_dir, config.cache_res)
    
    w2cs, Ks = load_cameras_from_json_custom(
        config.camera_json, H, W, frame_names, scene_norm_dict
    )
    
    if len(w2cs) != len(frame_names):
        guru.warning(f"Mismatch: {len(w2cs)} cameras vs {len(frame_names)} images")
        min_len = min(len(w2cs), len(frame_names))
        w2cs = w2cs[:min_len]
        Ks = Ks[:min_len]
        frame_names = frame_names[:min_len]
    
    w2cs = w2cs.to(device)
    Ks = Ks.to(device)
    
    os.makedirs(output_dir, exist_ok=True)
    
    guru.info(f"Starting rendering {len(frame_names)} frames...")
    
    renderer.model.training = False
    success_count = 0
    
    for i, frame_name in enumerate(frame_names):
        try:
            guru.info(f"Rendering frame {i+1}/{len(frame_names)}: {frame_name}")
            
            ts = extract_ts_from_frame_name(frame_name)
            w2c = w2cs[i:i+1]  # (1, 4, 4)
            K = Ks[i:i+1]      # (1, 3, 3)
            

            with torch.inference_mode():
                rendered = renderer.model.render(
                    ts,
                    w2c,
                    K,
                    img_wh,
                    return_depth=True,
                )
            
            if "img" in rendered:
                img_np = (rendered["img"][0].detach().cpu().numpy() * 255).astype(np.uint8)
                output_path = os.path.join(output_dir, f"{frame_name}_rendered.png")
                imageio.imwrite(output_path, img_np)
                
                if "depth" in rendered:
                    depth_np = rendered["depth"][0].detach().cpu().numpy()
                    depth_path = os.path.join(output_dir, f"{frame_name}_depth.npy")
                    np.save(depth_path, depth_np)
                
                success_count += 1
                guru.info(f"Saved: {output_path}")
                
            else:
                guru.error(f"No 'img' in rendered result for frame {frame_name}")
                
        except Exception as e:
            guru.error(f"Error rendering frame {frame_name}: {e}")
            continue
    
    guru.info(f"Successfully rendered {success_count}/{len(frame_names)} frames")
    return success_count


def main(config: JsonCameraInferenceConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if config.ckpt_path:
        ckpt_path = config.ckpt_path
        work_dir = os.path.dirname(os.path.dirname(ckpt_path))  # checkpoints/last.ckpt -> work_dir
        guru.info(f"Using direct checkpoint path: {ckpt_path}")
        
        if not config.cache_res:
            extracted_cache_res = extract_timestamp_from_path(ckpt_path)
            if extracted_cache_res:
                config.cache_res = extracted_cache_res
                guru.info(f"Auto-extracted cache_res: {config.cache_res}")
            
    elif config.work_dir:
        work_dir = config.work_dir
        ckpt_path = os.path.join(work_dir, "checkpoints", "last.ckpt")
        guru.info(f"Using work_dir: {work_dir}, checkpoint: {ckpt_path}")
        
        if not config.cache_res:
            extracted_cache_res = extract_timestamp_from_path(work_dir)
            if extracted_cache_res:
                config.cache_res = extracted_cache_res
                guru.info(f"Auto-extracted cache_res: {config.cache_res}")
    else:
        raise ValueError("Either ckpt_path or work_dir must be provided")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    train_cfg_path = os.path.join(work_dir, "cfg.yaml")
    if os.path.exists(train_cfg_path):
        with open(train_cfg_path, "r") as file:
            train_cfg = yaml.safe_load(file)
        use_2dgs = train_cfg.get("use_2dgs", False)
    else:
        guru.warning(f"cfg.yaml not found at {train_cfg_path}, using default use_2dgs=False")
        use_2dgs = False
    
    renderer = Renderer.init_from_checkpoint(
        ckpt_path,
        device,
        use_2dgs=use_2dgs,
        work_dir=work_dir,
        port=None,  
    )
    
    guru.info(f"Loaded model from {ckpt_path}")
    
    if not os.path.exists(config.image_dir):
        raise FileNotFoundError(f"Image directory not found: {config.image_dir}")
    if not os.path.exists(config.camera_json):
        raise FileNotFoundError(f"Camera JSON not found: {config.camera_json}")
    
    success_count = render_images_with_json_cameras(
        renderer,
        config.image_dir,
        config.camera_json,
        config.output_dir,
        config,
        device
    )
    
    guru.info(f"Rendering completed! {success_count} frames processed.")
    guru.info(f"Results saved to: {config.output_dir}")


if __name__ == "__main__":
    # python json_camera_inference.py \
    #   --work-dir /path/to/trained/model \
    #   --image-dir /131_data/intern/gunhee/PhysTrack/New/FLIP/ship/render/test/ \
    #   --camera-json /path/to/camera_info_test.json \
    #   --output-dir /path/to/output/
    
    # python json_camera_inference.py \
    #   --ckpt-path /path/to/model/checkpoints/last.ckpt \
    #   --image-dir /131_data/intern/gunhee/PhysTrack/New/FLIP/ship/render/test/ \
    #   --camera-json /path/to/camera_info_test.json \
    #   --output-dir /path/to/output/
    
    main(tyro.cli(JsonCameraInferenceConfig))