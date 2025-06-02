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
import fnmatch
from tqdm import tqdm
import imageio.v3 as iio

# Import evaluation metrics
from flow3d.metrics import mLPIPS, mPSNR, mSSIM


@dataclass
class EvalConfig:
    # Data paths
    data_dir: str  
    camera_json: str 
    output_dir: str 
    
    # Model paths
    work_dir: str = ""  
    ckpt_path: str = ""
    
    # Dataset settings
    res: str = "test"  
    matching_pattern: str = "*"  
    image_type: str = "render" 
    
    # Processing settings
    start_frame: int = 0  
    end_frame: int = -1  
    image_ext: str = ".png"


def extract_frame_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def extract_ts_from_frame_name(name: str) -> int:
    match = re.match(r"(\d+)_(\d+)", name)
    if match:
        return int(match.group(2))
    
    # Fallback: try to extract any number
    match = re.search(r"(\d+)", name)
    if match:
        return int(match.group(1))
    
    raise ValueError(f"Frame name {name} is not in the expected format (e.g., '0_123' or '1_456')")


def load_camera_params_from_json(json_path, img_files, img_dir, use_width_for_focal=False):
    import imageio as iio
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Camera JSON file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        meta = json.load(f)
    
    if "camera_angle_x" not in meta:
        raise ValueError(f"camera_angle_x not found in {json_path}")
    
    angle_x = meta["camera_angle_x"]
    
    frame_dict = {}
    for frame in meta["frames"]:
        file_path = frame["file_path"]  
        image_name = os.path.basename(file_path) + ".png"
        frame_dict[image_name] = frame
    
    num_imgs = len(img_files)
    Ks = np.zeros((num_imgs, 3, 3))
    w2cs = np.zeros((num_imgs, 4, 4))
    
    for idx, img_file in enumerate(img_files):
        image_name = os.path.basename(img_file)
        
        if image_name not in frame_dict:
            raise ValueError(f"Image {image_name} not found in camera JSON")
        
        frame = frame_dict[image_name]
        image_path = os.path.join(img_dir, image_name)
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        img = iio.imread(image_path)
        h, w = img.shape[:2]
        
        if use_width_for_focal:
            fx = fy = 0.5 * w / np.tan(angle_x / 2)  
        else:
            fx = fy = 0.5 * h / np.tan(angle_x / 2)
        cx, cy = w / 2, h / 2
        
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        Ks[idx] = K
        
        T_c2w = np.array(frame["transform_matrix"])
        flip = np.diag([1, -1, -1, 1])
        T_c2w_cv = T_c2w @ flip
        T_w2c = np.linalg.inv(T_c2w_cv)
        w2cs[idx] = T_w2c
    
    tstamps = np.arange(num_imgs, dtype=np.int32)
    return (
        torch.from_numpy(w2cs).float(),
        torch.from_numpy(Ks).float(),
        torch.from_numpy(tstamps),
    )


def load_scene_norm_dict(data_dir: str) -> dict:
    scene_norm_path = os.path.join(data_dir, f"flow3d_preprocessed/train", "scene_norm_dict.pth")
    
    if not os.path.exists(scene_norm_path):
        raise FileNotFoundError(f"Scene norm dict not found: {scene_norm_path}")
    
    guru.info(f"Loading scene norm dict from: {scene_norm_path}")
    scene_norm_dict = torch.load(scene_norm_path, map_location='cpu')
    return scene_norm_dict


def load_images_from_directory(image_dir: str, image_ext: str = ".png", matching_pattern: str = "*") -> list[str]:
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    all_files = [f for f in os.listdir(image_dir) if f.endswith(image_ext)]
    if not all_files:
        raise FileNotFoundError(f"No images found in {image_dir} with extension {image_ext}")
    
    frame_names = [os.path.splitext(p)[0] for p in sorted(all_files)]
    
    if matching_pattern != "*":
        filtered_frame_names = []
        for frame_name in frame_names:
            if fnmatch.fnmatch(frame_name, matching_pattern):
                filtered_frame_names.append(frame_name)
        frame_names = filtered_frame_names
        guru.info(f"Applied pattern '{matching_pattern}': {len(frame_names)} files matched")
    
    if not frame_names:
        raise ValueError(f"No frames matched pattern '{matching_pattern}' in {image_dir}")
    
    guru.info(f"Found {len(frame_names)} images in {image_dir}")
    return frame_names


def load_single_image(image_path: str) -> tuple[np.ndarray, tuple[int, int]]:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = imageio.imread(image_path)
    # Handle RGBA images
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]  # RGBA -> RGB
    
    H, W = img.shape[:2]
    return img, (W, H)


def render_images(
    renderer: Renderer,
    image_dir: str,
    camera_json: str,
    output_dir: str,
    config: EvalConfig,
    device: torch.device
) -> tuple[list[str], bool]:
    """Render images and return frame names and success status"""
    
    frame_names = load_images_from_directory(image_dir, config.image_ext, config.matching_pattern)
    
    if config.end_frame == -1:
        end_frame = len(frame_names)
    else:
        end_frame = min(config.end_frame, len(frame_names))
    
    frame_names = frame_names[config.start_frame:end_frame]
    guru.info(f"Processing frames {config.start_frame} to {end_frame-1} ({len(frame_names)} frames)")
    
    if len(frame_names) == 0:
        raise ValueError("No frames to process")
    
    first_image_path = os.path.join(image_dir, f"{frame_names[0]}{config.image_ext}")
    _, img_wh = load_single_image(first_image_path)
    W, H = img_wh
    
    guru.info(f"Image resolution: {W}x{H}")
    guru.info(f"Image dir: {image_dir}")
    guru.info(f"Data dir: {config.data_dir}")
    
    # Load scene normalization dict
    scene_norm_dict = load_scene_norm_dict(config.data_dir)
    
    # Load camera parameters
    img_files = [f"{name}{config.image_ext}" for name in frame_names]
    w2cs, Ks, tstamps = load_camera_params_from_json(config.camera_json, img_files, image_dir)
    
    if len(w2cs) != len(frame_names):
        guru.warning(f"Mismatch: {len(w2cs)} cameras vs {len(frame_names)} images")
        min_len = min(len(w2cs), len(frame_names))
        w2cs = w2cs[:min_len]
        Ks = Ks[:min_len]
        frame_names = frame_names[:min_len]
    
    # Apply scene normalization
    scale = scene_norm_dict["scale"]
    transform = scene_norm_dict["transfm"]
    guru.info(f"Applying scene normalization: scale={scale}")
    
    w2cs = torch.einsum("nij,jk->nik", w2cs, torch.linalg.inv(transform))
    w2cs[:, :3, 3] /= scale
    
    w2cs = w2cs.to(device)
    Ks = Ks.to(device)
    
    os.makedirs(output_dir, exist_ok=True)
    
    guru.info(f"Starting rendering {len(frame_names)} frames...")
    
    renderer.model.training = False
    success_count = 0
    failed_frames = []
    
    for i, frame_name in enumerate(tqdm(frame_names, desc="Rendering frames")):
        try:
            try:
                ts = extract_ts_from_frame_name(frame_name)
                if ts > 0:
                    ts = ts - 1
            except ValueError as e:
                guru.warning(f"Could not extract timestamp from {frame_name}, using index {i}")
                ts = i
            
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
                
            else:
                guru.error(f"No 'img' in rendered result for frame {frame_name}")
                failed_frames.append(frame_name)
                
        except Exception as e:
            guru.error(f"Error rendering frame {frame_name}: {e}")
            failed_frames.append(frame_name)
            continue
    
    guru.info(f"Successfully rendered {success_count}/{len(frame_names)} frames")
    if failed_frames:
        guru.warning(f"Failed frames: {failed_frames}")
    
    return frame_names, success_count > 0


def load_ground_truth_images(image_dir: str, frame_names: list[str], image_ext: str = ".png") -> np.ndarray:
    """Load ground truth images for evaluation"""
    gt_imgs = []
    
    for fn in tqdm(frame_names, desc="Loading ground truth images"):
        img_path = os.path.join(image_dir, f"{fn}{image_ext}")
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Ground truth image not found: {img_path}")
        
        img = iio.imread(img_path)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        gt_imgs.append(img)
    
    return np.array(gt_imgs)


def load_rendered_images(result_dir: str, frame_names: list[str]) -> np.ndarray:
    """Load rendered images for evaluation"""
    pred_imgs = []
    
    for name in frame_names:
        img_path = os.path.join(result_dir, f"{name}_rendered.png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Rendered image not found: {img_path}")
        pred_imgs.append(iio.imread(img_path))
    
    return np.array(pred_imgs)


def evaluate_metrics(gt_imgs: np.ndarray, pred_imgs: np.ndarray, device: torch.device) -> dict:
    """Evaluate PSNR, SSIM, and LPIPS metrics"""
    
    psnr_metric = mPSNR().to(device)
    ssim_metric = mSSIM().to(device)
    lpips_metric = mLPIPS().to(device)

    gt_imgs_tensor = torch.from_numpy(gt_imgs)[..., :3].to(device)
    pred_imgs_tensor = torch.from_numpy(pred_imgs)[..., :3].to(device)

    min_len = min(len(gt_imgs_tensor), len(pred_imgs_tensor))
    gt_imgs_tensor = gt_imgs_tensor[:min_len]
    pred_imgs_tensor = pred_imgs_tensor[:min_len]
    
    guru.info(f"Evaluating {min_len} image pairs")

    for i in range(min_len):
        gt_img = gt_imgs_tensor[i].clamp(0, 255) / 255.0
        pred_img = pred_imgs_tensor[i].clamp(0, 255) / 255.0

        psnr_metric.update(gt_img, pred_img)
        ssim_metric.update(gt_img[None], pred_img[None])
        lpips_metric.update(gt_img[None], pred_img[None])
    
    mpsnr = psnr_metric.compute().item()
    mssim = ssim_metric.compute().item()
    mlpips = lpips_metric.compute().item()
    
    return {
        "psnr": mpsnr,
        "ssim": mssim,
        "lpips": mlpips
    }


def main(config: EvalConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    guru.info(f"Using device: {device}")
    
    # Validate input paths
    if not os.path.exists(config.data_dir):
        raise FileNotFoundError(f"Data directory not found: {config.data_dir}")
    if not os.path.exists(config.camera_json):
        raise FileNotFoundError(f"Camera JSON not found: {config.camera_json}")
    
    # Setup model paths
    if config.ckpt_path:
        ckpt_path = config.ckpt_path
        work_dir = os.path.dirname(os.path.dirname(ckpt_path))
        guru.info(f"Using direct checkpoint path: {ckpt_path}")
    elif config.work_dir:
        work_dir = config.work_dir
        ckpt_path = os.path.join(work_dir, "checkpoints", "last.ckpt")
        guru.info(f"Using work_dir: {work_dir}, checkpoint: {ckpt_path}")
    else:
        raise ValueError("Either ckpt_path or work_dir must be provided")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    # Load training config
    train_cfg_path = os.path.join(work_dir, "cfg.yaml")
    if os.path.exists(train_cfg_path):
        with open(train_cfg_path, "r") as file:
            train_cfg = yaml.safe_load(file)
        use_2dgs = train_cfg.get("use_2dgs", False)
        guru.info(f"Loaded training config: use_2dgs={use_2dgs}")
    else:
        guru.warning(f"cfg.yaml not found at {train_cfg_path}, using default use_2dgs=False")
        use_2dgs = False
    
    # Initialize renderer
    try:
        renderer = Renderer.init_from_checkpoint(
            ckpt_path,
            device,
            use_2dgs=use_2dgs,
            work_dir=work_dir,
            port=None,  
        )
        guru.info(f"Successfully loaded model from {ckpt_path}")
    except Exception as e:
        guru.error(f"Failed to load renderer: {e}")
        raise
    
    # Setup image directory
    image_dir = os.path.join(config.data_dir, config.image_type, config.res)
    
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    # === INFERENCE ===
    guru.info("=" * 50)
    guru.info("STARTING INFERENCE")
    guru.info("=" * 50)
    
    try:
        frame_names, render_success = render_images(
            renderer,
            image_dir,
            config.camera_json,
            config.output_dir,
            config,
            device
        )
        
        if not render_success:
            guru.error("Rendering failed!")
            return
            
        guru.info(f"Inference completed! Results saved to: {config.output_dir}")
        
    except Exception as e:
        guru.error(f"Inference failed: {e}")
        raise
    
    # === EVALUATION ===
    guru.info("=" * 50)
    guru.info("STARTING EVALUATION")
    guru.info("=" * 50)
    
    try:
        # Load ground truth images
        gt_imgs = load_ground_truth_images(image_dir, frame_names, config.image_ext)
        
        # Load rendered images
        pred_imgs = load_rendered_images(config.output_dir, frame_names)
        
        # Evaluate metrics
        metrics = evaluate_metrics(gt_imgs, pred_imgs, device)
        
        # Print results
        guru.info("=" * 50)
        guru.info("EVALUATION RESULTS")
        guru.info("=" * 50)
        guru.info(f"PSNR: {metrics['psnr']:.4f}")
        guru.info(f"SSIM: {metrics['ssim']:.4f}")
        guru.info(f"LPIPS: {metrics['lpips']:.4f}")
        guru.info(f"Evaluated on {len(frame_names)} image pairs")
        
        # Save results
        results = {
            "metrics": metrics,
            "evaluation_info": {
                "frame_count": len(frame_names),
                "data_dir": config.data_dir,
                "res": config.res,
                "matching_pattern": config.matching_pattern,
                "image_type": config.image_type,
                "camera_json": config.camera_json
            },
            "config": {
                "work_dir": config.work_dir,
                "ckpt_path": ckpt_path,
                "output_dir": config.output_dir
            }
        }
        
        results_path = os.path.join(config.output_dir, "evaluation_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        guru.info(f"Evaluation results saved to: {results_path}")
        
    except Exception as e:
        guru.error(f"Evaluation failed: {e}")
        guru.info("Inference was successful, but evaluation failed. Check the evaluation settings and try again.")


if __name__ == "__main__":
    # Example usage:
    # 
    # Test evaluation (recommended after training):
    # python eval.py \
    #   --work-dir ./outputs/torus_experiment \
    #   --data-dir /131_data/intern/gunhee/PhysTrack/New/FLIP/ship \
    #   --camera-json /131_data/intern/gunhee/PhysTrack/New/FLIP/ship/camera_info_test.json \
    #   --output-dir ./outputs/torus_experiment/eval_test \
    #   --res test \
    #   --image-type render
    
    main(tyro.cli(EvalConfig))