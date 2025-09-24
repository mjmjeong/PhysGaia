import os
import cv2
import lpips
import torch
import numpy as np
from skimage.metrics import structural_similarity
import os.path as osp
import pandas as pd
import logging
from glob import glob


def im2tensor(img):
    return torch.Tensor(img.transpose(2, 0, 1) / 127.5 - 1.0)[None, ...]


def eval_physgaia_dir(gt_dir, pred_dir, report_dir, target_scene_id=1):
    lpips_loss = lpips.LPIPS(net="alex")
    assert osp.exists(gt_dir), f"GT directory does not exist: {gt_dir}"
    assert osp.exists(pred_dir), f"Prediction directory does not exist: {pred_dir}"
    
    os.makedirs(report_dir, exist_ok=True)
    viz_dir = osp.join(report_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)
    
    gt_pattern = osp.join(gt_dir, f"{target_scene_id}_*.png")
    gt_files = sorted(glob(gt_pattern))
    
    if not gt_files:
        raise ValueError(f"No GT files found for scene {target_scene_id} in {gt_dir}")
    
    logging.info(f"Found {len(gt_files)} GT files for scene {target_scene_id}")
    logging.info(f"GT files: {[osp.basename(f) for f in gt_files[:5]]}...")  
    
    timestamps = []
    for gt_file in gt_files:
        filename = osp.basename(gt_file)
        parts = filename.replace('.png', '').split('_')
        if len(parts) >= 2:
            try:
                timestamp = int(parts[1])
                timestamps.append(timestamp)
            except ValueError:
                logging.warning(f"Could not parse timestamp from {filename}")
                continue
    
    if not timestamps:
        raise ValueError("No valid timestamps found in GT files")
    
    timestamps.sort()
    logging.info(f"Timestamp range: {min(timestamps)} - {max(timestamps)}")
    
    psnr_list, ssim_list, lpips_list = [], [], []
    results = []
    
    for timestamp in timestamps:
        gt_fn = osp.join(gt_dir, f"{target_scene_id}_{timestamp:03d}.png")
        
        pred_fn = osp.join(pred_dir, f"{target_scene_id}_{timestamp:03d}.png")
        
        if not osp.exists(gt_fn):
            logging.warning(f"GT file not found: {gt_fn}")
            continue
        
        if not osp.exists(pred_fn):
            logging.warning(f"Prediction file not found: {pred_fn}")
            continue
        
        img_true = cv2.imread(gt_fn)
        img = cv2.imread(pred_fn)
        
        if img_true is None:
            logging.warning(f"Could not load GT image: {gt_fn}")
            continue
        
        if img is None:
            logging.warning(f"Could not load prediction image: {pred_fn}")
            continue
        
        if img.shape != img_true.shape:
            logging.info(f"Resizing prediction image from {img.shape} to {img_true.shape}")
            img = cv2.resize(img, (img_true.shape[1], img_true.shape[0]))
        
        _psnr = cv2.PSNR(img_true, img)
        _ssim = structural_similarity(img_true, img, multichannel=True)
        _lpips = lpips_loss.forward(im2tensor(img_true), im2tensor(img)).item()
        
        psnr_list.append(_psnr)
        ssim_list.append(_ssim)
        lpips_list.append(_lpips)
        
        results.append({
            "timestamp": timestamp,
            "scene_id": target_scene_id,
            "psnr": _psnr,
            "ssim": _ssim,
            "lpips": _lpips
        })
        
        errmap = np.sqrt(((img - img_true) ** 2).sum(-1)) / np.sqrt(3)
        factor = errmap.max()
        if factor > 0:
            errmap = errmap / factor * 255
        errmap = cv2.applyColorMap((errmap).astype(np.uint8), cv2.COLORMAP_JET)
        
        errmap = cv2.putText(
            errmap,
            f"Scene{target_scene_id}_T{timestamp:03d} NORM[{factor:.3f}] PSNR={_psnr:.3f}, SSIM={_ssim:.3f}, LPIPS={_lpips:.3f}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4, 
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        
        save_viz = np.concatenate([img_true, img, errmap], axis=1)
        save_name = osp.join(
            viz_dir, f"scene{target_scene_id}_t{timestamp:03d}_psnr={_psnr:.3f}.png"
        )
        cv2.imwrite(save_name, save_viz)
    
    if not psnr_list:
        raise ValueError("No valid image pairs found for evaluation")
    
    ave_psnr = np.mean(psnr_list)
    ave_ssim = np.mean(ssim_list)
    ave_lpips = np.mean(lpips_list)
    
    with open(osp.join(report_dir, "physgaia_render_metrics.txt"), "w") as f:
        f.write(f"Scene ID: {target_scene_id}\n")
        f.write(f"Number of frames: {len(psnr_list)}\n")
        f.write(f"Timestamp range: {min(timestamps)} - {max(timestamps)}\n")
        f.write(f"PSNR: {ave_psnr:.6f}\n")
        f.write(f"SSIM: {ave_ssim:.6f}\n")
        f.write(f"LPIPS: {ave_lpips:.6f}\n")
    
    summary_result = {
        "timestamp": "average",
        "scene_id": target_scene_id,
        "psnr": ave_psnr,
        "ssim": ave_ssim,
        "lpips": ave_lpips
    }
    results = [summary_result] + results
    
    df = pd.DataFrame(results)
    save_report_fn = osp.join(report_dir, "physgaia_render_metrics.xlsx")
    df.to_excel(save_report_fn, index=False)
    
    logging.info(f"Saved the evaluation report to {report_dir}")
    logging.info(
        f"Scene {target_scene_id} Metrics: PSNR={ave_psnr:.6f}, SSIM={ave_ssim:.6f}, LPIPS={ave_lpips:.6f}"
    )
    
    return ave_psnr, ave_ssim, ave_lpips


def eval_physgaia_multi_scenes(gt_dir, pred_dir, report_dir, scene_ids=None):
    if scene_ids is None:
        gt_files = glob(osp.join(gt_dir, "*.png"))
        scene_ids = set()
        for gt_file in gt_files:
            filename = osp.basename(gt_file)
            parts = filename.replace('.png', '').split('_')
            if len(parts) >= 2:
                try:
                    scene_id = int(parts[0])
                    scene_ids.add(scene_id)
                except ValueError:
                    continue
        scene_ids = sorted(list(scene_ids))
        logging.info(f"Auto-detected scene IDs: {scene_ids}")
    
    all_results = []
    total_psnr, total_ssim, total_lpips = [], [], []
    
    for scene_id in scene_ids:
        logging.info(f"Evaluating scene {scene_id}...")
        scene_report_dir = osp.join(report_dir, f"scene_{scene_id}")
        
        try:
            psnr, ssim, lpips_score = eval_physgaia_dir(
                gt_dir, pred_dir, scene_report_dir, target_scene_id=scene_id
            )
            
            all_results.append({
                "scene_id": scene_id,
                "psnr": psnr,
                "ssim": ssim,
                "lpips": lpips_score
            })
            
            total_psnr.append(psnr)
            total_ssim.append(ssim)
            total_lpips.append(lpips_score)
            
        except Exception as e:
            logging.error(f"Failed to evaluate scene {scene_id}: {e}")
            continue
    
    if total_psnr:
        ave_psnr = np.mean(total_psnr)
        ave_ssim = np.mean(total_ssim)
        ave_lpips = np.mean(total_lpips)
        
        summary_result = {
            "scene_id": "AVERAGE",
            "psnr": ave_psnr,
            "ssim": ave_ssim,
            "lpips": ave_lpips
        }
        all_results = [summary_result] + all_results
        
        df = pd.DataFrame(all_results)
        save_report_fn = osp.join(report_dir, "physgaia_all_scenes_metrics.xlsx")
        df.to_excel(save_report_fn, index=False)
        
        with open(osp.join(report_dir, "physgaia_all_scenes_summary.txt"), "w") as f:
            f.write(f"Evaluated scenes: {scene_ids}\n")
            f.write(f"Average PSNR: {ave_psnr:.6f}\n")
            f.write(f"Average SSIM: {ave_ssim:.6f}\n")
            f.write(f"Average LPIPS: {ave_lpips:.6f}\n")
        
        logging.info(f"Overall Average Metrics: PSNR={ave_psnr:.6f}, SSIM={ave_ssim:.6f}, LPIPS={ave_lpips:.6f}")
        return ave_psnr, ave_ssim, ave_lpips
    else:
        logging.error("No scenes were successfully evaluated")
        return None, None, None

if __name__ == "__main__":
    eval_physgaia_multi_scenes(
        gt_dir="PhysTrack/New/FLIP/torus_falling_into_water/render/test",
        pred_dir="/path/to/your/predictions", 
        report_dir="/path/to/evaluation/reports",
        scene_ids=[1, 2],
    )