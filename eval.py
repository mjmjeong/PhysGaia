import argparse
import json
import os.path as osp
from itertools import product
import os

import cv2
import imageio.v3 as iio
import numpy as np
import roma
import torch
from tqdm import tqdm

from flow3d.metrics import mLPIPS, mPSNR, mSSIM
from flow3d.transforms import rt_to_mat4, solve_procrustes
from loguru import logger as guru

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    type=str,
    help="Path to the data directory that contains all the sequences.",
)
parser.add_argument(
    "--result_dir",
    type=str,
    help="Path to the result directory that contains the results."
    "for batch evaluation, result_dir should contain subdirectories for each sequence. (result_dir/seq_name/results)"
    "for single sequence evaluation, result_dir should contain results directly (result_dir/results)",
)
parser.add_argument(
    "--seq_names",
    type=str,
    nargs="+",
    default=["casuals"],
    help="Sequence names to evaluate.",
)
args = parser.parse_args()
def load_data_dict(data_dir, res="", image_type="render/test",
                  depth_type="aligned_depth_anything_v2",
                  camera_type="droid_recon"):
    img_dir = f"{data_dir}/{image_type}/{res}"
    img_files = sorted(f for f in os.listdir(img_dir) if f.startswith("1_"))
    if not img_files:
        raise RuntimeError(f"No images found in {img_dir} with prefix '1_'")
    
    img_ext = os.path.splitext(img_files[0])[1]
    frame_names = [os.path.splitext(f)[0] for f in img_files]

    val_imgs = []
    for fn in tqdm(frame_names, desc="Loading images"):
        img_path = os.path.join(img_dir, f"{fn}{img_ext}")
        img = iio.imread(img_path)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        val_imgs.append(img)
    val_imgs = np.array(val_imgs)

    val_covisibles = np.ones_like(val_imgs[..., 0], dtype=np.float32) 

    return {
        "val_imgs": val_imgs,
        "val_covisibles": val_covisibles,
        "frame_names": frame_names,
    }

def load_result_dict(result_dir, frame_names):
    rgb_dir = osp.join(result_dir, "rgb")
    pred_val_imgs = []

    missing = 0
    for name in frame_names:
        found = False
        for ext in [".png", ".jpg", ".jpeg"]:
            img_path = osp.join(rgb_dir, f"{name}{ext}")
            if osp.exists(img_path):
                pred_val_imgs.append(iio.imread(img_path))
                found = True
                break
            if not found:
                print(f"Missing prediction: {name}")
                missing += 1
    print(f"Total missing: {missing}/{len(frame_names)}")

    pred_val_imgs = np.array(pred_val_imgs)

    return {
        "pred_val_imgs": pred_val_imgs,
    }


def evaluate_nv(data_dict, result_dict):
    if result_dict["pred_val_imgs"] is None:
        print("Cannot evaluate NV: missing predicted images")
        return 0.0, 0.0, 0.0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    psnr_metric = mPSNR().to(device)
    ssim_metric = mSSIM().to(device)
    lpips_metric = mLPIPS().to(device)

    val_imgs = torch.from_numpy(data_dict["val_imgs"])[..., :3].to(device)
    val_covisibles = torch.from_numpy(data_dict["val_covisibles"]).to(device)
    pred_val_imgs = torch.from_numpy(result_dict["pred_val_imgs"]).to(device)


    for i in range(len(val_imgs)):
        val_img = val_imgs[i].clamp(0, 255) / 255.0
        pred_val_img = pred_val_imgs[i].clamp(0, 255) / 255.0
        val_covisible = val_covisibles[i] 

        psnr_metric.update(val_img, pred_val_img)
        ssim_metric.update(val_img[None], pred_val_img[None])
        lpips_metric.update(val_img[None], pred_val_img[None])
    
    mpsnr = psnr_metric.compute().item()
    mssim = ssim_metric.compute().item()
    mlpips = lpips_metric.compute().item()
    
    print(f"NV mPSNR: {mpsnr:.4f}")
    print(f"NV mSSIM: {mssim:.4f}")
    print(f"NV mLPIPS: {mlpips:.4f}")
    
    return mpsnr, mssim, mlpips


if __name__ == "__main__":
    seq_names = args.seq_names

    mpsnr_all, mssim_all, mlpips_all = [], [], []

    for seq_name in seq_names:
        print("=========================================")
        print(f"Evaluating {seq_name}")
        print("=========================================")
        
        data_dir = osp.join(args.data_dir, seq_name)
        if not osp.exists(data_dir):
            data_dir = args.data_dir
        
        if not osp.exists(data_dir):
            print(f"Data directory {data_dir} does not exist, skipping.")
            continue
            
        result_dir = osp.join(args.result_dir, seq_name, "results")
        if not osp.exists(result_dir):
            result_dir = args.result_dir
        
        if not osp.exists(result_dir):
            print(f"Result directory {result_dir} does not exist, skipping.")
            continue
            
        # Load ground truth data
        # try:
        print("Loading data...")
        data_dict = load_data_dict(data_dir)
        frame_names = data_dict["frame_names"]
        
        print("Loading results...")
        result_dict = load_result_dict(result_dir, frame_names)
        
        # Evaluate novel view synthesis
        print("Evaluating novel view synthesis...")
        mpsnr, mssim, mlpips = evaluate_nv(data_dict, result_dict)
        mpsnr_all.append(mpsnr)
        mssim_all.append(mssim)
        mlpips_all.append(mlpips)
            
        # except Exception as e:
        #     print(f"Error evaluating {seq_name}: {e}")
        #     continue
    
    # Calculate and print overall metrics
    print("\n=========================================")
    print("Overall metrics:")
    print("=========================================")

    
    print(f"NV mPSNR: {np.mean(mpsnr_all):.4f}")
    print(f"NV mSSIM: {np.mean(mssim_all):.4f}")
    print(f"NV mLPIPS: {np.mean(mlpips_all):.4f}")
        
    # Save results to file
    results = {
        "seq_names": seq_names,
        "mpsnr": mpsnr_all,
        "mssim": mssim_all,
        "mlpips": mlpips_all,
    }
    
    os.makedirs("eval_results", exist_ok=True)
    with open(osp.join("eval_results", "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Results saved to eval_results/metrics.json")