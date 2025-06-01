import subprocess
from concurrent.futures import ProcessPoolExecutor

import tyro
from pathlib import Path


def main(
    img_dirs: list[str],
    gpus: list[int],
    img_name: str = "train",  # "train" or "test" 
    mask_name: str = "masks",
    metric_depth_name: str = "unidepth_disp",
    intrins_name: str = "unidepth_intrins",
    mono_depth_model: str = "depth-anything-v2",
    slam_name: str = "droid_recon",
    track_model: str = "bootstapir",
    tapir_torch: bool = True,
    matching_pattern: str = "0_*",
    sparse_dirs: list[str] = None,
    camera_json_paths: list[str] = None,
    ply_paths: list[str] = None,
    auto_camera_json: bool = True,
    auto_ply: bool = True,
):

    if len(img_dirs) > 0 and img_name not in img_dirs[0]:
        raise ValueError(f"Expecting '{img_name}' in {img_dirs[0]} (should be 'train' or 'test')")

    mono_depth_name = mono_depth_model.replace("-", "_")
    
    # Auto-generate paths if enabled
    if auto_camera_json and camera_json_paths is None:
        camera_json_paths = []
        for img_dir in img_dirs:
            base_dir = img_dir.replace("/render/train", "").replace("/render/test", "")
            base_dir = base_dir.replace("\\render\\train", "").replace("\\render\\test", "")
            
            if "train" in img_dir:
                camera_json_paths.append(f"{base_dir}/camera_info_train_mono.json")
            elif "test" in img_dir:
                camera_json_paths.append(f"{base_dir}/camera_info_test.json")
            else:
                # Fallback
                camera_json_paths.append(f"{base_dir}/camera_info_train_mono.json")
    
    if auto_ply and ply_paths is None:
        ply_paths = []
        for img_dir in img_dirs:
            base_dir = img_dir.replace("/render/train", "").replace("/render/test", "")
            base_dir = base_dir.replace("\\render\\train", "").replace("\\render\\test", "")
            ply_paths.append(f"{base_dir}/point_cloud.ply")
    
    num_dirs = len(img_dirs)
    if camera_json_paths and len(camera_json_paths) != num_dirs:
        raise ValueError(f"camera_json_paths length ({len(camera_json_paths)}) must match img_dirs length ({num_dirs})")
    if ply_paths and len(ply_paths) != num_dirs:
        raise ValueError(f"ply_paths length ({len(ply_paths)}) must match img_dirs length ({num_dirs})")
    if sparse_dirs and len(sparse_dirs) != num_dirs:
        raise ValueError(f"sparse_dirs length ({len(sparse_dirs)}) must match img_dirs length ({num_dirs})")
    
    with ProcessPoolExecutor(max_workers=len(gpus)) as exc:
        for i, img_dir in enumerate(img_dirs):
            gpu = gpus[i % len(gpus)]
            img_dir = img_dir.rstrip("/")
            
            camera_json_path = camera_json_paths[i] if camera_json_paths else None
            ply_path = ply_paths[i] if ply_paths else None
            sparse_dir = sparse_dirs[i] if sparse_dirs else None
            
            exc.submit(
                process_sequence,
                gpu,
                img_dir,
                f"{base_dir}/{mask_name}/{img_name}", 
                f"{base_dir}/{metric_depth_name}/{img_name}",  
                f"{base_dir}/{intrins_name}/{img_name}",  
                f"{base_dir}/{mono_depth_name}/{img_name}",  
                f"{base_dir}/aligned_{mono_depth_name}/{img_name}",  
                f"{base_dir}/{slam_name}/{img_name}",  
                f"{base_dir}/{track_model}/{img_name}", 
                mono_depth_model,
                track_model,
                tapir_torch,
                matching_pattern,
                sparse_dir,
                camera_json_path,
                ply_path,
            )


def process_sequence(
    gpu: int,
    img_dir: str,
    mask_dir: str,
    metric_depth_dir: str,
    intrins_name: str,
    mono_depth_dir: str,
    aligned_depth_dir: str,
    slam_path: str,
    track_dir: str,
    depth_model: str = "depth-anything",
    track_model: str = "bootstapir",
    tapir_torch: bool = True,
    matching_pattern: str = "0_*",
    sparse_dir: str = None,
    camera_json_path: str = None,
    ply_path: str = None,
):
    dev_arg = f"CUDA_VISIBLE_DEVICES={gpu} USE_MEM_EFFICIENT_ATTENTION=0"

    mono_depth_cmd = (
        f"{dev_arg} python compute_depth.py --img_dir {img_dir} "
        f"--out_raw_dir {mono_depth_dir} --out_aligned_dir {aligned_depth_dir} "
        f"--model {depth_model} --matching_pattern '{matching_pattern}'"
    )
    
    if ply_path:
        mono_depth_cmd += f" --ply_path {ply_path}"
    elif sparse_dir:
        mono_depth_cmd += f" --sparse_dir {sparse_dir}"
    elif metric_depth_dir:
        mono_depth_cmd += f" --metric_dir {metric_depth_dir}"
    
    if camera_json_path:
        mono_depth_cmd += f" --camera_json_path {camera_json_path}"
    
    print(f"[GPU {gpu}] {mono_depth_cmd}")
    subprocess.call(mono_depth_cmd, shell=True, executable="/bin/bash")

    slam_cmd = (
        f"{dev_arg} python recon_with_depth.py --img_dir {img_dir} "
        f"--calib {camera_json_path if camera_json_path else intrins_name + '.json'} "
        f"--depth_dir {aligned_depth_dir} --out_path {slam_path}, --matching_pattern {matching_pattern}"
    )
    print(f"[GPU {gpu}] {slam_cmd}")
    subprocess.call(slam_cmd, shell=True, executable="/bin/bash")

    track_script = "compute_tracks_torch.py" if tapir_torch else "compute_tracks_jax.py"
    track_cmd = (
        f"{dev_arg} python {track_script} --image_dir {img_dir} "
        f"--mask_dir {mask_dir} --out_dir {track_dir} --model_type {track_model} --matching_pattern {matching_pattern}"
    )
    print(f"[GPU {gpu}] {track_cmd}")
    subprocess.call(track_cmd, shell=True, executable="/bin/bash")


if __name__ == "__main__":
    tyro.cli(main)

