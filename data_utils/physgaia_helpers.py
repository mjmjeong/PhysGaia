import json
import numpy as np
import torch
import os.path as osp
from glob import glob
import imageio
import os

def load_camera_params_from_json(json_path, img_files, img_dir, use_width_for_focal=False, target_scene_id=None):
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
        path_parts = file_path.split('/')
        if len(path_parts) >= 2:
            filename_part = path_parts[-1]
            scene_id = int(filename_part.split('_')[0])
            
            if target_scene_id is None or scene_id == target_scene_id:
                image_name = os.path.basename(file_path) + ".png"
                frame_dict[image_name] = frame
    
    num_imgs = len(img_files)
    Ks = np.zeros((num_imgs, 3, 3))
    c2ws = np.zeros((num_imgs, 4, 4))  
    
    for idx, img_file in enumerate(img_files):
        image_name = os.path.basename(img_file)
        
        if image_name not in frame_dict:
            raise ValueError(f"Image {image_name} not found in camera JSON")
        
        frame = frame_dict[image_name]
        image_path = os.path.join(img_dir, image_name)
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        img = imageio.imread(image_path)
        h, w = img.shape[:2]
        
        if use_width_for_focal:
            fx = fy = 0.5 * w / np.tan(angle_x / 2)  
        else:
            fx = fy = 0.5 * h / np.tan(angle_x / 2)
        cx, cy = w / 2, h / 2
        
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        Ks[idx] = K
        
        T_c2w_nerf = np.array(frame["transform_matrix"])
        flip = np.diag([1, -1, -1, 1])
        T_c2w_cv = T_c2w_nerf @ flip
        
        c2ws[idx] = T_c2w_cv
    
    tstamps = np.arange(num_imgs, dtype=np.int32)
    return (
        torch.from_numpy(c2ws).float(), 
        torch.from_numpy(Ks).float(),
        torch.from_numpy(tstamps),
    )

def load_physgaia_gt_poses(data_root, camera_json_path, test_dir, camera_train_json_path, target_scene_id=None):
    train_img_dir = osp.join(data_root, "render", "train")
    if not osp.exists(train_img_dir):
        train_img_dir = osp.join(data_root, "images")
    if not osp.exists(train_img_dir):
        train_img_dir = osp.join(data_root, "train")
    
    train_img_files = sorted(glob(osp.join(train_img_dir, "0_*.png")))
    if len(train_img_files) == 0:
        raise ValueError(f"No training images found in {train_img_dir}")
    
    train_c2ws, train_Ks, train_tstamps = load_camera_params_from_json(
        camera_train_json_path, train_img_files, train_img_dir, use_width_for_focal=False, target_scene_id=0
    )
    
    with open(camera_train_json_path, 'r') as f:
        train_meta = json.load(f)
    gt_training_fov = np.rad2deg(train_meta["camera_angle_x"])
    
    test_img_files = sorted(glob(osp.join(test_dir, "*.png")))
    
    scenes = {}
    for file_path in test_img_files:
        filename = osp.basename(file_path)
        name_without_ext = filename.replace('.png', '')
        
        try:
            parts = name_without_ext.split('_')
            if len(parts) >= 2:
                scene_id = int(parts[0])
                timestamp = int(parts[1])
                
                if scene_id not in scenes:
                    scenes[scene_id] = []
                
                scenes[scene_id].append({
                    'filename': filename,
                    'timestamp': timestamp,
                    'scene_id': scene_id,
                    'file_path': file_path
                })
        except ValueError:
            continue
    
    if target_scene_id is None:
        target_scene_id = min(scenes.keys())
    elif target_scene_id not in scenes:
        raise ValueError(f"Scene {target_scene_id} not found. Available: {list(scenes.keys())}")
    
    test_scene_data = sorted(scenes[target_scene_id], key=lambda x: x['timestamp'])
    test_scene_files = [item['file_path'] for item in test_scene_data]
    test_timestamps = [item['timestamp'] for item in test_scene_data]
    test_filenames = [item['filename'].replace('.png', '') for item in test_scene_data]
    
    test_c2ws, test_Ks, test_tstamps = load_camera_params_from_json(
        camera_json_path, test_scene_files, test_dir, use_width_for_focal=False, target_scene_id=target_scene_id
    )
    
    with open(camera_json_path, 'r') as f:
        test_meta = json.load(f)
    gt_testing_fov_list = [np.rad2deg(test_meta["camera_angle_x"])]
    
    min_test_timestamp = min(test_timestamps)
    normalized_test_timestamps = [t - min_test_timestamp for t in test_timestamps]
    
    assert train_tstamps.min().item() == 0, f"Training timestamps should start from 0"
    assert min(normalized_test_timestamps) == 0, f"Test timestamps should start from 0"
    
    img_h, img_w = imageio.imread(test_scene_files[0]).shape[:2]
    cx = test_Ks[0][0, 2].item()
    cy = test_Ks[0][1, 2].item()
    cx_ratio = cx / img_w
    cy_ratio = cy / img_h
    
    train_img_h, train_img_w = imageio.imread(train_img_files[0]).shape[:2]
    train_cx = train_Ks[0][0, 2].item()
    train_cy = train_Ks[0][1, 2].item()
    train_cx_ratio = train_cx / train_img_w
    train_cy_ratio = train_cy / train_img_h
    
    return (
        train_c2ws,
        [test_c2ws],
        [normalized_test_timestamps],
        [test_filenames],
        gt_training_fov,
        gt_testing_fov_list,   
        [[train_cx_ratio, train_cy_ratio]],
        [[cx_ratio, cy_ratio]],
    )