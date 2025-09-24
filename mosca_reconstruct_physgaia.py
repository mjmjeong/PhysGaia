#!/usr/bin/env python3
"""
PhysGaia-specific MoSca reconstruction script
This is a specialized version of lite_moca_reconstruct.py for PhysGaia datasets only.
"""

import torch
import os
import os.path as osp
import logging
import shutil
from datetime import datetime
from omegaconf import OmegaConf

from lib_prior.prior_loading import Saved2D
from lib_moca.moca import moca_solve
from lib_moca.camera import MonocularCameras

from data_utils.physgaia_helpers import load_physgaia_gt_poses

from recon_utils import (
    seed_everything,
    setup_recon_ws,
    auto_get_depth_dir_tap_mode,
    SEED,
)


def load_gt_cam(ws, fit_cfg):
    """Load PhysGaia ground truth camera poses"""
    logging.info(f"Loading PhysGaia ground truth camera poses")
    
    camera_json_path = getattr(fit_cfg, "camera_json_path", 
                              osp.join(ws, "camera_info_test.json"))
                              
    camera_train_json_path = getattr(fit_cfg, "camera_train_json_path", 
                                   osp.join(ws, "camera_info_train.json"))
    test_dir = getattr(fit_cfg, "test_dir", 
                      osp.join(ws, "render/test"))
    target_scene_id = getattr(fit_cfg, "target_scene_id", None) 

    (
        gt_training_cam_T_wi,
        gt_testing_cam_T_wi_list,
        gt_testing_tids_list,
        gt_testing_fns_list,
        gt_training_fov,
        gt_testing_fov_list,
        _,
        gt_testing_cxcy_ratio_list,
    ) = load_physgaia_gt_poses(ws, camera_json_path, test_dir, 
                       camera_train_json_path, target_scene_id)
    
    return (
        gt_training_cam_T_wi,
        gt_testing_cam_T_wi_list,
        gt_testing_tids_list,
        gt_testing_fns_list,
        gt_training_fov,
        gt_testing_fov_list,
        _,
        gt_testing_cxcy_ratio_list,
    )


def static_reconstruct(ws, log_path, fit_cfg):
    seed_everything(SEED)
    DEPTH_DIR, TAP_MODE = auto_get_depth_dir_tap_mode(ws, fit_cfg)
    DEPTH_BOUNDARY_TH = getattr(fit_cfg, "depth_boundary_th", 1.0)
    INIT_GT_CAMERA_FLAG = getattr(fit_cfg, "init_gt_camera", False)
    DEP_MEDIAN = getattr(fit_cfg, "dep_median", 1.0)

    EPI_TH = getattr(fit_cfg, "ba_epi_th", getattr(fit_cfg, "epi_th", 1e-3))
    logging.info(f"Static BA with EPI_TH={EPI_TH}")
    print(f"Static BA with EPI_TH={EPI_TH}")
    device = torch.device("cuda:0")

    s2d: Saved2D = (
        Saved2D(ws)
        .load_epi()
        .load_dep(DEPTH_DIR, DEPTH_BOUNDARY_TH)
        .normalize_depth(median_depth=DEP_MEDIAN)
        .recompute_dep_mask(depth_boundary_th=DEPTH_BOUNDARY_TH)
        .load_track(
            f"*uniform*{TAP_MODE}",
            min_valid_cnt=getattr(fit_cfg, "tap_loading_min_valid_cnt", 4),
        )
        .load_vos()
    )

    if INIT_GT_CAMERA_FLAG:
        # if start form gt camera, load gt camera here
        logging.info(f"Initializing from GT camera")
        (
            gt_training_cam_T_wi,
            gt_testing_cam_T_wi_list,
            gt_testing_tids_list,
            gt_testing_fns_list,
            gt_training_fov,
            gt_testing_fov_list,
            gt_training_cxcy_ratio,
            gt_testing_cxcy_ratio_list,
        ) = load_gt_cam(ws, fit_cfg)
        gt_fovdeg = float(gt_training_fov)
        cxcy_ratio = gt_training_cxcy_ratio[0]  # gt camera center
        if getattr(fit_cfg, "init_gt_camera_focal_only", False):
            logging.info(f"Only init focal length")
            cams = MonocularCameras(
                n_time_steps=s2d.T,
                default_H=s2d.H,
                default_W=s2d.W,
                fxfycxcy=[gt_fovdeg, gt_fovdeg] + cxcy_ratio,
                delta_flag=True,
                init_camera_pose=torch.eye(4)
                .to(gt_training_cam_T_wi)[None]
                .expand(len(gt_training_cam_T_wi) - 1, -1, -1),
                iso_focal=getattr(fit_cfg, "iso_focal", False),
                dataset_mode=getattr(fit_cfg, "mode", "iphone"),
            )
        else:
            cams = MonocularCameras(
                n_time_steps=s2d.T,
                default_H=s2d.H,
                default_W=s2d.W,
                fxfycxcy=[gt_fovdeg, gt_fovdeg] + cxcy_ratio,
                delta_flag=False,
                init_camera_pose=gt_training_cam_T_wi,
                iso_focal=getattr(fit_cfg, "iso_focal", False),
                dataset_mode=getattr(fit_cfg, "mode", "iphone"),
            )
    else:
        cams = None

    logging.info("*" * 20 + "MoCa BA" + "*" * 20)
    cams, s2d, _ = moca_solve(
        ws=log_path,
        s2d=s2d,
        device=device,
        epi_th=EPI_TH,
        ba_total_steps=getattr(fit_cfg, "ba_total_steps", 2000),
        ba_switch_to_ind_step=getattr(fit_cfg, "ba_switch_to_ind_step", 500),
        ba_depth_correction_after_step=getattr(
            fit_cfg, "ba_depth_correction_after_step", 500
        ),
        ba_max_frames_per_step=32,
        static_id_mode="raft" if s2d.has_epi else "track",
        # * robust setting
        robust_depth_decay_th=getattr(fit_cfg, "robust_depth_decay_th", 2.0),
        robust_depth_decay_sigma=getattr(fit_cfg, "robust_depth_decay_sigma", 1.0),
        robust_std_decay_th=getattr(fit_cfg, "robust_std_decay_th", 0.2),
        robust_std_decay_sigma=getattr(fit_cfg, "robust_std_decay_sigma", 0.2),
        #
        gt_cam=cams,
        iso_focal=getattr(fit_cfg, "iso_focal", False),
        rescale_gt_cam_transl=getattr(fit_cfg, "rescale_gt_cam_transl", False),
        ba_lr_cam_f=getattr(fit_cfg, "ba_lr_cam_f", 0.0003),
        ba_lr_dep_c=getattr(fit_cfg, "ba_lr_dep_c", 0.001),
        ba_lr_dep_s=getattr(fit_cfg, "ba_lr_dep_s", 0.001),
        ba_lr_cam_q=getattr(fit_cfg, "ba_lr_cam_q", 0.0003),
        ba_lr_cam_t=getattr(fit_cfg, "ba_lr_cam_t", 0.0003),
        #
        ba_lambda_flow=getattr(fit_cfg, "ba_lambda_flow", 1.0),
        ba_lambda_depth=getattr(fit_cfg, "ba_lambda_depth", 0.1),
        ba_lambda_small_correction=getattr(fit_cfg, "ba_lambda_small_correction", 0.03),
        ba_lambda_cam_smooth_trans=getattr(fit_cfg, "ba_lambda_cam_smooth_trans", 0.0),
        ba_lambda_cam_smooth_rot=getattr(fit_cfg, "ba_lambda_cam_smooth_rot", 0.0),
        #
        depth_filter_th=getattr(fit_cfg, "ba_depth_remove_th", -1.0),
        init_cam_with_optimal_fov_results=getattr(
            fit_cfg, "init_cam_with_optimal_fov_results", True
        ),
        # fov
        fov_search_fallback=getattr(fit_cfg, "ba_fov_search_fallback", 53.0),
        fov_search_N=getattr(fit_cfg, "ba_fov_search_N", 100),
        fov_search_start=getattr(fit_cfg, "ba_fov_search_start", 30.0),
        fov_search_end=getattr(fit_cfg, "ba_fov_search_end", 90.0),
        viz_valid_ba_points=getattr(fit_cfg, "ba_viz_valid_points", False),
    )  # ! S2D is changed becuase the depth is re-scaled
    
    # Copy bundle_cams.pth to photometric_cam.pth for compatibility
    bundle_cams_path = osp.join(log_path, "bundle", "bundle_cams.pth")
    photometric_cam_path = osp.join(log_path, "photometric_cam.pth")
    if osp.exists(bundle_cams_path) and not osp.exists(photometric_cam_path):
        shutil.copy2(bundle_cams_path, photometric_cam_path)
        logging.info(f"Copied {bundle_cams_path} to {photometric_cam_path}")

    # PhysGaia doesn't need camera metrics testing
    logging.info("Skipping camera metrics testing for PhysGaia")

    return s2d


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("MoCa Reconstruction Camera Only")
    parser.add_argument("--ws", type=str, help="Source folder", required=True)
    parser.add_argument("--cfg", type=str, help="profile yaml file path", required=True)
    parser.add_argument("--full", action="store_true", help="Run full MoSca reconstruction", default=False)
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation", default=False)
    parser.add_argument("--logdir", type=str, help="Pre-reconstructed log directory for eval_only", default=None)
    parser.add_argument("--eval_suffix", type=str, help="Suffix for evaluation results directory", default="")
    parser.add_argument("--no_viz", action="store_true", help="no viz", default=False)
    parser.add_argument("--tto", action="store_true", help="no viz", default=False)

    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.load(args.cfg)
    cli_cfg = OmegaConf.from_dotlist([arg.lstrip("--") for arg in unknown])
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Skip reconstruction if eval_only is True
    if not args.eval_only:
        logdir = setup_recon_ws(args.ws, fit_cfg=cfg)
        # Always run full reconstruction by default
        static_reconstruct(args.ws, logdir, cfg)
        # Import inside to avoid circular import
        from mosca_reconstruct import photometric_warmup, scaffold_reconstruct, photometric_reconstruct
        photometric_warmup(args.ws, logdir, cfg)
        scaffold_reconstruct(args.ws, logdir, cfg)
        photometric_reconstruct(args.ws, logdir, cfg)
    else:
        # * EVALUATION ONLY
        if args.logdir is None:
            logs_base_dir = osp.join(args.ws, "logs")
            if not osp.exists(logs_base_dir):
                raise ValueError(f"Logs directory not found: {logs_base_dir}. Please specify --logdir or run without --eval_only")
            
            # Find the most recent log directory
            log_dirs = [d for d in os.listdir(logs_base_dir) if osp.isdir(osp.join(logs_base_dir, d))]
            if not log_dirs:
                raise ValueError(f"No log directories found in {logs_base_dir}. Please specify --logdir or run without --eval_only")
            
            log_dirs.sort(reverse=True)
            logdir = osp.join(logs_base_dir, log_dirs[0])
            logging.info(f"Auto-selected most recent log directory: {logdir}")
        else:
            logdir = args.logdir
        logging.info(f"Using pre-reconstructed workspace: {logdir}")

    # PhysGaia main test (rendering + evaluation)
    from mosca_evaluate import test_main
    
    # Save evaluation results in separate directory
    if args.eval_only:
        if args.eval_suffix:
            eval_dir = osp.join(logdir, f"eval_{args.eval_suffix}")
        else:
            # Auto-generate based on time
            now = datetime.now()
            eval_timestamp = now.strftime("%Y%m%d_%H%M%S")
            eval_dir = osp.join(logdir, f"eval_{eval_timestamp}")
        os.makedirs(eval_dir, exist_ok=True)
        
        # Create symbolic links for required files in evaluation directory
        from lib_render.render_helper import GS_BACKEND
        required_files = [
            "photometric_cam.pth",
            f"photometric_s_model_{GS_BACKEND.lower()}.pth",
            f"photometric_d_model_{GS_BACKEND.lower()}.pth",
            "bundle/bundle_cams.pth", 
            "bundle/bundle.pth",
            "mosca/mosca.pth"
        ]
        
        for file_path in required_files:
            src_path = osp.join(logdir, file_path)
            dst_path = osp.join(eval_dir, file_path)
            
            if osp.exists(src_path):
                # Create directory
                os.makedirs(osp.dirname(dst_path), exist_ok=True)
                # Create symbolic link (skip if already exists)
                if not osp.exists(dst_path):
                    try:
                        os.symlink(src_path, dst_path)
                    except OSError:
                        # Copy if symbolic link fails
                        shutil.copy2(src_path, dst_path)
                logging.info(f"Linked {file_path} to evaluation directory")
            else:
                logging.warning(f"Required file not found: {src_path}")
        
        logging.info(f"Evaluation results will be saved to: {eval_dir}")
    else:
        eval_dir = logdir
        
    test_main(
        cfg,
        saved_dir=eval_dir,
        data_root=args.ws,
        device=torch.device("cuda"),
        tto_flag=args.tto,
        eval_also_dyncheck_non_masked=False,
        skip_test_gen=False,
    )

    # PhysGaia visualization (if needed)
    if not args.no_viz:
        from mosca_viz import viz_main
        viz_main(
            save_dir=osp.join(logdir, "viz"),
            log_dir=logdir,
            cfg_fn=args.cfg,
            N=getattr(cfg, "viz_N", 5),
            move_angle_deg=getattr(cfg, "viz_move_angle_deg", 10.0),
            H_3d=getattr(cfg, "viz_H_3d", 960),
            W_3d=getattr(cfg, "viz_W_3d", 960),
            fov_3d=getattr(cfg, "viz_fov_3d", 70),
            back_ratio_3d=getattr(cfg, "viz_back_ratio_3d", 1.5),
            up_ratio=getattr(cfg, "viz_up_ratio", 0.05),
            bg_color=getattr(cfg, "photo_default_bg_color", [0.0, 0.0, 0.0]),
        )
