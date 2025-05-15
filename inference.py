import torch
import os
import os.path as osp
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from flow3d.trainer import Trainer
from flow3d.data import CasualDataset
from flow3d.configs import LossesConfig, OptimizerConfig, SceneLRConfig
from flow3d.data.utils import to_device
from dataclasses import dataclass, asdict, field
from typing import Annotated
import tyro
from flow3d.data import (
    iPhoneDataConfig,
    DavisDataConfig,
    CustomDataConfig,
    NvidiaDataConfig
)
from loguru import logger as guru

from flow3d.init_utils import (
    init_bg,
    init_fg_from_tracks_3d,
    init_motion_params_with_procrustes,
    run_initial_optim,
    vis_init_params,
    init_trainable_poses,
)
from flow3d.scene_model import SceneModel
from flow3d.tensor_dataclass import StaticObservations, TrackObservations
from flow3d.vis.utils import get_server


@dataclass
class InferenceConfig:
    ckpt_path: str
    save_dir: str
    data: (
        Annotated[iPhoneDataConfig, tyro.conf.subcommand(name="iphone")]
        | Annotated[DavisDataConfig, tyro.conf.subcommand(name="davis")]
        | Annotated[CustomDataConfig, tyro.conf.subcommand(name="custom")]
        | Annotated[NvidiaDataConfig, tyro.conf.subcommand(name="nvidia")]
    )
    lr: SceneLRConfig = field(default_factory=SceneLRConfig)
    loss: LossesConfig = field(default_factory=LossesConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    num_fg: int = 40_000
    num_bg: int = 100_000
    num_motion_bases: int = 10
    use_2dgs: bool = False
    port: int | None = None
    vis_debug: bool = False 


def initialize_and_checkpoint_model(
    cfg: InferenceConfig,
    train_dataset,
    device: torch.device,
    ckpt_path: str,
    vis: bool = False,
    port: int | None = None,
):
    if os.path.exists(ckpt_path):
        guru.info(f"model checkpoint exists at {ckpt_path}")
        return

    fg_params, motion_bases, bg_params, tracks_3d = init_model_from_tracks(
        train_dataset,
        cfg.num_fg,
        cfg.num_bg,
        cfg.num_motion_bases,
        vis=vis,
        port=port,
    )
    # run initial optimization
    Ks = train_dataset.get_Ks().to(device)
    w2cs = train_dataset.get_w2cs().to(device)
    run_initial_optim(fg_params, motion_bases, tracks_3d, Ks, w2cs)
    if vis and cfg.port is not None:
        server = get_server(port=cfg.port)
        vis_init_params(server, fg_params, motion_bases)

    camera_poses = init_trainable_poses(w2cs)

    model = SceneModel(
        Ks, 
        w2cs, 
        fg_params, 
        motion_bases, 
        camera_poses,
        bg_params,
        cfg.use_2dgs,
    )

    guru.info(f"Saving initialization to {ckpt_path}")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": 0, "global_step": 0}, ckpt_path)


def init_model_from_tracks(
    train_dataset,
    num_fg: int,
    num_bg: int,
    num_motion_bases: int,
    vis: bool = False,
    port: int | None = None,
):
    tracks_3d = TrackObservations(*train_dataset.get_tracks_3d(num_fg))
    print(
        f"{tracks_3d.xyz.shape=} {tracks_3d.visibles.shape=} "
        f"{tracks_3d.invisibles.shape=} {tracks_3d.confidences.shape} "
        f"{tracks_3d.colors.shape}"
    )
    if not tracks_3d.check_sizes():
        import ipdb
        ipdb.set_trace()

    rot_type = "6d"
    cano_t = int(tracks_3d.visibles.sum(dim=0).argmax().item())

    guru.info(f"{cano_t=} {num_fg=} {num_bg=} {num_motion_bases=}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    motion_bases, motion_coefs, tracks_3d = init_motion_params_with_procrustes(
        tracks_3d, num_motion_bases, rot_type, cano_t, vis=vis, port=port
    )
    motion_bases = motion_bases.to(device)

    fg_params = init_fg_from_tracks_3d(cano_t, tracks_3d, motion_coefs)
    fg_params = fg_params.to(device)

    bg_params = None
    if num_bg > 0:
        bg_points = StaticObservations(*train_dataset.get_bkgd_points(num_bg))
        assert bg_points.check_sizes()
        bg_params = init_bg(bg_points)
        bg_params = bg_params.to(device)

    tracks_3d = tracks_3d.to(device)
    return fg_params, motion_bases, bg_params, tracks_3d


def render_dataset(model, dataset, device, save_dir):
    """CasualDataset 렌더링 및 이미지 저장"""
    
    results_dir = osp.join(save_dir, "results")
    rgb_dir = osp.join(results_dir, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)
    
    guru.info(f"Created directories: {results_dir}, {rgb_dir}")
    
    for i in tqdm(range(len(dataset)), desc="Rendering frames"):
        try:
            data = dataset[i]
            frame_name = dataset.frame_names[i]
            
            ts = torch.tensor(i, device=device)
            w2cs = data["w2cs"].to(device).unsqueeze(0)  # (1, 4, 4)
            Ks = data["Ks"].to(device).unsqueeze(0)  # (1, 3, 3)
            
            img = data["imgs"]  # (H, W, 3)
            img_wh = img.shape[1::-1]  # (W, H)
            
            guru.debug(f"Rendering frame {i} ({frame_name})")
            rendered = model.render(
                ts.item(),
                w2cs,
                Ks,
                img_wh,
                return_depth=True,
            )
            
            if "img" in rendered:
                import imageio
                img_path = osp.join(rgb_dir, f"{frame_name}.png")
                img_np = (rendered["img"][0].detach().cpu().numpy() * 255).astype(np.uint8)
                imageio.imwrite(img_path, img_np)
                guru.debug(f"Saved image to {img_path}")
            else:
                guru.warning(f"No 'img' in rendered result for frame {i}")
                
        except Exception as e:
            guru.error(f"Error rendering frame {i}: {e}")
            continue
    
    guru.info(f"All rendered images saved to {rgb_dir}")
    return {"status": "success"}


def save_motion_params(model, save_dir):
    """Save motion coefficients and motion bases to disk."""
    fg = model.fg
    bases = model.motion_bases

    os.makedirs(save_dir, exist_ok=True)

    motion_coefs = fg.get_coefs().detach().cpu().numpy()  # (G, B)
    motion_bases_rot = bases.params["rots"].detach().cpu().numpy()  # (B, 3, 3)
    motion_bases_trans = bases.params["transls"].detach().cpu().numpy()  # (B, 3)

    np.save(os.path.join(save_dir, "motion_coefs.npy"), motion_coefs)
    np.savez(
        os.path.join(save_dir, "motion_bases.npz"),
        rots=motion_bases_rot,
        transls=motion_bases_trans,
    )

    print("✅ Saved motion_coefs.npy and motion_bases.npz")


def main(cfg: InferenceConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_dataset = CasualDataset(**asdict(cfg.data))
    guru.info(f"Test dataset has {test_dataset.num_frames} frames")

    if not os.path.exists(cfg.ckpt_path):
        initialize_and_checkpoint_model(
            cfg,
            test_dataset,
            device,
            cfg.ckpt_path,
            vis=cfg.vis_debug,
            port=cfg.port,
        )

    trainer, _ = Trainer.init_from_checkpoint(
        cfg.ckpt_path,
        device,
        cfg.use_2dgs,
        cfg.lr,
        cfg.loss,
        cfg.optim,
        work_dir=cfg.save_dir,
        port=cfg.port,
    )

    save_motion_params(trainer.model, cfg.save_dir)
    result = render_dataset(trainer.model, test_dataset, device, cfg.save_dir)
    print("Rendering result:", result)


if __name__ == "__main__":
    tyro.cli(main)