import os
import pickle
import numpy as np
import torch
import tyro
from loguru import logger as guru

from flow3d.data import (
    BaseDataset,
    DavisDataConfig,
    CustomDataConfig,
    get_train_val_datasets,
    iPhoneDataConfig,
    NvidiaDataConfig,
)
from flow3d.tensor_dataclass import TrackObservations
from dataclasses import dataclass
from typing import Annotated

from flow3d.configs import LossesConfig, OptimizerConfig, SceneLRConfig

torch.set_float32_matmul_precision("high")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

@dataclass
class SaveTracksConfig:
    data: (
        Annotated[iPhoneDataConfig, tyro.conf.subcommand(name="iphone")]
        | Annotated[DavisDataConfig, tyro.conf.subcommand(name="davis")]
        | Annotated[CustomDataConfig, tyro.conf.subcommand(name="custom")]
        | Annotated[NvidiaDataConfig, tyro.conf.subcommand(name="nvidia")]
    )
    num_fg: int = 40_000
    output_dir: str = "saved_tracks"


import torch
import os
import os.path as osp
import numpy as np

def save_motion_params(model, save_dir):
    """Save motion coefficients and motion bases to npy files."""
    fg = model.fg
    bases = model.motion_bases

    save_path_coef = osp.join(save_dir, "motion_coefs.npy")
    save_path_basis = osp.join(save_dir, "motion_bases.npy")

    motion_coefs = fg.get_coefs().detach().cpu().numpy()  # (G, B)
    motion_bases_rot = bases.params["rots"].detach().cpu().numpy()  # (B, 3, 3)
    motion_bases_trans = bases.params["transls"].detach().cpu().numpy()  # (B, 3)

    os.makedirs(save_dir, exist_ok=True)
    np.save(save_path_coef, motion_coefs)
    np.savez(
        save_path_basis,
        rots=motion_bases_rot,
        transls=motion_bases_trans
    )

    return {
        "motion_coefs_path": save_path_coef,
        "motion_bases_path": save_path_basis,
        "shape_coefs": motion_coefs.shape,
        "shape_rots": motion_bases_rot.shape,
        "shape_transls": motion_bases_trans.shape,
    }



def save_tracks_data(cfg: SaveTracksConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    train_dataset, _, _, _ = get_train_val_datasets(cfg.data, load_val=False)
    guru.info(f"Dataset loaded with {train_dataset.num_frames} frames")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tracks_3d = TrackObservations(*train_dataset.get_tracks_3d(cfg.num_fg))
    
    print(f"{tracks_3d.xyz.shape=} {tracks_3d.visibles.shape=} "
          f"{tracks_3d.invisibles.shape=} {tracks_3d.confidences.shape=} "
          f"{tracks_3d.colors.shape=}")
    
    num_frames = tracks_3d.xyz.shape[0]
    num_points = tracks_3d.xyz.shape[1]
    guru.info(f"Number of frames: {num_frames}")
    guru.info(f"Number of points: {num_points}")
    
    tracks_data = {
        'xyz': tracks_3d.xyz.cpu().numpy(),
        'visibles': tracks_3d.visibles.cpu().numpy(),
        'invisibles': tracks_3d.invisibles.cpu().numpy(),
        'confidences': tracks_3d.confidences.cpu().numpy() if tracks_3d.confidences is not None else None,
        'colors': tracks_3d.colors.cpu().numpy() if tracks_3d.colors is not None else None
    }
    
    save_path = os.path.join(cfg.output_dir, 'tracks_3d_data.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(tracks_data, f)
    guru.info(f"Saved tracks_3d data to {save_path}")
    
    np_save_path = os.path.join(cfg.output_dir, 'tracks_3d_data.npz')
    np.savez_compressed(np_save_path, **tracks_data)
    guru.info(f"Saved tracks_3d data to {np_save_path} (NumPy format)")
    
    visible_points_per_frame = tracks_3d.visibles.sum(dim=1).cpu().numpy()
    guru.info(f"Average visible points per frame: {visible_points_per_frame.mean():.2f}")
    guru.info(f"Min visible points in a frame: {visible_points_per_frame.min()}")
    guru.info(f"Max visible points in a frame: {visible_points_per_frame.max()}")


if __name__ == "__main__":
    save_tracks_data(tyro.cli(SaveTracksConfig))