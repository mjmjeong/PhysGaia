init_with_traj = True

grid_args = dict(
    canonical_num_levels=16,
    canonical_level_dim=2,
    canonical_base_resolution=16,
    canonical_desired_resolution=2048,
    canonical_log2_hashmap_size=19,

    deform_num_levels=32,
    deform_level_dim=2,
    deform_base_resolution=16,
    deform_desired_resolution=128,
    deform_log2_hashmap_size=19,

    bound=0.8,
)

network_args = dict(
    depth=1,
    width=256,
    directional=True,
)

grid_lr_scale = 10.0
network_lr_scale = 1.0

lambda_spatial_tv = 0.1
spatial_downsample_ratio = 1.0
spatial_perturb_range = 1e-3

lambda_temporal_tv = 0.1
temporal_downsample_ratio = 1.0
temporal_perturb_range = [1e-3, 1e-3, 1e-3, 0.0]