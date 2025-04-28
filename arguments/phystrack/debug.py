_base_ = './phystrack_default.py'

ModelParams = dict(
        init_with_traj=False,
        white_background=False
        )

ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 32,
     'resolution': [64, 64, 64, 25]
    },

    # deformation_lr_init = 0.001,
    # deformation_lr_final = 0.001,
    # deformation_lr_delay_mult = 0.01,
    # grid_lr_init = 0.001,
    # grid_lr_final = 0.001,
)
