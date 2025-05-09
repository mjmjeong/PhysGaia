# CUDA_VISIBLE_DEVICES=1,2,3 python compute_metric_depth.py \
#                 --img-dir /131_data/intern/gunhee/PhysTrack/New/MPM/bouncing_balls/render/train \
#                 --depth-dir /131_data/intern/gunhee/PhysTrack/New/MPM/bouncing_balls/flow3d_preprocessed/unidepth_disp \
#                 --intrins-file /131_data/intern/gunhee/PhysTrack/New/MPM/bouncing_balls/flow3d_preprocessed/unidepth_intrins.json

CUDA_VISIBLE_DEVICES=1,2,3 python compute_depth.py \
                --img_dir /131_data/intern/gunhee/PhysTrack/New/MPM/bouncing_balls/render/train \
                --out_raw_dir  /131_data/intern/gunhee/PhysTrack/New/MPM/bouncing_balls/flow3d_preprocessed/depth_anything_v2/0 \
                --out_aligned_dir /131_data/intern/gunhee/PhysTrack/New/MPM/bouncing_balls/flow3d_preprocessed/aligned_depth_anything_v2/0 \
                --model depth-anything-v2 \
                --matching_pattern "0_*"