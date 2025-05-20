#!/bin/bash

# Define the base command and parameters

# Set CUDA device number from argument
cuda_device=$1

# Check if cuda_device is provided
if [ -z "$cuda_device" ]; then
    echo "Usage: $0 <cuda_device>"
    exit 1
fi

# Loop through iterations
for iter in {1..10}; do
    CUDA_VISIBLE_DEVICES=$cuda_device python traj_visualize_2.py --input_dir /131_data/intern/gunhee/PhysTrack/New/Pyro/smoke_box \
    --object_name smoke --camera_json camera_info_train.json --camera_entry train/0_001 --sample_frame 1 \
    --output_image smoke_box_between_cuda${cuda_device}_iter${iter}.png --start_frame 1 --end_frame 240 --sample_num 20 --use_cuda
done