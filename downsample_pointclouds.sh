#!/bin/bash

# List of dataset paths to process
dataset_paths=(
    "/131_data/intern/gunhee/PhysTrack/New/MPM/bouncing_balls"
    "/131_data/intern/gunhee/PhysTrack/New/MPM/falling_jelly"
    "/131_data/intern/gunhee/PhysTrack/New/MPM/cow"
    "/131_data/intern/gunhee/PhysTrack/New/MPM/pancake"
    "/131_data/intern/gunhee/PhysTrack/New/Pyro/smoke_box"
    "/131_data/intern/gunhee/PhysTrack/New/Pyro/smoke_fall"
    "/131_data/intern/gunhee/PhysTrack/New/Pyro/simple_smoke"
    "/131_data/intern/gunhee/PhysTrack/New/Pyro/pisa"
    "/131_data/intern/gunhee/PhysTrack/New/FLIP/hanok"
    "/131_data/intern/gunhee/PhysTrack/New/FLIP/ship"
    "/131_data/intern/gunhee/PhysTrack/New/FLIP/torus_falling_into_water"
    "/131_data/intern/gunhee/PhysTrack/New/Vellum/box_falling_into_cloth"
    "/131_data/intern/gunhee/PhysTrack/New/Vellum/flags"
    "/131_data/intern/gunhee/PhysTrack/New/Vellum/pinned_flag"
    "/131_data/intern/gunhee/PhysTrack/New/Vellum/tube_flag"
    "/131_data/intern/gunhee/PhysTrack/New/Vellum/cloth_falling_onto_statue"
)

# Iterate through each dataset path
for dataset_path in "${dataset_paths[@]}"; do
    echo "Processing $dataset_path"
    
    # Process colmap_double directory
    echo "  Downsampling colmap_double"
    python scripts/downsample_point.py "$dataset_path/colmap_double/dense/0/fused.ply" "$dataset_path/colmap_double/dense/0/fused_downsampled.ply"
    
    # Process colmap_single directory
    echo "  Downsampling colmap_single"
    python scripts/downsample_point.py "$dataset_path/colmap_single/dense/0/fused.ply" "$dataset_path/colmap_single/dense/0/fused_downsampled.ply"
done

echo "All pointclouds downsampling completed." 