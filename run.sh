#!/bin/bash

# Usage: ./run.sh <data_directory> [view_mode] [num_cameras]
#   data_directory : the base directory containing hypernerf and iphone folders
#   view_mode      : visualization mode: "both", "frustums", or "path" (default: both)
#   num_cameras    : number of cameras to sample (default: 10)

if [ -z "$1" ]; then
    echo "Usage: $0 <data_directory> [view_mode] [num_cameras]"
    exit 1
fi

DATA_DIR="$1"
VIEW_MODE=${2:-both}
NUM_CAMERAS=${3:-10}

RESULT_DIR="results"
mkdir -p "$RESULT_DIR"

# Process each subfolder in data/hypernerf
for folder in "$DATA_DIR/hypernerf"/*/; do
    if [ -d "$folder" ]; then
        echo "Processing hypernerf folder: $folder"
        folder_name=$(basename "$folder")
        OUTPUT_DIR="${RESULT_DIR}/hypernerf/${folder_name}"
        if [ -d "$OUTPUT_DIR" ]; then
            rm -rf "$OUTPUT_DIR"
        fi
        mkdir -p "$OUTPUT_DIR"
        python hypernerf_visualize.py --path "$folder" --view_mode "$VIEW_MODE" --num_cameras "$NUM_CAMERAS" --save_dir "$OUTPUT_DIR"
    fi
done

# Process each subfolder in data/iphone
for folder in "$DATA_DIR/iphone"/*/; do
    if [ -d "$folder" ]; then
        echo "Processing iphone folder: $folder"
        folder_name=$(basename "$folder")
        OUTPUT_DIR="${RESULT_DIR}/iphone/${folder_name}"
        if [ -d "$OUTPUT_DIR" ]; then
            rm -rf "$OUTPUT_DIR"
        fi
        mkdir -p "$OUTPUT_DIR"
        python dycheck_visualize.py --path "$folder" --view_mode "$VIEW_MODE" --num_cameras "$NUM_CAMERAS" --save_dir "$OUTPUT_DIR"
    fi
done

echo "All tasks completed."