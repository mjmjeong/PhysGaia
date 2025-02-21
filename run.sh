#!/bin/bash

# Check if the data directory is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <data_directory>"
    exit 1
fi

DATA_DIR="$1"

# Create a directory to store all results
RESULT_DIR="results"
mkdir -p "$RESULT_DIR"

# Process each subfolder in data/hypernerf
for folder in "$DATA_DIR/hypernerf"/*/; do
    if [ -d "$folder" ]; then
        echo "Processing hypernerf folder: $folder"
        folder_name=$(basename "$folder")
        # Create a subdirectory for the current folder's output
        OUTPUT_DIR="${RESULT_DIR}/${folder_name}_hypernerf"
        mkdir -p "$OUTPUT_DIR"
        # Run hypernerf_visualize.py and save the geometries into the OUTPUT_DIR
        python hypernerf_visualize.py --path "$folder" --view_mode both --num_cameras 10 --save_dir "$OUTPUT_DIR"
    fi
done

# Process each subfolder in data/iphone
for folder in "$DATA_DIR/iphone"/*/; do
    if [ -d "$folder" ]; then
        echo "Processing iphone folder: $folder"
        folder_name=$(basename "$folder")
        OUTPUT_DIR="${RESULT_DIR}/${folder_name}_iphone"
        mkdir -p "$OUTPUT_DIR"
        python hypernerf_visualize.py --path "$folder" --view_mode both --num_cameras 10 --save_dir "$OUTPUT_DIR"
    fi
done

echo "All tasks completed."