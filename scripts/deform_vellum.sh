#!/bin/bash
BASE_PATH="/131_data/intern/gunhee/PhysTrack/New"
CUDA_DEVICE=0  # Parameterize CUDA device number here

SOLVER="Vellum"
ITEMS=("${SOLVER}/flags" "${SOLVER}/box_falling_into_cloth" "${SOLVER}/pinned_flag" "${SOLVER}/tube_flag" "${SOLVER}/cloth_falling_onto_statue")

# Mono - Traj
OUTPUT_PATH="output/mono/traj"
for ITEM in "${ITEMS[@]}"; do
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python train.py -s "${BASE_PATH}/${ITEM}" -m "${OUTPUT_PATH}/${ITEM}" \
        --init_with_traj --num_views single --eval --test_iterations 0
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python render.py -m "${OUTPUT_PATH}/${ITEM}" --mode render
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python metrics.py -m "${OUTPUT_PATH}/${ITEM}"
done

# Multi - Traj
OUTPUT_PATH="output/multi/traj"
for ITEM in "${ITEMS[@]}"; do
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python train.py -s "${BASE_PATH}/${ITEM}" -m "${OUTPUT_PATH}/${ITEM}" \
        --init_with_traj --num_views double --eval --test_iterations 0
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python render.py -m "${OUTPUT_PATH}/${ITEM}" --mode render
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python metrics.py -m "${OUTPUT_PATH}/${ITEM}"
done

# Mono - Colmap
OUTPUT_PATH="output/mono/colmap"
for ITEM in "${ITEMS[@]}"; do
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python train.py -s "${BASE_PATH}/${ITEM}" -m "${OUTPUT_PATH}/${ITEM}" \
        --num_views single --eval --test_iterations 0 
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python render.py -m "${OUTPUT_PATH}/${ITEM}" --mode render
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python metrics.py -m "${OUTPUT_PATH}/${ITEM}"
done

# Multi - Colmap
OUTPUT_PATH="output/multi/colmap"
for ITEM in "${ITEMS[@]}"; do
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python train.py -s "${BASE_PATH}/${ITEM}" -m "${OUTPUT_PATH}/${ITEM}" \
        --num_views double --eval --test_iterations 0
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python render.py -m "${OUTPUT_PATH}/${ITEM}" --mode render
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python metrics.py -m "${OUTPUT_PATH}/${ITEM}"
done