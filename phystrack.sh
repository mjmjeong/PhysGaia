#!/bin/bash
BASE_PATH="/131_data/intern/gunhee/PhysTrack/New"
CUDA_DEVICE=0  # Parameterize CUDA device number here

ITEMS=("Vellum/flags" "Vellum/box_falling_into_cloth" "Vellum/pinned_flag" "Vellum/tube_flag" "Vellum/cloth_falling_onto_statue" \
    "FLIP/fountain" "FLIP/hanok" "FLIP/ship" "FLIP/torus_falling_into_water" \
    "MPM/bouncing_balls" "MPM/cow" "MPM/falling_jelly" "MPM/pancake" \
    "Pyro/pisa" "Pyro/simple_smoke" "Pyro/smoke_box" "Pyro/smoke_fall"
    )

if [ $# -ne 1 ]; then
    echo "Usage: $0 <index>"
    exit 1
fi

INDEX=$1

if ! [[ "$INDEX" =~ ^[0-9]+$ ]] || [ "$INDEX" -lt 0 ] || [ "$INDEX" -ge "${#ITEMS[@]}" ]; then
    echo "Error: Index must be a valid number between 0 and $((${#ITEMS[@]} - 1))"
    exit 1
fi

ITEM=${ITEMS[$INDEX]}

# Mono - Traj
OUTPUT_PATH="log/mono/traj"
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python train.py --eval --config configs/base_phystrack_full.json --model_path "${OUTPUT_PATH}/${ITEM}" \
    --source_path "${BASE_PATH}/${ITEM}" --num_views single --init_with_traj
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python test.py --quiet --eval --skip_train --configpath configs/base_phystrack_full.json \
    --model_path "${OUTPUT_PATH}/${ITEM}" --source_path "${BASE_PATH}/${ITEM}"


OUTPUT_PATH="log/multi/traj"
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python train.py --eval --config configs/base_phystrack_full.json --model_path "${OUTPUT_PATH}/${ITEM}" \
    --source_path "${BASE_PATH}/${ITEM}" --num_views double --init_with_traj
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python test.py --quiet --eval --skip_train --configpath configs/base_phystrack_full.json \
    --model_path "${OUTPUT_PATH}/${ITEM}" --source_path "${BASE_PATH}/${ITEM}"

OUTPUT_PATH="log/mono/colmap"
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python train.py --eval --config configs/base_phystrack_full.json --model_path "${OUTPUT_PATH}/${ITEM}" \
    --source_path "${BASE_PATH}/${ITEM}" --num_views single
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python test.py --quiet --eval --skip_train --configpath configs/base_phystrack_full.json \
    --model_path "${OUTPUT_PATH}/${ITEM}" --source_path "${BASE_PATH}/${ITEM}"

OUTPUT_PATH="log/multi/colmap"
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python train.py --eval --config configs/base_phystrack_full.json --model_path "${OUTPUT_PATH}/${ITEM}" \
    --source_path "${BASE_PATH}/${ITEM}" --num_views double
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python test.py --quiet --eval --skip_train --configpath configs/base_phystrack_full.json \
    --model_path "${OUTPUT_PATH}/${ITEM}" --source_path "${BASE_PATH}/${ITEM}"