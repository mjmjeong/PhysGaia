#!/bin/bash

BASE_PATH="/131_data/intern/gunhee/PhysTrack/New"
CUDA_DEVICE=0  # Parameterize CUDA device number here

# Define solvers and their corresponding items
declare -A SOLVERS_ITEMS


SOLVERS_ITEMS["FLIP"]="${SOLVER}/fountain ${SOLVER}/hanok ${SOLVER}/ship ${SOLVER}/torus_falling_into_water"
SOLVERS_ITEMS["MPM"]="${SOLVER}/bouncing_balls ${SOLVER}/cow ${SOLVER}/falling_jelly ${SOLVER}/pancake"
SOLVERS_ITEMS["Pyro"]="${SOLVER}/pisa ${SOLVER}/simple_smoke ${SOLVER}/smoke_box ${SOLVER}/smoke_fall"
SOLVERS_ITEMS["Vellum"]="${SOLVER}/flags ${SOLVER}/box_falling_into_cloth ${SOLVER}/pinned_flag ${SOLVER}/tube_flag ${SOLVER}/cloth_falling_onto_statue"

# Get solver name and index passed as arguments
SOLVER_NAME=$1
INDEX=$2

# Ensure the solver is valid
if [ -z "${SOLVERS_ITEMS[$SOLVER_NAME]}" ]; then
  echo "Error: Invalid solver name $SOLVER_NAME"
  exit 1
fi

# Get the items for the specified solver
ITEMS=(${SOLVERS_ITEMS[$SOLVER_NAME]})

# Ensure the index is valid for the given solver
if [ -z "${ITEMS[$INDEX]}" ]; then
  echo "Error: Invalid index $INDEX for solver $SOLVER_NAME"
  exit 1
fi

# Get the item corresponding to the passed index
ITEM="${SOLVER_NAME}${ITEMS[$INDEX]}"
echo "Processing item: ${ITEM}"



##############################################################################

# Mono - Traj
OUTPUT_PATH="output/mono/traj"

CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python train.py -s "${BASE_PATH}/${ITEM}" -m "${OUTPUT_PATH}/${ITEM}" \
        --init_with_traj --num_views single --eval --test_iterations 0
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python render.py -m "${OUTPUT_PATH}/${ITEM}" --mode render
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python metrics.py -m "${OUTPUT_PATH}/${ITEM}"


## Multi - Traj
#OUTPUT_PATH="output/multi/traj"
#for ITEM in "${ITEMS[@]}"; do
#    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python train.py -s "${BASE_PATH}/${ITEM}" -m "${OUTPUT_PATH}/${ITEM}" \
#        --init_with_traj --num_views double --eval --test_iterations 0
#    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python render.py -m "${OUTPUT_PATH}/${ITEM}" --mode render
#    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python metrics.py -m "${OUTPUT_PATH}/${ITEM}"
#done

# Mono - Colmap
#OUTPUT_PATH="output/mono/colmap"
#for ITEM in "${ITEMS[@]}"; do
#    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python train.py -s "${BASE_PATH}/${ITEM}" -m "${OUTPUT_PATH}/${ITEM}" \
#        --num_views single --eval --test_iterations 0 
#    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python render.py -m "${OUTPUT_PATH}/${ITEM}" --mode render
#    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python metrics.py -m "${OUTPUT_PATH}/${ITEM}"
#done

# Multi - Colmap
#OUTPUT_PATH="output/multi/colmap"
#for ITEM in "${ITEMS[@]}"; do
#    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python train.py -s "${BASE_PATH}/${ITEM}" -m "${OUTPUT_PATH}/${ITEM}" \
#        --num_views double --eval --test_iterations 0
#    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python render.py -m "${OUTPUT_PATH}/${ITEM}" --mode render
#    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python metrics.py -m "${OUTPUT_PATH}/${ITEM}"
#done
