#!/bin/bash

# List of dataset paths to process
datasets=(
  "/131_data/intern/gunhee/PhysTrack/New/MPM/bouncing_balls"
  "/131_data/intern/gunhee/PhysTrack/New/MPM/falling_jelly" #(NOT READY)
  "/131_data/intern/gunhee/PhysTrack/New/MPM/cow"
  "/131_data/intern/gunhee/PhysTrack/New/MPM/pancake"
  "/131_data/intern/gunhee/PhysTrack/New/Pyro/smoke_box"
  "/131_data/intern/gunhee/PhysTrack/New/Pyro/smoke_fall"
  "/131_data/intern/gunhee/PhysTrack/New/Pyro/simple_smoke"
  "/131_data/intern/gunhee/PhysTrack/New/Pyro/pisa"
  "/131_data/intern/gunhee/PhysTrack/New/FLIP/hanok"
  "/131_data/intern/gunhee/PhysTrack/New/FLIP/fountain" #(NOT READY)
  #"/131_data/intern/gunhee/PhysTrack/New/FLIP/ship" (DONE)
  #"/131_data/intern/gunhee/PhysTrack/New/FLIP/torus_falling_into_water" (DONE)
  #"/131_data/intern/gunhee/PhysTrack/New/Vellum/box_falling_into_cloth" (DONE)
  #"/131_data/intern/gunhee/PhysTrack/New/Vellum/flags" (DONE)
  #"/131_data/intern/gunhee/PhysTrack/New/Vellum/pinned_flag" (DONE)
  "/131_data/intern/gunhee/PhysTrack/New/Vellum/tube_flag"
  "/131_data/intern/gunhee/PhysTrack/New/Vellum/cloth_falling_onto_statue"
)

# Loop through each dataset and run colmap.sh
for dataset in "${datasets[@]}"; do
  echo "Processing dataset: $dataset"
  bash colmap.sh "$dataset" phystrack
  echo "Finished processing: $dataset"
  echo "----------------------------------------"
done

echo "All datasets have been processed!" 