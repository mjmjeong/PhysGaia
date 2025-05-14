#!/usr/bin/env python
import os
import itertools
from pathlib import Path
import argparse
import json
from tabulate import tabulate

# Parse command line arguments
parser = argparse.ArgumentParser(description='Check if experiment directories exist in the target directory')
parser.add_argument('--output_dir', type=str, default="/131_data/wonjae/phystrack/outputs_deformable",
                   help='Base output directory to check')
args = parser.parse_args()

# Source paths from the original experiment script
source_paths = [
    "/131_data/intern/gunhee/PhysTrack/New/MPM/bouncing_balls",
    "/131_data/intern/gunhee/PhysTrack/New/MPM/falling_jelly",
    "/131_data/intern/gunhee/PhysTrack/New/MPM/cow",
    "/131_data/intern/gunhee/PhysTrack/New/MPM/pancake",
    "/131_data/intern/gunhee/PhysTrack/New/Pyro/smoke_box",
    "/131_data/intern/gunhee/PhysTrack/New/Pyro/smoke_fall",
    "/131_data/intern/gunhee/PhysTrack/New/Pyro/simple_smoke",
    "/131_data/intern/gunhee/PhysTrack/New/Pyro/pisa", 
     "/131_data/intern/gunhee/PhysTrack/New/FLIP/hanok",
    "/131_data/intern/gunhee/PhysTrack/New/FLIP/filling_cup",
     "/131_data/intern/gunhee/PhysTrack/New/FLIP/ship",
     "/131_data/intern/gunhee/PhysTrack/New/FLIP/torus_falling_into_water",
     "/131_data/intern/gunhee/PhysTrack/New/Vellum/box_falling_into_cloth",
    "/131_data/intern/gunhee/PhysTrack/New/Vellum/flags",
    "/131_data/intern/gunhee/PhysTrack/New/Vellum/pinned_flag",
    "/131_data/intern/gunhee/PhysTrack/New/Vellum/tube_flag", 
    "/131_data/intern/gunhee/PhysTrack/New/Vellum/cloth_falling_onto_statue", 
]

# Init options
init_options = [
    {"name": "traj", "args": ["--init_with_traj"]},
    {"name": "colmap", "args": []},  # No special initialization flag
    #{"name": "colmap_sparse", "args": ["--init_colmap_sparse"]}
]

# View options
num_views = [
    "single",
    "double",
    #"triple"
]

# Generate all experiment combinations
all_experiments = list(itertools.product(source_paths, init_options, num_views))
total_experiments = len(all_experiments)

print(f"Checking for {total_experiments} experiment directories in: {args.output_dir}")
print("-" * 80)

# Lists to track status
existing = []
missing = []

# Create a set of expected directory names
expected_dirs = set()

# Check each experiment combination
for i, (source_path, init_option, num_view) in enumerate(all_experiments):
    dataset_name = Path(source_path).name
    init_name = init_option["name"]
    dataset_category = Path(source_path).parent.name
    
    # Create the experiment name using the same format as the original script
    expname = f"{dataset_category}_{dataset_name}_{init_name}_{num_view}"
    exp_dir = os.path.join(args.output_dir, expname)
    
    # Add to expected directories set
    expected_dirs.add(expname)
    
    # Check if directory exists
    exists = os.path.isdir(exp_dir)
    
    # Store result
    exp_info = {
        "index": i,
        "experiment": expname,
        "directory": exp_dir,
        "exists": exists
    }
    
    if exists:
        existing.append(exp_info)
    else:
        missing.append(exp_info)

# Find unexpected directories (those in the output directory but not in our expected list)
unexpected = []
if os.path.exists(args.output_dir):
    for item in os.listdir(args.output_dir):
        full_path = os.path.join(args.output_dir, item)
        if os.path.isdir(full_path) and item not in expected_dirs:
            unexpected.append({
                "directory": full_path,
                "name": item
            })

# Print summary
print(f"Summary: {len(existing)}/{total_experiments} expected directories exist")
print(f"Missing: {len(missing)}/{total_experiments} directories")
print(f"Unexpected: {len(unexpected)} directories were found that don't match expected patterns")

# Show results in table format
if existing:
    print("\nExisting experiment directories:")
    table_data = [(i+1, exp["experiment"]) for i, exp in enumerate(existing)]
    print(tabulate(table_data, headers=["#", "Experiment Name"]))

if missing:
    print("\nMissing experiment directories:")
    table_data = [(i+1, exp["experiment"]) for i, exp in enumerate(missing)]
    print(tabulate(table_data, headers=["#", "Experiment Name"]))

if unexpected:
    print("\nUnexpected directories:")
    table_data = [(i+1, exp["name"]) for i, exp in enumerate(unexpected)]
    print(tabulate(table_data, headers=["#", "Directory Name"]))

# Save results to JSON file
results = {
    "output_directory": args.output_dir,
    "total_combinations": total_experiments,
    "existing_count": len(existing),
    "missing_count": len(missing),
    "unexpected_count": len(unexpected),
    "existing": existing,
    "missing": missing,
    "unexpected": unexpected
}

with open("experiment_directory_check.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDetailed results saved to: experiment_directory_check.json") 