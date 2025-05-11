#!/usr/bin/env python
import os
import subprocess
import itertools
import logging
import datetime
import argparse
import sys
import json
from pathlib import Path

DEBUG_ENUMERATION = False

# Set up argument parser for resuming
parser = argparse.ArgumentParser(description='Run multiple PhysTrack training experiments')
parser.add_argument('--resume_from', type=int, default=0, help='Resume from this experiment index')
parser.add_argument('--log_dir', type=str, default='experiment_logs', help='Directory to store logs')
args = parser.parse_args()

# Base output directory for experiments
output_base = "/131_data/wonjae/phystrack/outputs_deformable"

# Create log directory if it doesn't exist
os.makedirs(args.log_dir, exist_ok=True)

# Set up logging
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(args.log_dir, f'experiment_run_{timestamp}.log')
run_summary_file = os.path.join(args.log_dir, f'experiment_summary_{timestamp}.json')

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# These lists will be manually set later - using placeholders for now
source_paths = [
    "/131_data/intern/gunhee/PhysTrack/New/MPM/bouncing_balls",
    #"/131_data/intern/gunhee/PhysTrack/New/MPM/falling_jelly",
    "/131_data/intern/gunhee/PhysTrack/New/MPM/cow",
    "/131_data/intern/gunhee/PhysTrack/New/MPM/pancake",
    "/131_data/intern/gunhee/PhysTrack/New/Pyro/smoke_box",
    "/131_data/intern/gunhee/PhysTrack/New/Pyro/smoke_fall",
    "/131_data/intern/gunhee/PhysTrack/New/Pyro/simple_smoke",
    "/131_data/intern/gunhee/PhysTrack/New/Pyro/pisa",
    "/131_data/intern/gunhee/PhysTrack/New/FLIP/hanok",
    #"/131_data/intern/gunhee/PhysTrack/New/FLIP/filling_cup",
    "/131_data/intern/gunhee/PhysTrack/New/FLIP/ship",
    "/131_data/intern/gunhee/PhysTrack/New/FLIP/torus_falling_into_water",
    "/131_data/intern/gunhee/PhysTrack/New/Vellum/box_falling_into_cloth",
    "/131_data/intern/gunhee/PhysTrack/New/Vellum/flags",
    "/131_data/intern/gunhee/PhysTrack/New/Vellum/pinned_flag",
    "/131_data/intern/gunhee/PhysTrack/New/Vellum/tube_flag",
    "/131_data/intern/gunhee/PhysTrack/New/Vellum/cloth_falling_onto_statue",
]

# Init options - these will be passed as flags to train.py
# Since --init_with_traj is a store_true flag, and the other options don't exist,
# we will use a dictionary to represent different initialization modes
init_options = [
    {"name": "traj", "args": ["--init_with_traj"]},
    #{"name": "colmap", "args": []}  # No special initialization flag
]

# Single/double model options
num_views = [
    "single",
    #"double"
]

# Additional arguments that might vary
additional_args = [
    #"--configs arguments/phystrack/debug.py",
    "--eval",
]

# Generate all combinations and prepare to track results
all_experiments = list(itertools.product(source_paths, init_options, num_views))
total_experiments = len(all_experiments)
results = {
    "successful": [],
    "failed": [],
    "skipped": []
}

logger.info(f"Starting experiment run with {total_experiments} total combinations")
logger.info(f"Resuming from index {args.resume_from}")
logger.info(f"Logs will be saved to {log_file}")

# Record skipped experiments if resuming
if args.resume_from > 0:
    for i in range(args.resume_from):
        if i < len(all_experiments):
            source_path, init_option, num_view = all_experiments[i]
            dataset_name = Path(source_path).name
            init_name = init_option["name"]
            expname = f"{dataset_name}_{init_name}_{num_view}"
            
            # Build command for logging
            cmd_parts = ["python train.py", 
                        f"-s {source_path}", 
                        f"--model_path {output_base}/{expname}",
                        f"--num_views {num_view}"]
            cmd_parts.extend(init_option["args"])
            cmd_parts.extend(additional_args)
            cmd_str = " ".join(cmd_parts)
            
            results["skipped"].append({
                "index": i,
                "experiment": expname,
                "command": cmd_str
            })
            logger.info(f"Skipping experiment {i}/{total_experiments}: {expname}")

# Run experiments
for i, (source_path, init_option, num_view) in enumerate(all_experiments):
    # Skip if before resume point
    if i < args.resume_from:
        continue
    
    # Create a descriptive experiment name
    dataset_name = Path(source_path).name
    init_name = init_option["name"]
    
    expname = f"{dataset_name}_{init_name}_{num_view}"
    save_path = os.path.join(output_base, expname)
    
    # Create output directory for logs specific to this experiment
    exp_log_dir = os.path.join(args.log_dir, expname)
    try:
        os.makedirs(exp_log_dir, exist_ok=False)
    except FileExistsError:
        logger.info(f"Experiment {expname} already exists, skipping")
        continue
    
    # Create log files for this specific experiment
    stdout_log = os.path.join(exp_log_dir, "stdout.log")
    stderr_log = os.path.join(exp_log_dir, "stderr.log")
    
    # Build the command
    cmd = [
        "python", "train.py",
        "-s", source_path,
        "--model_path", save_path,
        "--num_views", num_view
    ]
    
    # Add initialization arguments
    cmd.extend(init_option["args"])
    
    # Add model-specific arguments
    cmd.extend(additional_args)
    
    # Convert to string for logging
    cmd_str = " ".join(cmd)
    logger.info(f"Running experiment {i}/{total_experiments}: {expname}")
    logger.info(f"Command: {cmd_str}")
    
    # Record the start time
    start_time = datetime.datetime.now()
    success = False
    error_message = None
    
    # Execute the command and capture output
    try:
        with open(stdout_log, 'w') as stdout_file, open(stderr_log, 'w') as stderr_file:
            if DEBUG_ENUMERATION:
                logger.info(f"DEBUG: Running command: {cmd_str}")
            else:
                result = subprocess.run(
                    cmd, 
                    check=True,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    text=True
                )
        success = True
        logger.info(f"Successfully completed experiment: {expname}")
        results["successful"].append({
            "index": i,
            "experiment": expname,
            "command": cmd_str,
            "stdout_log": stdout_log,
            "stderr_log": stderr_log
        })
    except subprocess.CalledProcessError as e:
        error_message = str(e)
        logger.error(f"Error running experiment {expname}: {e}")
        results["failed"].append({
            "index": i,
            "experiment": expname,
            "command": cmd_str,
            "error": error_message,
            "stdout_log": stdout_log,
            "stderr_log": stderr_log
        })
    
    # Record the end time and duration
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Log runtime information
    logger.info(f"Experiment {expname} completed in {duration:.2f} seconds")
    logger.info(f"Status: {'Success' if success else 'Failed'}")
    if error_message:
        logger.info(f"Error details: {error_message}")
    logger.info('-' * 80)  # Separator for clarity
    
    # Save the current results after each experiment
    with open(run_summary_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "total_experiments": total_experiments,
            "completed": i + 1,
            "results": results
        }, f, indent=2)

# Final summary
success_count = len(results["successful"])
failure_count = len(results["failed"])
skipped_count = len(results["skipped"])

logger.info("All experiments completed!")
logger.info(f"Summary: {success_count} successful, {failure_count} failed, {skipped_count} skipped")
logger.info(f"Success rate: {success_count/(success_count+failure_count)*100:.2f}%")
logger.info(f"Detailed run summary saved to: {run_summary_file}")

# Print failed experiments for easy reference
if failure_count > 0:
    logger.info("\nFailed experiments:")
    for i, exp in enumerate(results["failed"]):
        logger.info(f"{i+1}. {exp['experiment']} (index {exp['index']})")
        logger.info(f"   Command: {exp['command']}")
        logger.info(f"   Error: {exp['error']}")
