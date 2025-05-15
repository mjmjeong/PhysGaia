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

def notify_slack(message):
    import requests

    # Slack webhook URL
    webhook_url = "https://hooks.slack.com/services/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"

    try:
        payload = {
            "text": f"PhysTrack: {message}."
        }
        response = requests.post(webhook_url, json=payload)
        if response.status_code != 200:
            logger.info(f"Failed to send Slack notification: {response.text}")
    except Exception as e:
        logger.info(f"Error sending Slack notification: {str(e)}")

if __name__ == "__main__":

    # Set up argument parser for resuming
    parser = argparse.ArgumentParser(description='Run multiple PhysTrack evaluation experiments')
    parser.add_argument('--resume_from', type=int, default=0, help='Resume from this experiment index')
    parser.add_argument('--log_dir', type=str, default='evaluate_logs', help='Directory to store logs')
    parser.add_argument('--check_only', action='store_true', help='Check only, do not run evaluation')
    args = parser.parse_args()

    # Base output directory for experiments
    output_base = "/131_data/wonjae/phystrack/outputs_deformable"

    # Create log directory if it doesn't exist
    os.makedirs(args.log_dir, exist_ok=True)

    # Set up logging
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(args.log_dir, f'evaluation_run_{timestamp}.log')
    run_summary_file = os.path.join(args.log_dir, f'evaluation_summary_{timestamp}.json')

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
        # uncomment all
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

    # Init options - these will be passed as flags to train.py
    # Since --init_with_traj is a store_true flag, and the other options don't exist,
    # we will use a dictionary to represent different initialization modes
    init_options = [
        {"name": "traj", "args": []},
        {"name": "colmap", "args": []},  # No special initialization flag
        #{"name": "colmap_sparse", "args": ["--init_colmap_sparse"]}
    ]

    # Single/double model options
    num_views = [
        "single",
        "double",
        #"triple"
    ]

    # Additional arguments for render.py
    render_args = [

    ]

    # Generate all combinations and prepare to track results
    all_experiments = list(itertools.product(source_paths, init_options, num_views))
    total_experiments = len(all_experiments)
    results = {
        "successful_render": [],
        "failed_render": [],
        "successful_metrics": [],
        "failed_metrics": [],
        "skipped": []
    }

    logger.info(f"Starting evaluation run with {total_experiments} total combinations")
    logger.info(f"Resuming from index {args.resume_from}")
    logger.info(f"Logs will be saved to {log_file}")

    # Record skipped experiments if resuming
    if args.resume_from > 0:
        for i in range(args.resume_from):
            if i < len(all_experiments):
                source_path, init_option, num_view = all_experiments[i]
                dataset_name = Path(source_path).name
                dataset_category = Path(source_path).parent.name
                init_name = init_option["name"]
                expname = f"{dataset_category}_{dataset_name}_{init_name}_{num_view}"
                
                results["skipped"].append({
                    "index": i,
                    "experiment": expname
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
        dataset_category = Path(source_path).parent.name
        expname = f"{dataset_category}_{dataset_name}_{init_name}_{num_view}"
        save_path = os.path.join(output_base, expname)
        
        # Check if results.json file exists 
        metric_json_path = os.path.join(save_path, "results.json")
        if os.path.exists(metric_json_path):
            logger.info(f"'results.json' file already exists at {metric_json_path}, skipping experiment {expname}")
            results["skipped"].append({
                "index": i,
                "experiment": expname,
                "reason": "results.json file already exists"
            })
            continue
            
        if args.check_only:
            logger.info(f"There are missing experiment {expname}")
            continue
        
        # Create output directory for logs specific to this experiment
        exp_log_dir = os.path.join(args.log_dir, expname)
        os.makedirs(exp_log_dir, exist_ok=True)
        
        # Create log files for this specific experiment
        render_stdout_log = os.path.join(exp_log_dir, "render_stdout.log")
        render_stderr_log = os.path.join(exp_log_dir, "render_stderr.log")
        metrics_stdout_log = os.path.join(exp_log_dir, "metrics_stdout.log")
        metrics_stderr_log = os.path.join(exp_log_dir, "metrics_stderr.log")
        
        # Build the render command
        render_cmd = [
            "python", "render.py",
            "--model_path", save_path,
        ]
        
        # Add initialization arguments
        render_cmd.extend(init_option["args"])
        
        # Add render-specific arguments
        render_cmd.extend(render_args)
        
        # Convert to string for logging
        render_cmd_str = " ".join(render_cmd)
        logger.info(f"Running render for experiment {i}/{total_experiments}: {expname}")
        logger.info(f"Command: {render_cmd_str}")
        
        # Record the start time
        render_start_time = datetime.datetime.now()
        render_success = False
        render_error_message = None
        
        # Execute render command
        try:
            with open(render_stdout_log, 'w') as stdout_file, open(render_stderr_log, 'w') as stderr_file:
                if DEBUG_ENUMERATION:
                    logger.info(f"DEBUG: Running command: {render_cmd_str}")
                else:
                    result = subprocess.run(
                        render_cmd, 
                        check=True,
                        stdout=stdout_file,
                        stderr=stderr_file,
                        text=True
                    )
            render_success = True
            logger.info(f"Successfully completed rendering for experiment: {expname}")
            results["successful_render"].append({
                "index": i,
                "experiment": expname,
                "command": render_cmd_str,
                "stdout_log": render_stdout_log,
                "stderr_log": render_stderr_log
            })
        except subprocess.CalledProcessError as e:
            render_error_message = str(e)
            logger.error(f"Error running render for experiment {expname}: {e}")
            results["failed_render"].append({
                "index": i,
                "experiment": expname,
                "command": render_cmd_str,
                "error": render_error_message,
                "stdout_log": render_stdout_log,
                "stderr_log": render_stderr_log
            })
        
        # Record the end time and duration for render
        render_end_time = datetime.datetime.now()
        render_duration = (render_end_time - render_start_time).total_seconds()
        
        # Log render runtime information
        logger.info(f"Rendering for {expname} completed in {render_duration:.2f} seconds")
        logger.info(f"Render Status: {'Success' if render_success else 'Failed'}")
        
        # Only run metrics if render was successful
        if render_success:
            # Build the metrics command
            metrics_cmd = [
                "python", "metrics.py",
                "--model_path", save_path
            ]
            
            # Convert to string for logging
            metrics_cmd_str = " ".join(metrics_cmd)
            logger.info(f"Running metrics for experiment {i}/{total_experiments}: {expname}")
            logger.info(f"Command: {metrics_cmd_str}")
            
            # Record the start time for metrics
            metrics_start_time = datetime.datetime.now()
            metrics_success = False
            metrics_error_message = None
            
            # Execute metrics command
            try:
                with open(metrics_stdout_log, 'w') as stdout_file, open(metrics_stderr_log, 'w') as stderr_file:
                    if DEBUG_ENUMERATION:
                        logger.info(f"DEBUG: Running command: {metrics_cmd_str}")
                    else:
                        result = subprocess.run(
                            metrics_cmd, 
                            check=True,
                            stdout=stdout_file,
                            stderr=stderr_file,
                            text=True
                        )
                metrics_success = True
                logger.info(f"Successfully completed metrics for experiment: {expname}")
                results["successful_metrics"].append({
                    "index": i,
                    "experiment": expname,
                    "command": metrics_cmd_str,
                    "stdout_log": metrics_stdout_log,
                    "stderr_log": metrics_stderr_log
                })
            except subprocess.CalledProcessError as e:
                metrics_error_message = str(e)
                logger.error(f"Error running metrics for experiment {expname}: {e}")
                results["failed_metrics"].append({
                    "index": i,
                    "experiment": expname,
                    "command": metrics_cmd_str,
                    "error": metrics_error_message,
                    "stdout_log": metrics_stdout_log,
                    "stderr_log": metrics_stderr_log
                })
            
            # Record the end time and duration for metrics
            metrics_end_time = datetime.datetime.now()
            metrics_duration = (metrics_end_time - metrics_start_time).total_seconds()
            
            # Log metrics runtime information
            logger.info(f"Metrics for {expname} completed in {metrics_duration:.2f} seconds")
            logger.info(f"Metrics Status: {'Success' if metrics_success else 'Failed'}")
        
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
    render_success_count = len(results["successful_render"])
    render_failure_count = len(results["failed_render"])
    metrics_success_count = len(results["successful_metrics"])
    metrics_failure_count = len(results["failed_metrics"])
    skipped_count = len(results["skipped"])

    logger.info("All evaluations completed!")
    logger.info(f"Summary:")
    logger.info(f"  Render: {render_success_count} successful, {render_failure_count} failed")
    logger.info(f"  Metrics: {metrics_success_count} successful, {metrics_failure_count} failed")
    logger.info(f"  Skipped: {skipped_count}")
    
    if render_success_count + render_failure_count > 0:
        logger.info(f"Render success rate: {render_success_count/(render_success_count+render_failure_count)*100:.2f}%")
    
    if metrics_success_count + metrics_failure_count > 0:
        logger.info(f"Metrics success rate: {metrics_success_count/(metrics_success_count+metrics_failure_count)*100:.2f}%")
    
    logger.info(f"Detailed evaluation summary saved to: {run_summary_file}")

    # Print failed experiments for easy reference
    if render_failure_count > 0:
        logger.info("\nFailed render experiments:")
        for i, exp in enumerate(results["failed_render"]):
            logger.info(f"{i+1}. {exp['experiment']} (index {exp['index']})")
            logger.info(f"   Command: {exp['command']}")
            logger.info(f"   Error: {exp['error']}")
    
    if metrics_failure_count > 0:
        logger.info("\nFailed metrics experiments:")
        for i, exp in enumerate(results["failed_metrics"]):
            logger.info(f"{i+1}. {exp['experiment']} (index {exp['index']})")
            logger.info(f"   Command: {exp['command']}")
            logger.info(f"   Error: {exp['error']}")
    
    notify_slack(f"Evaluation completed. Results saved for {total_experiments - skipped_count} experiments.") 