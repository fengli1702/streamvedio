import os
import subprocess
import glob
import time
import sys
import argparse
from datetime import datetime

def run_experiment(script_path):
    """
    Run a single experiment script and capture its output
    
    Args:
        script_path: Path to the experiment script
    
    Returns:
        bool: True if experiment completed successfully, False otherwise
    """
    print(f"Running experiment: {os.path.basename(script_path)}")
    
    try:
        # Make sure the script is executable
        os.chmod(script_path, 0o755)
        
        # Run the experiment script and stream output in real-time
        process = subprocess.Popen(
            [script_path], 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='')
            sys.stdout.flush()
        
        # Wait for process to complete and get return code
        return_code = process.wait()
        
        return return_code == 0
            
    except Exception as e:
        print(f"Error running experiment {os.path.basename(script_path)}: {e}")
        return False

def main():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    all_experiment_scripts = sorted(glob.glob(os.path.join(script_dir, "exp*.sh")))
    
    if not all_experiment_scripts:
        print("No experiment scripts found in the experiments directory.")
        return
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run streaming-lvm experiments")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("-e", "--experiments", type=int, nargs="+", 
                        help="Specify experiment numbers to run (e.g., --experiments 1 3 5)")
    parser.add_argument("--list", action="store_true", 
                        help="List available experiments without running any")
    
    args = parser.parse_args()
    
    # List all available experiments
    if args.list:
        print(f"Found {len(all_experiment_scripts)} experiment scripts:")
        for i, script in enumerate(all_experiment_scripts):
            print(f"  {i+1}. {os.path.basename(script)}")
        return
    
    # Determine which experiments to run
    experiment_scripts = []
    if args.all:
        experiment_scripts = all_experiment_scripts
    elif args.experiments:
        for exp_num in args.experiments:
            if 0 <= exp_num <= len(all_experiment_scripts):
                experiment_scripts.append(all_experiment_scripts[exp_num-1])
            else:
                print(f"Warning: Experiment number {exp_num} is out of range, skipping.")
    else:
        parser.print_help()
        return
    
    if not experiment_scripts:
        print("No valid experiments selected to run.")
        return
    
    # Run selected experiments
    results = []    
    for script in experiment_scripts:
        success = run_experiment(script)
        results.append((os.path.basename(script), success))
    
    # Print summary
    successful = sum(1 for _, success in results if success)
    print(f"{successful}/{len(results)} experiments completed successfully")
    
    if successful < len(results):
        print("Failed experiments:")
        for script, success in results:
            if not success:
                print(f"  {script}")

if __name__ == "__main__":
    main()
