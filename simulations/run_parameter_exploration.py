import os
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime
import shutil

# Base directory for storing multiple simulation results
RESULTS_DIR = os.path.join('outputs', 'parameter_exploration')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Parameters to explore
diffusion_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
max_interaction_distances = [3.0, 5.0, 7.0, 10.0]
initial_num_civilizations = [3, 5, 10, 15]
use_dimensional_analysis = [True, False]

# Create a log file
log_file = open(os.path.join(RESULTS_DIR, 'exploration_log.txt'), 'w')


# Function to run a simulation with specific parameters
def run_simulation(params):
    # Create run ID and directory
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(RESULTS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Log parameters
    param_str = ', '.join([f"{k}={v}" for k, v in params.items()])
    log_file.write(f"Starting {run_id}: {param_str}\n")
    log_file.flush()

    # TODO: Modify this to actually use the parameters in your simulation
    # For now, we'll just run the standard simulation and capture its output
    try:
        # Run simulation script
        result = subprocess.run(['python', 'simulations/multi_civilization_simulation.py'],
                                capture_output=True, text=True, check=True)

        # Save stdout and stderr
        with open(os.path.join(run_dir, 'stdout.log'), 'w') as f:
            f.write(result.stdout)
        with open(os.path.join(run_dir, 'stderr.log'), 'w') as f:
            f.write(result.stderr)

        # Copy simulation output files to run directory
        for file_name in ['multi_civilization_statistics.csv',
                          'multi_civilization_events.csv',
                          'multi_civilization_stability.csv']:
            source = os.path.join('outputs', 'data', file_name)
            dest = os.path.join(run_dir, file_name)
            if os.path.exists(source):
                shutil.copy(source, dest)

        # Save parameters
        pd.DataFrame([params]).to_csv(os.path.join(run_dir, 'parameters.csv'), index=False)

        log_file.write(f"Completed {run_id} successfully\n")
        return True
    except Exception as e:
        log_file.write(f"Error in {run_id}: {str(e)}\n")
        return False


# Run simulations with different parameter combinations
# For demonstration, we'll just do a small subset of combinations
for diffusion_rate in diffusion_rates[:2]:  # Just first 2 values
    for max_distance in max_interaction_distances[:2]:  # Just first 2 values
        params = {
            'diffusion_rate': diffusion_rate,
            'max_interaction_distance': max_distance,
            'initial_num_civilizations': 5,  # Fixed value for this example
            'use_dimensional_analysis': True  # Fixed value for this example
        }
        run_simulation(params)

log_file.write("Parameter exploration completed\n")
log_file.close()
print("Parameter exploration completed. Results in", RESULTS_DIR)