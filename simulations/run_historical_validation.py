#!/usr/bin/env python3
"""
Script to run the historical validation module for the Axiomatic Intelligence Growth Simulation.
This compares simulation results with historical data and optimizes parameters.
"""

import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import validation module
from config.historical_validation import HistoricalValidation

# Import data generator (for creating example data if needed)
from config.historical_data_generator import generate_historical_data, visualize_historical_data


def main(args):
    """Run the historical validation process."""
    # Ensure output directories exist
    plots_dir = Path(args.output_dir) / 'plots'
    data_dir = Path(args.output_dir) / 'data'
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if historical data exists, generate if it doesn't
    historical_data_path = args.data
    if not os.path.exists(historical_data_path):
        print(f"Historical data file not found at {historical_data_path}")

        if args.generate_data:
            print("Generating synthetic historical data...")
            data = generate_historical_data(
                start_year=args.start_year,
                end_year=args.end_year,
                interval=args.interval,
                add_noise=True,
                save_path=historical_data_path
            )

            # Visualize generated data
            visualize_historical_data(
                data,
                save_path=plots_dir / "historical_data_visualization.png"
            )
        else:
            print("Use --generate-data flag to create synthetic historical data, or provide existing data with --data")
            return 1

    # Create validation object
    print(f"Creating historical validation object with data from {historical_data_path}")
    validator = HistoricalValidation(
        data_source=historical_data_path,
        start_year=args.start_year,
        end_year=args.end_year,
        interval=args.interval
    )

    # Run initial simulation
    print("Running simulation with default parameters...")
    validator.run_simulation()

    # Calculate initial error
    initial_error = validator.calculate_error()
    print(f"Error with default parameters: {initial_error:.4f}")

    # Create initial comparison visualization
    print("Generating initial comparison visualization...")
    validator.visualize_comparison(
        save_path=plots_dir / "historical_validation_initial.png"
    )

    # Optimize parameters if requested
    if args.optimize:
        print("Optimizing parameters...")

        # Define parameters to optimize based on command line arguments
        if args.optimize_params:
            params_to_optimize = args.optimize_params.split(',')
            print(f"Optimizing specified parameters: {params_to_optimize}")
        else:
            # Default parameters to optimize
            params_to_optimize = [
                "K_0", "S_0", "knowledge_growth_rate", "truth_adoption_rate",
                "suppression_decay", "alpha_feedback", "beta_feedback"
            ]
            print(f"Optimizing default parameters: {params_to_optimize}")

        # Run optimization
        optimized_params = validator.optimize_parameters(
            params_to_optimize=params_to_optimize,
            method=args.optimize_method
        )

        # Calculate final error
        final_error = validator.calculate_error()
        print(f"Error after optimization: {final_error:.4f}")
        print(f"Improvement: {initial_error - final_error:.4f} ({(1 - final_error / initial_error) * 100:.1f}%)")

        # Save optimized parameters
        pd.DataFrame([optimized_params]).to_csv(
            data_dir / "optimized_parameters.csv", index=False
        )
        print(f"Optimized parameters saved to {data_dir / 'optimized_parameters.csv'}")

    # Generate final visualizations
    print("Generating final visualizations...")
    validator.visualize_comparison(
        save_path=plots_dir / "historical_validation_comparison.png"
    )
    validator.visualize_key_periods(
        save_path=plots_dir / "historical_validation_periods.png"
    )

    # Export results
    print("Exporting simulation results...")
    validator.export_to_csv(
        data_dir / "historical_validation_results.csv"
    )

    print("Historical validation complete!")
    return 0


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run historical validation for the simulation framework")

    # Data parameters
    parser.add_argument('--data', default='outputs/data/historical_data.csv',
                        help='Path to historical data CSV file')
    parser.add_argument('--generate-data', action='store_true',
                        help='Generate synthetic historical data if none exists')
    parser.add_argument('--start-year', type=int, default=1000,
                        help='Start year for historical comparison')
    parser.add_argument('--end-year', type=int, default=2020,
                        help='End year for historical comparison')
    parser.add_argument('--interval', type=int, default=10,
                        help='Year interval for data points')

    # Optimization parameters
    parser.add_argument('--optimize', action='store_true',
                        help='Run parameter optimization')
    parser.add_argument('--optimize-params', type=str,
                        help='Comma-separated list of parameters to optimize')
    parser.add_argument('--optimize-method', default='SLSQP',
                        choices=['SLSQP', 'L-BFGS-B', 'TNC', 'COBYLA'],
                        help='Optimization method to use')

    # Output parameters
    parser.add_argument('--output-dir', default='outputs',
                        help='Directory for output files')

    # Parse arguments
    args = parser.parse_args()

    # Run main function
    sys.exit(main(args))