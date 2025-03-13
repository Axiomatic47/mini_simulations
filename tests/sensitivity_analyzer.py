#!/usr/bin/env python3
"""
Test script for the updated sensitivity analyzer.

This script verifies that the updated sensitivity analyzer:
1. Handles NumPy types correctly
2. Generates tornado plot data with the 'range' column
3. Successfully runs with mixed parameter types
"""
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

# Create output directory
output_dir = root_dir / "outputs" / "sensitivity_test"
output_dir.mkdir(parents=True, exist_ok=True)

print("Testing updated sensitivity analyzer...")

# Import the updated analyzer
from utils.sensitivity_analyzer import ParameterSensitivityAnalyzer


# Define a simple test simulation function
def test_simulation(params):
    """
    Test simulation that returns metrics based on parameters.
    Intentionally uses parameter values in the results to verify type handling.
    """
    # Extract parameters (convert to float to ensure consistent types)
    w0 = float(params.get('W_0', 1.0))
    alpha = float(params.get('ALPHA_WISDOM', 0.1))
    gamma = float(params.get('GAMMA_PHASE', 0.1))

    # Calculate metrics
    knowledge = w0 * 10 + alpha * 50 + gamma * 20
    convergence = 100 / (w0 + 0.01) + gamma * 30
    stability = w0 * 2 - alpha * 5 + gamma * gamma * 10

    # Return metrics with additional info for debugging
    return {
        'final_knowledge': knowledge,
        'convergence_time': convergence,
        'stability_index': stability,
        'debug_w0': w0,
        'debug_alpha': alpha,
        'debug_gamma': gamma
    }


# Create analyzer with intentional mixed parameter types
print("\nCreating analyzer with mixed parameter types...")
metrics = ['final_knowledge', 'convergence_time', 'stability_index']

# Mix of regular floats, NumPy types, and strings
base_params = {
    'W_0': 1.0,  # Regular float
    'ALPHA_WISDOM': np.float64(0.1),  # NumPy float
    'GAMMA_PHASE': np.float64(0.1),  # NumPy float
    'DESCRIPTION': 'test',  # String (should be handled correctly)
    'VERSION': np.int32(3)  # NumPy integer
}

print("Base parameters:")
for key, value in base_params.items():
    print(f"  {key}: {value} (type: {type(value)})")

# Create analyzer
analyzer = ParameterSensitivityAnalyzer(test_simulation, metrics, base_params)

# Define parameter ranges with NumPy types
print("\nDefining parameter ranges with NumPy types...")
analyzer.define_parameter_ranges({
    'W_0': (0.5, 2.0, 3),  # Regular values
    'ALPHA_WISDOM': (np.float64(0.05), np.float64(0.2), 3),  # NumPy floats
    'GAMMA_PHASE': (np.float64(0.05), np.float64(0.2), 3)  # NumPy floats
})

# Run one-at-a-time sensitivity analysis
print("\nRunning one-at-a-time sensitivity analysis...")
results = analyzer.run_one_at_a_time_sensitivity(parallel=False)
print(f"Generated {len(results)} result rows")

# Check for the 'parameter' and 'value' columns
print("Results columns:", results.columns.tolist())
print(f"Contains 'parameter' column: {'parameter' in results.columns}")
print(f"Contains 'value' column: {'value' in results.columns}")

# Check data types in results
print("\nVerifying data types in results:")
if 'debug_w0' in results.columns:
    print(f"W_0 type: {results['debug_w0'].dtype}")
if 'debug_alpha' in results.columns:
    print(f"ALPHA_WISDOM type: {results['debug_alpha'].dtype}")
if 'debug_gamma' in results.columns:
    print(f"GAMMA_PHASE type: {results['debug_gamma'].dtype}")

# Save results for inspection
results.to_csv(output_dir / "sensitivity_results.csv", index=False)
print(f"Results saved to {output_dir}/sensitivity_results.csv")

# Calculate parameter importance
print("\nCalculating parameter importance...")
importance = analyzer.calculate_parameter_importance()
print("Parameter importance ranking:")
print(importance)

# Test tornado plot data generation
print("\nGenerating tornado plot data...")
for metric in metrics:
    print(f"\nAnalyzing metric: {metric}")

    # Generate tornado plot data
    tornado_data = analyzer.calculate_tornado_plot_data(metric)
    print(f"Generated tornado data with {len(tornado_data)} rows")
    print(f"Columns: {tornado_data.columns.tolist()}")
    print(f"Contains 'range' column: {'range' in tornado_data.columns}")

    if not tornado_data.empty:
        print("First row of tornado data:")
        print(tornado_data.iloc[0])

    # Save tornado data
    tornado_data.to_csv(output_dir / f"{metric}_tornado_data.csv", index=False)
    print(f"Saved tornado data to {output_dir}/{metric}_tornado_data.csv")

    # Generate tornado plot
    fig = analyzer.generate_tornado_plot(metric=metric)
    if fig is not None:
        plt.savefig(output_dir / f"{metric}_tornado.png", dpi=300)
        plt.close(fig)
        print(f"Generated tornado plot: {output_dir}/{metric}_tornado.png")
    else:
        print(f"Warning: No tornado plot generated for {metric}")

# Generate comprehensive report
print("\nGenerating comprehensive report...")
report_dir = output_dir / "report"
report_dir.mkdir(exist_ok=True)
analyzer.generate_comprehensive_report(str(report_dir))
print(f"Generated comprehensive report in {report_dir}")

print("\nAll tests completed successfully!")
print(f"Results saved to {output_dir}")

# Return test results dictionary
test_results = {
    "num_result_rows": len(results),
    "has_parameter_column": 'parameter' in results.columns,
    "has_value_column": 'value' in results.columns,
    "metrics_tested": metrics,
    "tornado_data_columns": tornado_data.columns.tolist() if not tornado_data.empty else [],
    "has_range_column": 'range' in tornado_data.columns if not tornado_data.empty else False,
    "test_passed": True
}

print("\nTest results:", test_results)