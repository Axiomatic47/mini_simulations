# simple_validation.py
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def run_simplified_historical_validation():
    """
    Run a simplified version of historical validation that doesn't depend on imports.
    """
    print("Running simplified historical validation...")

    # 1. Generate synthetic historical data
    historical_data = generate_historical_data()

    # 2. Run a simple simulation
    simulation_results = run_simple_simulation(historical_data)

    # 3. Compare results
    error = calculate_comparison_error(historical_data, simulation_results)
    print(f"Validation error: {error:.2f}")

    # 4. Visualize comparison
    visualize_comparison(historical_data, simulation_results)

    # 5. Save results
    save_results(historical_data, simulation_results)

    return error


def generate_historical_data(start_year=1000, end_year=2020, interval=10):
    """Generate synthetic historical data."""
    years = np.arange(start_year, end_year + 1, interval)
    time_scale = (years - start_year) / (end_year - start_year)

    # Simple pattern generation
    knowledge = 10 * np.exp(2 * time_scale)
    suppression = 80 * np.exp(-time_scale) + 20
    intelligence = knowledge * (1 - 0.5 * suppression / 100)
    truth = 20 + 80 * (1 - np.exp(-time_scale))

    # Create DataFrame
    data = pd.DataFrame({
        "year": years,
        "knowledge_index": knowledge,
        "suppression_index": suppression,
        "intelligence_index": intelligence,
        "truth_index": truth
    })

    # Save data
    os.makedirs("validation/data", exist_ok=True)
    data.to_csv("validation/data/historical_data.csv", index=False)

    return data


def run_simple_simulation(historical_data):
    """Run a very simple simulation."""
    # Extract years from historical data
    years = historical_data["year"].values

    # Create simple simulation model
    time_scale = (years - years[0]) / (years[-1] - years[0])

    # Simple growth models
    knowledge = 5 * np.exp(2.2 * time_scale)
    suppression = 90 * np.exp(-1.1 * time_scale) + 10
    intelligence = knowledge * (1 - 0.6 * suppression / 100)
    truth = 15 + 85 * (1 - np.exp(-0.9 * time_scale))

    # Create DataFrame
    sim_results = pd.DataFrame({
        "year": years,
        "knowledge": knowledge,
        "suppression": suppression,
        "intelligence": intelligence,
        "truth": truth
    })

    return sim_results


def calculate_comparison_error(historical_data, simulation_results):
    """Calculate error between historical data and simulation results."""
    # Map simulation columns to historical column names
    sim_to_hist = {
        'knowledge': 'knowledge_index',
        'intelligence': 'intelligence_index',
        'suppression': 'suppression_index',
        'truth': 'truth_index'
    }

    # Calculate RMSE for each metric
    total_error = 0
    for sim_col, hist_col in sim_to_hist.items():
        # Normalize values to 0-100 scale
        hist_values = historical_data[hist_col].values
        sim_values = simulation_results[sim_col].values

        # Scale simulation values to match historical
        max_hist = np.max(hist_values)
        max_sim = np.max(sim_values)

        scaled_sim = sim_values * (max_hist / max_sim)

        # Calculate RMSE
        mse = np.mean((hist_values - scaled_sim) ** 2)
        rmse = np.sqrt(mse)

        print(f"{sim_col.capitalize()} RMSE: {rmse:.2f}")
        total_error += rmse

    # Average error
    return total_error / len(sim_to_hist)


def visualize_comparison(historical_data, simulation_results):
    """Visualize comparison between historical data and simulation."""
    # Map simulation columns to historical column names
    sim_to_hist = {
        'knowledge': 'knowledge_index',
        'intelligence': 'intelligence_index',
        'suppression': 'suppression_index',
        'truth': 'truth_index'
    }

    # Create grid of plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    # Plot each metric
    for i, (sim_col, hist_col) in enumerate(sim_to_hist.items()):
        ax = axes[i]

        # Scale simulation values to match historical range
        hist_values = historical_data[hist_col].values
        sim_values = simulation_results[sim_col].values

        max_hist = np.max(hist_values)
        max_sim = np.max(sim_values)

        scaled_sim = sim_values * (max_hist / max_sim)

        # Plot both
        ax.plot(historical_data['year'], hist_values, 'b-', label='Historical Data', linewidth=2)
        ax.plot(simulation_results['year'], scaled_sim, 'r--', label='Simulation', linewidth=2)

        # Add title and labels
        title = hist_col.replace('_', ' ').title()
        ax.set_title(title)
        ax.set_xlabel('Year')
        ax.set_ylabel('Index Value')
        ax.legend()
        ax.grid(True)

    # Add overall title
    fig.suptitle("Historical vs. Simulation Comparison", fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save figure
    os.makedirs("validation/reports/historical", exist_ok=True)
    plt.savefig("validation/reports/historical/comparison.png", dpi=300, bbox_inches='tight')

    # Don't call plt.show() if running in a script to avoid blocking


def save_results(historical_data, simulation_results):
    """Save validation results."""
    os.makedirs("validation/reports/historical", exist_ok=True)

    # Save historical data
    historical_data.to_csv("validation/reports/historical/historical_data.csv", index=False)

    # Save simulation results
    simulation_results.to_csv("validation/reports/historical/simulation_results.csv", index=False)

    # Save basic metrics
    metrics = {
        "rmse": calculate_comparison_error(historical_data, simulation_results),
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    pd.DataFrame([metrics]).to_csv("validation/reports/historical/error_metrics.csv", index=False)

    print("Results saved to validation/reports/historical/")


if __name__ == "__main__":
    run_simplified_historical_validation()