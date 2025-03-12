import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from utils.circuit_breaker import CircuitBreaker  # Import circuit breaker utility
from utils.dimensional_consistency import (
    Dimension, DimensionalValue,
    check_dimensional_consistency
)  # Import dimensional consistency utilities

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Setup output directories
BASE_DIR = Path(__file__).resolve().parent.parent
plots_dir = BASE_DIR / 'outputs' / 'plots'
data_dir = BASE_DIR / 'outputs' / 'data'
plots_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)

# Create circuit breaker for numerical stability
circuit_breaker = CircuitBreaker(
    threshold=1e-6,
    max_value=1e6,
    min_value=-1e6,
    max_rate_of_change=1e3
)

# Define bounds for outputs
MAX_INTELLIGENCE = 100.0
MAX_TRUTH = 100.0
MAX_SUPPRESSION = 100.0
MIN_VALUE = 0.0

# Enable or disable dimensional analysis
use_dimensional_analysis = True

# Define time range (e.g., years from 1000 AD to 2025 AD)
years = np.linspace(1000, 2025, 1025)


# Safely apply sin with bounds
def safe_sin(x):
    """Apply sin function with bounds to prevent extreme values."""
    return np.clip(np.sin(x), -1.0, 1.0)


# Safely apply exponential with bounds
def safe_exp(x):
    """Apply exponential function with bounds to prevent overflow."""
    # Limit the exponent to avoid overflow
    x = np.clip(x, -50.0, 50.0)
    return np.clip(np.exp(x), 0.0, 1e10)


# Define historical pattern functions with dimensional consistency
def historical_intelligence_growth(years):
    """Calculate intelligence growth with dimensional analysis."""
    raw_values = 50 * safe_sin(0.01 * (years - 1000)) - 20 * safe_exp(-0.002 * (years - 1000))
    clipped_values = np.clip(raw_values, MIN_VALUE, MAX_INTELLIGENCE)

    if use_dimensional_analysis:
        # Convert to dimensional values
        return DimensionalValue(clipped_values, Dimension.INTELLIGENCE)
    else:
        return clipped_values


def historical_truth_adoption(years):
    """Calculate truth adoption with dimensional analysis."""
    raw_values = 40 * (1 - safe_exp(-0.005 * (years - 1000)))
    clipped_values = np.clip(raw_values, MIN_VALUE, MAX_TRUTH)

    if use_dimensional_analysis:
        # Convert to dimensional values
        return DimensionalValue(clipped_values, Dimension.TRUTH)
    else:
        return clipped_values


def historical_suppression_feedback(years):
    """Calculate suppression feedback with dimensional analysis."""
    raw_values = safe_exp(-0.005 * (years - 1000)) + 0.1 * safe_sin(0.02 * (years - 1000))
    clipped_values = np.clip(raw_values, MIN_VALUE, MAX_SUPPRESSION)

    if use_dimensional_analysis:
        # Convert to dimensional values
        return DimensionalValue(clipped_values, Dimension.SUPPRESSION)
    else:
        return clipped_values


def historical_civilization_oscillation(years):
    """Calculate civilization oscillation with dimensional analysis."""
    raw_values = 0.05 * safe_exp(-0.002 * (years - 1000)) * safe_sin(0.01 * (years - 1000))
    clipped_values = np.clip(raw_values, -1.0, 1.0)

    if use_dimensional_analysis:
        # This is dimensionless (no specific physical dimension)
        return DimensionalValue(clipped_values, Dimension.DIMENSIONLESS)
    else:
        return clipped_values


# Calculate historical patterns
if use_dimensional_analysis:
    # Use dimensional versions
    intelligence_growth_dim = historical_intelligence_growth(years)
    truth_adoption_dim = historical_truth_adoption(years)
    suppression_feedback_dim = historical_suppression_feedback(years)
    civilization_oscillation_dim = historical_civilization_oscillation(years)

    # Extract raw values for plotting and saving
    intelligence_growth = intelligence_growth_dim.value
    truth_adoption = truth_adoption_dim.value
    suppression_feedback = suppression_feedback_dim.value
    civilization_oscillation = civilization_oscillation_dim.value

    print("Using dimensional analysis for historical simulation")

    # Verify dimensional consistency
    dimensional_equations = {
        'historical_intelligence_growth': historical_intelligence_growth,
        'historical_truth_adoption': historical_truth_adoption,
        'historical_suppression_feedback': historical_suppression_feedback,
        'historical_civilization_oscillation': historical_civilization_oscillation
    }

    try:
        consistency_results = check_dimensional_consistency(dimensional_equations)
        print("\nDimensional Consistency Check Results:")
        for name, result in consistency_results.items():
            print(f"{name}: {result['status']}")

        # Save consistency results
        consistency_df = pd.DataFrame([
            {"Function": name, "Status": result["status"], "Notes": result.get("message", "")}
            for name, result in consistency_results.items()
        ])
        consistency_df.to_csv(data_dir / "historical_dimensional_consistency.csv", index=False)
        print(f"Dimensional consistency results saved to: {data_dir / 'historical_dimensional_consistency.csv'}")
    except Exception as e:
        print(f"Error during dimensional consistency check: {e}")
else:
    # Use standard versions
    intelligence_growth = np.clip(50 * safe_sin(0.01 * (years - 1000)) - 20 * safe_exp(-0.002 * (years - 1000)),
                                  MIN_VALUE, MAX_INTELLIGENCE)

    truth_adoption = np.clip(40 * (1 - safe_exp(-0.005 * (years - 1000))),
                             MIN_VALUE, MAX_TRUTH)

    suppression_feedback = np.clip(safe_exp(-0.005 * (years - 1000)) + 0.1 * safe_sin(0.02 * (years - 1000)),
                                   MIN_VALUE, MAX_SUPPRESSION)

    civilization_oscillation = np.clip(0.05 * safe_exp(-0.002 * (years - 1000)) * safe_sin(0.01 * (years - 1000)),
                                       -1.0, 1.0)

# Check for any instabilities in the generated data
for array, name in [(intelligence_growth, "Intelligence Growth"),
                    (truth_adoption, "Truth Adoption"),
                    (suppression_feedback, "Suppression Feedback"),
                    (civilization_oscillation, "Civilization Oscillation")]:

    if circuit_breaker.check_array_stability(array):
        print(f"Warning: Potential instability detected in {name}. Applying additional bounds.")
        # If instability detected, apply more restrictive bounds
        array = np.nan_to_num(array, nan=0.0, posinf=MAX_INTELLIGENCE, neginf=MIN_VALUE)
        array = np.clip(array, MIN_VALUE, MAX_INTELLIGENCE)

# Create a DataFrame
df = pd.DataFrame({
    'Year': years,
    'Intelligence Growth': intelligence_growth,
    'Truth Adoption': truth_adoption,
    'Suppression Feedback': suppression_feedback,
    'Civilization Oscillation': civilization_oscillation
})

# Save to CSV
csv_filename = data_dir / "historical_simulation_results.csv"
df.to_csv(csv_filename, index=False)
print(f"CSV file saved: {csv_filename}")

# Save stability metrics if available
stability_metrics = {
    'Max_Intelligence': np.max(intelligence_growth),
    'Min_Intelligence': np.min(intelligence_growth),
    'Max_Truth': np.max(truth_adoption),
    'Min_Truth': np.min(truth_adoption),
    'Max_Suppression': np.max(suppression_feedback),
    'Min_Suppression': np.min(suppression_feedback),
    'Max_Oscillation': np.max(np.abs(civilization_oscillation)),
    'Stability_Violations': circuit_breaker.trigger_count,
    'Used_Dimensional_Analysis': use_dimensional_analysis
}

stability_df = pd.DataFrame([stability_metrics])
stability_filename = data_dir / "historical_simulation_stability.csv"
stability_df.to_csv(stability_filename, index=False)
print(f"Stability metrics saved: {stability_filename}")

# Plot results
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Intelligence Growth
axs[0, 0].plot(years, intelligence_growth, 'b', label='Intelligence Growth')
axs[0, 0].set_title('Intelligence Growth Over Time')
axs[0, 0].set_xlabel('Year')
axs[0, 0].set_ylabel('Intelligence')
axs[0, 0].legend()
axs[0, 0].grid(True)
axs[0, 0].set_ylim(0, MAX_INTELLIGENCE)  # Set consistent y-limits

# Truth Adoption
axs[0, 1].plot(years, truth_adoption, 'g', label='Truth Adoption')
axs[0, 1].set_title('Truth Adoption Model')
axs[0, 1].set_xlabel('Year')
axs[0, 1].set_ylabel('Truth Level')
axs[0, 1].legend()
axs[0, 1].grid(True)
axs[0, 1].set_ylim(0, MAX_TRUTH)  # Set consistent y-limits

# Suppression Feedback
axs[1, 0].plot(years, suppression_feedback, 'r', label='Suppression Feedback')
axs[1, 0].set_title('Suppression & Resistance Feedback')
axs[1, 0].set_xlabel('Year')
axs[1, 0].set_ylabel('Suppression Impact')
axs[1, 0].legend()
axs[1, 0].grid(True)
axs[1, 0].set_ylim(0, MAX_SUPPRESSION)  # Set consistent y-limits

# Civilization Oscillation
axs[1, 1].plot(years, civilization_oscillation, 'purple', label='Civilization Oscillation')
axs[1, 1].set_title('Civilization Oscillation Over Time')
axs[1, 1].set_xlabel('Year')
axs[1, 1].set_ylabel('Oscillation State')
axs[1, 1].legend()
axs[1, 1].grid(True)
axs[1, 1].set_ylim(-1, 1)  # Set consistent y-limits for oscillation

# Add annotation about dimensional analysis if used
if use_dimensional_analysis:
    plt.figtext(0.5, 0.01, "Using dimensional analysis", ha="center", fontsize=10,
                bbox={"facecolor": "lightgray", "alpha": 0.5, "pad": 5})

# Adjust layout and save figure
plt.tight_layout()
plot_filename = plots_dir / "historical_simulation_results.png"
plt.savefig(plot_filename)
print(f"Plot saved: {plot_filename}")

plt.show()