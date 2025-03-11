import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from utils.circuit_breaker import CircuitBreaker  # Import circuit breaker utility

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


# Define historical trends based on known suppression and knowledge revolutions
# Use bounds and safe functions to ensure numerical stability
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
    'Stability_Violations': circuit_breaker.trigger_count
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

# Adjust layout and save figure
plt.tight_layout()
plot_filename = plots_dir / "historical_simulation_results.png"
plt.savefig(plot_filename)
print(f"Plot saved: {plot_filename}")

plt.show()