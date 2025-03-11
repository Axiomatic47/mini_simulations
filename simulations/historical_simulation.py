import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Setup output directories
BASE_DIR = Path(__file__).resolve().parent.parent
plots_dir = BASE_DIR / 'outputs' / 'plots'
data_dir = BASE_DIR / 'outputs' / 'data'
plots_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)

# Define time range (e.g., years from 1000 AD to 2025 AD)
years = np.linspace(1000, 2025, 1025)

# Define historical trends based on known suppression and knowledge revolutions
intelligence_growth = 50 * np.sin(0.01 * (years - 1000)) - 20 * np.exp(-0.002 * (years - 1000))
truth_adoption = 40 * (1 - np.exp(-0.005 * (years - 1000)))
suppression_feedback = np.exp(-0.005 * (years - 1000)) + 0.1 * np.sin(0.02 * (years - 1000))
civilization_oscillation = 0.05 * np.exp(-0.002 * (years - 1000)) * np.sin(0.01 * (years - 1000))

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

# Plot results
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Intelligence Growth
axs[0, 0].plot(years, intelligence_growth, 'b', label='Intelligence Growth')
axs[0, 0].set_title('Intelligence Growth Over Time')
axs[0, 0].set_xlabel('Year')
axs[0, 0].set_ylabel('Intelligence')
axs[0, 0].legend()

# Truth Adoption
axs[0, 1].plot(years, truth_adoption, 'g', label='Truth Adoption')
axs[0, 1].set_title('Truth Adoption Model')
axs[0, 1].set_xlabel('Year')
axs[0, 1].set_ylabel('Truth Level')
axs[0, 1].legend()

# Suppression Feedback
axs[1, 0].plot(years, suppression_feedback, 'r', label='Suppression Feedback')
axs[1, 0].set_title('Suppression & Resistance Feedback')
axs[1, 0].set_xlabel('Year')
axs[1, 0].set_ylabel('Suppression Impact')
axs[1, 0].legend()

# Civilization Oscillation
axs[1, 1].plot(years, civilization_oscillation, 'purple', label='Civilization Oscillation')
axs[1, 1].set_title('Civilization Oscillation Over Time')
axs[1, 1].set_xlabel('Year')
axs[1, 1].set_ylabel('Oscillation State')
axs[1, 1].legend()

# Adjust layout and save figure
plt.tight_layout()
plot_filename = plots_dir / "historical_simulation_results.png"
plt.savefig(plot_filename)
print(f"Plot saved: {plot_filename}")

plt.show()