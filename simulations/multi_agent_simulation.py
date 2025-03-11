import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Ensure output directories exist
BASE_DIR = Path(__file__).resolve().parent.parent
plots_dir = BASE_DIR / 'outputs' / 'plots'
data_dir = BASE_DIR / 'outputs' / 'data'
plots_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)

# Simulation parameters
timesteps = 200
dt = 1
num_agents = 5

# Initialize arrays
K = np.zeros((num_agents, timesteps))  # Knowledge
S = np.zeros((num_agents, timesteps))  # Suppression
decision_probability = np.zeros((num_agents, timesteps))  # Decision probability

# Initial conditions
K[:, 0] = 1
S[:, 0] = np.linspace(3, 7, num_agents)  # Varying suppression per agent
decision_probability[:, 0] = 0.5

# Constants for simulation
A, gamma, T_crit = 1.2, 0.1, 5
lambda_s, beta = 0.05, 0.02
knowledge_cap, suppression_floor = 50, 0.1
momentum_factor = 0.1

# External suppression shock parameters
shock_times = [50, 120]
shock_magnitudes = [2.5, 3.0]

# Simulation arrays preparation
for t in range(1, timesteps):
    for agent in range(num_agents):
        # Knowledge growth (with logistic cap)
        growth = A * (1 - np.exp(-gamma * (K[agent, t-1] - T_crit)))
        knowledge_increment = growth * dt
        K[agent, t] = np.clip(K[agent, t-1] + knowledge_increment, 0, knowledge_cap)

        # Suppression decay with floor limit
        S[agent, t] = max(S[agent, t-1] * np.exp(-lambda_s * dt), suppression_floor)

        # Apply external shocks at specific times
        if t in shock_times:
            shock_idx = shock_times.index(t)
            S[agent, t] += shock_magnitudes[shock_idx]

        # Decision probability calculation with momentum
        raw_decision = 1 / (1 + np.exp(-(0.5 * K[agent, t] - 0.3 * S[agent, t])))
        decision_probability[agent, t] = (
            momentum_factor * decision_probability[agent, t-1] +
            (1 - momentum_factor) * raw_decision
        )
        # Ensure stability in probabilities
        decision_probability[agent, t] = np.clip(decision_probability[agent, t], 0, 1)

# Plot Knowledge and Suppression over Time
plt.figure(figsize=(12, 8))
for agent in range(num_agents):
    plt.plot(K[agent], label=f"Agent {agent+1} Knowledge")
    plt.plot(S[agent], linestyle='--', label=f"Agent {agent+1} Suppression")

plt.title("Knowledge vs. Suppression Dynamics with External Shocks")
plt.xlabel("Time Steps")
plt.ylabel("Level")
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig(plots_dir / "knowledge_suppression_simulation.png")

# Prepare and export simulation data to CSV
simulation_data = []

for agent in range(num_agents):
    for t in range(timesteps):
        simulation_data.append({
            "Agent": agent + 1,
            "Time": t,
            "Knowledge": K[agent, t],
            "Suppression": S[agent, t],
            "Decision_Probability": decision_probability[agent, t]
        })

df = pd.DataFrame(simulation_data)
df.to_csv(data_dir / "simulation_results.csv", index=False)
print(f"✅ Data saved at: {data_dir / 'simulation_results.csv'}")
print(f"✅ Plot saved at: {plots_dir / 'knowledge_suppression_simulation.png'}")

# Display plot
plt.show()