import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import circuit breaker for numerical stability
from utils.circuit_breaker import CircuitBreaker

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

# Bounds for numerical stability
MAX_KNOWLEDGE = 50.0  # Knowledge cap
MIN_KNOWLEDGE = 0.0
MAX_SUPPRESSION = 15.0  # Higher than initial to allow for shocks
MIN_SUPPRESSION = 0.1  # Suppression floor
MAX_PROB = 1.0
MIN_PROB = 0.0
MAX_DT = 2.0
MIN_DT = 0.2

# Initialize circuit breaker
circuit_breaker = CircuitBreaker(
    threshold=1e-6,
    max_value=MAX_KNOWLEDGE,
    min_value=MIN_KNOWLEDGE,
    max_rate_of_change=5.0  # Maximum allowed rate of change per timestep
)

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
knowledge_cap, suppression_floor = MAX_KNOWLEDGE, MIN_SUPPRESSION
momentum_factor = 0.1

# External suppression shock parameters
shock_times = [50, 120]
shock_magnitudes = [2.5, 3.0]

# Tracking for numerical stability
stability_issues = 0
adaptive_timestep = dt
enable_adaptive_timestep = True  # Enable adaptive timestep for stability


# Safe exponential function to prevent overflow
def safe_exp(x):
    """Apply exponential function with bounds to prevent overflow."""
    # Limit the exponent to avoid overflow
    x = np.clip(x, -20.0, 20.0)
    return np.exp(x)


print("Starting simulation...")

# Simulation arrays preparation
for t in range(1, timesteps):
    # Calculate adaptive timestep if enabled
    if enable_adaptive_timestep and t > 1:
        # Calculate maximum rate of change from previous step
        max_k_change = np.max(np.abs(K[:, t - 1] - K[:, t - 2]))
        max_s_change = np.max(np.abs(S[:, t - 1] - S[:, t - 2]))
        max_change = max(max_k_change, max_s_change)

        # Adjust timestep based on rate of change
        if max_change > 2.0:
            adaptive_timestep = max(MIN_DT, adaptive_timestep * 0.8)
        elif max_change < 0.5:
            adaptive_timestep = min(MAX_DT, adaptive_timestep * 1.2)

        current_dt = adaptive_timestep
    else:
        current_dt = dt

    for agent in range(num_agents):
        # Knowledge growth with logistic cap and safety measures
        growth_factor = 1 - np.exp(-gamma * np.clip(K[agent, t - 1] - T_crit, -20, 20))
        knowledge_increment = A * np.clip(growth_factor, 0, 1) * current_dt

        # Check for potential instability in knowledge growth
        if circuit_breaker.check_value_stability(knowledge_increment):
            knowledge_increment = np.clip(knowledge_increment, 0, 1.0)  # Limit growth to prevent jumps
            stability_issues += 1

        # Apply increment with bounds
        K[agent, t] = np.clip(K[agent, t - 1] + knowledge_increment, MIN_KNOWLEDGE, MAX_KNOWLEDGE)

        # Suppression decay with floor limit and safety check
        decay_factor = safe_exp(-lambda_s * current_dt)
        S[agent, t] = max(S[agent, t - 1] * decay_factor, suppression_floor)

        # Check for potential instability
        if circuit_breaker.check_value_stability(S[agent, t]):
            S[agent, t] = np.clip(S[agent, t], suppression_floor, MAX_SUPPRESSION)
            stability_issues += 1

        # Apply external shocks at specific times with bounds
        if t in shock_times:
            shock_idx = shock_times.index(t)
            shock_value = np.clip(shock_magnitudes[shock_idx], 0, MAX_SUPPRESSION - S[agent, t])
            S[agent, t] += shock_value

        # Decision probability calculation with safety for exponential
        raw_decision_input = np.clip(0.5 * K[agent, t] - 0.3 * S[agent, t], -10, 10)
        raw_decision = 1 / (1 + safe_exp(-raw_decision_input))

        # Apply momentum with bounds
        momentum_term = np.clip(momentum_factor * decision_probability[agent, t - 1], 0, 1)
        non_momentum_term = np.clip((1 - momentum_factor) * raw_decision, 0, 1)
        decision_probability[agent, t] = momentum_term + non_momentum_term

        # Ensure stability in probabilities with explicit bounds
        decision_probability[agent, t] = np.clip(decision_probability[agent, t], MIN_PROB, MAX_PROB)

        # Check for potential instability
        if circuit_breaker.check_value_stability(decision_probability[agent, t]):
            decision_probability[agent, t] = np.clip(decision_probability[agent, t], 0.1, 0.9)
            stability_issues += 1

    # Report progress
    if t % 50 == 0:
        print(
            f"Step {t}/{timesteps} completed. Current timestep: {current_dt:.4f}. Stability issues: {stability_issues}")

print(f"Simulation completed with {stability_issues} stability issues detected.")

# Replace any NaN or inf values that might have slipped through
K = np.nan_to_num(K, nan=0.0, posinf=MAX_KNOWLEDGE, neginf=0.0)
S = np.nan_to_num(S, nan=suppression_floor, posinf=MAX_SUPPRESSION, neginf=suppression_floor)
decision_probability = np.nan_to_num(decision_probability, nan=0.5, posinf=MAX_PROB, neginf=MIN_PROB)

# Plot Knowledge and Suppression over Time
plt.figure(figsize=(15, 10))

# Create a custom layout with subplots
plt.subplot(2, 2, 1)
for agent in range(num_agents):
    plt.plot(K[agent], label=f"Agent {agent + 1} Knowledge")
plt.title("Knowledge Growth Over Time")
plt.xlabel("Time Steps")
plt.ylabel("Knowledge Level")
plt.legend()
plt.grid(True)
plt.ylim(0, MAX_KNOWLEDGE)

plt.subplot(2, 2, 2)
for agent in range(num_agents):
    plt.plot(S[agent], label=f"Agent {agent + 1} Suppression")
plt.title("Suppression Dynamics with External Shocks")
plt.xlabel("Time Steps")
plt.ylabel("Suppression Level")
plt.legend()
plt.grid(True)
plt.ylim(0, MAX_SUPPRESSION)

# Add vertical lines for shock times
for shock_time in shock_times:
    plt.axvline(x=shock_time, color='red', linestyle='--', alpha=0.5)

plt.subplot(2, 2, 3)
for agent in range(num_agents):
    plt.plot(decision_probability[agent], label=f"Agent {agent + 1} Decision Probability")
plt.title("Decision Probability Over Time")
plt.xlabel("Time Steps")
plt.ylabel("Probability")
plt.legend()
plt.grid(True)
plt.ylim(0, 1)

# Plot timestep size if adaptive timestep was used
if enable_adaptive_timestep:
    plt.subplot(2, 2, 4)
    timestep_sizes = [dt]  # Initial timestep
    for t in range(2, timesteps):
        max_k_change = np.max(np.abs(K[:, t - 1] - K[:, t - 2]))
        max_s_change = np.max(np.abs(S[:, t - 1] - S[:, t - 2]))
        max_change = max(max_k_change, max_s_change)

        if max_change > 2.0:
            new_dt = max(MIN_DT, timestep_sizes[-1] * 0.8)
        elif max_change < 0.5:
            new_dt = min(MAX_DT, timestep_sizes[-1] * 1.2)
        else:
            new_dt = timestep_sizes[-1]

        timestep_sizes.append(new_dt)

    plt.plot(timestep_sizes, 'k-', label="Adaptive Timestep")
    plt.title("Adaptive Timestep Size")
    plt.xlabel("Time Steps")
    plt.ylabel("Timestep (dt)")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, MAX_DT)

# Save the plot
plt.tight_layout()
plt.savefig(plots_dir / "multi_agent_simulation_results.png")

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
df.to_csv(data_dir / "multi_agent_simulation_results.csv", index=False)

# Save stability metrics
stability_metrics = {
    'Total_Stability_Issues': stability_issues,
    'Circuit_Breaker_Triggers': circuit_breaker.trigger_count,
    'Max_Knowledge': np.max(K),
    'Min_Knowledge': np.min(K),
    'Max_Suppression': np.max(S),
    'Min_Suppression': np.min(S),
    'Final_Timestep': adaptive_timestep if enable_adaptive_timestep else dt
}

stability_df = pd.DataFrame([stability_metrics])
stability_df.to_csv(data_dir / "multi_agent_simulation_stability.csv", index=False)

print(f"✅ Data saved at: {data_dir / 'multi_agent_simulation_results.csv'}")
print(f"✅ Stability metrics saved at: {data_dir / 'multi_agent_simulation_stability.csv'}")
print(f"✅ Plot saved at: {plots_dir / 'multi_agent_simulation_results.png'}")

# Display plot
plt.show()