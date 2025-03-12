import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import equations from config directory
from config.equations import (
    intelligence_growth, truth_adoption, wisdom_field,
    resistance_resurgence, suppression_feedback,
    civilization_oscillation, knowledge_growth_phase_transition
)

# Import dimensional analysis tools
from utils.dimensional_consistency import (
    Dimension, DimensionalValue,
    intelligence_growth_with_dimensions,
    wisdom_field_with_dimensions,
    truth_adoption_with_dimensions,
    suppression_feedback_with_dimensions,
    resistance_resurgence_with_dimensions,
    check_dimensional_consistency
)

# Import circuit breaker for numerical stability
from utils.circuit_breaker import CircuitBreaker

# Parameters
timesteps, dt, num_agents = 400, 1, 5

# Bounds for numerical stability
MAX_KNOWLEDGE = 1000.0
MAX_SUPPRESSION = 100.0
MAX_INTELLIGENCE = 100.0
MAX_TRUTH = 100.0
MIN_VALUE = 0.0
MIN_TIMESTEP = 0.1
MAX_TIMESTEP = 5.0

# Directories for outputs
BASE_DIR = Path(__file__).resolve().parent.parent
plots_dir = BASE_DIR / 'outputs' / 'plots'
data_dir = BASE_DIR / 'outputs' / 'data'
plots_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)

# Initialize circuit breaker
circuit_breaker = CircuitBreaker(
    threshold=1e-6,
    max_value=MAX_KNOWLEDGE,
    min_value=MIN_VALUE,
    max_rate_of_change=100.0
)

# Arrays initialization
K, S, I = (np.zeros((num_agents, timesteps)) for _ in range(3))
T, E = np.zeros(timesteps), np.zeros(timesteps)

# Initial conditions
K[:, 0], S[:, 0], I[:, 0] = 1, np.linspace(5, 10, num_agents), 5.0
T[0], E[0], dE_dt = 1.0, 0.05, 0.0

# Constants (from latest adjustment)
W_0, alpha_wisdom = 1.0, 0.1
R_val, N = 2.0, 1.5
A_truth, T_max = 2.5, 40
lambda_decay = 0.05
alpha_feedback, beta_feedback = 0.1, 0.05
alpha_resurge, mu_resurge, t_crit_resurge = 5.0, 0.05, 150
K_0_phase, beta_decay_phase, A_phase, gamma_phase, T_crit_phase = 1.0, 0.02, 1.5, 0.1, 20
gamma_osc, omega_osc = 0.005, 0.3

# Create dimensional resistance value (used in multiple calculations)
R = DimensionalValue(R_val, Dimension.RESISTANCE)

# Initial oscillation conditions
E[0], dE_dt = 0.05, 0.0

# Stability tracking
stability_issues = 0
adaptive_timestep = dt
is_adaptive_timestep = True  # Enable adaptive timestep by default

# Set to True to use dimensional analysis for calculations
use_dimensional_analysis = True


# Safe mathematical operations
def safe_div(x, y, default=0.0):
    """Safe division with check for division by zero."""
    if abs(y) < 1e-10:
        return default
    return x / y


# Simulation loop
print("Starting simulation...")
if use_dimensional_analysis:
    print("Using dimensional analysis for key calculations")

for t in range(1, timesteps):
    # Use adaptive timestep if enabled
    if is_adaptive_timestep:
        # Adjust timestep based on rate of change in previous step
        if t > 1:
            max_change_k = np.max(np.abs(K[:, t - 1] - K[:, t - 2]))
            max_change_s = np.max(np.abs(S[:, t - 1] - S[:, t - 2]))
            max_change = max(max_change_k, max_change_s)

            if max_change > 10.0:
                adaptive_timestep = max(MIN_TIMESTEP, adaptive_timestep * 0.5)
            elif max_change < 1.0:
                adaptive_timestep = min(MAX_TIMESTEP, adaptive_timestep * 1.1)

        current_dt = adaptive_timestep
    else:
        current_dt = dt

    if use_dimensional_analysis:
        # Create dimensional value for Truth
        T_dim = DimensionalValue(T[t - 1], Dimension.TRUTH)

        # Truth adoption update with bounds using dimensional analysis
        truth_change_dim = truth_adoption_with_dimensions(T_dim, A_truth, T_max)
        # Extract the raw value from the dimensional result for the update
        T[t] = np.clip(T[t - 1] + truth_change_dim.value * current_dt, MIN_VALUE, MAX_TRUTH)
    else:
        # Truth adoption update with bounds (original version)
        truth_change = truth_adoption(T[t - 1], A_truth, T_max)
        T[t] = np.clip(T[t - 1] + truth_change * current_dt, MIN_VALUE, MAX_TRUTH)

    for agent in range(num_agents):
        if use_dimensional_analysis:
            # Convert raw values to dimensional values
            K_dim = DimensionalValue(K[agent, t - 1], Dimension.KNOWLEDGE)
            S_dim = DimensionalValue(S[agent, t - 1], Dimension.SUPPRESSION)

            # Calculate wisdom with dimensional analysis
            W_dim = wisdom_field_with_dimensions(W_0, alpha_wisdom, S_dim, R, K_dim)
        else:
            # Calculate wisdom with bounds (original version)
            W = wisdom_field(W_0, alpha_wisdom, S[agent, t - 1], R_val, K[agent, t - 1])

        # Knowledge growth with phase transition (using original function)
        # We don't have a dimensional version of this function yet
        k_growth = knowledge_growth_phase_transition(
            K[agent, t - 1], beta_decay_phase, t, A_phase, gamma_phase, T[t - 1], T_crit_phase)

        # Check for potential instability in knowledge growth
        if circuit_breaker.check_value_stability(k_growth):
            # If unstable, use a safer, limited growth rate
            stability_issues += 1
            k_growth = np.clip(k_growth, 0, K[agent, t - 1] * 0.1)  # Limit growth to 10%

        K[agent, t] = np.clip(K[agent, t - 1] + k_growth * current_dt, MIN_VALUE, MAX_KNOWLEDGE)

        if use_dimensional_analysis:
            # Update Knowledge dimensional value with the latest value
            K_dim_updated = DimensionalValue(K[agent, t], Dimension.KNOWLEDGE)

            # Calculate suppression base value with dimensional analysis
            s_base_dim = resistance_resurgence_with_dimensions(
                S[agent, 0], lambda_decay, t, alpha_resurge, mu_resurge, t_crit_resurge)

            # Calculate suppression feedback with dimensional analysis
            s_feedback_dim = suppression_feedback_with_dimensions(
                alpha_feedback, S_dim, beta_feedback, K_dim_updated)

            # Check for potential instability in suppression calculation
            if circuit_breaker.check_value_stability(s_feedback_dim.value):
                stability_issues += 1
                s_feedback = np.clip(s_feedback_dim.value, -1.0, 1.0)  # Limit feedback magnitude
            else:
                s_feedback = s_feedback_dim.value

            # Update suppression using the dimensional values
            S[agent, t] = np.clip(s_base_dim.value + s_feedback * current_dt, MIN_VALUE, MAX_SUPPRESSION)

            # Intelligence growth with dimensional analysis
            i_growth_dim = intelligence_growth_with_dimensions(
                K_dim_updated,
                W_dim,
                R,
                DimensionalValue(S[agent, t], Dimension.SUPPRESSION),
                N
            )

            # Check for potential instability in intelligence growth
            if circuit_breaker.check_value_stability(i_growth_dim.value):
                stability_issues += 1
                i_growth = np.clip(i_growth_dim.value, -1.0, 1.0)  # Limit growth/decline rate
            else:
                i_growth = i_growth_dim.value

        else:
            # Suppression update with bounds (original version)
            s_base = resistance_resurgence(
                S[agent, 0], lambda_decay, t, alpha_resurge, mu_resurge, t_crit_resurge)

            s_feedback = suppression_feedback(alpha_feedback, S[agent, t - 1], beta_feedback, K[agent, t - 1])

            # Check for potential instability in suppression calculation
            if circuit_breaker.check_value_stability(s_feedback):
                stability_issues += 1
                s_feedback = np.clip(s_feedback, -1.0, 1.0)  # Limit feedback magnitude

            S[agent, t] = np.clip(s_base + s_feedback * current_dt, MIN_VALUE, MAX_SUPPRESSION)

            # Intelligence growth with bounds (original version)
            i_growth = intelligence_growth(K[agent, t], W, R_val, S[agent, t], N)

            # Check for potential instability in intelligence growth
            if circuit_breaker.check_value_stability(i_growth):
                stability_issues += 1
                i_growth = np.clip(i_growth, -1.0, 1.0)  # Limit growth/decline rate

        I[agent, t] = np.clip(I[agent, t - 1] + i_growth * current_dt, MIN_VALUE, MAX_INTELLIGENCE)

    # Civilization oscillation with bounds
    # We don't have a dimensional version of this function yet
    osc_acceleration = civilization_oscillation(E[t - 1], dE_dt, gamma_osc, omega_osc)

    # Check for potential instability in oscillation
    if circuit_breaker.check_value_stability(osc_acceleration):
        stability_issues += 1
        osc_acceleration = np.clip(osc_acceleration, -0.1, 0.1)  # Limit acceleration

    dE_dt = np.clip(dE_dt + osc_acceleration * current_dt, -1.0, 1.0)
    E[t] = np.clip(E[t - 1] + dE_dt * current_dt, -1.0, 1.0)

    # Output progress and any stability issues
    if t % 100 == 0:
        print(f"Step {t}/{timesteps} completed. Stability issues: {stability_issues}")

print(f"Simulation completed with {stability_issues} stability issues detected.")

# Replace any remaining NaN or inf values that might have slipped through
K = np.nan_to_num(K, nan=0.0, posinf=MAX_KNOWLEDGE, neginf=0.0)
S = np.nan_to_num(S, nan=0.0, posinf=MAX_SUPPRESSION, neginf=0.0)
I = np.nan_to_num(I, nan=0.0, posinf=MAX_INTELLIGENCE, neginf=0.0)
T = np.nan_to_num(T, nan=0.0, posinf=MAX_TRUTH, neginf=0.0)
E = np.nan_to_num(E, nan=0.0, posinf=1.0, neginf=-1.0)

# Dimensional consistency check
if use_dimensional_analysis:
    try:
        # Define dimensional equations to check
        dimensional_equations = {
            'intelligence_growth_with_dimensions': intelligence_growth_with_dimensions,
            'wisdom_field_with_dimensions': wisdom_field_with_dimensions,
            'truth_adoption_with_dimensions': truth_adoption_with_dimensions,
            'suppression_feedback_with_dimensions': suppression_feedback_with_dimensions,
            'resistance_resurgence_with_dimensions': resistance_resurgence_with_dimensions
        }

        # Check dimensional consistency
        consistency_results = check_dimensional_consistency(dimensional_equations)
        print("\nDimensional Consistency Check Results:")
        for name, result in consistency_results.items():
            print(f"{name}: {result['status']}")

        # Save results to file
        consistency_df = pd.DataFrame([
            {"Equation": name, "Status": result["status"], "Notes": result.get("message", "")}
            for name, result in consistency_results.items()
        ])
        consistency_df.to_csv(data_dir / "dimensional_consistency.csv", index=False)
        print(f"Dimensional consistency results saved to: {data_dir / 'dimensional_consistency.csv'}")

    except Exception as e:
        print(f"Error during dimensional consistency check: {e}")

# Plotting
plt.figure(figsize=(12, 10))

# Plot data
plot_data = [
    (np.mean(I, axis=0), 'Intelligence', 'blue', 0, MAX_INTELLIGENCE),
    (T, 'Truth Adoption', 'green', 0, MAX_TRUTH),
    (np.mean(S, axis=0), 'Suppression Level', 'red', 0, MAX_SUPPRESSION),
    (E, 'Civilization Oscillation', 'purple', -1, 1)
]

for i, (data, title, color, y_min, y_max) in enumerate(plot_data):
    plt.subplot(2, 2, i + 1)
    plt.plot(np.arange(timesteps), data, color=color, linewidth=2)
    plt.title(f'{title} Dynamics')
    plt.xlabel('Time Steps')
    plt.ylabel(title)
    plt.ylim(y_min, y_max)
    plt.legend([title])
    plt.grid(True)

plt.tight_layout()
plt.savefig(plots_dir / "comprehensive_simulation_results.png")

# Export data
df_results = pd.DataFrame({
    'Time': np.arange(timesteps),
    'Avg_Intelligence': np.mean(I, axis=0),
    'Truth_Adoption': T,
    'Avg_Suppression': np.mean(S, axis=0),
    'Civilization_Oscillation': E
})

df_results.to_csv(data_dir / "comprehensive_simulation_results.csv", index=False)

# Save stability metrics
stability_metrics = {
    'Total_Stability_Issues': stability_issues,
    'Circuit_Breaker_Triggers': circuit_breaker.trigger_count,
    'Max_Knowledge': np.max(K),
    'Max_Suppression': np.max(S),
    'Max_Intelligence': np.max(I),
    'Max_Truth': np.max(T),
    'Final_Timestep': adaptive_timestep if is_adaptive_timestep else dt,
    'Used_Dimensional_Analysis': use_dimensional_analysis
}

stability_df = pd.DataFrame([stability_metrics])
stability_df.to_csv(data_dir / "comprehensive_simulation_stability.csv", index=False)
print(f"Stability metrics saved to: {data_dir / 'comprehensive_simulation_stability.csv'}")

plt.show()