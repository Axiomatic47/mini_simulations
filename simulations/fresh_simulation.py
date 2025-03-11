import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import from config directory
from config.equations import (
    intelligence_growth, free_will_decision, truth_adoption,
    wisdom_field, resistance_resurgence, suppression_feedback,
    civilization_oscillation, knowledge_growth_phase_transition
)

# Import circuit breaker for numerical stability
from utils.circuit_breaker import CircuitBreaker

# Simulation parameters
timesteps = 400
dt = 1
num_agents = 5

# Bounds for numerical stability
MAX_KNOWLEDGE = 100.0
MAX_SUPPRESSION = 50.0
MAX_INTELLIGENCE = 100.0
MAX_TRUTH = 100.0
MIN_VALUE = 0.0
MAX_OSCILLATION = 1.0
MIN_OSCILLATION = -1.0

# Initialize circuit breaker
circuit_breaker = CircuitBreaker(
    threshold=1e-6,
    max_value=MAX_KNOWLEDGE,
    min_value=MIN_VALUE,
    max_rate_of_change=5.0
)

# Enable adaptive timestep
enable_adaptive_timestep = True
MIN_TIMESTEP = 0.2
MAX_TIMESTEP = 2.0
adaptive_timestep = dt

# Stability tracking
stability_issues = 0
timestep_history = [dt]  # Track timestep changes


# Safe mathematical operations
def safe_exp(x, max_result=1e10):
    """Apply exponential function with bounds to prevent overflow."""
    # Limit the exponent to avoid overflow
    x = np.clip(x, -50.0, 50.0)
    return np.clip(np.exp(x), 0.0, max_result)


def safe_div(x, y, default=0.0):
    """Safe division with check for division by zero."""
    if abs(y) < 1e-10:
        return default
    return x / y


# Explicit absolute paths for outputs
BASE_DIR = Path(__file__).resolve().parent.parent
plots_dir = BASE_DIR / 'outputs' / 'plots'
data_dir = BASE_DIR / 'outputs' / 'data'
plots_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)

# Initialize arrays
K = np.zeros((num_agents, timesteps))
S = np.zeros((num_agents, timesteps))
I = np.zeros((num_agents, timesteps))
T = np.zeros(timesteps)
E = np.zeros(timesteps)
dE_dt = 0.0

# Initial conditions
K[:, 0] = 1.0
S[:, 0] = np.linspace(5, 10, num_agents)
I[:, 0] = 5.0
T[0] = 1.0
E[0] = 0.05  # small perturbation for clear oscillations

# Constants
W_0, alpha_wisdom = 1.0, 0.1
R, N = 2.0, 1.5
A_truth, T_max = 2.5, 40
lambda_decay = 0.05
alpha_feedback, beta_feedback = 0.1, 0.05
alpha_resurge, mu_resurge, t_crit_resurge = 5.0, 0.05, 150
K_0_phase, beta_decay_phase, A_phase, gamma_phase, T_crit_phase = 1.0, 0.02, 1.5, 0.1, 20
gamma_osc, omega_osc = 0.005, 0.3

print("Starting simulation...")

# Main simulation loop
for t in range(1, timesteps):
    # Calculate adaptive timestep if enabled
    if enable_adaptive_timestep and t > 1:
        # Calculate maximum rate of change from previous step
        max_k_change = np.max(np.abs(K[:, t - 1] - K[:, t - 2]))
        max_s_change = np.max(np.abs(S[:, t - 1] - S[:, t - 2]))
        max_i_change = np.max(np.abs(I[:, t - 1] - I[:, t - 2]))
        max_change = max(max_k_change, max_s_change, max_i_change)

        # Adjust timestep based on rate of change
        if max_change > 2.0:
            adaptive_timestep = max(MIN_TIMESTEP, adaptive_timestep * 0.8)
        elif max_change < 0.5:
            adaptive_timestep = min(MAX_TIMESTEP, adaptive_timestep * 1.2)

        current_dt = adaptive_timestep
    else:
        current_dt = dt

    # Track timestep
    timestep_history.append(current_dt)

    # Output progress periodically
    if t % 100 == 0:
        print(
            f"Step {t}/{timesteps} completed. Current timestep: {current_dt:.4f}. Stability issues: {stability_issues}")

    # Truth adoption update with bounds and stability check
    try:
        truth_change = truth_adoption(
            np.clip(T[t - 1], MIN_VALUE, MAX_TRUTH),
            A_truth,
            T_max
        )

        # Check for excessive change
        if circuit_breaker.check_value_stability(truth_change):
            truth_change = np.clip(truth_change, -1.0, 1.0)
            stability_issues += 1

        T[t] = np.clip(T[t - 1] + truth_change * current_dt, MIN_VALUE, MAX_TRUTH)
    except Exception as e:
        print(f"Warning: Error in truth adoption at t={t}: {e}")
        T[t] = T[t - 1]  # Use previous value
        stability_issues += 1

    for agent in range(num_agents):
        try:
            # Wisdom calculation with bounds
            W = wisdom_field(
                W_0,
                alpha_wisdom,
                np.clip(S[agent, t - 1], MIN_VALUE, MAX_SUPPRESSION),
                R,
                np.clip(K[agent, t - 1], MIN_VALUE, MAX_KNOWLEDGE)
            )

            # Knowledge growth update with bounds and stability check
            k_growth = knowledge_growth_phase_transition(
                np.clip(K[agent, t - 1], MIN_VALUE, MAX_KNOWLEDGE),
                beta_decay_phase,
                t,
                A_phase,
                gamma_phase,
                np.clip(T[t - 1], MIN_VALUE, MAX_TRUTH),
                T_crit_phase
            )

            # Check for stability
            if circuit_breaker.check_value_stability(k_growth):
                k_growth = np.clip(k_growth, K[agent, t - 1] * 0.9, K[agent, t - 1] * 1.1)
                stability_issues += 1

            K[agent, t] = np.clip(k_growth, MIN_VALUE, MAX_KNOWLEDGE)

            # Suppression update (with resurgence) with bounds
            s_resurgence = resistance_resurgence(
                S[agent, 0],
                lambda_decay,
                t,
                alpha_resurge,
                mu_resurge,
                t_crit_resurge
            )

            # Check for stability
            if circuit_breaker.check_value_stability(s_resurgence):
                s_resurgence = np.clip(s_resurgence, 0.0, MAX_SUPPRESSION)
                stability_issues += 1

            S[agent, t] = np.clip(s_resurgence, MIN_VALUE, MAX_SUPPRESSION)

            # Suppression feedback with bounds
            Fs = suppression_feedback(
                alpha_feedback,
                np.clip(S[agent, t - 1], MIN_VALUE, MAX_SUPPRESSION),
                beta_feedback,
                np.clip(K[agent, t - 1], MIN_VALUE, MAX_KNOWLEDGE)
            )

            # Check for stability
            if circuit_breaker.check_value_stability(Fs):
                Fs = np.clip(Fs, -1.0, 1.0)
                stability_issues += 1

            S[agent, t] = np.clip(S[agent, t] + Fs * current_dt, MIN_VALUE, MAX_SUPPRESSION)

            # Intelligence growth update with bounds
            i_growth = intelligence_growth(
                np.clip(K[agent, t], MIN_VALUE, MAX_KNOWLEDGE),
                W,
                R,
                np.clip(S[agent, t], MIN_VALUE, MAX_SUPPRESSION),
                N
            )

            # Check for stability
            if circuit_breaker.check_value_stability(i_growth):
                i_growth = np.clip(i_growth, -1.0, 1.0)
                stability_issues += 1

            I[agent, t] = np.clip(I[agent, t - 1] + i_growth * current_dt, MIN_VALUE, MAX_INTELLIGENCE)

        except Exception as e:
            print(f"Warning: Error updating agent {agent} at t={t}: {e}")
            # Fall back to previous values in case of error
            K[agent, t] = K[agent, t - 1]
            S[agent, t] = S[agent, t - 1]
            I[agent, t] = I[agent, t - 1]
            stability_issues += 1

    # Civilization oscillation dynamics with bounds
    try:
        osc_acceleration = civilization_oscillation(
            np.clip(E[t - 1], MIN_OSCILLATION, MAX_OSCILLATION),
            np.clip(dE_dt, -1.0, 1.0),
            gamma_osc,
            omega_osc
        )

        # Check for stability
        if circuit_breaker.check_value_stability(osc_acceleration):
            osc_acceleration = np.clip(osc_acceleration, -0.1, 0.1)
            stability_issues += 1

        dE_dt = np.clip(dE_dt + osc_acceleration * current_dt, -1.0, 1.0)
        E[t] = np.clip(E[t - 1] + dE_dt * current_dt, MIN_OSCILLATION, MAX_OSCILLATION)
    except Exception as e:
        print(f"Warning: Error in oscillation calculation at t={t}: {e}")
        dE_dt = 0.0  # Reset acceleration
        E[t] = E[t - 1]  # Use previous value
        stability_issues += 1

print(f"Simulation completed with {stability_issues} stability issues detected.")

# Replace any NaN or inf values that might have slipped through
K = np.nan_to_num(K, nan=1.0, posinf=MAX_KNOWLEDGE, neginf=MIN_VALUE)
S = np.nan_to_num(S, nan=1.0, posinf=MAX_SUPPRESSION, neginf=MIN_VALUE)
I = np.nan_to_num(I, nan=1.0, posinf=MAX_INTELLIGENCE, neginf=MIN_VALUE)
T = np.nan_to_num(T, nan=1.0, posinf=MAX_TRUTH, neginf=MIN_VALUE)
E = np.nan_to_num(E, nan=0.0, posinf=MAX_OSCILLATION, neginf=MIN_OSCILLATION)

# Visualization
time_range = np.arange(timesteps)

plt.figure(figsize=(15, 12))  # Increased size for additional plot

# Intelligence Growth
plt.subplot(3, 2, 1)
plt.plot(time_range, np.mean(I, axis=0), 'b-', linewidth=2, label='Avg Intelligence')
plt.title('Intelligence Growth Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Intelligence')
plt.legend()
plt.grid(True)
plt.ylim(0, np.max(np.mean(I, axis=0)) * 1.1)  # Set appropriate y-limit

# Truth Adoption
plt.subplot(3, 2, 2)
plt.plot(time_range, T, 'g-', linewidth=2, label='Truth Adoption')
plt.title('Truth Adoption Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Truth Level')
plt.legend()
plt.grid(True)
plt.ylim(0, MAX_TRUTH)  # Set appropriate y-limit

# Suppression Dynamics
plt.subplot(3, 2, 3)
plt.plot(time_range, np.mean(S, axis=0), 'r-', linewidth=2, label='Avg Suppression')
plt.title('Suppression Dynamics')
plt.xlabel('Time Steps')
plt.ylabel('Suppression Level')
plt.legend()
plt.grid(True)
plt.ylim(0, np.max(np.mean(S, axis=0)) * 1.1)  # Set appropriate y-limit

# Civilization Oscillation Dynamics
plt.subplot(3, 2, 4)
plt.plot(time_range, E, 'm-', linewidth=2, label='Civilization Oscillation')
plt.title('Civilization Oscillation Dynamics')
plt.xlabel('Time Steps')
plt.ylabel('Oscillation State')
plt.legend()
plt.grid(True)
plt.ylim(MIN_OSCILLATION, MAX_OSCILLATION)  # Set appropriate y-limit

# Adaptive Timestep (NEW)
plt.subplot(3, 2, 5)
plt.plot(time_range, timestep_history, 'k-', linewidth=2, label='Adaptive Timestep')
plt.title('Adaptive Timestep')
plt.xlabel('Time Steps')
plt.ylabel('Timestep Size (dt)')
plt.legend()
plt.grid(True)
plt.ylim(0, MAX_TIMESTEP * 1.1)  # Set appropriate y-limit

# Free Will Decision (NEW - example application)
plt.subplot(3, 2, 6)
decision_probabilities = np.zeros(timesteps)
for t in range(timesteps):
    # Calculate a proxy for the free will decision based on knowledge vs suppression
    # This is just an example visualization
    avg_k = np.mean(K[:, t])
    avg_s = np.mean(S[:, t])
    # Apply free_will_decision with bounds
    try:
        decision = free_will_decision(
            np.clip(avg_k / 10, 0, 10),  # q_Id (scaled knowledge)
            1.0,  # E_K (constant field for example)
            np.clip(avg_s / 10, 0, 10),  # q_R (scaled suppression)
            0.5  # E_F (constant field for example)
        )
        decision_probabilities[t] = decision
    except:
        # Fallback if calculation fails
        decision_probabilities[t] = 0.0 if t == 0 else decision_probabilities[t - 1]

plt.plot(time_range, decision_probabilities, 'y-', linewidth=2, label='Decision Probability')
plt.title('Free Will Decision Dynamics')
plt.xlabel('Time Steps')
plt.ylabel('Decision Probability')
plt.legend()
plt.grid(True)
plt.ylim(-1.1, 1.1)  # Decision probability is bounded by tanh

plt.tight_layout()

# Save plot explicitly and confirm save
plot_file = plots_dir / "enhanced_simulation_results.png"
plt.savefig(str(plot_file))
print(f"✅ Plot saved at: {plot_file}")

# Save CSV explicitly and confirm save
csv_file = data_dir / "enhanced_simulation_results.csv"
df_results = pd.DataFrame({
    'Time': time_range,
    'Avg_Intelligence': np.mean(I, axis=0),
    'Truth_Adoption': T,
    'Avg_Suppression': np.mean(S, axis=0),
    'Civilization_Oscillation': E,
    'Adaptive_Timestep': timestep_history,
    'Decision_Probability': decision_probabilities
})
df_results.to_csv(csv_file, index=False)
print(f"✅ Data saved at: {csv_file}")

# Save stability metrics
stability_metrics = {
    'Total_Stability_Issues': stability_issues,
    'Circuit_Breaker_Triggers': circuit_breaker.trigger_count,
    'Max_Knowledge': np.max(K),
    'Max_Suppression': np.max(S),
    'Max_Intelligence': np.max(I),
    'Max_Truth': np.max(T),
    'Initial_Timestep': dt,
    'Final_Timestep': timestep_history[-1] if timestep_history else dt,
    'Min_Timestep_Used': min(timestep_history) if timestep_history else dt,
    'Max_Timestep_Used': max(timestep_history) if timestep_history else dt
}

stability_df = pd.DataFrame([stability_metrics])
stability_df.to_csv(data_dir / "simulation_stability.csv", index=False)
print(f"✅ Stability metrics saved at: {data_dir / 'simulation_stability.csv'}")

plt.show()