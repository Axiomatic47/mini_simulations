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

# Simulation parameters
timesteps = 400
dt = 1
num_agents = 5

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

# Main simulation loop
for t in range(1, timesteps):
    # Truth adoption update
    T[t] = T[t - 1] + truth_adoption(T[t - 1], A_truth, T_max) * dt

    for agent in range(num_agents):
        # Wisdom calculation
        W = wisdom_field(W_0, alpha_wisdom, S[agent, t - 1], R, K[agent, t - 1])

        # Knowledge growth update
        K[agent, t] = knowledge_growth_phase_transition(
            K[agent, t - 1], beta_decay_phase, t, A_phase, gamma_phase, T[t - 1], T_crit_phase
        )

        # Suppression update (with resurgence)
        S[agent, t] = resistance_resurgence(
            S[agent, 0], lambda_decay, t, alpha_resurge, mu_resurge, t_crit_resurge
        )

        # Suppression feedback
        Fs = suppression_feedback(alpha_feedback, S[agent, t - 1], beta_feedback, K[agent, t - 1])
        S[agent, t] += Fs * dt

        # Intelligence growth update
        I[agent, t] = I[agent, t - 1] + intelligence_growth(
            K[agent, t], W, R, S[agent, t], N
        ) * dt

    # Civilization oscillation dynamics
    osc_acceleration = civilization_oscillation(E[t - 1], dE_dt, gamma_osc, omega_osc)
    dE_dt += osc_acceleration * dt
    E[t] = E[t - 1] + dE_dt * dt

# Visualization
time_range = np.arange(timesteps)

plt.figure(figsize=(14, 10))

# Intelligence Growth
plt.subplot(2, 2, 1)
plt.plot(time_range, np.mean(I, axis=0), 'b-', linewidth=2, label='Avg Intelligence')
plt.title('Intelligence Growth Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Intelligence')
plt.legend()
plt.grid(True)

# Truth Adoption
plt.subplot(2, 2, 2)
plt.plot(time_range, T, 'g-', linewidth=2, label='Truth Adoption')
plt.title('Truth Adoption Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Truth Level')
plt.legend()
plt.grid(True)

# Suppression Dynamics
plt.subplot(2, 2, 3)
plt.plot(time_range, np.mean(S, axis=0), 'r-', linewidth=2, label='Avg Suppression')
plt.title('Suppression Dynamics')
plt.xlabel('Time Steps')
plt.ylabel('Suppression Level')
plt.legend()
plt.grid(True)

# Civilization Oscillation Dynamics
plt.subplot(2, 2, 4)
plt.plot(time_range, E, 'm-', linewidth=2, label='Civilization Oscillation')
plt.title('Civilization Oscillation Dynamics')
plt.xlabel('Time Steps')
plt.ylabel('Oscillation State')
plt.legend()
plt.grid(True)

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
    'Civilization_Oscillation': E
})
df_results.to_csv(csv_file, index=False)
print(f"✅ Data saved at: {csv_file}")

plt.show()