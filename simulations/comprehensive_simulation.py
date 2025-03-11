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

# Parameters
timesteps, dt, num_agents = 400, 1, 5

# Directories for outputs
BASE_DIR = Path(__file__).resolve().parent.parent
plots_dir = BASE_DIR / 'outputs' / 'plots'
data_dir = BASE_DIR / 'outputs' / 'data'
plots_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)

# Arrays initialization
K, S, I = (np.zeros((num_agents, timesteps)) for _ in range(3))
T, E = np.zeros(timesteps), np.zeros(timesteps)

# Initial conditions
K[:, 0], S[:, 0], I[:, 0] = 1, np.linspace(5, 10, num_agents), 5.0
T[0], E[0], dE_dt = 1.0, 0.05, 0.0

# Constants (from latest adjustment)
W_0, alpha_wisdom = 1.0, 0.1
R, N = 2.0, 1.5
A_truth, T_max = 2.5, 40
lambda_decay = 0.05
alpha_feedback, beta_feedback = 0.1, 0.05
alpha_resurge, mu_resurge, t_crit_resurge = 5.0, 0.05, 150
K_0_phase, beta_decay_phase, A_phase, gamma_phase, T_crit_phase = 1.0, 0.02, 1.5, 0.1, 20
gamma_osc, omega_osc = 0.005, 0.3

# Initial oscillation conditions
E[0], dE_dt = 0.05, 0.0

# Simulation loop
for t in range(1, timesteps):
    T[t] = T[t - 1] + truth_adoption(T[t - 1], A_truth, T_max) * dt
    for agent in range(num_agents):
        W = wisdom_field(W_0, alpha_wisdom, S[agent, t-1], R, K[agent, t-1])
        K[agent, t] = knowledge_growth_phase_transition(
            K[agent, t-1], beta_decay_phase, t, A_phase, gamma_phase, T[t-1], T_crit_phase)
        S[agent, t] = resistance_resurgence(
            S[agent, 0], lambda_decay, t, alpha_resurge, mu_resurge, t_crit_resurge)
        S[agent, t] += suppression_feedback(alpha_feedback, S[agent, t-1], beta_feedback, K[agent, t-1]) * dt
        I[agent, t] = I[agent, t-1] + intelligence_growth(K[agent, t], W, R, S[agent, t], N) * dt

    osc_acceleration = civilization_oscillation(E[t-1], dE_dt, gamma_osc, omega_osc)
    dE_dt += osc_acceleration * dt
    E[t] = E[t-1] + dE_dt * dt

# Plotting
plt.figure(figsize=(12, 10))

# Plot data
plot_data = [
    (np.mean(I, axis=0), 'Intelligence', 'blue'),
    (T, 'Truth Adoption', 'green'),
    (np.mean(S, axis=0), 'Suppression Level', 'red'),
    (E, 'Civilization Oscillation', 'purple')
]

for i, (data, title, color) in enumerate(plot_data):
    plt.subplot(2, 2, i + 1)
    plt.plot(np.arange(timesteps), data, color=color, linewidth=2)
    plt.title(f'{title} Dynamics')
    plt.xlabel('Time Steps')
    plt.ylabel(title)
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

plt.show()