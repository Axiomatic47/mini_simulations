import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Add parent directory to path to find modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import original equations from config directory
try:
    from config.equations import (
        intelligence_growth, truth_adoption, wisdom_field,
        resistance_resurgence, suppression_feedback,
        civilization_oscillation, knowledge_growth_phase_transition
    )
except ImportError:
    print("Error: Could not import from 'config/equations.py'. Check that the file exists.")
    sys.exit(1)

# Import quantum_em_extensions module from config directory
try:
    from config.quantum_em_extensions import (
        knowledge_field_influence, quantum_entanglement_correlation,
        knowledge_field_gradient, build_entanglement_network,
        quantum_tunneling_probability
    )
except ImportError:
    print("Error: Could not import from 'config/quantum_em_extensions.py'. Check that the file exists.")
    sys.exit(1)

# Simulation parameters
timesteps = 400
dt = 1
num_agents = 5

# Ensure output directories exist
BASE_DIR = Path(__file__).resolve().parent.parent
plots_dir = BASE_DIR / 'outputs' / 'plots'
data_dir = BASE_DIR / 'outputs' / 'data'
plots_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)

print(f"Base directory: {BASE_DIR}")
print(f"Plots directory: {plots_dir}")
print(f"Data directory: {data_dir}")

# Initialize arrays
K = np.zeros((num_agents, timesteps))  # Knowledge
S = np.zeros((num_agents, timesteps))  # Suppression
I = np.zeros((num_agents, timesteps))  # Intelligence
T = np.zeros(timesteps)  # Truth adoption
E = np.zeros(timesteps)  # Civilization oscillation state

# New arrays for electromagnetism and quantum entanglement
em_influence = np.zeros((num_agents, timesteps))  # Electromagnetic field influence
qe_correlation = np.zeros((num_agents, timesteps))  # Quantum entanglement correlation
tunneling_events = np.zeros(timesteps)  # Record of tunneling breakthroughs

# Initial conditions
K[:, 0] = 1.0
S[:, 0] = np.linspace(5, 10, num_agents)
I[:, 0] = 5.0
T[0] = 1.0
E[0] = 0.05  # Small perturbation for clear oscillations
dE_dt = 0.0  # Initial oscillation velocity

# Conceptual positions for agents in knowledge space (for field gradients)
# Represents their "location" in conceptual/knowledge domain
agent_positions = np.random.rand(num_agents, 2)  # 2D conceptual space

# Constants from original simulation
W_0, alpha_wisdom = 1.0, 0.1
R, N = 2.0, 1.5
A_truth, T_max = 2.5, 40
lambda_decay = 0.05
alpha_feedback, beta_feedback = 0.1, 0.05
alpha_resurge, mu_resurge, t_crit_resurge = 5.0, 0.05, 150
K_0_phase, beta_decay_phase, A_phase, gamma_phase, T_crit_phase = 1.0, 0.02, 1.5, 0.1, 20
gamma_osc, omega_osc = 0.005, 0.3

# New constants for EM and QE effects
kappa_em = 0.025  # Knowledge field permeability
rho_entanglement = 0.08  # Maximum entanglement strength
sigma_entanglement = 0.15  # Entanglement decay rate
tunneling_threshold = 0.7  # Threshold for tunneling events

print("Starting simulation...")

# Main simulation loop
for t in range(1, timesteps):
    # Truth adoption update (same as original)
    T[t] = T[t - 1] + truth_adoption(T[t - 1], A_truth, T_max) * dt

    # Precompute the entanglement network for this timestep
    entanglement_matrix = build_entanglement_network(
        K[:, t - 1], max_entanglement=rho_entanglement, decay_rate=sigma_entanglement
    )

    # Calculate knowledge field gradients
    k_gradients = knowledge_field_gradient(K[:, t - 1], agent_positions, field_strength=0.1)

    # Update each agent
    for agent in range(num_agents):
        # Original wisdom calculation
        W = wisdom_field(W_0, alpha_wisdom, S[agent, t - 1], R, K[agent, t - 1])

        # Electromagnetic-like knowledge field influence
        em_sum = 0
        for other in range(num_agents):
            if agent != other:
                # Calculate conceptual distance between agents
                distance = np.linalg.norm(agent_positions[agent] - agent_positions[other])

                # Apply electromagnetic-like knowledge field influence
                em_effect = knowledge_field_influence(
                    K[agent, t - 1], K[other, t - 1], distance, kappa=kappa_em
                )
                em_sum += em_effect

        # Store electromagnetic influence
        em_influence[agent, t] = em_sum

        # Calculate quantum entanglement effect (sum of all entanglements)
        qe_sum = np.sum(entanglement_matrix[agent, :]) - entanglement_matrix[agent, agent]
        qe_correlation[agent, t] = qe_sum

        # Original knowledge growth with phase transition
        K[agent, t] = knowledge_growth_phase_transition(
            K[agent, t - 1], beta_decay_phase, t, A_phase, gamma_phase, T[t - 1], T_crit_phase
        )

        # Add electromagnetic and quantum effects to knowledge
        K[agent, t] += (em_sum + qe_sum * 0.1) * dt

        # Suppression update with resistance resurgence (same as original)
        S[agent, t] = resistance_resurgence(
            S[agent, 0], lambda_decay, t, alpha_resurge, mu_resurge, t_crit_resurge
        )

        # Add suppression feedback
        S[agent, t] += suppression_feedback(alpha_feedback, S[agent, t - 1], beta_feedback, K[agent, t - 1]) * dt

        # Check for quantum tunneling breakthrough events
        # (allows knowledge to "tunnel" through suppression barriers)
        if S[agent, t] > 0.5:  # Only consider significant suppression barriers
            tunnel_prob = quantum_tunneling_probability(
                barrier_height=S[agent, t],
                barrier_width=1.0,  # Constant width for simplicity
                energy_level=K[agent, t]
            )

            # If tunneling occurs, reduce suppression
            if np.random.random() < tunnel_prob:
                S[agent, t] *= 0.5  # Reduce suppression by half
                tunneling_events[t] += 1  # Record tunneling event

        # Update agent's conceptual position based on knowledge field gradient
        agent_positions[agent] += k_gradients[agent] * dt

        # Original intelligence growth with additional factors
        I[agent, t] = I[agent, t - 1] + intelligence_growth(
            K[agent, t], W, R, S[agent, t], N
        ) * dt

        # Add electromagnetic and quantum entanglement effects to intelligence
        I[agent, t] += (em_sum * 0.05 + qe_sum * 0.02) * dt

    # Civilization oscillation dynamics (same as original)
    osc_acceleration = civilization_oscillation(E[t - 1], dE_dt, gamma_osc, omega_osc)
    dE_dt += osc_acceleration * dt
    E[t] = E[t - 1] + dE_dt * dt

print("Simulation completed.")
print("Preparing visualization...")

# Prepare for visualization
time_range = np.arange(timesteps)

# Create a 3x2 grid of plots to show all dynamics
plt.figure(figsize=(15, 12))

# 1. Intelligence Growth
plt.subplot(3, 2, 1)
plt.plot(time_range, np.mean(I, axis=0), 'b-', linewidth=2, label='Avg Intelligence')
plt.title('Intelligence Growth Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Intelligence')
plt.legend()
plt.grid(True)

# 2. Truth Adoption
plt.subplot(3, 2, 2)
plt.plot(time_range, T, 'g-', linewidth=2, label='Truth Adoption')
plt.title('Truth Adoption Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Truth Level')
plt.legend()
plt.grid(True)

# 3. Suppression Dynamics
plt.subplot(3, 2, 3)
plt.plot(time_range, np.mean(S, axis=0), 'r-', linewidth=2, label='Avg Suppression')
plt.title('Suppression Dynamics')
plt.xlabel('Time Steps')
plt.ylabel('Suppression Level')
plt.legend()
plt.grid(True)

# 4. Civilization Oscillation
plt.subplot(3, 2, 4)
plt.plot(time_range, E, 'm-', linewidth=2, label='Civilization Oscillation')
plt.title('Civilization Oscillation Dynamics')
plt.xlabel('Time Steps')
plt.ylabel('Oscillation State')
plt.legend()
plt.grid(True)

# 5. Electromagnetic Field Influence (NEW)
plt.subplot(3, 2, 5)
plt.plot(time_range, np.mean(em_influence, axis=0), 'c-', linewidth=2, label='EM Field Influence')
plt.title('Knowledge Field (Electromagnetic) Influence')
plt.xlabel('Time Steps')
plt.ylabel('Field Strength')
plt.legend()
plt.grid(True)

# 6. Quantum Entanglement Correlation (NEW)
plt.subplot(3, 2, 6)
plt.plot(time_range, np.mean(qe_correlation, axis=0), 'y-', linewidth=2, label='Quantum Entanglement')
plt.plot(time_range, tunneling_events * 0.2, 'k--', linewidth=1, label='Tunneling Events')
plt.title('Quantum Entanglement & Tunneling')
plt.xlabel('Time Steps')
plt.ylabel('Correlation / Events')
plt.legend()
plt.grid(True)

plt.tight_layout()

# Save the enhanced plot
plot_file = plots_dir / "quantum_em_enhanced_simulation.png"
plt.savefig(str(plot_file))
print(f"✅ Enhanced plot saved at: {plot_file}")

# Save the simulation results to CSV
df_results = pd.DataFrame({
    'Time': time_range,
    'Avg_Intelligence': np.mean(I, axis=0),
    'Truth_Adoption': T,
    'Avg_Suppression': np.mean(S, axis=0),
    'Civilization_Oscillation': E,
    'Avg_EM_Influence': np.mean(em_influence, axis=0),
    'Avg_Quantum_Correlation': np.mean(qe_correlation, axis=0),
    'Tunneling_Events': tunneling_events
})

csv_file = data_dir / "quantum_em_enhanced_results.csv"
df_results.to_csv(csv_file, index=False)
print(f"✅ Enhanced data saved at: {csv_file}")

print("Displaying plot...")
plt.show()