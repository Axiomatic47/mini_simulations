import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

# Add parent directory to path to find modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import original equations and astrophysics extensions
from config.equations import (
    intelligence_growth, truth_adoption, wisdom_field,
    resistance_resurgence, suppression_feedback,
    civilization_oscillation, knowledge_growth_phase_transition
)

# Import astrophysics extensions
from config.astrophysics_extensions import (
    civilization_lifecycle_phase, suppression_event_horizon,
    cosmic_background_knowledge, knowledge_inflation,
    knowledge_gravitational_lensing, dark_energy_knowledge_acceleration,
    galactic_structure_model
)

# Import parameters
from config.parameters import (
    TIMESTEPS, DT, NUM_AGENTS, W_0, ALPHA_WISDOM, RESISTANCE, NETWORK_EFFECT,
    A_TRUTH, T_MAX, LAMBDA_DECAY, ALPHA_FEEDBACK, BETA_FEEDBACK,
    ALPHA_RESURGE, MU_RESURGE, T_CRIT_RESURGE,
    K_0_PHASE, A_PHASE, GAMMA_PHASE, T_CRIT_PHASE,
    GAMMA_OSC, OMEGA_OSC
)

# Ensure output directories exist
BASE_DIR = Path(__file__).resolve().parent.parent
plots_dir = BASE_DIR / 'outputs' / 'plots'
data_dir = BASE_DIR / 'outputs' / 'data'
plots_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)

print(f"Base directory: {BASE_DIR}")
print(f"Plots directory: {plots_dir}")
print(f"Data directory: {data_dir}")

# Astrophysics-specific parameters - refined values
LIFECYCLE_THRESHOLDS = np.array([50, 100, 200, 300, 350])  # Age thresholds for lifecycle phases
LIFECYCLE_INTENSITIES = np.array([0.5, 1.0, 1.5, 0.8, 1.2, 0.6])  # Intensity modifiers for each phase
INFLATION_THRESHOLD = 15  # Truth adoption threshold to trigger knowledge inflation
CRITICAL_CONSTANT = 1.5  # Constant for event horizon calculation (reduced to avoid extreme spikes)
BASE_BACKGROUND_KNOWLEDGE = 0.3  # Increased base level of cosmic background knowledge
DARK_ENERGY_ONSET = 100  # Earlier onset of dark energy effects
MINIMUM_INTELLIGENCE = 0.0  # Minimum bound for intelligence (prevent negative values)
REBIRTH_INTENSITY = 0.8  # Reduced rebirth intensity to avoid extreme spikes

# Simulation parameters
timesteps = TIMESTEPS
dt = DT
num_agents = NUM_AGENTS

# Initialize arrays
K = np.zeros((num_agents, timesteps))  # Knowledge
S = np.zeros((num_agents, timesteps))  # Suppression
I = np.zeros((num_agents, timesteps))  # Intelligence
T = np.zeros(timesteps)  # Truth adoption
E = np.zeros(timesteps)  # Civilization oscillation state

# Astrophysics-specific arrays
lifecycle_intensity = np.zeros(timesteps)
lifecycle_phase = np.zeros(timesteps, dtype=int)
event_horizon = np.zeros(timesteps)
beyond_horizon = np.zeros(timesteps, dtype=bool)
background_knowledge = np.zeros(timesteps)
inflation_multiplier = np.zeros(timesteps)
is_inflating = np.zeros(timesteps, dtype=bool)
apparent_truth = np.zeros(timesteps)
truth_distortion = np.zeros(timesteps)
dark_energy_effect = np.zeros(timesteps)
inflation_duration = np.zeros(timesteps)
knowledge_suppression_ratio = np.zeros(timesteps)  # New metric
critical_transitions = np.zeros(timesteps, dtype=bool)  # Track phase transitions

# Initial conditions
K[:, 0] = 1.0
S[:, 0] = np.linspace(5, 10, num_agents)
I[:, 0] = 5.0
T[0] = 1.0
E[0] = 0.05
dE_dt = 0.0  # Initial oscillation velocity

# Generate galactic structure influence matrix
influence_matrix = galactic_structure_model(num_agents)

# Track if inflation is active and for how long
inflation_active = False
inflation_start_time = 0

print("Starting simulation...")

# Main simulation loop
for t in range(1, timesteps):
    # Calculate astrophysics effects

    # 1. Stellar lifecycle analog
    lifecycle_intensity[t], lifecycle_phase[t] = civilization_lifecycle_phase(
        t, 1.0, LIFECYCLE_THRESHOLDS, LIFECYCLE_INTENSITIES
    )

    # Track phase transitions (for critical points visualization)
    if t > 1 and lifecycle_phase[t] != lifecycle_phase[t - 1]:
        critical_transitions[t] = True
        print(f"Phase transition at t={t}: Phase {lifecycle_phase[t - 1]} → {lifecycle_phase[t]}")

    # 2. Event horizon calculation
    avg_suppression = np.mean(S[:, t - 1])
    avg_knowledge = np.mean(K[:, t - 1])
    event_horizon[t], beyond_horizon[t] = suppression_event_horizon(
        avg_suppression, avg_knowledge, CRITICAL_CONSTANT
    )

    # Calculate knowledge/suppression ratio (for phase diagrams)
    knowledge_suppression_ratio[t] = avg_knowledge / max(0.1, avg_suppression)

    # 3. Cosmic background knowledge - with more realistic time dependency
    background_knowledge[t] = cosmic_background_knowledge(
        t, BASE_BACKGROUND_KNOWLEDGE * (1 + 0.005 * np.sqrt(t))  # Gradually increasing base knowledge
    )

    # 4. Check for inflation trigger
    if not inflation_active and T[t - 1] > INFLATION_THRESHOLD:
        inflation_active = True
        inflation_start_time = t
        critical_transitions[t] = True  # Mark as critical transition
        print(f"Knowledge inflation triggered at t={t}, T={T[t - 1]:.2f}")

    inflation_duration[t] = t - inflation_start_time if inflation_active else 0
    inflation_multiplier[t], is_inflating[t] = knowledge_inflation(
        np.mean(K[:, t - 1]), T[t - 1], INFLATION_THRESHOLD,
        expansion_rate=1.8, duration=inflation_duration[t]  # Reduced expansion rate
    )

    # 5. Calculate gravitational lensing effect (truth distortion)
    observer_distance = max(1.0, 10.0 - avg_suppression / 2)  # Closer with higher suppression
    apparent_truth[t], truth_distortion[t] = knowledge_gravitational_lensing(
        T[t - 1], avg_suppression, observer_distance
    )

    # 6. Calculate dark energy effect - earlier onset and more gradual growth
    if t > DARK_ENERGY_ONSET:
        dark_energy_effect[t] = dark_energy_knowledge_acceleration(
            t - DARK_ENERGY_ONSET, np.mean(K[:, t - 1]), unexplained_factor=0.01 + 0.0001 * t
        )
    else:
        dark_energy_effect[t] = 0

    # Truth adoption update with gravitational lensing effect
    # Use apparent truth for decision-making but actual truth for storage
    effective_truth_rate = truth_adoption(apparent_truth[t], A_TRUTH, T_MAX)
    T[t] = T[t - 1] + effective_truth_rate * dt

    # Update each agent
    for agent in range(num_agents):
        # Calculate wisdom
        W = wisdom_field(W_0, ALPHA_WISDOM, S[agent, t - 1], RESISTANCE, K[agent, t - 1])

        # Calculate social influence from galactic structure
        social_influence = 0
        for other in range(num_agents):
            if agent != other:
                social_influence += influence_matrix[agent, other] * K[other, t - 1] * 0.01

        # Knowledge growth with inflation effect and dark energy
        base_knowledge_growth = knowledge_growth_phase_transition(
            K[agent, t - 1], 0.01, t, A_PHASE, GAMMA_PHASE, T[t - 1], T_CRIT_PHASE
        )

        # Apply inflation multiplier if active, and ensure minimum background knowledge
        K[agent, t] = max(
            background_knowledge[t],
            base_knowledge_growth * inflation_multiplier[t] + dark_energy_effect[t] + social_influence
        )

        # Calculate suppression with lifecycle effects
        base_suppression = resistance_resurgence(
            S[agent, 0], LAMBDA_DECAY, t, ALPHA_RESURGE, MU_RESURGE, T_CRIT_RESURGE
        )

        # Add suppression feedback - more responsive to knowledge
        S[agent, t] = base_suppression + suppression_feedback(
            ALPHA_FEEDBACK, S[agent, t - 1], BETA_FEEDBACK * 1.5, K[agent, t - 1]  # Increased feedback
        ) * dt

        # Modify suppression based on lifecycle phase
        if lifecycle_phase[t] == 4:  # Collapse phase
            S[agent, t] *= 1.2  # Increased suppression during collapse
        elif lifecycle_phase[t] == 5:  # Remnant/rebirth phase
            S[agent, t] *= 0.8  # Decreased suppression during rebirth

        # Event horizon effect - if beyond horizon, knowledge growth is severely constrained
        if beyond_horizon[t]:
            # Knowledge is limited when beyond event horizon
            K[agent, t] = min(K[agent, t], K[agent, t - 1] * (1 - 0.05 * dt))
            S[agent, t] *= 1.1  # Suppression increases faster

        # Update intelligence with lifecycle intensity modifiers
        intelligence_change = intelligence_growth(
            K[agent, t], W, RESISTANCE, S[agent, t], NETWORK_EFFECT
        ) * dt * lifecycle_intensity[t]

        # Apply limits to intelligence - prevent excessive spikes and negative values
        if lifecycle_phase[t] == 5:  # Rebirth phase
            # Limit rebirth intensity to avoid extreme spikes
            intelligence_change *= REBIRTH_INTENSITY

        I[agent, t] = max(MINIMUM_INTELLIGENCE, I[agent, t - 1] + intelligence_change)

    # Civilization oscillation dynamics
    osc_acceleration = civilization_oscillation(E[t - 1], dE_dt, GAMMA_OSC, OMEGA_OSC)
    dE_dt += osc_acceleration * dt
    E[t] = E[t - 1] + dE_dt * dt

print("Simulation completed.")
print("Preparing visualization...")

# Prepare for visualization
time_range = np.arange(timesteps)

# Create a custom layout with GridSpec for better control
plt.figure(figsize=(16, 14))
gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1.5])

# 1. Intelligence & Knowledge Growth
ax1 = plt.subplot(gs[0, 0])
ax1.plot(time_range, np.mean(I, axis=0), 'b-', linewidth=2, label='Avg Intelligence')
ax1.plot(time_range, np.mean(K, axis=0), 'g--', linewidth=2, label='Avg Knowledge')
ax1.set_title('Intelligence & Knowledge Growth')
ax1.set_xlabel('Time Steps')
ax1.set_ylabel('Level')
ax1.legend()
ax1.grid(True)

# 2. Truth Adoption and Distortion
ax2 = plt.subplot(gs[0, 1])
ax2.plot(time_range, T, 'g-', linewidth=2, label='Actual Truth')
ax2.plot(time_range, apparent_truth, 'g--', linewidth=2, label='Apparent Truth')
ax2.fill_between(time_range, T, apparent_truth, color='g', alpha=0.3, label='Distortion')
ax2.set_title('Truth Adoption & Gravitational Lensing')
ax2.set_xlabel('Time Steps')
ax2.set_ylabel('Truth Level')
ax2.legend()
ax2.grid(True)

# 3. Suppression and Event Horizon - with log scale
ax3 = plt.subplot(gs[1, 0])
ax3.plot(time_range, np.mean(S, axis=0), 'r-', linewidth=2, label='Avg Suppression')
ax3.set_title('Suppression Dynamics')
ax3.set_xlabel('Time Steps')
ax3.set_ylabel('Suppression Level')
ax3.legend()
ax3.grid(True)

# 3b. Event Horizon plotted separately with log scale
ax3b = plt.subplot(gs[1, 1])
ax3b.semilogy(time_range, event_horizon, 'k-', linewidth=2, label='Event Horizon')
ax3b.fill_between(time_range, 0.1, 100, where=beyond_horizon, color='k', alpha=0.3, label='Beyond Horizon')
# Mark critical transitions on the event horizon plot
for t in np.where(critical_transitions)[0]:
    if t > 0:  # Skip first timestep
        ax3b.axvline(x=t, color='r', linestyle='--', alpha=0.5)
ax3b.set_title('Event Horizon (Log Scale)')
ax3b.set_xlabel('Time Steps')
ax3b.set_ylabel('Horizon Threshold (log scale)')
ax3b.set_ylim(0.1, max(100, np.max(event_horizon)))
ax3b.legend()
ax3b.grid(True)

# 4. Civilization Lifecycle
ax4 = plt.subplot(gs[2, 0])
ax4.plot(time_range, lifecycle_intensity, 'm-', linewidth=2, label='Lifecycle Intensity')
scatter_colors = ['blue', 'green', 'purple', 'orange', 'red', 'brown']
phase_names = ['Formation', 'Growth', 'Peak', 'Decline', 'Collapse', 'Rebirth']
for phase in range(6):
    phase_times = np.where(lifecycle_phase == phase)[0]
    if len(phase_times) > 0:
        ax4.scatter(phase_times, lifecycle_intensity[phase_times],
                    color=scatter_colors[phase], s=10,
                    label=f'Phase {phase}: {phase_names[phase]}')

ax4.set_title('Civilization Lifecycle (Stellar Analogy)')
ax4.set_xlabel('Time Steps')
ax4.set_ylabel('Intensity')
ax4.legend()
ax4.grid(True)

# 5. Knowledge Inflation & Dark Energy
ax5 = plt.subplot(gs[2, 1])
ax5.plot(time_range, inflation_multiplier, 'c-', linewidth=2, label='Inflation Multiplier')
ax5.plot(time_range, dark_energy_effect, 'y-', linewidth=2, label='Dark Energy Effect')
ax5.fill_between(time_range, 1, inflation_multiplier, where=is_inflating,
                 color='c', alpha=0.3, label='Inflation Active')
ax5.axhline(y=1.0, color='gray', linestyle='--')
ax5.set_title('Knowledge Inflation & Dark Energy')
ax5.set_xlabel('Time Steps')
ax5.set_ylabel('Multiplier / Effect')
ax5.legend()
ax5.grid(True)

# 6. Phase Diagram: Knowledge vs. Suppression
ax6 = plt.subplot(gs[3, :])
# Create a scatter plot with color representing time
scatter = ax6.scatter(
    np.mean(S, axis=0),
    np.mean(K, axis=0),
    c=time_range,
    cmap='viridis',
    s=10,
    alpha=0.7
)

# Add colorbar for time reference
cbar = plt.colorbar(scatter, ax=ax6)
cbar.set_label('Time Steps')

# Plot the trajectory with a line
ax6.plot(np.mean(S, axis=0), np.mean(K, axis=0), 'k-', alpha=0.3, linewidth=1)

# Mark critical transitions and phase changes
for t in np.where(critical_transitions)[0]:
    if t > 0:  # Skip first timestep
        ax6.plot(np.mean(S[:, t]), np.mean(K[:, t]), 'ro', markersize=8)
        ax6.annotate(f't={t}',
                     xy=(np.mean(S[:, t]), np.mean(K[:, t])),
                     xytext=(10, 10),
                     textcoords='offset points',
                     fontsize=8)

# Draw event horizon boundary
# This is an approximation of the S/K² = constant curve
S_range = np.linspace(0.1, max(np.mean(S, axis=0)) * 1.2, 100)
for critical_value in [0.5, 1.0, 2.0]:
    K_boundary = np.sqrt(CRITICAL_CONSTANT * S_range / critical_value)
    ax6.plot(S_range, K_boundary, 'k--', alpha=0.5, linewidth=1)
    # Add label at the middle of the curve
    middle_idx = len(S_range) // 2
    ax6.annotate(f'Horizon {critical_value}',
                 xy=(S_range[middle_idx], K_boundary[middle_idx]),
                 xytext=(10, 0),
                 textcoords='offset points',
                 fontsize=8,
                 alpha=0.7)

ax6.set_title('Phase Diagram: Knowledge vs. Suppression')
ax6.set_xlabel('Suppression Level')
ax6.set_ylabel('Knowledge Level')
ax6.grid(True)

plt.tight_layout()

# Save the enhanced plot
plot_file = plots_dir / "astrophysics_simulation_improved_results.png"
plt.savefig(str(plot_file))
print(f"✅ Enhanced plot saved at: {plot_file}")

# Create a correlation matrix
correlation_data = pd.DataFrame({
    'Intelligence': np.mean(I, axis=0),
    'Knowledge': np.mean(K, axis=0),
    'Truth': T,
    'Suppression': np.mean(S, axis=0),
    'EventHorizon': event_horizon,
    'LifecycleIntensity': lifecycle_intensity,
    'Phase': lifecycle_phase,
    'Inflation': inflation_multiplier,
    'DarkEnergy': dark_energy_effect
})

# Calculate correlation matrix
correlation_matrix = correlation_data.corr()

# Plot the correlation matrix
plt.figure(figsize=(12, 10))
plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Correlation Coefficient')
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45, ha='right')
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title('Correlation Matrix of Simulation Variables')

# Add correlation values
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                 ha='center', va='center',
                 color='white' if abs(correlation_matrix.iloc[i, j]) > 0.5 else 'black')

plt.tight_layout()
corr_file = plots_dir / "astrophysics_correlation_matrix.png"
plt.savefig(str(corr_file))
print(f"✅ Correlation matrix saved at: {corr_file}")

# Save the simulation results to CSV
df_results = pd.DataFrame({
    'Time': time_range,
    'Avg_Intelligence': np.mean(I, axis=0),
    'Avg_Knowledge': np.mean(K, axis=0),
    'Truth_Adoption': T,
    'Apparent_Truth': apparent_truth,
    'Truth_Distortion': truth_distortion,
    'Avg_Suppression': np.mean(S, axis=0),
    'Event_Horizon': event_horizon,
    'Beyond_Horizon': beyond_horizon,
    'Lifecycle_Intensity': lifecycle_intensity,
    'Lifecycle_Phase': lifecycle_phase,
    'Inflation_Multiplier': inflation_multiplier,
    'Is_Inflating': is_inflating,
    'Dark_Energy_Effect': dark_energy_effect,
    'Background_Knowledge': background_knowledge,
    'Knowledge_Suppression_Ratio': knowledge_suppression_ratio,
    'Critical_Transition': critical_transitions
})

csv_file = data_dir / "astrophysics_simulation_improved_results.csv"
df_results.to_csv(csv_file, index=False)
print(f"✅ Enhanced data saved at: {csv_file}")

print("Displaying plots...")
plt.show()