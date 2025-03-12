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

# Import dimensional consistency tools
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

# Bounds for numerical stability
MAX_KNOWLEDGE = 500.0
MAX_SUPPRESSION = 100.0
MAX_INTELLIGENCE = 100.0
MAX_TRUTH = 100.0
MIN_KNOWLEDGE = 0.0
MIN_SUPPRESSION = 0.0
MIN_INTELLIGENCE = 0.0
MIN_TRUTH = 0.0
MAX_APPARENT_TRUTH = 100.0
MIN_APPARENT_TRUTH = 0.0
MIN_DISTANCE = 0.1  # Minimum distance to prevent division by zero

# Enable or disable dimensional analysis
use_dimensional_analysis = True

# Initialize circuit breaker
circuit_breaker = CircuitBreaker(
    threshold=1e-6,
    max_value=MAX_KNOWLEDGE,
    min_value=MIN_KNOWLEDGE,
    max_rate_of_change=10.0
)


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


# Astrophysics-specific parameters - refined values with safety bounds
LIFECYCLE_THRESHOLDS = np.array([50, 100, 200, 300, 350])  # Age thresholds for lifecycle phases
LIFECYCLE_INTENSITIES = np.array([0.5, 1.0, 1.5, 0.8, 1.2, 0.6])  # Intensity modifiers for each phase
INFLATION_THRESHOLD = 15  # Truth adoption threshold to trigger knowledge inflation
CRITICAL_CONSTANT = 1.5  # Constant for event horizon calculation (reduced to avoid extreme spikes)
BASE_BACKGROUND_KNOWLEDGE = 0.3  # Increased base level of cosmic background knowledge
DARK_ENERGY_ONSET = 100  # Earlier onset of dark energy effects
MINIMUM_INTELLIGENCE = 0.0  # Minimum bound for intelligence (prevent negative values)
REBIRTH_INTENSITY = 0.8  # Reduced rebirth intensity to avoid extreme spikes
MAX_INFLATION_MULTIPLIER = 5.0  # Maximum inflation multiplier to prevent runaway growth

# Simulation parameters
timesteps = TIMESTEPS
dt = DT
num_agents = NUM_AGENTS

# Enable adaptive timestep for stability
enable_adaptive_timestep = True
adaptive_timestep = dt
MIN_TIMESTEP = 0.2
MAX_TIMESTEP = 2.0

# Stability tracking
stability_issues = 0

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

# If using dimensional analysis, set up dimensional containers
if use_dimensional_analysis:
    print("Using dimensional analysis for calculations")
    # Arrays to store dimensional values during simulation
    K_dim = [[None for _ in range(timesteps)] for _ in range(num_agents)]
    S_dim = [[None for _ in range(timesteps)] for _ in range(num_agents)]
    I_dim = [[None for _ in range(timesteps)] for _ in range(num_agents)]
    T_dim = [None for _ in range(timesteps)]
    E_dim = [None for _ in range(timesteps)]

    # Initialize first timestep with dimensional values
    for agent in range(num_agents):
        K_dim[agent][0] = DimensionalValue(K[agent, 0], Dimension.KNOWLEDGE)
        S_dim[agent][0] = DimensionalValue(S[agent, 0], Dimension.SUPPRESSION)
        I_dim[agent][0] = DimensionalValue(I[agent, 0], Dimension.INTELLIGENCE)

    T_dim[0] = DimensionalValue(T[0], Dimension.TRUTH)
    E_dim[0] = DimensionalValue(E[0], Dimension.DIMENSIONLESS)

    # Create dimensional constants
    R = DimensionalValue(RESISTANCE, Dimension.RESISTANCE)

# Generate galactic structure influence matrix
influence_matrix = galactic_structure_model(num_agents)

# Track if inflation is active and for how long
inflation_active = False
inflation_start_time = 0

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
        if max_change > 5.0:
            adaptive_timestep = max(MIN_TIMESTEP, adaptive_timestep * 0.8)
        elif max_change < 0.5:
            adaptive_timestep = min(MAX_TIMESTEP, adaptive_timestep * 1.2)

        current_dt = adaptive_timestep
    else:
        current_dt = dt

    # Calculate astrophysics effects

    # 1. Stellar lifecycle analog with safety checks
    try:
        lifecycle_intensity[t], lifecycle_phase[t] = civilization_lifecycle_phase(
            t, 1.0, LIFECYCLE_THRESHOLDS, LIFECYCLE_INTENSITIES
        )
        lifecycle_intensity[t] = np.clip(lifecycle_intensity[t], 0.1, 2.0)  # Bound intensity
    except Exception as e:
        print(f"Warning: Error in lifecycle calculation at t={t}: {e}")
        lifecycle_intensity[t] = lifecycle_intensity[t - 1]  # Use previous value
        lifecycle_phase[t] = lifecycle_phase[t - 1]
        stability_issues += 1

    # Track phase transitions (for critical points visualization)
    if t > 1 and lifecycle_phase[t] != lifecycle_phase[t - 1]:
        critical_transitions[t] = True
        print(f"Phase transition at t={t}: Phase {lifecycle_phase[t - 1]} → {lifecycle_phase[t]}")

    # 2. Event horizon calculation with safety measures
    try:
        avg_suppression = np.mean(S[:, t - 1])
        avg_knowledge = np.mean(K[:, t - 1])

        # Safety check to prevent division by zero or extreme values
        event_horizon[t], beyond_horizon[t] = suppression_event_horizon(
            np.clip(avg_suppression, 0.1, MAX_SUPPRESSION),
            np.clip(avg_knowledge, 0.1, MAX_KNOWLEDGE),
            np.clip(CRITICAL_CONSTANT, 0.1, 5.0)
        )

        # Additional check on event horizon value
        event_horizon[t] = np.clip(event_horizon[t], 0.1, 100.0)
    except Exception as e:
        print(f"Warning: Error in event horizon calculation at t={t}: {e}")
        event_horizon[t] = event_horizon[t - 1]  # Use previous value
        beyond_horizon[t] = beyond_horizon[t - 1]
        stability_issues += 1

    # Calculate knowledge/suppression ratio (for phase diagrams) with safety
    if avg_suppression > 0.1:
        knowledge_suppression_ratio[t] = safe_div(avg_knowledge, avg_suppression, default=1.0)
    else:
        knowledge_suppression_ratio[t] = 10.0  # Default high ratio when suppression is near zero

    # 3. Cosmic background knowledge - with more realistic time dependency and bounds
    try:
        background_knowledge[t] = cosmic_background_knowledge(
            t, np.clip(BASE_BACKGROUND_KNOWLEDGE * (1 + 0.005 * np.sqrt(t)), 0.1, 10.0)
        )
        background_knowledge[t] = np.clip(background_knowledge[t], 0.1, 10.0)
    except Exception as e:
        print(f"Warning: Error in background knowledge calculation at t={t}: {e}")
        background_knowledge[t] = background_knowledge[t - 1]  # Use previous value
        stability_issues += 1

    # 4. Check for inflation trigger with safety checks
    if not inflation_active and T[t - 1] > INFLATION_THRESHOLD:
        inflation_active = True
        inflation_start_time = t
        critical_transitions[t] = True  # Mark as critical transition
        print(f"Knowledge inflation triggered at t={t}, T={T[t - 1]:.2f}")

    inflation_duration[t] = t - inflation_start_time if inflation_active else 0

    try:
        inflation_multiplier[t], is_inflating[t] = knowledge_inflation(
            np.clip(np.mean(K[:, t - 1]), 0.1, MAX_KNOWLEDGE),
            np.clip(T[t - 1], 0.1, MAX_TRUTH),
            INFLATION_THRESHOLD,
            expansion_rate=np.clip(1.8, 1.0, 3.0),
            duration=np.clip(inflation_duration[t], 0, 100)
        )

        # Bound inflation multiplier to prevent runaway growth
        inflation_multiplier[t] = np.clip(inflation_multiplier[t], 1.0, MAX_INFLATION_MULTIPLIER)
    except Exception as e:
        print(f"Warning: Error in inflation calculation at t={t}: {e}")
        inflation_multiplier[t] = 1.0  # Safe default
        is_inflating[t] = False
        stability_issues += 1

    # 5. Calculate gravitational lensing effect (truth distortion) with safety
    try:
        # Ensure minimum distance to prevent extreme distortion
        observer_distance = max(MIN_DISTANCE, 10.0 - avg_suppression / 2)
        apparent_truth[t], truth_distortion[t] = knowledge_gravitational_lensing(
            np.clip(T[t - 1], MIN_TRUTH, MAX_TRUTH),
            np.clip(avg_suppression, MIN_SUPPRESSION, MAX_SUPPRESSION),
            observer_distance
        )

        # Bound apparent truth to reasonable values
        apparent_truth[t] = np.clip(apparent_truth[t], MIN_APPARENT_TRUTH, MAX_APPARENT_TRUTH)
        truth_distortion[t] = np.clip(truth_distortion[t], -MAX_TRUTH, MAX_TRUTH)
    except Exception as e:
        print(f"Warning: Error in gravitational lensing calculation at t={t}: {e}")
        apparent_truth[t] = T[t - 1]  # Default to actual truth
        truth_distortion[t] = 0.0
        stability_issues += 1

    # 6. Calculate dark energy effect with safety measures
    try:
        if t > DARK_ENERGY_ONSET:
            dark_energy_effect[t] = dark_energy_knowledge_acceleration(
                np.clip(t - DARK_ENERGY_ONSET, 0, 1000),
                np.clip(np.mean(K[:, t - 1]), MIN_KNOWLEDGE, MAX_KNOWLEDGE),
                unexplained_factor=np.clip(0.01 + 0.0001 * t, 0.001, 0.1)
            )

            # Bound dark energy effect
            dark_energy_effect[t] = np.clip(dark_energy_effect[t], 0.0, 5.0)
        else:
            dark_energy_effect[t] = 0.0
    except Exception as e:
        print(f"Warning: Error in dark energy calculation at t={t}: {e}")
        dark_energy_effect[t] = dark_energy_effect[t - 1]  # Use previous value
        stability_issues += 1

    # Truth adoption update with gravitational lensing effect
    # Use apparent truth for decision-making but actual truth for storage
    try:
        if use_dimensional_analysis:
            # Create dimensional value for apparent truth
            apparent_truth_dim = DimensionalValue(apparent_truth[t], Dimension.TRUTH)

            # Use dimensional version
            effective_truth_rate_dim = truth_adoption_with_dimensions(
                apparent_truth_dim, A_TRUTH, T_MAX
            )
            effective_truth_rate = effective_truth_rate_dim.value
        else:
            # Use original version
            effective_truth_rate = truth_adoption(
                np.clip(apparent_truth[t], MIN_APPARENT_TRUTH, MAX_APPARENT_TRUTH),
                A_TRUTH,
                T_MAX
            )

        # Update truth with stability check
        if circuit_breaker.check_value_stability(effective_truth_rate):
            effective_truth_rate = np.clip(effective_truth_rate, -2.0, 2.0)
            stability_issues += 1

        T[t] = np.clip(T[t - 1] + effective_truth_rate * current_dt, MIN_TRUTH, MAX_TRUTH)

        # Store dimensional value if using dimensional analysis
        if use_dimensional_analysis:
            T_dim[t] = DimensionalValue(T[t], Dimension.TRUTH)
    except Exception as e:
        print(f"Warning: Error in truth adoption calculation at t={t}: {e}")
        T[t] = T[t - 1]  # Use previous value in case of error
        if use_dimensional_analysis:
            T_dim[t] = T_dim[t - 1]  # Use previous dimensional value
        stability_issues += 1

    # Update each agent
    for agent in range(num_agents):
        try:
            if use_dimensional_analysis:
                # Use dimensional versions
                K_curr_dim = K_dim[agent][t - 1]
                S_curr_dim = S_dim[agent][t - 1]

                # Calculate wisdom with dimensional analysis
                W_dim = wisdom_field_with_dimensions(
                    W_0, ALPHA_WISDOM, S_curr_dim, R, K_curr_dim)
                W = W_dim.value  # Extract raw value for other calculations
            else:
                # Calculate wisdom with bounds (original version)
                W = wisdom_field(
                    W_0,
                    ALPHA_WISDOM,
                    np.clip(S[agent, t - 1], MIN_SUPPRESSION, MAX_SUPPRESSION),
                    RESISTANCE,
                    np.clip(K[agent, t - 1], MIN_KNOWLEDGE, MAX_KNOWLEDGE)
                )

            # Calculate social influence from galactic structure with bounds
            social_influence = 0
            for other in range(num_agents):
                if agent != other:
                    influence_value = np.clip(
                        influence_matrix[agent, other] * K[other, t - 1] * 0.01,
                        -5.0, 5.0  # Bound influence
                    )
                    social_influence += influence_value

            # Knowledge growth with inflation effect and dark energy
            base_knowledge_growth = knowledge_growth_phase_transition(
                np.clip(K[agent, t - 1], MIN_KNOWLEDGE, MAX_KNOWLEDGE),
                0.01,
                t,
                A_PHASE,
                GAMMA_PHASE,
                np.clip(T[t - 1], MIN_TRUTH, MAX_TRUTH),
                T_CRIT_PHASE
            )

            # Check for stability in knowledge growth
            if circuit_breaker.check_value_stability(base_knowledge_growth):
                base_knowledge_growth = np.clip(base_knowledge_growth, 0.0, K[agent, t - 1] * 0.1)
                stability_issues += 1

            # Apply inflation multiplier if active, and ensure minimum background knowledge
            knowledge_increment = (
                                          base_knowledge_growth * inflation_multiplier[t] +
                                          dark_energy_effect[t] +
                                          social_influence
                                  ) * current_dt

            # Apply bounds to knowledge increment
            knowledge_increment = np.clip(knowledge_increment, -MAX_KNOWLEDGE * 0.1, MAX_KNOWLEDGE * 0.1)

            # Update knowledge with bounds
            K[agent, t] = np.clip(
                max(background_knowledge[t], K[agent, t - 1] + knowledge_increment),
                MIN_KNOWLEDGE,
                MAX_KNOWLEDGE
            )

            # Store dimensional value if using dimensional analysis
            if use_dimensional_analysis:
                K_dim[agent][t] = DimensionalValue(K[agent, t], Dimension.KNOWLEDGE)

            # Calculate suppression with lifecycle effects
            if use_dimensional_analysis:
                # Use dimensional version
                s_base_dim = resistance_resurgence_with_dimensions(
                    S[agent, 0], LAMBDA_DECAY, t, ALPHA_RESURGE, MU_RESURGE, T_CRIT_RESURGE)
                s_base = s_base_dim.value
            else:
                # Use original version
                s_base = resistance_resurgence(
                    S[agent, 0],
                    LAMBDA_DECAY,
                    t,
                    ALPHA_RESURGE,
                    MU_RESURGE,
                    T_CRIT_RESURGE
                )

            # Add suppression feedback - more responsive to knowledge
            if use_dimensional_analysis:
                # Use dimensional version
                K_updated_dim = K_dim[agent][t]
                S_updated_dim = DimensionalValue(S[agent, t - 1], Dimension.SUPPRESSION)

                s_feedback_dim = suppression_feedback_with_dimensions(
                    ALPHA_FEEDBACK, S_updated_dim,
                    BETA_FEEDBACK * 1.5, K_updated_dim)
                s_feedback = s_feedback_dim.value
            else:
                # Use original version
                s_feedback = suppression_feedback(
                    ALPHA_FEEDBACK,
                    np.clip(S[agent, t - 1], MIN_SUPPRESSION, MAX_SUPPRESSION),
                    BETA_FEEDBACK * 1.5,
                    np.clip(K[agent, t - 1], MIN_KNOWLEDGE, MAX_KNOWLEDGE)
                ) * current_dt

            # Check for stability in suppression calculation
            if circuit_breaker.check_value_stability(s_feedback):
                s_feedback = np.clip(s_feedback, -2.0, 2.0)
                stability_issues += 1

            # Apply current_dt to s_feedback if using dimensional version (not included in the function)
            if use_dimensional_analysis:
                s_feedback *= current_dt

            # Update suppression with bounds
            suppression_increment = s_base + s_feedback
            S[agent, t] = np.clip(suppression_increment, MIN_SUPPRESSION, MAX_SUPPRESSION)

            # Modify suppression based on lifecycle phase
            if lifecycle_phase[t] == 4:  # Collapse phase
                S[agent, t] = np.clip(S[agent, t] * 1.2, MIN_SUPPRESSION, MAX_SUPPRESSION)
            elif lifecycle_phase[t] == 5:  # Remnant/rebirth phase
                S[agent, t] = np.clip(S[agent, t] * 0.8, MIN_SUPPRESSION, MAX_SUPPRESSION)

            # Store dimensional value if using dimensional analysis
            if use_dimensional_analysis:
                S_dim[agent][t] = DimensionalValue(S[agent, t], Dimension.SUPPRESSION)

            # Event horizon effect - if beyond horizon, knowledge growth is severely constrained
            if beyond_horizon[t]:
                # Knowledge is limited when beyond event horizon
                K[agent, t] = np.clip(
                    min(K[agent, t], K[agent, t - 1] * (1 - 0.05 * current_dt)),
                    MIN_KNOWLEDGE,
                    MAX_KNOWLEDGE
                )
                S[agent, t] = np.clip(S[agent, t] * 1.1, MIN_SUPPRESSION, MAX_SUPPRESSION)

                # Update dimensional value if changed
                if use_dimensional_analysis:
                    K_dim[agent][t] = DimensionalValue(K[agent, t], Dimension.KNOWLEDGE)
                    S_dim[agent][t] = DimensionalValue(S[agent, t], Dimension.SUPPRESSION)

            # Update intelligence with lifecycle intensity modifiers
            if use_dimensional_analysis:
                # Use dimensional version
                K_final_dim = K_dim[agent][t]
                S_final_dim = S_dim[agent][t]

                i_growth_dim = intelligence_growth_with_dimensions(
                    K_final_dim, W_dim, R, S_final_dim, NETWORK_EFFECT)
                i_growth = i_growth_dim.value
            else:
                # Use original version
                i_growth = intelligence_growth(
                    np.clip(K[agent, t], MIN_KNOWLEDGE, MAX_KNOWLEDGE),
                    W,
                    RESISTANCE,
                    np.clip(S[agent, t], MIN_SUPPRESSION, MAX_SUPPRESSION),
                    NETWORK_EFFECT
                )

            # Apply lifecycle intensity to intelligence growth
            i_growth *= current_dt * lifecycle_intensity[t]

            # Check for stability in intelligence growth
            if circuit_breaker.check_value_stability(i_growth):
                i_growth = np.clip(i_growth, -5.0, 5.0)
                stability_issues += 1

            # Apply limits to intelligence - prevent excessive spikes and negative values
            if lifecycle_phase[t] == 5:  # Rebirth phase
                # Limit rebirth intensity to avoid extreme spikes
                i_growth *= np.clip(REBIRTH_INTENSITY, 0.1, 1.0)

            I[agent, t] = np.clip(
                max(MINIMUM_INTELLIGENCE, I[agent, t - 1] + i_growth),
                MIN_INTELLIGENCE,
                MAX_INTELLIGENCE
            )

            # Store dimensional value if using dimensional analysis
            if use_dimensional_analysis:
                I_dim[agent][t] = DimensionalValue(I[agent, t], Dimension.INTELLIGENCE)

        except Exception as e:
            print(f"Warning: Error updating agent {agent} at t={t}: {e}")
            # Fall back to previous values in case of error
            K[agent, t] = K[agent, t - 1]
            S[agent, t] = S[agent, t - 1]
            I[agent, t] = I[agent, t - 1]

            # Store dimensional value if using dimensional analysis
            if use_dimensional_analysis:
                K_dim[agent][t] = K_dim[agent][t - 1]
                S_dim[agent][t] = S_dim[agent][t - 1]
                I_dim[agent][t] = I_dim[agent][t - 1]

            stability_issues += 1

    # Civilization oscillation dynamics with bounds
    try:
        osc_acceleration = civilization_oscillation(E[t - 1], dE_dt, GAMMA_OSC, OMEGA_OSC)

        # Check for stability in oscillation
        if circuit_breaker.check_value_stability(osc_acceleration):
            osc_acceleration = np.clip(osc_acceleration, -0.1, 0.1)
            stability_issues += 1

        dE_dt = np.clip(dE_dt + osc_acceleration * current_dt, -1.0, 1.0)
        E[t] = np.clip(E[t - 1] + dE_dt * current_dt, -1.0, 1.0)

        # Store dimensional value if using dimensional analysis
        if use_dimensional_analysis:
            E_dim[t] = DimensionalValue(E[t], Dimension.DIMENSIONLESS)
    except Exception as e:
        print(f"Warning: Error in oscillation calculation at t={t}: {e}")
        dE_dt = 0.0  # Reset acceleration
        E[t] = E[t - 1]  # Use previous value

        # Store dimensional value if using dimensional analysis
        if use_dimensional_analysis:
            E_dim[t] = E_dim[t - 1]

        stability_issues += 1

    # Output progress and current stability state
    if t % 100 == 0 or stability_issues > t / 10:  # Report more frequently if many issues
        print(
            f"Step {t}/{timesteps} completed. Current timestep: {current_dt:.4f}. Stability issues: {stability_issues}")

print(f"Simulation completed with {stability_issues} stability issues detected.")

# Dimensional consistency check if enabled
if use_dimensional_analysis:
    try:
        # Define the dimensional functions to check
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
            {"Function": name, "Status": result["status"], "Notes": result.get("message", "")}
            for name, result in consistency_results.items()
        ])
        consistency_df.to_csv(data_dir / "astrophysics_dimensional_consistency.csv", index=False)
        print(f"Dimensional consistency results saved to: {data_dir / 'astrophysics_dimensional_consistency.csv'}")
    except Exception as e:
        print(f"Error during dimensional consistency check: {e}")

# Replace any NaN or inf values that might have slipped through
K = np.nan_to_num(K, nan=1.0, posinf=MAX_KNOWLEDGE, neginf=MIN_KNOWLEDGE)
S = np.nan_to_num(S, nan=1.0, posinf=MAX_SUPPRESSION, neginf=MIN_SUPPRESSION)
I = np.nan_to_num(I, nan=1.0, posinf=MAX_INTELLIGENCE, neginf=MIN_INTELLIGENCE)
T = np.nan_to_num(T, nan=1.0, posinf=MAX_TRUTH, neginf=MIN_TRUTH)
E = np.nan_to_num(E, nan=0.0, posinf=1.0, neginf=-1.0)
lifecycle_intensity = np.nan_to_num(lifecycle_intensity, nan=1.0, posinf=2.0, neginf=0.1)
event_horizon = np.nan_to_num(event_horizon, nan=1.0, posinf=100.0, neginf=0.1)
inflation_multiplier = np.nan_to_num(inflation_multiplier, nan=1.0, posinf=MAX_INFLATION_MULTIPLIER, neginf=1.0)
apparent_truth = np.nan_to_num(apparent_truth, nan=1.0, posinf=MAX_APPARENT_TRUTH, neginf=MIN_APPARENT_TRUTH)
truth_distortion = np.nan_to_num(truth_distortion, nan=0.0, posinf=MAX_TRUTH, neginf=-MAX_TRUTH)
dark_energy_effect = np.nan_to_num(dark_energy_effect, nan=0.0, posinf=5.0, neginf=0.0)
knowledge_suppression_ratio = np.nan_to_num(knowledge_suppression_ratio, nan=1.0, posinf=10.0, neginf=0.1)

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
ax1.set_ylim(0, np.max(np.mean(K, axis=0)) * 1.1)  # Set appropriate y-limit

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
ax2.set_ylim(0, MAX_TRUTH)  # Set appropriate y-limit

# 3. Suppression and Event Horizon - with log scale
ax3 = plt.subplot(gs[1, 0])
ax3.plot(time_range, np.mean(S, axis=0), 'r-', linewidth=2, label='Avg Suppression')
ax3.set_title('Suppression Dynamics')
ax3.set_xlabel('Time Steps')
ax3.set_ylabel('Suppression Level')
ax3.legend()
ax3.grid(True)
ax3.set_ylim(0, np.max(np.mean(S, axis=0)) * 1.1)  # Set appropriate y-limit

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
ax4.set_ylim(0, 2)  # Set appropriate y-limit

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
ax5.set_ylim(0, np.max(inflation_multiplier) * 1.1)  # Set appropriate y-limit

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
    K_boundary = np.sqrt(safe_div(CRITICAL_CONSTANT * S_range, critical_value, default=0.0))
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

# Add annotation about dimensional analysis if used
if use_dimensional_analysis:
    plt.figtext(0.5, 0.01, "Using dimensional analysis", ha="center", fontsize=10,
                bbox={"facecolor": "lightgray", "alpha": 0.5, "pad": 5})

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

# Save stability metrics
stability_metrics = {
    'Total_Stability_Issues': stability_issues,
    'Circuit_Breaker_Triggers': circuit_breaker.trigger_count,
    'Max_Knowledge': np.max(K),
    'Max_Suppression': np.max(S),
    'Max_Intelligence': np.max(I),
    'Max_Truth': np.max(T),
    'Max_Inflation': np.max(inflation_multiplier),
    'Final_Timestep': adaptive_timestep if enable_adaptive_timestep else dt,
    'Used_Dimensional_Analysis': use_dimensional_analysis
}

stability_df = pd.DataFrame([stability_metrics])
stability_df.to_csv(data_dir / "astrophysics_simulation_stability.csv", index=False)
print(f"✅ Stability metrics saved at: {data_dir / 'astrophysics_simulation_stability.csv'}")

print("Displaying plots...")
plt.show()