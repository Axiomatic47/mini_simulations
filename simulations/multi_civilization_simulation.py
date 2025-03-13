# simulations/multi_civilization_simulation.py

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

# Add parent directory to path to find modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import regular extensions
from config.equations import (
    intelligence_growth, truth_adoption, wisdom_field,
    resistance_resurgence, suppression_feedback
)

# Import astrophysics extensions
from config.astrophysics_extensions import (
    civilization_lifecycle_phase, suppression_event_horizon,
    cosmic_background_knowledge, knowledge_inflation
)

# Import multi-civilization extensions
from config.multi_civilization_extensions import (
    initialize_civilizations, calculate_distance_matrix, calculate_interaction_strength,
    process_all_civilization_interactions
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

# Ensure output directories exist
BASE_DIR = Path(__file__).resolve().parent.parent
plots_dir = BASE_DIR / 'outputs' / 'plots'
data_dir = BASE_DIR / 'outputs' / 'data'
animation_dir = BASE_DIR / 'outputs' / 'animations'
plots_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)
animation_dir.mkdir(parents=True, exist_ok=True)

print(f"Base directory: {BASE_DIR}")
print(f"Plots directory: {plots_dir}")
print(f"Data directory: {data_dir}")
print(f"Animation directory: {animation_dir}")

# Enable or disable dimensional analysis
use_dimensional_analysis = True

# Numerical stability parameters
MAX_KNOWLEDGE = 100.0
MAX_SUPPRESSION = 50.0
MAX_INTELLIGENCE = 100.0
MAX_TRUTH = 100.0
MAX_INFLUENCE = 10.0
MAX_RESOURCES = 1000.0
MIN_VALUE = 0.0
MIN_DISTANCE = 0.1
MAX_CIVILIZATION_SIZE = 20.0
MIN_CIVILIZATION_SIZE = 0.5
MAX_POSITION = 10.0
MIN_POSITION = 0.0

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
MAX_TIMESTEP = 1.0
default_dt = 1.0
adaptive_timestep = default_dt


# Safe mathematical operations
def safe_div(x, y, default=0.0):
    """Safe division with check for division by zero."""
    if abs(y) < 1e-10:
        return default
    return x / y


def safe_exp(x, max_result=1e10):
    """Apply exponential function with bounds to prevent overflow."""
    # Limit the exponent to avoid overflow
    x = np.clip(x, -50.0, 50.0)
    return np.clip(np.exp(x), 0.0, max_result)


def safe_sqrt(x, default=0.0):
    """Safe square root with check for negative values."""
    if x <= 0:
        return default
    return np.sqrt(x)


def safe_distance(p1, p2, min_dist=MIN_DISTANCE):
    """Calculate distance between points with minimum bound to prevent division by zero."""
    try:
        dist = np.linalg.norm(p1 - p2)
        return max(dist, min_dist)
    except Exception:
        return min_dist


# Simulation parameters
timesteps = 150  # Reduced for faster execution
dt = 1.0
initial_num_civilizations = 5

# Initialize event log
event_log = []

# Stability tracking
stability_issues = 0

print("Initializing civilizations...")
# Initialize arrays for all civilizations
civilizations = initialize_civilizations(initial_num_civilizations)

# Fix: Convert ages to float type to avoid casting error
civilizations["ages"] = civilizations["ages"].astype(float)

# Ensure positions are within bounds
civilizations["positions"] = np.clip(civilizations["positions"], MIN_POSITION, MAX_POSITION)

# Ensure sizes are within bounds
civilizations["sizes"] = np.clip(civilizations["sizes"], MIN_CIVILIZATION_SIZE, MAX_CIVILIZATION_SIZE)

# Initialize knowledge arrays with bounds
knowledge_array = np.clip(1.0 + 2.0 * np.random.rand(initial_num_civilizations), MIN_VALUE, MAX_KNOWLEDGE)
suppression_array = np.clip(3.0 + 3.0 * np.random.rand(initial_num_civilizations), MIN_VALUE, MAX_SUPPRESSION)
intelligence_array = np.clip(5.0 + 5.0 * np.random.rand(initial_num_civilizations), MIN_VALUE, MAX_INTELLIGENCE)
truth_array = np.clip(1.0 + 1.0 * np.random.rand(initial_num_civilizations), MIN_VALUE, MAX_TRUTH)
influence_array = np.clip(civilizations["influence"].copy(), MIN_VALUE, MAX_INFLUENCE)
resources_array = np.clip(civilizations["resources"].copy(), MIN_VALUE, MAX_RESOURCES)

# If using dimensional analysis, set up dimensional arrays
if use_dimensional_analysis:
    print("Using dimensional analysis for multi-civilization simulation")

    # Create dimensional values for initial state
    knowledge_dim_array = [DimensionalValue(k, Dimension.KNOWLEDGE) for k in knowledge_array]
    suppression_dim_array = [DimensionalValue(s, Dimension.SUPPRESSION) for s in suppression_array]
    intelligence_dim_array = [DimensionalValue(i, Dimension.INTELLIGENCE) for i in intelligence_array]
    truth_dim_array = [DimensionalValue(t, Dimension.TRUTH) for t in truth_array]
    influence_dim_array = [DimensionalValue(inf, Dimension.INFLUENCE) for inf in influence_array]
    resources_dim_array = [DimensionalValue(r, Dimension.RESOURCES) for r in resources_array]

    # Create dimensional constant
    R = DimensionalValue(2.0, Dimension.RESISTANCE)  # Resistance constant


    # Define dimensional functions for multi-civilization context

    @check_dimensional_consistency
    def multi_civ_intelligence_growth(K_dim, W_dim, R_dim, S_dim, N_factor):
        """Dimensional intelligence growth for multi-civilization context."""
        if K_dim.dimension != Dimension.KNOWLEDGE:
            raise ValueError(f"Expected KNOWLEDGE dimension, got {K_dim.dimension}")
        if W_dim.dimension != Dimension.WISDOM:
            raise ValueError(f"Expected WISDOM dimension, got {W_dim.dimension}")
        if R_dim.dimension != Dimension.RESISTANCE:
            raise ValueError(f"Expected RESISTANCE dimension, got {R_dim.dimension}")
        if S_dim.dimension != Dimension.SUPPRESSION:
            raise ValueError(f"Expected SUPPRESSION dimension, got {S_dim.dimension}")

        # Perform calculation with dimension tracking
        growth_term = (K_dim.value * W_dim.value) / (1.0 + K_dim.value / 100.0)
        result = growth_term - R_dim.value - S_dim.value + N_factor

        return DimensionalValue(result, Dimension.INTELLIGENCE)


    @check_dimensional_consistency
    def multi_civ_decision_probability(K_dim, S_dim):
        """Calculate decision probability with dimensional values."""
        if K_dim.dimension != Dimension.KNOWLEDGE:
            raise ValueError(f"Expected KNOWLEDGE dimension, got {K_dim.dimension}")
        if S_dim.dimension != Dimension.SUPPRESSION:
            raise ValueError(f"Expected SUPPRESSION dimension, got {S_dim.dimension}")

        # Calculate the raw probability
        raw_input = 0.5 * K_dim.value - 0.3 * S_dim.value
        raw_input = np.clip(raw_input, -10, 10)
        probability = 1 / (1 + safe_exp(-raw_input))

        return DimensionalValue(probability, Dimension.PROBABILITY)

# Arrays to store data for all timesteps
# These are lists since the number of civilizations can change
time_history = []
civilization_count_history = []
knowledge_history = []
suppression_history = []
intelligence_history = []
truth_history = []
influence_history = []
resources_history = []
position_history = []
size_history = []
event_history = []
age_history = []
beyond_horizon_history = []
stability_history = []  # Track stability issues over time
timestep_history = []  # Track adaptive timestep changes

# Additional arrays for dimensional values if using dimensional analysis
if use_dimensional_analysis:
    knowledge_dim_history = []
    suppression_dim_history = []
    intelligence_dim_history = []
    truth_dim_history = []
    influence_dim_history = []
    resources_dim_history = []

print("Starting simulation...")

# Main simulation loop
for t in range(timesteps):
    current_num_civilizations = len(knowledge_array)

    # Skip timestep if no civilizations exist
    if current_num_civilizations == 0:
        # Record empty state
        time_history.append(t)
        civilization_count_history.append(0)
        knowledge_history.append([])
        suppression_history.append([])
        intelligence_history.append([])
        truth_history.append([])
        influence_history.append([])
        resources_history.append([])
        position_history.append([])
        size_history.append([])
        event_history.append([])
        age_history.append([])
        beyond_horizon_history.append([])
        stability_history.append(stability_issues)
        timestep_history.append(adaptive_timestep)

        # Store empty dimensional arrays if using dimensional analysis
        if use_dimensional_analysis:
            knowledge_dim_history.append([])
            suppression_dim_history.append([])
            intelligence_dim_history.append([])
            truth_dim_history.append([])
            influence_dim_history.append([])
            resources_dim_history.append([])

        continue

    # Calculate adaptive timestep if enabled and civilization count changes slowly
    # (abrupt changes in civilization count might need smaller timesteps)
    if enable_adaptive_timestep and t > 0:
        # Calculate maximum rate of change
        if len(knowledge_history[-1]) > 0 and len(knowledge_array) > 0:
            # If civilization counts match, calculate max change in knowledge
            if len(knowledge_history[-1]) == len(knowledge_array):
                max_k_change = np.max(np.abs(np.array(knowledge_array) - np.array(knowledge_history[-1])))
                # Adjust timestep based on rate of change
                if max_k_change > 2.0:
                    adaptive_timestep = max(MIN_TIMESTEP, adaptive_timestep * 0.8)
                elif max_k_change < 0.5:
                    adaptive_timestep = min(MAX_TIMESTEP, adaptive_timestep * 1.2)
            else:
                # If civilization count changed, use a smaller timestep for stability
                adaptive_timestep = max(MIN_TIMESTEP, adaptive_timestep * 0.5)

        # Use adaptive timestep
        current_dt = adaptive_timestep
    else:
        current_dt = dt

    # Track timestep
    timestep_history.append(current_dt)

    # Occasionally output progress
    if t % 50 == 0:
        print(
            f"Timestep {t}: {current_num_civilizations} civilizations. Current dt: {current_dt:.4f}, Stability issues: {stability_issues}")

    # Increment civilization ages with bounds
    civilizations["ages"] = np.clip(civilizations["ages"] + current_dt, 0, 1000)

    # Calculate civilization lifecycle phases
    lifecycle_intensities = np.zeros(current_num_civilizations)
    lifecycle_phases = np.zeros(current_num_civilizations, dtype=int)

    # Define consistent phase thresholds for all civilizations
    phase_thresholds = np.array([50, 100, 200, 300, 350])
    phase_intensities = np.array([0.5, 1.0, 1.5, 0.8, 1.2, 0.6])

    # Process each civilization's internal dynamics
    for i in range(current_num_civilizations):
        try:
            # Calculate lifecycle phase with safety
            intensity, phase = civilization_lifecycle_phase(
                civilizations["ages"][i], 1.0, phase_thresholds, phase_intensities
            )
            lifecycle_intensities[i] = np.clip(intensity, 0.1, 2.0)  # Bound intensity
            lifecycle_phases[i] = phase

            if use_dimensional_analysis:
                # Get dimensional values
                K_dim = knowledge_dim_array[i]
                S_dim = suppression_dim_array[i]
                T_dim = truth_dim_array[i]

                # Calculate wisdom with dimensional analysis
                W_dim = wisdom_field_with_dimensions(
                    1.0, 0.1, S_dim, R, K_dim)

                # Update knowledge with civilization's innovation rate
                # Bound innovation rates
                innovation_rate = np.clip(civilizations["innovation_rates"][i], 0.01, 10.0)

                # We don't have a dimensional version of knowledge growth yet,
                # so use the regular version with raw values
                knowledge_base_growth = np.clip(knowledge_array[i] * 0.05 * innovation_rate, -5.0, 5.0)
            else:
                # Calculate wisdom field with bounds (original version)
                W = wisdom_field(
                    1.0,
                    0.1,
                    np.clip(suppression_array[i], MIN_VALUE, MAX_SUPPRESSION),
                    2.0,
                    np.clip(knowledge_array[i], MIN_VALUE, MAX_KNOWLEDGE)
                )

                # Update knowledge with civilization's innovation rate
                # Bound innovation rates
                innovation_rate = np.clip(civilizations["innovation_rates"][i], 0.01, 10.0)
                knowledge_base_growth = np.clip(knowledge_array[i] * 0.05 * innovation_rate, -5.0, 5.0)

            # Add inflation effect for rapidly growing civilizations with safety
            inflation_multiplier = 1.0
            if truth_array[i] > 10 and phase == 2:  # High truth in peak phase
                try:
                    inflation_multiplier, _ = knowledge_inflation(
                        np.clip(knowledge_array[i], MIN_VALUE, MAX_KNOWLEDGE),
                        np.clip(truth_array[i], MIN_VALUE, MAX_TRUTH),
                        inflation_threshold=10,
                        expansion_rate=1.5,
                        duration=10
                    )
                    # Bound inflation multiplier
                    inflation_multiplier = np.clip(inflation_multiplier, 1.0, 5.0)
                except Exception as e:
                    print(f"Warning: Error in inflation calculation for civilization {i}: {e}")
                    inflation_multiplier = 1.0
                    stability_issues += 1

            # Update knowledge with bounds
            knowledge_increment = knowledge_base_growth * inflation_multiplier * current_dt

            # Check for stability
            if circuit_breaker.check_value_stability(knowledge_increment):
                knowledge_increment = np.clip(knowledge_increment, -2.0, 2.0)
                stability_issues += 1

            # Update knowledge
            knowledge_array[i] = np.clip(
                knowledge_array[i] + knowledge_increment,
                MIN_VALUE,
                MAX_KNOWLEDGE
            )

            # Update dimensional knowledge if using dimensional analysis
            if use_dimensional_analysis:
                knowledge_dim_array[i] = DimensionalValue(knowledge_array[i], Dimension.KNOWLEDGE)

            # Update suppression based on internal dynamics with bounds
            if use_dimensional_analysis:
                # Use dimensional version for suppression feedback
                suppression_resistance = np.clip(civilizations["suppression_resistance"][i], 0.1, 10.0)

                suppression_change_dim = suppression_feedback_with_dimensions(
                    0.1,
                    S_dim,
                    0.05 * suppression_resistance,
                    knowledge_dim_array[i]  # Use updated knowledge
                )
                suppression_change = suppression_change_dim.value
            else:
                # Civilizations with high suppression resistance experience less suppression
                suppression_resistance = np.clip(civilizations["suppression_resistance"][i], 0.1, 10.0)
                suppression_change = suppression_feedback(
                    0.1,
                    np.clip(suppression_array[i], MIN_VALUE, MAX_SUPPRESSION),
                    0.05 * suppression_resistance,
                    np.clip(knowledge_array[i], MIN_VALUE, MAX_KNOWLEDGE)
                )

            # Check for stability
            if circuit_breaker.check_value_stability(suppression_change):
                suppression_change = np.clip(suppression_change, -2.0, 2.0)
                stability_issues += 1

            # Update suppression
            suppression_array[i] = np.clip(
                suppression_array[i] + suppression_change * current_dt,
                MIN_VALUE,
                MAX_SUPPRESSION
            )

            # Update dimensional suppression if using dimensional analysis
            if use_dimensional_analysis:
                suppression_dim_array[i] = DimensionalValue(suppression_array[i], Dimension.SUPPRESSION)

            # Ensure minimum suppression
            suppression_array[i] = max(0.5, suppression_array[i])
            if use_dimensional_analysis:
                suppression_dim_array[i] = DimensionalValue(suppression_array[i], Dimension.SUPPRESSION)

            # Update intelligence with bounds
            if use_dimensional_analysis:
                # Use dimensional version
                intelligence_change_dim = multi_civ_intelligence_growth(
                    knowledge_dim_array[i],  # Updated knowledge
                    W_dim,
                    R,
                    suppression_dim_array[i],  # Updated suppression
                    1.5  # Network effect
                )
                intelligence_change = intelligence_change_dim.value
            else:
                # Use original version
                intelligence_change = intelligence_growth(
                    np.clip(knowledge_array[i], MIN_VALUE, MAX_KNOWLEDGE),
                    W,
                    2.0,
                    np.clip(suppression_array[i], MIN_VALUE, MAX_SUPPRESSION),
                    1.5
                )

            # Check for stability
            if circuit_breaker.check_value_stability(intelligence_change):
                intelligence_change = np.clip(intelligence_change, -2.0, 2.0)
                stability_issues += 1

            # Update intelligence
            intelligence_array[i] = np.clip(
                intelligence_array[i] + intelligence_change * lifecycle_intensities[i] * current_dt,
                MIN_VALUE,
                MAX_INTELLIGENCE
            )

            # Update dimensional intelligence if using dimensional analysis
            if use_dimensional_analysis:
                intelligence_dim_array[i] = DimensionalValue(intelligence_array[i], Dimension.INTELLIGENCE)

            # Update truth adoption with bounds
            if use_dimensional_analysis:
                # Use dimensional version
                truth_change_dim = truth_adoption_with_dimensions(
                    T_dim,
                    0.5,
                    40.0
                )
                truth_change = truth_change_dim.value
            else:
                # Use original version
                truth_change = truth_adoption(
                    np.clip(truth_array[i], MIN_VALUE, MAX_TRUTH),
                    0.5,
                    40.0
                )

            # Check for stability
            if circuit_breaker.check_value_stability(truth_change):
                truth_change = np.clip(truth_change, -2.0, 2.0)
                stability_issues += 1

            # Update truth
            truth_array[i] = np.clip(
                truth_array[i] + truth_change * current_dt,
                MIN_VALUE,
                MAX_TRUTH
            )

            # Update dimensional truth if using dimensional analysis
            if use_dimensional_analysis:
                truth_dim_array[i] = DimensionalValue(truth_array[i], Dimension.TRUTH)

        except Exception as e:
            print(f"Warning: Error processing internal dynamics for civilization {i}: {e}")
            stability_issues += 1
            # If error occurs, keep previous values to maintain stability
            if i < len(lifecycle_intensities) and i < len(lifecycle_phases):
                lifecycle_intensities[i] = 1.0  # Fallback to neutral intensity
                lifecycle_phases[i] = 2  # Fallback to peak phase

    # Calculate event horizons for each civilization with safety
    beyond_horizon = np.zeros(current_num_civilizations, dtype=bool)
    for i in range(current_num_civilizations):
        try:
            _, is_beyond = suppression_event_horizon(
                np.clip(suppression_array[i], 0.1, MAX_SUPPRESSION),  # Ensure non-zero
                np.clip(knowledge_array[i], 0.1, MAX_KNOWLEDGE),  # Ensure non-zero
                critical_constant=1.5
            )
            beyond_horizon[i] = is_beyond
        except Exception as e:
            print(f"Warning: Error in event horizon calculation for civilization {i}: {e}")
            beyond_horizon[i] = False  # Safe default
            stability_issues += 1

    # Process all inter-civilization interactions with safety
    try:
        (civilizations, knowledge_array, suppression_array,
         influence_array, resources_array, events) = process_all_civilization_interactions(
            civilizations, knowledge_array, suppression_array,
            influence_array, resources_array, current_dt,
            # Add stability parameters
            min_distance=MIN_DISTANCE,
            max_change=5.0,
            max_knowledge=MAX_KNOWLEDGE,
            max_suppression=MAX_SUPPRESSION,
            max_influence=MAX_INFLUENCE,
            max_resources=MAX_RESOURCES,
            min_division=0.1  # Minimum for division operations
        )

        # Ensure all values are within bounds after interactions
        knowledge_array = np.clip(knowledge_array, MIN_VALUE, MAX_KNOWLEDGE)
        suppression_array = np.clip(suppression_array, MIN_VALUE, MAX_SUPPRESSION)
        influence_array = np.clip(influence_array, MIN_VALUE, MAX_INFLUENCE)
        resources_array = np.clip(resources_array, MIN_VALUE, MAX_RESOURCES)
        civilizations["positions"] = np.clip(civilizations["positions"], MIN_POSITION, MAX_POSITION)
        civilizations["sizes"] = np.clip(civilizations["sizes"], MIN_CIVILIZATION_SIZE, MAX_CIVILIZATION_SIZE)

        # Update dimensional arrays if using dimensional analysis
        if use_dimensional_analysis:
            # Create new dimensional arrays since the number of civilizations may have changed
            knowledge_dim_array = [DimensionalValue(k, Dimension.KNOWLEDGE) for k in knowledge_array]
            suppression_dim_array = [DimensionalValue(s, Dimension.SUPPRESSION) for s in suppression_array]
            intelligence_dim_array = [DimensionalValue(i, Dimension.INTELLIGENCE) for i in intelligence_array]
            truth_dim_array = [DimensionalValue(t, Dimension.TRUTH) for t in truth_array]
            influence_dim_array = [DimensionalValue(inf, Dimension.INFLUENCE) for inf in influence_array]
            resources_dim_array = [DimensionalValue(r, Dimension.RESOURCES) for r in resources_array]

    except Exception as e:
        print(f"Warning: Error in civilization interactions at timestep {t}: {e}")
        # If interaction processing fails, keep the civilizations unchanged
        events = []
        stability_issues += 1

    # Record events with timestamp
    for event in events:
        event["time"] = t
        event_log.append(event)

    # Store current state with safety for empty arrays
    time_history.append(t)
    civilization_count_history.append(len(knowledge_array))
    knowledge_history.append(knowledge_array.copy() if len(knowledge_array) > 0 else np.array([]))
    suppression_history.append(suppression_array.copy() if len(suppression_array) > 0 else np.array([]))
    intelligence_history.append(intelligence_array.copy() if len(intelligence_array) > 0 else np.array([]))
    truth_history.append(truth_array.copy() if len(truth_array) > 0 else np.array([]))
    influence_history.append(influence_array.copy() if len(influence_array) > 0 else np.array([]))
    resources_history.append(resources_array.copy() if len(resources_array) > 0 else np.array([]))

    # Store dimensional arrays if using dimensional analysis
    if use_dimensional_analysis:
        knowledge_dim_history.append(knowledge_dim_array.copy() if len(knowledge_dim_array) > 0 else [])
        suppression_dim_history.append(suppression_dim_array.copy() if len(suppression_dim_array) > 0 else [])
        intelligence_dim_history.append(intelligence_dim_array.copy() if len(intelligence_dim_array) > 0 else [])
        truth_dim_history.append(truth_dim_array.copy() if len(truth_dim_array) > 0 else [])
        influence_dim_history.append(influence_dim_array.copy() if len(influence_dim_array) > 0 else [])
        resources_dim_history.append(resources_dim_array.copy() if len(resources_dim_array) > 0 else [])

    # Ensure position and size arrays exist before copying
    if hasattr(civilizations, "positions") and len(civilizations["positions"]) > 0:
        position_history.append(civilizations["positions"].copy())
    else:
        position_history.append(np.array([]))

    if hasattr(civilizations, "sizes") and len(civilizations["sizes"]) > 0:
        size_history.append(civilizations["sizes"].copy())
    else:
        size_history.append(np.array([]))

    event_history.append(events)

    if hasattr(civilizations, "ages") and len(civilizations["ages"]) > 0:
        age_history.append(civilizations["ages"].copy())
    else:
        age_history.append(np.array([]))

    beyond_horizon_history.append(beyond_horizon.copy() if len(beyond_horizon) > 0 else np.array([]))
    stability_history.append(stability_issues)

print("Simulation completed.")
print(f"Final number of civilizations: {len(knowledge_array)}")
print(f"Total number of events: {len(event_log)}")
print(f"Total stability issues: {stability_issues}")

# Dimensional consistency check if enabled
if use_dimensional_analysis:
    try:
        # Define the dimensional functions to check
        dimensional_equations = {
            'multi_civ_intelligence_growth': multi_civ_intelligence_growth,
            'multi_civ_decision_probability': multi_civ_decision_probability,
            'wisdom_field_with_dimensions': wisdom_field_with_dimensions,
            'truth_adoption_with_dimensions': truth_adoption_with_dimensions,
            'suppression_feedback_with_dimensions': suppression_feedback_with_dimensions
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
        consistency_df.to_csv(data_dir / "multi_civilization_dimensional_consistency.csv", index=False)
        print(
            f"Dimensional consistency results saved to: {data_dir / 'multi_civilization_dimensional_consistency.csv'}")
    except Exception as e:
        print(f"Error during dimensional consistency check: {e}")

print("Preparing visualization...")

# Replace any NaN or inf values that might have slipped through
for i in range(len(knowledge_history)):
    if len(knowledge_history[i]) > 0:
        knowledge_history[i] = np.nan_to_num(knowledge_history[i], nan=1.0, posinf=MAX_KNOWLEDGE, neginf=MIN_VALUE)
    if len(suppression_history[i]) > 0:
        suppression_history[i] = np.nan_to_num(suppression_history[i], nan=1.0, posinf=MAX_SUPPRESSION,
                                               neginf=MIN_VALUE)
    if len(intelligence_history[i]) > 0:
        intelligence_history[i] = np.nan_to_num(intelligence_history[i], nan=1.0, posinf=MAX_INTELLIGENCE,
                                                neginf=MIN_VALUE)
    if len(truth_history[i]) > 0:
        truth_history[i] = np.nan_to_num(truth_history[i], nan=1.0, posinf=MAX_TRUTH, neginf=MIN_VALUE)


# Create a function to get statistical data across all civilizations
def get_civilization_stats(history_arrays):
    """Calculate statistics for civilization metrics over time."""
    stats = {}

    for name, history in history_arrays.items():
        stats[f"{name}_count"] = []
        stats[f"{name}_mean"] = []
        stats[f"{name}_max"] = []
        stats[f"{name}_min"] = []
        stats[f"{name}_total"] = []

        for t in range(len(history)):
            if len(history[t]) > 0:
                stats[f"{name}_count"].append(len(history[t]))
                stats[f"{name}_mean"].append(np.mean(history[t]))
                stats[f"{name}_max"].append(np.max(history[t]))
                stats[f"{name}_min"].append(np.min(history[t]))
                stats[f"{name}_total"].append(np.sum(history[t]))
            else:
                stats[f"{name}_count"].append(0)
                stats[f"{name}_mean"].append(np.nan)
                stats[f"{name}_max"].append(np.nan)
                stats[f"{name}_min"].append(np.nan)
                stats[f"{name}_total"].append(0)

    return stats


# Prepare data for statistical analysis
history_arrays = {
    "knowledge": knowledge_history,
    "suppression": suppression_history,
    "intelligence": intelligence_history,
    "truth": truth_history,
    "influence": influence_history,
    "resources": resources_history,
    "size": size_history
}

# Calculate statistics
stats = get_civilization_stats(history_arrays)

# Create a figure for statistical plots
plt.figure(figsize=(15, 14))  # Increased height for additional subplot
gs = gridspec.GridSpec(4, 2)  # Added one more row for stability metrics

# 1. Civilization count over time
ax1 = plt.subplot(gs[0, 0])
ax1.plot(time_history, civilization_count_history, 'b-', linewidth=2)
ax1.set_title('Civilization Count Over Time')
ax1.set_xlabel('Time Steps')
ax1.set_ylabel('Number of Civilizations')
ax1.grid(True)

# 2. Average Knowledge and Suppression
ax2 = plt.subplot(gs[0, 1])
ax2.plot(time_history, stats["knowledge_mean"], 'g-', linewidth=2, label='Avg Knowledge')
ax2.plot(time_history, stats["suppression_mean"], 'r-', linewidth=2, label='Avg Suppression')
ax2.set_title('Average Knowledge and Suppression')
ax2.set_xlabel('Time Steps')
ax2.set_ylabel('Level')
ax2.legend()
ax2.grid(True)

# 3. Max and Min Knowledge
ax3 = plt.subplot(gs[1, 0])
ax3.plot(time_history, stats["knowledge_max"], 'g-', linewidth=2, label='Max Knowledge')
ax3.plot(time_history, stats["knowledge_min"], 'g--', linewidth=2, label='Min Knowledge')
ax3.fill_between(time_history, stats["knowledge_min"], stats["knowledge_max"], color='g', alpha=0.2)
ax3.set_title('Knowledge Range Across Civilizations')
ax3.set_xlabel('Time Steps')
ax3.set_ylabel('Knowledge Level')
ax3.legend()
ax3.grid(True)

# 4. Average Intelligence and Truth
ax4 = plt.subplot(gs[1, 1])
ax4.plot(time_history, stats["intelligence_mean"], 'b-', linewidth=2, label='Avg Intelligence')
ax4.plot(time_history, stats["truth_mean"], 'c-', linewidth=2, label='Avg Truth')
ax4.set_title('Average Intelligence and Truth')
ax4.set_xlabel('Time Steps')
ax4.set_ylabel('Level')
ax4.legend()
ax4.grid(True)

# 5. Total Resources and Influence
ax5 = plt.subplot(gs[2, 0])
ax5.plot(time_history, stats["resources_total"], 'y-', linewidth=2, label='Total Resources')
ax5.set_title('Total Resources Across All Civilizations')
ax5.set_xlabel('Time Steps')
ax5.set_ylabel('Total Resources')
ax5.grid(True)

# 6. Event counts
ax6 = plt.subplot(gs[2, 1])
# Count events by type
event_counts = {"collision": 0, "merger": 0, "collapse": 0, "spawn": 0, "new_civilization": 0}
event_times = {"collision": [], "merger": [], "collapse": [], "spawn": [], "new_civilization": []}

for event in event_log:
    event_type = event["type"]
    if event_type in event_counts:
        event_counts[event_type] += 1
        event_times[event_type].append(event["time"])

# Create bar chart of event counts
event_types = list(event_counts.keys())
event_values = [event_counts[et] for et in event_types]
ax6.bar(event_types, event_values, color=['red', 'purple', 'black', 'green', 'blue'])
ax6.set_title('Event Counts by Type')
ax6.set_xlabel('Event Type')
ax6.set_ylabel('Count')
ax6.grid(True, axis='y')

# 7. Stability Metrics (New)
ax7 = plt.subplot(gs[3, :])
ax7.plot(time_history, stability_history, 'r-', linewidth=2, label='Cumulative Stability Issues')
ax7.plot(time_history, timestep_history, 'b--', linewidth=2, label='Adaptive Timestep')

# Mark major events on the stability plot
for event_type, times in event_times.items():
    if times:  # If there are events of this type
        # Use different markers for different event types
        marker = 'o' if event_type == 'collision' else '^' if event_type == 'merger' else 's'
        color = 'red' if event_type == 'collision' else 'purple' if event_type == 'merger' else 'green'

        # Plot small markers at event times
        for event_time in times:
            # Find the stability value at this time
            stability_value = stability_history[time_history.index(event_time)] if event_time in time_history else 0
            ax7.plot(event_time, stability_value, marker=marker, color=color, markersize=6, alpha=0.7)

ax7.set_title('Numerical Stability Metrics')
ax7.set_xlabel('Time Steps')
ax7.set_ylabel('Value')
ax7.legend()
ax7.grid(True)

# Add secondary y-axis for timestep
ax7_twin = ax7.twinx()
ax7_twin.set_ylabel('Timestep Size (dt)', color='blue')
ax7_twin.tick_params(axis='y', labelcolor='blue')
ax7_twin.set_ylim(0, max(timestep_history) * 1.2)

# Add annotation about dimensional analysis if used
if use_dimensional_analysis:
    plt.figtext(0.5, 0.01, "Using dimensional analysis", ha="center", fontsize=10,
                bbox={"facecolor": "lightgray", "alpha": 0.5, "pad": 5})

plt.tight_layout()
stats_plot_file = plots_dir / "multi_civilization_statistics.png"
plt.savefig(str(stats_plot_file))
print(f"✅ Statistics plot saved at: {stats_plot_file}")

# Create a spatial visualization of civilization positions and sizes
plt.figure(figsize=(12, 12))

# Select time points for snapshots
snapshot_times = [0, int(timesteps * 0.25), int(timesteps * 0.5), int(timesteps * 0.75), timesteps - 1]
snapshot_times = [t for t in snapshot_times if t < len(time_history)]

# Create subplots for each snapshot
for i, t in enumerate(snapshot_times):
    ax = plt.subplot(2, 3, i + 1)

    if len(position_history[t]) > 0:
        # Get the data for this time point
        positions = position_history[t]
        sizes = size_history[t]
        knowledge_values = knowledge_history[t]
        suppression_values = suppression_history[t]
        beyond_horizon_values = beyond_horizon_history[t]

        # Create normalized values for color mapping
        if len(knowledge_values) > 0:
            norm_knowledge = (knowledge_values - np.min(knowledge_values)) / max(1e-10,
                                                                                 np.max(knowledge_values) - np.min(
                                                                                     knowledge_values))
        else:
            norm_knowledge = np.array([])

        # Plot each civilization
        for j in range(len(positions)):
            # Circle size based on civilization size
            marker_size = sizes[j] * 100

            # Color based on knowledge (green) to suppression (red) ratio
            if j < len(beyond_horizon_values) and beyond_horizon_values[j]:
                # Black for civilizations beyond event horizon
                color = 'black'
                edge_color = 'white'
            else:
                # Otherwise, use knowledge level for color (green = high, red = low)
                if j < len(norm_knowledge):
                    color_val = norm_knowledge[j]
                    color = plt.cm.RdYlGn(color_val)
                    edge_color = 'black'
                else:
                    color = 'gray'
                    edge_color = 'black'

            # Plot the civilization
            ax.scatter(positions[j, 0], positions[j, 1], s=marker_size,
                       c=[color], edgecolors=edge_color, alpha=0.7)

    # Set plot limits consistent across all snapshots
    ax.set_xlim(-2, 12)
    ax.set_ylim(-2, 12)
    ax.set_title(f'Time: {time_history[t]}')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.grid(True)

# Add a legend in the last subplot space
ax_legend = plt.subplot(2, 3, 6)
ax_legend.axis('off')

# Add legend elements
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=15, label='High Knowledge'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=15, label='Medium Knowledge'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=15, label='Low Knowledge'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=15, label='Beyond Event Horizon')
]

# Add civilization size examples
for size, label in [(5, 'Small'), (10, 'Medium'), (15, 'Large')]:
    legend_elements.append(
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markersize=size / 2, label=f'{label} Civilization')
    )

ax_legend.legend(handles=legend_elements, loc='center', fontsize=12)
ax_legend.text(0.5, 0.1, 'Civilization Attributes', horizontalalignment='center',
               fontsize=14, fontweight='bold', transform=ax_legend.transAxes)

# Add annotation about dimensional analysis if used
if use_dimensional_analysis:
    ax_legend.text(0.5, 0.05, "Using dimensional analysis", ha="center", fontsize=10,
                   transform=ax_legend.transAxes, bbox={"facecolor": "lightgray", "alpha": 0.5, "pad": 5})

plt.tight_layout()
positions_plot_file = plots_dir / "multi_civilization_positions.png"
plt.savefig(str(positions_plot_file))
print(f"✅ Positions plot saved at: {positions_plot_file}")

# Skip animation for performance
print("Skipping animation for performance. Uncomment the animation code if needed.")

# Save event log to CSV
event_df = pd.DataFrame(event_log)
event_csv_file = data_dir / "multi_civilization_events.csv"
if not event_df.empty:
    event_df.to_csv(event_csv_file, index=False)
    print(f"✅ Event log saved at: {event_csv_file}")
else:
    print("No events to save.")

# Prepare statistics dataframe
stats_data = {
    'Time': time_history,
    'Civilization_Count': civilization_count_history,
    'Stability_Issues': stability_history,
    'Timestep': timestep_history
}

# Add all calculated statistics
for key, values in stats.items():
    stats_data[key] = values

# Create and save statistics dataframe
stats_df = pd.DataFrame(stats_data)
stats_csv_file = data_dir / "multi_civilization_statistics.csv"
stats_df.to_csv(stats_csv_file, index=False)
print(f"✅ Statistics data saved at: {stats_csv_file}")

# Save stability metrics
stability_metrics = {
    'Total_Stability_Issues': stability_issues,
    'Circuit_Breaker_Triggers': circuit_breaker.trigger_count,
    'Max_Knowledge': np.max([np.max(k) if len(k) > 0 else 0 for k in knowledge_history]),
    'Max_Suppression': np.max([np.max(s) if len(s) > 0 else 0 for s in suppression_history]),
    'Max_Intelligence': np.max([np.max(i) if len(i) > 0 else 0 for i in intelligence_history]),
    'Max_Truth': np.max([np.max(t) if len(t) > 0 else 0 for t in truth_history]),
    'Initial_Timestep': dt,
    'Final_Timestep': timestep_history[-1] if timestep_history else dt,
    'Min_Timestep_Used': min(timestep_history) if timestep_history else dt,
    'Total_Collisions': event_counts.get('collision', 0),
    'Total_Mergers': event_counts.get('merger', 0),
    'Total_Collapses': event_counts.get('collapse', 0),
    'Total_Spawns': event_counts.get('spawn', 0),
    'Total_New_Civilizations': event_counts.get('new_civilization', 0),
    'Used_Dimensional_Analysis': use_dimensional_analysis
}

stability_df = pd.DataFrame([stability_metrics])
stability_df.to_csv(data_dir / "multi_civilization_stability.csv", index=False)
print(f"✅ Stability metrics saved at: {data_dir / 'multi_civilization_stability.csv'}")

print("All visualizations and data exports completed.")
plt.show()