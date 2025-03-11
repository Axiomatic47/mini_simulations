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

# Simulation parameters
timesteps = 150  # Reduced for faster execution
dt = 1.0
initial_num_civilizations = 5

# Initialize event log
event_log = []

print("Initializing civilizations...")
# Initialize arrays for all civilizations
civilizations = initialize_civilizations(initial_num_civilizations)

# Fix: Convert ages to float type to avoid casting error
civilizations["ages"] = civilizations["ages"].astype(float)

# Initialize knowledge arrays
knowledge_array = 1.0 + 2.0 * np.random.rand(initial_num_civilizations)
suppression_array = 3.0 + 3.0 * np.random.rand(initial_num_civilizations)
intelligence_array = 5.0 + 5.0 * np.random.rand(initial_num_civilizations)
truth_array = 1.0 + 1.0 * np.random.rand(initial_num_civilizations)
influence_array = civilizations["influence"].copy()
resources_array = civilizations["resources"].copy()

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
        continue

    # Occasionally output progress
    if t % 50 == 0:
        print(f"Timestep {t}: {current_num_civilizations} civilizations")

    # Increment civilization ages
    civilizations["ages"] += dt

    # Calculate civilization lifecycle phases
    lifecycle_intensities = np.zeros(current_num_civilizations)
    lifecycle_phases = np.zeros(current_num_civilizations, dtype=int)

    # Define consistent phase thresholds for all civilizations
    phase_thresholds = np.array([50, 100, 200, 300, 350])
    phase_intensities = np.array([0.5, 1.0, 1.5, 0.8, 1.2, 0.6])

    # Process each civilization's internal dynamics
    for i in range(current_num_civilizations):
        # Calculate lifecycle phase
        intensity, phase = civilization_lifecycle_phase(
            civilizations["ages"][i], 1.0, phase_thresholds, phase_intensities
        )
        lifecycle_intensities[i] = intensity
        lifecycle_phases[i] = phase

        # Calculate wisdom field
        W = wisdom_field(1.0, 0.1, suppression_array[i], 2.0, knowledge_array[i])

        # Update knowledge with civilization's innovation rate
        knowledge_base_growth = knowledge_array[i] * 0.05 * civilizations["innovation_rates"][i]

        # Add inflation effect for rapidly growing civilizations
        inflation_multiplier = 1.0
        if truth_array[i] > 10 and phase == 2:  # High truth in peak phase
            inflation_multiplier, _ = knowledge_inflation(
                knowledge_array[i], truth_array[i], inflation_threshold=10,
                expansion_rate=1.5, duration=10
            )

        # Update knowledge
        knowledge_array[i] += knowledge_base_growth * inflation_multiplier * dt

        # Update suppression based on internal dynamics
        # Civilizations with high suppression resistance experience less suppression
        suppression_change = suppression_feedback(
            0.1, suppression_array[i],
            0.05 * civilizations["suppression_resistance"][i], knowledge_array[i]
        )
        suppression_array[i] += suppression_change * dt

        # Ensure minimum suppression
        suppression_array[i] = max(0.5, suppression_array[i])

        # Update intelligence
        intelligence_change = intelligence_growth(
            knowledge_array[i], W, 2.0, suppression_array[i], 1.5
        )
        intelligence_array[i] += intelligence_change * lifecycle_intensities[i] * dt

        # Ensure minimum intelligence
        intelligence_array[i] = max(0.1, intelligence_array[i])

        # Update truth adoption
        truth_change = truth_adoption(truth_array[i], 0.5, 40.0)
        truth_array[i] += truth_change * dt

    # Calculate event horizons for each civilization
    beyond_horizon = np.zeros(current_num_civilizations, dtype=bool)
    for i in range(current_num_civilizations):
        _, is_beyond = suppression_event_horizon(
            suppression_array[i], knowledge_array[i], critical_constant=1.5
        )
        beyond_horizon[i] = is_beyond

    # Process all inter-civilization interactions
    (civilizations, knowledge_array, suppression_array,
     influence_array, resources_array, events) = process_all_civilization_interactions(
        civilizations, knowledge_array, suppression_array,
        influence_array, resources_array, dt
    )

    # Record events with timestamp
    for event in events:
        event["time"] = t
        event_log.append(event)

    # Store current state
    time_history.append(t)
    civilization_count_history.append(len(knowledge_array))
    knowledge_history.append(knowledge_array.copy())
    suppression_history.append(suppression_array.copy())
    intelligence_history.append(intelligence_array.copy())
    truth_history.append(truth_array.copy())
    influence_history.append(influence_array.copy())
    resources_history.append(resources_array.copy())
    position_history.append(civilizations["positions"].copy())
    size_history.append(civilizations["sizes"].copy())
    event_history.append(events)
    age_history.append(civilizations["ages"].copy())
    beyond_horizon_history.append(beyond_horizon.copy())

print("Simulation completed.")
print(f"Final number of civilizations: {len(knowledge_array)}")
print(f"Total number of events: {len(event_log)}")

print("Preparing visualization...")


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
plt.figure(figsize=(15, 12))
gs = gridspec.GridSpec(3, 2)

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

plt.tight_layout()
stats_plot_file = plots_dir / "multi_civilization_statistics.png"
plt.savefig(str(stats_plot_file))
print(f"✅ Statistics plot saved at: {stats_plot_file}")

# Create a spatial visualization of civilization positions and sizes
plt.figure(figsize=(12, 10))

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
            if beyond_horizon_values[j]:
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
    'Civilization_Count': civilization_count_history
}

# Add all calculated statistics
for key, values in stats.items():
    stats_data[key] = values

# Create and save statistics dataframe
stats_df = pd.DataFrame(stats_data)
stats_csv_file = data_dir / "multi_civilization_statistics.csv"
stats_df.to_csv(stats_csv_file, index=False)
print(f"✅ Statistics data saved at: {stats_csv_file}")

print("All visualizations and data exports completed.")
plt.show()