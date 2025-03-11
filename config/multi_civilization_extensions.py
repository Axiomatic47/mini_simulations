import numpy as np
from scipy.spatial.distance import cdist


# Multi-Civilization Interaction Models

def initialize_civilizations(num_civilizations, max_age_variance=100):
    """
    Initialize multiple civilizations with different starting ages and positions.

    Parameters:
        num_civilizations (int): Number of civilizations to initialize
        max_age_variance (int): Maximum variance in starting ages

    Returns:
        dict: Dictionary containing civilization parameters
    """
    # Generate random starting ages, placing civilizations at different lifecycle phases
    starting_ages = np.random.randint(0, max_age_variance, num_civilizations)

    # Generate positions in a conceptual 2D space
    # Civilizations close to each other will interact more strongly
    positions = np.random.rand(num_civilizations, 2) * 10

    # Generate intrinsic parameters for each civilization
    innovation_rates = 0.8 + 0.4 * np.random.rand(num_civilizations)  # 0.8-1.2
    suppression_resistance = 0.7 + 0.6 * np.random.rand(num_civilizations)  # 0.7-1.3
    knowledge_retention = 0.6 + 0.4 * np.random.rand(num_civilizations)  # 0.6-1.0
    expansion_tendency = 0.5 + 1.0 * np.random.rand(num_civilizations)  # 0.5-1.5

    # Starting resources and influence levels
    resources = 10 + 10 * np.random.rand(num_civilizations)
    influence = 5 + 5 * np.random.rand(num_civilizations)

    # Create dictionary of civilization parameters
    return {
        "ages": starting_ages,
        "positions": positions,
        "innovation_rates": innovation_rates,
        "suppression_resistance": suppression_resistance,
        "knowledge_retention": knowledge_retention,
        "expansion_tendency": expansion_tendency,
        "resources": resources,
        "influence": influence,
        "velocities": np.zeros((num_civilizations, 2)),  # Initially static
        "sizes": 1.0 + np.random.rand(num_civilizations)  # Initial size/scale
    }


def calculate_distance_matrix(positions):
    """
    Calculate distances between all pairs of civilizations.

    Parameters:
        positions (array): 2D array of civilization positions

    Returns:
        array: Matrix of distances between civilizations
    """
    return cdist(positions, positions)


def calculate_interaction_strength(distance_matrix, max_interaction_distance=5.0):
    """
    Calculate interaction strength between civilizations based on distance.

    Parameters:
        distance_matrix (array): Matrix of distances between civilizations
        max_interaction_distance (float): Maximum distance for interactions

    Returns:
        array: Matrix of interaction strengths
    """
    # Create interaction strength matrix with inverse square law
    interaction_strength = 1.0 / (1.0 + distance_matrix ** 2)

    # Set diagonal (self-interaction) to 0
    np.fill_diagonal(interaction_strength, 0)

    # Zero out interactions beyond max distance
    interaction_strength[distance_matrix > max_interaction_distance] = 0

    return interaction_strength


def galactic_collision_effect(civ_i, civ_j, collision_threshold=1.0):
    """
    Model effects of a collision or close encounter between civilizations.

    Parameters:
        civ_i (dict): Parameters of the first civilization
        civ_j (dict): Parameters of the second civilization
        collision_threshold (float): Distance threshold for collision effects

    Returns:
        tuple: Knowledge transfer, suppression effect, resource exchange
    """
    # Calculate distance between civilizations
    distance = np.linalg.norm(civ_i["position"] - civ_j["position"])

    # If civilizations are very close, model collision effects
    if distance < collision_threshold:
        # Calculate relative development levels
        knowledge_ratio = civ_i["knowledge"] / max(0.1, civ_j["knowledge"])

        # Knowledge transfer (more advanced civ transfers knowledge to less advanced)
        if knowledge_ratio > 1:
            # i transfers knowledge to j
            knowledge_transfer = 0.1 * (knowledge_ratio - 1) * civ_i["knowledge"]
        else:
            # j transfers knowledge to i
            knowledge_transfer = -0.1 * (1 - knowledge_ratio) * civ_j["knowledge"]

        # Suppression effect (stronger civ may suppress weaker)
        power_ratio = civ_i["influence"] / max(0.1, civ_j["influence"])
        if power_ratio > 1.5:
            # i suppresses j
            suppression_effect = 0.05 * power_ratio * civ_i["influence"]
        elif power_ratio < 0.67:  # 1/1.5
            # j suppresses i
            suppression_effect = -0.05 * (1 / power_ratio) * civ_j["influence"]
        else:
            # No significant suppression
            suppression_effect = 0

        # Resource exchange (can be positive or negative)
        resource_differential = civ_i["resources"] - civ_j["resources"]
        resource_exchange = 0.02 * resource_differential

        return knowledge_transfer, suppression_effect, resource_exchange

    # No collision effects if distance is above threshold
    return 0, 0, 0


def knowledge_diffusion(civilizations, knowledge_array, interaction_strength, diffusion_rate=0.01):
    """
    Model knowledge diffusion between civilizations.

    Parameters:
        civilizations (dict): Civilization parameters
        knowledge_array (array): Current knowledge levels for all civilizations
        interaction_strength (array): Matrix of interaction strengths
        diffusion_rate (float): Base rate of knowledge diffusion

    Returns:
        array: Knowledge change due to diffusion
    """
    num_civilizations = len(knowledge_array)
    knowledge_change = np.zeros(num_civilizations)

    # Calculate knowledge diffusion for each civilization
    for i in range(num_civilizations):
        for j in range(num_civilizations):
            if i != j and interaction_strength[i, j] > 0:
                # Knowledge flows from higher to lower levels
                knowledge_diff = knowledge_array[j] - knowledge_array[i]
                if knowledge_diff > 0:
                    # Receiving knowledge
                    # Affected by innovation rate (how well civ can adopt external ideas)
                    knowledge_change[i] += (diffusion_rate *
                                            interaction_strength[i, j] *
                                            knowledge_diff *
                                            civilizations["innovation_rates"][i])
                else:
                    # Giving knowledge - reduced outflow based on knowledge retention
                    knowledge_change[i] += (diffusion_rate *
                                            interaction_strength[i, j] *
                                            knowledge_diff *
                                            (1 - civilizations["knowledge_retention"][i]))

    return knowledge_change


def cultural_influence(civilizations, influence_array, interaction_strength, base_influence_rate=0.02):
    """
    Model cultural and ideological influence between civilizations.

    Parameters:
        civilizations (dict): Civilization parameters
        influence_array (array): Current influence levels for all civilizations
        interaction_strength (array): Matrix of interaction strengths
        base_influence_rate (float): Base rate of influence spread

    Returns:
        array: Influence change due to cultural exchange
    """
    num_civilizations = len(influence_array)
    influence_change = np.zeros(num_civilizations)

    # Calculate influence spread for each civilization
    for i in range(num_civilizations):
        for j in range(num_civilizations):
            if i != j and interaction_strength[i, j] > 0:
                # Influence effect based on relative sizes
                size_factor = civilizations["sizes"][i] / max(0.1, civilizations["sizes"][j])

                # Calculate influence exchange
                influence_diff = influence_array[j] - influence_array[i]
                influence_direction = 1 if influence_diff > 0 else -1

                # Influence change depends on difference, size, and expansion tendency
                influence_change[i] += (base_influence_rate *
                                        interaction_strength[i, j] *
                                        influence_direction *
                                        abs(influence_diff) ** 0.5 *
                                        size_factor *
                                        civilizations["expansion_tendency"][i])

    return influence_change


def resource_competition(civilizations, resources_array, interaction_strength, competition_rate=0.01):
    """
    Model competition for resources between civilizations.

    Parameters:
        civilizations (dict): Civilization parameters
        resources_array (array): Current resource levels for all civilizations
        interaction_strength (array): Matrix of interaction strengths
        competition_rate (float): Base rate of resource competition

    Returns:
        array: Resource change due to competition
    """
    num_civilizations = len(resources_array)
    resource_change = np.zeros(num_civilizations)

    # Calculate resource competition effects
    for i in range(num_civilizations):
        for j in range(num_civilizations):
            if i != j and interaction_strength[i, j] > 0:
                # Calculate relative power (combination of knowledge and influence)
                power_i = (resources_array[i] +
                           civilizations["influence"][i] +
                           civilizations["knowledge_retention"][i] * 10)

                power_j = (resources_array[j] +
                           civilizations["influence"][j] +
                           civilizations["knowledge_retention"][j] * 10)

                # Power ratio determines resource flow
                power_ratio = power_i / max(0.1, power_j)

                # Resources flow from weaker to stronger civilizations
                if power_ratio > 1:
                    # i gains resources from j
                    flow = competition_rate * interaction_strength[i, j] * np.log(power_ratio)
                    resource_change[i] += flow
                    # This will be negative for j in its own calculation

    return resource_change


def civilization_movement(civilizations, interaction_strength, dt=1.0, attraction_factor=0.01, repulsion_threshold=1.0):
    """
    Update positions of civilizations based on attractive and repulsive forces.

    Parameters:
        civilizations (dict): Civilization parameters
        interaction_strength (array): Matrix of interaction strengths
        dt (float): Time step
        attraction_factor (float): Strength of attraction between civilizations
        repulsion_threshold (float): Distance threshold for repulsion

    Returns:
        array: Updated positions for all civilizations
    """
    num_civilizations = len(civilizations["positions"])

    # Calculate forces between civilizations
    forces = np.zeros((num_civilizations, 2))

    for i in range(num_civilizations):
        for j in range(num_civilizations):
            if i != j:
                # Calculate direction vector
                direction = civilizations["positions"][j] - civilizations["positions"][i]
                distance = np.linalg.norm(direction)

                # Normalize direction
                if distance > 0:
                    direction = direction / distance

                # Calculate attractive force based on influence and knowledge
                attraction = (attraction_factor *
                              interaction_strength[i, j] *
                              civilizations["influence"][j] *
                              civilizations["expansion_tendency"][i])

                # Calculate repulsive force if civilizations are too close
                repulsion = 0
                if distance < repulsion_threshold:
                    repulsion = -attraction_factor * (repulsion_threshold - distance) * 5

                # Combined force
                net_force = (attraction + repulsion) * direction
                forces[i] += net_force

    # Update velocities (with damping)
    civilizations["velocities"] = 0.9 * civilizations["velocities"] + forces * dt

    # Update positions
    civilizations["positions"] += civilizations["velocities"] * dt

    return civilizations["positions"]


def update_civilization_sizes(civilizations, knowledge_array, influence_array, growth_factor=0.01):
    """
    Update the sizes of civilizations based on their knowledge and influence.

    Parameters:
        civilizations (dict): Civilization parameters
        knowledge_array (array): Knowledge levels for all civilizations
        influence_array (array): Influence levels for all civilizations
        growth_factor (float): Base growth rate for civilization sizes

    Returns:
        array: Updated sizes for all civilizations
    """
    num_civilizations = len(civilizations["sizes"])

    # Calculate size changes based on knowledge and influence
    for i in range(num_civilizations):
        knowledge_effect = np.log1p(knowledge_array[i]) * growth_factor
        influence_effect = np.sqrt(influence_array[i]) * growth_factor * 0.5

        # Combined effect (knowledge has stronger impact than influence)
        size_change = knowledge_effect + influence_effect

        # Update size
        civilizations["sizes"][i] *= (1 + size_change)

        # Ensure minimum size
        civilizations["sizes"][i] = max(0.1, civilizations["sizes"][i])

    return civilizations["sizes"]


def detect_civilization_collapse(knowledge_array, suppression_array, threshold=0.1):
    """
    Detect civilizations that have collapsed due to high suppression and low knowledge.

    Parameters:
        knowledge_array (array): Knowledge levels for all civilizations
        suppression_array (array): Suppression levels for all civilizations
        threshold (float): Collapse threshold for knowledge/suppression ratio

    Returns:
        array: Boolean array indicating collapsed civilizations
    """
    # Calculate knowledge to suppression ratio
    k_s_ratio = knowledge_array / np.maximum(0.1, suppression_array)

    # Detect collapses where ratio falls below threshold
    return k_s_ratio < threshold


def detect_civilization_mergers(civilizations, distance_threshold=0.5, size_ratio_threshold=3.0):
    """
    Detect and process civilization mergers when they become very close.

    Parameters:
        civilizations (dict): Civilization parameters
        distance_threshold (float): Distance threshold for mergers
        size_ratio_threshold (float): Size ratio beyond which larger absorbs smaller

    Returns:
        list: Pairs of civilizations that should merge [[(i, j), ...]]
    """
    num_civilizations = len(civilizations["positions"])
    distance_matrix = calculate_distance_matrix(civilizations["positions"])
    mergers = []

    # Check all pairs of civilizations
    for i in range(num_civilizations):
        for j in range(i + 1, num_civilizations):
            if distance_matrix[i, j] < distance_threshold:
                # Check size ratio for absorption
                size_ratio = civilizations["sizes"][i] / max(0.1, civilizations["sizes"][j])

                if size_ratio > size_ratio_threshold:
                    # i absorbs j
                    mergers.append((i, j))
                elif 1 / size_ratio > size_ratio_threshold:
                    # j absorbs i
                    mergers.append((j, i))
                # If size ratio is moderate, no merger occurs

    return mergers


def process_civilization_merger(civilizations, knowledge_array, i, j):
    """
    Process a merger between two civilizations (i absorbs j).

    Parameters:
        civilizations (dict): Civilization parameters
        knowledge_array (array): Knowledge levels for all civilizations
        i (int): Index of absorbing civilization
        j (int): Index of absorbed civilization

    Returns:
        dict: Updated civilization parameters
        array: Updated knowledge array
    """
    # Calculate new attributes for combined civilization

    # Knowledge combines with diminishing returns
    knowledge_array[i] = knowledge_array[i] + 0.8 * knowledge_array[j]

    # Resources add linearly
    civilizations["resources"][i] += civilizations["resources"][j]

    # Influence combines with bonus
    civilizations["influence"][i] += 0.9 * civilizations["influence"][j]

    # Size increases based on absorbed civilization
    civilizations["sizes"][i] += 0.7 * civilizations["sizes"][j]

    # Weighted average of innovation rate
    civilizations["innovation_rates"][i] = (
            (civilizations["innovation_rates"][i] * knowledge_array[i] +
             civilizations["innovation_rates"][j] * knowledge_array[j]) /
            (knowledge_array[i] + knowledge_array[j])
    )

    # Weighted average of other traits
    civilizations["suppression_resistance"][i] = (
            (civilizations["suppression_resistance"][i] * civilizations["influence"][i] +
             civilizations["suppression_resistance"][j] * civilizations["influence"][j]) /
            (civilizations["influence"][i] + civilizations["influence"][j])
    )

    # Return updated parameters
    return civilizations, knowledge_array


def spawn_new_civilization(civilizations, knowledge_array, suppression_array, position, parent_idx=None):
    """
    Spawn a new civilization, either randomly or as offspring of an existing one.

    Parameters:
        civilizations (dict): Civilization parameters
        knowledge_array (array): Knowledge levels for all civilizations
        suppression_array (array): Suppression levels for all civilizations
        position (array): Starting position for new civilization
        parent_idx (int): Index of parent civilization (None for random)

    Returns:
        dict: Updated civilization parameters
        array: Updated knowledge array
        array: Updated suppression array
    """
    num_civilizations = len(knowledge_array)

    # Make deep copies to avoid modifying the originals
    civilizations = {k: v.copy() for k, v in civilizations.items()}
    knowledge_array = knowledge_array.copy()
    suppression_array = suppression_array.copy()

    # Create containers for new civilization
    for key in civilizations:
        if isinstance(civilizations[key], np.ndarray):
            # Add one more slot to each parameter array
            if civilizations[key].ndim == 1:
                # 1D arrays
                civilizations[key] = np.append(civilizations[key],
                                               np.zeros(1, dtype=civilizations[key].dtype))
            else:
                # 2D arrays (positions, velocities)
                # First add a row of zeros
                civilizations[key] = np.vstack([civilizations[key],
                                                np.zeros((1, civilizations[key].shape[1]),
                                                         dtype=civilizations[key].dtype)])

    # Extend knowledge and suppression arrays
    knowledge_array = np.append(knowledge_array, [0])
    suppression_array = np.append(suppression_array, [0])

    # Set initial position - now we can safely set it since we've reshaped the array
    civilizations["positions"][num_civilizations] = position
    civilizations["velocities"][num_civilizations] = [0, 0]

    if parent_idx is not None:
        # Spawned from parent with moderate inheritance and mutation
        civilizations["ages"][num_civilizations] = 0  # New civilization

        # Inherit with variation from parent
        mutation_factor = 0.2

        civilizations["innovation_rates"][num_civilizations] = (
                civilizations["innovation_rates"][parent_idx] * (1 + mutation_factor * (np.random.rand() - 0.5))
        )

        civilizations["suppression_resistance"][num_civilizations] = (
                civilizations["suppression_resistance"][parent_idx] * (1 + mutation_factor * (np.random.rand() - 0.5))
        )

        civilizations["knowledge_retention"][num_civilizations] = (
                civilizations["knowledge_retention"][parent_idx] * (1 + mutation_factor * (np.random.rand() - 0.5))
        )

        civilizations["expansion_tendency"][num_civilizations] = (
                civilizations["expansion_tendency"][parent_idx] * (1 + mutation_factor * (np.random.rand() - 0.5))
        )

        # Initial resources and influence transferred from parent
        resource_transfer = 0.2 * civilizations["resources"][parent_idx]
        civilizations["resources"][parent_idx] -= resource_transfer
        civilizations["resources"][num_civilizations] = resource_transfer

        influence_transfer = 0.1 * civilizations["influence"][parent_idx]
        civilizations["influence"][parent_idx] -= influence_transfer
        civilizations["influence"][num_civilizations] = influence_transfer

        # Initial knowledge transfer
        knowledge_transfer = 0.3 * knowledge_array[parent_idx]
        knowledge_array[num_civilizations] = knowledge_transfer

        # Initial size
        civilizations["sizes"][num_civilizations] = 0.5 * civilizations["sizes"][parent_idx]

    else:
        # Random new civilization
        civilizations["ages"][num_civilizations] = 0
        civilizations["innovation_rates"][num_civilizations] = 0.8 + 0.4 * np.random.rand()
        civilizations["suppression_resistance"][num_civilizations] = 0.7 + 0.6 * np.random.rand()
        civilizations["knowledge_retention"][num_civilizations] = 0.6 + 0.4 * np.random.rand()
        civilizations["expansion_tendency"][num_civilizations] = 0.5 + 1.0 * np.random.rand()
        civilizations["resources"][num_civilizations] = 5 + 5 * np.random.rand()
        civilizations["influence"][num_civilizations] = 2 + 3 * np.random.rand()
        civilizations["sizes"][num_civilizations] = 0.5 + 0.5 * np.random.rand()
        knowledge_array[num_civilizations] = 0.5 + 1.5 * np.random.rand()
        suppression_array[num_civilizations] = 1 + 2 * np.random.rand()

    return civilizations, knowledge_array, suppression_array


def remove_civilization(civilizations, knowledge_array, suppression_array, idx):
    """
    Remove a civilization from the simulation (due to collapse or merger).

    Parameters:
        civilizations (dict): Civilization parameters
        knowledge_array (array): Knowledge levels for all civilizations
        suppression_array (array): Suppression levels for all civilizations
        idx (int): Index of civilization to remove

    Returns:
        dict: Updated civilization parameters
        array: Updated knowledge array
        array: Updated suppression array
    """
    # Create mask for all civilizations except the one to remove
    num_civilizations = len(knowledge_array)
    mask = np.ones(num_civilizations, dtype=bool)
    mask[idx] = False

    # Apply mask to all arrays
    for key in civilizations:
        if isinstance(civilizations[key], np.ndarray):
            if civilizations[key].ndim == 1:
                civilizations[key] = civilizations[key][mask]
            else:
                # Handle 2D arrays (positions, velocities)
                civilizations[key] = civilizations[key][mask, :]

    # Apply mask to knowledge and suppression arrays
    knowledge_array = knowledge_array[mask]
    suppression_array = suppression_array[mask]

    return civilizations, knowledge_array, suppression_array


def process_all_civilization_interactions(civilizations, knowledge_array, suppression_array,
                                          influence_array, resources_array, dt=1.0):
    """
    Process all interactions between civilizations in a single time step.

    Parameters:
        civilizations (dict): Civilization parameters
        knowledge_array (array): Knowledge levels for all civilizations
        suppression_array (array): Suppression levels for all civilizations
        influence_array (array): Influence levels for all civilizations
        resources_array (array): Resource levels for all civilizations
        dt (float): Time step size

    Returns:
        dict: Updated civilization parameters
        array: Updated knowledge array
        array: Updated suppression array
        array: Updated influence array
        array: Updated resources array
        list: Information about key events
    """
    num_civilizations = len(knowledge_array)
    events = []  # Track significant events

    # Skip processing if only 0 or 1 civilization
    if num_civilizations <= 1:
        return (civilizations, knowledge_array, suppression_array,
                influence_array, resources_array, events)

    # Calculate distance and interaction matrices
    distance_matrix = calculate_distance_matrix(civilizations["positions"])
    interaction_strength = calculate_interaction_strength(distance_matrix)

    # Process knowledge diffusion
    knowledge_change = knowledge_diffusion(civilizations, knowledge_array,
                                           interaction_strength)

    # Process cultural influence
    influence_change = cultural_influence(civilizations, influence_array,
                                          interaction_strength)

    # Process resource competition
    resource_change = resource_competition(civilizations, resources_array,
                                           interaction_strength)

    # Detect close encounters/collisions
    for i in range(num_civilizations):
        for j in range(i + 1, num_civilizations):
            if distance_matrix[i, j] < 1.5:
                # Process collision effects
                k_transfer, s_effect, r_exchange = galactic_collision_effect(
                    {"position": civilizations["positions"][i],
                     "knowledge": knowledge_array[i],
                     "influence": influence_array[i],
                     "resources": resources_array[i]},
                    {"position": civilizations["positions"][j],
                     "knowledge": knowledge_array[j],
                     "influence": influence_array[j],
                     "resources": resources_array[j]}
                )

                # Apply collision effects
                knowledge_change[i] += k_transfer
                knowledge_change[j] -= k_transfer

                suppression_array[j] += s_effect
                suppression_array[i] -= s_effect

                resource_change[i] += r_exchange
                resource_change[j] -= r_exchange

                # Record significant collision
                if abs(k_transfer) > 0.5 or abs(s_effect) > 0.5 or abs(r_exchange) > 0.5:
                    events.append({
                        "type": "collision",
                        "civilizations": (i, j),
                        "knowledge_transfer": k_transfer,
                        "suppression_effect": s_effect,
                        "resource_exchange": r_exchange
                    })

    # Apply all changes
    knowledge_array += knowledge_change * dt
    influence_array += influence_change * dt
    resources_array += resource_change * dt

    # Ensure non-negative values
    knowledge_array = np.maximum(0, knowledge_array)
    influence_array = np.maximum(0, influence_array)
    resources_array = np.maximum(0, resources_array)

    # Update civilization positions
    civilization_movement(civilizations, interaction_strength, dt)

    # Update civilization sizes
    update_civilization_sizes(civilizations, knowledge_array, influence_array)

    # Detect collapses
    collapses = detect_civilization_collapse(knowledge_array, suppression_array)
    collapsed_indices = np.where(collapses)[0]

    # Process collapses from highest index to lowest to avoid reindexing issues
    for idx in sorted(collapsed_indices, reverse=True):
        events.append({
            "type": "collapse",
            "civilization": idx,
            "knowledge": knowledge_array[idx],
            "suppression": suppression_array[idx]
        })

        # Remove collapsed civilization
        civilizations, knowledge_array, suppression_array = remove_civilization(
            civilizations, knowledge_array, suppression_array, idx
        )

        # Update arrays after removal
        if idx < len(influence_array):
            influence_array = np.delete(influence_array, idx)
        if idx < len(resources_array):
            resources_array = np.delete(resources_array, idx)

    # Detect mergers
    mergers = detect_civilization_mergers(civilizations)

    # Process mergers from highest indices to lowest
    mergers.sort(key=lambda pair: (pair[1], pair[0]), reverse=True)

    for absorber, absorbed in mergers:
        events.append({
            "type": "merger",
            "absorber": absorber,
            "absorbed": absorbed,
            "absorber_size": civilizations["sizes"][absorber],
            "absorbed_size": civilizations["sizes"][absorbed]
        })

        # Process merger
        civilizations, knowledge_array = process_civilization_merger(
            civilizations, knowledge_array, absorber, absorbed
        )

        # Remove absorbed civilization
        civilizations, knowledge_array, suppression_array = remove_civilization(
            civilizations, knowledge_array, suppression_array, absorbed
        )

        # Update arrays after removal
        if absorbed < len(influence_array):
            influence_array = np.delete(influence_array, absorbed)
        if absorbed < len(resources_array):
            resources_array = np.delete(resources_array, absorbed)

    # Check for civilization spawning
    num_current_civilizations = len(knowledge_array)
    for i in range(num_current_civilizations):
        # Check if a civilization is large and prosperous enough to spawn an offshoot
        if (civilizations["sizes"][i] > 3.0 and
                knowledge_array[i] > 5.0 and
                resources_array[i] > 20.0 and
                np.random.random() < 0.05):  # 5% chance per timestep

            # Generate position near parent
            spawn_position = (civilizations["positions"][i] +
                              0.5 * np.random.rand(2) *
                              civilizations["expansion_tendency"][i])

            # Spawn new civilization
            civilizations, knowledge_array, suppression_array = spawn_new_civilization(
                civilizations, knowledge_array, suppression_array, spawn_position, parent_idx=i
            )

            # Extend influence and resource arrays
            influence_array = np.append(influence_array, [0.1 * influence_array[i]])
            resources_array = np.append(resources_array, [0.2 * resources_array[i]])

            # Record event
            events.append({
                "type": "spawn",
                "parent": i,
                "position": spawn_position,
                "initial_knowledge": knowledge_array[-1],
                "initial_size": civilizations["sizes"][-1]
            })

    # Occasional random new civilization (cosmic origin)
    if num_current_civilizations < 10 and np.random.random() < 0.01:  # 1% chance per timestep
        # Generate random position
        new_position = 10 * np.random.rand(2)

        # Spawn random new civilization
        civilizations, knowledge_array, suppression_array = spawn_new_civilization(
            civilizations, knowledge_array, suppression_array, new_position
        )

        # Extend influence and resource arrays
        influence_array = np.append(influence_array, [2 + 3 * np.random.rand()])
        resources_array = np.append(resources_array, [5 + 5 * np.random.rand()])

        # Record event
        events.append({
            "type": "new_civilization",
            "position": new_position,
            "initial_knowledge": knowledge_array[-1],
            "initial_size": civilizations["sizes"][-1]
        })

    return (civilizations, knowledge_array, suppression_array,
            influence_array, resources_array, events)