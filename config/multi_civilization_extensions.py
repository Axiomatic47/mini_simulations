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
    # Ensure valid parameters
    num_civilizations = max(1, int(num_civilizations))
    max_age_variance = max(1, int(max_age_variance))

    # Generate random starting ages, placing civilizations at different lifecycle phases
    starting_ages = np.random.randint(0, max_age_variance, num_civilizations)

    # Generate positions in a conceptual 2D space
    # Civilizations close to each other will interact more strongly
    positions = np.random.rand(num_civilizations, 2) * 10

    # Generate intrinsic parameters for each civilization with bounds
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
    # Handle empty or invalid input
    if positions.size == 0 or positions.ndim != 2:
        return np.array([[0.0]])

    return cdist(positions, positions)


def calculate_interaction_strength(distance_matrix, max_interaction_distance=5.0, min_distance=0.1):
    """
    Calculate interaction strength between civilizations based on distance.

    Parameters:
        distance_matrix (array): Matrix of distances between civilizations
        max_interaction_distance (float): Maximum distance for interactions
        min_distance (float): Minimum distance to prevent division by zero

    Returns:
        array: Matrix of interaction strengths
    """
    # Apply parameter bounds
    max_interaction_distance = max(min_distance, max_interaction_distance)

    # Create a copy to avoid modifying the original
    safe_distances = np.copy(distance_matrix)

    # Apply minimum distance to prevent division by zero
    safe_distances = np.maximum(safe_distances, min_distance)

    # Create interaction strength matrix with inverse square law
    interaction_strength = 1.0 / (1.0 + safe_distances ** 2)

    # Set diagonal (self-interaction) to 0
    np.fill_diagonal(interaction_strength, 0)

    # Zero out interactions beyond max distance
    interaction_strength[distance_matrix > max_interaction_distance] = 0

    return interaction_strength


def galactic_collision_effect(civ_i, civ_j, collision_threshold=1.0,
                              max_transfer=0.5, min_division=0.1):
    """
    Model effects of a collision or close encounter between civilizations.

    Parameters:
        civ_i (dict): Parameters of the first civilization
        civ_j (dict): Parameters of the second civilization
        collision_threshold (float): Distance threshold for collision effects
        max_transfer (float): Maximum knowledge transfer ratio
        min_division (float): Minimum value for division safety

    Returns:
        tuple: Knowledge transfer, suppression effect, resource exchange
    """
    # Calculate distance between civilizations
    distance = np.linalg.norm(civ_i["position"] - civ_j["position"])

    # If civilizations are very close, model collision effects
    if distance < collision_threshold:
        # Calculate relative development levels with safety minimum
        knowledge_ratio = civ_i["knowledge"] / max(min_division, civ_j["knowledge"])

        # Knowledge transfer (more advanced civ transfers knowledge to less advanced)
        if knowledge_ratio > 1:
            # i transfers knowledge to j
            knowledge_transfer = min(max_transfer * civ_i["knowledge"],
                                     0.1 * (knowledge_ratio - 1) * civ_i["knowledge"])
        else:
            # j transfers knowledge to i
            knowledge_transfer = -min(max_transfer * civ_j["knowledge"],
                                      0.1 * (1 - knowledge_ratio) * civ_j["knowledge"])

        # Suppression effect (stronger civ may suppress weaker)
        power_ratio = civ_i["influence"] / max(min_division, civ_j["influence"])
        if power_ratio > 1.5:
            # i suppresses j
            suppression_effect = min(civ_i["influence"], 0.05 * power_ratio * civ_i["influence"])
        elif power_ratio < 0.67:  # 1/1.5
            # j suppresses i
            suppression_effect = -min(civ_j["influence"],
                                      0.05 * (1 / max(min_division, power_ratio)) * civ_j["influence"])
        else:
            # No significant suppression
            suppression_effect = 0

        # Resource exchange (can be positive or negative)
        resource_differential = civ_i["resources"] - civ_j["resources"]
        resource_exchange = min(abs(resource_differential), 0.02 * abs(resource_differential)) * np.sign(
            resource_differential)

        return knowledge_transfer, suppression_effect, resource_exchange

    # No collision effects if distance is above threshold
    return 0, 0, 0


def knowledge_diffusion(civilizations, knowledge_array, interaction_strength,
                      diffusion_rate=0.01, max_diffusion=0.5, min_division=1e-10):
    """
    Model knowledge diffusion between civilizations with numerical stability safeguards.

    Parameters:
        civilizations (dict): Civilization parameters
        knowledge_array (array): Current knowledge levels for all civilizations
        interaction_strength (array): Matrix of interaction strengths
        diffusion_rate (float): Base rate of knowledge diffusion
        max_diffusion (float): Maximum diffusion amount per time step
        min_division (float): Minimum value to prevent division by zero

    Returns:
        array: Knowledge change due to diffusion
    """
    # Apply parameter bounds
    diffusion_rate = max(0, min(1, diffusion_rate))

    # Get dimensions and ensure they match
    num_civilizations = len(knowledge_array)
    knowledge_change = np.zeros(num_civilizations)

    # Handle empty case
    if num_civilizations == 0:
        return knowledge_change

    # Safety check for interaction_strength dimensions
    if interaction_strength.shape[0] != num_civilizations or interaction_strength.shape[1] != num_civilizations:
        # Resize interaction matrix if dimensions don't match
        temp_interaction = np.zeros((num_civilizations, num_civilizations))
        # Copy available values
        for i in range(min(interaction_strength.shape[0], num_civilizations)):
            for j in range(min(interaction_strength.shape[1], num_civilizations)):
                temp_interaction[i, j] = interaction_strength[i, j]
        interaction_strength = temp_interaction
        print(f"Warning: Interaction strength matrix resized to match {num_civilizations} civilizations")

    # Ensure civilization parameters arrays exist and have correct length
    for key in ["innovation_rates", "knowledge_retention"]:
        if key not in civilizations or len(civilizations[key]) != num_civilizations:
            # Create or resize array with default values
            default_val = 0.5  # middle value as default
            civilizations[key] = np.full(num_civilizations, default_val)
            print(f"Warning: {key} array resized with default values")

    # Calculate knowledge diffusion for each civilization
    for i in range(num_civilizations):
        for j in range(num_civilizations):
            # Skip self-interaction or negative/zero interaction strength
            if i == j or interaction_strength[i, j] <= 0:
                continue

            try:
                # Knowledge flows from higher to lower levels
                knowledge_diff = knowledge_array[j] - knowledge_array[i]

                # Apply diffusion with appropriate bounds
                if knowledge_diff > 0:
                    # Receiving knowledge
                    # Ensure innovation rate is within bounds
                    innovation_rate = np.clip(civilizations["innovation_rates"][i], 0.01, 10.0)

                    # Calculate diffusion amount with bounds
                    diffusion_amount = (diffusion_rate *
                                        interaction_strength[i, j] *
                                        knowledge_diff *
                                        innovation_rate)

                    # Check for NaN or Inf
                    if np.isnan(diffusion_amount) or np.isinf(diffusion_amount):
                        diffusion_amount = max_diffusion * 0.01  # Safe default

                    # Apply maximum diffusion bound
                    knowledge_change[i] += np.clip(diffusion_amount, 0, max_diffusion)
                else:
                    # Giving knowledge - reduced outflow based on knowledge retention
                    # Ensure knowledge retention is within bounds
                    knowledge_retention = np.clip(civilizations["knowledge_retention"][i], 0.0, 1.0)

                    # Calculate diffusion amount with bounds
                    diffusion_amount = (diffusion_rate *
                                        interaction_strength[i, j] *
                                        knowledge_diff *
                                        (1 - knowledge_retention))

                    # Check for NaN or Inf
                    if np.isnan(diffusion_amount) or np.isinf(diffusion_amount):
                        diffusion_amount = -max_diffusion * 0.01  # Safe default

                    # Apply maximum diffusion bound
                    knowledge_change[i] += np.clip(diffusion_amount, -max_diffusion, 0)
            except Exception as e:
                print(f"Warning: Error in knowledge diffusion calculation: {e}")
                # Skip this interaction pair
                continue

    # Final check for NaN or Inf values
    knowledge_change = np.nan_to_num(knowledge_change, nan=0.0, posinf=max_diffusion, neginf=-max_diffusion)

    return knowledge_change


def cultural_influence(civilizations, influence_array, interaction_strength,
                       base_influence_rate=0.02, max_influence_change=1.0, min_division=0.1):
    """
    Model cultural and ideological influence between civilizations.

    Parameters:
        civilizations (dict): Civilization parameters
        influence_array (array): Current influence levels for all civilizations
        interaction_strength (array): Matrix of interaction strengths
        base_influence_rate (float): Base rate of influence spread
        max_influence_change (float): Maximum influence change per time step
        min_division (float): Minimum value for division safety

    Returns:
        array: Influence change due to cultural exchange
    """
    # Apply parameter bounds
    base_influence_rate = max(0, min(1, base_influence_rate))

    num_civilizations = len(influence_array)
    influence_change = np.zeros(num_civilizations)

    # Handle empty case
    if num_civilizations == 0:
        return influence_change

    # Calculate influence spread for each civilization
    for i in range(num_civilizations):
        for j in range(num_civilizations):
            if i != j and interaction_strength[i, j] > 0:
                # Influence effect based on relative sizes
                size_factor = civilizations["sizes"][i] / max(min_division, civilizations["sizes"][j])

                # Calculate influence exchange
                influence_diff = influence_array[j] - influence_array[i]
                influence_direction = 1 if influence_diff > 0 else -1

                # Influence change depends on difference, size, and expansion tendency
                change_amount = (base_influence_rate *
                                 interaction_strength[i, j] *
                                 influence_direction *
                                 np.sqrt(max(0, abs(influence_diff))) *
                                 size_factor *
                                 civilizations["expansion_tendency"][i])

                # Apply maximum change bound
                influence_change[i] += max(-max_influence_change, min(max_influence_change, change_amount))

    return influence_change


def resource_competition(civilizations, resources_array, interaction_strength,
                         competition_rate=0.01, max_resource_change=2.0, min_division=0.1):
    """
    Model competition for resources between civilizations.

    Parameters:
        civilizations (dict): Civilization parameters
        resources_array (array): Current resource levels for all civilizations
        interaction_strength (array): Matrix of interaction strengths
        competition_rate (float): Base rate of resource competition
        max_resource_change (float): Maximum resource change per time step
        min_division (float): Minimum value for division safety

    Returns:
        array: Resource change due to competition
    """
    # Apply parameter bounds
    competition_rate = max(0, min(1, competition_rate))

    num_civilizations = len(resources_array)
    resource_change = np.zeros(num_civilizations)

    # Handle empty case
    if num_civilizations == 0:
        return resource_change

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
                power_ratio = power_i / max(min_division, power_j)

                # Resources flow from weaker to stronger civilizations
                if power_ratio > 1:
                    # i gains resources from j
                    # Use bounded log to prevent extreme values
                    log_ratio = min(10, np.log(max(1, power_ratio)))
                    flow = competition_rate * interaction_strength[i, j] * log_ratio
                    # Apply maximum change bound
                    resource_change[i] += min(max_resource_change, flow)
                    # This will be negative for j in its own calculation

    return resource_change


def civilization_movement(civilizations, interaction_strength, dt=1.0,
                          attraction_factor=0.01, repulsion_threshold=1.0,
                          max_velocity=1.0, damping=0.9):
    """
    Update positions of civilizations based on attractive and repulsive forces.

    Parameters:
        civilizations (dict): Civilization parameters
        interaction_strength (array): Matrix of interaction strengths
        dt (float): Time step
        attraction_factor (float): Strength of attraction between civilizations
        repulsion_threshold (float): Distance threshold for repulsion
        max_velocity (float): Maximum velocity for stability
        damping (float): Velocity damping factor (0-1)

    Returns:
        array: Updated positions for all civilizations
    """
    # Apply parameter bounds
    dt = max(0.01, min(2.0, dt))
    damping = max(0, min(1, damping))

    num_civilizations = len(civilizations["positions"])

    # Handle empty case
    if num_civilizations == 0:
        return civilizations["positions"]

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
    civilizations["velocities"] = damping * civilizations["velocities"] + forces * dt

    # Apply velocity limits for stability
    velocity_magnitudes = np.linalg.norm(civilizations["velocities"], axis=1)
    for i in range(num_civilizations):
        if velocity_magnitudes[i] > max_velocity:
            civilizations["velocities"][i] = (civilizations["velocities"][i] /
                                              velocity_magnitudes[i] * max_velocity)

    # Update positions
    civilizations["positions"] += civilizations["velocities"] * dt

    return civilizations["positions"]


def update_civilization_sizes(civilizations, knowledge_array, influence_array,
                              growth_factor=0.01, max_growth=0.05, min_size=0.1, max_size=10.0):
    """
    Update the sizes of civilizations based on their knowledge and influence.

    Parameters:
        civilizations (dict): Civilization parameters
        knowledge_array (array): Knowledge levels for all civilizations
        influence_array (array): Influence levels for all civilizations
        growth_factor (float): Base growth rate for civilization sizes
        max_growth (float): Maximum growth factor per time step
        min_size (float): Minimum civilization size
        max_size (float): Maximum civilization size

    Returns:
        array: Updated sizes for all civilizations
    """
    # Apply parameter bounds
    growth_factor = max(0, min(0.1, growth_factor))

    num_civilizations = len(civilizations["sizes"])

    # Handle empty case
    if num_civilizations == 0:
        return civilizations["sizes"]

    # Calculate size changes based on knowledge and influence
    for i in range(num_civilizations):
        # Use bounded log functions to prevent extreme values
        log_knowledge = np.log1p(max(0, knowledge_array[i]))
        sqrt_influence = np.sqrt(max(0, influence_array[i]))

        knowledge_effect = log_knowledge * growth_factor
        influence_effect = sqrt_influence * growth_factor * 0.5

        # Combined effect (knowledge has stronger impact than influence)
        size_change = min(max_growth, knowledge_effect + influence_effect)

        # Update size
        civilizations["sizes"][i] *= (1 + size_change)

        # Ensure size is within bounds
        civilizations["sizes"][i] = max(min_size, min(max_size, civilizations["sizes"][i]))

    return civilizations["sizes"]


def detect_civilization_collapse(knowledge_array, suppression_array, threshold=0.1, min_division=0.1):
    """
    Detect civilizations that have collapsed due to high suppression and low knowledge.

    Parameters:
        knowledge_array (array): Knowledge levels for all civilizations
        suppression_array (array): Suppression levels for all civilizations
        threshold (float): Collapse threshold for knowledge/suppression ratio
        min_division (float): Minimum value for division safety

    Returns:
        array: Boolean array indicating collapsed civilizations
    """
    # Ensure arrays are not empty
    if len(knowledge_array) == 0 or len(suppression_array) == 0:
        return np.array([], dtype=bool)

    # Ensure suppression values aren't too small
    safe_suppression = np.maximum(min_division, suppression_array)

    # Calculate knowledge to suppression ratio
    k_s_ratio = knowledge_array / safe_suppression

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

    # Handle empty case
    if num_civilizations <= 1:
        return []

    distance_matrix = calculate_distance_matrix(civilizations["positions"])
    mergers = []

    # Check all pairs of civilizations
    for i in range(num_civilizations):
        for j in range(i + 1, num_civilizations):
            if distance_matrix[i, j] < distance_threshold:
                # FIX: Check size ratio for absorption with safer approach
                # Apply safer bounds to prevent division by zero
                size_i = max(0.01, civilizations["sizes"][i])
                size_j = max(0.01, civilizations["sizes"][j])

                # FIX: Calculate size ratio safely without division by zero risk
                if size_i >= size_j:
                    size_ratio = size_i / size_j
                    if size_ratio > size_ratio_threshold:
                        # i absorbs j
                        mergers.append((i, j))
                else:
                    # Avoid division when size_i is smaller
                    inverse_ratio = size_j / size_i
                    if inverse_ratio > size_ratio_threshold:
                        # j absorbs i
                        mergers.append((j, i))
                # If size ratio is moderate, no merger occurs

    return mergers


def process_civilization_merger(civilizations, knowledge_array, i, j,
                                knowledge_transfer_ratio=0.8, resource_transfer_ratio=1.0,
                                influence_transfer_ratio=0.9, size_transfer_ratio=0.7):
    """
    Process a merger between two civilizations (i absorbs j).

    Parameters:
        civilizations (dict): Civilization parameters
        knowledge_array (array): Knowledge levels for all civilizations
        i (int): Index of absorbing civilization
        j (int): Index of absorbed civilization
        knowledge_transfer_ratio (float): How much knowledge is retained in merger
        resource_transfer_ratio (float): How much resources are retained in merger
        influence_transfer_ratio (float): How much influence is retained in merger
        size_transfer_ratio (float): How much size is retained in merger

    Returns:
        dict: Updated civilization parameters
        array: Updated knowledge array
    """
    # Apply parameter bounds
    knowledge_transfer_ratio = max(0, min(1, knowledge_transfer_ratio))
    resource_transfer_ratio = max(0, min(1, resource_transfer_ratio))
    influence_transfer_ratio = max(0, min(1, influence_transfer_ratio))
    size_transfer_ratio = max(0, min(1, size_transfer_ratio))

    # Ensure indices are valid
    num_civilizations = len(knowledge_array)
    if i >= num_civilizations or j >= num_civilizations or i < 0 or j < 0 or i == j:
        return civilizations, knowledge_array

    # Ensure values are positive
    absorber_knowledge = max(0, knowledge_array[i])
    absorbed_knowledge = max(0, knowledge_array[j])

    # Knowledge combines with diminishing returns
    knowledge_array[i] = absorber_knowledge + knowledge_transfer_ratio * absorbed_knowledge

    # Resources add linearly
    civilizations["resources"][i] += resource_transfer_ratio * civilizations["resources"][j]

    # Influence combines with bonus
    civilizations["influence"][i] += influence_transfer_ratio * civilizations["influence"][j]

    # Size increases based on absorbed civilization
    civilizations["sizes"][i] += size_transfer_ratio * civilizations["sizes"][j]

    # Weighted average of innovation rate (protect against division by zero)
    total_knowledge = max(0.01, absorber_knowledge + absorbed_knowledge)
    civilizations["innovation_rates"][i] = (
            (civilizations["innovation_rates"][i] * absorber_knowledge +
             civilizations["innovation_rates"][j] * absorbed_knowledge) /
            total_knowledge
    )

    # Weighted average of other traits
    total_influence = max(0.01, civilizations["influence"][i] + civilizations["influence"][j])
    civilizations["suppression_resistance"][i] = (
            (civilizations["suppression_resistance"][i] * civilizations["influence"][i] +
             civilizations["suppression_resistance"][j] * civilizations["influence"][j]) /
            total_influence
    )

    # Return updated parameters
    return civilizations, knowledge_array


def spawn_new_civilization(civilizations, knowledge_array, suppression_array, position, parent_idx=None,
                           mutation_factor=0.2, min_size=0.1, max_size=10.0,
                           resource_transfer_ratio=0.2, influence_transfer_ratio=0.1,
                           knowledge_transfer_ratio=0.3):
    """
    Spawn a new civilization, either randomly or as offspring of an existing one.

    Parameters:
        civilizations (dict): Civilization parameters
        knowledge_array (array): Knowledge levels for all civilizations
        suppression_array (array): Suppression levels for all civilizations
        position (array): Starting position for new civilization
        parent_idx (int): Index of parent civilization (None for random)
        mutation_factor (float): How much offspring parameters vary from parent
        min_size (float): Minimum civilization size
        max_size (float): Maximum civilization size
        resource_transfer_ratio (float): Ratio of resources transferred from parent
        influence_transfer_ratio (float): Ratio of influence transferred from parent
        knowledge_transfer_ratio (float): Ratio of knowledge transferred from parent

    Returns:
        dict: Updated civilization parameters
        array: Updated knowledge array
        array: Updated suppression array
    """
    # Apply parameter bounds
    mutation_factor = max(0, min(1, mutation_factor))
    resource_transfer_ratio = max(0, min(1, resource_transfer_ratio))
    influence_transfer_ratio = max(0, min(1, influence_transfer_ratio))
    knowledge_transfer_ratio = max(0, min(1, knowledge_transfer_ratio))

    num_civilizations = len(knowledge_array)

    # FIX: Check that all arrays have the same length
    for key, array in civilizations.items():
        if isinstance(array, np.ndarray) and array.ndim == 1:
            if len(array) != num_civilizations:
                print(f"Warning: Fixing {key} array dimension from {len(array)} to {num_civilizations}")
                if len(array) < num_civilizations:
                    # Extend array with default values
                    padding = np.zeros(num_civilizations - len(array))
                    civilizations[key] = np.append(array, padding)
                else:
                    # Trim array
                    civilizations[key] = array[:num_civilizations]
        elif isinstance(array, np.ndarray) and array.ndim == 2:
            if array.shape[0] != num_civilizations:
                print(f"Warning: Fixing {key} 2D array dimension from {array.shape[0]} to {num_civilizations}")
                if array.shape[0] < num_civilizations:
                    # Extend array with rows of zeros
                    padding = np.zeros((num_civilizations - array.shape[0], array.shape[1]))
                    civilizations[key] = np.vstack([array, padding])
                else:
                    # Trim array
                    civilizations[key] = array[:num_civilizations, :]

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

    if parent_idx is not None and 0 <= parent_idx < num_civilizations:
        # Spawned from parent with moderate inheritance and mutation
        civilizations["ages"][num_civilizations] = 0  # New civilization

        # Inherit with variation from parent
        # Bounded mutation using sigmoid function
        def mutate(value):
            mutation = mutation_factor * (np.random.rand() - 0.5)
            # Use tanh to bound the mutation effect
            return value * (1 + np.tanh(mutation))

        civilizations["innovation_rates"][num_civilizations] = mutate(
            civilizations["innovation_rates"][parent_idx]
        )

        civilizations["suppression_resistance"][num_civilizations] = mutate(
            civilizations["suppression_resistance"][parent_idx]
        )

        civilizations["knowledge_retention"][num_civilizations] = mutate(
            civilizations["knowledge_retention"][parent_idx]
        )

        civilizations["expansion_tendency"][num_civilizations] = mutate(
            civilizations["expansion_tendency"][parent_idx]
        )

        # Initial resources and influence transferred from parent
        resource_transfer = resource_transfer_ratio * civilizations["resources"][parent_idx]
        civilizations["resources"][parent_idx] -= resource_transfer
        civilizations["resources"][num_civilizations] = resource_transfer

        influence_transfer = influence_transfer_ratio * civilizations["influence"][parent_idx]
        civilizations["influence"][parent_idx] -= influence_transfer
        civilizations["influence"][num_civilizations] = influence_transfer

        # Initial knowledge transfer
        knowledge_transfer = knowledge_transfer_ratio * knowledge_array[parent_idx]
        knowledge_array[num_civilizations] = knowledge_transfer

        # Initial size
        civilizations["sizes"][num_civilizations] = min(max_size,
                                                        max(min_size,
                                                            0.5 * civilizations["sizes"][parent_idx]))

    else:
        # Random new civilization with bounded values
        civilizations["ages"][num_civilizations] = 0
        civilizations["innovation_rates"][num_civilizations] = 0.8 + 0.4 * np.random.rand()
        civilizations["suppression_resistance"][num_civilizations] = 0.7 + 0.6 * np.random.rand()
        civilizations["knowledge_retention"][num_civilizations] = 0.6 + 0.4 * np.random.rand()
        civilizations["expansion_tendency"][num_civilizations] = 0.5 + 1.0 * np.random.rand()
        civilizations["resources"][num_civilizations] = 5 + 5 * np.random.rand()
        civilizations["influence"][num_civilizations] = 2 + 3 * np.random.rand()
        civilizations["sizes"][num_civilizations] = max(min_size,
                                                        min(max_size,
                                                            0.5 + 0.5 * np.random.rand()))
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
    # Get number of civilizations and validate index
    num_civilizations = len(knowledge_array)
    if idx < 0 or idx >= num_civilizations:
        return civilizations, knowledge_array, suppression_array

    # Create mask for all civilizations except the one to remove
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
                                          influence_array, resources_array, dt=1.0,
                                          max_spawn_probability=0.05, max_random_spawn_probability=0.01,
                                          max_civilizations=20, min_division=0.01):
    """
    Process all interactions between civilizations in a single time step.

    Parameters:
        civilizations (dict): Civilization parameters
        knowledge_array (array): Knowledge levels for all civilizations
        suppression_array (array): Suppression levels for all civilizations
        influence_array (array): Influence levels for all civilizations
        resources_array (array): Resource levels for all civilizations
        dt (float): Time step size
        max_spawn_probability (float): Maximum probability for spawning new civilizations
        max_random_spawn_probability (float): Maximum probability for random new civilizations
        max_civilizations (int): Maximum number of civilizations allowed
        min_division (float): Minimum value for division safety

    Returns:
        dict: Updated civilization parameters
        array: Updated knowledge array
        array: Updated suppression array
        array: Updated influence array
        array: Updated resources array
        list: Information about key events
    """
    # Apply parameter bounds
    dt = max(0.01, min(2.0, dt))
    max_spawn_probability = max(0, min(1, max_spawn_probability))
    max_random_spawn_probability = max(0, min(1, max_random_spawn_probability))
    max_civilizations = max(1, max_civilizations)

    num_civilizations = len(knowledge_array)
    events = []  # Track significant events

    # Skip processing if only 0 or 1 civilization
    if num_civilizations <= 1:
        return (civilizations, knowledge_array, suppression_array,
                influence_array, resources_array, events)

    # Ensure all civilization arrays are of the correct length
    for key in ["positions", "innovation_rates", "knowledge_retention", "expansion_tendency", "sizes"]:
        if key not in civilizations or len(civilizations[key]) != num_civilizations:
            # Create or resize array with default values
            default_val = 1.0 if key == "sizes" else 0.5
            new_array = np.full(num_civilizations, default_val)

            # For positions, we need a 2D array
            if key == "positions" and key in civilizations:
                new_array = np.zeros((num_civilizations, 2))
                existing_length = min(len(civilizations[key]), num_civilizations)
                for i in range(existing_length):
                    new_array[i] = civilizations[key][i]
                for i in range(existing_length, num_civilizations):
                    new_array[i] = np.random.rand(2) * 10  # Random position
            # Copy any existing values for other arrays
            elif key in civilizations and len(civilizations[key]) > 0:
                existing_length = min(len(civilizations[key]), num_civilizations)
                new_array[:existing_length] = civilizations[key][:existing_length]

            civilizations[key] = new_array
            print(f"Warning: {key} array resized to match {num_civilizations} civilizations")

    # Calculate distance and interaction matrices with array dimension checks
    try:
        distance_matrix = calculate_distance_matrix(civilizations["positions"])

        # Check dimensions match the number of civilizations
        if distance_matrix.shape[0] != num_civilizations or distance_matrix.shape[1] != num_civilizations:
            print(
                f"Warning: Distance matrix dimensions {distance_matrix.shape} don't match number of civilizations {num_civilizations}")
            # Create a safe distance matrix with correct dimensions
            safe_distance = np.zeros((num_civilizations, num_civilizations))
            # Copy values that fit
            rows = min(distance_matrix.shape[0], num_civilizations)
            cols = min(distance_matrix.shape[1], num_civilizations)
            safe_distance[:rows, :cols] = distance_matrix[:rows, :cols]
            distance_matrix = safe_distance

        interaction_strength = calculate_interaction_strength(distance_matrix)
    except Exception as e:
        print(f"Error calculating distance matrix: {e}")
        # Create safe default matrices
        distance_matrix = np.ones((num_civilizations, num_civilizations)) * 5.0  # Default large distance
        np.fill_diagonal(distance_matrix, 0.0)  # Zero distance to self
        interaction_strength = np.zeros((num_civilizations, num_civilizations))  # No interaction by default

    # Process knowledge diffusion with error handling
    try:
        knowledge_change = knowledge_diffusion(
            civilizations,
            knowledge_array,
            interaction_strength,
            min_division=min_division
        )

        # Ensure knowledge_change is not None
        if knowledge_change is None:
            print("Warning: knowledge_diffusion returned None. Using zero-filled array.")
            knowledge_change = np.zeros_like(knowledge_array)
    except Exception as e:
        print(f"Error in knowledge diffusion: {e}")
        knowledge_change = np.zeros_like(knowledge_array)

    # Process cultural influence with error handling
    try:
        influence_change = cultural_influence(
            civilizations,
            influence_array,
            interaction_strength,
            min_division=min_division
        )

        # Ensure influence_change is not None
        if influence_change is None:
            print("Warning: cultural_influence returned None. Using zero-filled array.")
            influence_change = np.zeros_like(influence_array)
    except Exception as e:
        print(f"Error in cultural influence: {e}")
        influence_change = np.zeros_like(influence_array)

    # Process resource competition with error handling
    try:
        resource_change = resource_competition(
            civilizations,
            resources_array,
            interaction_strength,
            min_division=min_division
        )

        # Ensure resource_change is not None
        if resource_change is None:
            print("Warning: resource_competition returned None. Using zero-filled array.")
            resource_change = np.zeros_like(resources_array)
    except Exception as e:
        print(f"Error in resource competition: {e}")
        resource_change = np.zeros_like(resources_array)

    # Detect close encounters/collisions
    for i in range(num_civilizations):
        for j in range(i + 1, num_civilizations):
            try:
                if i < distance_matrix.shape[0] and j < distance_matrix.shape[1] and distance_matrix[i, j] < 1.5:
                    # Process collision effects
                    k_transfer, s_effect, r_exchange = galactic_collision_effect(
                        {"position": civilizations["positions"][i],
                         "knowledge": knowledge_array[i],
                         "influence": influence_array[i],
                         "resources": resources_array[i]},
                        {"position": civilizations["positions"][j],
                         "knowledge": knowledge_array[j],
                         "influence": influence_array[j],
                         "resources": resources_array[j]},
                        min_division=min_division
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
            except Exception as e:
                print(f"Error processing collision between civilizations {i} and {j}: {e}")

    # Apply all changes with bounded growth
    knowledge_array += np.clip(knowledge_change * dt, -knowledge_array * 0.5, knowledge_array * 2)
    influence_array += np.clip(influence_change * dt, -influence_array * 0.5, influence_array * 2)
    resources_array += np.clip(resource_change * dt, -resources_array * 0.5, resources_array * 2)

    # Ensure non-negative values
    knowledge_array = np.maximum(0, knowledge_array)
    influence_array = np.maximum(0, influence_array)
    resources_array = np.maximum(0, resources_array)

    # Update civilization positions with error handling
    try:
        civilization_movement(civilizations, interaction_strength, dt)
    except Exception as e:
        print(f"Error in civilization movement: {e}")

    # Update civilization sizes with error handling
    try:
        update_civilization_sizes(civilizations, knowledge_array, influence_array)
    except Exception as e:
        print(f"Error updating civilization sizes: {e}")

    # Detect collapses
    try:
        collapses = detect_civilization_collapse(knowledge_array, suppression_array, min_division=min_division)
        # Ensure collapses array matches the civilization count
        if len(collapses) != num_civilizations:
            print(f"Warning: collapse array size {len(collapses)} doesn't match civilization count {num_civilizations}")
            # Create a properly sized array of False values
            collapses = np.zeros(num_civilizations, dtype=bool)

        collapsed_indices = np.where(collapses)[0]
    except Exception as e:
        print(f"Error detecting collapses: {e}")
        collapsed_indices = []

    # Process collapses from highest index to lowest to avoid reindexing issues
    for idx in sorted(collapsed_indices, reverse=True):
        if idx < num_civilizations:  # Make sure index is valid
            events.append({
                "type": "collapse",
                "civilization": idx,
                "knowledge": knowledge_array[idx],
                "suppression": suppression_array[idx]
            })

            # Remove collapsed civilization
            try:
                civilizations, knowledge_array, suppression_array = remove_civilization(
                    civilizations, knowledge_array, suppression_array, idx
                )

                # Update arrays after removal
                if idx < len(influence_array):
                    influence_array = np.delete(influence_array, idx)
                if idx < len(resources_array):
                    resources_array = np.delete(resources_array, idx)
            except Exception as e:
                print(f"Error removing collapsed civilization {idx}: {e}")

    # Update num_civilizations after collapses
    num_civilizations = len(knowledge_array)

    # Detect mergers
    try:
        mergers = detect_civilization_mergers(civilizations)
    except Exception as e:
        print(f"Error detecting mergers: {e}")
        mergers = []

    # Filter mergers to ensure valid indices
    valid_mergers = []
    for absorber, absorbed in mergers:
        if 0 <= absorber < num_civilizations and 0 <= absorbed < num_civilizations:
            valid_mergers.append((absorber, absorbed))
    mergers = valid_mergers

    # Process mergers from highest indices to lowest
    mergers.sort(key=lambda pair: (pair[1], pair[0]), reverse=True)

    for absorber, absorbed in mergers:
        try:
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
        except Exception as e:
            print(f"Error processing merger between {absorber} and {absorbed}: {e}")

    # Update num_civilizations after mergers
    num_civilizations = len(knowledge_array)

    # Check for civilization spawning (with bounds on maximum number)
    if num_civilizations < max_civilizations:
        for i in range(min(num_civilizations, len(civilizations["sizes"]))):
            try:
                # Check if a civilization is large and prosperous enough to spawn an offshoot
                spawn_probability = min(max_spawn_probability,
                                        0.05 * (civilizations["sizes"][i] > 3.0) *
                                        (knowledge_array[i] > 5.0) *
                                        (i < len(resources_array) and resources_array[i] > 20.0))

                if np.random.random() < spawn_probability:
                    # Ensure index is valid for expansion_tendency
                    if i < len(civilizations["expansion_tendency"]):
                        expansion = civilizations["expansion_tendency"][i]
                    else:
                        expansion = 1.0  # Default

                    # Generate position near parent
                    spawn_position = (civilizations["positions"][i] +
                                      0.5 * np.random.rand(2) * expansion)

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
            except Exception as e:
                print(f"Error in civilization spawn from parent {i}: {e}")

        # Occasional random new civilization (cosmic origin)
        try:
            random_spawn_probability = min(max_random_spawn_probability,
                                           0.01 * (num_civilizations < 10))
            if np.random.random() < random_spawn_probability:
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
        except Exception as e:
            print(f"Error spawning random new civilization: {e}")

    return (civilizations, knowledge_array, suppression_array,
            influence_array, resources_array, events)