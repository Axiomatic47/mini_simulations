"""
Process all interactions between civilizations in a single time step.
"""

import numpy as np
from utils.dim_handler import safe_process_civilization_interactions
from .calculate_distance_matrix import calculate_distance_matrix
from .calculate_interaction_strength import calculate_interaction_strength
from .knowledge_diffusion import knowledge_diffusion
from .cultural_influence import cultural_influence
from .resource_competition import resource_competition
from .civilization_movement import civilization_movement
from .update_civilization_sizes import update_civilization_sizes
from .detect_civilization_collapse import detect_civilization_collapse
from .detect_civilization_mergers import detect_civilization_mergers
from .process_civilization_merger import process_civilization_merger
from .remove_civilization import remove_civilization
from .spawn_new_civilization import spawn_new_civilization


def _process_interactions(civs_data, num_civs):
    """
    Internal helper function for processing civilization interactions.

    Physics Domain: multi_system
    Scale Level: multi_civilization
    Application Domains: civilization
    """
    # Unpack data
    civilizations = civs_data['civilizations']
    knowledge_array = civs_data['knowledge_array']
    suppression_array = civs_data['suppression_array']
    influence_array = civs_data['influence_array']
    resources_array = civs_data['resources_array']
    dt = civs_data.get('dt', 1.0)
    max_spawn_probability = civs_data.get('max_spawn_probability', 0.05)
    max_random_spawn_probability = civs_data.get('max_random_spawn_probability', 0.01)
    max_civilizations = civs_data.get('max_civilizations', 20)
    min_division = civs_data.get('min_division', 0.01)

    events = []  # Track significant events

    # Skip processing if only 0 or 1 civilization
    if num_civs <= 1:
        return {
            'civilizations': civilizations,
            'knowledge_array': knowledge_array,
            'suppression_array': suppression_array,
            'influence_array': influence_array,
            'resources_array': resources_array,
            'events': events
        }

    # Calculate distance and interaction matrices with array dimension checks
    try:
        distance_matrix = calculate_distance_matrix(civilizations["positions"])
        interaction_strength = calculate_interaction_strength(distance_matrix)
    except Exception as e:
        print(f"Error calculating distance matrix: {e}")
        # Create safe default matrices
        distance_matrix = np.ones((num_civs, num_civs)) * 5.0  # Default large distance
        np.fill_diagonal(distance_matrix, 0.0)  # Zero distance to self
        interaction_strength = np.zeros((num_civs, num_civs))  # No interaction by default

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
    for i in range(num_civs):
        for j in range(i + 1, num_civs):
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
        if len(collapses) != num_civs:
            print(f"Warning: collapse array size {len(collapses)} doesn't match civilization count {num_civs}")
            # Create a properly sized array of False values
            collapses = np.zeros(num_civs, dtype=bool)

        collapsed_indices = np.where(collapses)[0]
    except Exception as e:
        print(f"Error detecting collapses: {e}")
        collapsed_indices = []

    # Process collapses from highest index to lowest to avoid reindexing issues
    for idx in sorted(collapsed_indices, reverse=True):
        if idx < num_civs:  # Make sure index is valid
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

    # Update num_civs after collapses
    num_civs = len(knowledge_array)

    # Detect mergers
    try:
        mergers = detect_civilization_mergers(civilizations)
    except Exception as e:
        print(f"Error detecting mergers: {e}")
        mergers = []

    # Filter mergers to ensure valid indices
    valid_mergers = []
    for absorber, absorbed in mergers:
        if 0 <= absorber < num_civs and 0 <= absorbed < num_civs:
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

    # Update num_civs after mergers
    num_civs = len(knowledge_array)

    # Check for civilization spawning (with bounds on maximum number)
    if num_civs < max_civilizations:
        for i in range(min(num_civs, len(civilizations["sizes"]))):
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
                                           0.01 * (num_civs < 10))
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

    # Return updated data
    return {
        'civilizations': civilizations,
        'knowledge_array': knowledge_array,
        'suppression_array': suppression_array,
        'influence_array': influence_array,
        'resources_array': resources_array,
        'events': events
    }


def process_all_civilization_interactions(civilizations, knowledge_array, suppression_array, influence_array,
                                          resources_array, dt=1.0, max_spawn_probability=0.05,
                                          max_random_spawn_probability=0.01, max_civilizations=20, min_division=0.01):
    """
    Process all interactions between civilizations in a single time step.

    Physics Domain: multi_system
    Scale Level: multi_civilization
    Application Domains: civilization, knowledge, suppression

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
    # Package data
    civs_data = {
        'civilizations': civilizations,
        'knowledge_array': knowledge_array,
        'suppression_array': suppression_array,
        'influence_array': influence_array,
        'resources_array': resources_array,
        'dt': dt,
        'max_spawn_probability': max_spawn_probability,
        'max_random_spawn_probability': max_random_spawn_probability,
        'max_civilizations': max_civilizations,
        'min_division': min_division
    }

    # Use the safe version
    result = safe_process_civilization_interactions(civs_data, _process_interactions)

    # Unpack results
    return (
        result['civilizations'],
        result['knowledge_array'],
        result['suppression_array'],
        result['influence_array'],
        result['resources_array'],
        result['events']
    )