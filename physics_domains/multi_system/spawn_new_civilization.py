"""
Spawn a new civilization, either randomly or as offspring of an existing one.
"""

import numpy as np


def spawn_new_civilization(civilizations, knowledge_array, suppression_array, position, parent_idx=None,
                           mutation_factor=0.2, min_size=0.1, max_size=10.0,
                           resource_transfer_ratio=0.2, influence_transfer_ratio=0.1,
                           knowledge_transfer_ratio=0.3):
    """
    Spawn a new civilization, either randomly or as offspring of an existing one.

    Physics Domain: multi_system
    Scale Level: multi_civilization
    Application Domains: civilization, knowledge

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