"""
Remove a civilization from the simulation (due to collapse or merger).
"""

import numpy as np


def remove_civilization(civilizations, knowledge_array, suppression_array, idx):
    """
    Remove a civilization from the simulation (due to collapse or merger).

    Physics Domain: multi_system
    Scale Level: multi_civilization
    Application Domains: civilization

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