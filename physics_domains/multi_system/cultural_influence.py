"""
Model cultural and ideological influence between civilizations.
"""

import numpy as np


def cultural_influence(civilizations, influence_array, interaction_strength,
                       base_influence_rate=0.02, max_influence_change=1.0, min_division=0.1):
    """
    Model cultural and ideological influence between civilizations.

    Physics Domain: multi_system
    Scale Level: multi_civilization
    Application Domains: civilization

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