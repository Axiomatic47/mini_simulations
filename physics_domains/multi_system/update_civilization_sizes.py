"""
Update the sizes of civilizations based on their knowledge and influence.
"""

import numpy as np


def update_civilization_sizes(civilizations, knowledge_array, influence_array,
                              growth_factor=0.01, max_growth=0.05, min_size=0.1, max_size=10.0):
    """
    Update the sizes of civilizations based on their knowledge and influence.

    Physics Domain: multi_system
    Scale Level: multi_civilization
    Application Domains: civilization

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