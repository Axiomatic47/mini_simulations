"""
Process a merger between two civilizations (i absorbs j).
"""

import numpy as np


def process_civilization_merger(civilizations, knowledge_array, i, j,
                                knowledge_transfer_ratio=0.8, resource_transfer_ratio=1.0,
                                influence_transfer_ratio=0.9, size_transfer_ratio=0.7):
    """
    Process a merger between two civilizations (i absorbs j).

    Physics Domain: multi_system
    Scale Level: multi_civilization
    Application Domains: civilization, knowledge

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