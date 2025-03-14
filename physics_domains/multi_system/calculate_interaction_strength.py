"""
Calculate interaction strength between civilizations based on distance.
"""

import numpy as np


def calculate_interaction_strength(distance_matrix, max_interaction_distance=5.0, min_distance=0.1):
    """
    Calculate interaction strength between civilizations based on distance.

    Physics Domain: multi_system
    Scale Level: multi_civilization
    Application Domains: civilization

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