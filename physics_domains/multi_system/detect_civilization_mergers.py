"""
Detect and process civilization mergers when they become very close.
"""

import numpy as np
from .calculate_distance_matrix import calculate_distance_matrix


def detect_civilization_mergers(civilizations, distance_threshold=0.5, size_ratio_threshold=3.0):
    """
    Detect and process civilization mergers when they become very close.

    Physics Domain: multi_system
    Scale Level: multi_civilization
    Application Domains: civilization

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