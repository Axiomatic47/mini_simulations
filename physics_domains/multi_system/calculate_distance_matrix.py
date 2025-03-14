"""
Calculate distances between all pairs of civilizations.
"""

import numpy as np
from utils.dim_handler import safe_calculate_distance_matrix


def calculate_distance_matrix(positions):
    """
    Calculate distances between all pairs of civilizations.

    Physics Domain: multi_system
    Scale Level: multi_civilization
    Application Domains: civilization

    Parameters:
        positions (array): 2D array of civilization positions

    Returns:
        array: Matrix of distances between civilizations
    """
    # Use the safe version
    return safe_calculate_distance_matrix(positions)