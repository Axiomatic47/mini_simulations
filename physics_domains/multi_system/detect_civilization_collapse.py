"""
Detect civilizations that have collapsed due to high suppression and low knowledge.
"""

import numpy as np


def detect_civilization_collapse(knowledge_array, suppression_array, threshold=0.1, min_division=0.1):
    """
    Detect civilizations that have collapsed due to high suppression and low knowledge.

    Physics Domain: multi_system
    Scale Level: multi_civilization
    Application Domains: civilization, suppression

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