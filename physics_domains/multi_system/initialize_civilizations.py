"""
Initialize multiple civilizations with different starting ages and positions.
"""

import numpy as np


def initialize_civilizations(num_civilizations, max_age_variance=100):
    """
    Initialize multiple civilizations with different starting ages and positions.

    Physics Domain: multi_system
    Scale Level: multi_civilization
    Application Domains: civilization

    Parameters:
        num_civilizations (int): Number of civilizations to initialize
        max_age_variance (int): Maximum variance in starting ages

    Returns:
        dict: Dictionary containing civilization parameters
    """
    # Ensure valid parameters
    num_civilizations = max(1, int(num_civilizations))
    max_age_variance = max(1, int(max_age_variance))

    # Generate random starting ages, placing civilizations at different lifecycle phases
    starting_ages = np.random.randint(0, max_age_variance, num_civilizations)

    # Generate positions in a conceptual 2D space
    # Civilizations close to each other will interact more strongly
    positions = np.random.rand(num_civilizations, 2) * 10

    # Generate intrinsic parameters for each civilization with bounds
    innovation_rates = 0.8 + 0.4 * np.random.rand(num_civilizations)  # 0.8-1.2
    suppression_resistance = 0.7 + 0.6 * np.random.rand(num_civilizations)  # 0.7-1.3
    knowledge_retention = 0.6 + 0.4 * np.random.rand(num_civilizations)  # 0.6-1.0
    expansion_tendency = 0.5 + 1.0 * np.random.rand(num_civilizations)  # 0.5-1.5

    # Starting resources and influence levels
    resources = 10 + 10 * np.random.rand(num_civilizations)
    influence = 5 + 5 * np.random.rand(num_civilizations)

    # Create dictionary of civilization parameters
    return {
        "ages": starting_ages,
        "positions": positions,
        "innovation_rates": innovation_rates,
        "suppression_resistance": suppression_resistance,
        "knowledge_retention": knowledge_retention,
        "expansion_tendency": expansion_tendency,
        "resources": resources,
        "influence": influence,
        "velocities": np.zeros((num_civilizations, 2)),  # Initially static
        "sizes": 1.0 + np.random.rand(num_civilizations)  # Initial size/scale
    }