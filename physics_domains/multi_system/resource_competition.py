"""
Model competition for resources between civilizations.
"""

import numpy as np


def resource_competition(civilizations, resources_array, interaction_strength,
                         competition_rate=0.01, max_resource_change=2.0, min_division=0.1):
    """
    Model competition for resources between civilizations.

    Physics Domain: multi_system
    Scale Level: multi_civilization
    Application Domains: civilization

    Parameters:
        civilizations (dict): Civilization parameters
        resources_array (array): Current resource levels for all civilizations
        interaction_strength (array): Matrix of interaction strengths
        competition_rate (float): Base rate of resource competition
        max_resource_change (float): Maximum resource change per time step
        min_division (float): Minimum value for division safety

    Returns:
        array: Resource change due to competition
    """
    # Apply parameter bounds
    competition_rate = max(0, min(1, competition_rate))

    num_civilizations = len(resources_array)
    resource_change = np.zeros(num_civilizations)

    # Handle empty case
    if num_civilizations == 0:
        return resource_change

    # Calculate resource competition effects
    for i in range(num_civilizations):
        for j in range(num_civilizations):
            if i != j and interaction_strength[i, j] > 0:
                # Calculate relative power (combination of knowledge and influence)
                power_i = (resources_array[i] +
                           civilizations["influence"][i] +
                           civilizations["knowledge_retention"][i] * 10)

                power_j = (resources_array[j] +
                           civilizations["influence"][j] +
                           civilizations["knowledge_retention"][j] * 10)

                # Power ratio determines resource flow
                power_ratio = power_i / max(min_division, power_j)

                # Resources flow from weaker to stronger civilizations
                if power_ratio > 1:
                    # i gains resources from j
                    # Use bounded log to prevent extreme values
                    log_ratio = min(10, np.log(max(1, power_ratio)))
                    flow = competition_rate * interaction_strength[i, j] * log_ratio
                    # Apply maximum change bound
                    resource_change[i] += min(max_resource_change, flow)
                    # This will be negative for j in its own calculation

    return resource_change