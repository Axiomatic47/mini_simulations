"""
Model effects of a collision or close encounter between civilizations.
"""

import numpy as np

def galactic_collision_effect(civ_i, civ_j, collision_threshold=1.0,
                             max_transfer=0.5, min_division=0.1):
    """
    Model effects of a collision or close encounter between civilizations.

    Physics Domain: multi_system
    Scale Level: multi_civilization
    Application Domains: civilization, knowledge, suppression

    Parameters:
        civ_i (dict): Parameters of the first civilization
        civ_j (dict): Parameters of the second civilization
        collision_threshold (float): Distance threshold for collision effects
        max_transfer (float): Maximum knowledge transfer ratio
        min_division (float): Minimum value for division safety

    Returns:
        tuple: Knowledge transfer, suppression effect, resource exchange
    """
    # Calculate distance between civilizations
    distance = np.linalg.norm(civ_i["position"] - civ_j["position"])

    # If civilizations are very close, model collision effects
    if distance < collision_threshold:
        # Calculate relative development levels with safety minimum
        knowledge_ratio = civ_i["knowledge"] / max(min_division, civ_j["knowledge"])

        # Knowledge transfer (more advanced civ transfers knowledge to less advanced)
        if knowledge_ratio > 1:
            # i transfers knowledge to j
            knowledge_transfer = min(max_transfer * civ_i["knowledge"],
                                     0.1 * (knowledge_ratio - 1) * civ_i["knowledge"])
        else:
            # j transfers knowledge to i
            knowledge_transfer = -min(max_transfer * civ_j["knowledge"],
                                      0.1 * (1 - knowledge_ratio) * civ_j["knowledge"])

        # Suppression effect (stronger civ may suppress weaker)
        power_ratio = civ_i["influence"] / max(min_division, civ_j["influence"])
        if power_ratio > 1.5:
            # i suppresses j
            suppression_effect = min(civ_i["influence"], 0.05 * power_ratio * civ_i["influence"])
        elif power_ratio < 0.67:  # 1/1.5
            # j suppresses i
            suppression_effect = -min(civ_j["influence"],
                                     0.05 * (1 / max(min_division, power_ratio)) * civ_j["influence"])
        else:
            # No significant suppression
            suppression_effect = 0

        # Resource exchange (can be positive or negative)
        resource_differential = civ_i["resources"] - civ_j["resources"]
        resource_exchange = min(abs(resource_differential), 0.02 * abs(resource_differential)) * np.sign(
            resource_differential)

        return knowledge_transfer, suppression_effect, resource_exchange

    # No collision effects if distance is above threshold
    return 0, 0, 0