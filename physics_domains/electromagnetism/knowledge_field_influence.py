"""
Electromagnetism Analogy: Knowledge Field Influence
"""

import numpy as np

def knowledge_field_influence(K_i, K_j, r_ij, kappa=0.05, K_max=1000.0, r_min=0.1):
    """
    Calculates the electromagnetic-like influence of knowledge fields with improved stability.

    Physics Domain: electromagnetism
    Scale Level: group
    Application Domains: knowledge

    Parameters:
        K_i (float): Knowledge state of agent i
        K_j (float): Knowledge state of agent j
        r_ij (float): Relational or conceptual distance between agents
        kappa (float): Knowledge permeability constant
        K_max (float): Maximum knowledge value for stability
        r_min (float): Minimum distance to prevent division by zero

    Returns:
        float: Knowledge field influence (analogous to electromagnetic force)
    """
    # Enforce parameter bounds
    K_i_safe = min(K_max, max(0.0, K_i))
    K_j_safe = min(K_max, max(0.0, K_j))
    r_ij_safe = max(r_min, r_ij)
    kappa_safe = min(1.0, max(0.0, kappa))

    # Coulomb's Law analog for knowledge field influence
    return kappa_safe * K_i_safe * K_j_safe / (r_ij_safe ** 2)