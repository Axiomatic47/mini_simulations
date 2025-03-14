"""
Models quantum entanglement-like correlations with bounded knowledge difference.
"""

import numpy as np


def quantum_entanglement_correlation(K_i, K_j, rho=0.1, sigma=0.05, K_diff_max=100.0):
    """
    Models quantum entanglement-like correlations with bounded knowledge difference.

    Physics Domain: quantum_mechanics
    Scale Level: agent
    Application Domains: knowledge

    Parameters:
        K_i (float): Knowledge state of agent i
        K_j (float): Knowledge state of agent j
        rho (float): Maximum entanglement strength
        sigma (float): Entanglement decay rate based on knowledge difference
        K_diff_max (float): Maximum knowledge difference for calculation

    Returns:
        float: Entanglement correlation factor (nonlocal connection strength)
    """
    # Enforce parameter bounds
    rho_safe = min(1.0, max(0.0, rho))
    sigma_safe = min(1.0, max(0.001, sigma))

    # Calculate bounded knowledge difference
    K_diff = min(K_diff_max, abs(K_i - K_j))

    # Exponential decay of entanglement with increasing difference
    # Clip exponent to prevent underflow
    exponent = -sigma_safe * K_diff
    exponent = max(-50.0, exponent)  # Prevent extreme negative values

    return rho_safe * np.exp(exponent)