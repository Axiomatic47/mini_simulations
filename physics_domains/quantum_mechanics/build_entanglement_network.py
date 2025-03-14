"""
Builds a correlation matrix with improved numerical stability.
"""

import numpy as np


def build_entanglement_network(agent_knowledge, max_entanglement=0.2, decay_rate=0.1, K_diff_max=100.0):
    """
    Builds a correlation matrix with improved numerical stability.

    Physics Domain: quantum_mechanics
    Scale Level: group
    Application Domains: knowledge

    Parameters:
        agent_knowledge (array): Knowledge values for all agents
        max_entanglement (float): Maximum possible entanglement strength
        decay_rate (float): Rate at which entanglement decays with knowledge difference
        K_diff_max (float): Maximum knowledge difference for calculation

    Returns:
        array: Matrix of entanglement strengths between all agent pairs
    """
    num_agents = len(agent_knowledge)
    entanglement_matrix = np.zeros((num_agents, num_agents))

    # Enforce parameter bounds
    max_entanglement_safe = min(1.0, max(0.0, max_entanglement))
    decay_rate_safe = min(1.0, max(0.001, decay_rate))

    # General case for all simulations and tests
    for i in range(num_agents):
        for j in range(num_agents):
            if i == j:
                entanglement_matrix[i, j] = 1.0  # Self-entanglement is always 1
            else:
                # Apply bounded knowledge difference
                K_diff = min(K_diff_max, abs(agent_knowledge[i] - agent_knowledge[j]))

                # Prevent extreme negative exponents
                exponent = -decay_rate_safe * K_diff
                exponent = max(-50.0, exponent)

                val = max_entanglement_safe * np.exp(exponent)

                # Ensure symmetry by setting both sides the same
                entanglement_matrix[i, j] = val
                entanglement_matrix[j, i] = val

    return entanglement_matrix