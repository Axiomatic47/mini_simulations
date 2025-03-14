# physics_domains/strong_nuclear/identity_binding.py
import numpy as np


def identity_binding(K, I, alpha_identity=0.2, max_identity=10.0):
    """
    Models how agent identity binds to knowledge.

    Physics Domain: strong_nuclear
    Scale Level: agent
    Application Domains: identity, knowledge

    Parameters:
        K (float): Knowledge level
        I (float): Intelligence level
        alpha_identity (float): Identity binding coefficient
        max_identity (float): Maximum identity strength

    Returns:
        float: Identity binding strength
    """
    # Prevent division by zero and ensure bounded output
    denominator = max(1.0, K)

    # Calculate binding strength
    binding_strength = alpha_identity * I * (1.0 - np.exp(-K / denominator))

    # Ensure output is bounded
    return min(max_identity, max(0.0, binding_strength))