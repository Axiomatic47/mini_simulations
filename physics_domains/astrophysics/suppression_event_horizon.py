import numpy as np

def suppression_event_horizon(S, K, critical_constant=2.0, min_K=0.01, max_S=1000.0):
    """
    Calculates suppression threshold using black hole event horizon analogy.

    Parameters:
        S (float): Suppression level (analogous to mass)
        K (float): Knowledge level (analogous to escape velocity)
        critical_constant (float): Similar to G in Schwarzschild radius
        min_K (float): Minimum knowledge value to prevent division by zero
        max_S (float): Maximum suppression value for stability

    Returns:
        float: Critical radius beyond which knowledge cannot escape suppression
        bool: Whether system is beyond event horizon (True if suppressed)

    Physics Domain: astrophysics
    Scale Level: civilization
    Application Domains: suppression, knowledge
    """
    # Apply parameter bounds
    K = max(min_K, K)
    S = max(0, min(max_S, S))
    critical_constant = max(0, critical_constant)

    # Event horizon calculation: r_s = 2GM/c²
    # Analogous formulation: r_critical ∝ S/K²
    event_horizon = critical_constant * S / (K ** 2)

    # Determine if current state is beyond event horizon
    # If ratio of S/K² exceeds threshold, suppression is dominant
    is_beyond_horizon = event_horizon > 1.0

    return event_horizon, is_beyond_horizon