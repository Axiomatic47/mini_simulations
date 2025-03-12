def intelligence_growth(K, W, R, S, N, K_max=100.0):
    """
    Computes intelligence growth with saturation to prevent unbounded growth.
    Includes comprehensive numerical stability safeguards.

    Parameters:
        K (float): Knowledge level
        W (float): Wisdom factor (knowledge integration efficiency)
        R (float): Resistance level
        S (float): Suppression level
        N (float): Network effect (mutual learning contribution)
        K_max (float): Maximum knowledge capacity (prevents unbounded growth)

    Returns:
        float: Intelligence growth rate
    """
    K_safe = min(K_max, max(0.0, K))
    W_safe = min(10.0, max(0.0, W))
    R_safe = min(100.0, max(0.0, R))
    S_safe = min(100.0, max(0.0, S))
    N_safe = min(10.0, max(-10.0, N))
    numerator = K_safe * W_safe
    denominator = 1.0 + K_safe / max(1e-10, K_max)
    growth_term = circuit_breaker.safe_div(numerator, denominator)
    result = growth_term - R_safe - S_safe + N_safe
    return circuit_breaker.check_and_fix(result)