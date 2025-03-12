import numpy as np
from utils.circuit_breaker import CircuitBreaker

def intelligence_growth(K, W, R, S, N, K_max=100.0):
    """
    Computes intelligence growth with saturation to prevent unbounded growth.

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
    # Initialize circuit breaker for numerical stability
    circuit_breaker = CircuitBreaker(
        threshold=1e-10,
        max_value=1e10,
        min_value=1e-10,
        max_rate_of_change=1.0
    )

    # Apply saturation term to prevent unbounded growth
    # Cap knowledge to prevent overflow
    K_safe = min(K_max, max(0.0, K))

    # Apply safe bounds to other parameters
    W_safe = min(10.0, max(0.0, W))
    R_safe = min(100.0, max(0.0, R))
    S_safe = min(100.0, max(0.0, S))
    N_safe = min(10.0, max(-10.0, N))

    # Use saturation term in denominator to limit growth
    numerator = K_safe * W_safe
    denominator = 1.0 + K_safe / K_max

    # Safe division with circuit breaker
    growth_term = circuit_breaker.safe_div(numerator, denominator)

    # Calculate final result
    result = growth_term - R_safe - S_safe + N_safe

    # Final stability check
    return circuit_breaker.check_and_fix(result)
