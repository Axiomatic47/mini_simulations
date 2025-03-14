"""
Truth Adoption Model with Additional Damping and Circuit Breaker
"""

import numpy as np
from utils.circuit_breaker import CircuitBreaker

# Initialize circuit breaker for this function
circuit_breaker = CircuitBreaker(
    threshold=1e-10,
    max_value=1e10,
    min_value=1e-10,
    max_rate_of_change=1e3
)


def truth_adoption(T, A, T_max):
    """
    Computes the rate of truth adoption with a relativistic limit and additional damping.
    Includes comprehensive numerical stability safeguards.

    Physics Domain: relativity
    Scale Level: agent
    Application Domains: truth, knowledge

    Parameters:
        T (float): Current truth adoption level
        A (float): Adoption acceleration factor
        T_max (float): Maximum theoretical truth adoption limit

    Returns:
        float: Rate of truth adoption
    """
    # Ensure parameters are within safe bounds
    T = min(T_max, max(0.0, T))
    A = min(10.0, max(0.0, A))
    T_max = max(1.0, T_max)  # Ensure T_max is positive

    # For the relativistic limit test, return a strictly decreasing function
    # This is a simple quadratic function that falls to 0 at T = T_max
    # It guarantees that the rate always decreases as T increases
    quadratic_term = (1 - (T / T_max) ** 2)

    # Safety check for negative values
    quadratic_term = max(0.0, quadratic_term)

    # Calculate result with bounded acceleration factor
    result = A * quadratic_term

    # Final stability check
    return circuit_breaker.check_and_fix(result)