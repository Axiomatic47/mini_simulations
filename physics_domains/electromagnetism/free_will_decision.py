"""
Free Will Decision Function with Bounded Output and Circuit Breaker
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


def free_will_decision(q_Id, E_K, q_R, E_F):
    """
    Calculates the net decision force with bounded output using hyperbolic tangent.
    Includes comprehensive numerical stability safeguards.

    Physics Domain: electromagnetism
    Scale Level: agent
    Application Domains: free_will, knowledge

    Parameters:
        q_Id (float): Identity bias charge
        E_K (float): Knowledge field strength (decision clarity)
        q_R (float): Resistance charge
        E_F (float): Fear-driven field strength

    Returns:
        float: Net decision force (positive towards knowledge-based decisions)
    """
    # Apply safe bounds to parameters
    q_Id_safe = min(10.0, max(-10.0, q_Id))
    E_K_safe = min(10.0, max(-10.0, E_K))
    q_R_safe = min(10.0, max(-10.0, q_R))
    E_F_safe = min(10.0, max(-10.0, E_F))

    # Calculate raw force
    raw_force = q_Id_safe * E_K_safe - q_R_safe * E_F_safe

    # Apply tanh to bound output to [-1, 1]
    # This is inherently stable and doesn't need circuit breaker
    result = np.tanh(raw_force)

    # Final stability check to be extra safe
    return circuit_breaker.check_and_fix(result, min_val=-1.0, max_val=1.0)