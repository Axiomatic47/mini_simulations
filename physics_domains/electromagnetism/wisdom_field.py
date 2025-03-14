"""
Wisdom Field with Smooth Transitions and Circuit Breaker
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


def wisdom_field(W_0, alpha, S, R, K, max_growth=5.0):
    """
    Computes wisdom field strength with numerical safeguards and smooth transitions.
    Includes comprehensive numerical stability safeguards.

    Physics Domain: electromagnetism
    Scale Level: agent
    Application Domains: wisdom, knowledge, suppression

    Parameters:
        W_0 (float): Base wisdom level
        alpha (float): Suppression impact factor
        S (float): Suppression level
        R (float): Resistance level
        K (float): Knowledge level
        max_growth (float): Maximum growth multiplier

    Returns:
        float: Wisdom field strength
    """
    # Apply safe bounds to all parameters
    W_0_safe = min(10.0, max(0.01, W_0))
    alpha_safe = min(1.0, max(0.0, alpha))
    S_safe = min(100.0, max(0.0, S))
    R_safe = min(100.0, max(0.0, R))
    K_safe = max(0.001, K)  # Prevent division by zero

    # Suppression effect with smoother transition for high suppression
    if S_safe > 25.0:
        # Gradually reduce sensitivity to suppression at high levels
        effective_alpha = alpha_safe / (1.0 + 0.01 * (S_safe - 25.0))
        suppression_effect = circuit_breaker.safe_exp(-effective_alpha * S_safe)
    else:
        suppression_effect = circuit_breaker.safe_exp(-alpha_safe * S_safe)

    # Knowledge integration with smooth transitions
    # Use sigmoid for R/K ratio to create smoother behavior around thresholds
    R_capped = min(R_safe, 10.0)  # Cap resistance to prevent explosion

    # Apply progressive smoothing based on K value
    if K_safe < 1.0:
        # For very low K, smooth transition to prevent extreme growth
        k_factor = K_safe
        r_k_ratio = R_capped / (K_safe + 1.0)
        integration_factor = 1.0 + r_k_ratio * k_factor
    else:
        # Normal case with sigmoid-like smooth growth curve
        r_k_ratio = R_capped / K_safe
        sigmoid_input = r_k_ratio - 1.0  # Center sigmoid around r_k_ratio = 1
        sigmoid_factor = 1.0 / (1.0 + np.exp(-sigmoid_input))

        # Map sigmoid output to [1, max_growth] range
        integration_factor = 1.0 + (max_growth - 1.0) * sigmoid_factor

    # Final smooth capping to max_growth
    if integration_factor > 0.9 * max_growth:
        # Soft maximum approaching max_growth
        excess = integration_factor - 0.9 * max_growth
        soft_excess = excess / (1.0 + 0.1 * excess)
        integration_factor = 0.9 * max_growth + soft_excess

    # Combine effects with smooth bounds
    result = W_0_safe * suppression_effect * integration_factor

    # Final soft maximum
    if result > W_0_safe * max_growth * 0.95:
        # Smooth approach to absolute maximum
        excess = result - W_0_safe * max_growth * 0.95
        soft_excess = excess / (1.0 + excess / (W_0_safe * max_growth * 0.05))
        result = W_0_safe * max_growth * 0.95 + soft_excess

    # Final stability check with hard cap as safety
    return circuit_breaker.check_and_fix(result, max_val=W_0_safe * max_growth)


def wisdom_field_enhanced(W_0, alpha, S, R, K, max_growth=5.0):
    """
    Enhanced version of wisdom field equation with additional stabilization.
    This function is kept for backward compatibility.
    The main wisdom_field function now includes all of these enhancements.

    Physics Domain: electromagnetism
    Scale Level: agent
    Application Domains: wisdom, knowledge, suppression

    Parameters:
        W_0 (float): Base wisdom level
        alpha (float): Suppression impact factor
        S (float): Suppression level
        R (float): Resistance level
        K (float): Knowledge level
        max_growth (float): Maximum growth multiplier

    Returns:
        float: Wisdom field strength
    """
    # Simply call the updated wisdom_field function which now has all enhancements
    return wisdom_field(W_0, alpha, S, R, K, max_growth)