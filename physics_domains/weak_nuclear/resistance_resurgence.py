"""
Resistance Resurgence with Smooth Transitions and Circuit Breaker
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

def resistance_resurgence(S_0, lambda_decay, t, alpha_resurge, mu_resurge, t_crit):
    """
    Computes resistance resurgence and decay with smooth transitions at thresholds.
    Includes comprehensive numerical stability safeguards.

    Physics Domain: weak_nuclear
    Scale Level: group
    Application Domains: resistance, suppression

    Parameters:
        S_0 (float): Initial suppression level
        lambda_decay (float): Exponential decay rate
        t (float): Time step
        alpha_resurge (float): Resurgence intensity
        mu_resurge (float): Resurgence decay rate
        t_crit (float): Critical time for resurgence

    Returns:
        float: Suppression level with resurgence
    """
    # Apply safe bounds to all parameters
    S_0_safe = min(100.0, max(0.0, S_0))
    lambda_decay_safe = min(1.0, max(0.0001, lambda_decay))
    t_safe = min(1000.0, max(0.0, t))  # Cap time to prevent overflow
    alpha_resurge_safe = min(20.0, max(0.0, alpha_resurge))
    mu_resurge_safe = min(1.0, max(0.0001, mu_resurge))

    # Base exponential decay with smooth rate transition at very long times
    # Use a smoothly decreasing decay rate for very long times
    effective_lambda = lambda_decay_safe
    if t_safe > 500:
        decay_damping = 1.0 - 0.5 * min(1.0, (t_safe - 500) / 500)  # Gradually reduce decay rate
        effective_lambda *= decay_damping

    base_suppression = S_0_safe * circuit_breaker.safe_exp(-effective_lambda * t_safe)

    # Smooth transition around critical time for resurgence
    resurgence = 0.0
    # Create transition window around critical time
    transition_width = 5.0  # Width of transition window

    if t > t_crit - transition_width and t <= t_crit:
        # Pre-critical smooth ramp-up
        transition_factor = (t - (t_crit - transition_width)) / transition_width
        transition_factor = 0.5 * (1 - np.cos(np.pi * transition_factor))  # Cosine smoothing

        # Calculate early resurgence with gradual onset
        time_diff = 0.0  # At critical time, time_diff will be 0
        resurgence_exp = circuit_breaker.safe_exp(-mu_resurge_safe * time_diff)
        early_resurgence = alpha_resurge_safe * resurgence_exp * 0.1  # Start at 10% of full strength

        # Scale by transition factor
        resurgence = early_resurgence * transition_factor

    elif t > t_crit:
        # Post-critical full resurgence with smooth decay
        time_diff = min(500.0, t - t_crit)

        # Calculate full resurgence with smoother decay
        resurgence_exp = circuit_breaker.safe_exp(-mu_resurge_safe * time_diff)
        resurgence = alpha_resurge_safe * resurgence_exp

        # Add smoother long-term damping
        if time_diff > 100.0:
            # Use sigmoid function for smoother damping
            sigmoid_factor = 1.0 / (1.0 + np.exp((time_diff - 300.0) / 50.0))
            # Ensure minimum of 10% strength remains for very long times
            damping_factor = 0.1 + 0.9 * sigmoid_factor
            resurgence *= damping_factor

    # Combine effects with smooth lower bound
    # Use soft minimum to avoid abrupt transitions to zero
    result = base_suppression + resurgence
    if result < 0.1:
        # Soft minimum that approaches but never reaches exactly zero
        result = 0.1 * np.exp(10 * (result - 0.1))

    # Final stability check
    return circuit_breaker.check_and_fix(result)