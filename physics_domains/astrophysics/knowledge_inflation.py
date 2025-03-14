import numpy as np

def knowledge_inflation(K, T, inflation_threshold, expansion_rate=2.0, duration=10,
                        min_multiplier=1.0, max_multiplier=5.0):
    """
    Models rapid knowledge expansion after critical threshold,
    analogous to cosmic inflation.

    Parameters:
        K (float): Current knowledge level
        T (float): Truth adoption level
        inflation_threshold (float): Threshold for triggering inflation
        expansion_rate (float): Base rate of inflation
        duration (float): How long since threshold crossing
        min_multiplier (float): Minimum multiplication factor
        max_multiplier (float): Maximum multiplication factor

    Returns:
        float: Knowledge expansion multiplier
        bool: Whether inflation is active

    Physics Domain: astrophysics
    Scale Level: civilization
    Application Domains: knowledge, truth
    """
    # Apply parameter bounds
    K = max(0, K)
    T = max(0, T)
    inflation_threshold = max(0, inflation_threshold)
    expansion_rate = max(1.0, min(10.0, expansion_rate))
    duration = max(0, duration)

    # Check if inflation threshold has been reached
    is_inflating = T > inflation_threshold

    if not is_inflating:
        return min_multiplier, False

    # Initial exponential growth, then stabilization
    if duration <= 0:
        return min_multiplier, is_inflating

    # Rapid initial expansion that gradually stabilizes
    if duration < 10:
        # Bound exponential input to prevent overflow
        exp_input = max(-10, min(10, -0.3 * (duration - 1)))
        # Exponential growth phase
        multiplier = 1.0 + (expansion_rate - 1.0) * np.exp(exp_input)
    else:
        # Stabilization phase
        multiplier = 1.0 + 0.1 * expansion_rate

    # Ensure multiplier is within bounds
    multiplier = max(min_multiplier, min(max_multiplier, multiplier))

    return multiplier, is_inflating