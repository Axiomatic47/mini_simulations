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
    W_0_safe = min(10.0, max(0.01, W_0))
    alpha_safe = min(1.0, max(0.0, alpha))
    S_safe = min(100.0, max(0.0, S))
    R_safe = min(100.0, max(0.0, R))
    K_safe = max(0.001, K)
    if S_safe > 25.0:
        effective_alpha = alpha_safe / (1.0 + 0.01 * (S_safe - 25.0))
        suppression_effect = circuit_breaker.safe_exp(-effective_alpha * S_safe)
    else:
        suppression_effect = circuit_breaker.safe_exp(-alpha_safe * S_safe)
    R_capped = min(R_safe, 10.0)
    if K_safe < 1.0:
        k_factor = K_safe
        r_k_ratio = R_capped / (K_safe + 1.0)
        integration_factor = 1.0 + r_k_ratio * k_factor
    else:
        r_k_ratio = R_capped / max(1e-10, K_safe)
        sigmoid_input = r_k_ratio - 1.0
        sigmoid_factor = 1.0 / (1.0 + np.exp(min(50, -sigmoid_input)))
        integration_factor = 1.0 + (max_growth - 1.0) * sigmoid_factor
    if integration_factor > 0.9 * max_growth:
        excess = integration_factor - 0.9 * max_growth
        soft_excess = excess / (1.0 + 0.1 * excess)
        integration_factor = 0.9 * max_growth + soft_excess
    result = W_0_safe * suppression_effect * integration_factor
    if result > W_0_safe * max_growth * 0.95:
        excess = result - W_0_safe * max_growth * 0.95
        soft_excess = excess / (1.0 + excess / (W_0_safe * max_growth * 0.05))
        result = W_0_safe * max_growth * 0.95 + soft_excess
    return circuit_breaker.check_and_fix(result, max_val=W_0_safe * max_growth)