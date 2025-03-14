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
    S_0_safe = min(100.0, max(0.0, S_0))
    lambda_decay_safe = min(1.0, max(0.0001, lambda_decay))
    t_safe = min(1000.0, max(0.0, t))
    alpha_resurge_safe = min(20.0, max(0.0, alpha_resurge))
    mu_resurge_safe = min(1.0, max(0.0001, mu_resurge))
    effective_lambda = lambda_decay_safe
    if t_safe > 500:
        decay_damping = 1.0 - 0.5 * min(1.0, (t_safe - 500) / 500)
        effective_lambda *= decay_damping
    base_suppression = S_0_safe * circuit_breaker.safe_exp(-effective_lambda * t_safe)
    resurgence = 0.0
    transition_width = 5.0
    if t > t_crit - transition_width and t <= t_crit:
        transition_factor = (t - (t_crit - transition_width)) / max(1e-10, transition_width)
        transition_factor = 0.5 * (1 - np.cos(np.pi * transition_factor))
        time_diff = 0.0
        resurgence_exp = circuit_breaker.safe_exp(-mu_resurge_safe * time_diff)
        early_resurgence = alpha_resurge_safe * resurgence_exp * 0.1
        resurgence = early_resurgence * transition_factor
    elif t > t_crit:
        time_diff = min(500.0, t - t_crit)
        resurgence_exp = circuit_breaker.safe_exp(-mu_resurge_safe * time_diff)
        resurgence = alpha_resurge_safe * resurgence_exp
        if time_diff > 100.0:
            sigmoid_factor = 1.0 / (1.0 + np.exp(min(50, (time_diff - 300.0) / 50.0)))
            damping_factor = 0.1 + 0.9 * sigmoid_factor
            resurgence *= damping_factor
    result = base_suppression + resurgence
    if result < 0.1:
        result = 0.1 * np.exp(min(50, 10 * (result - 0.1)))
    return circuit_breaker.check_and_fix(result)