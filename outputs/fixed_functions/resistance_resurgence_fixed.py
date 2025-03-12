def resistance_resurgence(S_0, lambda_decay, t, alpha_resurge, mu_resurge, t_crit):
    """
    Computes resistance resurgence and decay with time bounds.
    Includes comprehensive numerical stability safeguards.

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
    base_suppression = S_0_safe * circuit_breaker.safe_exp(-lambda_decay_safe * t_safe)
    resurgence = 0.0
    if t > t_crit:
        time_diff = min(500.0, t - t_crit)
        resurgence_exp = circuit_breaker.safe_exp(-mu_resurge_safe * time_diff)
        resurgence = alpha_resurge_safe * resurgence_exp
        if time_diff > 100.0:
            damping_factor = 1.0 - (time_diff - 100.0) / 900.0
            damping_factor = max(0.1, damping_factor)
            resurgence *= damping_factor
    result = max(0.0, base_suppression + resurgence)
    return circuit_breaker.check_and_fix(result)