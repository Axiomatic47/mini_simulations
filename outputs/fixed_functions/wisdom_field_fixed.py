def wisdom_field(W_0, alpha, S, R, K):
    """
    Computes wisdom field strength with numerical safeguards.

    Parameters:
        W_0 (float): Base wisdom level
        alpha (float): Suppression impact factor
        S (float): Suppression level
        R (float): Resistance level
        K (float): Knowledge level

    Returns:
        float: Wisdom field strength
    """
    W_0_safe = min(10.0, max(0.01, W_0))
    alpha_safe = min(1.0, max(0.0, alpha))
    S_safe = min(100.0, max(0.0, S))
    R_safe = min(100.0, max(0.0, R))
    K_safe = max(0.001, K)
    return W_0_safe * np.exp(min(50, -alpha_safe * S_safe)) * (1.0 + min(R_safe, 10.0) / max(1e-10, K_safe))