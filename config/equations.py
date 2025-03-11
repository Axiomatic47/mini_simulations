import numpy as np


# 1. Intelligence Growth Equation
def intelligence_growth(K, W, R, S, N):
    """
    Computes intelligence growth.

    Parameters:
        K (float): Knowledge level
        W (float): Wisdom factor (knowledge integration efficiency)
        R (float): Resistance level
        S (float): Suppression level
        N (float): Network effect (mutual learning contribution)

    Returns:
        float: Intelligence growth rate
    """
    return K * W - R - S + N


# 2. Free Will Decision Function (Electromagnetic analogy)
def free_will_decision(q_Id, E_K, q_R, E_F):
    """
    Calculates the net decision force based on identity, knowledge, resistance, and fear.

    Parameters:
        q_Id (float): Identity bias charge
        E_K (float): Knowledge field strength (decision clarity)
        q_R (float): Resistance charge
        E_F (float): Fear-driven field strength

    Returns:
        float: Net decision force (positive towards knowledge-based decisions)
    """
    return q_Id * E_K - q_R * E_F


# 3. Truth Adoption Model (Relativistic analogy)
def truth_adoption(T, A, T_max):
    """
    Computes the rate of truth adoption with a relativistic limit.

    Parameters:
        T (float): Current truth adoption level
        A (float): Adoption acceleration factor
        T_max (float): Maximum theoretical truth adoption limit

    Returns:
        float: Rate of truth adoption
    """
    # Ensure T doesn't exceed T_max
    T = min(T, T_max)

    # Apply relativistic limit - adoption rate approaches zero as T approaches T_max
    return A / (1 + (T ** 2 / T_max ** 2))


# 4. Wisdom Field (Electromagnetic field analogy)
def wisdom_field(W_0, alpha, S, R, K):
    """
    Computes wisdom field strength based on knowledge and suppression.

    Parameters:
        W_0 (float): Base wisdom level
        alpha (float): Suppression impact factor
        S (float): Suppression level
        R (float): Resistance level
        K (float): Knowledge level

    Returns:
        float: Wisdom field strength
    """
    # Ensure K is not zero to avoid division issues
    K = max(K, 0.001)

    # Wisdom decreases with suppression and resistance-to-knowledge ratio
    return W_0 * np.exp(-alpha * S) * (1 + R / K)


# 5. Resistance Resurgence (Nuclear decay analogy)
def resistance_resurgence(S_0, lambda_decay, t, alpha_resurge, mu_resurge, t_crit):
    """
    Computes resistance resurgence and decay.

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
    # Base exponential decay
    base_suppression = S_0 * np.exp(-lambda_decay * t)

    # Resurgence after critical time
    resurgence = 0
    if t > t_crit:
        resurgence = alpha_resurge * np.exp(-mu_resurge * (t - t_crit))

    return base_suppression + resurgence


# 6. Suppression Feedback (Weak nuclear force analogy)
def suppression_feedback(alpha, S, beta, K):
    """
    Computes suppression feedback based on current suppression and knowledge.

    Parameters:
        alpha (float): Suppression reinforcement coefficient
        S (float): Current suppression level
        beta (float): Knowledge disruption coefficient
        K (float): Current knowledge level

    Returns:
        float: Suppression feedback effect
    """
    return alpha * S - beta * K


# 7. Civilization Oscillation (Quantum neutrino oscillation analogy)
def civilization_oscillation(E, dE_dt, gamma, omega):
    """
    Computes civilization oscillation (egalitarian vs. hierarchical).

    Parameters:
        E (float): Current civilization state
        dE_dt (float): Rate of change of state
        gamma (float): Damping factor
        omega (float): Natural oscillation frequency

    Returns:
        float: Acceleration of civilization state
    """
    # Damped harmonic oscillator equation (2nd order ODE)
    return -gamma * dE_dt - (omega ** 2) * E


# 8. Knowledge Growth Phase Transition (Weak nuclear transformation analogy)
def knowledge_growth_phase_transition(K_0, beta_decay, t, A, gamma, T, T_crit):
    """
    Computes knowledge growth with phase transition behavior.

    Parameters:
        K_0 (float): Base knowledge level
        beta_decay (float): Knowledge decay rate
        t (float): Time step
        A (float): Growth amplitude
        gamma (float): Transition sharpness
        T (float): Truth adoption level
        T_crit (float): Critical threshold for phase transition

    Returns:
        float: Knowledge level after phase transition
    """
    # Potential knowledge decay in suppressed state
    decay_term = K_0 * np.exp(-beta_decay * t)

    # Phase transition to rapid growth after threshold
    # When T exceeds T_crit, growth accelerates
    growth_term = A * (1 - np.exp(-gamma * (T - T_crit)))

    # Combine effects (ensuring knowledge is non-negative)
    return max(0, decay_term + growth_term)