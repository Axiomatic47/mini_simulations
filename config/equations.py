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
    return A / (1 + (T ** 2 / T_max ** 2))


# 4. Identity Binding Equation (Nuclear analogy)
def identity_binding(K_critical, lambda_decay, distance):
    """
    Computes identity binding strength.

    Parameters:
        K_critical (float): Cohesion threshold
        lambda_decay (float): Decay constant
        distance (float): Distance between identities

    Returns:
        float: Identity binding force
    """
    return -K_critical * np.exp(-lambda_decay * distance)


# 5. Suppression & Resistance Feedback Loops
def suppression_feedback(alpha, S, beta, K):
    """
    Computes suppression feedback dynamics.

    Parameters:
        alpha (float): Suppression reinforcement coefficient
        S (float): Current suppression level
        beta (float): Knowledge disruption coefficient
        K (float): Current knowledge level

    Returns:
        float: Feedback affecting suppression
    """
    return alpha * S - beta * K


# 6. Intelligence Entropy Function (Thermodynamic analogy)
def intelligence_entropy(S_0, lambda_decay, T):
    """
    Computes intelligence entropy based on truth adoption.

    Parameters:
        S_0 (float): Initial suppression
        lambda_decay (float): Suppression decay factor
        T (float): Truth adoption level

    Returns:
        float: Intelligence entropy (suppression)
    """
    return S_0 * np.exp(-lambda_decay * T)


# 7. Civilization Oscillation Model (Damped Wave)
def civilization_oscillation(E, dE_dt, gamma, omega):
    """
    Models oscillations between hierarchical and egalitarian states.

    Parameters:
        E (float): Egalitarian state
        dE_dt (float): First derivative of egalitarian state
        gamma (float): Damping factor (suppression)
        omega (float): Natural frequency

    Returns:
        float: Second derivative (acceleration) of civilization state
    """
    return -gamma * dE_dt - (omega ** 2) * E


# 8. Resistance Resurgence Model
def resistance_resurgence(S_0, lambda_decay, t, alpha_resurge, mu_resurge, t_crit):
    """
    Models delayed resurgence in suppression.

    Parameters:
        S_0 (float): Initial suppression level
        lambda_decay (float): Decay constant
        t (float): Current time step
        alpha_resurge (float): Resurgence intensity
        mu_resurge (float): Decay rate of resurgence
        t_crit (float): Critical time of resurgence

    Returns:
        float: Total suppression including resurgence
    """
    resurgence = alpha_resurge * np.exp(-mu_resurge * (t - t_crit)) if t > t_crit else 0
    return S_0 * np.exp(-lambda_decay * t) + resurgence


# 9. Phase Transition for Knowledge Growth
def knowledge_growth_phase_transition(K_0, beta_decay, t, A, gamma, T, T_crit):
    """
    Computes knowledge growth with phase transition behavior.

    Parameters:
        K_0 (float): Initial knowledge
        beta_decay (float): Knowledge decay rate
        t (float): Time step
        A (float): Knowledge growth amplitude
        gamma (float): Phase transition sharpness
        T (float): Truth adoption level
        T_crit (float): Critical threshold for transition

    Returns:
        float: Knowledge level at time t
    """
    return K_0 * np.exp(-beta_decay * t) + A * (1 - np.exp(-gamma * (T - T_crit)))


# 10. Wisdom as a Guiding Electromagnetic Field
def wisdom_field(W_0, alpha, S, R, K):
    """
    Calculates wisdom’s guiding effect on knowledge integration.

    Parameters:
        W_0 (float): Base wisdom level
        alpha (float): Suppression impact on wisdom
        S (float): Suppression level
        R (float): Resistance level
        K (float): Knowledge level

    Returns:
        float: Adjusted wisdom level
    """
    return W_0 * np.exp(-alpha * S) * (1 + (R / K))
