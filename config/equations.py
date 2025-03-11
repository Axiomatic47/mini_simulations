import numpy as np


# 1. Intelligence Growth Equation with Saturation
def intelligence_growth(K, W, R, S, N, K_max=100.0):
    """
    Computes intelligence growth with saturation to prevent unbounded growth.

    Parameters:
        K (float): Knowledge level
        W (float): Wisdom factor (knowledge integration efficiency)
        R (float): Resistance level
        S (float): Suppression level
        N (float): Network effect (mutual learning contribution)
        K_max (float): Maximum knowledge capacity (prevents unbounded growth)

    Returns:
        float: Intelligence growth rate
    """
    # Apply saturation term to prevent unbounded growth
    # Cap knowledge to prevent overflow
    K_safe = min(K_max, max(0.0, K))

    # Apply safe bounds to other parameters
    W_safe = min(10.0, max(0.0, W))
    R_safe = min(100.0, max(0.0, R))
    S_safe = min(100.0, max(0.0, S))
    N_safe = min(10.0, max(-10.0, N))

    # Use saturation term in denominator to limit growth
    return K_safe * W_safe / (1.0 + K_safe / K_max) - R_safe - S_safe + N_safe


# 2. Free Will Decision Function with Bounded Output
def free_will_decision(q_Id, E_K, q_R, E_F):
    """
    Calculates the net decision force with bounded output using hyperbolic tangent.

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
    return np.tanh(raw_force)


# 3. Truth Adoption Model with Additional Damping
def truth_adoption(T, A, T_max):
    """
    Computes the rate of truth adoption with a relativistic limit and additional damping.

    Parameters:
        T (float): Current truth adoption level
        A (float): Adoption acceleration factor
        T_max (float): Maximum theoretical truth adoption limit

    Returns:
        float: Rate of truth adoption
    """
    # Ensure parameters are within safe bounds
    T = min(T_max, max(0.0, T))
    A = min(10.0, max(0.0, A))
    T_max = max(1.0, T_max)  # Ensure T_max is positive

    # Apply relativistic limit with additional damping factor
    return A / (1.0 + (T / T_max) ** 2) * (1.0 - T / T_max)


# 4. Wisdom Field with Numerical Safeguards
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
    # Apply safe bounds to all parameters
    W_0_safe = min(10.0, max(0.01, W_0))
    alpha_safe = min(1.0, max(0.0, alpha))
    S_safe = min(100.0, max(0.0, S))
    R_safe = min(100.0, max(0.0, R))
    K_safe = max(0.001, K)  # Prevent division by zero

    # Apply wisdom calculation with safety caps
    return W_0_safe * np.exp(-alpha_safe * S_safe) * (1.0 + min(R_safe, 10.0) / K_safe)


# 5. Resistance Resurgence with Time Bounds
def resistance_resurgence(S_0, lambda_decay, t, alpha_resurge, mu_resurge, t_crit):
    """
    Computes resistance resurgence and decay with time bounds.

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
    t_safe = min(1000.0, max(0.0, t))  # Cap time to prevent overflow in exponential
    alpha_resurge_safe = min(20.0, max(0.0, alpha_resurge))
    mu_resurge_safe = min(1.0, max(0.0001, mu_resurge))

    # Base exponential decay with time limit
    base_suppression = S_0_safe * np.exp(-lambda_decay_safe * t_safe)

    # Resurgence after critical time
    resurgence = 0.0
    if t > t_crit:
        # Cap the time difference to prevent overflow
        time_diff = min(500.0, t - t_crit)
        resurgence = alpha_resurge_safe * np.exp(-mu_resurge_safe * time_diff)

    # Combine effects with a minimum bound
    return max(0.0, base_suppression + resurgence)


# 6. Suppression Feedback with Parameter Bounds
def suppression_feedback(alpha, S, beta, K):
    """
    Computes suppression feedback with bounded parameters.

    Parameters:
        alpha (float): Suppression reinforcement coefficient
        S (float): Current suppression level
        beta (float): Knowledge disruption coefficient
        K (float): Current knowledge level

    Returns:
        float: Suppression feedback effect
    """
    # Apply safe bounds to all parameters
    alpha_safe = min(1.0, max(0.0, alpha))
    S_safe = min(100.0, max(0.0, S))
    beta_safe = min(1.0, max(0.0, beta))
    K_safe = min(1000.0, max(0.0, K))

    # Calculate bounded feedback
    return alpha_safe * S_safe - beta_safe * K_safe


# 7. Civilization Oscillation as First-Order System
class CivilizationOscillator:
    """
    Class implementing a first-order system for civilization oscillation.
    This is more numerically stable than the second-order implementation.
    """

    def __init__(self, E_0=0.0, V_0=0.0):
        """Initialize the oscillator with starting values."""
        self.E = E_0  # Egalitarian state
        self.V = V_0  # Rate of change of state

    def update(self, gamma, omega, dt=1.0):
        """
        Update the oscillator by one time step.

        Parameters:
            gamma (float): Damping factor
            omega (float): Natural oscillation frequency
            dt (float): Time step size

        Returns:
            tuple: Current state and velocity (E, V)
        """
        # Apply safe bounds to parameters
        gamma_safe = min(1.0, max(0.0, gamma))
        omega_safe = min(1.0, max(0.0, omega))
        dt_safe = min(1.0, max(0.001, dt))

        # Calculate acceleration (second derivative)
        accel = -gamma_safe * self.V - (omega_safe ** 2) * self.E

        # Update velocity (first derivative)
        self.V += accel * dt_safe

        # Update position (state)
        self.E += self.V * dt_safe

        return self.E, self.V


# Legacy function for backward compatibility
def civilization_oscillation(E, dE_dt, gamma, omega):
    """
    Legacy function for civilization oscillation (second-order ODE).
    Consider using the CivilizationOscillator class for better stability.

    Parameters:
        E (float): Current civilization state
        dE_dt (float): Rate of change of state
        gamma (float): Damping factor
        omega (float): Natural oscillation frequency

    Returns:
        float: Acceleration of civilization state
    """
    # Apply safe bounds to parameters
    gamma_safe = min(1.0, max(0.0, gamma))
    omega_safe = min(1.0, max(0.0, omega))
    E_safe = min(10.0, max(-10.0, E))
    dE_dt_safe = min(10.0, max(-10.0, dE_dt))

    # Damped harmonic oscillator equation (2nd order ODE)
    return -gamma_safe * dE_dt_safe - (omega_safe ** 2) * E_safe


# 8. Knowledge Growth Phase Transition with Sigmoid
def knowledge_growth_phase_transition(K_0, beta_decay, t, A, gamma, T, T_crit):
    """
    Computes knowledge growth with smooth sigmoid phase transition.

    Parameters:
        K_0 (float): Base knowledge level
        beta_decay (float): Knowledge decay rate
        t (float): Time step
        A (float): Growth amplitude
        gamma (float): Transition sharpness
        T (float): Truth adoption level
        T_crit (float): Critical threshold for transition

    Returns:
        float: Knowledge level after phase transition
    """
    # Apply safe bounds to all parameters
    K_0_safe = min(100.0, max(0.0, K_0))
    beta_decay_safe = min(0.5, max(0.0, beta_decay))
    t_safe = min(1000.0, max(0.0, t))  # Cap time to prevent overflow
    A_safe = min(10.0, max(0.0, A))
    gamma_safe = min(1.0, max(0.0, gamma))
    T_safe = min(100.0, max(0.0, T))
    T_crit_safe = min(100.0, max(0.0, T_crit))

    # Potential knowledge decay in suppressed state (with time limit)
    decay_term = K_0_safe * np.exp(-beta_decay_safe * min(500.0, t_safe))

    # Phase transition using sigmoid function (smoother than exponential)
    sigmoid = 1.0 / (1.0 + np.exp(-gamma_safe * (T_safe - T_crit_safe)))
    growth_term = A_safe * sigmoid

    # Combine effects (ensuring knowledge is non-negative)
    return max(0.0, decay_term + growth_term)