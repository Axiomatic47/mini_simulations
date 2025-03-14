# physics_domains/strong_nuclear/civilization_oscillation.py
import numpy as np


def civilization_oscillation(t, K, gamma_osc=0.005, omega_osc=0.3, amplitude=1.0):
    """
    Models oscillation patterns in civilization development.

    Physics Domain: strong_nuclear
    Scale Level: civilization
    Application Domains: cycles, development

    Parameters:
        t (float): Time variable
        K (float): Knowledge level
        gamma_osc (float): Oscillation damping factor
        omega_osc (float): Natural oscillation frequency
        amplitude (float): Base oscillation amplitude

    Returns:
        float: Oscillation contribution to system
    """
    # Bounded input to prevent overflow
    safe_t = min(1000.0, max(0.0, t))

    # Knowledge-dependent amplitude that decreases as knowledge grows
    k_amplitude = amplitude / (1.0 + 0.01 * K)

    # Calculate damped oscillation
    oscillation = k_amplitude * np.exp(-gamma_osc * safe_t) * np.cos(omega_osc * safe_t)

    return oscillation