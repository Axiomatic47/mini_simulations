"""
Knowledge Growth Phase Transition Module

This module models the phase transition in knowledge growth using a sigmoid function
to represent transitions between different growth regimes.
"""

import numpy as np



def knowledge_growth_phase_transition(K, K_0, a, gamma, t, t_crit, *args):
    '''
    Calculate knowledge growth rate with phase transition effect.
    This function accepts extra arguments but only uses the first 6.

    Parameters:
        K (float): Current knowledge level
        K_0 (float): Base knowledge growth rate
        a (float): Amplitude of the transition effect
        gamma (float): Steepness of the transition
        t (float): Current time
        t_crit (float): Critical time for phase transition
        *args: Additional arguments (ignored)

    Returns:
        float: The adjusted growth rate
    '''
    import numpy as np
    # Simple sigmoid transition
    transition = 1.0 / (1.0 + np.exp(-gamma * (t - t_crit)))
    growth_rate = K_0 + a * transition
    return growth_rate
