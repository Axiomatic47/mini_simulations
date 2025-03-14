"""
Stabilized versions of core equations with enhanced numerical stability safeguards.
These versions incorporate circuit breaker integration, safe bounds, and protection
against common numerical issues like division by zero and overflow.
All functions feature smooth transitions at critical thresholds.

This module imports and re-exports functions from their respective physics domains
to maintain backward compatibility.
"""

# Import all functions from their respective physics domains
from physics_domains.thermodynamics import intelligence_growth
from physics_domains.electromagnetism import free_will_decision, wisdom_field, wisdom_field_enhanced, knowledge_field_influence, knowledge_field_gradient
from physics_domains.relativity import truth_adoption
from physics_domains.weak_nuclear import resistance_resurgence, suppression_feedback
from physics_domains.quantum_mechanics import quantum_tunneling_probability

# Re-export all functions for backward compatibility
__all__ = [
    # Thermodynamics
    'intelligence_growth',

    # Electromagnetism
    'free_will_decision',
    'wisdom_field',
    'wisdom_field_enhanced',
    'knowledge_field_influence',
    'knowledge_field_gradient',

    # Relativity
    'truth_adoption',

    # Weak Nuclear
    'resistance_resurgence',
    'suppression_feedback',

    # Quantum Mechanics
    'quantum_tunneling_probability',
]

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
