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