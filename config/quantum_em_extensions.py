"""
Quantum and electromagnetic extensions for modeling quantum phenomena
and field interactions in knowledge systems.

This module imports and re-exports quantum and electromagnetic functions
from the physics_domains packages for backward compatibility.
"""

# Import all functions from the physics domains
from physics_domains.electromagnetism import (
    knowledge_field_influence,
    knowledge_field_gradient
)

from physics_domains.quantum_mechanics import (
    quantum_entanglement_correlation,
    build_entanglement_network,
    quantum_tunneling_probability
)

# Re-export all functions for backward compatibility
__all__ = [
    # Electromagnetism
    'knowledge_field_influence',
    'knowledge_field_gradient',

    # Quantum Mechanics
    'quantum_entanglement_correlation',
    'build_entanglement_network',
    'quantum_tunneling_probability',
]