"""
Quantum Mechanics Physics Domain

This module contains functions that model quantum phenomena including
tunneling, entanglement, and other quantum effects in knowledge systems.
"""

from .quantum_entanglement_correlation import quantum_entanglement_correlation
from .build_entanglement_network import build_entanglement_network
from .quantum_tunneling_probability import quantum_tunneling_probability

__all__ = [
    'quantum_entanglement_correlation',
    'build_entanglement_network',
    'quantum_tunneling_probability',
]