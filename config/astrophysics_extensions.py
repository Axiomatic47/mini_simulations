"""
Astrophysics Extensions

This module imports and re-exports astrophysics-related functions from the physics_domains package.
This file maintains backward compatibility with existing code that imports from config.
"""

from physics_domains.astrophysics import (
    civilization_lifecycle_phase,
    suppression_event_horizon,
    cosmic_background_knowledge,
    knowledge_inflation,
    knowledge_gravitational_lensing,
    dark_energy_knowledge_acceleration,
    galactic_structure_model,
)

# Re-export all functions for backward compatibility
__all__ = [
    'civilization_lifecycle_phase',
    'suppression_event_horizon',
    'cosmic_background_knowledge',
    'knowledge_inflation',
    'knowledge_gravitational_lensing',
    'dark_energy_knowledge_acceleration',
    'galactic_structure_model',
]