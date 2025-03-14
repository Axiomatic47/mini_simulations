"""
Multi-civilization extensions for modeling interactions between multiple civilization systems.

This module imports and re-exports multi-civilization related functions from
the physics_domains.multi_system package for backward compatibility.
"""

from utils.dim_handler import DimensionHandler, safe_calculate_distance_matrix, safe_process_civilization_interactions

# Create a module-level instance
dim_handler = DimensionHandler(verbose=True, auto_fix=True)

# Import all functions from the physics domain
from physics_domains.multi_system import (
    initialize_civilizations,
    calculate_distance_matrix,
    calculate_interaction_strength,
    galactic_collision_effect,
    knowledge_diffusion,
    cultural_influence,
    resource_competition,
    civilization_movement,
    update_civilization_sizes,
    detect_civilization_collapse,
    detect_civilization_mergers,
    process_civilization_merger,
    spawn_new_civilization,
    remove_civilization,
    process_all_civilization_interactions
)

# Re-export all functions for backward compatibility
__all__ = [
    'initialize_civilizations',
    'calculate_distance_matrix',
    'calculate_interaction_strength',
    'galactic_collision_effect',
    'knowledge_diffusion',
    'cultural_influence',
    'resource_competition',
    'civilization_movement',
    'update_civilization_sizes',
    'detect_civilization_collapse',
    'detect_civilization_mergers',
    'process_civilization_merger',
    'spawn_new_civilization',
    'remove_civilization',
    'process_all_civilization_interactions',
    'dim_handler',  # Also export the module-level dimension handler
]