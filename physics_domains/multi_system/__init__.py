"""
Multi-System Physics Domain

This module contains functions for modeling interactions between multiple
civilization systems, including knowledge diffusion, cultural influence,
resource competition, and other multi-civilization dynamics.
"""

from .initialize_civilizations import initialize_civilizations
from .calculate_distance_matrix import calculate_distance_matrix
from .calculate_interaction_strength import calculate_interaction_strength
from .galactic_collision_effect import galactic_collision_effect
from .knowledge_diffusion import knowledge_diffusion
from .cultural_influence import cultural_influence
from .resource_competition import resource_competition
from .civilization_movement import civilization_movement
from .update_civilization_sizes import update_civilization_sizes
from .detect_civilization_collapse import detect_civilization_collapse
from .detect_civilization_mergers import detect_civilization_mergers
from .process_civilization_merger import process_civilization_merger
from .spawn_new_civilization import spawn_new_civilization
from .remove_civilization import remove_civilization
from .process_all_civilization_interactions import process_all_civilization_interactions

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
]