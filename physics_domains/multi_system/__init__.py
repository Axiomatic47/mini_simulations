# physics_domains/multi_system/__init__.py

from .process_civilization_merger import process_civilization_merger
from .calculate_distance_matrix import calculate_distance_matrix
from .initialize_civilizations import initialize_civilizations
from .spawn_new_civilization import spawn_new_civilization
from .cultural_influence import cultural_influence
from .update_civilization_sizes import update_civilization_sizes
from .process_all_civilization_interactions import process_all_civilization_interactions
from .galactic_collision_effect import galactic_collision_effect
from .remove_civilization import remove_civilization
from .detect_civilization_mergers import detect_civilization_mergers
from .knowledge_diffusion import knowledge_diffusion
from .civilization_movement import civilization_movement
from .resource_competition import resource_competition
from .calculate_interaction_strength import calculate_interaction_strength
from .detect_civilization_collapse import detect_civilization_collapse

# Export all functions
__all__ = [
    'process_civilization_merger',
    'calculate_distance_matrix',
    'initialize_civilizations',
    'spawn_new_civilization',
    'cultural_influence',
    'update_civilization_sizes',
    'process_all_civilization_interactions',
    'galactic_collision_effect',
    'remove_civilization',
    'detect_civilization_mergers',
    'knowledge_diffusion',
    'civilization_movement',
    'resource_competition',
    'calculate_interaction_strength',
    'detect_civilization_collapse',
]
