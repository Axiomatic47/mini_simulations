"""
Module Mapper

This module maps functions to their physics domains based on directory structure
and provides utilities for analyzing the organization of equation functions.
"""

import os
import sys
import inspect
import logging
from pathlib import Path
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhysicsDomain(Enum):
    """Enumeration of physics domains used in the simulation."""
    THERMODYNAMICS = "thermodynamics"
    RELATIVITY = "relativity"
    ELECTROMAGNETISM = "electromagnetism"
    STRONG_NUCLEAR = "strong_nuclear"
    WEAK_NUCLEAR = "weak_nuclear"
    QUANTUM_MECHANICS = "quantum_mechanics"
    ASTROPHYSICS = "astrophysics"
    MULTI_SYSTEM = "multi_system"
    UTILS = "utils"  # For utility functions
    BRIDGE = "bridge"  # For bridge functions connecting domains
    UNKNOWN = "unknown"

def map_function_to_domain():
    """
    Maps functions to their physics domains based on directory structure.

    Returns:
        dict: Dictionary mapping function names to domain names
    """
    function_map = {}

    # Get the base path for physics_domains
    base_path = Path(__file__).resolve().parent / "physics_domains"

    if not base_path.exists():
        logger.warning(f"Physics domains directory not found at {base_path}")
        return _get_fallback_function_map()

    # Iterate through each domain directory
    for domain_dir in base_path.iterdir():
        if domain_dir.is_dir() and not domain_dir.name.startswith('__'):
            domain_name = domain_dir.name.upper()

            # Iterate through each Python file in the domain directory
            for py_file in domain_dir.glob('*.py'):
                if py_file.name.startswith('__'):
                    continue

                # Module name is the filename without extension
                module_name = py_file.stem

                # Add to the function map - the function name is typically the same as the file name
                function_map[module_name] = domain_name

    # Add special functions from bridge modules
    try:
        bridge_path = Path(__file__).resolve().parent / "bridge_functions"
        if bridge_path.exists():
            for py_file in bridge_path.glob('*.py'):
                if py_file.name.startswith('__'):
                    continue

                module_name = py_file.stem
                function_map[module_name] = "BRIDGE"
        else:
            logger.warning(f"Bridge functions directory not found at {bridge_path}")
    except Exception as e:
        logger.warning(f"Error processing bridge functions: {e}")

    # Add fallback mappings for functions that might not follow the filename pattern
    fallback_mappings = _get_fallback_function_map()

    # Merge the discovered mappings with the fallback mappings
    # Discovered mappings take precedence
    merged_map = {**fallback_mappings, **function_map}

    return merged_map

def _get_fallback_function_map():
    """
    Provides a fallback mapping of functions to domains when directory scanning fails.

    Returns:
        dict: Manual mapping of function names to domains
    """
    return {
        # Thermodynamics domain
        "intelligence_growth": "THERMODYNAMICS",
        "knowledge_growth_phase_transition": "THERMODYNAMICS",

        # Relativity domain
        "truth_adoption": "RELATIVITY",

        # Electromagnetism domain
        "wisdom_field": "ELECTROMAGNETISM",
        "wisdom_field_enhanced": "ELECTROMAGNETISM",
        "free_will_decision": "ELECTROMAGNETISM",
        "knowledge_field_gradient": "ELECTROMAGNETISM",
        "knowledge_field_influence": "ELECTROMAGNETISM",

        # Strong nuclear domain
        "civilization_oscillation": "STRONG_NUCLEAR",
        "identity_binding": "STRONG_NUCLEAR",

        # Weak nuclear domain
        "suppression_feedback": "WEAK_NUCLEAR",
        "resistance_resurgence": "WEAK_NUCLEAR",

        # Quantum mechanics domain
        "build_entanglement_network": "QUANTUM_MECHANICS",
        "quantum_tunneling_probability": "QUANTUM_MECHANICS",
        "quantum_entanglement_correlation": "QUANTUM_MECHANICS",

        # Astrophysics domain
        "suppression_event_horizon": "ASTROPHYSICS",
        "civilization_lifecycle_phase": "ASTROPHYSICS",
        "cosmic_background_knowledge": "ASTROPHYSICS",
        "dark_energy_knowledge_acceleration": "ASTROPHYSICS",
        "galactic_structure_model": "ASTROPHYSICS",
        "knowledge_inflation": "ASTROPHYSICS",
        "knowledge_gravitational_lensing": "ASTROPHYSICS",

        # Multi-system domain
        "calculate_distance_matrix": "MULTI_SYSTEM",
        "safe_calculate_distance_matrix": "MULTI_SYSTEM",
        "initialize_civilizations": "MULTI_SYSTEM",
        "spawn_new_civilization": "MULTI_SYSTEM",
        "cultural_influence": "MULTI_SYSTEM",
        "update_civilization_sizes": "MULTI_SYSTEM",
        "calculate_interaction_strength": "MULTI_SYSTEM",
        "civilization_movement": "MULTI_SYSTEM",
        "detect_civilization_collapse": "MULTI_SYSTEM",
        "detect_civilization_mergers": "MULTI_SYSTEM",
        "knowledge_diffusion": "MULTI_SYSTEM",
        "process_all_civilization_interactions": "MULTI_SYSTEM",
        "safe_process_civilization_interactions": "MULTI_SYSTEM",
        "process_civilization_merger": "MULTI_SYSTEM",
        "remove_civilization": "MULTI_SYSTEM",
        "resource_competition": "MULTI_SYSTEM",
        "galactic_collision_effect": "MULTI_SYSTEM",

        # Utility functions
        "safe_exp": "UTILS",
        "safe_div": "UTILS",
        "safe_sqrt": "UTILS",
        "safe_log": "UTILS",
    }

def get_functions_by_domain():
    """
    Group functions by their physics domain.

    Returns:
        dict: Dictionary mapping domain names to lists of function names
    """
    domain_functions = {}
    function_map = map_function_to_domain()

    for func_name, domain in function_map.items():
        if domain not in domain_functions:
            domain_functions[domain] = []
        domain_functions[domain].append(func_name)

    return domain_functions

def print_function_map():
    """Print the function map grouped by domain."""
    domain_functions = get_functions_by_domain()

    print("\n==== Physics Domain Function Map ====\n")

    total_functions = 0
    for domain, functions in sorted(domain_functions.items()):
        print(f"{domain} ({len(functions)} functions):")
        for func in sorted(functions):
            print(f"  - {func}")
        total_functions += len(functions)
        print()

    print(f"Total discovered functions: {total_functions}")

def get_domain_for_function(func_name):
    """
    Get the physics domain for a specific function.

    Args:
        func_name (str): Name of the function

    Returns:
        str: Name of the physics domain, or None if not found
    """
    function_map = map_function_to_domain()
    return function_map.get(func_name, "UNKNOWN")

def load_function_from_domain(func_name, default_func=None):
    """
    Attempt to load a function from its physics domain.

    Args:
        func_name (str): Name of the function to load
        default_func: Default function to return if loading fails

    Returns:
        function: The loaded function, or default_func if loading fails
    """
    function_map = map_function_to_domain()

    # Skip if function name is not mapped
    if func_name not in function_map:
        logger.warning(f"Unknown domain for function {func_name}")
        return default_func

    domain = function_map[func_name].lower()

    # Try to import from physics domains
    try:
        # Form import path
        module_path = f"physics_domains.{domain}.{func_name}"

        # Dynamically import
        module_parts = module_path.split('.')
        imported = __import__(module_path, fromlist=[module_parts[-1]])

        # Get the function
        if hasattr(imported, func_name):
            return getattr(imported, func_name)
    except ImportError:
        # Try importing from config as fallback
        try:
            # Form fallback import path
            if domain in ["thermodynamics", "relativity", "electromagnetism", "strong_nuclear", "weak_nuclear"]:
                # Core equations
                from config.equations import func_name
                return func_name
            elif domain == "astrophysics":
                # Astrophysics extensions
                from config.astrophysics_extensions import func_name
                return func_name
            elif domain == "multi_system":
                # Multi-civilization extensions
                from config.multi_civilization_extensions import func_name
                return func_name
            elif domain == "quantum_mechanics":
                # Quantum extensions
                from config.quantum_em_extensions import func_name
                return func_name
        except (ImportError, AttributeError):
            logger.warning(f"Could not import {func_name} from any location")

    return default_func

if __name__ == "__main__":
    # Print the function map when the script is run directly
    print_function_map()