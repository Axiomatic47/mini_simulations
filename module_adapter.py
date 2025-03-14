"""
Module adapter for bridging between physics_domains and config modules.
This allows code that expects equations in config to find them in physics_domains.
"""

import importlib
import inspect
import logging
from pathlib import Path

logger = logging.getLogger("ModuleAdapter")


def get_equation_function(function_name):
    """
    Find a function either in physics_domains or config.

    Args:
        function_name: Name of the function to find

    Returns:
        The function if found, or a dummy function
    """
    # Try to import from module_map if available
    try:
        from module_map import find_function, map_function_to_domain
        domain, module_name, function = find_function(function_name)
        if function:
            logger.info(f"Found {function_name} in {module_name}")
            return function
    except ImportError:
        logger.warning("module_map not available, using direct imports")

    # List of possible module locations
    module_locations = [
        # Physics domains
        f"physics_domains.thermodynamics.{function_name}",
        f"physics_domains.relativity.{function_name}",
        f"physics_domains.electromagnetism.{function_name}",
        f"physics_domains.quantum_mechanics.{function_name}",
        f"physics_domains.weak_nuclear.{function_name}",
        f"physics_domains.strong_nuclear.{function_name}",
        f"physics_domains.astrophysics.{function_name}",
        f"physics_domains.multi_system.{function_name}",

        # Config modules
        "config.equations",
        "config.astrophysics_extensions",
        "config.quantum_em_extensions",
        "config.multi_civilization_extensions"
    ]

    # Try to import from each location
    for module_path in module_locations:
        try:
            # For physics_domains, the function name is part of the module path
            if module_path.count('.') == 2:
                module = importlib.import_module(module_path)
                if hasattr(module, function_name):
                    return getattr(module, function_name)
            # For config modules, try to find the function within the module
            else:
                module = importlib.import_module(module_path)
                if hasattr(module, function_name):
                    return getattr(module, function_name)
        except ImportError:
            continue

    # Create a dummy function if not found
    logger.warning(f"Function {function_name} not found, using dummy")

    def dummy_function(*args, **kwargs):
        logger.warning(f"Using dummy implementation of {function_name}")
        return 1.0

    return dummy_function


def get_simulation_function(simulation_name):
    """
    Get a simulation function by name.

    Args:
        simulation_name: Name of the simulation

    Returns:
        The simulation function if found, or a dummy function
    """
    try:
        module = importlib.import_module(f"simulations.{simulation_name}")

        # Check for run_simulation function
        if hasattr(module, "run_simulation"):
            return getattr(module, "run_simulation")

        # Look for alternative function names
        for name in dir(module):
            func = getattr(module, name)
            if callable(func) and ("run" in name.lower() or "simulation" in name.lower()):
                return func
    except ImportError:
        logger.warning(f"Simulation module {simulation_name} not found")

    # Return dummy function
    import numpy as np

    def dummy_simulation(*args, **kwargs):
        logger.warning(f"Using dummy {simulation_name} simulation")
        timesteps = 100
        return {
            'time': np.arange(timesteps),
            'knowledge': np.random.rand(timesteps) * 10,
            'suppression': np.random.rand(timesteps) * 5,
            'intelligence': np.random.rand(timesteps) * 15
        }

    return dummy_simulation