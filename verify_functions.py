#!/usr/bin/env python
"""
Script to verify that functions can be discovered in the refactored physics domains.
This helps diagnose issues with the validation framework's function discovery mechanism.
"""

import importlib
import inspect
import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Add current directory to path to help with imports
sys.path.insert(0, os.getcwd())

# List of core functions to check
CORE_FUNCTIONS = [
    # Strong Nuclear
    "civilization_oscillation",
    "identity_binding",

    # Weak Nuclear
    "suppression_feedback",
    "resistance_resurgence",

    # Thermodynamics
    "intelligence_growth",
    "knowledge_growth_phase_transition",

    # Relativity
    "truth_adoption",

    # Electromagnetism
    "wisdom_field",
    "knowledge_field_influence",
    "free_will_decision",

    # Quantum Mechanics
    "quantum_tunneling_probability",
    "quantum_entanglement_correlation",

    # Astrophysics
    "civilization_lifecycle_phase",
    "suppression_event_horizon",
    "knowledge_gravitational_lensing",

    # Multi-system
    "knowledge_diffusion",
    "cultural_influence",
    "initialize_civilizations"
]

def try_module_map_import(func_name):
    """Try to import a function using the module map."""
    try:
        # Try to import module_map
        module_map = importlib.import_module("module_map")

        # Check if it has the necessary functions
        if hasattr(module_map, "load_function_from_domain"):
            func = module_map.load_function_from_domain(func_name)
            return func, True if func else False
        else:
            logger.warning("Module map not available or find_function not found")
            return None, False
    except ImportError:
        logger.warning("Could not import module_map")
        return None, False
    except Exception as e:
        logger.warning(f"Error trying to use module_map: {e}")
        return None, False

def try_direct_physics_domains_import(func_name):
    """Try to directly import from physics domains."""
    domains = [
        "thermodynamics", "relativity", "electromagnetism",
        "weak_nuclear", "strong_nuclear", "quantum_mechanics",
        "astrophysics", "multi_system"
    ]

    for domain in domains:
        try:
            module_name = f"physics_domains.{domain}.{func_name}"
            module = importlib.import_module(module_name)
            if hasattr(module, func_name):
                return getattr(module, func_name), module_name
        except ImportError:
            continue
        except Exception as e:
            logger.warning(f"Error trying to import {module_name}: {e}")

    return None, None

def try_config_import(func_name):
    """Try to import from the config modules."""
    try:
        # Try config.equations first
        try:
            module = importlib.import_module("config.equations")
            if hasattr(module, func_name):
                return getattr(module, func_name), "config.equations"
        except ImportError:
            pass

        # Try more specific config modules
        specific_modules = [
            "config.quantum_em_extensions",
            "config.astrophysics_extensions",
            "config.multi_civilization_extensions"
        ]

        for module_name in specific_modules:
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, func_name):
                    return getattr(module, func_name), module_name
            except ImportError:
                pass
    except Exception as e:
        logger.warning(f"Error trying to import {func_name} from config: {e}")

    return None, None

def try_validation_adapter_import(func_name):
    """Try to import using the validation adapter."""
    try:
        from validation_adapter import FunctionAdapter
        adapter = FunctionAdapter()
        func = adapter.get_physics_function(func_name)
        return func, "validation_adapter"
    except ImportError:
        logger.warning("Could not import validation_adapter")
        return None, None
    except Exception as e:
        logger.warning(f"Error trying to use validation_adapter: {e}")
        return None, None

def check_function_docstring(func, func_name):
    """Check if the function has the necessary docstring metadata."""
    issues = []

    if not func.__doc__:
        issues.append(f"{func_name}: Missing docstring")
        return issues

    # Check for required metadata
    if "Physics Domain:" not in func.__doc__:
        issues.append(f"{func_name}: Missing 'Physics Domain:' in docstring")

    if "Scale Level:" not in func.__doc__:
        issues.append(f"{func_name}: Missing 'Scale Level:' in docstring")

    if "Application Domains:" not in func.__doc__:
        issues.append(f"{func_name}: Missing 'Application Domains:' in docstring")

    return issues

def verify_function(func_name):
    """Verify that a function can be discovered and has proper docstring metadata."""
    logger.info(f"Checking {func_name}...")

    # Test all discovery methods

    # 1. Try using module map first
    func, found_with_map = try_module_map_import(func_name)
    if found_with_map:
        logger.info(f"  ✓ Found with module_map.load_function_from_domain")
    else:
        logger.info(f"  ✗ Not found with module_map")

    # 2. Try direct import from physics_domains
    direct_func, module_name = try_direct_physics_domains_import(func_name)
    if direct_func:
        logger.info(f"  ✓ Found with direct import in {module_name}")
    else:
        logger.info(f"  ✗ Not found with direct import from physics_domains")

    # 3. Try import from config
    config_func, config_module = try_config_import(func_name)
    if config_func:
        logger.info(f"  ✓ Found with import from {config_module}")
    else:
        logger.info(f"  ✗ Not found in config modules")

    # 4. Try using the validation adapter
    adapter_func, adapter_info = try_validation_adapter_import(func_name)
    if adapter_func:
        logger.info(f"  ✓ Found with validation adapter")
    else:
        logger.info(f"  ✗ Not found with validation adapter")

    # Return the first function found, in order of preference
    if found_with_map and func:
        return func
    if direct_func:
        return direct_func
    if config_func:
        return config_func
    if adapter_func:
        return adapter_func

    # If not found at all
    logger.info(f"  ! Function {func_name} not found with any method")
    return None

def main():
    """Main function that verifies the discoverability of core functions."""
    # Track function discovery stats
    found_funcs = {}
    not_found = []
    with_module_map = 0
    with_direct_import = 0
    with_config = 0
    with_adapter = 0

    # Check core functions
    for func_name in CORE_FUNCTIONS:
        # First check if module map can find it
        func, found_with_map = try_module_map_import(func_name)
        if found_with_map and func:
            found_funcs[func_name] = func
            with_module_map += 1
            continue

        # Then try direct import
        direct_func, _ = try_direct_physics_domains_import(func_name)
        if direct_func:
            found_funcs[func_name] = direct_func
            with_direct_import += 1
            continue

        # Then try config import
        config_func, _ = try_config_import(func_name)
        if config_func:
            found_funcs[func_name] = config_func
            with_config += 1
            continue

        # Finally try adapter
        adapter_func, _ = try_validation_adapter_import(func_name)
        if adapter_func:
            found_funcs[func_name] = adapter_func
            with_adapter += 1
            continue

        # If not found by any method
        not_found.append(func_name)

    # Print summary
    print("\nSummary:")
    print(f"Total functions checked: {len(CORE_FUNCTIONS)}")
    print(f"Found with module_map: {with_module_map}")
    print(f"Found with direct import: {with_direct_import}")
    print(f"Found with config import: {with_config}")
    print(f"Found with validation adapter: {with_adapter}")
    print(f"Not found: {len(not_found)}")

    if not_found:
        print("\nFunctions not found:")
        for func_name in not_found:
            print(f"  - {func_name}")

    # Check docstrings for found functions
    print("\nChecking docstrings for found functions...")
    all_issues = []

    for func_name, func in found_funcs.items():
        issues = check_function_docstring(func, func_name)
        all_issues.extend(issues)

    if all_issues:
        print("\nDocstring issues found:")
        for issue in all_issues:
            print(f"  - {issue}")
    else:
        print("No docstring issues found!")

    # Check for module map usage
    if with_module_map == 0:
        print("\nWarning: module_map.load_function_from_domain is not finding any functions")
        print("Recommendations:")
        print("  1. Ensure module_map.py is in your project directory")
        print("  2. Check if module_map.load_function_from_domain is implemented correctly")
        print("  3. Verify physics_domains directory structure")

    return 0 if len(not_found) == 0 and len(all_issues) == 0 else 1

if __name__ == "__main__":
    sys.exit(main())