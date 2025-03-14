"""
Module mapping utility to dynamically discover and load equation functions
across the physics_domains directory structure.
"""

import os
import importlib
import inspect
import sys
from pathlib import Path
from typing import Dict, Callable, List, Any, Tuple, Optional

# Physics domains we want to scan
PHYSICS_DOMAINS = [
    "astrophysics",
    "electromagnetism",
    "multi_system",
    "quantum_mechanics",
    "relativity",
    "strong_nuclear",
    "thermodynamics",
    "weak_nuclear"
]


def discover_equation_functions() -> Dict[str, Callable]:
    """
    Dynamically discover all equation functions across physics_domains,
    loading each file directly to avoid circular imports.

    Returns:
        Dictionary mapping function names to function objects
    """
    equation_functions = {}
    base_dir = Path(__file__).parent
    physics_dir = base_dir / "physics_domains"

    # Ensure physics_domains is in the Python path
    sys.path.insert(0, str(base_dir))

    # Import functions from legacy config files first
    for config_file in ['equations.py', 'quantum_em_extensions.py', 'astrophysics_extensions.py',
                        'multi_civilization_extensions.py']:
        try:
            config_path = base_dir / 'config' / config_file
            if config_path.exists():
                # Extract function objects defined in this file
                extract_functions_from_file(str(config_path), equation_functions)
        except Exception as e:
            print(f"Error processing config file {config_file}: {e}")

    # Process each domain directory
    for domain in PHYSICS_DOMAINS:
        domain_path = physics_dir / domain
        if not domain_path.is_dir():
            print(f"Warning: Domain directory {domain} not found")
            continue

        # Get all Python files in the domain directory
        py_files = list(domain_path.glob("*.py"))
        for py_file in py_files:
            if py_file.name == "__init__.py":
                continue

            # Process this file directly
            try:
                extract_functions_from_file(str(py_file), equation_functions)
            except Exception as e:
                print(f"Error extracting functions from {py_file}: {e}")

    return equation_functions

def discover_equation_functions() -> Dict[str, Callable]:
    """
    Dynamically discover all equation functions across physics_domains,
    loading each file directly to avoid circular imports.

    Returns:
        Dictionary mapping function names to function objects
    """
    equation_functions = {}
    base_dir = Path(__file__).parent
    physics_dir = base_dir / "physics_domains"

    # Ensure physics_domains is in the Python path
    sys.path.insert(0, str(base_dir))

    # Import functions from legacy config files first
    for config_file in ['equations.py', 'quantum_em_extensions.py', 'astrophysics_extensions.py',
                        'multi_civilization_extensions.py']:
        try:
            config_path = base_dir / 'config' / config_file
            if config_path.exists():
                # Extract function objects defined in this file
                extract_functions_from_file(str(config_path), equation_functions)
        except Exception as e:
            print(f"Error processing config file {config_file}: {e}")

    # Process each domain directory
    for domain in PHYSICS_DOMAINS:
        domain_path = physics_dir / domain
        if not domain_path.is_dir():
            print(f"Warning: Domain directory {domain} not found")
            continue

        # Get all Python files in the domain directory
        py_files = list(domain_path.glob("*.py"))
        for py_file in py_files:
            if py_file.name == "__init__.py":
                continue

            # Process this file directly
            try:
                extract_functions_from_file(str(py_file), equation_functions)
            except Exception as e:
                print(f"Error extracting functions from {py_file}: {e}")

    return equation_functions

def extract_functions_from_file(file_path, function_dict):
    """
    Extract function objects from a file without importing the module.

    Args:
        file_path: Path to the Python file
        function_dict: Dictionary to store extracted functions
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Parse the file to find function names
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_name = node.name

                # Skip private functions
                if function_name.startswith('_'):
                    continue

                # For each function, store its name and location
                function_dict[function_name] = {
                    'name': function_name,
                    'file_path': file_path,
                    'docstring': ast.get_docstring(node),
                    'domain': file_path.split('physics_domains/')[1].split('/')[0]
                    if 'physics_domains/' in file_path else 'config'
                }
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")

def map_function_to_domain(func_name: str) -> Tuple[str, str]:
    """
    Maps a function name to its physics domain and file.

    Args:
        func_name: Name of the function

    Returns:
        Tuple of (domain, file_name)
    """
    base_dir = Path(__file__).parent.parent
    physics_dir = base_dir / "physics_domains"

    for domain in PHYSICS_DOMAINS:
        domain_path = physics_dir / domain
        if not domain_path.is_dir():
            continue

        # Check each Python file in the domain
        for py_file in domain_path.glob("*.py"):
            if py_file.name == "__init__.py":
                continue

            # Read the file to check if it contains the function definition
            content = py_file.read_text()
            if f"def {func_name}" in content:
                return domain, py_file.stem

    # Check for functions that might still be in config
    config_dir = base_dir / "config"
    for py_file in config_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue

        content = py_file.read_text()
        if f"def {func_name}" in content:
            return "config", py_file.stem

    return "unknown", "unknown"


def get_function_hierarchy() -> Dict[str, List[str]]:
    """
    Organize functions by their physics domain.

    Returns:
        Dictionary mapping domain names to lists of function names
    """
    functions = discover_equation_functions()
    hierarchy = {domain: [] for domain in PHYSICS_DOMAINS}
    hierarchy["config"] = []  # For any remaining functions in config

    for func_name in functions:
        domain, _ = map_function_to_domain(func_name)
        if domain in hierarchy:
            hierarchy[domain].append(func_name)
        else:
            print(f"Warning: Unknown domain for function {func_name}")

    return hierarchy


def get_bridge_functions() -> Dict[str, Callable]:
    """
    Discover all bridge functions across scale levels.

    Returns:
        Dictionary mapping bridge function names to function objects
    """
    bridge_functions = {}
    base_dir = Path(__file__).parent.parent
    bridge_dir = base_dir / "bridge_functions"

    if not bridge_dir.is_dir():
        print(f"Warning: Bridge functions directory not found at {bridge_dir}")
        return bridge_functions

    # Add bridge_functions to Python path
    sys.path.insert(0, str(base_dir))

    # Import each bridge function module
    for py_file in bridge_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue

        # Convert file path to module path
        rel_path = py_file.relative_to(base_dir)
        module_path = str(rel_path.with_suffix("")).replace(os.sep, ".")

        try:
            # Import the module
            module = importlib.import_module(module_path)

            # Get all functions from the module
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                # Skip private functions
                if name.startswith("_"):
                    continue

                # Add function to our dictionary
                bridge_functions[name] = obj

        except ImportError as e:
            print(f"Error importing {module_path}: {e}")

    return bridge_functions


def discover_equation_functions() -> Dict[str, Callable]:
    """
    Dynamically discover all equation functions across physics_domains.

    Returns:
        Dictionary mapping function names to function objects
    """
    equation_functions = {}
    base_dir = Path(__file__).parent
    physics_dir = base_dir / "physics_domains"

    # Ensure physics_domains is in the Python path
    sys.path.insert(0, str(base_dir))

    # Import each domain module
    for domain in PHYSICS_DOMAINS:
        domain_path = physics_dir / domain
        if not domain_path.is_dir():
            print(f"Warning: Domain directory {domain} not found")
            continue

        # Get all Python files in the domain directory
        py_files = list(domain_path.glob("*.py"))
        for py_file in py_files:
            if py_file.name == "__init__.py":
                continue

            # Convert file path to module path
            rel_path = py_file.relative_to(base_dir)
            module_path = str(rel_path.with_suffix("")).replace(os.sep, ".")

            try:
                # Import the module
                module = importlib.import_module(module_path)

                # Get all functions from the module
                for name, obj in inspect.getmembers(module, inspect.isfunction):
                    # Skip private functions
                    if name.startswith("_"):
                        continue

                    # Add function to our dictionary
                    equation_functions[name] = obj

            except ImportError as e:
                print(f"Error importing {module_path}: {e}")
            except Exception as e:
                print(f"Unexpected error importing {module_path}: {e}")

    return equation_functions

def print_module_map():
    """Print a summary of discovered functions by domain"""
    hierarchy = get_function_hierarchy()

    print("\n==== Physics Domain Function Map ====")
    for domain, functions in hierarchy.items():
        if not functions:
            continue
        print(f"\n{domain.upper()} ({len(functions)} functions):")
        for func in sorted(functions):
            print(f"  - {func}")

    bridge_functions = get_bridge_functions()
    if bridge_functions:
        print("\nBRIDGE FUNCTIONS:")
        for func in sorted(bridge_functions.keys()):
            print(f"  - {func}")

    # Overall stats
    total_funcs = sum(len(funcs) for funcs in hierarchy.values()) + len(bridge_functions)
    print(f"\nTotal discovered functions: {total_funcs}")


if __name__ == "__main__":
    print_module_map()