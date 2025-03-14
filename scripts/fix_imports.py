#!/usr/bin/env python3
"""
Script to automatically generate proper __init__.py files for all physics domain directories.
This helps resolve circular import issues and ensures all functions are properly exported.
"""

import os
import re
from pathlib import Path


def generate_init_files():
    base_dir = Path(__file__).parent
    domains_dir = base_dir / "physics_domains"

    # List of physical domains
    domains = [d.name for d in domains_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]

    for domain in domains:
        domain_dir = domains_dir / domain

        # Get all Python files (except __init__.py)
        py_files = [f for f in domain_dir.glob("*.py") if f.name != "__init__.py"]

        # Extract function names from files
        functions = []
        for py_file in py_files:
            # Function name is typically the same as the file name
            function_name = py_file.stem
            functions.append(function_name)

        # Generate __init__.py content
        init_content = f"# physics_domains/{domain}/__init__.py\n\n"

        # Import statements
        for function in functions:
            init_content += f"from .{function} import {function}\n"

        init_content += "\n# Export all functions\n"
        init_content += "__all__ = [\n"
        for function in functions:
            init_content += f"    '{function}',\n"
        init_content += "]\n"

        # Write the init file
        init_file = domain_dir / "__init__.py"
        with open(init_file, "w") as f:
            f.write(init_content)

        print(f"Generated {init_file}")


if __name__ == "__main__":
    generate_init_files()