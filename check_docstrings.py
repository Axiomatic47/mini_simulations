#!/usr/bin/env python
"""
Script to check docstrings in physics domain functions for validation consistency.
This will help ensure all functions are discovered properly by the validation framework.
"""

import os
import re
import inspect
import importlib
import argparse
from pathlib import Path

# Define the expected patterns
EXPECTED_PATTERNS = [
    r"Physics Domain:\s*(\w+)",
    r"Scale Level:\s*(\w+)",
    r"Application Domains:\s*([\w\s,]+)"
]

KNOWN_PHYSICS_DOMAINS = [
    "thermodynamics", "relativity", "electromagnetism", "quantum_mechanics",
    "weak_nuclear", "strong_nuclear", "astrophysics", "multi_system"
]

KNOWN_SCALE_LEVELS = [
    "quantum", "agent", "group", "civilization", "multi_civilization", "cosmic"
]

def check_function(module_name, func_name, obj):
    """Check a single function's docstring for validation metadata."""
    issues = []

    docstring = inspect.getdoc(obj)
    if not docstring:
        issues.append(f"{module_name}.{func_name}: Missing docstring")
        return issues

    # Check for expected patterns
    missing_patterns = []
    for pattern in EXPECTED_PATTERNS:
        if not re.search(pattern, docstring, re.IGNORECASE):
            pattern_name = pattern.split(':')[0]
            missing_patterns.append(pattern_name)

    if missing_patterns:
        issues.append(f"{module_name}.{func_name}: Missing metadata - {', '.join(missing_patterns)}")

    # Check if physics domain is valid
    match = re.search(EXPECTED_PATTERNS[0], docstring, re.IGNORECASE)
    if match:
        domain = match.group(1).lower()
        if domain not in KNOWN_PHYSICS_DOMAINS:
            issues.append(f"{module_name}.{func_name}: Unknown physics domain '{domain}'")

    # Check if scale level is valid
    match = re.search(EXPECTED_PATTERNS[1], docstring, re.IGNORECASE)
    if match:
        level = match.group(1).lower()
        if level not in KNOWN_SCALE_LEVELS:
            issues.append(f"{module_name}.{func_name}: Unknown scale level '{level}'")

    return issues

def scan_physics_domains():
    """Scan the physics_domains directory structure."""
    print("Scanning physics_domains directory...")
    issues = []

    base_dir = Path("physics_domains")
    if not base_dir.exists():
        print(f"Warning: {base_dir} directory not found")
        return issues

    # Check all domains
    for domain in KNOWN_PHYSICS_DOMAINS:
        domain_dir = base_dir / domain
        if domain_dir.exists() and domain_dir.is_dir():
            print(f"Checking domain: {domain}")
            # Scan all Python files in this domain
            for py_file in domain_dir.glob("*.py"):
                if py_file.name == "__init__.py":
                    continue

                # Import the module and check functions
                module_name = f"physics_domains.{domain}.{py_file.stem}"
                try:
                    # Add parent directory to path if needed
                    if str(base_dir.parent) not in sys.path:
                        sys.path.insert(0, str(base_dir.parent))

                    module = importlib.import_module(module_name)
                    for name, obj in inspect.getmembers(module):
                        if inspect.isfunction(obj) and not name.startswith("_"):
                            func_issues = check_function(module_name, name, obj)
                            issues.extend(func_issues)
                except ImportError as e:
                    issues.append(f"Could not import module {module_name}: {e}")
                except Exception as e:
                    issues.append(f"Error checking module {module_name}: {e}")

    return issues

def fix_docstring(file_path, function_name, physics_domain=None, scale_level=None, application_domains=None):
    """Attempt to fix a docstring in a file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except (FileNotFoundError, PermissionError) as e:
        print(f"Could not open file {file_path}: {e}")
        return False

    # Find the function definition
    function_pattern = re.compile(r'def\s+' + re.escape(function_name) + r'\s*\(.*?\):\s*"""', re.DOTALL)
    match = function_pattern.search(content)
    if not match:
        print(f"Could not find function {function_name} in {file_path}")
        return False

    # Find the end of the docstring
    start_pos = match.end()
    triple_quote_pos = content.find('"""', start_pos)
    if triple_quote_pos == -1:
        print(f"Could not find end of docstring for {function_name} in {file_path}")
        return False

    docstring = content[start_pos:triple_quote_pos]

    # Add missing metadata
    new_docstring = docstring

    # Check if Physics Domain needs to be added
    if physics_domain and "Physics Domain:" not in docstring:
        if new_docstring.strip() and not new_docstring.endswith('\n\n'):
            new_docstring += "\n\n" if not new_docstring.endswith('\n') else "\n"
        new_docstring += f"Physics Domain: {physics_domain}\n"

    # Check if Scale Level needs to be added
    if scale_level and "Scale Level:" not in docstring:
        if "Physics Domain:" not in new_docstring and not new_docstring.endswith('\n\n'):
            new_docstring += "\n\n" if not new_docstring.endswith('\n') else "\n"
        new_docstring += f"Scale Level: {scale_level}\n"

    # Check if Application Domains need to be added
    if application_domains and "Application Domains:" not in docstring:
        if "Scale Level:" not in new_docstring and "Physics Domain:" not in new_docstring and not new_docstring.endswith('\n\n'):
            new_docstring += "\n\n" if not new_docstring.endswith('\n') else "\n"
        domains_str = ", ".join(application_domains) if isinstance(application_domains, list) else application_domains
        new_docstring += f"Application Domains: {domains_str}\n"

    # Replace the docstring
    new_content = content[:start_pos] + new_docstring + content[triple_quote_pos:]

    try:
        with open(file_path, 'w') as f:
            f.write(new_content)
        print(f"Fixed docstring for {function_name} in {file_path}")
        return True
    except (PermissionError, IOError) as e:
        print(f"Could not write to file {file_path}: {e}")
        return False

def get_domain_from_path(file_path):
    """Extract the domain from a file path."""
    path = Path(file_path)
    parts = path.parts

    # Look for one of the known domains in the path
    for part in parts:
        if part.lower() in [d.lower() for d in KNOWN_PHYSICS_DOMAINS]:
            return part.lower()

    return None

def guess_scale_level(function_name, domain):
    """Guess the scale level based on function name and domain."""
    if domain == "quantum_mechanics":
        return "quantum"
    elif domain == "multi_system":
        return "multi_civilization"
    elif "civilization" in function_name:
        return "civilization"
    elif "group" in function_name or "network" in function_name:
        return "group"
    elif "agent" in function_name or "individual" in function_name:
        return "agent"
    elif domain == "astrophysics":
        return "cosmic"

    # Default to civilization level for most functions
    return "civilization"

def guess_application_domains(function_name, docstring):
    """Guess application domains based on function name and docstring."""
    app_domains = []

    # Extract keywords from function name
    keywords = re.findall(r'[a-z]+', function_name.lower())

    # Map common keywords to domains
    domain_mapping = {
        "knowledge": "knowledge",
        "learn": "learning",
        "intelligence": "intelligence",
        "truth": "truth",
        "wisdom": "wisdom",
        "suppression": "suppression",
        "resistance": "resistance",
        "field": "field theory",
        "oscillation": "cycles",
        "civilization": "civilization",
        "culture": "culture",
        "diffusion": "diffusion",
        "grow": "growth",
        "phase": "phase transition",
        "transition": "transition",
        "quantum": "quantum effects",
        "entangle": "entanglement",
        "influence": "influence",
        "interaction": "interaction"
    }

    for keyword in keywords:
        if keyword in domain_mapping:
            app_domains.append(domain_mapping[keyword])

    # Also extract from docstring if available
    if docstring:
        # Look for terms like "models", "calculates", "simulates"
        description_match = re.search(r'(models|calculates|computes|simulates|determines)\s+([\w\s]+)',
                                      docstring, re.IGNORECASE)
        if description_match:
            described = description_match.group(2).lower()
            for term, domain in domain_mapping.items():
                if term in described and domain not in app_domains:
                    app_domains.append(domain)

    # Ensure we have at least one application domain
    if not app_domains:
        if "growth" in function_name or "knowledge" in function_name:
            app_domains = ["knowledge", "growth"]
        elif "truth" in function_name:
            app_domains = ["truth", "adoption"]
        elif "field" in function_name:
            app_domains = ["field theory", "influence"]
        elif "suppression" in function_name:
            app_domains = ["suppression", "resistance"]
        elif "civilization" in function_name:
            app_domains = ["civilization", "development"]
        else:
            app_domains = ["system dynamics"]

    return app_domains

def main():
    """Main function for checking and fixing docstrings."""
    import sys

    # Add current directory to path to help with imports
    sys.path.insert(0, os.getcwd())

    parser = argparse.ArgumentParser(description="Check docstrings in physics domain functions for validation consistency.")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix issues")
    parser.add_argument("--physics-domain", help="Default physics domain for fixing")
    parser.add_argument("--scale-level", help="Default scale level for fixing")
    parser.add_argument("--application-domains", help="Default application domains for fixing (comma-separated)")
    parser.add_argument("--auto-guess", action="store_true", help="Automatically guess missing metadata")
    args = parser.parse_args()

    print("Checking docstrings in physics_domains...")
    issues = scan_physics_domains()

    if not issues:
        print("No issues found! All docstrings are properly formatted.")
        return

    print("\nIssues found:")
    for issue in issues:
        print(f"- {issue}")

    if args.fix or args.auto_guess:
        print("\nAttempting to fix issues...")

        for issue in issues:
            if "Missing metadata" in issue:
                parts = issue.split(": ")
                if len(parts) < 3:
                    continue

                module_func = parts[0]
                module_name, func_name = module_func.rsplit(".", 1)

                # Convert module name to file path
                file_path = module_name.replace(".", "/") + ".py"

                # Check if file exists
                if not os.path.exists(file_path):
                    print(f"File not found: {file_path}")
                    continue

                # Determine what's missing
                missing = parts[2].split(", ")

                # Determine the physics domain
                physics_domain = args.physics_domain
                if args.auto_guess and "Physics Domain" in missing and not physics_domain:
                    # Try to guess from the file path
                    guessed_domain = get_domain_from_path(file_path)
                    if guessed_domain:
                        physics_domain = guessed_domain

                # Determine the scale level
                scale_level = args.scale_level
                if args.auto_guess and "Scale Level" in missing and not scale_level:
                    # Try to guess from the function name and domain
                    guessed_scale = guess_scale_level(func_name, physics_domain if physics_domain else "")
                    if guessed_scale:
                        scale_level = guessed_scale

                # Determine application domains
                application_domains = args.application_domains
                if args.application_domains:
                    application_domains = args.application_domains.split(",")

                if args.auto_guess and "Application Domains" in missing and not application_domains:
                    # Try to get the docstring to guess from
                    try:
                        module = importlib.import_module(module_name)
                        func = getattr(module, func_name)
                        docstring = inspect.getdoc(func)
                        guessed_domains = guess_application_domains(func_name, docstring)
                        if guessed_domains:
                            application_domains = guessed_domains
                    except (ImportError, AttributeError):
                        pass

                # Fix the docstring
                fix_docstring(file_path, func_name, physics_domain, scale_level, application_domains)

if __name__ == "__main__":
    main()