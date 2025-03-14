# Equation Validation Framework Fix Guide

This guide explains how to fix issues with the Axiomatic Equation Validation Framework and ensure it correctly detects all your equations, physics domains, and scale levels.

## Understanding the Issues

The validation framework is failing due to several interrelated issues:

1. **Symbolic Link Problems**: The symbolic links in the `physics_domains` directory point to incorrect paths (`/Users/joseph/Git/mini_simulations/equations.py` instead of the correct `/Users/joseph/Git/mini_simulations/config/equations.py`).

2. **Missing Docstring Annotations**: The validation framework looks for specific docstring patterns:
   - `Physics Domain: domain_name`
   - `Scale Level: level_name`
   - `Application Domains: domain1, domain2`

3. **Module Import Paths**: The validation framework expects specific import paths for your modules.

4. **Bridge Functions and Strong Nuclear Domain**: These aren't being properly detected.

## How to Fix These Issues

### Step 1: Fix Symbolic Links

Run the `fix_symlinks.py` script to update all symbolic links to point to the correct locations:

```bash
python fix_symlinks.py
```

This script:
- Recreates symbolic links for all physics domains
- Points them to the correct files in your `config` directory
- Ensures all necessary directories exist

### Step 2: Update Docstring Annotations

Run the `fix_docstrings.py` script to update function docstrings with required annotations:

```bash
python fix_docstrings.py
```

This script:
- Parses your `EQUATIONS_MAP.md` file to extract function metadata
- Analyzes each Python file to find functions
- Adds missing annotations to function docstrings

### Step 3: Run Validation with Correct Parameters

Run the validation with explicit module paths:

```bash
bash run_validation_fixed.sh
```

This script:
1. Fixes symbolic links
2. Updates docstrings
3. Runs validation with correct module paths

## Manual Fixes (If Needed)

If automatic scripts don't resolve all issues:

### Bridge Functions

Ensure each bridge function has these annotations in its docstring:

```python
"""
Function description...

Physics Domain: domain_name
Scale Levels: level1, level2
Application Domains: domain1, domain2
"""
```

### Strong Nuclear Domain Functions

For strong nuclear functions, ensure:

```python
"""
Function description...

Physics Domain: strong_nuclear
Scale Level: agent  # or appropriate level
Application Domains: knowledge, identity  # or appropriate domains
"""
```

## Future Maintenance

When adding new functions:

1. Add them to `EQUATIONS_MAP.md`
2. Include proper docstring annotations
3. If creating symbolic links, point to the correct file in the `config` directory

## Troubleshooting

If validation still doesn't detect all functions:

1. Check the validation logs in `validation/logs/`
2. Verify docstring annotations match exactly what's expected
3. Use the `--verbose` flag with the validation script
4. Try running with `--focus coverage` to focus on equation detection

## Reference

Physics Domains:
- thermodynamics
- electromagnetism
- strong_nuclear
- weak_nuclear
- quantum_mechanics
- relativity
- astrophysics
- multi_system

Scale Levels:
- quantum
- agent
- group
- civilization
- multi_civilization
- cosmic