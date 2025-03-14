#!/usr/bin/env python3
"""
Main validation entry point for Axiomatic Intelligence Growth Simulation.

This script coordinates the execution of various validation components
and provides a unified interface for running validation.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("validation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("run_validation")


def ensure_utils_and_validation_available():
    """Check if required validation utilities are available."""
    try:
        import utils.circuit_breaker
        import utils.dim_handler
        import utils.edge_case_checker
        import utils.cross_level_validator
        import utils.sensitivity_analyzer
        logger.info("Required utils modules are available")
        return True
    except ImportError as e:
        logger.error(f"Required utils are missing: {e}")
        logger.error("Please ensure utils directory is in PYTHONPATH")
        return False


def run_module_mapping():
    """Run module mapping to discover all functions."""
    try:
        # Import our module mapper
        from module_map import discover_equation_functions, print_module_map

        # Print function map
        logger.info("Running module mapping...")
        print_module_map()

        # Get function count
        equations = discover_equation_functions()
        logger.info(f"Discovered {len(equations)} equation functions")

        return len(equations) > 0
    except ImportError:
        logger.error("module_map.py not found. Please create it first.")
        logger.error("See documentation for instructions on setting up module mapping")
        return False


def create_validation_dirs():
    """Create necessary directories for validation outputs."""
    dirs = [
        "validation/reports",
        "validation/reports/edge_case",
        "validation/reports/cross_level",
        "validation/reports/sensitivity",
        "validation/reports/historical",
        "validation/reports/dimensional",
        "validation/reports/unified",
        "validation/logs"
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)

    logger.info(f"Created validation directories")


def run_unified_validation(components=None):
    """
    Run unified validation with specified components.

    Args:
        components: List of components to validate, or None for defaults
    """
    try:
        # Import unified validator
        sys.path.append(".")  # Ensure current directory is in path
        from unified_validator import run_validation

        # Run validation
        validator = run_validation(components)

        # Check for errors
        error_components = [c for c, r in validator.validation_results.items()
                            if r["status"] == "Error" and r["status"] != "Not Run"]

        if error_components:
            logger.error(f"Validation errors in components: {', '.join(error_components)}")
            return False
        else:
            logger.info("Unified validation completed successfully")
            return True
    except ImportError:
        logger.error("unified_validator.py not found. Please create it first.")
        return False


def run_individual_validations(components):
    """
    Run individual validation components separately.

    Args:
        components: List of components to validate
    """
    success = True

    # Edge case validation
    if "edge_cases" in components:
        try:
            logger.info("Running edge case validation...")
            from utils.edge_case_checker import EdgeCaseChecker, run_edge_case_check

            # Run edge case checking
            run_edge_case_check("validation/reports/edge_case")
            logger.info("Edge case validation completed")
        except Exception as e:
            logger.error(f"Edge case validation failed: {e}")
            success = False

    # Cross-level validation
    if "cross_level" in components:
        try:
            logger.info("Running cross-level validation...")
            from utils.cross_level_validator import run_cross_level_validation

            # Run cross-level validation
            run_cross_level_validation("validation/reports/cross_level")
            logger.info("Cross-level validation completed")
        except Exception as e:
            logger.error(f"Cross-level validation failed: {e}")
            success = False

    # Historical validation
    if "historical" in components:
        try:
            logger.info("Running historical validation...")
            from config.historical_validation import run_historical_validation

            # Run historical validation
            validator = run_historical_validation(
                output_dir="validation/reports/historical",
                optimize=True,
                visualize=True
            )
            logger.info(f"Historical validation completed with error: {validator.calculate_error():.2f}")
        except Exception as e:
            logger.error(f"Historical validation failed: {e}")
            success = False

    # Dimensional consistency
    if "dimensions" in components:
        try:
            logger.info("Running dimensional consistency validation...")
            from utils.dimensional_consistency import run_dimensional_validation

            # Run dimensional validation
            run_dimensional_validation("validation/reports/dimensional")
            logger.info("Dimensional consistency validation completed")
        except Exception as e:
            logger.error(f"Dimensional consistency validation failed: {e}")
            success = False

    # Sensitivity analysis
    if "sensitivity" in components:
        try:
            logger.info("Running sensitivity analysis...")
            from utils.sensitivity_analyzer import run_sensitivity_analysis

            # Run sensitivity analysis
            run_sensitivity_analysis("validation/reports/sensitivity")
            logger.info("Sensitivity analysis completed")
        except Exception as e:
            logger.error(f"Sensitivity analysis failed: {e}")
            success = False

    return success


def main():
    """Main validation entry point."""
    parser = argparse.ArgumentParser(description="Run validation for simulation framework")
    parser.add_argument("--components", nargs="*",
                        choices=["edge_cases", "dimensions", "cross_level", "historical", "sensitivity", "all"],
                        default=["edge_cases", "dimensions", "cross_level", "historical"],
                        help="Components to validate")
    parser.add_argument("--unified", action="store_true",
                        help="Use unified validator instead of individual components")
    parser.add_argument("--no-module-map", action="store_true",
                        help="Skip module mapping")

    args = parser.parse_args()

    # Check if required utils are available
    if not ensure_utils_and_validation_available():
        logger.error("Required validation utilities not available. Exiting.")
        return 1

    # Create validation directories
    create_validation_dirs()

    # Handle "all" components choice
    if "all" in args.components:
        args.components = ["edge_cases", "dimensions", "cross_level", "historical", "sensitivity"]

    # Run module mapping if requested
    if not args.no_module_map:
        if not run_module_mapping():
            logger.error("Module mapping failed. Continuing with validation anyway...")

    success = True

    # Run validation
    if args.unified:
        success = run_unified_validation(args.components)
    else:
        success = run_individual_validations(args.components)

    # Report result
    if success:
        logger.info("Validation completed successfully")
        return 0
    else:
        logger.error("Validation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())