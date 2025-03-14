#!/usr/bin/env python
"""
Run Validation

This script runs the validation framework on the refactored physics domains.
It uses the validation adapter to bridge the gap between the validation framework
and the refactored codebase structure.
"""

import os
import sys
import logging
import argparse
import importlib
import subprocess
import webbrowser
import platform
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("validation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("run_validation")


def open_report(report_path):
    """
    Open the generated report in the default browser.

    Args:
        report_path: Path to the report file
    """
    try:
        # Convert to absolute path if needed
        if not os.path.isabs(report_path):
            report_path = os.path.abspath(report_path)

        logger.info(f"Opening report at {report_path}")

        # Open using the system's default program
        if platform.system() == 'Darwin':  # macOS
            subprocess.call(['open', report_path])
        elif platform.system() == 'Windows':
            os.startfile(report_path)
        else:  # Linux and other
            subprocess.call(['xdg-open', report_path])
    except Exception as e:
        logger.warning(f"Could not open report automatically: {e}")
        logger.info(f"Report is available at: {report_path}")


def open_validation_reports(status):
    """
    Open validation reports if validation completed successfully.

    Args:
        status: Validation status string

    Returns:
        status: The original status
    """
    if status in ["success", "warning"]:
        logger.info("Opening validation reports...")

        # Open the unified report
        unified_report_path = "validation/reports/unified/unified_validation_report.html"
        if os.path.exists(unified_report_path):
            open_report(unified_report_path)

        # Open the historical validation report
        historical_report_path = "validation/reports/historical/historical_validation_report.html"
        if os.path.exists(historical_report_path):
            open_report(historical_report_path)

    return status


def run_validation(args):
    """
    Run the validation suite with the adapter.

    Args:
        args: Command-line arguments

    Returns:
        str: Validation status
    """
    from validation_adapter import ValidationAdapter

    adapter = ValidationAdapter()

    # Check if physics_domains directory exists
    physics_path = Path("physics_domains")
    if not physics_path.exists():
        logger.error("physics_domains directory not found. Make sure you're in the right directory.")
        return "error"

    # Create output directory
    output_dir = args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Run verification if requested
    if args.verify_only:
        # Run docstring checker
        try:
            import check_docstrings
            check_docstrings.main()
        except ImportError:
            logger.warning("check_docstrings module not found, skipping docstring verification")

        # Run function verification
        try:
            import verify_functions
            verify_functions.main()
        except ImportError:
            logger.warning("verify_functions module not found, skipping function verification")

        logger.info("Verification completed")
        return "success"

    # Run historical validation if requested
    if args.historical_only:
        try:
            logger.info("Running historical validation...")

            # Try to import ValidationSuite
            try:
                from validation.validation_suite import ValidationSuite
                validation_suite = ValidationSuite(adapter)
                historical_status = validation_suite.run_historical_validation()

                logger.info(f"Historical validation completed: {historical_status}")

                # Open report if requested
                if args.open:
                    historical_report_path = "validation/reports/historical/historical_validation_report.html"
                    if os.path.exists(historical_report_path):
                        open_report(historical_report_path)

                return historical_status
            except ImportError:
                logger.error("Could not import ValidationSuite, historical validation not available")
                return "error"
        except Exception as e:
            logger.error(f"Error in historical validation: {e}")
            return "error"

    # Run unified validation
    try:
        logger.info("Running unified validation...")

        # Create output directory
        if not output_dir:
            output_dir = "validation/reports"

        # Run unified validation
        results = adapter.run_unified_validation(output_dir)

        overall_status = results.get('overall_status', 'error')
        logger.info(f"Unified validation completed. Overall status: {overall_status}")

        # Open report if requested
        if args.open:
            unified_report_path = f"{output_dir}/unified/unified_validation_report.html"
            if os.path.exists(unified_report_path):
                open_report(unified_report_path)

        return overall_status
    except Exception as e:
        logger.error(f"Error in unified validation: {e}")
        return "error"


def main():
    """
    Main function for running validation.

    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(description="Run validation on refactored physics domains")
    parser.add_argument("--open", action="store_true", help="Open reports automatically")
    parser.add_argument("--verify-only", action="store_true", help="Only verify functions and docstrings")
    parser.add_argument("--historical-only", action="store_true", help="Only run historical validation")
    parser.add_argument("--output-dir", default="validation/reports", help="Directory to save validation reports")
    args = parser.parse_args()

    logger.info("Starting validation...")

    # Run validation
    status = run_validation(args)

    # Handle opening reports
    if args.open and not args.verify_only:
        status = open_validation_reports(status)

    logger.info(f"Validation complete - overall status: {status}")

    # Return exit code
    return 0 if status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())