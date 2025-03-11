#!/usr/bin/env python
"""
Comprehensive test runner for the Axiomatic Intelligence Growth Simulation Framework.
This script discovers and runs all tests in the tests directory.
"""

import unittest
import sys
import os
import argparse
import time
from pathlib import Path
from unittest import TestLoader, TestSuite

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent))


def run_tests(specific_test=None, test_module=None):
    """
    Discover and run all tests, or a specific test if specified.

    Args:
        specific_test (str): Optional name of a specific test to run
        test_module (str): Optional name of a specific test module to run

    Returns:
        int: 0 if all tests passed, 1 otherwise
    """
    # Create colorful output
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

    # Print header
    print(f"{BLUE}{BOLD}{'=' * 70}{END}")
    print(f"{BLUE}{BOLD}   Axiomatic Intelligence Growth Simulation - Test Suite{END}")
    print(f"{BLUE}{BOLD}{'=' * 70}{END}")

    # Find test directory
    test_dir = Path(__file__).resolve().parent / 'tests'

    # Create a loader
    loader = TestLoader()

    # Create the test suite
    suite = TestSuite()

    # Prepare for timing
    start_time = time.time()

    # Load tests based on input
    if specific_test:
        print(f"{YELLOW}Running specific test: {specific_test}{END}")
        try:
            # Try to load the specific test
            parts = specific_test.split('.')
            if len(parts) == 2:
                # We have a class.method format
                class_name, method_name = parts
                module_path = f"tests.test_{class_name.lower()}"

                # Import the module dynamically
                __import__(module_path)
                module = sys.modules[module_path]

                # Find the test class
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and issubclass(attr,
                                                             unittest.TestCase) and attr_name.lower() == class_name.lower():
                        # Add the specific test method
                        suite.addTest(attr(method_name))
                        break
            else:
                # Assume it's a full test file name
                if specific_test.startswith('test_'):
                    # It's already prefixed with test_
                    file_path = test_dir / specific_test
                else:
                    # Add the test_ prefix
                    file_path = test_dir / f"test_{specific_test}"

                # Add .py if needed
                if not file_path.suffix:
                    file_path = file_path.with_suffix('.py')

                if not file_path.exists():
                    print(f"{RED}Error: Test file {file_path} does not exist{END}")
                    return 1

                suite = loader.discover(start_dir=test_dir, pattern=file_path.name)
        except Exception as e:
            print(f"{RED}Error loading specific test: {e}{END}")
            return 1
    elif test_module:
        print(f"{YELLOW}Running tests from module: {test_module}{END}")
        pattern = f"test_{test_module}.py"
        suite = loader.discover(start_dir=test_dir, pattern=pattern)
    else:
        print(f"{YELLOW}Running all tests...{END}")
        suite = loader.discover(start_dir=test_dir, pattern='test_*.py')

    # Run the test suite
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Calculate execution time
    execution_time = time.time() - start_time

    # Output summary
    print(f"\n{BLUE}{BOLD}{'=' * 70}{END}")
    print(f"{BOLD}Test Summary:{END}")
    print(f"  • Tests run: {result.testsRun}")
    if result.failures:
        print(f"  • {RED}Failures: {len(result.failures)}{END}")
    else:
        print(f"  • Failures: 0")

    if result.errors:
        print(f"  • {RED}Errors: {len(result.errors)}{END}")
    else:
        print(f"  • Errors: 0")

    if result.skipped:
        print(f"  • {YELLOW}Skipped: {len(result.skipped)}{END}")
    else:
        print(f"  • Skipped: 0")

    print(f"  • Execution time: {execution_time:.2f} seconds")

    # Overall result
    if result.wasSuccessful():
        print(f"\n{GREEN}{BOLD}✅ All tests passed successfully!{END}")
    else:
        print(f"\n{RED}{BOLD}❌ Some tests failed. Please check the output above for details.{END}")

    # Return appropriate exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run the test suite for the Axiomatic Intelligence Growth Simulation Framework.')
    parser.add_argument('--test', '-t',
                        help='Specific test to run (e.g., "astrophysics_extensions" or "TestAstrophysicsExtensions.test_civilization_lifecycle_phase")')
    parser.add_argument('--module', '-m', help='Test module to run (e.g., "astrophysics_extensions")')

    args = parser.parse_args()

    # Run the tests
    sys.exit(run_tests(specific_test=args.test, test_module=args.module))