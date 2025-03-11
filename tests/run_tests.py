#!/usr/bin/env python
"""
Comprehensive test runner for the Axiomatic Intelligence Growth Simulation Framework.
This script discovers and runs all tests in the tests directory.
"""

import unittest
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))


def run_tests():
    """Discover and run all tests."""
    print("=" * 70)
    print("   Axiomatic Intelligence Growth Simulation - Test Suite")
    print("=" * 70)

    # Find test directory
    test_dir = Path(__file__).resolve().parent

    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=test_dir, pattern='test_*.py')

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Output summary
    print("\n" + "=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    # Return appropriate exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())