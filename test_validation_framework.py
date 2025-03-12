# test_validation_framework.py

import sys
import os
import traceback


def header(text):
    """Print a header with the given text."""
    print("\n" + "=" * 80)
    print(f" {text} ".center(80, "="))
    print("=" * 80)


def test_circuit_breaker():
    """Test the circuit breaker functionality."""
    header("TESTING CIRCUIT BREAKER")

    try:
        from utils.circuit_breaker import CircuitBreaker

        # Create a circuit breaker instance
        cb = CircuitBreaker()
        print("✓ Successfully imported and created CircuitBreaker")

        # Test value stability checking
        stable = cb.check_value_stability(50.0)
        unstable = cb.check_value_stability(float('inf'))
        print(f"✓ Value stability check: stable={stable}, unstable={unstable}")

        # Test array stability checking
        import numpy as np
        stable_array = cb.check_array_stability(np.array([1.0, 2.0, 3.0]))
        unstable_array = cb.check_array_stability(np.array([1.0, float('nan'), 3.0]))
        print(f"✓ Array stability check: stable={stable_array}, unstable={unstable_array}")

        # Test safe operations
        safe_div_result = cb.safe_div(10.0, 0.0, default=999.0)
        print(f"✓ Safe division: 10.0/0.0 = {safe_div_result}")

        safe_exp_result = cb.safe_exp(100.0)
        print(f"✓ Safe exponential: exp(100.0) = {safe_exp_result}")

        # Get status report
        status = cb.get_status_report()
        print(f"✓ Status report: {status}")

        return True

    except Exception as e:
        print(f"✗ Error testing CircuitBreaker: {e}")
        traceback.print_exc()
        return False


def test_dim_handler():
    """Test the dimension handler functionality."""
    header("TESTING DIMENSION HANDLER")

    try:
        from utils.dim_handler import DimensionHandler
        import numpy as np

        # Create a handler
        handler = DimensionHandler(verbose=True)
        print("✓ Successfully imported and created DimensionHandler")

        # Test dimension verification and fixing
        positions = np.random.random((3, 2))
        influence = np.array([1.0, 2.0])  # Should be length 3
        resources = np.array([[1.0, 2.0], [3.0, 4.0]])  # Should be 3x2

        expected_shapes = {
            'positions': (3, 2),
            'influence': (3,),
            'resources': (3, 2)
        }

        arrays = {
            'positions': positions,
            'influence': influence,
            'resources': resources
        }

        fixed_arrays = handler.verify_and_fix_if_needed(arrays, expected_shapes, "test")
        print("✓ Successfully verified and fixed dimensions")

        # Check results
        for name, array in fixed_arrays.items():
            expected_shape = expected_shapes[name]
            actual_shape = array.shape
            if actual_shape == expected_shape:
                print(f"✓ {name}: Shape {actual_shape} matches expected {expected_shape}")
            else:
                print(f"✗ {name}: Shape {actual_shape} doesn't match expected {expected_shape}")

        return True

    except Exception as e:
        print(f"✗ Error testing DimensionHandler: {e}")
        traceback.print_exc()
        return False


def test_dimensional_consistency():
    """Test the dimensional consistency functionality."""
    header("TESTING DIMENSIONAL CONSISTENCY")

    try:
        from utils.dimensional_consistency import Dimension, DimensionalValue

        # Create dimensional values
        k = DimensionalValue(10.0, Dimension.KNOWLEDGE)
        s = DimensionalValue(5.0, Dimension.SUPPRESSION)
        w = DimensionalValue(1.0, Dimension.WISDOM)
        r = DimensionalValue(2.0, Dimension.RESISTANCE)

        print("✓ Successfully imported and created DimensionalValue objects")

        # Test dimensional operations
        try:
            k_plus_k = k + k
            print(f"✓ Addition: {k} + {k} = {k_plus_k}")

            k_times_2 = k * 2.0
            print(f"✓ Multiplication: {k} * 2.0 = {k_times_2}")

            # This should raise a ValueError
            try:
                k_plus_s = k + s
                print(f"✗ Addition of different dimensions didn't raise error: {k} + {s} = {k_plus_s}")
            except ValueError:
                print("✓ Addition of different dimensions correctly raised ValueError")

            return True

        except Exception as e:
            print(f"✗ Error testing dimensional operations: {e}")
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"✗ Error testing DimensionalConsistency: {e}")
        traceback.print_exc()
        return False


def test_sensitivity_analyzer():
    """Test basic functionality of the sensitivity analyzer."""
    header("TESTING SENSITIVITY ANALYZER")

    try:
        from utils.sensitivity_analyzer import ParameterSensitivityAnalyzer
        import numpy as np

        # Define a simple test function
        def test_func(params):
            x = params.get('x', 0)
            y = params.get('y', 0)
            return {
                'sum': x + y,
                'product': x * y
            }

        # Create analyzer
        base_params = {'x': 2.0, 'y': 3.0}
        metrics = ['sum', 'product']
        analyzer = ParameterSensitivityAnalyzer(test_func, metrics, base_params)
        print("✓ Successfully imported and created ParameterSensitivityAnalyzer")

        # Define parameter ranges
        analyzer.define_parameter_ranges({
            'x': (1.0, 5.0, 3),
            'y': (1.0, 5.0, 3)
        })
        print("✓ Successfully defined parameter ranges")

        # Run basic analysis
        try:
            results = analyzer.run_one_at_a_time_sensitivity(parallel=False)
            print("✓ Successfully ran sensitivity analysis")
            print(f"Results shape: {results.shape}")

            # Calculate correlations and importance
            correlations = analyzer.calculate_parameter_correlations()
            print("✓ Successfully calculated parameter correlations")

            importance = analyzer.calculate_parameter_importance()
            print("✓ Successfully calculated parameter importance")
            print(f"Importance: {importance}")

            return True

        except Exception as e:
            print(f"✗ Error running sensitivity analysis: {e}")
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"✗ Error testing SensitivityAnalyzer: {e}")
        traceback.print_exc()
        return False


def test_edge_case_checker():
    """Test basic functionality of the edge case checker."""
    header("TESTING EDGE CASE CHECKER")

    try:
        from utils.edge_case_checker import EdgeCaseChecker

        # Define a test function with edge cases
        def test_func_with_edge_cases(x, y):
            """Test function with potential edge cases."""
            return x / y + np.sqrt(x) + np.log(y)

        # Create checker
        equation_functions = {
            'test_func': test_func_with_edge_cases
        }

        checker = EdgeCaseChecker(equation_functions)
        print("✓ Successfully imported and created EdgeCaseChecker")

        # Try to analyze function
        try:
            result = checker.analyze_function('test_func')
            print("✓ Successfully analyzed function")

            # Check if expected edge cases were found
            patterns_found = result.get('patterns_found', {})
            print(f"Edge cases found: {list(patterns_found.keys())}")

            if 'division_by_zero' in patterns_found:
                print("✓ Correctly identified division by zero risk")

            if 'sqrt_of_negative' in patterns_found:
                print("✓ Correctly identified square root of negative risk")

            if 'log_of_non_positive' in patterns_found:
                print("✓ Correctly identified log of non-positive risk")

            # Generate recommendations
            recommendations = checker.generate_recommendations()
            print("✓ Successfully generated recommendations")
            print(f"Recommendations: {recommendations.get('test_func', [])}")

            return True

        except Exception as e:
            print(f"✗ Error analyzing function: {e}")
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"✗ Error testing EdgeCaseChecker: {e}")
        traceback.print_exc()
        return False


def test_cross_level_validator():
    """Test basic functionality of the cross-level validator."""
    header("TESTING CROSS-LEVEL VALIDATOR")

    try:
        from utils.cross_level_validator import CrossLevelValidator

        # Define some test functions
        def level1_func(x):
            return x * 2

        def level2_func(x):
            return level1_func(x) + 5

        # Create hierarchy
        equation_functions = {
            'level1_func': level1_func,
            'level2_func': level2_func
        }

        hierarchy_levels = {
            'Level 1': ['level1_func'],
            'Level 2': ['level2_func']
        }

        # Create validator
        validator = CrossLevelValidator(equation_functions, hierarchy_levels)
        print("✓ Successfully imported and created CrossLevelValidator")

        # Try to build dependency graph
        try:
            validator.build_dependency_graph()
            print("✓ Successfully built dependency graph")

            # Validate level dependencies
            dependency_results = validator.validate_level_dependencies()
            print("✓ Successfully validated level dependencies")
            print(f"Dependency validation result: {dependency_results}")

            # Detect feedback loops
            feedback_loops = validator.detect_feedback_loops()
            print("✓ Successfully detected feedback loops")
            print(f"Found {len(feedback_loops)} feedback loops")

            return True

        except Exception as e:
            print(f"✗ Error in CrossLevelValidator operations: {e}")
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"✗ Error testing CrossLevelValidator: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all validation framework tests."""
    header("VALIDATION FRAMEWORK TESTING")

    results = {}

    results['circuit_breaker'] = test_circuit_breaker()
    results['dim_handler'] = test_dim_handler()
    results['dimensional_consistency'] = test_dimensional_consistency()
    results['sensitivity_analyzer'] = test_sensitivity_analyzer()
    results['edge_case_checker'] = test_edge_case_checker()
    results['cross_level_validator'] = test_cross_level_validator()

    # Print summary
    header("TEST SUMMARY")

    all_pass = True
    for test, passed in results.items():
        if passed:
            print(f"✓ {test}: PASSED")
        else:
            print(f"✗ {test}: FAILED")
            all_pass = False

    if all_pass:
        print("\nAll tests passed successfully!")
    else:
        print("\nSome tests failed. See details above.")


if __name__ == "__main__":
    run_all_tests()