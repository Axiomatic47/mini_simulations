#!/usr/bin/env python
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Now we can import from utils
from utils.edge_case_checker import EdgeCaseChecker

# Import from config
from config.equations import (
    intelligence_growth, truth_adoption, wisdom_field,
    resistance_resurgence, suppression_feedback
)


def generate_circuit_breaker_version_of_intelligence_growth():
    """Generate fixed version of intelligence_growth with circuit breaker."""
    fixed_code = """import numpy as np
from utils.circuit_breaker import CircuitBreaker

def intelligence_growth(K, W, R, S, N, K_max=100.0):
    \"\"\"
    Computes intelligence growth with saturation to prevent unbounded growth.

    Parameters:
        K (float): Knowledge level
        W (float): Wisdom factor (knowledge integration efficiency)
        R (float): Resistance level
        S (float): Suppression level
        N (float): Network effect (mutual learning contribution)
        K_max (float): Maximum knowledge capacity (prevents unbounded growth)

    Returns:
        float: Intelligence growth rate
    \"\"\"
    # Initialize circuit breaker for numerical stability
    circuit_breaker = CircuitBreaker(
        threshold=1e-10,
        max_value=1e10,
        min_value=1e-10,
        max_rate_of_change=1.0
    )

    # Apply saturation term to prevent unbounded growth
    # Cap knowledge to prevent overflow
    K_safe = min(K_max, max(0.0, K))

    # Apply safe bounds to other parameters
    W_safe = min(10.0, max(0.0, W))
    R_safe = min(100.0, max(0.0, R))
    S_safe = min(100.0, max(0.0, S))
    N_safe = min(10.0, max(-10.0, N))

    # Use saturation term in denominator to limit growth
    numerator = K_safe * W_safe
    denominator = 1.0 + K_safe / K_max

    # Safe division with circuit breaker
    growth_term = circuit_breaker.safe_div(numerator, denominator)

    # Calculate final result
    result = growth_term - R_safe - S_safe + N_safe

    # Final stability check
    return circuit_breaker.check_and_fix(result)
"""
    return fixed_code


def main():
    print("Running edge case analysis and fixes...")

    # Create output directories
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_dir = Path("validation/reports/edge_case")
    report_dir.mkdir(parents=True, exist_ok=True)

    # Define equation functions to check
    equation_functions = {
        'intelligence_growth': intelligence_growth,
        'truth_adoption': truth_adoption,
        'wisdom_field': wisdom_field,
        'resistance_resurgence': resistance_resurgence,
        'suppression_feedback': suppression_feedback
    }

    # Create checker
    checker = EdgeCaseChecker(equation_functions)

    # Analyze all functions
    print("Analyzing functions for edge cases...")
    results = checker.analyze_all_functions()

    # Display summary of analysis
    print("\nEdge Case Analysis Summary:")
    print("=" * 40)

    total_edge_cases = 0
    high_risk_funcs = []

    for func_name, result in results.items():
        if 'error' in result:
            print(f"Error analyzing {func_name}: {result['error']}")
            continue

        edge_count = result.get('edge_case_count', 0)
        safety_score = result.get('safety_score', 100)
        risk_level = result.get('risk_level', 'low')

        print(f"{func_name}:")
        print(f"  - Edge cases: {edge_count}")
        print(f"  - Safety score: {safety_score:.1f}/100")
        print(f"  - Risk level: {risk_level.upper()}")

        total_edge_cases += edge_count

        if risk_level == 'high':
            high_risk_funcs.append(func_name)

    print("\nTotal edge cases found: ", total_edge_cases)
    if high_risk_funcs:
        print("High risk functions that need attention:")
        for func in high_risk_funcs:
            print(f"  - {func}")

    # Generate recommendations
    print("\nGenerating improvement recommendations...")
    recommendations = checker.generate_recommendations()

    # Display top recommendations
    print("\nTop Recommendations:")
    print("=" * 40)
    for func_name, recs in recommendations.items():
        if recs:
            print(f"\n{func_name}: {len(recs)} recommendations")
            # Show up to 3 recommendations per function
            for i, rec in enumerate(recs[:3], 1):
                print(f"  {i}. {rec['issue']}")
                print(f"     → {rec['recommendation']}")

            if len(recs) > 3:
                print(f"  ... and {len(recs) - 3} more recommendations")

    # Generate report
    print("\nGenerating comprehensive report...")
    checker.generate_edge_case_completion_report(str(report_dir))
    print(f"Report generated in {report_dir}")

    # Example of fixing specific functions
    if 'wisdom_field' in equation_functions:
        print("\nGenerating fixes for wisdom_field:")
        fixed_code = checker.generate_fixes('wisdom_field')
        print(fixed_code)

        # Save fixed code to file
        fixed_dir = output_dir / "fixed_functions"
        fixed_dir.mkdir(exist_ok=True)
        with open(fixed_dir / "wisdom_field_fixed.py", "w") as f:
            f.write(fixed_code)
        print(f"Fixed code saved to {fixed_dir / 'wisdom_field_fixed.py'}")

    # Add circuit breaker example
    if 'intelligence_growth' in equation_functions:
        print("\nAdding circuit breaker to intelligence_growth:")
        cb_code = generate_circuit_breaker_version_of_intelligence_growth()
        print(cb_code)

        # Save circuit breaker code to file
        fixed_dir = output_dir / "fixed_functions"
        fixed_dir.mkdir(exist_ok=True)
        with open(fixed_dir / "intelligence_growth_with_cb.py", "w") as f:
            f.write(cb_code)
        print(f"Circuit breaker code saved to {fixed_dir / 'intelligence_growth_with_cb.py'}")

    # Generate test cases
    if 'truth_adoption' in equation_functions:
        print("\nGenerating test cases for edge cases:")
        test_code = checker.generate_test_cases('truth_adoption')

        # Save test code to file
        test_dir = output_dir / "test_cases"
        test_dir.mkdir(exist_ok=True)
        with open(test_dir / "test_truth_adoption_edge_cases.py", "w") as f:
            f.write(test_code)
        print(f"Test cases saved to {test_dir / 'test_truth_adoption_edge_cases.py'}")

    print("\nEdge case analysis and fixes complete.")


if __name__ == "__main__":
    main()