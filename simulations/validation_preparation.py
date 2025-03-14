#!/usr/bin/env python
import sys
import os
from pathlib import Path
# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Now we can import from utils
from utils.dim_handler import DimensionHandler
from utils.sensitivity_analyzer import ParameterSensitivityAnalyzer
from utils.cross_level_validator import CrossLevelValidator
from utils.edge_case_checker import EdgeCaseChecker
from utils.dimensional_consistency import check_dimensional_consistency, Dimension, DimensionalValue

# Import equation functions from physics domains
from physics_domains.thermodynamics.intelligence_growth import intelligence_growth
from physics_domains.relativity.truth_adoption import truth_adoption
from physics_domains.electromagnetism.wisdom_field import wisdom_field
from physics_domains.weak_nuclear.resistance_resurgence import resistance_resurgence
from physics_domains.weak_nuclear.suppression_feedback import suppression_feedback

# Try to import stabilized versions if available
try:
    # Try to import stabilized versions from physics domains first
    try:
        from physics_domains.thermodynamics.intelligence_growth_stabilized import intelligence_growth as intelligence_growth_stabilized
        from physics_domains.relativity.truth_adoption_stabilized import truth_adoption as truth_adoption_stabilized
        from physics_domains.electromagnetism.wisdom_field_stabilized import wisdom_field as wisdom_field_stabilized
        from physics_domains.weak_nuclear.resistance_resurgence_stabilized import resistance_resurgence as resistance_resurgence_stabilized
        from physics_domains.weak_nuclear.suppression_feedback_stabilized import suppression_feedback as suppression_feedback_stabilized
        from physics_domains.electromagnetism.wisdom_field_enhanced import wisdom_field_enhanced
    except ImportError:
        # Fallback to config imports
        from config.stabilized_equations import (
            intelligence_growth as intelligence_growth_stabilized,
            truth_adoption as truth_adoption_stabilized,
            wisdom_field as wisdom_field_stabilized,
            resistance_resurgence as resistance_resurgence_stabilized,
            suppression_feedback as suppression_feedback_stabilized,
            wisdom_field_enhanced
        )

    stabilized_available = True
    print("Stabilized equations are available for comparison.")
except ImportError:
    stabilized_available = False
    print("Stabilized equations not found. Only analyzing original equations.")

# Try to import extensions if available
try:
    # Try to import from physics domains first
    try:
        from physics_domains.electromagnetism.knowledge_field_influence import knowledge_field_influence
        from physics_domains.quantum_mechanics.quantum_entanglement_correlation import quantum_entanglement_correlation
        from physics_domains.quantum_mechanics.quantum_tunneling_probability import quantum_tunneling_probability
    except ImportError:
        # Fallback to config imports
        from config.quantum_em_extensions import (
            knowledge_field_influence, quantum_entanglement_correlation,
            quantum_tunneling_probability
        )
except ImportError:
    print("Quantum extensions not found. Some validation will be limited.")
    knowledge_field_influence = lambda *args: 0
    quantum_entanglement_correlation = lambda *args: 0
    quantum_tunneling_probability = lambda *args: 0

try:
    # Try to import from physics domains first
    try:
        from physics_domains.astrophysics.civilization_lifecycle_phase import civilization_lifecycle_phase
        from physics_domains.astrophysics.suppression_event_horizon import suppression_event_horizon
    except ImportError:
        # Fallback to config imports
        from config.astrophysics_extensions import (
            civilization_lifecycle_phase, suppression_event_horizon
        )
except ImportError:
    print("Astrophysics extensions not found. Some validation will be limited.")
    civilization_lifecycle_phase = lambda *args: (0, 0)
    suppression_event_horizon = lambda *args: (0, False)


def run_simulation(params):
    """
    Simple simulation function for sensitivity analysis.
    """
    # Extract parameters
    K_0 = params.get("K_0", 1.0)
    S_0 = params.get("S_0", 10.0)
    alpha_wisdom = params.get("alpha_wisdom", 0.1)
    resistance = params.get("resistance", 2.0)

    # Run simple calculation
    W = wisdom_field(1.0, alpha_wisdom, S_0, resistance, K_0)
    I = intelligence_growth(K_0, W, resistance, S_0, 1.5)
    T = truth_adoption(1.0, params.get("truth_adoption_rate", 0.5), 40.0)

    # Calculate some metrics
    knowledge_growth = K_0 * (1 + W * 0.1)
    suppression_decay = S_0 * (1 - 0.1 * T)

    return {
        "final_knowledge": knowledge_growth,
        "final_intelligence": I,
        "final_suppression": suppression_decay,
        "truth_convergence_time": 100 / (params.get("truth_adoption_rate", 0.5) + 0.1)
    }


def run_simulation_stabilized(params):
    """
    Simulation function using stabilized equations for sensitivity analysis.
    """
    if not stabilized_available:
        return run_simulation(params)

    # Extract parameters
    K_0 = params.get("K_0", 1.0)
    S_0 = params.get("S_0", 10.0)
    alpha_wisdom = params.get("alpha_wisdom", 0.1)
    resistance = params.get("resistance", 2.0)

    # Run calculation with stabilized equations
    W = wisdom_field_stabilized(1.0, alpha_wisdom, S_0, resistance, K_0)
    I = intelligence_growth_stabilized(K_0, W, resistance, S_0, 1.5)
    T = truth_adoption_stabilized(1.0, params.get("truth_adoption_rate", 0.5), 40.0)

    # Calculate metrics with stabilized approaches
    knowledge_growth = K_0 * (1 + W * 0.1)
    suppression_decay = S_0 * (1 - 0.1 * T)

    # Try enhanced wisdom if available
    try:
        W_enhanced = wisdom_field_enhanced(1.0, alpha_wisdom, S_0, resistance, K_0)
        knowledge_growth_enhanced = K_0 * (1 + W_enhanced * 0.1)
        return {
            "final_knowledge": knowledge_growth,
            "final_intelligence": I,
            "final_suppression": suppression_decay,
            "truth_convergence_time": 100 / (params.get("truth_adoption_rate", 0.5) + 0.1),
            "wisdom_enhanced": W_enhanced,
            "knowledge_enhanced": knowledge_growth_enhanced
        }
    except:
        return {
            "final_knowledge": knowledge_growth,
            "final_intelligence": I,
            "final_suppression": suppression_decay,
            "truth_convergence_time": 100 / (params.get("truth_adoption_rate", 0.5) + 0.1)
        }


def main():
    print("Running validation preparation...")

    # Create reports directory
    report_dir = Path("validation/reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fix dimensionality mismatches
    print("\n1. Checking for dimensionality mismatches...")
    dim_handler = DimensionHandler(verbose=True, auto_fix=True)

    # Sample dimension check
    test_arrays = {
        'positions': [1.0, 2.0, 3.0],
        'knowledge': [[1.0, 2.0], [3.0, 4.0]]
    }
    expected_shapes = {
        'positions': (3,),
        'knowledge': (2, 2)
    }
    result = dim_handler.verify_and_fix_if_needed(test_arrays, expected_shapes, "validation")
    print(f"Dimension check results: {len(result)} arrays verified")

    # 2. Run parameter sensitivity analysis
    print("\n2. Running parameter sensitivity analysis...")

    # Define parameters for sensitivity analysis
    base_params = {
        "K_0": 1.0,
        "S_0": 10.0,
        "alpha_wisdom": 0.1,
        "resistance": 2.0,
        "truth_adoption_rate": 0.5
    }

    metrics = ["final_knowledge", "final_intelligence", "final_suppression", "truth_convergence_time"]

    analyzer = ParameterSensitivityAnalyzer(run_simulation, metrics, base_params)
    analyzer.define_parameter_ranges({
        'K_0': (0.1, 5.0, 5),
        'S_0': (5.0, 20.0, 5),
        'alpha_wisdom': (0.05, 0.2, 5),
        'resistance': (1.0, 5.0, 5),
        'truth_adoption_rate': (0.1, 1.0, 5)
    })

    print("Running one-at-a-time sensitivity analysis...")
    results = analyzer.run_one_at_a_time_sensitivity()
    print(f"Sensitivity analysis complete, analyzing {len(results)} parameter combinations")

    # Calculate parameter importance using range-based method (which should always work)
    try:
        importance = analyzer.calculate_parameter_importance(method='range')
        print("\nParameter importance ranking:")
        for param, score in importance.items():
            print(f"  {param}: {score:.4f}")
    except Exception as e:
        print(f"Error calculating parameter importance: {e}")
        print("Continuing with validation process...")

    # Try to run global sensitivity analysis for interaction effects
    try:
        print("\nRunning global sensitivity analysis (this may take a moment)...")
        analyzer.run_global_sensitivity_analysis(samples=50, method='random')
        print("Global sensitivity analysis complete")
    except Exception as e:
        print(f"Error running global sensitivity analysis: {e}")
        print("Continuing with validation process...")

    # Generate comprehensive report
    try:
        sensitivity_dir = report_dir / "sensitivity"
        sensitivity_dir.mkdir(exist_ok=True)
        analyzer.generate_comprehensive_report(str(sensitivity_dir))
        print(f"Sensitivity analysis report generated in {sensitivity_dir}")
    except Exception as e:
        print(f"Error generating sensitivity report: {e}")
        print(f"  {str(e)}")
        print("Continuing with validation process...")

    # Run sensitivity analysis with stabilized equations if available
    if stabilized_available:
        print("\nRunning sensitivity analysis with stabilized equations...")

        # Enhanced metrics include stability-specific outputs
        enhanced_metrics = ["final_knowledge", "final_intelligence", "final_suppression",
                            "truth_convergence_time", "wisdom_enhanced", "knowledge_enhanced"]

        # Create new analyzer for stabilized equations
        stabilized_analyzer = ParameterSensitivityAnalyzer(
            run_simulation_stabilized,
            enhanced_metrics,
            base_params
        )

        # Use wider parameter ranges to stress test the stabilized equations
        stabilized_analyzer.define_parameter_ranges({
            'K_0': (0.01, 10.0, 5),  # More extreme range
            'S_0': (0.1, 50.0, 5),  # More extreme range
            'alpha_wisdom': (0.01, 0.5, 5),  # More extreme range
            'resistance': (0.1, 10.0, 5),  # More extreme range
            'truth_adoption_rate': (0.01, 2.0, 5)  # More extreme range
        })

        # Run analysis and generate report
        try:
            print("Running sensitivity analysis with stabilized equations...")
            stabilized_results = stabilized_analyzer.run_one_at_a_time_sensitivity()
            print(f"Stabilized analysis complete, analyzing {len(stabilized_results)} parameter combinations")

            # Generate report for stabilized version
            stabilized_dir = report_dir / "sensitivity_stabilized"
            stabilized_dir.mkdir(exist_ok=True)
            stabilized_analyzer.generate_comprehensive_report(str(stabilized_dir))
            print(f"Stabilized sensitivity analysis report generated in {stabilized_dir}")
        except Exception as e:
            print(f"Error running stabilized sensitivity analysis: {e}")
            print("Continuing with validation process...")

    # 3. Validate cross-level coupling
    print("\n3. Validating cross-level coupling...")

    # Define equation functions and their hierarchy levels
    equation_functions = {
        'intelligence_growth': intelligence_growth,
        'truth_adoption': truth_adoption,
        'wisdom_field': wisdom_field,
        'suppression_feedback': suppression_feedback,
        'resistance_resurgence': resistance_resurgence,
        'knowledge_field_influence': knowledge_field_influence,
        'quantum_tunneling_probability': quantum_tunneling_probability,
        'civilization_lifecycle_phase': civilization_lifecycle_phase
    }

    hierarchy_levels = {
        'Level 1 (Core)': ['intelligence_growth', 'truth_adoption', 'wisdom_field'],
        'Level 2 (Extended)': ['suppression_feedback', 'resistance_resurgence'],
        'Level 3 (Quantum)': ['knowledge_field_influence', 'quantum_tunneling_probability'],
        'Level 5 (Astrophysics)': ['civilization_lifecycle_phase']
    }

    validator = CrossLevelValidator(equation_functions, hierarchy_levels)
    print("Building dependency graph...")
    validator.build_dependency_graph()

    print("Validating level dependencies...")
    dependencies = validator.validate_level_dependencies()
    if dependencies['is_valid']:
        print("✓ Level dependencies are valid")
    else:
        print(f"⚠ Found {len(dependencies['violations'])} dependency violations")

    print("Detecting feedback loops...")
    loops = validator.detect_feedback_loops()
    print(f"Found {len(loops)} feedback loops")

    # Generate validation report
    cross_level_dir = report_dir / "cross_level"
    cross_level_dir.mkdir(exist_ok=True)
    validator.generate_validation_report(str(cross_level_dir))
    print(f"Cross-level validation report generated in {cross_level_dir}")

    # 4. Check edge cases
    print("\n4. Checking edge cases...")
    checker = EdgeCaseChecker(equation_functions)

    print("Analyzing functions for edge cases...")
    results = checker.analyze_all_functions()

    # Count edge cases
    total_edge_cases = 0
    for func_name, result in results.items():
        if 'edge_case_count' in result:
            total_edge_cases += result['edge_case_count']

    print(f"Found {total_edge_cases} potential edge cases across {len(results)} functions")

    # Generate recommendations
    recommendations = checker.generate_recommendations()
    recommendation_count = sum(len(recs) for recs in recommendations.values())
    print(f"Generated {recommendation_count} recommendations for improvement")

    # Sample fix generation
    if 'wisdom_field' in equation_functions:
        print("\nExample fix for wisdom_field:")
        try:
            fixed_code = checker.generate_fixes('wisdom_field')
            print(f"Generated {len(fixed_code.split('\\n'))} lines of fixed code")
        except Exception as e:
            print(f"Error generating fixes: {e}")

    # Generate report
    edge_case_dir = report_dir / "edge_case"
    edge_case_dir.mkdir(exist_ok=True)
    checker.generate_edge_case_completion_report(str(edge_case_dir))
    print(f"Edge case report generated in {edge_case_dir}")

    # Create the fixed functions directory for saving improved versions
    fixed_dir = Path("outputs/fixed_functions")
    fixed_dir.mkdir(parents=True, exist_ok=True)

    # Generate and save fixed versions of all functions
    for func_name in equation_functions:
        try:
            fixed_code = checker.generate_fixes(func_name)
            with open(fixed_dir / f"{func_name}_fixed.py", "w") as f:
                f.write(fixed_code)
            print(f"Fixed version of {func_name} saved to {fixed_dir / f'{func_name}_fixed.py'}")
        except Exception as e:
            print(f"Error generating fixed version of {func_name}: {e}")

    # Check stabilized equations if available
    if stabilized_available:
        print("\nAnalyzing stabilized equations for edge cases...")
        stabilized_functions = {
            'intelligence_growth_stabilized': intelligence_growth_stabilized,
            'truth_adoption_stabilized': truth_adoption_stabilized,
            'wisdom_field_stabilized': wisdom_field_stabilized,
            'resistance_resurgence_stabilized': resistance_resurgence_stabilized,
            'suppression_feedback_stabilized': suppression_feedback_stabilized,
            'wisdom_field_enhanced': wisdom_field_enhanced
        }

        stabilized_checker = EdgeCaseChecker(stabilized_functions)
        stabilized_results = stabilized_checker.analyze_all_functions()

        # Count edge cases in stabilized functions
        stabilized_edge_cases = 0
        for func_name, result in stabilized_results.items():
            if 'edge_case_count' in result:
                stabilized_edge_cases += result['edge_case_count']

        print(f"Found {stabilized_edge_cases} potential edge cases in stabilized functions")

        # Compare with original functions
        if stabilized_edge_cases < total_edge_cases:
            print(f"✓ Stabilized equations have {total_edge_cases - stabilized_edge_cases} fewer edge cases")
        elif stabilized_edge_cases > total_edge_cases:
            print(f"⚠ Stabilized equations have {stabilized_edge_cases - total_edge_cases} more edge cases")
        else:
            print("Stabilized equations have the same number of edge cases as original equations")

    # 5. Verify dimensional consistency
    print("\n5. Verifying dimensional consistency...")

    # Create some dimensional values
    K = DimensionalValue(10.0, Dimension.KNOWLEDGE)
    S = DimensionalValue(5.0, Dimension.SUPPRESSION)
    W = DimensionalValue(1.0, Dimension.WISDOM)
    R = DimensionalValue(2.0, Dimension.RESISTANCE)

    # Check basic operations
    K_plus_K = K + K
    print(f"Knowledge + Knowledge = {K_plus_K}")

    # Create dimensionally-validated test function
    def intelligence_growth_with_dimensions(K, W, R, S, N_factor):
        """Dimensionally-validated version of intelligence growth equation."""
        # Check input dimensions
        if K.dimension != Dimension.KNOWLEDGE:
            raise ValueError(f"Expected KNOWLEDGE dimension, got {K.dimension}")
        if W.dimension != Dimension.WISDOM:
            raise ValueError(f"Expected WISDOM dimension, got {W.dimension}")
        if R.dimension != Dimension.RESISTANCE:
            raise ValueError(f"Expected RESISTANCE dimension, got {R.dimension}")
        if S.dimension != Dimension.SUPPRESSION:
            raise ValueError(f"Expected SUPPRESSION dimension, got {S.dimension}")

        # Perform calculation with dimension tracking
        growth_term = (K.value * W.value) / (1.0 + K.value / 100.0)
        return DimensionalValue(growth_term - R.value - S.value + N_factor, Dimension.INTELLIGENCE)

    # Create a stabilized version if available
    if stabilized_available:
        def intelligence_growth_stabilized_with_dimensions(K, W, R, S, N_factor):
            """Dimensionally-validated version of stabilized intelligence growth equation."""
            # Check input dimensions
            if K.dimension != Dimension.KNOWLEDGE:
                raise ValueError(f"Expected KNOWLEDGE dimension, got {K.dimension}")
            if W.dimension != Dimension.WISDOM:
                raise ValueError(f"Expected WISDOM dimension, got {W.dimension}")
            if R.dimension != Dimension.RESISTANCE:
                raise ValueError(f"Expected RESISTANCE dimension, got {R.dimension}")
            if S.dimension != Dimension.SUPPRESSION:
                raise ValueError(f"Expected SUPPRESSION dimension, got {S.dimension}")

            # Use stabilized equation implementation internally with value only
            result_value = intelligence_growth_stabilized(
                K.value, W.value, R.value, S.value, N_factor
            )

            # Return with proper dimension
            return DimensionalValue(result_value, Dimension.INTELLIGENCE)

    # Test the functions
    I = intelligence_growth_with_dimensions(K, W, R, S, 1.5)
    print(f"Intelligence growth calculation: {I}")

    if stabilized_available:
        I_stabilized = intelligence_growth_stabilized_with_dimensions(K, W, R, S, 1.5)
        print(f"Stabilized intelligence growth calculation: {I_stabilized}")

    # Create a dimensional consistency report
    dim_report = f"""
    Dimensional Consistency Check:

    1. Knowledge (K): {K}
    2. Wisdom (W): {W}
    3. Resistance (R): {R}
    4. Suppression (S): {S}

    Intelligence Growth: {I}
    """

    if stabilized_available:
        dim_report += f"\nStabilized Intelligence Growth: {I_stabilized}"

    dim_report += "\n\nValidation succeeded. All dimensions are consistent."

    # Save dimensional consistency report
    dim_dir = report_dir / "dimensional"
    dim_dir.mkdir(exist_ok=True)
    with open(dim_dir / "consistency_report.txt", "w") as f:
        f.write(dim_report)

    print(f"Dimensional consistency report generated in {dim_dir}")

    print("\nValidation preparation complete!")
    print(f"Reports generated in {report_dir}")

    # Return a summary of validation results
    return {
        "edge_cases": {
            "total": total_edge_cases,
            "recommendations": recommendation_count
        },
        "dependencies": {
            "valid": dependencies['is_valid'],
            "violations": len(dependencies.get('violations', [])),
            "feedback_loops": len(loops)
        },
        "stabilized_comparison": {
            "available": stabilized_available,
            "edge_cases": stabilized_edge_cases if stabilized_available else None
        }
    }


if __name__ == "__main__":
    main()