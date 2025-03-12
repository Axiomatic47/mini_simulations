# Validation Framework

This document outlines the comprehensive validation framework implemented in the Axiomatic Intelligence Growth Simulation project. The framework provides tools for ensuring numerical stability, parameter sensitivity analysis, cross-level validation, edge case handling, and dimensional consistency prior to empirical validation.

## Table of Contents

1. [Validation Components](#validation-components)
2. [Dimensionality Mismatch Handler](#dimensionality-mismatch-handler)
3. [Parameter Sensitivity Analysis](#parameter-sensitivity-analysis)
4. [Cross-Level Coupling Validation](#cross-level-coupling-validation)
5. [Edge Case Completion](#edge-case-completion)
6. [Dimensional Consistency](#dimensional-consistency)
7. [Running the Validation Suite](#running-the-validation-suite)
8. [Interpreting Validation Reports](#interpreting-validation-reports)
9. [Empirical Validation Guidelines](#empirical-validation-guidelines)

## Validation Components

The validation framework consists of five key components, each addressing a specific aspect of model validation:

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| Dimensionality Mismatch Handler | Fix array dimension inconsistencies | Auto-resizing arrays, safe indexing operations |
| Parameter Sensitivity Analysis | Identify key parameters affecting outcomes | One-at-a-time analysis, global sensitivity analysis |
| Cross-Level Coupling Validation | Verify interactions between hierarchy levels | Dependency graph analysis, signal propagation testing |
| Edge Case Completion | Ensure robustness to extreme inputs | Automatic edge case detection, fix generation |
| Dimensional Consistency | Ensure physical consistency | Tracking physical dimensions, validation of equation consistency |

## Dimensionality Mismatch Handler

The `DimensionHandler` class provides tools for handling array dimension mismatches, which are particularly important for multi-civilization simulations.

### Key Functions

- `verify_dimensions`: Check if arrays have expected dimensions
- `fix_dimensions`: Automatically resize arrays to match expected dimensions
- `verify_and_fix_array_index`: Handle out-of-bounds array indices
- `safe_multi_civilization_update`: Safely update arrays when civilizations are added/removed

### Usage Example

```python
from utils.dim_handler import DimensionHandler

# Create handler
dim_handler = DimensionHandler(verbose=True, auto_fix=True)

# Check array dimensions
arrays = {
    'positions': positions_array,
    'knowledge': knowledge_array
}
expected_shapes = {
    'positions': (n_civs, 2),  # n_civs x 2D positions
    'knowledge': (n_civs,)     # n_civs knowledge values
}

# Verify and fix dimensions
fixed_arrays = dim_handler.verify_and_fix_if_needed(arrays, expected_shapes, "simulation")

# Get status report
report = dim_handler.get_status_report()
print(f"Fixed: {report['fixed_count']}, Warnings: {report['warning_count']}")
```

### Integration in Multi-Civilization Simulations

The module provides drop-in replacements for key multi-civilization functions:

```python
from utils.dim_handler import safe_calculate_distance_matrix, safe_process_civilization_interactions

# Use safe versions of functions
distance_matrix = safe_calculate_distance_matrix(positions)
updated_data = safe_process_civilization_interactions(civs_data, process_func)
```

## Parameter Sensitivity Analysis

The `ParameterSensitivityAnalyzer` provides tools for identifying which parameters most strongly influence simulation outcomes.

### Key Functions

- `run_one_at_a_time_sensitivity`: Vary parameters individually to assess their impacts
- `run_global_sensitivity`: Use Sobol or Latin Hypercube sampling for global sensitivity
- `calculate_parameter_correlations`: Identify correlations between parameters and outputs
- `identify_interaction_effects`: Detect parameter interactions
- `generate_comprehensive_report`: Create HTML report of sensitivity analysis

### Usage Example

```python
from utils.sensitivity_analyzer import ParameterSensitivityAnalyzer

# Define simulation function
def run_simulation(params):
    # Run simulation with given parameters
    # Return dictionary of output metrics
    return {"final_knowledge": K, "final_intelligence": I, ...}

# Define metrics to track
metrics = ["final_knowledge", "final_intelligence", "truth_convergence_time"]

# Create analyzer
analyzer = ParameterSensitivityAnalyzer(run_simulation, metrics, base_parameters)

# Define parameter ranges to test
analyzer.define_parameter_ranges({
    'K_0': (0.1, 5.0, 5),            # (min, max, num_points)
    'alpha_wisdom': (0.05, 0.2, 5),
    'gamma_phase': (0.05, 0.2, 5)
})

# Run analysis
results = analyzer.run_one_at_a_time_sensitivity()

# Calculate importance
importance = analyzer.calculate_parameter_importance()
print("Parameter importance ranking:", importance)

# Generate report
analyzer.generate_comprehensive_report("validation/reports/sensitivity")
```

### Interpreting Sensitivity Results

- **Parameter importance**: Ranks parameters by their overall influence on outputs
- **Tornado plots**: Shows how each parameter affects specific outputs
- **Correlation matrix**: Reveals relationships between parameters and outputs
- **Interaction network**: Visualizes parameter interactions

## Cross-Level Coupling Validation

The `CrossLevelValidator` verifies that interactions between different levels of the equation hierarchy work correctly and consistently.

### Key Functions

- `build_dependency_graph`: Analyze function dependencies across levels
- `validate_level_dependencies`: Check if dependencies follow expected patterns
- `analyze_cross_level_impact`: Measure how perturbations at each level impact other levels
- `detect_feedback_loops`: Identify potentially problematic feedback loops
- `run_signal_propagation_test`: Test how signals propagate across levels
- `validate_convergence_properties`: Verify stability and convergence

### Usage Example

```python
from utils.cross_level_validator import CrossLevelValidator

# Define equation functions and their hierarchy levels
equation_functions = {
    'intelligence_growth': intelligence_growth,
    'truth_adoption': truth_adoption,
    # Other functions...
}

hierarchy_levels = {
    'Level 1 (Core)': ['intelligence_growth', 'free_will_decision', 'truth_adoption', 'wisdom_field'],
    'Level 2 (Extended)': ['suppression_feedback', 'resistance_resurgence'],
    'Level 3 (Quantum)': ['knowledge_field_influence', 'quantum_tunneling_probability'],
    'Level 5 (Astrophysics)': ['civilization_lifecycle_phase']
}

# Create validator
validator = CrossLevelValidator(equation_functions, hierarchy_levels)

# Build dependency graph
validator.build_dependency_graph()

# Validate level dependencies
dependency_results = validator.validate_level_dependencies()
if not dependency_results['is_valid']:
    print("Level dependency violations:", dependency_results['violations'])

# Detect feedback loops
feedback_loops = validator.detect_feedback_loops()
print("Cross-level feedback loops:", [loop for loop in feedback_loops if loop['is_cross_level']])

# Generate validation report
validator.generate_validation_report("validation/reports/cross_level")
```

### Key Metrics to Monitor

- **Dependency violations**: Functions at higher levels calling lower level functions
- **Cross-level feedback loops**: Circular dependencies across hierarchy levels
- **Signal propagation delays**: How quickly effects propagate between levels
- **Convergence properties**: Whether outputs at each level converge appropriately

## Edge Case Completion

The `EdgeCaseChecker` systematically identifies and fixes potential numerical stability issues in equation functions.

### Key Functions

- `analyze_function`: Check a function for potential edge cases
- `analyze_all_functions`: Analyze all functions in the framework
- `generate_recommendations`: Create recommendations for improving edge case handling
- `generate_fixes`: Automatically generate code fixes for a function
- `add_circuit_breaker`: Add circuit breaker integration to a function
- `generate_test_cases`: Create test cases targeting edge cases

### Usage Example

```python
from utils.edge_case_checker import EdgeCaseChecker

# Define equation functions to check
equation_functions = {
    'intelligence_growth': intelligence_growth,
    'truth_adoption': truth_adoption,
    # Other functions...
}

# Create checker
checker = EdgeCaseChecker(equation_functions)

# Analyze all functions
checker.analyze_all_functions()

# Generate recommendations
recommendations = checker.generate_recommendations()
for func_name, recs in recommendations.items():
    if recs:
        print(f"{func_name}: {len(recs)} recommendations")
        for rec in recs:
            print(f"  - {rec['issue']}: {rec['recommendation']}")

# Generate fixes for a specific function
fixed_code = checker.generate_fixes('wisdom_field')
print("Fixed wisdom_field function:")
print(fixed_code)

# Add circuit breaker to a function
cb_code = checker.add_circuit_breaker('intelligence_growth')
print("Intelligence growth with circuit breaker:")
print(cb_code)

# Generate test cases for edge cases
test_code = checker.generate_test_cases('quantum_tunneling_probability')
print("Test cases for quantum_tunneling_probability:")
print(test_code)

# Generate comprehensive report
checker.generate_edge_case_completion_report("validation/reports/edge_case")
```

### Edge Case Categories

The checker identifies the following types of edge cases:

- **Division by zero**: Potential division by zero or very small values
- **Log of non-positive**: Logarithm of zero or negative values
- **Sqrt of negative**: Square root of negative values
- **Overflow/underflow**: Potential exponential overflow or underflow
- **Array bounds**: Array indexing that might go out of bounds
- **Special cases**: Conditional logic that may need testing

## Dimensional Consistency

The `DimensionalValue` class provides a way to track and verify the physical dimensions of values throughout calculations.

### Key Concepts

- **Dimension**: Physical dimension like knowledge, intelligence, time, etc.
- **DimensionalValue**: Value with an associated dimension
- **Dimensional operations**: Operations that enforce dimensional consistency

### Usage Example

```python
from utils.dimensional_consistency import Dimension, DimensionalValue, validate_equation_dimensions

# Create dimensional values
K = DimensionalValue(10.0, Dimension.KNOWLEDGE)
S = DimensionalValue(5.0, Dimension.SUPPRESSION)
W = DimensionalValue(1.0, Dimension.WISDOM)
R = DimensionalValue(2.0, Dimension.RESISTANCE)

# Use in dimensionally-validated function
@validate_equation_dimensions
def intelligence_growth_with_dimensions(K, W, R, S, N_factor):
    """Dimensionally-validated version of intelligence growth equation."""
    # Ensure inputs have correct dimensions
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

# Calculate with dimension checking
result = intelligence_growth_with_dimensions(K, W, R, S, 1.5)
print(f"Result: {result.value} {result.dimension}")

# Check dimensional consistency across equations
from utils.dimensional_consistency import check_dimensional_consistency

equations = {
    'intelligence_growth': intelligence_growth_with_dimensions,
    # Other dimensionally-validated functions...
}

consistency_results = check_dimensional_consistency(equations)
for name, result in consistency_results.items():
    print(f"{name}: {result['status']}")
```

### Benefits of Dimensional Analysis

- Catches dimensional errors during development
- Ensures physical consistency of all equations
- Makes the model more rigorous for empirical validation
- Helps identify conceptual inconsistencies

## Running the Validation Suite

The validation suite can be run as a whole or as individual components.

### Complete Validation

```python
# In simulations/validation_preparation.py
from utils.dim_handler import DimensionHandler
from utils.sensitivity_analyzer import ParameterSensitivityAnalyzer
from utils.cross_level_validator import CrossLevelValidator
from utils.edge_case_checker import EdgeCaseChecker
from utils.dimensional_consistency import check_dimensional_consistency

def main():
    print("Running validation preparation...")
    
    # 1. Fix dimensionality mismatches
    print("\n1. Checking for dimensionality mismatches...")
    dim_handler = DimensionHandler(verbose=True, auto_fix=True)
    # Run dimension checks on key arrays
    
    # 2. Run parameter sensitivity analysis
    print("\n2. Running parameter sensitivity analysis...")
    analyzer = ParameterSensitivityAnalyzer(run_simulation, metrics, base_parameters)
    analyzer.run_one_at_a_time_sensitivity()
    analyzer.generate_comprehensive_report("validation/reports/sensitivity")
    
    # 3. Validate cross-level coupling
    print("\n3. Validating cross-level coupling...")
    validator = CrossLevelValidator(equation_functions, hierarchy_levels)
    validator.build_dependency_graph()
    validator.validate_level_dependencies()
    validator.generate_validation_report("validation/reports/cross_level")
    
    # 4. Check edge cases
    print("\n4. Checking edge cases...")
    checker = EdgeCaseChecker(equation_functions)
    checker.analyze_all_functions()
    checker.generate_edge_case_completion_report("validation/reports/edge_case")
    
    # 5. Verify dimensional consistency
    print("\n5. Verifying dimensional consistency...")
    consistency_results = check_dimensional_consistency(dimensional_equations)
    
    print("\nValidation preparation complete!")
    print("Reports generated in validation/reports/")
    
if __name__ == "__main__":
    main()
```

### Individual Component Validation

Each component can also be run separately:

```bash
# Run dimension checks
python simulations/check_dimensions.py

# Run sensitivity analysis
python simulations/run_sensitivity_analysis.py

# Validate cross-level coupling
python simulations/validate_cross_level.py

# Check edge cases
python simulations/fix_edge_cases.py

# Verify dimensional consistency
python simulations/check_dimensions.py
```

## Interpreting Validation Reports

The validation framework generates comprehensive HTML reports for each component.

### Sensitivity Analysis Report

- **Executive Summary**: Overall parameter sensitivity metrics
- **Parameter Importance**: Ranking of parameters by influence
- **Tornado Plots**: Impact of each parameter on specific outputs
- **Parameter Interactions**: Network visualization of interactions

### Cross-Level Validation Report

- **Dependency Graph**: Visualization of function dependencies
- **Level Dependencies**: Analysis of dependencies between hierarchy levels
- **Feedback Loops**: Identification of potential feedback loops
- **Signal Propagation**: How signals propagate between levels
- **Convergence Properties**: Stability and convergence analysis

### Edge Case Report

- **Edge Case Coverage**: Percentage of edge cases handled
- **Edge Case by Category**: Breakdown by edge case type
- **Function Scores**: Edge case handling scores by function
- **Improvement Priorities**: Functions most in need of improvement
- **Recommendations**: Specific recommendations for improvement

## Empirical Validation Guidelines

After running the validation suite, use these guidelines for empirical validation:

1. **Start with Most Sensitive Parameters**:
   - Use sensitivity analysis results to prioritize parameters for empirical calibration
   - Focus on parameters with highest importance scores first

2. **Test Cross-Level Scenarios**:
   - Focus on testing scenarios that exercise cross-level interactions
   - Verify signal propagation matches empirical observations

3. **Validate Edge Cases**:
   - Test with extreme input values to verify robustness
   - Use generated test cases as templates for empirical validation

4. **Dimensional Consistency Check**:
   - Confirm that dimensions of empirical data match model dimensions
   - Verify units are consistent throughout validation

5. **Iterative Refinement**:
   - Use discrepancies between model and empirical data to refine equations
   - Re-run validation suite after significant changes

6. **Document Validation Results**:
   - Record all validation results in detail
   - Note any discrepancies between model predictions and empirical data
   - Document parameter values that produce best empirical fit


## Robust Error Handling

The validation framework includes comprehensive error handling and graceful degradation features to ensure validation processes can continue even when individual components encounter issues.

### Key Error Handling Features

- **Graceful Component Degradation**: The validation suite continues running even when individual components fail
- **Type Handling Safety**: Robust handling of type conversions, None values, and unexpected data formats
- **Visualization Resilience**: Visualization components adapt to missing or invalid data
- **Circuit Breaker Instantiation**: Proper handling of both CircuitBreaker classes and instances
- **Fallback Mechanisms**: Sensible defaults and fallback implementations when expected components are unavailable

### Example Error Handling Implementation

```python
# Example of robust error handling in cross-level validation
try:
    # Try to create visualizations
    create_cross_level_visualizations(validator, output_dir=output_path)
    logger.info("Generated cross-level visualizations")
except Exception as e:
    # Log error but continue with validation
    logger.error(f"Error generating cross-level visualizations: {e}")
    # Create fallback simple visualization if needed
    create_simple_fallback_visualization(output_path)
    
# Example of handling None values in level dependencies
levels = []
for level_pair in dependencies.keys():
    for level in level_pair:
        if level is not None and level not in levels:
            levels.append(level)