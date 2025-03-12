#!/usr/bin/env python
import unittest
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Now we can import from utils
from utils.cross_level_validator import CrossLevelValidator

# Import equation functions
from config.equations import (
    intelligence_growth, free_will_decision, truth_adoption,
    wisdom_field, suppression_feedback
)

# Try to import extensions if available
try:
    from config.quantum_em_extensions import (
        knowledge_field_influence, quantum_entanglement_correlation,
        quantum_tunneling_probability
    )
except ImportError:
    print("Quantum extensions not found. Some tests will be skipped.")
    knowledge_field_influence = lambda *args: 0
    quantum_entanglement_correlation = lambda *args: 0
    quantum_tunneling_probability = lambda *args: 0

try:
    from config.astrophysics_extensions import (
        civilization_lifecycle_phase, suppression_event_horizon
    )
except ImportError:
    print("Astrophysics extensions not found. Some tests will be skipped.")
    civilization_lifecycle_phase = lambda *args: (0, 0)
    suppression_event_horizon = lambda *args: (0, False)


class TestCrossLevelCoupling(unittest.TestCase):
    def setUp(self):
        """Set up the test environment."""
        # Define equation functions and their hierarchy levels
        self.equation_functions = {
            'intelligence_growth': intelligence_growth,
            'free_will_decision': free_will_decision,
            'truth_adoption': truth_adoption,
            'wisdom_field': wisdom_field,
            'suppression_feedback': suppression_feedback,
            'knowledge_field_influence': knowledge_field_influence,
            'quantum_tunneling_probability': quantum_tunneling_probability,
            'civilization_lifecycle_phase': civilization_lifecycle_phase
        }

        self.hierarchy_levels = {
            'Level 1 (Core)': [
                'intelligence_growth', 'free_will_decision',
                'truth_adoption', 'wisdom_field'
            ],
            'Level 2 (Extended)': [
                'suppression_feedback'
            ],
            'Level 3 (Quantum)': [
                'knowledge_field_influence', 'quantum_tunneling_probability'
            ],
            'Level 5 (Astrophysics)': [
                'civilization_lifecycle_phase'
            ]
        }

        # Create the validator
        self.validator = CrossLevelValidator(self.equation_functions, self.hierarchy_levels)

        # Build the dependency graph
        self.validator.build_dependency_graph()

    def test_build_dependency_graph(self):
        """Test that the dependency graph is built correctly."""
        self.assertIsNotNone(self.validator.dependency_graph)

        # Check that all functions are in the graph
        for func_name in self.equation_functions:
            self.assertIn(func_name, self.validator.dependency_graph.nodes)

        # Print some information about the graph
        print(f"\nDependency graph has {len(self.validator.dependency_graph.nodes)} nodes")
        print(f"Dependency graph has {len(self.validator.dependency_graph.edges)} edges")

    def test_level_dependencies(self):
        """Test that level dependencies are valid."""
        result = self.validator.validate_level_dependencies()

        # Print dependency validation results
        print("\nLevel dependency validation results:")
        print(f"Is valid: {result['is_valid']}")
        print(f"Number of violations: {len(result['violations'])}")

        for level_pair, deps in result['level_dependencies'].items():
            from_level, to_level = level_pair
            print(f"From {from_level} to {to_level}: {len(deps)} dependencies")

        # Assert that the validation passed
        self.assertTrue(result['is_valid'], "Level dependencies should be valid")

        # Print any violations if they exist
        if not result['is_valid']:
            print("\nViolations:")
            for violation in result['violations']:
                print(f"From {violation['from_function']} ({violation['from_level']}) "
                      f"to {violation['to_function']} ({violation['to_level']})")

    def test_feedback_loops(self):
        """Test detection of feedback loops."""
        loops = self.validator.detect_feedback_loops()

        # Print feedback loop information
        print(f"\nDetected {len(loops)} feedback loops")

        # Count cross-level loops
        cross_level_loops = [loop for loop in loops if loop['is_cross_level']]
        print(f"Cross-level loops: {len(cross_level_loops)}")

        # Print details of any cross-level loops
        if cross_level_loops:
            print("\nCross-level loops:")
            for loop in cross_level_loops:
                print(f"- {' → '.join(loop['functions'])}")
                print(f"  Levels: {', '.join(loop['unique_levels'])}")

    def test_hierarchy_consistency(self):
        """Test overall hierarchy consistency."""
        result = self.validator.validate_hierarchy_consistency()

        # Print hierarchy consistency results
        print(f"\nHierarchy consistency: {'PASS' if result['is_consistent'] else 'FAIL'}")
        print(f"Number of issues: {len(result['issues'])}")

        # Print any issues
        if not result['is_consistent']:
            print("\nConsistency issues:")
            for issue in result['issues']:
                print(f"- {issue['issue_type']}: {issue.get('from_level', issue.get('level', 'Unknown'))}")

        # Assert that the hierarchy is consistent
        self.assertTrue(result['is_consistent'], "Hierarchy should be consistent")

    def test_signal_propagation(self):
        """Test signal propagation through the hierarchy levels."""

        # Define a simple test simulation function
        def test_simulation(params, state=None, is_init=True):
            if is_init:
                # Initialize state with parameters
                return {
                    'core_metric': params.get('core_param', 1.0),
                    'extended_metric': 0.0,
                    'quantum_metric': 0.0,
                    'astrophysics_metric': 0.0
                }
            else:
                # Update state based on previous state
                new_state = state.copy()

                # Level 1 (Core) affects Level 2 (Extended)
                new_state['extended_metric'] = 0.5 * state['core_metric']

                # Level 2 (Extended) affects Level 3 (Quantum)
                new_state['quantum_metric'] = 0.3 * state['extended_metric']

                # Level 3 (Quantum) affects Level 5 (Astrophysics)
                new_state['astrophysics_metric'] = 0.2 * state['quantum_metric']

                return new_state

        # Define base parameters and metrics to track
        base_params = {'core_param': 1.0}
        perturb_params = {'core_param': 2.0}  # Double the core parameter value
        metrics = ['core_metric', 'extended_metric', 'quantum_metric', 'astrophysics_metric']

        # Run the signal propagation test
        result = self.validator.run_signal_propagation_test(
            test_simulation, base_params, perturb_params, metrics, time_steps=20
        )

        # Print signal propagation statistics
        print("\nSignal propagation statistics:")
        for metric in metrics:
            first_response = result['statistics']['first_response'][metric]
            peak_response = result['statistics']['peak_response'][metric]
            settling_time = result['statistics']['settling_time'][metric]

            print(f"{metric}:")
            print(f"  First response at step: {first_response}")
            print(f"  Peak response at step: {peak_response}")
            print(f"  Settling time at step: {settling_time}")

        # Assert that signals propagate through all levels
        self.assertLess(
            result['statistics']['first_response']['core_metric'],
            result['statistics']['first_response']['extended_metric'],
            "Core metrics should respond before extended metrics"
        )

        self.assertLess(
            result['statistics']['first_response']['extended_metric'],
            result['statistics']['first_response']['quantum_metric'],
            "Extended metrics should respond before quantum metrics"
        )


if __name__ == '__main__':
    unittest.main()