"""
Comprehensive validation suite that integrates all validation components.
This script orchestrates the full validation process with simplified loading
that works with a single version of each function.
"""

import os
import sys
import time
import logging
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import importlib
import traceback
from datetime import datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
log_dir = Path("validation/logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "validation_suite.log"),
        logging.StreamHandler()
    ]
)
# Create the global logger
logger = logging.getLogger("validation_suite")


class ValidationSuite:
    def __init__(self, output_dir=None):
        # Use the global logger
        self.logger = logger

        # Set up output directory
        self.output_dir = Path(output_dir) if output_dir else Path("validation/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize results dictionary to store validation results
        self.results = {}
        self.run_components = set()

        # Initialize components to None
        self.components = {}
        self.dim_handler = None
        self.sensitivity_analyzer = None
        self.cross_level_validator = None
        self.edge_case_checker = None
        self.dimensional_validator = None
        self.circuit_breaker = None
        self.historical_validator = None
        self.reporter = None

        # Additional initialization
        self.equation_functions = {}
        self.simulation_func = None
        self.historical_data = None

        self.logger.info("Validation suite initialized")

    def load_components(self):
        """Load validation components."""
        self.logger.info("Loading validation components")

        # Track which components are required vs optional
        required_components = []
        optional_components = ['historical_validator', 'reporter']

        try:
            # Load dimension handler
            try:
                from utils.dim_handler import DimensionHandler
                self.dim_handler = DimensionHandler(verbose=True, auto_fix=True)
                self.components['dim_handler'] = self.dim_handler
                self.logger.info("Loaded dimension handler")
            except ImportError:
                self.logger.warning("Dimension handler not available")
                self.dim_handler = None
                required_components.append('dim_handler')

            # Load sensitivity analyzer
            try:
                from utils.sensitivity_analyzer import ParameterSensitivityAnalyzer
                self.sensitivity_analyzer = ParameterSensitivityAnalyzer
                self.components['sensitivity_analyzer'] = self.sensitivity_analyzer
                self.logger.info("Loaded sensitivity analyzer")
            except ImportError:
                self.logger.warning("Sensitivity analyzer not available")
                self.sensitivity_analyzer = None
                required_components.append('sensitivity_analyzer')

            # Load cross-level validator
            try:
                from utils.cross_level_validator import CrossLevelValidator
                self.cross_level_validator = CrossLevelValidator
                self.components['cross_level_validator'] = self.cross_level_validator
                self.logger.info("Loaded cross-level validator")
            except ImportError:
                self.logger.warning("Cross-level validator not available")
                self.cross_level_validator = None

            # Load edge case checker
            try:
                from utils.edge_case_checker import EdgeCaseChecker
                self.edge_case_checker = EdgeCaseChecker
                self.components['edge_case_checker'] = self.edge_case_checker
                self.logger.info("Loaded edge case checker")
            except ImportError:
                self.logger.warning("Edge case checker not available")
                self.edge_case_checker = None
                required_components.append('edge_case_checker')

            # Load dimensional validator
            try:
                from utils.dimensional_consistency import check_dimensional_consistency
                self.dimensional_validator = check_dimensional_consistency
                self.components['dimensional_validator'] = self.dimensional_validator
                self.logger.info("Loaded dimensional validator")
            except ImportError:
                self.logger.warning("Dimensional validator not available")
                self.dimensional_validator = None

            # Load circuit breaker
            try:
                from utils.circuit_breaker import CircuitBreaker
                self.circuit_breaker = CircuitBreaker
                self.components['circuit_breaker'] = self.circuit_breaker
                self.logger.info("Loaded circuit breaker")
            except ImportError:
                self.logger.warning("Circuit breaker not available")
                self.circuit_breaker = None

            # Load historical validator (OPTIONAL)
            try:
                from validation.historical_integration import HistoricalValidator
                self.historical_validator = HistoricalValidator
                self.components['historical_validator'] = self.historical_validator
                self.logger.info("Loaded historical validator")
            except ImportError:
                self.logger.warning("Historical validator not available")
                self.historical_validator = None

            # Load validation reporter (OPTIONAL)
            try:
                from validation.validation_reporter import ValidationReporter
                self.reporter = ValidationReporter(str(self.output_dir))
                self.components['reporter'] = self.reporter
                self.logger.info("Loaded validation reporter")
            except ImportError:
                self.logger.warning("Validation reporter not available")
                self.reporter = None

            # Check if any required components are missing
            missing_required = [comp for comp in required_components if getattr(self, comp) is None]
            if missing_required:
                self.logger.error(f"Missing required components: {', '.join(missing_required)}")
                return False

            # Final setup steps - place any final initialization here
            self.logger.info("All components loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load validation components: {str(e)}")
            # Print stack trace for debugging
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def load_models_and_data(self):
        """
        Load all necessary models and data for validation.

        Returns:
            Dictionary with loaded components
        """
        logger.info("Loading models and data")

        try:
            # Dictionary to store loaded components
            loaded = {}

            # Load equation functions
            try:
                from config import equations
                loaded["equations"] = equations
                logger.info("Loaded equations module")
            except ImportError:
                logger.warning("Equations module not found")

            # Load quantum & EM extensions
            try:
                import quantum_em_extensions
                loaded["quantum_em_extensions"] = quantum_em_extensions
                logger.info("Loaded quantum EM extensions")
            except ImportError:
                logger.warning("Quantum EM extensions not found")

            # Load parameters
            try:
                import parameters
                loaded["parameters"] = parameters
                logger.info("Loaded parameters")
            except ImportError:
                logger.warning("Parameters not found")

            # Load historical data
            hist_data_path = Path("outputs/data/historical_data.csv")
            if hist_data_path.exists():
                try:
                    loaded["historical_data"] = pd.read_csv(hist_data_path)
                    logger.info(f"Loaded historical data: {len(loaded['historical_data'])} records")
                except Exception as e:
                    logger.error(f"Error loading historical data: {e}")
            else:
                logger.warning(f"Historical data not found at {hist_data_path}")

            return loaded

        except Exception as e:
            logger.error(f"Error loading models and data: {e}")
            traceback.print_exc()
            return {}

    def run_dimension_validation(self, sample_arrays=None):
        """
        Run dimension validation.

        Parameters:
            sample_arrays: Sample arrays to validate dimensions

        Returns:
            Validation results
        """
        logger.info("Running dimension validation")

        if self.dim_handler is None:
            logger.error("Dimension handler not initialized")
            return None

        try:
            # Create sample arrays if not provided
            if sample_arrays is None:
                # Default sample arrays with dimension issues
                sample_arrays = {
                    "positions": np.random.random((3, 2)),
                    "knowledge": np.array([1.0, 2.0]),  # Should be length 3
                    "influence": np.array([[1.0, 2.0], [3.0, 4.0]])  # Should be shape (3,)
                }

            # Define expected shapes
            expected_shapes = {
                "positions": (3, 2),
                "knowledge": (3,),
                "influence": (3,)
            }

            # Verify and fix dimensions
            fixed_arrays = self.dim_handler.verify_and_fix_if_needed(
                sample_arrays, expected_shapes, "validation_suite"
            )

            # Get status report
            status_report = self.dim_handler.get_status_report()

            # Store and return results
            results = {
                "fixed_count": status_report.get("fixed_count", 0),
                "warning_count": status_report.get("warning_count", 0),
                "error_count": status_report.get("error_count", 0),
                "status": "Success" if status_report.get("fixed_count", 0) == 0 else "Warning"
            }

            self.results["dimension"] = results
            self.run_components.add("dimension")

            if self.reporter:
                self.reporter.set_component_result("dimension", results)

            logger.info(f"Dimension validation completed: {results['status']}")
            return results

        except Exception as e:
            logger.error(f"Error in dimension validation: {e}")
            traceback.print_exc()
            return None

    def run_sensitivity_analysis(self, simulation_func, base_parameters, metrics, parameter_ranges=None):
        """
        Run parameter sensitivity analysis.

        Parameters:
            simulation_func: Function to run simulation
            base_parameters: Base parameter values
            metrics: List of metrics to analyze
            parameter_ranges: Dictionary of parameter ranges to test

        Returns:
            Sensitivity analysis results
        """
        logger.info("Running parameter sensitivity analysis")

        try:
            # Import sensitivity analyzer
            from utils.sensitivity_analyzer import ParameterSensitivityAnalyzer

            # Create analyzer
            analyzer = ParameterSensitivityAnalyzer(
                simulation_func, metrics, base_parameters
            )

            # Define parameter ranges if not provided
            if parameter_ranges is None:
                # Default parameter ranges (adjust based on your model)
                parameter_ranges = {
                    param: (value * 0.5, value * 1.5, 5)
                    for param, value in base_parameters.items()
                    if isinstance(value, (int, float)) and param not in
                       ["timesteps", "dt", "t_max"]  # Skip time parameters
                }

            # Set parameter ranges
            analyzer.define_parameter_ranges(parameter_ranges)

            # Run analysis
            logger.info("Running one-at-a-time sensitivity analysis")
            results = analyzer.run_one_at_a_time_sensitivity(parallel=False)

            # Calculate parameter importance
            importance = analyzer.calculate_parameter_importance()

            # Generate visualizations
            try:
                from validation.validation_visualizers import create_parameter_sensitivity_visualizations
                create_parameter_sensitivity_visualizations(
                    analyzer, output_dir=str(self.output_dir / "sensitivity")
                )
                logger.info("Generated sensitivity visualizations")
            except ImportError:
                logger.warning("Could not generate sensitivity visualizations")

            # Store and return results
            results_dict = {
                "parameter_importance": importance.to_dict(),
                "analyzed_parameters": list(parameter_ranges.keys()),
                "metrics": metrics,
                "status": "Success"
            }

            self.results["sensitivity"] = results_dict
            self.run_components.add("sensitivity")

            if self.reporter:
                self.reporter.set_component_result("sensitivity", results_dict)

            logger.info(f"Sensitivity analysis completed: {len(importance)} parameters analyzed")

            # Save analyzer for later use
            self.sensitivity_analyzer = analyzer

            return results_dict

        except Exception as e:
            logger.error(f"Error in sensitivity analysis: {e}")
            traceback.print_exc()
            return None

    def run_edge_case_detection(self, equation_functions=None):
        """
        Run edge case detection on equation functions.

        Parameters:
            equation_functions: Dictionary of equation functions to check

        Returns:
            Edge case detection results
        """
        self.logger.info("Running edge case detection")
        status = "Error"  # Default status

        try:
            # Import edge case checker
            from utils.edge_case_checker import EdgeCaseChecker

            # Use loaded equation functions if not provided
            if equation_functions is None:
                try:
                    # Try to load equations module
                    import equations

                    # Extract functions from module
                    equation_functions = {}
                    for name in dir(equations):
                        if name.startswith("_"):
                            continue
                        item = getattr(equations, name)
                        if callable(item):
                            equation_functions[name] = item

                    self.logger.info(f"Loaded {len(equation_functions)} functions from equations module")

                except ImportError:
                    self.logger.error("Equations module not found and no functions provided")
                    return None

            # Create checker
            checker = EdgeCaseChecker(equation_functions)

            # Analyze all functions
            self.logger.info(f"Analyzing {len(equation_functions)} functions for edge cases")
            checker.analyze_all_functions()

            # Generate recommendations
            recommendations = checker.generate_recommendations()

            # Count issues by severity
            severity_counts = {
                "low": 0,
                "medium": 0,
                "high": 0,
                "critical": 0
            }

            for func_name, func_recs in recommendations.items():
                for rec in func_recs:
                    severity = rec.get("severity", "low")
                    severity_counts[severity] += 1

            # Generate visualizations
            try:
                from validation.validation_visualizers import create_edge_case_visualizations
                create_edge_case_visualizations(
                    checker, output_dir=str(self.output_dir / "edge_case")
                )
                self.logger.info("Generated edge case visualizations")
            except Exception as e:
                self.logger.error(f"Error generating edge case visualizations: {e}")
                # This is a non-critical error - continue without failing

            # Generate fixed code for functions with critical or high issues
            fixed_code = {}
            for func_name, func_recs in recommendations.items():
                critical_issues = any(rec.get("severity") in ["critical", "high"] for rec in func_recs)
                if critical_issues:
                    fixed_code[func_name] = checker.generate_fixes(func_name)

            # Determine status based on analysis results
            critical_issues = []

            # Check if there are any critical recommendations
            for func_name, recs in recommendations.items():
                for rec in recs:
                    if rec.get('severity', 'warning') == 'critical':
                        critical_issues.append(f"{func_name}: {rec['issue']}")

            # Set status based on critical issues
            if critical_issues:
                self.logger.warning(f"Edge case detection found critical issues: {critical_issues}")
                status = "Warning"
            else:
                status = "Success"

            # Store results
            results = {
                "analyzed_functions": len(equation_functions),
                "severity_counts": severity_counts,
                "has_critical_issues": severity_counts["critical"] > 0,
                "has_high_issues": severity_counts["high"] > 0,
                "status": status
            }

            # Store detailed results for reporting
            detailed_results = {
                "analyzed_functions": len(equation_functions),
                "severity_counts": severity_counts,
                "recommendations": recommendations,
                "fixed_code": fixed_code,
                "status": status
            }

            # Save detected edge cases
            self.results["edge_case"] = results
            self.run_components.add("edge_case")

            if self.reporter:
                self.reporter.set_component_result("edge_case", checker.results)

            self.logger.info(f"Edge case detection completed: {status}")

            # Save checker for later use
            self.edge_case_checker = checker

            return detailed_results

        except Exception as e:
            self.logger.error(f"Error in edge case detection: {e}")
            traceback.print_exc()
            status = "Error"
            return None
        finally:
            self.logger.info(f"Edge case detection completed: {status}")

    def create_dummy_simulation(self):
        """Create a robust dummy simulation function for testing."""
        try:
            from validation.historical_integration import create_dummy_simulation_function
            return create_dummy_simulation_function()
        except ImportError:
            self.logger.error("Could not import create_dummy_simulation_function")

            # Define a basic fallback function
            def basic_dummy_simulation(params):
                """Generate synthetic data for testing."""
                import numpy as np
                import random

                # Fix random seed for reproducibility
                random.seed(42)
                np.random.seed(42)

                # Handle parameter structure
                if not isinstance(params, dict):
                    params = {
                        'ALPHA_WISDOM': 0.1,
                        'BETA_FEEDBACK': 0.05,
                        'GAMMA_PHASE': 0.1,
                        'LAMBDA_DECAY': 0.05
                    }

                # Get scale factor from params (bounded between 0.5 and 2.0)
                param_sum = sum(float(v) for v in params.values() if isinstance(v, (int, float)))
                scale = max(0.5, min(2.0, param_sum / 0.3))

                # Generate simple time series
                timesteps = 100
                time = np.arange(timesteps)
                knowledge = 10.0 + 0.5 * time * scale
                suppression = 20.0 * np.exp(-0.05 * time * scale)
                intelligence = 5.0 + 0.3 * knowledge

                # Add some random noise
                knowledge += np.random.normal(0, 1, timesteps)
                suppression += np.random.normal(0, 0.5, timesteps)
                intelligence += np.random.normal(0, 0.5, timesteps)

                # Ensure all values are positive
                knowledge = np.maximum(0, knowledge)
                suppression = np.maximum(0, suppression)
                intelligence = np.maximum(0, intelligence)

                return {
                    'time': time,
                    'knowledge': knowledge,
                    'suppression': suppression,
                    'intelligence': intelligence
                }

            return basic_dummy_simulation

    # Add a helper function to check if dependencies for visualization are available
    def check_visualization_dependencies(self):
        """Check if required visualization dependencies are available."""
        try:
            import matplotlib
            import seaborn
            import networkx
            return True
        except ImportError as e:
            self.logger.warning(f"Visualization dependency missing: {e}")
            return False

    def run_cross_level_validation(self, equation_functions=None, hierarchy_levels=None):
        """
        Run cross-level validation.

        Parameters:
            equation_functions: Dictionary of equation functions
            hierarchy_levels: Dictionary mapping levels to function names

        Returns:
            Cross-level validation results
        """
        self.logger.info("Running cross-level validation")
        status = "Error"  # Default status

        try:
            # Import cross-level validator
            from utils.cross_level_validator import CrossLevelValidator

            # Use loaded equation functions if not provided
            if equation_functions is None:
                try:
                    # Try to load equations module
                    import equations
                    import quantum_em_extensions

                    # Extract functions from modules
                    equation_functions = {}

                    for name in dir(equations):
                        if name.startswith("_"):
                            continue
                        item = getattr(equations, name)
                        if callable(item):
                            equation_functions[name] = item

                    for name in dir(quantum_em_extensions):
                        if name.startswith("_"):
                            continue
                        item = getattr(quantum_em_extensions, name)
                        if callable(item):
                            equation_functions[name] = item

                    self.logger.info(f"Loaded {len(equation_functions)} functions for cross-level validation")

                except ImportError:
                    self.logger.error("Required modules not found and no functions provided")
                    return None

            # Define hierarchy levels if not provided
            if hierarchy_levels is None:
                # Default hierarchy levels (adjust based on your model)
                hierarchy_levels = {
                    'Level 1 (Core)': [
                        'intelligence_growth', 'free_will_decision',
                        'truth_adoption', 'wisdom_field'
                    ],
                    'Level 2 (Extended)': [
                        'suppression_feedback', 'resistance_resurgence'
                    ],
                    'Level 3 (Quantum)': [
                        'quantum_tunneling_probability', 'knowledge_field_influence',
                        'quantum_entanglement_correlation', 'knowledge_field_gradient',
                        'build_entanglement_network'
                    ]
                }

            # Create validator
            validator = CrossLevelValidator(equation_functions, hierarchy_levels)

            # Build dependency graph
            self.logger.info("Building dependency graph")
            validator.build_dependency_graph()

            # Validate level dependencies
            self.logger.info("Validating level dependencies")
            dependency_results = validator.validate_level_dependencies()

            # Detect feedback loops
            self.logger.info("Detecting feedback loops")
            feedback_loops = validator.detect_feedback_loops()

            # Generate visualizations
            try:
                from validation.validation_visualizers import create_cross_level_visualizations
                create_cross_level_visualizations(
                    validator, output_dir=str(self.output_dir / "cross_level")
                )
                self.logger.info("Generated cross-level visualizations")
            except Exception as e:
                self.logger.error(f"Error generating cross-level visualizations: {e}")
                # This is a non-critical error - continue without failing

            # Determine status based on validation results
            validation_issues = []

            # Check if any level dependencies violate expectations
            if dependency_results and not dependency_results.get('is_valid', True):
                validation_issues.append('Level dependencies violate expectations')

            # Check if there are cross-level feedback loops
            if feedback_loops:
                cross_level_loops = [loop for loop in feedback_loops if loop.get('is_cross_level', False)]
                if cross_level_loops:
                    validation_issues.append('Cross-level feedback loops detected')

            # Set status based on issues
            if validation_issues:
                self.logger.warning(f"Cross-level validation found issues: {validation_issues}")
                status = "Warning"
            else:
                status = "Success"

            # Store results
            violations = dependency_results.get('violations', [])
            cross_level_loops = [loop for loop in feedback_loops if loop.get('is_cross_level', False)]

            results = {
                "dependency_violations": len(violations),
                "feedback_loops": len(feedback_loops),
                "cross_level_loops": len(cross_level_loops),
                "status": status
            }

            # Store detailed results for reporting
            detailed_results = {
                "level_dependencies": dependency_results,
                "feedback_loops": feedback_loops,
                "analyzed_functions": len(equation_functions),
                "hierarchy_levels": {level: len(funcs) for level, funcs in hierarchy_levels.items()},
                "status": status
            }

            self.results["cross_level"] = results
            self.run_components.add("cross_level")

            if self.reporter:
                self.reporter.set_component_result("cross_level", detailed_results)

            # Save validator for later use
            self.cross_level_validator = validator

            return detailed_results

        except Exception as e:
            self.logger.error(f"Error in cross-level validation: {e}")
            traceback.print_exc()
            status = "Error"
            return None
        finally:
            self.logger.info(f"Cross-level validation completed: {status}")

    def run_dimensional_consistency_validation(self, dimensional_equations=None):
        """
        Run dimensional consistency validation.

        Parameters:
            dimensional_equations: Dictionary of dimensionally-validated equations

        Returns:
            Dimensional consistency validation results
        """
        logger.info("Running dimensional consistency validation")

        try:
            # Return warning if validator not initialized
            if self.dimensional_validator is None:
                logger.error("Dimensional validator not initialized")
                return None

            # Use provided equations if available
            if dimensional_equations is None:
                logger.warning("No dimensional equations provided, skipping validation")
                return None

            # Run validation
            consistency_results = self.dimensional_validator(dimensional_equations)

            # Generate visualizations
            try:
                from validation.validation_visualizers import create_dimensional_consistency_visualizations
                create_dimensional_consistency_visualizations(
                    consistency_results, output_dir=str(self.output_dir / "dimensional")
                )
                logger.info("Generated dimensional consistency visualizations")
            except ImportError:
                logger.warning("Could not generate dimensional consistency visualizations")

            # Store results
            inconsistent_count = sum(1 for res in consistency_results.values()
                                     if res.get('status') == 'INCONSISTENT')

            results = {
                "analyzed_equations": len(consistency_results),
                "inconsistent_count": inconsistent_count,
                "consistent_count": len(consistency_results) - inconsistent_count,
                "status": "Error" if inconsistent_count > 0 else "Success"
            }

            self.results["dimensional"] = results
            self.run_components.add("dimensional")

            if self.reporter:
                self.reporter.set_component_result("dimensional", consistency_results)

            logger.info(f"Dimensional consistency validation completed: {results['status']}")

            return consistency_results

        except Exception as e:
            logger.error(f"Error in dimensional consistency validation: {e}")
            traceback.print_exc()
            return None

    def run_historical_validation(self, simulation_func=None, parameter_ranges=None):
        """
        Run historical validation.

        Parameters:
            simulation_func: Function to run simulation
            parameter_ranges: Dictionary of parameter ranges for optimization

        Returns:
            Historical validation results
        """
        self.logger.info("Running historical validation")

        try:
            # Import historical validator
            try:
                from validation.historical_integration import integrate_historical_validation, create_dummy_simulation_function
            except ImportError:
                self.logger.error("Failed to import historical integration functions")
                return None

            # Use dummy function if none provided
            if simulation_func is None or not callable(simulation_func):
                self.logger.warning("No simulation function provided, using dummy function")
                simulation_func = create_dummy_simulation_function()

            # Create parameter ranges if not provided
            if parameter_ranges is None:
                parameter_ranges = {
                    'ALPHA_WISDOM': (0.05, 0.2),
                    'BETA_FEEDBACK': (0.01, 0.1),
                    'GAMMA_PHASE': (0.05, 0.2),
                    'LAMBDA_DECAY': (0.01, 0.1)
                }

            # Set up validation components - create instances where needed
            validation_components = {}
            if self.dim_handler:
                validation_components['dimension_handler'] = self.dim_handler
            if self.circuit_breaker:
                # Create instance if passing the class
                if isinstance(self.circuit_breaker, type):
                    validation_components['circuit_breaker'] = self.circuit_breaker()
                else:
                    validation_components['circuit_breaker'] = self.circuit_breaker
            if self.edge_case_checker:
                if isinstance(self.edge_case_checker, type):
                    validation_components['edge_case_checker'] = self.edge_case_checker({})
                else:
                    validation_components['edge_case_checker'] = self.edge_case_checker

            # Create validator
            validator = integrate_historical_validation(
                simulation_func, validation_components, parameter_ranges
            )

            # Find best parameters with error handling
            self.logger.info("Finding best parameters for historical fit")
            try:
                best_params = validator.find_best_parameters(max_evals=20)
            except Exception as e:
                self.logger.error(f"Error finding best parameters: {e}")
                # Create backup parameters
                best_params = {param: (min_val + max_val)/2 for param, (min_val, max_val) in parameter_ranges.items()}

            # Generate comparison visualizations with error handling
            self.logger.info("Generating historical comparison visualizations")
            try:
                errors = validator.compare_to_historical_data()
            except Exception as e:
                self.logger.error(f"Error generating comparisons: {e}")
                errors = {'overall': {'rmse': float('inf')}}

            # Generate report with error handling
            try:
                self.logger.info("Generating historical validation report")
                report_path = validator.generate_historical_validation_report()
            except Exception as e:
                self.logger.error(f"Error generating report: {e}")
                report_path = self.output_dir / "historical_report_error.txt"
                with open(report_path, 'w') as f:
                    f.write(f"Error generating report: {e}")

            # Store results
            overall_rmse = errors.get('overall', {}).get('rmse', float('inf'))
            results = {
                "best_parameters": best_params,
                "overall_rmse": overall_rmse,
                "report_path": str(report_path),
                "status": "Success" if overall_rmse < 10.0 else "Warning"
            }

            self.results["historical"] = results
            self.run_components.add("historical")

            if self.reporter:
                self.reporter.set_component_result("historical", errors)

            self.logger.info(f"Historical validation completed: {results['status']}")

            # Save validator for later use
            self.historical_validator = validator

            return errors

        except Exception as e:
            self.logger.error(f"Error in historical validation: {e}")
            import traceback
            traceback.print_exc()
            return None


    def run_full_validation(self, simulation_func=None, options=None):
        """
        Run the full validation suite.

        Parameters:
            simulation_func: Function to run simulation
            options: Dictionary of validation options

        Returns:
            Dictionary of validation results
        """
        logger.info("Running full validation suite")

        # Define default options
        default_options = {
            "dimension": True,
            "sensitivity": True,
            "edge_case": True,
            "cross_level": True,
            "dimensional": True,
            "historical": True
        }

        # Use provided options or defaults
        options = options or default_options

        # Load components if not already loaded
        if not self.dim_handler:
            success = self.load_components()
            if not success:
                return None

        # Load models and data
        loaded = self.load_models_and_data()

        # Define sample simulation function if not provided
        if simulation_func is None:
            logger.warning("No simulation function provided, using dummy function")

            # Define dummy simulation function
            def dummy_simulation(params):
                """Improved dummy simulation that properly handles parameters."""
                # Generate more stable random results
                timesteps = 100
                np.random.seed(42)  # Use fixed seed for reproducibility

                # Use parameters to influence the outputs rather than pure randomness
                param_sum = sum(params.values()) if params else 1.0
                scaling = max(0.1, min(10.0, param_sum))

                return {
                    "time": np.arange(timesteps),
                    "knowledge": np.random.rand(timesteps) * 10 * scaling,
                    "suppression": np.random.rand(timesteps) * 5 * scaling,
                    "intelligence": np.random.rand(timesteps) * 15 * scaling
                }

            simulation_func = dummy_simulation

        # Run individual validation components
        results = {}

        # 1. Run dimension validation
        if options.get("dimension", True):
            logger.info("Running dimension validation")
            dim_results = self.run_dimension_validation()
            results["dimension"] = dim_results

        # 2. Run sensitivity analysis
        if options.get("sensitivity", True) and "parameters" in loaded:
            logger.info("Running sensitivity analysis")

            # Get base parameters from loaded parameters module
            params_module = loaded["parameters"]
            base_parameters = {
                name: value for name, value in params_module.__dict__.items()
                if isinstance(value, (int, float)) and not name.startswith("_")
                   and name.isupper()  # Only include constants
            }

            # Define metrics to track
            metrics = ["knowledge", "suppression", "intelligence"]

            # Define parameter ranges (adjust based on your model)
            parameter_ranges = {
                "W_0": (0.5, 2.0, 5),
                "ALPHA_WISDOM": (0.05, 0.2, 5),
                "ALPHA_FEEDBACK": (0.05, 0.2, 5),
                "GAMMA_PHASE": (0.05, 0.2, 5)
            }

            sensitivity_results = self.run_sensitivity_analysis(
                simulation_func, base_parameters, metrics, parameter_ranges
            )
            results["sensitivity"] = sensitivity_results

        # 3. Run edge case detection
        if options.get("edge_case", True):
            logger.info("Running edge case detection")

            # Combine functions from all relevant modules
            equation_functions = {}

            # Add core equations
            if "equations" in loaded:
                for name in dir(loaded["equations"]):
                    if name.startswith("_"):
                        continue
                    item = getattr(loaded["equations"], name)
                    if callable(item):
                        equation_functions[name] = item

            # Add quantum EM extensions
            if "quantum_em_extensions" in loaded:
                for name in dir(loaded["quantum_em_extensions"]):
                    if name.startswith("_"):
                        continue
                    item = getattr(loaded["quantum_em_extensions"], name)
                    if callable(item):
                        equation_functions[name] = item

            edge_case_results = self.run_edge_case_detection(equation_functions)
            results["edge_case"] = edge_case_results

        # 4. Run cross-level validation
        if options.get("cross_level", True):
            logger.info("Running cross-level validation")

            # Use the same combined functions as for edge case detection
            equation_functions = {}

            # Add core equations
            if "equations" in loaded:
                for name in dir(loaded["equations"]):
                    if name.startswith("_"):
                        continue
                    item = getattr(loaded["equations"], name)
                    if callable(item):
                        equation_functions[name] = item

            # Add quantum EM extensions
            if "quantum_em_extensions" in loaded:
                for name in dir(loaded["quantum_em_extensions"]):
                    if name.startswith("_"):
                        continue
                    item = getattr(loaded["quantum_em_extensions"], name)
                    if callable(item):
                        equation_functions[name] = item

            # Define hierarchy levels (adjust based on your model)
            hierarchy_levels = {
                'Level 1 (Core)': [
                    'intelligence_growth', 'free_will_decision',
                    'truth_adoption', 'wisdom_field'
                ],
                'Level 2 (Extended)': [
                    'suppression_feedback', 'resistance_resurgence'
                ],
                'Level 3 (Quantum)': [
                    'quantum_tunneling_probability', 'knowledge_field_influence',
                    'quantum_entanglement_correlation', 'knowledge_field_gradient',
                    'build_entanglement_network'
                ]
            }

            cross_level_results = self.run_cross_level_validation(
                equation_functions, hierarchy_levels
            )
            results["cross_level"] = cross_level_results

        # 5. Run dimensional consistency validation
        if options.get("dimensional", True):
            logger.info("Running dimensional consistency validation")

            # We need dimensionally-validated equations for this
            # In a real implementation, you would have these available
            # For now, we'll skip this step unless specific equations are provided
            dimensional_results = self.run_dimensional_consistency_validation()
            results["dimensional"] = dimensional_results

        # 6. Run historical validation
        if options.get("historical", True) and "historical_data" in loaded:
            logger.info("Running historical validation")

            # Define parameter ranges for optimization (adjust based on your model)
            parameter_ranges = {
                "ALPHA_WISDOM": (0.05, 0.2),
                "BETA_FEEDBACK": (0.01, 0.1),
                "GAMMA_PHASE": (0.05, 0.2),
                "LAMBDA_DECAY": (0.01, 0.1)
            }

            historical_results = self.run_historical_validation(
                simulation_func, parameter_ranges
            )
            results["historical"] = historical_results

        # Generate comprehensive report
        if self.reporter:
            logger.info("Generating comprehensive validation report")
            report_path = self.reporter.generate_report()
            logger.info(f"Report generated: {report_path}")
            results["report_path"] = str(report_path)

        logger.info(f"Full validation completed: {len(results)} components run")
        return results


def run_validation():
    """
    Run the validation suite from the command line.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the validation suite")
    parser.add_argument("--output-dir", default="validation/reports", help="Output directory for reports")
    parser.add_argument("--component",
                        choices=["all", "dimension", "sensitivity", "edge_case", "cross_level", "dimensional",
                                 "historical"],
                        default="all", help="Component to run")
    parser.add_argument("--simulation-func", help="Path to simulation function")
    parser.add_argument("--skip-report", action="store_true", help="Skip report generation")

    args = parser.parse_args()

    # Create validation suite
    suite = ValidationSuite(output_dir=args.output_dir)

    # Load components
    success = suite.load_components()
    if not success:
        logger.error("Failed to load validation components")
        return False

    # Load simulation function if provided
    simulation_func = None
    if args.simulation_func:
        try:
            # Try to import the function
            module_path, func_name = args.simulation_func.rsplit(".", 1)
            module = importlib.import_module(module_path)
            simulation_func = getattr(module, func_name)
            logger.info(f"Loaded simulation function: {args.simulation_func}")
        except Exception as e:
            logger.error(f"Error loading simulation function: {e}")
            return False

    # Define validation options
    options = {
        "dimension": args.component in ["all", "dimension"],
        "sensitivity": args.component in ["all", "sensitivity"],
        "edge_case": args.component in ["all", "edge_case"],
        "cross_level": args.component in ["all", "cross_level"],
        "dimensional": args.component in ["all", "dimensional"],
        "historical": args.component in ["all", "historical"]
    }

    # Run validation
    results = suite.run_full_validation(simulation_func, options)

    # Generate report unless skipped
    if not args.skip_report and suite.reporter:
        report_path = suite.reporter.generate_report()
        logger.info(f"Final report generated: {report_path}")

    return results is not None


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)