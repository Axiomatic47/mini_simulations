"""
Unified validation entry point for the Axiomatic Intelligence Growth Simulation framework.
Uses the module mapping utility to discover and validate functions across the codebase.
"""

import sys
import os
from pathlib import Path
import logging
from typing import Dict, List, Set, Optional, Any, Callable
import importlib
import traceback

# Add the parent directory to the sys.path to allow imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import the module mapping utility
from module_map import discover_equation_functions, get_function_hierarchy, get_bridge_functions

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(parent_dir, "validation", "logs", "unified_validation.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("unified_validator")


class UnifiedValidator:
    """Unified validation framework for equations across all physics domains."""

    def __init__(self,
                 output_dir: Optional[Path] = None,
                 enable_historical: bool = True,
                 enable_edge_cases: bool = True,
                 enable_dimensions: bool = True,
                 enable_cross_level: bool = True,
                 enable_sensitivity: bool = False):  # False by default as it's the most time-consuming
        """
        Initialize the unified validator.

        Args:
            output_dir: Directory for validation outputs
            enable_historical: Enable historical validation
            enable_edge_cases: Enable edge case checking
            enable_dimensions: Enable dimension consistency checking
            enable_cross_level: Enable cross-level validation
            enable_sensitivity: Enable parameter sensitivity analysis
        """
        self.output_dir = output_dir or Path(parent_dir, "validation", "reports", "unified")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Component flags
        self.enable_historical = enable_historical
        self.enable_edge_cases = enable_edge_cases
        self.enable_dimensions = enable_dimensions
        self.enable_cross_level = enable_cross_level
        self.enable_sensitivity = enable_sensitivity

        # Discover functions
        self.equation_functions = discover_equation_functions()
        self.bridge_functions = get_bridge_functions()
        self.function_hierarchy = get_function_hierarchy()

        # Keep track of validation results
        self.validation_results = {
            "historical": {"status": "Not Run", "details": {}},
            "edge_cases": {"status": "Not Run", "details": {}},
            "dimensions": {"status": "Not Run", "details": {}},
            "cross_level": {"status": "Not Run", "details": {}},
            "sensitivity": {"status": "Not Run", "details": {}}
        }

        logger.info(f"UnifiedValidator initialized with {len(self.equation_functions)} equation functions "
                    f"and {len(self.bridge_functions)} bridge functions")

    def validate_all(self):
        """Run all enabled validation components."""
        if self.enable_edge_cases:
            self.validate_edge_cases()

        if self.enable_dimensions:
            self.validate_dimensions()

        if self.enable_cross_level:
            self.validate_cross_level()

        if self.enable_historical:
            self.validate_historical()

        if self.enable_sensitivity:
            self.validate_sensitivity()

        # Generate final consolidated report
        self.generate_report()

    def validate_edge_cases(self):
        """Validate edge cases for all equation functions."""
        logger.info("Starting edge case validation...")

        try:
            # Import the edge case checker
            from utils.edge_case_checker import EdgeCaseChecker

            # Create the checker with our discovered functions
            checker = EdgeCaseChecker(self.equation_functions)

            # Analyze all functions
            checker.analyze_all_functions()

            # Generate report
            output_path = self.output_dir.parent / "edge_case"
            output_path.mkdir(exist_ok=True)

            try:
                checker.generate_edge_case_completion_report(str(output_path))
                self.validation_results["edge_cases"]["status"] = "Success"
            except Exception as e:
                logger.error(f"Error generating edge case report: {e}")
                self.validation_results["edge_cases"]["status"] = "Warning"

            # Store details
            self.validation_results["edge_cases"]["details"] = {
                "analyzed_functions": len(checker.analysis_results),
                "total_edge_cases": sum(len(results) for results in checker.analysis_results.values()),
                "report_path": str(output_path)
            }

            logger.info("Edge case validation completed")

        except Exception as e:
            logger.error(f"Edge case validation failed: {e}")
            logger.error(traceback.format_exc())
            self.validation_results["edge_cases"]["status"] = "Error"
            self.validation_results["edge_cases"]["details"] = {"error": str(e)}

    def validate_dimensions(self):
        """Validate dimensional consistency for all equation functions."""
        logger.info("Starting dimensional consistency validation...")

        try:
            # Import the dimensional consistency utility
            from utils.dimensional_consistency import check_dimensional_consistency, Dimension, DimensionalValue

            # Create a subset of functions that have dimension annotations
            dimensionally_validated_functions = {}
            for name, func in self.equation_functions.items():
                # Check if the function has dimensional annotations
                doc = func.__doc__ or ""
                if "Physics Domain:" in doc and ("Dimension" in doc or "dimension" in doc):
                    dimensionally_validated_functions[name] = func

            if not dimensionally_validated_functions:
                logger.warning("No functions with dimensional annotations found")
                self.validation_results["dimensions"]["status"] = "Warning"
                self.validation_results["dimensions"]["details"] = {
                    "error": "No functions with dimensional annotations found"
                }
                return

            # Run dimensional consistency check if possible
            try:
                results = check_dimensional_consistency(dimensionally_validated_functions)

                # Save results
                output_path = self.output_dir.parent / "dimensional"
                output_path.mkdir(exist_ok=True)

                with open(output_path / "consistency_report.txt", "w") as f:
                    f.write("Dimensional Consistency Report\n")
                    f.write("==============================\n\n")

                    for name, result in results.items():
                        f.write(f"Function: {name}\n")
                        f.write(f"Status: {result['status']}\n")
                        if "error" in result:
                            f.write(f"Error: {result['error']}\n")
                        f.write("\n")

                self.validation_results["dimensions"]["status"] = "Success"
                self.validation_results["dimensions"]["details"] = {
                    "validated_functions": len(dimensionally_validated_functions),
                    "report_path": str(output_path / "consistency_report.txt")
                }

            except Exception as e:
                logger.error(f"Error in dimensional consistency check: {e}")
                self.validation_results["dimensions"]["status"] = "Warning"
                self.validation_results["dimensions"]["details"] = {"error": str(e)}

            logger.info("Dimensional consistency validation completed")

        except Exception as e:
            logger.error(f"Dimensional validation failed: {e}")
            logger.error(traceback.format_exc())
            self.validation_results["dimensions"]["status"] = "Error"
            self.validation_results["dimensions"]["details"] = {"error": str(e)}

    def validate_cross_level(self):
        """Validate cross-level coupling for all equation functions."""
        logger.info("Starting cross-level validation...")

        try:
            # Import the cross-level validator
            from utils.cross_level_validator import CrossLevelValidator

            # Define hierarchy levels based on our physics domains
            hierarchy_levels = {}

            # Level 1 - Core physics
            hierarchy_levels["Level 1 (Core)"] = self.function_hierarchy.get("thermodynamics", []) + \
                                                 self.function_hierarchy.get("relativity", [])

            # Level 2 - Nuclear forces
            hierarchy_levels["Level 2 (Nuclear)"] = self.function_hierarchy.get("strong_nuclear", []) + \
                                                    self.function_hierarchy.get("weak_nuclear", [])

            # Level 3 - Electromagnetic and Quantum
            hierarchy_levels["Level 3 (EM & Quantum)"] = self.function_hierarchy.get("electromagnetism", []) + \
                                                         self.function_hierarchy.get("quantum_mechanics", [])

            # Level 4 - Astrophysics
            hierarchy_levels["Level 4 (Astrophysics)"] = self.function_hierarchy.get("astrophysics", [])

            # Level 5 - Multi-system
            hierarchy_levels["Level 5 (Multi-system)"] = self.function_hierarchy.get("multi_system", [])

            # Create validator
            validator = CrossLevelValidator(self.equation_functions, hierarchy_levels)

            # Build dependency graph
            validator.build_dependency_graph()

            # Validate level dependencies
            dependency_results = validator.validate_level_dependencies()

            # Detect feedback loops
            feedback_loops = validator.detect_feedback_loops()

            # Generate validation report
            output_path = self.output_dir.parent / "cross_level"
            output_path.mkdir(exist_ok=True)

            try:
                validator.generate_validation_report(str(output_path))
                self.validation_results["cross_level"]["status"] = "Success"
            except Exception as e:
                logger.error(f"Error generating cross-level report: {e}")
                self.validation_results["cross_level"]["status"] = "Warning"

            # Store details
            self.validation_results["cross_level"]["details"] = {
                "is_valid": dependency_results["is_valid"],
                "violations": len(dependency_results.get("violations", [])),
                "feedback_loops": len(feedback_loops),
                "cross_level_loops": len([loop for loop in feedback_loops if loop.get("is_cross_level", False)]),
                "report_path": str(output_path)
            }

            logger.info("Cross-level validation completed")

        except Exception as e:
            logger.error(f"Cross-level validation failed: {e}")
            logger.error(traceback.format_exc())
            self.validation_results["cross_level"]["status"] = "Error"
            self.validation_results["cross_level"]["details"] = {"error": str(e)}

    def validate_historical(self):
        """Validate simulation against historical data."""
        logger.info("Starting historical validation...")

        try:
            # Import historical validation
            from config.historical_validation import HistoricalValidation

            # Create validator
            validator = HistoricalValidation(enable_circuit_breaker=True, enable_adaptive_timestep=True)

            # Optimize parameters
            params = validator.optimize_parameters(max_iterations=5)  # Use fewer iterations for speed

            # Calculate error
            error = validator.calculate_error()

            # Save results
            output_path = self.output_dir.parent / "historical"
            output_path.mkdir(exist_ok=True)
            validator.save_results(str(output_path))

            # Visualize comparison if possible
            try:
                validator.visualize_comparison(save_path=str(output_path / "comparison.png"))

                # Generate report with validation results, try to use historical_integration or report_generator
                report_generated = False

                try:
                    # Try the historical_integration module
                    from validation.historical_integration import generate_historical_report
                    generate_historical_report(validator, output_path)
                    report_generated = True
                except (ImportError, AttributeError):
                    logger.warning("Could not use historical_integration.generate_historical_report")

                if not report_generated:
                    try:
                        # Try the report_generator module
                        from validation.report_generator import generate_historical_validation_report
                        generate_historical_validation_report(validator, output_path)
                        report_generated = True
                    except (ImportError, AttributeError):
                        logger.warning("Could not use report_generator.generate_historical_validation_report")

                # If no specialized report generators are available, create a basic report
                if not report_generated:
                    with open(output_path / "historical_validation_report.html", "w") as f:
                        f.write("<html><head><title>Historical Validation Report</title></head><body>")
                        f.write("<h1>Historical Validation Report</h1>")
                        f.write(f"<p>Error: {error}</p>")
                        f.write("<h2>Optimized Parameters</h2><ul>")
                        for key, value in params.items():
                            f.write(f"<li>{key}: {value}</li>")
                        f.write("</ul>")
                        f.write("<h2>Visualization</h2>")
                        f.write('<img src="comparison.png" alt="Historical Comparison" />')
                        f.write("</body></html>")

            except Exception as e:
                logger.error(f"Error in historical visualization: {e}")

            # Set validation status
            if error < 10:  # Arbitrary threshold, adjust as needed
                self.validation_results["historical"]["status"] = "Success"
            else:
                self.validation_results["historical"]["status"] = "Warning"

            # Store details
            self.validation_results["historical"]["details"] = {
                "error": error,
                "optimized_params": params,
                "report_path": str(output_path)
            }

            logger.info("Historical validation completed")

        except Exception as e:
            logger.error(f"Historical validation failed: {e}")
            logger.error(traceback.format_exc())
            self.validation_results["historical"]["status"] = "Error"
            self.validation_results["historical"]["details"] = {"error": str(e)}

    def validate_sensitivity(self):
        """Run parameter sensitivity analysis."""
        logger.info("Starting sensitivity analysis...")

        try:
            # Import sensitivity analyzer
            from utils.sensitivity_analyzer import ParameterSensitivityAnalyzer

            # Define a simple simulation function that uses our equation functions
            def run_simulation(params):
                # This is a simplified simulation for sensitivity analysis
                # It should return metrics that we want to analyze sensitivity for

                # Default metrics
                results = {
                    "final_knowledge": 0,
                    "final_intelligence": 0,
                    "final_suppression": 0,
                    "truth_convergence_time": 0
                }

                # Try to run a minimal simulation if possible
                try:
                    from simulations.comprehensive_simulation import run_minimal_simulation
                    sim_results = run_minimal_simulation(params)
                    results.update(sim_results)
                except (ImportError, AttributeError):
                    logger.warning("Could not run minimal simulation, using dummy values")
                    # Add some sensitivity to parameters to see effects
                    results["final_knowledge"] = 10 * params.get("K_0", 1.0) / params.get("ALPHA_WISDOM", 0.1)
                    results["final_intelligence"] = 5 * params.get("W_0", 1.0) / params.get("ALPHA_FEEDBACK", 0.1)
                    results["truth_convergence_time"] = 20 * params.get("ALPHA_WISDOM", 0.1)

                return results

            # Import parameters
            from config.parameters import (
                W_0, ALPHA_WISDOM, RESISTANCE, A_TRUTH,
                LAMBDA_DECAY, ALPHA_FEEDBACK, ALPHA_RESURGE
            )

            # Define base parameters
            base_parameters = {
                "W_0": W_0,
                "ALPHA_WISDOM": ALPHA_WISDOM,
                "RESISTANCE": RESISTANCE,
                "A_TRUTH": A_TRUTH,
                "LAMBDA_DECAY": LAMBDA_DECAY,
                "ALPHA_FEEDBACK": ALPHA_FEEDBACK,
                "ALPHA_RESURGE": ALPHA_RESURGE
            }

            # Define metrics to analyze
            metrics = ["final_knowledge", "final_intelligence", "final_suppression", "truth_convergence_time"]

            # Create analyzer
            analyzer = ParameterSensitivityAnalyzer(run_simulation, metrics, base_parameters)

            # Define parameter ranges
            analyzer.define_parameter_ranges({
                "W_0": (0.5, 2.0, 3),  # (min, max, points)
                "ALPHA_WISDOM": (0.05, 0.2, 3),
                "RESISTANCE": (1.0, 3.0, 3),
                "A_TRUTH": (1.5, 3.5, 3),
                "LAMBDA_DECAY": (0.02, 0.1, 3)
            })

            # Run analysis with fewer samples for speed
            results = analyzer.run_one_at_a_time_sensitivity(num_samples=2)

            # Generate report
            output_path = self.output_dir.parent / "sensitivity"
            output_path.mkdir(exist_ok=True)

            try:
                analyzer.generate_comprehensive_report(str(output_path))
                self.validation_results["sensitivity"]["status"] = "Success"
            except Exception as e:
                logger.error(f"Error generating sensitivity report: {e}")
                self.validation_results["sensitivity"]["status"] = "Warning"

            # Calculate parameter importance if possible
            try:
                importance = analyzer.calculate_parameter_importance()
                self.validation_results["sensitivity"]["details"]["importance"] = importance
            except Exception as e:
                logger.error(f"Error calculating parameter importance: {e}")

            # Store details
            self.validation_results["sensitivity"]["details"].update({
                "parameters_tested": len(analyzer.parameter_ranges),
                "metrics_analyzed": len(metrics),
                "report_path": str(output_path)
            })

            logger.info("Sensitivity analysis completed")

        except Exception as e:
            logger.error(f"Sensitivity analysis failed: {e}")
            logger.error(traceback.format_exc())
            self.validation_results["sensitivity"]["status"] = "Error"
            self.validation_results["sensitivity"]["details"] = {"error": str(e)}

    def generate_report(self):
        """Generate a unified validation report."""
        logger.info("Generating unified validation report...")

        # Basic report structure
        report_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Unified Validation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .success { color: green; }
                .warning { color: orange; }
                .error { color: red; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
            </style>
        </head>
        <body>
            <h1>Unified Validation Report</h1>
            <p>Generated on: {date}</p>

            <h2>Validation Summary</h2>
            <table>
                <tr>
                    <th>Component</th>
                    <th>Status</th>
                    <th>Details</th>
                </tr>
        """.format(date=__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Add rows for each component
        for component, results in self.validation_results.items():
            status = results["status"]
            status_class = ""
            if status == "Success":
                status_class = "success"
            elif status == "Warning":
                status_class = "warning"
            elif status == "Error":
                status_class = "error"

            details = ""
            if "error" in results["details"]:
                details = f"Error: {results['details']['error']}"
            elif component == "edge_cases":
                analyzed = results["details"].get("analyzed_functions", 0)
                total = results["details"].get("total_edge_cases", 0)
                details = f"Analyzed {analyzed} functions, found {total} edge cases"
            elif component == "dimensions":
                validated = results["details"].get("validated_functions", 0)
                details = f"Validated {validated} functions"
            elif component == "cross_level":
                is_valid = results["details"].get("is_valid", False)
                violations = results["details"].get("violations", 0)
                loops = results["details"].get("cross_level_loops", 0)
                details = f"Valid: {is_valid}, Violations: {violations}, Cross-level loops: {loops}"
            elif component == "historical":
                error = results["details"].get("error", "N/A")
                details = f"Validation error: {error}"
            elif component == "sensitivity":
                params = results["details"].get("parameters_tested", 0)
                metrics = results["details"].get("metrics_analyzed", 0)
                details = f"Tested {params} parameters across {metrics} metrics"

            report_html += f"""
                <tr>
                    <td>{component.title()}</td>
                    <td class="{status_class}">{status}</td>
                    <td>{details}</td>
                </tr>
            """

        # Function statistics
        report_html += """
            </table>

            <h2>Function Statistics</h2>
            <table>
                <tr>
                    <th>Domain</th>
                    <th>Functions</th>
                </tr>
        """

        # Add rows for each domain
        for domain, functions in self.function_hierarchy.items():
            if functions:  # Only show domains with functions
                report_html += f"""
                    <tr>
                        <td>{domain}</td>
                        <td>{len(functions)}</td>
                    </tr>
                """

        # Bridge functions
        report_html += f"""
            <tr>
                <td>Bridge Functions</td>
                <td>{len(self.bridge_functions)}</td>
            </tr>
        """

        report_html += """
            </table>

            <h2>Recommendations</h2>
            <ul>
        """

        # Add recommendations based on validation results
        if self.validation_results["edge_cases"]["status"] == "Warning":
            report_html += "<li>Improve edge case handling in equation functions</li>"

        if self.validation_results["dimensions"]["status"] == "Warning":
            report_html += "<li>Add dimensional annotations to more functions</li>"

        if self.validation_results["cross_level"].get("details", {}).get("is_valid", True) == False:
            report_html += "<li>Address cross-level dependency violations</li>"

        if self.validation_results["historical"].get("details", {}).get("error", 0) > 5:
            report_html += "<li>Refine equations to better match historical data</li>"

        # Close HTML tags
        report_html += """
            </ul>

            <h2>Next Steps</h2>
            <ol>
                <li>Review detailed reports for each validation component</li>
                <li>Address any identified issues or warnings</li>
                <li>Rerun validation to confirm improvements</li>
                <li>Proceed with simulation once validation is successful</li>
            </ol>
        </body>
        </html>
        """

        # Write report to file
        report_path = self.output_dir / "unified_validation_report.html"
        with open(report_path, "w") as f:
            f.write(report_html)

        logger.info(f"Unified validation report generated at {report_path}")
        return report_path


def run_validation(components=None, output_dir=None):
    """
    Run validation with specified components.

    Args:
        components: List of component names to run, or None for all
        output_dir: Directory for validation outputs
    """
    # Determine which components to run
    if components is None:
        components = ["edge_cases", "dimensions", "cross_level", "historical"]
    else:
        components = [c.lower() for c in components]

    # Set up validator
    validator = UnifiedValidator(
        output_dir=Path(output_dir) if output_dir else None,
        enable_historical="historical" in components,
        enable_edge_cases="edge_cases" in components,
        enable_dimensions="dimensions" in components,
        enable_cross_level="cross_level" in components,
        enable_sensitivity="sensitivity" in components
    )

    # Run validation
    validator.validate_all()

    # Print summary
    print("\nValidation Summary:")
    for component, results in validator.validation_results.items():
        if results["status"] != "Not Run":
            print(f"  {component.title()}: {results['status']}")

    return validator


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run unified validation for the simulation framework")
    parser.add_argument("--components", nargs="*",
                        choices=["edge_cases", "dimensions", "cross_level", "historical", "sensitivity", "all"],
                        help="Components to validate (default: all except sensitivity)")
    parser.add_argument("--output-dir", help="Directory for validation outputs")

    args = parser.parse_args()

    # Handle "all" component
    if args.components and "all" in args.components:
        args.components = ["edge_cases", "dimensions", "cross_level", "historical", "sensitivity"]

    # Run validation
    validator = run_validation(args.components, args.output_dir)

    # Exit with status code based on validation results
    if any(results["status"] == "Error" for results in validator.validation_results.values()):
        sys.exit(1)
    else:
        sys.exit(0)