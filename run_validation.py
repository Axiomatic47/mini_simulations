#!/usr/bin/env python
"""
Unified Validation Integration Runner
This script runs the complete equation validation and optimization process,
integrating all validation components into a single workflow.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
import importlib

# Add project root to path to ensure imports work
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("validation/logs/unified_validation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("IntegrationRunner")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run unified equation validation and optimization.")

    parser.add_argument(
        "--output-dir",
        default="validation/reports/unified",
        help="Directory for output files (default: validation/reports/unified)"
    )

    parser.add_argument(
        "--modules",
        nargs="+",
        default=["equations", "astrophysics_extensions", "quantum_em_extensions", "multi_civilization_extensions"],
        help="Equation modules to validate (default: all modules)"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick validation without simulations (faster but less comprehensive)"
    )

    parser.add_argument(
        "--focus",
        choices=["coverage", "cross-scale", "simulation", "all"],
        default="all",
        help="Focus validation on a specific aspect (default: all)"
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=300,
        help="Number of timesteps for simulations (default: 300)"
    )

    parser.add_argument(
        "--generate-plan",
        action="store_true",
        help="Generate optimization plan (default: True)"
    )

    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Only generate optimization plan without running validation"
    )

    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Skip generating visualizations (faster)"
    )

    return parser.parse_args()


def import_validator_components():
    """Import the various validator components."""
    components = {}

    try:
        # Import UnifiedValidator
        from validation.unified_validator import UnifiedValidator
        components["unified_validator"] = UnifiedValidator
    except ImportError:
        logger.warning("Could not import UnifiedValidator, will create it")
        # We'll create it below if needed

    try:
        # Import EquationCoverageAnalyzer
        from validation.equation_coverage_analyzer import EquationCoverageAnalyzer
        components["equation_analyzer"] = EquationCoverageAnalyzer
    except ImportError:
        logger.warning("Could not import EquationCoverageAnalyzer, will create it")

        # Try to create it
        try:
            # Create module directory
            Path("validation/equation_coverage_analyzer.py").parent.mkdir(parents=True, exist_ok=True)

            # Copy file content from artifacts/equation-coverage-analyzer.py if it exists
            analyzer_path = Path("artifacts/equation-coverage-analyzer.py")
            if analyzer_path.exists():
                with open(analyzer_path, "r") as f_in:
                    content = f_in.read()

                with open("validation/equation_coverage_analyzer.py", "w") as f_out:
                    f_out.write(content)

                logger.info("Created equation_coverage_analyzer.py from artifact")

                # Import it
                from validation.equation_coverage_analyzer import EquationCoverageAnalyzer
                components["equation_analyzer"] = EquationCoverageAnalyzer
            else:
                logger.error("Could not create EquationCoverageAnalyzer, analyzer artifact not found")
        except Exception as e:
            logger.error(f"Error creating EquationCoverageAnalyzer: {e}")

    try:
        # Import CrossScaleValidator
        from validation.cross_scale_validator import CrossScaleValidator
        components["cross_validator"] = CrossScaleValidator
    except ImportError:
        logger.warning("Could not import CrossScaleValidator, will create it")

        # Try to create it
        try:
            # Copy file content from artifacts/cross-scale-validator.py if it exists
            validator_path = Path("artifacts/cross-scale-validator.py")
            if validator_path.exists():
                with open(validator_path, "r") as f_in:
                    content = f_in.read()

                with open("validation/cross_scale_validator.py", "w") as f_out:
                    f_out.write(content)

                logger.info("Created cross_scale_validator.py from artifact")

                # Import it
                from validation.cross_scale_validator import CrossScaleValidator
                components["cross_validator"] = CrossScaleValidator
            else:
                logger.error("Could not create CrossScaleValidator, validator artifact not found")
        except Exception as e:
            logger.error(f"Error creating CrossScaleValidator: {e}")

    try:
        # Import ComparativeSimulationAnalyzer
        from validation.comparative_analyzer import ComparativeSimulationAnalyzer
        components["comparative_analyzer"] = ComparativeSimulationAnalyzer
    except ImportError:
        logger.warning("Could not import ComparativeSimulationAnalyzer, will create it")

        # Try to create it
        try:
            # Copy file content from artifacts/comparative-simulation-analyzer.py if it exists
            analyzer_path = Path("artifacts/comparative-simulation-analyzer.py")
            if analyzer_path.exists():
                with open(analyzer_path, "r") as f_in:
                    content = f_in.read()

                with open("validation/comparative_analyzer.py", "w") as f_out:
                    f_out.write(content)

                logger.info("Created comparative_analyzer.py from artifact")

                # Import it
                from validation.comparative_analyzer import ComparativeSimulationAnalyzer
                components["comparative_analyzer"] = ComparativeSimulationAnalyzer
            else:
                logger.error("Could not create ComparativeSimulationAnalyzer, analyzer artifact not found")
        except Exception as e:
            logger.error(f"Error creating ComparativeSimulationAnalyzer: {e}")

    try:
        # Import ReportGenerator
        from validation.report_generator import ReportGenerator
        components["report_generator"] = ReportGenerator
    except ImportError:
        logger.warning("Could not import ReportGenerator, will create it")

        # Try to create it
        try:
            # Copy file content from artifacts/report-generator.py if it exists
            generator_path = Path("artifacts/report-generator.py")
            if generator_path.exists():
                with open(generator_path, "r") as f_in:
                    content = f_in.read()

                with open("validation/report_generator.py", "w") as f_out:
                    f_out.write(content)

                logger.info("Created report_generator.py from artifact")

                # Import it
                from validation.report_generator import ReportGenerator
                components["report_generator"] = ReportGenerator
            else:
                logger.error("Could not create ReportGenerator, generator artifact not found")
        except Exception as e:
            logger.error(f"Error creating ReportGenerator: {e}")

    # Check if we need to create UnifiedValidator from scratch
    if "unified_validator" not in components:
        try:
            # Copy file content from artifacts/unified-validation-controller.py if it exists
            controller_path = Path("artifacts/unified-validation-controller.py")
            if controller_path.exists():
                with open(controller_path, "r") as f_in:
                    content = f_in.read()

                with open("validation/unified_validator.py", "w") as f_out:
                    f_out.write(content)

                logger.info("Created unified_validator.py from artifact")

                # Import it
                from validation.unified_validator import UnifiedValidator
                components["unified_validator"] = UnifiedValidator
            else:
                logger.error("Could not create UnifiedValidator, controller artifact not found")
        except Exception as e:
            logger.error(f"Error creating UnifiedValidator: {e}")

    return components


def run_validation(args):
    """Run the unified validation process with the specified arguments."""
    logger.info("Starting unified validation process")

    # Import validator components
    components = import_validator_components()

    # Check if we have the necessary components
    if "unified_validator" in components:
        # Use the unified validator
        logger.info("Using UnifiedValidator")
        validator = components["unified_validator"](output_dir=args.output_dir)

        # Set focus areas based on args
        focus_areas = []
        if args.focus == "all" or args.focus == "coverage":
            focus_areas.append("equation_completeness")
        if args.focus == "all" or args.focus == "cross-scale":
            focus_areas.append("cross_level_interaction")
        if args.focus == "all" or args.focus == "simulation":
            focus_areas.append("parameter_sensitivity")

        # Run full validation
        results = validator.run_full_validation(
            equation_modules=args.modules,
            focus_areas=focus_areas,
            generate_report=True
        )

        # If requested, generate optimization plan
        if args.generate_plan:
            plan = validator.generate_optimization_plan()
            logger.info(f"Optimization plan generated at {validator.output_dir}/optimization_plan.md")

        # Return results for possible further processing
        return results

    else:
        # If UnifiedValidator is not available, use individual components
        logger.info("Using individual validator components")

        # Initialize results dictionary
        results = {}

        # Run equation coverage analysis if available
        if "equation_analyzer" in components and (args.focus == "all" or args.focus == "coverage"):
            logger.info("Running equation coverage analysis")
            analyzer = components["equation_analyzer"]()
            results["equation_coverage"] = analyzer.analyze_equation_set(
                equation_modules=args.modules,
                identify_gaps=True
            )

            # Visualize coverage if requested
            if not args.no_visualizations:
                analyzer.visualize_coverage(results["equation_coverage"]["coverage"],
                                            f"{args.output_dir}/equation_coverage.png")

                if results["equation_coverage"]["gaps"]:
                    analyzer.visualize_gaps(results["equation_coverage"]["gaps"],
                                            f"{args.output_dir}/equation_gaps.png")

        # Run cross-scale interaction validation if available
        if "cross_validator" in components and (args.focus == "all" or args.focus == "cross-scale"):
            logger.info("Running cross-scale interaction validation")
            validator = components["cross_validator"]()
            results["cross_scale"] = validator.evaluate_cross_scale_interactions(
                levels=None,
                key_transitions=None,
                source_modules=args.modules
            )

            # Visualize cross-scale interactions if requested
            if not args.no_visualizations:
                validator.visualize_dependency_graph(results["cross_scale"]["dependency_graph"],
                                                     f"{args.output_dir}/dependency_graph.png")
                validator.visualize_scale_adjacency(results["cross_scale"]["dependency_graph"],
                                                    f"{args.output_dir}/scale_adjacency.png")
                validator.visualize_signal_propagation(results["cross_scale"]["signal_propagation"],
                                                       f"{args.output_dir}/signal_propagation.png")

        # Run comparative simulation analysis if available
        if "comparative_analyzer" in components and (
                args.focus == "all" or args.focus == "simulation") and not args.quick:
            logger.info("Running comparative simulation analysis")
            analyzer = components["comparative_analyzer"](output_dir=args.output_dir)

            # Define configurations
            configurations = [
                {"name": "base", "description": "Core equations only"},
                {"name": "quantum", "description": "With quantum effects"},
                {"name": "astrophysics", "description": "With astrophysics analogies"},
                {"name": "multi_civ", "description": "With multi-civilization dynamics"},
                {"name": "integrated", "description": "All extensions integrated"}
            ]

            # Run analysis
            results["comparative"] = analyzer.run_comparative_analysis(
                configurations=configurations,
                metrics=None,
                timesteps=args.timesteps
            )

        # Generate comprehensive report if report generator is available
        if "report_generator" in components:
            logger.info("Generating comprehensive report")
            generator = components["report_generator"]()

            # Create output directory if it doesn't exist
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)

            # Generate report
            report_path = generator.generate_unified_report(
                results,
                f"{args.output_dir}/unified_validation_report.html",
                include_visualizations=not args.no_visualizations
            )

            logger.info(f"Comprehensive report generated at {report_path}")

        # Generate optimization plan if requested
        if args.generate_plan:
            logger.info("Generating optimization plan")

            # Identify optimization opportunities
            opportunities = identify_optimization_opportunities(results)
            results["opportunities"] = opportunities

            # Generate plan
            plan = generate_optimization_plan(results, f"{args.output_dir}/optimization_plan.md")

            logger.info(f"Optimization plan generated at {args.output_dir}/optimization_plan.md")

        return results


def identify_optimization_opportunities(results):
    """
    Identify opportunities for equation optimization based on validation results.

    Args:
        results: Dictionary containing all validation results

    Returns:
        Dictionary of optimization opportunities
    """
    opportunities = {
        "equation_gaps": [],
        "cross_scale_improvements": [],
        "stability_enhancements": [],
        "parameter_optimizations": [],
        "integration_opportunities": []
    }

    # Identify equation gaps
    if "equation_coverage" in results and "gaps" in results["equation_coverage"]:
        for gap in results["equation_coverage"]["gaps"]:
            severity = gap.get("severity", "medium")
            priority = "High" if severity == "high" else ("Medium" if severity == "medium" else "Low")

            opportunities["equation_gaps"].append({
                "description": gap.get("description", ""),
                "priority": priority,
                "recommendation": gap.get("recommendation", "Fill this gap with an appropriate equation")
            })

    # Identify cross-scale improvements
    if "cross_scale" in results:
        # Check transitions
        if "scale_transitions" in results["cross_scale"] and "transitions" in results["cross_scale"][
            "scale_transitions"]:
            transitions = results["cross_scale"]["scale_transitions"]["transitions"]

            for (scale1, scale2), details in transitions.items():
                quality = details.get("quality", 0)
                if quality < 0.4:
                    priority = "High" if quality < 0.2 else "Medium"

                    opportunities["cross_scale_improvements"].append({
                        "transition": (scale1, scale2),
                        "quality": quality,
                        "priority": priority,
                        "recommendation": f"Improve integration between {scale1} and {scale2} scales"
                    })

        # Check key transitions
        if "transition_quality" in results["cross_scale"]:
            for (eq1, eq2), quality in results["cross_scale"]["transition_quality"].items():
                if quality < 0.4:
                    priority = "High" if quality < 0.2 else "Medium"

                    opportunities["cross_scale_improvements"].append({
                        "transition": (eq1, eq2),
                        "quality": quality,
                        "priority": priority,
                        "recommendation": f"Strengthen connection between {eq1} and {eq2}"
                    })

    # Identify parameter optimizations from comparative analysis
    if "comparative" in results and "comparison" in results["comparative"]:
        comparison = results["comparative"]["comparison"]

        # Find metrics where different configurations perform best
        for key in comparison:
            if key.startswith("best_") and key != "best_overall":
                metric = key.replace("best_", "")
                best_config = comparison[key].get("configuration", "")

                if "best_overall" in comparison and best_config != comparison["best_overall"].get("configuration", ""):
                    # This metric performs best in a different configuration than the overall best
                    opportunities["parameter_optimizations"].append({
                        "parameter": metric,
                        "best_config": best_config,
                        "priority": "Medium",
                        "recommendation": f"Study {best_config} configuration to optimize {metric}"
                    })

    # Suggest integration opportunities
    if "comparative" in results and "comparison" in results["comparative"]:
        comparison = results["comparative"]["comparison"]

        if "best_overall" in comparison:
            best_overall = comparison["best_overall"].get("configuration", "")

            opportunities["integration_opportunities"].append({
                "best_config": best_overall,
                "priority": "High",
                "recommendation": f"Integrate successful patterns from {best_overall} configuration into other configurations"
            })

    return opportunities


def generate_optimization_plan(results, output_path):
    """
    Generate a concrete plan for optimization based on identified opportunities.

    Args:
        results: Dictionary containing validation results
        output_path: Path to save the optimization plan

    Returns:
        String containing the optimization plan
    """
    # Get optimization opportunities
    opportunities = results.get("opportunities", {})

    # Create the plan
    plan = "# Equation Optimization Plan\n\n"
    plan += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Add high priority items first
    high_priority = []
    for category, items in opportunities.items():
        for item in items:
            if item.get("priority") == "High":
                high_priority.append({
                    "category": category,
                    "item": item
                })

    if high_priority:
        plan += "## High Priority Optimizations\n\n"
        for i, opportunity in enumerate(high_priority, 1):
            category = opportunity["category"].replace("_", " ").title()
            item = opportunity["item"]
            plan += f"### {i}. {category}: {item.get('description', item.get('best_config', 'Optimization'))}\n\n"
            plan += f"**Recommendation:** {item.get('recommendation', 'No specific recommendation')}\n\n"

    # Add categorized items
    for category, items in opportunities.items():
        if items:
            category_title = category.replace("_", " ").title()
            plan += f"## {category_title}\n\n"

            for i, item in enumerate(items, 1):
                if item.get("priority") == "High":
                    continue  # Skip high priority items, already included above

                plan += f"### {i}. {item.get('description', item.get('best_config', 'Optimization'))}\n\n"
                plan += f"**Priority:** {item.get('priority', 'Medium')}\n\n"
                plan += f"**Recommendation:** {item.get('recommendation', 'No specific recommendation')}\n\n"

    # Add implementation steps
    plan += "## Implementation Steps\n\n"
    plan += "1. **Address High Priority Items First**\n"
    plan += "   - Focus on stability enhancements before adding new features\n"
    plan += "   - Improve cross-scale transitions with poor quality scores\n\n"

    plan += "2. **Refine Existing Equations**\n"
    plan += "   - Optimize sensitive parameters identified in the analysis\n"
    plan += "   - Enhance stability of equations with numerical issues\n\n"

    plan += "3. **Fill Identified Gaps**\n"
    plan += "   - Develop new equations for missing physical analogies\n"
    plan += "   - Ensure dimensional consistency in new equations\n\n"

    plan += "4. **Improve Integration**\n"
    plan += "   - Strengthen connections between quantum and astrophysical scales\n"
    plan += "   - Ensure consistent parameter usage across all scales\n\n"

    plan += "5. **Validate and Test**\n"
    plan += "   - Re-run validation suite after each major change\n"
    plan += "   - Ensure no regressions in existing functionality\n\n"

    # Save the plan to a file
    plan_path = output_path
    os.makedirs(os.path.dirname(plan_path), exist_ok=True)

    with open(plan_path, "w") as f:
        f.write(plan)

    return plan


def main():
    """Main function to run the unified validation process."""
    # Parse command line arguments
    args = parse_arguments()

    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Run the validation process if not just generating plan
    if not args.plan_only:
        results = run_validation(args)
    else:
        # Load previous results if available
        results_path = f"{args.output_dir}/validation_results.json"
        if os.path.exists(results_path):
            import json
            with open(results_path, "r") as f:
                results = json.load(f)
        else:
            logger.error("Cannot generate plan only without previous results")
            sys.exit(1)

    logger.info("Unified validation process completed successfully")


if __name__ == "__main__":
    main()