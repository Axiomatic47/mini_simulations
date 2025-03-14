"""
Unified Validation Controller for Equation Optimization
This controller coordinates all validation components to provide a comprehensive
assessment of equation performance, completeness, and optimization opportunities.
"""

import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path to ensure imports work
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import validation components
from validation.validation_suite import ValidationSuite
from validation.validation_visualizers import ValidationVisualizer
from report_generator import ReportGenerator
from utils.edge_case_checker import EdgeCaseChecker
from utils.cross_level_validator import CrossLevelValidator
from utils.dimensional_consistency import DimensionalValidator
from utils.sensitivity_analyzer import ParameterSensitivityAnalyzer

# Import simulation tools for comparative analysis
from simulations.comprehensive_simulation import run_simulation as run_comprehensive
from simulations.quantum_em_simulation import run_simulation as run_quantum
from simulations.astrophysics_simulation import run_simulation as run_astrophysics
from simulations.multi_civilization_simulation import run_simulation as run_multi_civ

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("validation/logs/unified_validation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("UnifiedValidator")



def _dummy_simulation(params):
    """Dummy simulation function for ParameterSensitivityAnalyzer initialization."""
    import numpy as np
    # Return synthetic data structure similar to simulation results
    timesteps = 100
    return {
        'time': np.arange(timesteps),
        'knowledge': np.random.rand(timesteps) * 10,
        'suppression': np.random.rand(timesteps) * 5, 
        'intelligence': np.random.rand(timesteps) * 15
    }


class UnifiedValidator:
    """
    Unified system for validating and optimizing equations across all scales.
    """

    def __init__(self, output_dir="validation/reports/unified"):
        """Initialize the unified validation system."""
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Initialize validation components
        self.validation_suite = ValidationSuite()
        self.visualizer = ValidationVisualizer()
        self.report_generator = ReportGenerator()
        self.edge_checker = EdgeCaseChecker({})
        self.cross_validator = CrossLevelValidator({}, {'Level 1': [], 'Level 2': []})
        self.dim_validator = DimensionalValidator()
        self.sensitivity_analyzer = ParameterSensitivityAnalyzer(_dummy_simulation, ['knowledge', 'suppression', 'intelligence'], {'alpha': 0.1, 'beta': 0.2})

        # Track results
        self.results = {}
        self.start_time = None

        logger.info("Unified Validator initialized")

    def run_full_validation(self, equation_modules=None, focus_areas=None, generate_report=True):
        """
        Run comprehensive validation across all components.

        Args:
            equation_modules: List of equation module names to validate
            focus_areas: List of specific areas to focus validation on
            generate_report: Whether to generate a comprehensive report

        Returns:
            Dictionary of validation results
        """
        self.start_time = time.time()
        logger.info("Starting full validation process")

        if equation_modules is None:
            equation_modules = ["equations", "astrophysics_extensions",
                                "quantum_em_extensions", "multi_civilization_extensions"]

        if focus_areas is None:
            focus_areas = ["equation_completeness", "cross_level_interaction",
                           "edge_cases", "dimensional_consistency", "parameter_sensitivity"]

        # Step 1: Run basic validation suite
        logger.info("Running basic validation suite")
        self.results["basic_validation"] = self.validation_suite.run_full_validation(
            focus_areas=focus_areas,
            generate_report=False  # We'll generate a unified report later
        )

        # Step 2: Analysis of equation coverage and gaps
        logger.info("Analyzing equation coverage and gaps")
        self.results["equation_coverage"] = self.edge_checker.analyze_equation_set(
            equation_modules=equation_modules,
            identify_gaps=True
        )

        # Step 3: Cross-scale interaction validation
        logger.info("Validating cross-scale interactions")
        self.results["cross_scale"] = self.cross_validator.evaluate_cross_scale_interactions(
            levels=["quantum", "agent", "civilization", "multi_civilization"],
            key_transitions=[
                ("quantum_tunneling", "suppression_event_horizon"),
                ("knowledge_field_influence", "civilization_lifecycle_phase"),
                ("entanglement_network", "galactic_structure_model")
            ]
        )

        # Step 4: Dimensional consistency checks
        logger.info("Checking dimensional consistency")
        self.results["dimensional"] = self.dim_validator.validate_dimensional_consistency(
            equation_modules=equation_modules
        )

        # Step 5: Parameter sensitivity analysis
        logger.info("Analyzing parameter sensitivity")
        self.results["sensitivity"] = self.sensitivity_analyzer.analyze_equation_sensitivity(
            target_metrics=["knowledge_growth", "truth_adoption", "system_stability"]
        )

        # Step 6: Comparative simulation analysis
        logger.info("Running comparative simulation analysis")
        self.results["comparative"] = self.run_comparative_simulations()

        # Generate comprehensive report if requested
        if generate_report:
            self.generate_comprehensive_report()

        elapsed_time = time.time() - self.start_time
        logger.info(f"Full validation completed in {elapsed_time:.2f} seconds")

        return self.results

    def run_comparative_simulations(self, configurations=None):
        """
        Run and compare simulations with different equation configurations.

        Args:
            configurations: List of configuration dictionaries

        Returns:
            Dictionary of comparative results
        """
        if configurations is None:
            configurations = [
                {"name": "base", "run_func": run_comprehensive, "extensions": []},
                {"name": "quantum", "run_func": run_quantum, "extensions": ["quantum_em_extensions"]},
                {"name": "astrophysics", "run_func": run_astrophysics, "extensions": ["astrophysics_extensions"]},
                {"name": "multi_civ", "run_func": run_multi_civ, "extensions": ["multi_civilization_extensions"]},
                {"name": "integrated", "run_func": self.run_integrated_simulation,
                 "extensions": ["quantum_em_extensions", "astrophysics_extensions", "multi_civilization_extensions"]}
            ]

        results = {}
        for config in configurations:
            logger.info(f"Running {config['name']} simulation")
            try:
                sim_results = config["run_func"]()
                results[config["name"]] = {
                    "success": True,
                    "results": sim_results,
                    "metrics": self.calculate_simulation_metrics(sim_results)
                }
            except Exception as e:
                logger.error(f"Error running {config['name']} simulation: {str(e)}")
                results[config["name"]] = {
                    "success": False,
                    "error": str(e)
                }

        # Compare results across configurations
        results["comparison"] = self.compare_simulation_results(results)

        return results

    def run_integrated_simulation(self, timesteps=500, agents=10, civilizations=5):
        """
        Run an integrated simulation combining all extension types.

        This simulation tests how quantum effects influence astrophysical models
        at both individual agent and multi-civilization scales.

        Args:
            timesteps: Number of simulation timesteps
            agents: Number of agents per civilization
            civilizations: Number of civilizations

        Returns:
            Dictionary of simulation results
        """
        logger.info(f"Running integrated simulation with {timesteps} steps, {agents} agents, {civilizations} civs")

        # This is a placeholder for an actual integrated simulation
        # In a real implementation, you would create a new simulation that combines
        # quantum, astrophysics, and multi-civilization features

        # For now, let's call the appropriate modules and combine results
        import numpy as np
        from config import equations, quantum_em_extensions, astrophysics_extensions, multi_civilization_extensions

        # Initialize structures for agents and civilizations
        civ_data = multi_civilization_extensions.initialize_civilizations(civilizations)
        knowledge_array = np.random.uniform(1.0, 10.0, civilizations)
        suppression_array = np.random.uniform(0.5, 5.0, civilizations)
        influence_array = np.random.uniform(1.0, 8.0, civilizations)
        resources_array = np.random.uniform(5.0, 20.0, civilizations)

        # Track results
        results = {
            "time": np.arange(timesteps),
            "knowledge": np.zeros((timesteps, civilizations)),
            "suppression": np.zeros((timesteps, civilizations)),
            "influence": np.zeros((timesteps, civilizations)),
            "resources": np.zeros((timesteps, civilizations)),
            "events": []
        }

        # Store initial values
        results["knowledge"][0] = knowledge_array
        results["suppression"][0] = suppression_array
        results["influence"][0] = influence_array
        results["resources"][0] = resources_array

        # Run simulation
        for t in range(1, timesteps):
            # 1. Process civilization lifecycles with astrophysics extensions
            for i in range(civilizations):
                # Apply lifecycle phase model
                phase_intensity, phase = astrophysics_extensions.civilization_lifecycle_phase(
                    civ_data["ages"][i], knowledge_array[i],
                    np.array([10, 50, 100, 150, 200]),
                    np.array([0.5, 1.0, 1.5, 1.0, 0.7, 0.3])
                )

                # Apply event horizon model
                horizon_radius, beyond_horizon = astrophysics_extensions.suppression_event_horizon(
                    suppression_array[i], knowledge_array[i]
                )

                # Apply knowledge gravitational lensing
                truth_value = 5.0 + 0.1 * knowledge_array[i]
                apparent_truth, distortion = astrophysics_extensions.knowledge_gravitational_lensing(
                    truth_value, suppression_array[i], 2.0
                )

                # Modify knowledge based on cosmic inflation if appropriate
                if apparent_truth > 10.0:
                    multiplier, is_inflating = astrophysics_extensions.knowledge_inflation(
                        knowledge_array[i], apparent_truth, 10.0, duration=t - 100 if t > 100 else 0
                    )
                    if is_inflating:
                        knowledge_array[i] *= multiplier

            # 2. Apply quantum effects across civilizations
            for i in range(civilizations):
                # Check for quantum tunneling events
                tunneling_prob = quantum_em_extensions.quantum_tunneling_probability(
                    suppression_array[i], 1.0, knowledge_array[i]
                )

                # If tunneling occurs, reduce suppression
                if np.random.random() < tunneling_prob:
                    suppression_array[i] *= 0.8
                    results["events"].append({
                        "time": t,
                        "type": "tunneling",
                        "civilization": i,
                        "probability": tunneling_prob
                    })

            # 3. Process civilization interactions
            civ_data, knowledge_array, suppression_array, influence_array, resources_array, events = \
                multi_civilization_extensions.process_all_civilization_interactions(
                    civ_data, knowledge_array, suppression_array, influence_array, resources_array
                )

            # Add events to results
            for event in events:
                event["time"] = t
                results["events"].append(event)

            # Store current values
            if t < timesteps:
                results["knowledge"][t] = knowledge_array
                results["suppression"][t] = suppression_array
                results["influence"][t] = influence_array
                results["resources"][t] = resources_array

        return results

    def calculate_simulation_metrics(self, sim_results):
        """
        Calculate comparative metrics from simulation results.

        Args:
            sim_results: Results from a simulation run

        Returns:
            Dictionary of metrics
        """
        import numpy as np

        metrics = {}

        # Not all simulations will have the same output structure,
        # so we need to handle different formats

        # Handle time series data if available
        if isinstance(sim_results, dict) and "knowledge" in sim_results and len(sim_results["knowledge"]) > 0:
            knowledge = sim_results["knowledge"]

            if isinstance(knowledge, np.ndarray) and knowledge.ndim >= 1:
                # For multi-agent or multi-civilization data
                if knowledge.ndim > 1:
                    metrics["final_knowledge_mean"] = np.mean(knowledge[-1])
                    metrics["final_knowledge_std"] = np.std(knowledge[-1])
                    metrics["knowledge_growth_rate"] = np.mean(knowledge[-1] - knowledge[0]) / len(knowledge)
                else:
                    # For single-agent data
                    metrics["final_knowledge"] = knowledge[-1]
                    metrics["knowledge_growth_rate"] = (knowledge[-1] - knowledge[0]) / len(knowledge)

        # Handle suppression if available
        if isinstance(sim_results, dict) and "suppression" in sim_results and len(sim_results["suppression"]) > 0:
            suppression = sim_results["suppression"]

            if isinstance(suppression, np.ndarray) and suppression.ndim >= 1:
                if suppression.ndim > 1:
                    metrics["final_suppression_mean"] = np.mean(suppression[-1])
                    metrics["suppression_decay_rate"] = np.mean(suppression[0] - suppression[-1]) / len(suppression)
                else:
                    metrics["final_suppression"] = suppression[-1]
                    metrics["suppression_decay_rate"] = (suppression[0] - suppression[-1]) / len(suppression)

        # Handle events if available
        if isinstance(sim_results, dict) and "events" in sim_results:
            events = sim_results["events"]
            metrics["event_count"] = len(events)

            # Count event types
            event_types = {}
            for event in events:
                if "type" in event:
                    event_type = event["type"]
                    event_types[event_type] = event_types.get(event_type, 0) + 1

            metrics["event_types"] = event_types

        # Calculate overall complexity metric
        metrics["complexity"] = len(metrics)

        return metrics

    def compare_simulation_results(self, all_results):
        """
        Compare results across different simulation configurations.

        Args:
            all_results: Dictionary of results from different configurations

        Returns:
            Dictionary of comparative analyses
        """
        comparison = {
            "best_knowledge_growth": None,
            "most_stable": None,
            "most_complex": None,
            "unique_events": {},
            "common_metrics": []
        }

        # Find common metrics across all successful simulations
        common_metrics = set()
        first = True

        for config_name, config_results in all_results.items():
            if config_name == "comparison" or not config_results.get("success", False):
                continue

            if "metrics" in config_results:
                metrics = set(config_results["metrics"].keys())
                if first:
                    common_metrics = metrics
                    first = False
                else:
                    common_metrics = common_metrics.intersection(metrics)

        comparison["common_metrics"] = list(common_metrics)

        # Compare across common metrics
        for metric in common_metrics:
            comparison[f"best_{metric}"] = {
                "value": None,
                "configuration": None
            }

            comparison[f"{metric}_by_config"] = {}

            for config_name, config_results in all_results.items():
                if config_name == "comparison" or not config_results.get("success", False):
                    continue

                if "metrics" in config_results and metric in config_results["metrics"]:
                    value = config_results["metrics"][metric]
                    comparison[f"{metric}_by_config"][config_name] = value

                    # For some metrics, higher is better
                    if metric in ["knowledge_growth_rate", "final_knowledge", "final_knowledge_mean", "complexity"]:
                        if (comparison[f"best_{metric}"]["value"] is None or
                                value > comparison[f"best_{metric}"]["value"]):
                            comparison[f"best_{metric}"]["value"] = value
                            comparison[f"best_{metric}"]["configuration"] = config_name

                    # For some metrics, lower is better
                    elif metric in ["final_suppression", "final_suppression_mean"]:
                        if (comparison[f"best_{metric}"]["value"] is None or
                                value < comparison[f"best_{metric}"]["value"]):
                            comparison[f"best_{metric}"]["value"] = value
                            comparison[f"best_{metric}"]["configuration"] = config_name

        # Identify the overall best configuration based on multiple metrics
        scores = {}
        for config_name in all_results:
            if config_name == "comparison" or not all_results[config_name].get("success", False):
                continue

            scores[config_name] = 0

            # Award points for being the best in different metrics
            for metric in common_metrics:
                best_config = comparison.get(f"best_{metric}", {}).get("configuration")
                if best_config == config_name:
                    scores[config_name] += 1

        comparison["overall_scores"] = scores
        comparison["best_overall"] = max(scores.items(), key=lambda x: x[1])[0] if scores else None

        return comparison

    def generate_comprehensive_report(self):
        """
        Generate a comprehensive report of all validation results.
        """
        logger.info("Generating comprehensive validation report")

        # Create report directory
        report_dir = Path(self.output_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

        # Generate main HTML report
        report_path = report_dir / "unified_validation_report.html"

        # Use the report generator to create the report
        self.report_generator.generate_unified_report(
            self.results,
            report_path,
            include_visualizations=True
        )

        # Generate supplementary reports and visualizations
        self._generate_supplementary_reports(report_dir)

        logger.info(f"Comprehensive report generated at {report_path}")
        return str(report_path)

    def _generate_supplementary_reports(self, report_dir):
        """
        Generate supplementary reports and visualizations.

        Args:
            report_dir: Directory to save reports
        """
        # Equation coverage visualization
        if "equation_coverage" in self.results:
            self.visualizer.plot_equation_coverage(
                self.results["equation_coverage"],
                str(report_dir / "equation_coverage.png")
            )

        # Cross-scale interaction visualization
        if "cross_scale" in self.results:
            self.visualizer.plot_cross_scale_interactions(
                self.results["cross_scale"],
                str(report_dir / "cross_scale_interactions.png")
            )

        # Parameter sensitivity visualization
        if "sensitivity" in self.results:
            self.visualizer.plot_sensitivity_heatmap(
                self.results["sensitivity"],
                str(report_dir / "sensitivity_heatmap.png")
            )

        # Comparative simulation results
        if "comparative" in self.results and "comparison" in self.results["comparative"]:
            self.visualizer.plot_simulation_comparison(
                self.results["comparative"]["comparison"],
                str(report_dir / "simulation_comparison.png")
            )

    def identify_optimization_opportunities(self):
        """
        Identify opportunities for equation optimization based on validation results.

        Returns:
            Dictionary of optimization opportunities
        """
        if not self.results:
            logger.warning("No validation results available for identifying optimization opportunities")
            return {}

        opportunities = {
            "equation_gaps": [],
            "cross_scale_improvements": [],
            "stability_enhancements": [],
            "parameter_optimizations": [],
            "integration_opportunities": []
        }

        # Identify equation gaps
        if "equation_coverage" in self.results:
            coverage = self.results["equation_coverage"]
            for gap in coverage.get("gaps", []):
                opportunities["equation_gaps"].append({
                    "description": gap["description"],
                    "affected_modules": gap["modules"],
                    "priority": gap["severity"],
                    "recommendation": gap.get("recommendation", "Fill this gap with an appropriate equation")
                })

        # Identify cross-scale improvements
        if "cross_scale" in self.results:
            cross_scale = self.results["cross_scale"]
            for transition, quality in cross_scale.get("transition_quality", {}).items():
                if quality < 0.7:  # Arbitrary threshold
                    opportunities["cross_scale_improvements"].append({
                        "transition": transition,
                        "quality": quality,
                        "priority": "High" if quality < 0.5 else "Medium",
                        "recommendation": f"Improve integration between {transition[0]} and {transition[1]}"
                    })

        # Identify stability enhancements
        if "basic_validation" in self.results:
            stability = self.results["basic_validation"].get("stability", {})
            for equation, issues in stability.get("issues_by_equation", {}).items():
                if issues:
                    opportunities["stability_enhancements"].append({
                        "equation": equation,
                        "issues": issues,
                        "priority": "High",
                        "recommendation": f"Enhance stability of {equation} by addressing {len(issues)} issues"
                    })

        # Identify parameter optimizations
        if "sensitivity" in self.results:
            sensitivity = self.results["sensitivity"]
            for param, importance in sensitivity.get("parameter_importance", {}).items():
                if importance > 0.1:  # Arbitrary threshold for significant parameters
                    opportunities["parameter_optimizations"].append({
                        "parameter": param,
                        "importance": importance,
                        "priority": "High" if importance > 0.2 else "Medium",
                        "recommendation": f"Optimize {param} for better performance"
                    })

        # Identify integration opportunities
        if "comparative" in self.results and "comparison" in self.results["comparative"]:
            comparison = self.results["comparative"]["comparison"]
            best_config = comparison.get("best_overall")

            if best_config == "integrated":
                # If integrated configuration is best, recommend enhancing specific interactions
                for metric, metric_by_config in comparison.items():
                    if metric.endswith("_by_config") and "integrated" in metric_by_config:
                        base_metric = metric.replace("_by_config", "")
                        if best_config != comparison.get(f"best_{base_metric}", {}).get("configuration"):
                            better_config = comparison.get(f"best_{base_metric}", {}).get("configuration")
                            if better_config:
                                opportunities["integration_opportunities"].append({
                                    "metric": base_metric,
                                    "better_config": better_config,
                                    "priority": "Medium",
                                    "recommendation": f"Enhance {base_metric} in integrated model by learning from {better_config} implementation"
                                })
            else:
                # Recommend integrating the best configuration into others
                opportunities["integration_opportunities"].append({
                    "best_config": best_config,
                    "priority": "High",
                    "recommendation": f"Integrate successful patterns from {best_config} into other configurations"
                })

        return opportunities

    def generate_optimization_plan(self, opportunities=None):
        """
        Generate a concrete plan for optimization based on identified opportunities.

        Args:
            opportunities: Dictionary of optimization opportunities (if None, will be identified)

        Returns:
            String containing the optimization plan
        """
        if opportunities is None:
            opportunities = self.identify_optimization_opportunities()

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
                plan += f"### {i}. {category}: {item.get('description', item.get('equation', item.get('parameter', item.get('transition', item.get('best_config', 'Optimization')))))}\n\n"
                plan += f"**Recommendation:** {item.get('recommendation', 'No specific recommendation')}\n\n"
                if "affected_modules" in item:
                    plan += f"**Affected Modules:** {', '.join(item['affected_modules'])}\n\n"

        # Add categorized items
        for category, items in opportunities.items():
            if items:
                category_title = category.replace("_", " ").title()
                plan += f"## {category_title}\n\n"

                for i, item in enumerate(items, 1):
                    if item.get("priority") == "High":
                        continue  # Skip high priority items, already included above

                    plan += f"### {i}. {item.get('description', item.get('equation', item.get('parameter', item.get('transition', item.get('best_config', 'Optimization')))))}\n\n"
                    plan += f"**Priority:** {item.get('priority', 'Medium')}\n\n"
                    plan += f"**Recommendation:** {item.get('recommendation', 'No specific recommendation')}\n\n"

                    # Add any additional details
                    for key, value in item.items():
                        if key not in ["description", "priority", "recommendation", "affected_modules", "equation",
                                       "parameter", "transition", "best_config"]:
                            plan += f"**{key.replace('_', ' ').title()}:** {value}\n\n"

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
        plan_path = Path(self.output_dir) / "optimization_plan.md"
        with open(plan_path, "w") as f:
            f.write(plan)

        logger.info(f"Optimization plan generated and saved to {plan_path}")

        return plan

    def generate_comprehensive_report(self):
        """
        Generate a comprehensive report of all validation results.
        """
        logger.info("Generating comprehensive validation report")

        # Create report directory
        report_dir = Path(self.output_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

        # Generate main HTML report
        report_path = report_dir / "unified_validation_report.html"

        # Use the report generator to create the report
        self.report_generator.generate_unified_report(
            self.results,
            report_path,
            include_visualizations=True
        )

        # Generate supplementary reports and visualizations
        self._generate_supplementary_reports(report_dir)

        logger.info(f"Comprehensive report generated at {report_path}")

        # Automatically open the report
        self.open_report(report_path)

        return str(report_path)


# Example usage
if __name__ == "__main__":
    validator = UnifiedValidator()

    # Run full validation
    results = validator.run_full_validation()

    # Identify optimization opportunities
    opportunities = validator.identify_optimization_opportunities()

    # Generate optimization plan
    plan = validator.generate_optimization_plan(opportunities)

    print(f"Validation complete. Check reports in {validator.output_dir}")
    print(f"Optimization plan: {validator.output_dir}/optimization_plan.md")