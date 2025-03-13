"""
Comparative Simulation Analyzer
This tool runs and compares simulations with different equation configurations
to identify optimization opportunities and equation interactions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import time
import logging
from pathlib import Path
import sys

# Add project root to path to ensure imports work
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ComparativeAnalyzer")


class ComparativeSimulationAnalyzer:
    """
    Analyzes and compares simulations with different equation configurations.
    """

    def __init__(self, output_dir="validation/reports/unified"):
        """Initialize the comparative simulation analyzer."""
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Default configurations
        self.default_configurations = [
            {"name": "base", "description": "Core equations only"},
            {"name": "quantum", "description": "With quantum effects"},
            {"name": "astrophysics", "description": "With astrophysics analogies"},
            {"name": "multi_civ", "description": "With multi-civilization dynamics"},
            {"name": "integrated", "description": "All extensions integrated"}
        ]

        # Default metrics to track
        self.default_metrics = [
            "final_knowledge",
            "knowledge_growth_rate",
            "final_suppression",
            "suppression_decay_rate",
            "convergence_time",
            "stability_issues",
            "event_count"
        ]

        # Import necessary simulation modules
        self._import_simulation_modules()

    def _import_simulation_modules(self):
        """Import necessary simulation modules."""
        try:
            # Import core equations
            import config.equations
            self.equations = config.equations

            # Try to import extensions
            try:
                import config.quantum_em_extensions
                self.quantum_extensions = config.quantum_em_extensions
            except ImportError:
                logger.warning("Could not import quantum_em_extensions")
                self.quantum_extensions = None

            try:
                import config.astrophysics_extensions
                self.astrophysics_extensions = config.astrophysics_extensions
            except ImportError:
                logger.warning("Could not import astrophysics_extensions")
                self.astrophysics_extensions = None

            try:
                import config.multi_civilization_extensions
                self.multi_civ_extensions = config.multi_civilization_extensions
            except ImportError:
                logger.warning("Could not import multi_civilization_extensions")
                self.multi_civ_extensions = None

            # Import parameter module
            try:
                import config.parameters
                self.parameters = config.parameters
            except ImportError:
                logger.warning("Could not import parameters, using defaults")
                self.parameters = None

            # Import simulation modules
            try:
                import simulations.comprehensive_simulation
                self.comprehensive_sim = simulations.comprehensive_simulation
            except ImportError:
                logger.warning("Could not import comprehensive_simulation")
                self.comprehensive_sim = None

            try:
                import simulations.quantum_em_simulation
                self.quantum_sim = simulations.quantum_em_simulation
            except ImportError:
                logger.warning("Could not import quantum_em_simulation")
                self.quantum_sim = None

            try:
                import simulations.astrophysics_simulation
                self.astrophysics_sim = simulations.astrophysics_simulation
            except ImportError:
                logger.warning("Could not import astrophysics_simulation")
                self.astrophysics_sim = None

            try:
                import simulations.multi_civilization_simulation
                self.multi_civ_sim = simulations.multi_civilization_simulation
            except ImportError:
                logger.warning("Could not import multi_civilization_simulation")
                self.multi_civ_sim = None

            logger.info("Successfully imported simulation modules")

        except ImportError as e:
            logger.error(f"Error importing required modules: {str(e)}")

    def run_comparative_analysis(self, configurations=None, metrics=None, timesteps=300):
        """
        Run and compare simulations with different equation configurations.

        Args:
            configurations: List of configuration dictionaries
            metrics: List of metrics to track and compare
            timesteps: Number of timesteps for simulations

        Returns:
            Dictionary containing comparison results
        """
        if configurations is None:
            configurations = self.default_configurations

        if metrics is None:
            metrics = self.default_metrics

        logger.info(f"Running comparative analysis with {len(configurations)} configurations")

        results = {}

        # Run simulations for each configuration
        for config in configurations:
            logger.info(f"Running {config['name']} simulation")
            start_time = time.time()

            # Run simulation
            try:
                sim_results = self._run_simulation_with_config(config, timesteps)
                sim_metrics = self._extract_metrics(sim_results, metrics)

                # Store results
                results[config["name"]] = {
                    "description": config.get("description", ""),
                    "results": sim_results,
                    "metrics": sim_metrics,
                    "runtime": time.time() - start_time,
                    "success": True
                }

                logger.info(
                    f"Successfully ran {config['name']} simulation in {results[config['name']]['runtime']:.2f} seconds")

            except Exception as e:
                logger.error(f"Error running {config['name']} simulation: {str(e)}")
                results[config["name"]] = {
                    "description": config.get("description", ""),
                    "error": str(e),
                    "success": False
                }

        # Compare results
        comparison = self._compare_results(results, metrics)

        # Store comparison results
        results["comparison"] = comparison

        # Save results to CSV
        self._save_comparison_to_csv(results, metrics)

        # Visualize comparison
        self._visualize_comparison(results, metrics)

        return results

    def _run_simulation_with_config(self, config, timesteps):
        """
        Run a simulation with the specified configuration.

        Args:
            config: Configuration dictionary
            timesteps: Number of timesteps

        Returns:
            Dictionary of simulation results
        """
        config_name = config["name"]

        # Determine which simulation to run based on configuration
        if config_name == "base" and self.comprehensive_sim:
            # Run core simulation with base equations only
            return self.comprehensive_sim.run_simulation(timesteps=timesteps, use_extensions=False)

        elif config_name == "quantum" and self.quantum_sim:
            # Run quantum simulation
            return self.quantum_sim.run_simulation(timesteps=timesteps)

        elif config_name == "astrophysics" and self.astrophysics_sim:
            # Run astrophysics simulation
            return self.astrophysics_sim.run_simulation(timesteps=timesteps)

        elif config_name == "multi_civ" and self.multi_civ_sim:
            # Run multi-civilization simulation
            return self.multi_civ_sim.run_simulation(timesteps=timesteps)

        elif config_name == "integrated":
            # Run integrated simulation
            return self._run_integrated_simulation(timesteps)

        elif "custom" in config_name and "run_func" in config:
            # Run custom simulation function
            return config["run_func"](timesteps=timesteps)

        else:
            # Fallback to comprehensive simulation with all extensions
            if self.comprehensive_sim:
                return self.comprehensive_sim.run_simulation(timesteps=timesteps, use_extensions=True)
            else:
                raise ValueError(f"No simulation module available for configuration: {config_name}")

    def _run_integrated_simulation(self, timesteps):
        """
        Run an integrated simulation combining all extension types.

        Args:
            timesteps: Number of timesteps

        Returns:
            Dictionary of simulation results
        """
        logger.info(f"Running integrated simulation with {timesteps} timesteps")

        # This is an example of how the integrated simulation might be implemented
        # In a real implementation, you would create a custom simulation that properly
        # integrates all extension types

        # Initialize results structure
        results = {
            "time": np.arange(timesteps),
            "knowledge": np.zeros(timesteps),
            "suppression": np.zeros(timesteps),
            "intelligence": np.zeros(timesteps),
            "truth": np.zeros(timesteps),
            "events": []
        }

        # Define initial conditions
        K_0 = 1.0  # Initial knowledge
        S_0 = 10.0  # Initial suppression
        I_0 = 5.0  # Initial intelligence
        T_0 = 1.0  # Initial truth

        # Store initial values
        results["knowledge"][0] = K_0
        results["suppression"][0] = S_0
        results["intelligence"][0] = I_0
        results["truth"][0] = T_0

        try:
            # Initialize multi-civilization layer if available
            if self.multi_civ_extensions:
                num_civilizations = 3
                civ_data = self.multi_civ_extensions.initialize_civilizations(num_civilizations)
                knowledge_array = np.ones(num_civilizations) * K_0
                suppression_array = np.ones(num_civilizations) * S_0
                influence_array = np.random.uniform(1.0, 8.0, num_civilizations)
                resources_array = np.random.uniform(5.0, 20.0, num_civilizations)

                # Add multi-civilization metrics to results
                results["num_civilizations"] = np.zeros(timesteps, dtype=int)
                results["knowledge_array"] = [knowledge_array.copy()]
                results["suppression_array"] = [suppression_array.copy()]
                results["influence_array"] = [influence_array.copy()]
                results["resources_array"] = [resources_array.copy()]
                results["num_civilizations"][0] = num_civilizations

            # Run simulation
            for t in range(1, timesteps):
                # 1. Apply core equations
                W = self.equations.wisdom_field(1.0, 0.1, results["suppression"][t - 1], 2.0,
                                                results["knowledge"][t - 1])

                truth_change = self.equations.truth_adoption(results["truth"][t - 1], 0.5, 40.0)
                results["truth"][t] = max(0, results["truth"][t - 1] + truth_change)

                intelligence_change = self.equations.intelligence_growth(
                    results["knowledge"][t - 1], W, 2.0, results["suppression"][t - 1], 1.5)
                results["intelligence"][t] = max(0, results["intelligence"][t - 1] + intelligence_change)

                # 2. Apply quantum extensions if available
                if self.quantum_extensions:
                    # Quantum tunneling probability
                    tunneling_prob = self.quantum_extensions.quantum_tunneling_probability(
                        results["suppression"][t - 1], 1.0, results["knowledge"][t - 1])

                    # Apply tunneling effect
                    if np.random.random() < tunneling_prob:
                        # Knowledge breakthroughs due to tunneling
                        results["knowledge"][t] = results["knowledge"][t - 1] * 1.2
                        results["events"].append({
                            "time": t,
                            "type": "tunneling",
                            "probability": tunneling_prob
                        })
                    else:
                        # Normal knowledge growth
                        results["knowledge"][t] = results["knowledge"][t - 1] * 1.05
                else:
                    # Default knowledge growth
                    results["knowledge"][t] = results["knowledge"][t - 1] * 1.05

                # 3. Apply astrophysics extensions if available
                if self.astrophysics_extensions:
                    # Check for event horizon
                    horizon_radius, beyond_horizon = self.astrophysics_extensions.suppression_event_horizon(
                        results["suppression"][t - 1], results["knowledge"][t - 1])

                    if beyond_horizon:
                        # Knowledge is trapped
                        results["knowledge"][t] *= 0.95
                        results["events"].append({
                            "time": t,
                            "type": "event_horizon",
                            "radius": horizon_radius
                        })

                    # Check for inflation
                    multiplier, is_inflating = self.astrophysics_extensions.knowledge_inflation(
                        results["knowledge"][t - 1], results["truth"][t - 1], 10.0, duration=t - 100 if t > 100 else 0)

                    if is_inflating:
                        results["knowledge"][t] *= multiplier
                        results["events"].append({
                            "time": t,
                            "type": "inflation",
                            "multiplier": multiplier
                        })

                # 4. Suppression dynamics
                suppress_change = self.equations.suppression_feedback(
                    0.1, results["suppression"][t - 1], 0.05, results["knowledge"][t - 1])
                results["suppression"][t] = max(0, results["suppression"][t - 1] + suppress_change)

                # 5. Multi-civilization dynamics if available
                if self.multi_civ_extensions and t % 10 == 0:  # Only update every 10 steps for efficiency
                    try:
                        # Process civilization interactions
                        civ_data, knowledge_array, suppression_array, influence_array, resources_array, events = \
                            self.multi_civ_extensions.process_all_civilization_interactions(
                                civ_data, knowledge_array, suppression_array, influence_array, resources_array)

                        # Record events
                        for event in events:
                            event["time"] = t
                            results["events"].append(event)

                        # Update aggregate values
                        results["knowledge"][t] = np.mean(knowledge_array)
                        results["suppression"][t] = np.mean(suppression_array)

                        # Store multi-civilization data
                        results["knowledge_array"].append(knowledge_array.copy())
                        results["suppression_array"].append(suppression_array.copy())
                        results["influence_array"].append(influence_array.copy())
                        results["resources_array"].append(resources_array.copy())
                        results["num_civilizations"][t] = len(knowledge_array)
                    except Exception as e:
                        logger.error(f"Error in multi-civilization processing: {str(e)}")
                        # Ensure we have placeholder data
                        if hasattr(results, "knowledge_array"):
                            results["knowledge_array"].append(results["knowledge_array"][-1])
                            results["suppression_array"].append(results["suppression_array"][-1])
                            results["influence_array"].append(results["influence_array"][-1])
                            results["resources_array"].append(results["resources_array"][-1])

            # If we have multi-civilization data, ensure arrays are properly filled
            if "knowledge_array" in results and len(results["knowledge_array"]) < timesteps:
                # Fill in gaps with repeated data
                last_idx = len(results["knowledge_array"]) - 1
                for t in range(last_idx + 1, timesteps):
                    results["knowledge_array"].append(results["knowledge_array"][last_idx])
                    results["suppression_array"].append(results["suppression_array"][last_idx])
                    results["influence_array"].append(results["influence_array"][last_idx])
                    results["resources_array"].append(results["resources_array"][last_idx])

            # Add system stability metric
            stability_issues = np.count_nonzero(np.isnan(results["knowledge"]))
            stability_issues += np.count_nonzero(np.isnan(results["suppression"]))
            results["stability_issues"] = stability_issues

            logger.info(f"Completed integrated simulation with {len(results['events'])} events")

        except Exception as e:
            logger.error(f"Error in integrated simulation: {str(e)}")
            # Ensure we return what we have
            results["error"] = str(e)

        return results

    def _extract_metrics(self, sim_results, metrics):
        """
        Extract metrics from simulation results.

        Args:
            sim_results: Dictionary of simulation results
            metrics: List of metrics to extract

        Returns:
            Dictionary of extracted metrics
        """
        extracted = {}

        # Handle case where sim_results is not a dictionary
        if not isinstance(sim_results, dict):
            logger.warning("Simulation results is not a dictionary")
            return {}

        for metric in metrics:
            try:
                if metric == "final_knowledge":
                    if "knowledge" in sim_results and len(sim_results["knowledge"]) > 0:
                        extracted[metric] = float(sim_results["knowledge"][-1])

                elif metric == "knowledge_growth_rate":
                    if "knowledge" in sim_results and len(sim_results["knowledge"]) > 1:
                        k_start = sim_results["knowledge"][0]
                        k_end = sim_results["knowledge"][-1]
                        timesteps = len(sim_results["knowledge"])
                        if k_start > 0:
                            # Use compound annual growth rate formula
                            extracted[metric] = (k_end / k_start) ** (1 / timesteps) - 1
                        else:
                            # Fallback to simple average growth
                            extracted[metric] = (k_end - k_start) / timesteps

                elif metric == "final_suppression":
                    if "suppression" in sim_results and len(sim_results["suppression"]) > 0:
                        extracted[metric] = float(sim_results["suppression"][-1])

                elif metric == "suppression_decay_rate":
                    if "suppression" in sim_results and len(sim_results["suppression"]) > 1:
                        s_start = sim_results["suppression"][0]
                        s_end = sim_results["suppression"][-1]
                        timesteps = len(sim_results["suppression"])
                        if s_start > 0:
                            # Use decay rate formula
                            extracted[metric] = 1 - (s_end / s_start) ** (1 / timesteps)
                        else:
                            # Fallback to simple average decay
                            extracted[metric] = (s_start - s_end) / timesteps

                elif metric == "convergence_time":
                    # Define convergence as knowledge reaching 90% of its final value
                    if "knowledge" in sim_results and len(sim_results["knowledge"]) > 0:
                        k_final = sim_results["knowledge"][-1]
                        threshold = 0.9 * k_final
                        for t, k in enumerate(sim_results["knowledge"]):
                            if k >= threshold:
                                extracted[metric] = t
                                break
                        else:
                            # No convergence found
                            extracted[metric] = len(sim_results["knowledge"])

                elif metric == "stability_issues":
                    if "stability_issues" in sim_results:
                        extracted[metric] = sim_results["stability_issues"]
                    else:
                        # Calculate stability issues from NaN values
                        stability_issues = 0
                        for key in ["knowledge", "suppression", "intelligence", "truth"]:
                            if key in sim_results and isinstance(sim_results[key], (list, np.ndarray)):
                                stability_issues += np.count_nonzero(np.isnan(sim_results[key]))
                        extracted[metric] = stability_issues

                elif metric == "event_count":
                    if "events" in sim_results and isinstance(sim_results["events"], list):
                        extracted[metric] = len(sim_results["events"])

                elif metric in sim_results:
                    # Direct extraction for metrics that exist in results
                    value = sim_results[metric]
                    if isinstance(value, (int, float, bool)):
                        extracted[metric] = value
                    elif isinstance(value, (list, np.ndarray)) and len(value) > 0:
                        extracted[metric] = value[-1]  # Use final value

            except Exception as e:
                logger.error(f"Error extracting metric {metric}: {str(e)}")

        return extracted

    def _compare_results(self, results, metrics):
        """
        Compare results across different configurations.

        Args:
            results: Dictionary of results by configuration
            metrics: List of metrics to compare

        Returns:
            Dictionary of comparison results
        """
        comparison = {}

        # Collect metrics by configuration
        metrics_by_config = {}
        for config_name, config_results in results.items():
            if config_results.get("success", False) and "metrics" in config_results:
                metrics_by_config[config_name] = config_results["metrics"]

        if not metrics_by_config:
            logger.warning("No successful configurations to compare")
            return {"error": "No successful configurations to compare"}

        # Find best configuration for each metric
        for metric in metrics:
            comparison[f"{metric}_by_config"] = {}
            best_value = None
            best_config = None

            for config_name, config_metrics in metrics_by_config.items():
                if metric in config_metrics:
                    value = config_metrics[metric]
                    comparison[f"{metric}_by_config"][config_name] = value

                    # Determine if this is the best value
                    # For some metrics, higher is better
                    if metric in ["final_knowledge", "knowledge_growth_rate"]:
                        if best_value is None or value > best_value:
                            best_value = value
                            best_config = config_name

                    # For some metrics, lower is better
                    elif metric in ["final_suppression", "convergence_time", "stability_issues"]:
                        if best_value is None or value < best_value:
                            best_value = value
                            best_config = config_name

                    # For event_count, more events could indicate either more interesting
                    # dynamics or more issues - so we don't assign a best value

            if best_value is not None:
                comparison[f"best_{metric}"] = {
                    "value": best_value,
                    "configuration": best_config
                }

        # Calculate overall score for each configuration
        scores = {}
        for config_name in metrics_by_config.keys():
            score = 0

            # Award points for having the best metric values
            for metric in metrics:
                best_info = comparison.get(f"best_{metric}")
                if best_info and best_info.get("configuration") == config_name:
                    score += 1

            # Award points for stability
            if config_name in comparison.get("stability_issues_by_config", {}):
                stability_issues = comparison["stability_issues_by_config"][config_name]
                if stability_issues == 0:
                    score += 1
                elif stability_issues < 10:
                    score += 0.5

            scores[config_name] = score

        comparison["scores"] = scores

        # Identify best overall configuration
        if scores:
            best_config = max(scores.items(), key=lambda x: x[1])
            comparison["best_overall"] = {
                "configuration": best_config[0],
                "score": best_config[1]
            }

        return comparison

    def _save_comparison_to_csv(self, results, metrics):
        """
        Save comparison results to CSV files.

        Args:
            results: Dictionary of results by configuration
            metrics: List of metrics to compare
        """
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Extract metrics by configuration
        data = []
        for config_name, config_results in results.items():
            if config_name == "comparison":
                continue

            row = {"configuration": config_name}

            if config_results.get("success", False) and "metrics" in config_results:
                for metric, value in config_results["metrics"].items():
                    row[metric] = value

            if "runtime" in config_results:
                row["runtime"] = config_results["runtime"]

            if "success" in config_results:
                row["success"] = config_results["success"]

            if "error" in config_results:
                row["error"] = config_results["error"]

            data.append(row)

        # Create DataFrame and save to CSV
        if data:
            df = pd.DataFrame(data)
            csv_path = Path(self.output_dir) / "simulation_comparison.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved comparison results to {csv_path}")

        # Save comparison metrics
        if "comparison" in results:
            comparison = results["comparison"]

            # Extract metric comparisons
            metric_data = []
            for metric in metrics:
                if f"{metric}_by_config" in comparison:
                    row = {"metric": metric}
                    row.update(comparison[f"{metric}_by_config"])
                    metric_data.append(row)

            if metric_data:
                df = pd.DataFrame(metric_data)
                csv_path = Path(self.output_dir) / "metric_comparison.csv"
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved metric comparison to {csv_path}")

            # Save scores
            if "scores" in comparison:
                scores_data = [{"configuration": config, "score": score}
                               for config, score in comparison["scores"].items()]
                df = pd.DataFrame(scores_data)
                csv_path = Path(self.output_dir) / "configuration_scores.csv"
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved configuration scores to {csv_path}")

    def _visualize_comparison(self, results, metrics):
        """
        Create visualizations comparing results across configurations.

        Args:
            results: Dictionary of results by configuration
            metrics: List of metrics to compare
        """
        if "comparison" not in results:
            logger.warning("No comparison results to visualize")
            return

        comparison = results["comparison"]

        # Collect configurations that succeeded
        successful_configs = [config for config, result in results.items()
                              if config != "comparison" and result.get("success", False)]

        if not successful_configs:
            logger.warning("No successful configurations to visualize")
            return

        # Create metric comparison chart
        self._create_metric_comparison_chart(comparison, metrics, successful_configs)

        # Create time series comparison for each metric
        for metric in ["knowledge", "suppression", "intelligence", "truth"]:
            self._create_time_series_comparison(results, metric, successful_configs)

        # Create configuration scores chart
        if "scores" in comparison:
            self._create_scores_chart(comparison["scores"])

    def _create_metric_comparison_chart(self, comparison, metrics, configurations):
        """
        Create a chart comparing metrics across configurations.

        Args:
            comparison: Dictionary of comparison results
            metrics: List of metrics to compare
            configurations: List of configuration names
        """
        # Filter metrics actually present in comparison
        available_metrics = [m for m in metrics
                             if f"{m}_by_config" in comparison and comparison[f"{m}_by_config"]]

        if not available_metrics:
            logger.warning("No metrics available for comparison chart")
            return

        # Create figure with subplots for each metric
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(15, 5))

        # Handle case of single metric
        if n_metrics == 1:
            axes = [axes]

        # Plot each metric
        for i, metric in enumerate(available_metrics):
            ax = axes[i]

            # Get metric values by configuration
            metric_by_config = comparison[f"{metric}_by_config"]
            configs = list(metric_by_config.keys())
            values = list(metric_by_config.values())

            # Plot bar chart
            if metric in ["stability_issues", "final_suppression", "convergence_time"]:
                # For these metrics, lower is better
                bars = ax.bar(configs, values)
                # Color the best bar green
                if values:
                    min_idx = values.index(min(values))
                    for j, bar in enumerate(bars):
                        bar.set_color('red' if j == min_idx else 'blue')
            else:
                # For other metrics, higher is generally better
                bars = ax.bar(configs, values)
                # Color the best bar green
                if values:
                    max_idx = values.index(max(values))
                    for j, bar in enumerate(bars):
                        bar.set_color('green' if j == max_idx else 'blue')

            # Set title and labels
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.tick_params(axis='x', rotation=45)

        # Adjust layout
        plt.tight_layout()

        # Save figure
        fig_path = Path(self.output_dir) / "metric_comparison.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Saved metric comparison chart to {fig_path}")

    def _create_time_series_comparison(self, results, metric, configurations):
        """
        Create a chart comparing time series data across configurations.

        Args:
            results: Dictionary of results by configuration
            metric: Metric to compare
            configurations: List of configuration names
        """
        # Check if metric exists in results
        has_data = False
        for config in configurations:
            if config in results and metric in results[config].get("results", {}):
                has_data = True
                break

        if not has_data:
            logger.warning(f"No time series data available for {metric}")
            return

        # Create figure
        plt.figure(figsize=(10, 6))

        # Plot time series for each configuration
        for config in configurations:
            if config in results and "results" in results[config]:
                sim_results = results[config]["results"]
                if metric in sim_results and "time" in sim_results:
                    time = sim_results["time"]
                    data = sim_results[metric]

                    # Ensure time and data have the same length
                    min_len = min(len(time), len(data))
                    plt.plot(time[:min_len], data[:min_len], label=config)

        # Set title and labels
        plt.title(f"{metric.title()} Over Time")
        plt.xlabel("Time")
        plt.ylabel(metric.title())
        plt.grid(True)
        plt.legend()

        # Save figure
        fig_path = Path(self.output_dir) / f"{metric}_comparison.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved {metric} time series comparison to {fig_path}")

    def _create_scores_chart(self, scores):
        """
        Create a chart showing overall scores for each configuration.

        Args:
            scores: Dictionary of scores by configuration
        """
        # Create figure
        plt.figure(figsize=(10, 6))

        # Sort configurations by score
        sorted_configs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        configs = [x[0] for x in sorted_configs]
        values = [x[1] for x in sorted_configs]

        # Plot bar chart
        bars = plt.bar(configs, values)

        # Color the best bar green
        if values:
            bars[0].set_color('green')

        # Set title and labels
        plt.title("Overall Configuration Scores")
        plt.xlabel("Configuration")
        plt.ylabel("Score")
        plt.grid(True, axis='y')
        plt.tick_params(axis='x', rotation=45)

        # Save figure
        fig_path = Path(self.output_dir) / "configuration_scores.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved configuration scores chart to {fig_path}")


# Example usage
if __name__ == "__main__":
    analyzer = ComparativeSimulationAnalyzer()

    # Define configurations
    configurations = [
        {"name": "base", "description": "Core equations only"},
        {"name": "quantum", "description": "With quantum effects"},
        {"name": "astrophysics", "description": "With astrophysics analogies"},
        {"name": "multi_civ", "description": "With multi-civilization dynamics"},
        {"name": "integrated", "description": "All extensions integrated"}
    ]

    # Run comparative analysis
    results = analyzer.run_comparative_analysis(configurations)

    # Print summary
    if "comparison" in results and "best_overall" in results["comparison"]:
        best = results["comparison"]["best_overall"]
        print(f"Best configuration: {best['configuration']} with score {best['score']}")

    print("Comparative analysis complete. See output directory for results.")