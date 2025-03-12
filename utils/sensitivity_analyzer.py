"""
Parameter sensitivity analysis utilities for the Axiomatic Intelligence Growth Simulation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import itertools
import os


class ParameterSensitivityAnalyzer:
    """
    Tool for analyzing parameter sensitivity in simulations.

    This class provides methods for:
    1. One-at-a-time sensitivity analysis
    2. Global sensitivity analysis
    3. Parameter correlation analysis
    4. Visualization of sensitivity results
    """

    def __init__(self, simulation_function, metrics, base_parameters, random_seed=42):
        """
        Initialize sensitivity analyzer.

        Args:
            simulation_function: Function that takes parameters and returns metrics dictionary
            metrics: List of metric names to analyze
            base_parameters: Dictionary of baseline parameter values
            random_seed: Random seed for reproducibility
        """
        self.simulation_function = simulation_function
        self.metrics = metrics
        self.base_parameters = base_parameters
        self.random_seed = random_seed

        # Initialize state
        self.parameter_ranges = {}
        self.results = None
        self.global_results = None
        self.correlations = {}
        self.interaction_effects = {}
        self.sobol_results = None

        # Set random seed
        np.random.seed(random_seed)

    def define_parameter_ranges(self, parameter_ranges):
        """
        Define parameter ranges for sensitivity analysis.

        Args:
            parameter_ranges: Dictionary mapping parameter names to (min, max, num_points) tuples
        """
        self.parameter_ranges = parameter_ranges

    def run_one_at_a_time_sensitivity(self, parallel=True):
        """
        Run one-at-a-time sensitivity analysis.

        Args:
            parallel: Whether to run simulations in parallel

        Returns:
            DataFrame: Results of sensitivity analysis
        """
        # Ensure parameter ranges are defined
        if not self.parameter_ranges:
            raise ValueError("Parameter ranges must be defined before running sensitivity analysis")

        # Generate parameter combinations
        parameter_values = {}
        for param, (min_val, max_val, num_points) in self.parameter_ranges.items():
            parameter_values[param] = np.linspace(min_val, max_val, num_points)

        # Generate parameter sets
        parameter_sets = []
        for param, values in parameter_values.items():
            for value in values:
                # Create parameter set with base values and one changed parameter
                params = self.base_parameters.copy()
                params[param] = value
                params['changed_parameter'] = param
                parameter_sets.append(params)

        # Run simulations
        if parallel and len(parameter_sets) > 1:
            results = self._run_parallel_simulations(parameter_sets)
        else:
            results = self._run_sequential_simulations(parameter_sets)

        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        self.results = results_df

        # Calculate correlations
        self._calculate_correlations()

        return results_df

    def run_global_sensitivity_analysis(self, samples=100, method='saltelli'):
        """
        Run global sensitivity analysis using Sobol method.

        Args:
            samples: Number of samples per parameter
            method: Sampling method ('saltelli', 'latin', 'random')

        Returns:
            dict: Sobol indices for each metric
        """
        try:
            from SALib.sample import saltelli, latin
            from SALib.analyze import sobol
        except ImportError:
            print("Warning: SALib not installed. Using simpler sampling method.")
            method = 'random'

        # Ensure parameter ranges are defined
        if not self.parameter_ranges:
            raise ValueError("Parameter ranges must be defined before running sensitivity analysis")

        # Define problem for SALib
        problem = {
            'num_vars': len(self.parameter_ranges),
            'names': list(self.parameter_ranges.keys()),
            'bounds': [[r[0], r[1]] for r in self.parameter_ranges.values()]
        }

        # Generate samples
        if method == 'saltelli' and 'saltelli' in locals():
            # Saltelli sampling for Sobol analysis
            param_values = saltelli.sample(problem, samples)
        elif method == 'latin' and 'latin' in locals():
            # Latin hypercube sampling
            param_values = latin.sample(problem, samples)
        else:
            # Random sampling
            param_values = self._generate_random_samples(problem, samples)

        # Convert to parameter dictionaries
        parameter_sets = []
        for values in param_values:
            params = self.base_parameters.copy()
            for i, param in enumerate(problem['names']):
                params[param] = values[i]
            parameter_sets.append(params)

        # Run simulations
        if len(parameter_sets) > 10:
            results = self._run_parallel_simulations(parameter_sets)
        else:
            results = self._run_sequential_simulations(parameter_sets)

        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        self.global_results = results_df

        # Calculate Sobol indices if possible
        sobol_indices = {}
        if method == 'saltelli' and 'sobol' in locals():
            for metric in self.metrics:
                try:
                    Y = results_df[metric].values
                    Si = sobol.analyze(problem, Y, print_to_console=False)
                    sobol_indices[metric] = Si
                except Exception as e:
                    print(f"Error calculating Sobol indices for {metric}: {e}")

        self.sobol_results = sobol_indices

        # Calculate interaction effects
        self._calculate_interaction_effects()

        return sobol_indices

    def calculate_parameter_importance(self, method='range'):
        """
        Calculate parameter importance using different methods.

        Args:
            method (str): Method to use for importance calculation:
                'sobol' - Sobol indices (requires SALib)
                'correlation' - Correlation-based importance
                'range' - Range-based importance
                'variance' - Variance-based importance

        Returns:
            pd.Series: Parameter importance scores, sorted in descending order
        """
        # If we have no results yet, run analysis
        if self.results is None:
            self.run_one_at_a_time_sensitivity()

        if method == 'sobol' and self.sobol_results:
            # Use Sobol indices if available
            first_metric = next(iter(self.sobol_results))
            importance = pd.Series({
                param: self.sobol_results[first_metric]['S1'][i]
                for i, param in enumerate(self.parameter_ranges.keys())
            })

        elif method == 'correlation' and self.correlations:
            # Use correlation-based importance if available
            importance = pd.Series({
                param: max([abs(corr) for metric, corr in param_corrs.items()])
                for param, param_corrs in self.correlations.items()
            })

        else:
            # Fallback to range-based importance (always works)
            importance = {}

            # Get unique parameter values from the results
            for param in self.parameter_ranges.keys():
                if param not in self.results.columns:
                    # Skip parameters not in results
                    importance[param] = 0.0
                    continue

                param_values = self.results[param].unique()

                if len(param_values) <= 1:
                    # Skip parameters that don't vary
                    importance[param] = 0.0
                    continue

                # Calculate total impact across all metrics
                total_impact = 0.0
                for metric in self.metrics:
                    if metric not in self.results.columns:
                        continue

                    # Calculate max variation for each metric
                    metric_values = []
                    for param_value in param_values:
                        # Get average metric value for this parameter value
                        mask = self.results[param] == param_value
                        if mask.any():
                            avg_value = self.results.loc[mask, metric].mean()
                            metric_values.append(avg_value)

                    if metric_values:
                        # Calculate normalized range
                        metric_range = max(metric_values) - min(metric_values)
                        # Add to total impact
                        total_impact += metric_range

                importance[param] = total_impact

            # Convert to Series
            importance = pd.Series(importance)

        # Normalize to sum to 1.0 if there are any non-zero values
        if importance.sum() > 0:
            importance = importance / importance.sum()

        # Sort in descending order
        return importance.sort_values(ascending=False)

    def calculate_parameter_correlations(self):
        """
        Calculate correlations between parameters and metrics.

        Returns:
            dict: Correlations between parameters and metrics
        """
        if self.results is None:
            self.run_one_at_a_time_sensitivity()

        self._calculate_correlations()
        return self.correlations

    def identify_interaction_effects(self):
        """
        Identify interaction effects between parameters.

        Returns:
            dict: Interaction effects between parameters
        """
        if self.global_results is None:
            # We need global sensitivity analysis results
            try:
                self.run_global_sensitivity_analysis()
            except Exception as e:
                raise ValueError(f"Must run global sensitivity analysis before identifying interactions: {e}")

        self._calculate_interaction_effects()
        return self.interaction_effects

    def generate_tornado_plot(self, metric=None, figsize=(10, 8), top_n=None):
        """
        Generate a tornado plot showing parameter sensitivity.

        Args:
            metric (str): Metric to plot, or None to use first metric
            figsize (tuple): Figure size
            top_n (int): Number of top parameters to show

        Returns:
            matplotlib.figure.Figure: Tornado plot
        """
        if self.results is None:
            self.run_one_at_a_time_sensitivity()

        # Choose metric
        if metric is None:
            metric = self.metrics[0]
        elif metric not in self.metrics:
            raise ValueError(f"Metric {metric} not in available metrics: {self.metrics}")

        # Calculate baseline value
        baseline_mask = True
        for param in self.parameter_ranges:
            baseline_mask &= (self.results[param] == self.base_parameters[param])
        baseline_value = self.results.loc[baseline_mask, metric].values[0] if baseline_mask.any() else 0

        # Calculate parameter impacts
        impacts = {}
        for param, (min_val, max_val, _) in self.parameter_ranges.items():
            min_mask = (self.results[param] == min_val) & (self.results['changed_parameter'] == param)
            max_mask = (self.results[param] == max_val) & (self.results['changed_parameter'] == param)

            if min_mask.any() and max_mask.any():
                min_value = self.results.loc[min_mask, metric].values[0]
                max_value = self.results.loc[max_mask, metric].values[0]
                impacts[param] = (min_value - baseline_value, max_value - baseline_value)

        # Sort by absolute impact
        sorted_impacts = sorted(
            impacts.items(),
            key=lambda x: max(abs(x[1][0]), abs(x[1][1])),
            reverse=True
        )

        # Limit to top N if specified
        if top_n is not None:
            sorted_impacts = sorted_impacts[:top_n]

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        # Plot parameters
        param_names = [p[0] for p in sorted_impacts]
        y_pos = np.arange(len(param_names))

        # Plot bars
        for i, (param, (min_impact, max_impact)) in enumerate(sorted_impacts):
            # Plot decreasing impact in blue
            ax.barh(
                y_pos[i],
                min_impact if min_impact < 0 else 0,
                align='center',
                color='blue',
                alpha=0.6,
                height=0.5
            )

            # Plot increasing impact in red
            ax.barh(
                y_pos[i],
                max_impact if max_impact > 0 else 0,
                align='center',
                color='red',
                alpha=0.6,
                height=0.5
            )

        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(param_names)
        ax.invert_yaxis()  # Highest values at top
        ax.set_xlabel(f'Change in {metric}')
        ax.set_title(f'Tornado Plot - Impact on {metric}')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)

        return fig

    def generate_sensitivity_heatmap(self, figsize=(12, 10)):
        """
        Generate a heatmap of parameter sensitivity across all metrics.

        Args:
            figsize (tuple): Figure size

        Returns:
            matplotlib.figure.Figure: Heatmap plot
        """
        # Ensure we have correlation results
        if not self.correlations and self.results is not None:
            self._calculate_correlations()

        if not self.correlations:
            raise ValueError("No correlation data available. Run sensitivity analysis first.")

        # Create correlation matrix
        corr_matrix = np.zeros((len(self.parameter_ranges), len(self.metrics)))
        parameters = list(self.parameter_ranges.keys())

        for i, param in enumerate(parameters):
            if param in self.correlations:
                for j, metric in enumerate(self.metrics):
                    if metric in self.correlations[param]:
                        corr_matrix[i, j] = self.correlations[param][metric]

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)

        # Set ticks and labels
        ax.set_xticks(np.arange(len(self.metrics)))
        ax.set_yticks(np.arange(len(parameters)))
        ax.set_xticklabels(self.metrics)
        ax.set_yticklabels(parameters)

        # Rotate x labels for readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        for i in range(len(parameters)):
            for j in range(len(self.metrics)):
                ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                        ha="center", va="center",
                        color="black" if abs(corr_matrix[i, j]) < 0.5 else "white")

        # Add title and labels
        ax.set_title("Parameter-Metric Sensitivity Correlation")

        fig.tight_layout()
        return fig

    def generate_interaction_network(self, figsize=(12, 10), threshold=0.1):
        """
        Generate a network visualization of parameter interactions.

        Args:
            figsize (tuple): Figure size
            threshold (float): Minimum interaction strength to display

        Returns:
            matplotlib.figure.Figure: Network plot
        """
        # Ensure we have interaction data
        if not self.interaction_effects and self.global_results is not None:
            self._calculate_interaction_effects()

        if not self.interaction_effects:
            # Try to run global sensitivity first
            try:
                self.run_global_sensitivity_analysis(samples=50)
            except Exception:
                raise ValueError("No interaction data available and could not run global sensitivity analysis")

        # Import networkx for graph visualization
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx is required for interaction network visualization")

        # Create graph
        G = nx.Graph()

        # Add nodes (parameters)
        parameters = list(self.parameter_ranges.keys())
        for param in parameters:
            G.add_node(param)

        # Add edges (interactions)
        for p1, p2, strength in self.interaction_effects:
            if abs(strength) >= threshold:
                G.add_edge(p1, p2, weight=abs(strength), color='red' if strength > 0 else 'blue')

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        # Use spring layout for node positions
        pos = nx.spring_layout(G, seed=42)

        # Get edge colors and weights
        edge_colors = [G[u][v]['color'] for u, v in G.edges()]
        edge_weights = [2 * G[u][v]['weight'] for u, v in G.edges()]

        # Draw network
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=500, alpha=0.8)
        nx.draw_networkx_labels(G, pos, ax=ax)
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_weights, edge_color=edge_colors, alpha=0.7)

        # Add legend
        ax.plot([0], [0], color='red', linewidth=2, label='Positive Interaction')
        ax.plot([0], [0], color='blue', linewidth=2, label='Negative Interaction')
        ax.legend()

        # Remove axis
        ax.set_axis_off()

        # Add title
        ax.set_title("Parameter Interaction Network")

        return fig

    def generate_comprehensive_report(self, output_dir=None):
        """
        Generate a comprehensive report of sensitivity analysis results.

        Args:
            output_dir (str): Directory to save the report

        Returns:
            str: Path to the report
        """
        import os

        # Create output directory if needed
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        # Generate plots
        try:
            # Tornado plot for each metric
            for metric in self.metrics:
                fig = self.generate_tornado_plot(metric=metric)
                if output_dir is not None:
                    plt.savefig(os.path.join(output_dir, f"tornado_{metric}.png"), dpi=300)
                plt.close(fig)

            # Sensitivity heatmap
            fig = self.generate_sensitivity_heatmap()
            if output_dir is not None:
                plt.savefig(os.path.join(output_dir, "sensitivity_heatmap.png"), dpi=300)
            plt.close(fig)

            # Parameter importance
            importance = self.calculate_parameter_importance()
            fig, ax = plt.subplots(figsize=(10, 6))
            importance.plot(kind='bar', ax=ax)
            ax.set_title("Parameter Importance")
            ax.set_ylabel("Normalized Importance")
            plt.tight_layout()
            if output_dir is not None:
                plt.savefig(os.path.join(output_dir, "parameter_importance.png"), dpi=300)
            plt.close(fig)

            # Try to generate interaction network if we have global results
            try:
                if self.global_results is not None or self.interaction_effects:
                    fig = self.generate_interaction_network()
                    if output_dir is not None:
                        plt.savefig(os.path.join(output_dir, "interaction_network.png"), dpi=300)
                    plt.close(fig)
            except Exception as e:
                print(f"Could not generate interaction network: {e}")

            # Generate HTML report
            if output_dir is not None:
                self._generate_html_report(output_dir)

            print(f"Comprehensive report generated in {output_dir}")
            return output_dir

        except Exception as e:
            print(f"Error generating comprehensive report: {e}")
            if output_dir is not None:
                # Try to generate a basic report
                self._generate_basic_report(output_dir)
                return output_dir
            else:
                return None

    def _generate_random_samples(self, problem, samples):
        """Generate random samples for parameters."""
        num_vars = problem['num_vars']
        bounds = problem['bounds']

        # Generate random values within bounds
        x = np.random.random((samples, num_vars))
        for i, bound in enumerate(bounds):
            x[:, i] = x[:, i] * (bound[1] - bound[0]) + bound[0]

        return x

    def _run_sequential_simulations(self, parameter_sets):
        """Run simulations sequentially."""
        results = []
        for params in tqdm(parameter_sets, desc="Running simulations"):
            try:
                # Run simulation
                sim_result = self.simulation_function(params)

                # Add parameters to result
                result = params.copy()
                result.update(sim_result)

                results.append(result)
            except Exception as e:
                print(f"Error running simulation with parameters {params}: {e}")

        return results

    def _run_parallel_simulations(self, parameter_sets):
        """Run simulations in parallel."""
        # Determine number of workers
        try:
            num_cpus = multiprocessing.cpu_count()
            num_workers = max(1, num_cpus - 1)  # Leave one CPU free
        except:
            num_workers = 2

        # Run simulations in parallel
        results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self._run_single_simulation, params) for params in parameter_sets]

            # Collect results as they complete
            for future in tqdm(futures, desc="Running simulations"):
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    print(f"Error in simulation: {e}")

        return results

    def _run_single_simulation(self, params):
        """Run a single simulation and handle errors."""
        try:
            # Run simulation
            sim_result = self.simulation_function(params)

            # Add parameters to result
            result = params.copy()
            result.update(sim_result)

            return result
        except Exception as e:
            print(f"Error running simulation with parameters {params}: {e}")
            return None

    def _calculate_correlations(self):
        """Calculate correlations between parameters and metrics."""
        if self.results is None:
            return

        self.correlations = {}

        # Calculate correlations for each parameter and metric
        for param in self.parameter_ranges:
            if param not in self.results.columns:
                continue

            self.correlations[param] = {}

            for metric in self.metrics:
                if metric not in self.results.columns:
                    continue

                # Calculate correlation for this parameter-metric pair
                mask = self.results['changed_parameter'] == param
                if mask.any():
                    df = self.results.loc[mask, [param, metric]]
                    if len(df) > 1:
                        corr = df.corr().iloc[0, 1]
                        self.correlations[param][metric] = corr

    def run_global_sensitivity(self, num_samples=128):
        """
        Run global sensitivity analysis using Sobol method from SALib.

        Parameters:
            num_samples (int): Number of samples for Sobol sequence

        Returns:
            dict: Sensitivity indices and results
        """
        try:
            from SALib.sample import saltelli
            from SALib.analyze import sobol

            # Define problem for SALib
            problem = {
                'num_vars': len(self.parameter_ranges),
                'names': list(self.parameter_ranges.keys()),
                'bounds': [self.parameter_ranges[name][:2] for name in problem['names']]
            }

            # Generate samples using Saltelli's extension of Sobol sequence
            param_values = saltelli.sample(problem, num_samples)
            print(f"Running {param_values.shape[0]} simulations for global sensitivity analysis...")

            # Parallel computation of model outputs
            Y = {}
            for metric in self.metrics:
                Y[metric] = np.zeros(param_values.shape[0])

            # Run simulations with parameter combinations
            for i, X in enumerate(param_values):
                params_dict = dict(zip(problem['names'], X))
                self.current_params = {**self.base_parameters, **params_dict}
                sim_results = self.simulation_function(self.current_params)

                for metric in self.metrics:
                    Y[metric][i] = sim_results[metric]

                if i % 100 == 0:
                    print(f"  Completed {i}/{param_values.shape[0]} simulations")

            # Perform Sobol analysis for each metric
            results = {}
            for metric in self.metrics:
                Si = sobol.analyze(problem, Y[metric], print_to_console=False)
                results[metric] = {
                    'S1': Si['S1'],  # First-order indices
                    'S2': Si['S2'],  # Second-order indices
                    'ST': Si['ST'],  # Total-order indices
                    'parameter_names': problem['names']
                }

            return results

        except ImportError:
            print("SALib not installed. Using simpler sampling method.")
            return self._run_simple_global_sensitivity(num_samples)

    def _calculate_interaction_effects(self):
        """Calculate interaction effects between parameters."""
        if self.global_results is None:
            return

        self.interaction_effects = []

        # Calculate interactions for each parameter pair and metric
        parameters = list(self.parameter_ranges.keys())

        for i, p1 in enumerate(parameters):
            for j, p2 in enumerate(parameters):
                if j <= i:
                    continue  # Skip duplicate pairs and self-interactions

                # Calculate interaction for this parameter pair
                if p1 in self.global_results.columns and p2 in self.global_results.columns:
                    interaction_strength = self._calculate_pair_interaction(p1, p2)
                    self.interaction_effects.append((p1, p2, interaction_strength))

    def _calculate_pair_interaction(self, p1, p2):
        """Calculate interaction strength between two parameters."""
        # Simple method: calculate correlation between parameters in their effect on metrics
        interaction_strength = 0

        for metric in self.metrics:
            if metric not in self.global_results.columns:
                continue

            # Calculate parameter effects on metric
            effects_p1 = {}
            effects_p2 = {}

            # Discretize parameter values
            p1_vals = pd.qcut(self.global_results[p1], 5, duplicates='drop').cat.codes
            p2_vals = pd.qcut(self.global_results[p2], 5, duplicates='drop').cat.codes

            # Calculate average metric value for each parameter value
            for val in np.unique(p1_vals):
                mask = p1_vals == val
                effects_p1[val] = self.global_results.loc[mask, metric].mean()

            for val in np.unique(p2_vals):
                mask = p2_vals == val
                effects_p2[val] = self.global_results.loc[mask, metric].mean()

            # Calculate interaction score based on joint effects
            total_effect = 0
            for val1 in np.unique(p1_vals):
                for val2 in np.unique(p2_vals):
                    mask = (p1_vals == val1) & (p2_vals == val2)
                    if mask.any():
                        joint_effect = self.global_results.loc[mask, metric].mean()
                        independent_effect = effects_p1[val1] + effects_p2[val2] - self.global_results[metric].mean()

                        # Interaction is difference between joint and independent effects
                        diff = joint_effect - independent_effect
                        total_effect += diff * mask.sum() / len(self.global_results)

            interaction_strength += total_effect

        # Average across metrics
        if len(self.metrics) > 0:
            interaction_strength /= len(self.metrics)

        return interaction_strength

    def _generate_html_report(self, output_dir):
        """Generate HTML report of sensitivity analysis results."""
        from datetime import datetime

        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Parameter Sensitivity Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; }
                h1, h2, h3 { color: #2c3e50; }
                .section { margin-bottom: 30px; border-bottom: 1px solid #eee; padding-bottom: 20px; }
                .image-container { margin: 20px 0; text-align: center; }
                img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .footer { margin-top: 40px; text-align: center; font-size: 0.8em; color: #777; }
            </style>
        </head>
        <body>
            <h1>Parameter Sensitivity Analysis Report</h1>
            <p>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>

            <div class="section">
                <h2>Executive Summary</h2>
                <p>This report analyzes the sensitivity of simulation outputs to changes in input parameters.</p>
            </div>

            <div class="section">
                <h2>Parameter Importance</h2>
                <p>The chart below shows the relative importance of each parameter:</p>
                <div class="image-container">
                    <img src="parameter_importance.png" alt="Parameter Importance">
                </div>

                <h3>Parameter Importance Values</h3>
                <table>
                    <tr>
                        <th>Parameter</th>
                        <th>Importance</th>
                    </tr>
        """

        # Add parameter importance values
        importance = self.calculate_parameter_importance()
        for param, value in importance.items():
            html += f"""
                    <tr>
                        <td>{param}</td>
                        <td>{value:.4f}</td>
                    </tr>
            """

        html += """
                </table>
            </div>

            <div class="section">
                <h2>Tornado Plots</h2>
                <p>These plots show how varying each parameter affects specific metrics:</p>
        """

        # Add tornado plots for each metric
        for metric in self.metrics:
            html += f"""
                <h3>Impact on {metric}</h3>
                <div class="image-container">
                    <img src="tornado_{metric}.png" alt="Tornado Plot for {metric}">
                </div>
            """

        html += """
            </div>

            <div class="section">
                <h2>Parameter-Metric Sensitivity Heatmap</h2>
                <p>This heatmap shows the correlation between parameters and metrics:</p>
                <div class="image-container">
                    <img src="sensitivity_heatmap.png" alt="Sensitivity Heatmap">
                </div>
            </div>
        """

        # Add interaction network if available
        if os.path.exists(os.path.join(output_dir, "interaction_network.png")):
            html += """
            <div class="section">
                <h2>Parameter Interaction Network</h2>
                <p>This network visualizes interactions between parameters:</p>
                <div class="image-container">
                    <img src="interaction_network.png" alt="Parameter Interaction Network">
                </div>
            </div>
            """

        html += """
            <div class="footer">
                <p>Generated using ParameterSensitivityAnalyzer</p>
            </div>
        </body>
        </html>
        """

        # Write HTML to file
        with open(os.path.join(output_dir, "sensitivity_report.html"), "w") as f:
            f.write(html)

    def _generate_basic_report(self, output_dir):
        """Generate a basic text report of sensitivity analysis results."""
        from datetime import datetime

        report = f"""
        Parameter Sensitivity Analysis Report
        ====================================
        Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

        Executive Summary
        -----------------
        This report analyzes the sensitivity of simulation outputs to changes in input parameters.

        Parameter Importance
        -------------------
        """

        # Add parameter importance values
        importance = self.calculate_parameter_importance()
        for param, value in importance.items():
            report += f"{param}: {value:.4f}\n"

        # Write report to file
        with open(os.path.join(output_dir, "sensitivity_report.txt"), "w") as f:
            f.write(report)