import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from functools import partial
import multiprocessing as mp
from tqdm import tqdm


class ParameterSensitivityAnalyzer:
    """
    A class for analyzing parameter sensitivity across the equation hierarchy.
    """

    def __init__(self, simulation_runner, output_metrics, base_parameters):
        """
        Initialize the sensitivity analyzer.

        Args:
            simulation_runner: Function that runs the simulation with given parameters
                               and returns the results
            output_metrics: List of output metric names to track
            base_parameters: Dictionary of baseline parameters
        """
        self.simulation_runner = simulation_runner
        self.output_metrics = output_metrics
        self.base_parameters = base_parameters
        self.results = None
        self.parameter_ranges = {}
        self.parameter_groups = {}
        self.correlation_matrix = None
        self.parameter_importance = None

    def define_parameter_ranges(self, ranges_dict):
        """
        Define ranges for parameters to analyze.

        Args:
            ranges_dict: Dictionary mapping parameter names to (min, max, num_points)
        """
        self.parameter_ranges = {}
        for param, (min_val, max_val, num_points) in ranges_dict.items():
            self.parameter_ranges[param] = np.linspace(min_val, max_val, num_points)

    def define_parameter_groups(self, groups_dict):
        """
        Define parameter groups for hierarchical analysis.

        Args:
            groups_dict: Dictionary mapping group names to lists of parameter names
        """
        self.parameter_groups = groups_dict

    def run_one_at_a_time_sensitivity(self, parallel=True, num_processes=None):
        """
        Run one-at-a-time sensitivity analysis, varying each parameter individually.

        Args:
            parallel: Whether to run simulations in parallel
            num_processes: Number of processes to use for parallel execution

        Returns:
            DataFrame containing sensitivity results
        """
        results = []

        # Create list of parameter combinations to test
        param_combinations = []
        param_names = []

        for param_name, param_values in self.parameter_ranges.items():
            base_params = self.base_parameters.copy()
            for value in param_values:
                if value != self.base_parameters.get(param_name, None):
                    test_params = base_params.copy()
                    test_params[param_name] = value
                    param_combinations.append(test_params)
                    param_names.append((param_name, value))

        # Add baseline
        param_combinations.append(self.base_parameters.copy())
        param_names.append(("baseline", None))

        # Run simulations
        if parallel and mp.cpu_count() > 1:
            num_proc = num_processes if num_processes else max(1, mp.cpu_count() - 1)
            with mp.Pool(num_proc) as pool:
                simulation_results = list(tqdm(
                    pool.imap(self.simulation_runner, param_combinations),
                    total=len(param_combinations)
                ))
        else:
            simulation_results = []
            for params in tqdm(param_combinations):
                simulation_results.append(self.simulation_runner(params))

        # Process results
        for (param_name, param_value), sim_result in zip(param_names, simulation_results):
            result_entry = {
                "parameter": param_name,
                "value": param_value
            }

            # Extract output metrics
            for metric in self.output_metrics:
                if isinstance(sim_result, dict) and metric in sim_result:
                    result_entry[metric] = sim_result[metric]
                elif hasattr(sim_result, metric):
                    result_entry[metric] = getattr(sim_result, metric)
                else:
                    result_entry[metric] = None

            results.append(result_entry)

        # Convert to DataFrame
        self.results = pd.DataFrame(results)
        return self.results

    def run_global_sensitivity(self, num_samples=100, method='sobol', parallel=True, num_processes=None):
        """
        Run global sensitivity analysis using Sobol or Latin Hypercube sampling.

        Args:
            num_samples: Number of parameter combinations to sample
            method: Sampling method ('sobol' or 'latin')
            parallel: Whether to run simulations in parallel
            num_processes: Number of processes to use for parallel execution

        Returns:
            DataFrame containing sensitivity results
        """
        try:
            from SALib.sample import sobol as sobol_sample
            from SALib.sample import latin as latin_sample
            from SALib.analyze import sobol as sobol_analyze
            import numpy as np
        except ImportError:
            print("SALib package is required for global sensitivity analysis.")
            print("Install it with: pip install SALib")
            return None

        # Define problem
        problem = {
            'num_vars': len(self.parameter_ranges),
            'names': list(self.parameter_ranges.keys()),
            'bounds': [[min(values), max(values)] for values in self.parameter_ranges.values()]
        }

        # Generate samples
        if method == 'sobol':
            param_values = sobol_sample.sample(problem, num_samples)
        elif method == 'latin':
            param_values = latin_sample.sample(problem, num_samples)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        # Create parameter combinations
        param_combinations = []
        for sample in param_values:
            params = self.base_parameters.copy()
            for i, param_name in enumerate(problem['names']):
                params[param_name] = sample[i]
            param_combinations.append(params)

        # Run simulations
        if parallel and mp.cpu_count() > 1:
            num_proc = num_processes if num_processes else max(1, mp.cpu_count() - 1)
            with mp.Pool(num_proc) as pool:
                simulation_results = list(tqdm(
                    pool.imap(self.simulation_runner, param_combinations),
                    total=len(param_combinations)
                ))
        else:
            simulation_results = []
            for params in tqdm(param_combinations):
                simulation_results.append(self.simulation_runner(params))

        # Process results for each output metric
        results = []
        for i, params in enumerate(param_combinations):
            result_entry = {param: value for param, value in params.items()}

            # Extract output metrics
            sim_result = simulation_results[i]
            for metric in self.output_metrics:
                if isinstance(sim_result, dict) and metric in sim_result:
                    result_entry[metric] = sim_result[metric]
                elif hasattr(sim_result, metric):
                    result_entry[metric] = getattr(sim_result, metric)
                else:
                    result_entry[metric] = None

            results.append(result_entry)

        # Convert to DataFrame
        self.results = pd.DataFrame(results)

        # Calculate Sobol indices for each output metric
        sensitivity_indices = {}
        for metric in self.output_metrics:
            if not all(self.results[metric].notna()):
                continue

            # Create Y values for Sobol analysis
            Y = self.results[metric].values

            try:
                # Run Sobol analysis
                Si = sobol_analyze.analyze(problem, Y, print_to_console=False)
                sensitivity_indices[metric] = {
                    'S1': {problem['names'][i]: Si['S1'][i] for i in range(len(problem['names']))},
                    'ST': {problem['names'][i]: Si['ST'][i] for i in range(len(problem['names']))},
                }
            except Exception as e:
                print(f"Error calculating Sobol indices for {metric}: {e}")

        self.sensitivity_indices = sensitivity_indices
        return self.results, sensitivity_indices

    def calculate_parameter_correlations(self):
        """
        Calculate parameter correlations with output metrics.

        Returns:
            DataFrame containing correlation coefficients
        """
        if self.results is None:
            raise ValueError("Must run sensitivity analysis before calculating correlations")

        # Extract parameter columns and output metrics
        param_cols = [col for col in self.results.columns
                      if col not in self.output_metrics and col not in ['parameter', 'value']]

        # Calculate correlation matrix
        data_for_corr = self.results[param_cols + self.output_metrics].copy()
        self.correlation_matrix = data_for_corr.corr()

        # Extract just the parameter-to-output correlations
        param_output_corr = self.correlation_matrix.loc[param_cols, self.output_metrics]

        return param_output_corr

    def calculate_parameter_importance(self, method='sobol'):
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

        if method == 'sobol' and hasattr(self, 'sobol_results'):
            # Use Sobol indices if available
            importance = pd.Series({
                param: self.sobol_results['S1'][i]
                for i, param in enumerate(self.parameters)
            })
        elif method == 'correlation' and hasattr(self, 'correlations'):
            # Use correlation-based importance if available
            importance = pd.Series({
                param: self.correlations.get(param, {}).max()
                for param in self.parameters
            })
        else:
            # Fallback to range-based importance (always works)
            importance = {}

            # Get unique parameter values
            for param in self.parameters:
                param_values = self.results[param].unique()

                if len(param_values) <= 1:
                    # Skip parameters that don't vary
                    importance[param] = 0.0
                    continue

                # Calculate total impact across all metrics
                total_impact = 0.0
                for metric in self.metrics:
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

    def identify_interaction_effects(self):
        """
        Identify parameters with strong interaction effects.

        Returns:
            DataFrame of parameter pairs with interaction measures
        """
        if not hasattr(self, 'sensitivity_indices'):
            raise ValueError("Must run global sensitivity analysis before identifying interactions")

        interactions = []

        for metric, indices in self.sensitivity_indices.items():
            # Sobol total effect - first order effect = interaction effect
            for param in indices['S1'].keys():
                interaction = indices['ST'][param] - indices['S1'][param]
                interactions.append({
                    'metric': metric,
                    'parameter': param,
                    'interaction_effect': interaction
                })

        return pd.DataFrame(interactions)

    def run_sequential_dependency_analysis(self, dependency_chains):
        """
        Analyze how parameter changes propagate through dependency chains.

        Args:
            dependency_chains: List of parameter chains to analyze, where each chain
                              is a list of parameters in sequence of dependency

        Returns:
            DataFrame with propagation analysis results
        """
        results = []

        for chain in dependency_chains:
            # Modify first parameter in chain
            for param_name in chain:
                if param_name not in self.parameter_ranges:
                    continue

                param_values = self.parameter_ranges[param_name]

                for value in param_values:
                    # Create parameter set with modified value
                    params = self.base_parameters.copy()
                    params[param_name] = value

                    # Run simulation
                    result = self.simulation_runner(params)

                    # Record impact on each parameter in the chain
                    entry = {
                        'chain': '->'.join(chain),
                        'modified_param': param_name,
                        'value': value
                    }

                    # Extract relevant metrics for parameters in chain
                    for chain_param in chain:
                        for metric in self.output_metrics:
                            if chain_param in metric:  # Simple heuristic to find relevant metrics
                                if isinstance(result, dict) and metric in result:
                                    entry[f"{chain_param}_{metric}"] = result[metric]
                                elif hasattr(result, metric):
                                    entry[f"{chain_param}_{metric}"] = getattr(result, metric)

                    results.append(entry)

        return pd.DataFrame(results)

    def plot_sensitivity_tornado(self, metric, top_n=10):
        """
        Create tornado plot for sensitivity of one output metric.

        Args:
            metric: Name of output metric to analyze
            top_n: Number of most important parameters to include

        Returns:
            matplotlib Figure
        """
        if self.results is None:
            raise ValueError("Must run sensitivity analysis before plotting")

        # Get baseline value
        baseline = self.results[self.results['parameter'] == 'baseline'][metric].values[0]

        # Calculate parameter impacts
        impacts = []

        for param in self.parameter_ranges.keys():
            param_results = self.results[self.results['parameter'] == param]
            if param_results.empty:
                continue

            min_val = param_results[metric].min()
            max_val = param_results[metric].max()

            impacts.append({
                'parameter': param,
                'min_impact': min_val - baseline,
                'max_impact': max_val - baseline,
                'range': max_val - min_val
            })

        # Convert to DataFrame and sort by impact range
        impacts_df = pd.DataFrame(impacts).sort_values('range', ascending=False).head(top_n)

        # Create tornado plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot horizontal bars
        y_pos = np.arange(len(impacts_df))
        ax.barh(y_pos, impacts_df['max_impact'], left=baseline, color='green', alpha=0.6)
        ax.barh(y_pos, impacts_df['min_impact'], left=baseline, color='red', alpha=0.6)

        # Add parameter names
        ax.set_yticks(y_pos)
        ax.set_yticklabels(impacts_df['parameter'])

        # Add vertical line at baseline
        ax.axvline(x=baseline, color='black', linestyle='--')

        # Add labels
        ax.set_xlabel(f'Impact on {metric}')
        ax.set_title(f'Sensitivity of {metric} to Parameters')

        plt.tight_layout()
        return fig

    def plot_parameter_importance(self, method='correlation'):
        """
        Plot overall parameter importance across all outputs.

        Args:
            method: Method to use for importance calculation

        Returns:
            matplotlib Figure
        """
        importance = self.calculate_parameter_importance(method)

        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 8))
        importance.plot(kind='bar', ax=ax)

        # Add labels
        ax.set_ylabel('Importance Score')
        ax.set_title(f'Parameter Importance ({method})')

        plt.tight_layout()
        return fig

    def plot_parameter_interactions(self, threshold=0.1):
        """
        Plot parameter interaction network.

        Args:
            threshold: Minimum interaction strength to include

        Returns:
            matplotlib Figure
        """
        try:
            import networkx as nx
            from matplotlib.cm import get_cmap
        except ImportError:
            print("networkx is required for interaction networks.")
            print("Install it with: pip install networkx")
            return None

        interactions_df = self.identify_interaction_effects()

        # Filter by threshold
        strong_interactions = interactions_df[interactions_df['interaction_effect'] > threshold]

        # Create graph
        G = nx.Graph()

        # Add nodes (parameters)
        for param in self.parameter_ranges.keys():
            G.add_node(param)

        # Add edges (interactions)
        for _, row in strong_interactions.iterrows():
            param = row['parameter']
            for other_param in self.parameter_ranges.keys():
                if param != other_param:
                    # Check if this pair has strong interactions
                    other_row = strong_interactions[
                        (strong_interactions['parameter'] == other_param) &
                        (strong_interactions['metric'] == row['metric'])
                        ]

                    if not other_row.empty:
                        # Add edge with weight based on interaction effect
                        weight = (row['interaction_effect'] + other_row['interaction_effect'].values[0]) / 2
                        if weight > threshold:
                            G.add_edge(param, other_param, weight=weight, metric=row['metric'])

        # Plot network
        fig, ax = plt.subplots(figsize=(12, 10))

        # Layout
        pos = nx.spring_layout(G, seed=42)

        # Get edge weights for width and color
        edges = G.edges()
        weights = [G[u][v]['weight'] * 5 for u, v in edges]  # Scale for visibility

        # Color map based on associated metric
        unique_metrics = interactions_df['metric'].unique()
        color_map = get_cmap('tab10', len(unique_metrics))
        metric_colors = {metric: color_map(i) for i, metric in enumerate(unique_metrics)}

        edge_colors = [metric_colors[G[u][v]['metric']] for u, v in edges]

        # Draw
        nx.draw_networkx_nodes(G, pos, node_size=500, alpha=0.8)
        nx.draw_networkx_labels(G, pos, font_size=10)
        nx.draw_networkx_edges(G, pos, width=weights, edge_color=edge_colors, alpha=0.7)

        # Add legend for metrics
        handles = [plt.Line2D([0], [0], color=color, lw=4) for color in metric_colors.values()]
        labels = list(metric_colors.keys())
        plt.legend(handles, labels, title='Metrics')

        plt.title('Parameter Interaction Network')
        plt.axis('off')

        return fig

    def generate_comprehensive_report(self, output_dir):
        """
        Generate comprehensive HTML report of sensitivity analysis.

        Args:
            output_dir: Directory to save report files
        """
        try:
            import json
            from pathlib import Path
            import base64
            from io import BytesIO
        except ImportError:
            print("Required packages missing for report generation.")
            return

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Generate all plots
        plots = {}

        # Sensitivity tornados for each metric
        for metric in self.output_metrics:
            fig = self.plot_sensitivity_tornado(metric)
            buf = BytesIO()
            fig.savefig(buf, format='png')
            plt.close(fig)
            plots[f"tornado_{metric}"] = base64.b64encode(buf.getbuffer()).decode("ascii")

        # Parameter importance
        fig = self.plot_parameter_importance()
        buf = BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        plots["parameter_importance"] = base64.b64encode(buf.getbuffer()).decode("ascii")

        # Parameter interactions
        fig = self.plot_parameter_interactions()
        if fig:
            buf = BytesIO()
            fig.savefig(buf, format='png')
            plt.close(fig)
            plots["parameter_interactions"] = base64.b64encode(buf.getbuffer()).decode("ascii")

        # Create data for report
        report_data = {
            "plots": plots,
            "parameter_importance": self.parameter_importance.to_dict() if self.parameter_importance is not None else {},
            "parameter_correlations": self.correlation_matrix.to_dict() if self.correlation_matrix is not None else {},
            "parameter_ranges": {k: list(v) for k, v in self.parameter_ranges.items()},
            "output_metrics": self.output_metrics
        }

        # Save data as JSON
        with open(Path(output_dir) / "sensitivity_data.json", "w") as f:
            json.dump(report_data, f)

        # Create HTML report
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Parameter Sensitivity Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                h1, h2, h3 { color: #333; }
                .container { max-width: 1200px; margin: 0 auto; }
                .plot-container { margin: 20px 0; }
                .plot { max-width: 100%; height: auto; border: 1px solid #ddd; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .metric-section { margin-bottom: 40px; border-bottom: 1px solid #eee; padding-bottom: 20px; }
                .summary { background-color: #f8f8f8; padding: 15px; border-radius: 5px; margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Parameter Sensitivity Analysis Report</h1>

                <div class="summary">
                    <h2>Executive Summary</h2>
                    <p>This report analyzes the sensitivity of simulation outputs to various input parameters across the equation hierarchy.</p>
                </div>

                <h2>Overall Parameter Importance</h2>
                <div class="plot-container">
                    <img class="plot" src="data:image/png;base64,{plots['parameter_importance']}" alt="Parameter Importance">
                </div>

                <h2>Parameter Interactions</h2>
                <div class="plot-container">
                    <img class="plot" src="data:image/png;base64,{plots.get('parameter_interactions', '')}" alt="Parameter Interactions">
                </div>

                <h2>Sensitivity Analysis by Metric</h2>
        """

        # Add tornado plots for each metric
        for metric in self.output_metrics:
            html_content += f"""
                <div class="metric-section">
                    <h3>Sensitivity of {metric}</h3>
                    <div class="plot-container">
                        <img class="plot" src="data:image/png;base64,{plots[f'tornado_{metric}']}" alt="Tornado Plot for {metric}">
                    </div>
                </div>
            """

        # Add importance table
        html_content += """
                <h2>Parameter Importance Rankings</h2>
                <table>
                    <tr>
                        <th>Parameter</th>
                        <th>Importance Score</th>
                    </tr>
        """

        for param, score in self.parameter_importance.items():
            html_content += f"""
                    <tr>
                        <td>{param}</td>
                        <td>{score:.4f}</td>
                    </tr>
            """

        html_content += """
                </table>

                <h2>Recommendations</h2>
                <div class="summary">
                    <p>Based on the sensitivity analysis, the following parameters should be prioritized for empirical validation:</p>
                    <ul>
        """

        # Add recommendations for top 5 parameters
        for param in self.parameter_importance.index[:5]:
            html_content += f"""
                        <li><strong>{param}</strong>: High sensitivity across multiple outputs</li>
            """

        html_content += """
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """

        # Save HTML report
        with open(Path(output_dir) / "sensitivity_report.html", "w") as f:
            f.write(html_content.format(plots=plots))

        print(f"Report generated in {output_dir}")


# Example usage:
def run_example_sensitivity_analysis():
    """
    Example of how to use the sensitivity analyzer with the axiomatic framework.
    """
    # Import necessary components
    from config.equations import (
        intelligence_growth, truth_adoption, wisdom_field,
        resistance_resurgence, suppression_feedback
    )

    # Define a simple simulation runner for demonstration
    def simple_simulation(params):
        """Simple simulation that returns key metrics."""
        # Initial values
        I = params.get('I_0', 5.0)
        K = params.get('K_0', 1.0)
        S = params.get('S_0', 10.0)
        T = params.get('T_0', 1.0)

        # Time steps
        dt = 1.0
        steps = 100

        # Results storage
        results = {
            'I': np.zeros(steps),
            'K': np.zeros(steps),
            'S': np.zeros(steps),
            'T': np.zeros(steps)
        }

        results['I'][0] = I
        results['K'][0] = K
        results['S'][0] = S
        results['T'][0] = T

        # Simulation loop
        for t in range(1, steps):
            # Calculate wisdom
            W = wisdom_field(
                1.0,
                params.get('alpha_wisdom', 0.1),
                results['S'][t - 1],
                params.get('resistance', 2.0),
                results['K'][t - 1]
            )

            # Update intelligence
            i_growth = intelligence_growth(
                results['K'][t - 1],
                W,
                params.get('resistance', 2.0),
                results['S'][t - 1],
                1.5
            )
            results['I'][t] = max(0, results['I'][t - 1] + i_growth * dt)

            # Update truth
            truth_change = truth_adoption(
                results['T'][t - 1],
                params.get('truth_adoption_rate', 0.5),
                params.get('truth_max', 40.0)
            )
            results['T'][t] = max(0, results['T'][t - 1] + truth_change * dt)

            # Update suppression
            suppression_fb = suppression_feedback(
                params.get('alpha_feedback', 0.1),
                results['S'][t - 1],
                params.get('beta_feedback', 0.05),
                results['K'][t - 1]
            )
            results['S'][t] = max(0, results['S'][t - 1] + suppression_fb * dt)

            # Update knowledge
            if results['T'][t - 1] > params.get('t_crit_phase', 20.0):
                # Phase transition
                growth_term = params.get('knowledge_growth_rate', 0.05) * results['K'][t - 1] * (
                        1 + params.get('gamma_phase', 0.1) * (results['T'][t - 1] - params.get('t_crit_phase', 20.0)) /
                        (1 + abs(results['T'][t - 1] - params.get('t_crit_phase', 20.0)))
                )
            else:
                # Simple growth
                growth_term = params.get('knowledge_growth_rate', 0.05) * results['K'][t - 1]

            results['K'][t] = max(0, results['K'][t - 1] + growth_term * dt)

        # Calculate summary metrics
        return {
            'final_intelligence': results['I'][-1],
            'final_knowledge': results['K'][-1],
            'final_suppression': results['S'][-1],
            'final_truth': results['T'][-1],
            'max_intelligence': np.max(results['I']),
            'knowledge_growth_rate': (results['K'][-1] - results['K'][0]) / steps,
            'truth_convergence_time': np.argmax(results['T'] > 0.9 * params.get('truth_max', 40.0)) if np.any(
                results['T'] > 0.9 * params.get('truth_max', 40.0)) else steps,
            'suppression_decay_rate': (results['S'][0] - results['S'][-1]) / steps if results['S'][0] > results['S'][
                -1] else 0
        }

    # Base parameters
    base_params = {
        'K_0': 1.0,  # Initial knowledge
        'S_0': 10.0,  # Initial suppression
        'I_0': 5.0,  # Initial intelligence
        'T_0': 1.0,  # Initial truth
        'knowledge_growth_rate': 0.05,  # Base knowledge growth rate
        'truth_adoption_rate': 0.5,  # Truth adoption acceleration
        'truth_max': 40.0,  # Maximum theoretical truth
        'suppression_decay': 0.05,  # Suppression decay rate
        'alpha_feedback': 0.1,  # Suppression reinforcement coefficient
        'beta_feedback': 0.05,  # Knowledge disruption coefficient
        'alpha_wisdom': 0.1,  # Wisdom scaling with suppression
        'resistance': 2.0,  # Base resistance level
        'gamma_phase': 0.1,  # Phase transition sharpness
        't_crit_phase': 20.0  # Critical threshold for transition
    }

    # Output metrics to track
    output_metrics = [
        'final_intelligence',
        'final_knowledge',
        'final_suppression',
        'final_truth',
        'max_intelligence',
        'knowledge_growth_rate',
        'truth_convergence_time',
        'suppression_decay_rate'
    ]

    # Create sensitivity analyzer
    analyzer = ParameterSensitivityAnalyzer(
        simple_simulation,
        output_metrics,
        base_params
    )

    # Define parameter ranges
    analyzer.define_parameter_ranges({
        'K_0': (0.1, 5.0, 5),
        'S_0': (5.0, 20.0, 5),
        'truth_adoption_rate': (0.1, 1.0, 5),
        'alpha_feedback': (0.05, 0.2, 5),
        'beta_feedback': (0.01, 0.1, 5),
        'gamma_phase': (0.05, 0.2, 5),
        't_crit_phase': (10.0, 30.0, 5)
    })

    # Define parameter groups by equation level
    analyzer.define_parameter_groups({
        'Level 1 (Core)': ['K_0', 'S_0', 'I_0', 'T_0', 'truth_adoption_rate'],
        'Level 2 (Extended)': ['alpha_feedback', 'beta_feedback', 'gamma_phase', 't_crit_phase'],
        'Level 3 (Quantum)': [],  # No quantum parameters in this simple example
        'Level 4 (Multi-Civ)': [],  # No multi-civilization parameters in this simple example
        'Level 5 (Astrophysics)': []  # No astrophysics parameters in this simple example
    })

    # Run one-at-a-time sensitivity analysis
    results = analyzer.run_one_at_a_time_sensitivity()

    # Calculate correlations
    correlations = analyzer.calculate_parameter_correlations()

    # Calculate parameter importance
    importance = analyzer.calculate_parameter_importance()

    # Generate report
    analyzer.generate_comprehensive_report("sensitivity_analysis_report")

    return analyzer