import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm


class CrossLevelValidator:
    """
    Framework for validating interactions between different levels of the equation hierarchy.
    """

    def __init__(self, equation_functions, hierarchy_levels):
        """
        Initialize the cross-level validator.

        Args:
            equation_functions: Dictionary mapping function names to their implementations
            hierarchy_levels: Dictionary mapping level names to lists of function names
        """
        self.equation_functions = equation_functions
        self.hierarchy_levels = hierarchy_levels
        self.dependency_graph = None
        self.validation_results = {}
        self.level_metrics = {}

    def build_dependency_graph(self):
        """
        Build a directed graph of function dependencies across hierarchy levels.

        Returns:
            networkx.DiGraph: Dependency graph
        """
        try:
            import networkx as nx
            import inspect
        except ImportError:
            print("networkx and inspect are required for dependency graph analysis.")
            return None

        # Create directed graph
        G = nx.DiGraph()

        # Add nodes for all functions
        for func_name in self.equation_functions:
            # Determine level
            level = None
            for lvl, funcs in self.hierarchy_levels.items():
                if func_name in funcs:
                    level = lvl
                    break

            G.add_node(func_name, level=level)

        # Analyze function calls to determine edges
        for caller_name, caller_func in self.equation_functions.items():
            # Get function source code
            try:
                source = inspect.getsource(caller_func)
            except (TypeError, OSError):
                # Handle case where source code is not available
                continue

            # Check for calls to other functions
            for callee_name in self.equation_functions:
                if callee_name != caller_name and callee_name in source:
                    # Add edge from caller to callee
                    G.add_edge(caller_name, callee_name)

        self.dependency_graph = G
        return G

    def validate_level_dependencies(self, expected_dependencies=None):
        """
        Validate that dependencies between levels follow expected patterns.

        Args:
            expected_dependencies: Dictionary mapping tuple (from_level, to_level)
                                  to True/False indicating if dependencies are expected

        Returns:
            dict: Validation results
        """
        if self.dependency_graph is None:
            self.build_dependency_graph()

        if expected_dependencies is None:
            # Default expectation: Each level can call its own level or lower levels
            expected_dependencies = {}
            levels = sorted(set(level for level in self.hierarchy_levels.keys()))
            for i, from_level in enumerate(levels):
                for j, to_level in enumerate(levels):
                    # Can call same or lower level
                    expected_dependencies[(from_level, to_level)] = (j <= i)

        # Check actual dependencies
        level_dependencies = {}
        violations = []

        for u, v in self.dependency_graph.edges():
            u_level = self.dependency_graph.nodes[u]['level']
            v_level = self.dependency_graph.nodes[v]['level']

            # Record dependency
            if (u_level, v_level) not in level_dependencies:
                level_dependencies[(u_level, v_level)] = []

            level_dependencies[(u_level, v_level)].append((u, v))

            # Check if this is an unexpected dependency
            if not expected_dependencies.get((u_level, v_level), False):
                violations.append({
                    'from_function': u,
                    'to_function': v,
                    'from_level': u_level,
                    'to_level': v_level
                })

        results = {
            'level_dependencies': level_dependencies,
            'violations': violations,
            'is_valid': len(violations) == 0
        }

        self.validation_results['level_dependencies'] = results
        return results

    def detect_feedback_loops(self):
        """
        Detect potential feedback loops across hierarchy levels.

        Returns:
            list: Detected feedback loops
        """
        if self.dependency_graph is None:
            self.build_dependency_graph()

        # Find cycles in the graph
        try:
            cycles = list(nx.simple_cycles(self.dependency_graph))
        except nx.NetworkXNoCycle:
            cycles = []
        except AttributeError:
            # Handle case where simple_cycles might not be available
            cycles = []

        # Analyze each cycle to categorize by levels involved
        categorized_cycles = []

        for cycle in cycles:
            # Get levels for each function in the cycle
            cycle_levels = [self.dependency_graph.nodes[func]['level'] for func in cycle]

            # Calculate properties
            cycle_info = {
                'functions': cycle,
                'levels': cycle_levels,
                'unique_levels': list(set(cycle_levels)),
                'is_cross_level': len(set(cycle_levels)) > 1,
                'length': len(cycle)
            }

            categorized_cycles.append(cycle_info)

        # Sort by cross-level status and cycle length
        categorized_cycles.sort(key=lambda x: (-int(x['is_cross_level']), -x['length']))

        self.validation_results['feedback_loops'] = categorized_cycles
        return categorized_cycles

    def validate_hierarchy_consistency(self):
        """
        Validate that the overall hierarchy is consistent and well-structured.

        Returns:
            dict: Consistency validation results
        """
        # Get sorted list of levels
        levels = sorted(set(level for level in self.hierarchy_levels.keys()))

        # Create directed graph of expected level relationships
        level_graph = nx.DiGraph()

        # Add nodes for levels
        for level in levels:
            level_graph.add_node(level)

        # Add edges for expected dependencies (higher can call lower)
        for i, upper_level in enumerate(levels):
            for j, lower_level in enumerate(levels):
                if j <= i:  # Can call same or lower level
                    level_graph.add_edge(upper_level, lower_level)

        # Check if actual dependencies violate expected structure
        consistency_issues = []

        if self.dependency_graph is not None:
            # Get actual level dependencies
            actual_deps = set()
            for u, v in self.dependency_graph.edges():
                u_level = self.dependency_graph.nodes[u]['level']
                v_level = self.dependency_graph.nodes[v]['level']
                actual_deps.add((u_level, v_level))

            # Check for violations
            for from_level, to_level in actual_deps:
                if (from_level, to_level) not in level_graph.edges():
                    consistency_issues.append({
                        'issue_type': 'unexpected_dependency',
                        'from_level': from_level,
                        'to_level': to_level
                    })

        # Check for dangling levels (no functions)
        for level, funcs in self.hierarchy_levels.items():
            if not funcs:
                consistency_issues.append({
                    'issue_type': 'empty_level',
                    'level': level
                })

        results = {
            'is_consistent': len(consistency_issues) == 0,
            'issues': consistency_issues
        }

        self.validation_results['hierarchy_consistency'] = results
        return results

    def analyze_cross_level_impact(self, simulation_function, base_parameters,
                                   parameter_levels, output_metrics,
                                   perturbation_size=0.1, num_samples=10):
        """
        Analyze how perturbations at each level impact outputs at other levels.

        Args:
            simulation_function: Function that runs simulation with given parameters
            base_parameters: Dictionary of baseline parameters
            parameter_levels: Dictionary mapping parameter names to their hierarchy levels
            output_metrics: Dictionary mapping output metric names to their hierarchy levels
            perturbation_size: Relative size of parameter perturbations
            num_samples: Number of Monte Carlo samples for each parameter

        Returns:
            dict: Cross-level impact analysis results
        """
        impacts = []

        # Organize parameters by level
        params_by_level = {}
        for param, level in parameter_levels.items():
            if level not in params_by_level:
                params_by_level[level] = []
            params_by_level[level].append(param)

        # Organize metrics by level
        metrics_by_level = {}
        for metric, level in output_metrics.items():
            if level not in metrics_by_level:
                metrics_by_level[level] = []
            metrics_by_level[level].append(metric)

        # Run baseline simulation
        baseline_result = simulation_function(base_parameters)

        # For each level and parameter, measure impacts on all output levels
        for param_level, params in params_by_level.items():
            for param in params:
                if param not in base_parameters:
                    continue

                base_value = base_parameters[param]

                # Skip if base value is zero (can't do relative perturbation)
                if base_value == 0:
                    continue

                # Create perturbed values
                perturbations = np.linspace(
                    base_value * (1 - perturbation_size),
                    base_value * (1 + perturbation_size),
                    num_samples
                )

                for perturbed_value in perturbations:
                    # Create parameter set with perturbation
                    params = base_parameters.copy()
                    params[param] = perturbed_value

                    # Run simulation
                    result = simulation_function(params)

                    # For each output metric, calculate impact
                    for metric, metric_level in output_metrics.items():
                        if metric in result and metric in baseline_result:
                            # Calculate relative change
                            if baseline_result[metric] == 0:
                                rel_change = (result[metric] - baseline_result[metric])
                            else:
                                rel_change = (result[metric] - baseline_result[metric]) / abs(baseline_result[metric])

                            # Calculate sensitivity
                            param_rel_change = (perturbed_value - base_value) / abs(base_value)
                            if param_rel_change == 0:
                                sensitivity = 0
                            else:
                                sensitivity = rel_change / param_rel_change

                            impacts.append({
                                'param': param,
                                'param_level': param_level,
                                'param_value': perturbed_value,
                                'param_base_value': base_value,
                                'metric': metric,
                                'metric_level': metric_level,
                                'metric_value': result[metric],
                                'metric_base_value': baseline_result[metric],
                                'rel_change': rel_change,
                                'sensitivity': sensitivity
                            })

        # Convert to DataFrame
        impacts_df = pd.DataFrame(impacts)

        # Calculate average sensitivities between levels
        level_sensitivities = {}
        for param_level in params_by_level.keys():
            for metric_level in metrics_by_level.keys():
                mask = (impacts_df['param_level'] == param_level) & (impacts_df['metric_level'] == metric_level)
                if mask.any():
                    sensitivity = impacts_df.loc[mask, 'sensitivity'].abs().mean()
                    level_sensitivities[(param_level, metric_level)] = sensitivity

        results = {
            'impacts': impacts_df,
            'level_sensitivities': level_sensitivities
        }

        self.validation_results['cross_level_impact'] = results
        return results

    def run_signal_propagation_test(self, simulation_function, base_parameters,
                                    start_params, output_metrics, time_steps=100):
        """
        Test how signals propagate from lower to higher levels over time.

        Args:
            simulation_function: Function that runs single step with state and parameters
            base_parameters: Dictionary of baseline parameters
            start_params: Dictionary of parameters to perturb at time step 0
            output_metrics: List of output metrics to track
            time_steps: Number of time steps to simulate

        Returns:
            dict: Signal propagation results
        """
        # Initialize results
        propagation_results = {
            'time': np.arange(time_steps),
            'metrics': {metric: np.zeros(time_steps) for metric in output_metrics}
        }

        # Run baseline simulation for comparison
        baseline_state = {}
        current_state = {}

        # Run perturbed simulation
        for t in range(time_steps):
            # Apply perturbation only at t=0
            params = base_parameters.copy()
            if t == 0:
                for param, value in start_params.items():
                    params[param] = value

            # Run simulation step
            if t == 0:
                # Initialize state
                current_state = simulation_function(params, None, is_init=True)
                # Run baseline with unperturbed parameters
                baseline_state = simulation_function(base_parameters, None, is_init=True)
            else:
                # Continue simulation with current state
                current_state = simulation_function(params, current_state, is_init=False)
                # Continue baseline
                baseline_state = simulation_function(base_parameters, baseline_state, is_init=False)

            # Record metric values (as deltas from baseline)
            for metric in output_metrics:
                if metric in current_state and metric in baseline_state:
                    propagation_results['metrics'][metric][t] = current_state[metric] - baseline_state[metric]

        # Calculate propagation statistics
        propagation_stats = {
            'first_response': {},
            'peak_response': {},
            'settling_time': {}
        }

        for metric in output_metrics:
            signal = propagation_results['metrics'][metric]
            abs_signal = np.abs(signal)

            # Calculate when signal first exceeds threshold
            threshold = 0.05 * np.max(abs_signal) if np.max(abs_signal) > 0 else 0.01
            first_response_idx = np.argmax(abs_signal > threshold)
            if first_response_idx == 0 and abs_signal[0] <= threshold:
                first_response_idx = time_steps  # No response

            # Calculate peak response
            peak_idx = np.argmax(abs_signal)

            # Calculate settling time (when signal stays within threshold of final value)
            settle_threshold = 0.1 * np.max(abs_signal) if np.max(abs_signal) > 0 else 0.01
            final_value = signal[-1]
            settling_idx = time_steps

            for i in range(time_steps - 1, 0, -1):
                if abs(signal[i] - final_value) > settle_threshold:
                    settling_idx = i + 1
                    break

            propagation_stats['first_response'][metric] = first_response_idx
            propagation_stats['peak_response'][metric] = peak_idx
            propagation_stats['settling_time'][metric] = settling_idx

        results = {
            'time_series': propagation_results,
            'statistics': propagation_stats
        }

        self.validation_results['signal_propagation'] = results
        return results

    def validate_convergence_properties(self, simulation_function, base_parameters,
                                        output_metrics, time_steps=1000, dt=0.1):
        """
        Validate convergence properties of the system across levels.

        Args:
            simulation_function: Function that runs simulation with given parameters
            base_parameters: Dictionary of baseline parameters
            output_metrics: List of output metrics to track
            time_steps: Number of time steps to simulate
            dt: Time step size

        Returns:
            dict: Convergence validation results
        """
        # Run long simulation
        results = {
            'time': np.arange(time_steps) * dt,
            'metrics': {metric: np.zeros(time_steps) for metric in output_metrics}
        }

        # Simulate
        state = simulation_function(base_parameters, None, is_init=True)

        # Record initial values
        for metric in output_metrics:
            if metric in state:
                results['metrics'][metric][0] = state[metric]

        # Continue simulation
        for t in range(1, time_steps):
            state = simulation_function(base_parameters, state, is_init=False)

            # Record values
            for metric in output_metrics:
                if metric in state:
                    results['metrics'][metric][t] = state[metric]

        # Analyze convergence for each metric
        convergence_stats = {}

        for metric in output_metrics:
            signal = results['metrics'][metric]

            # Calculate if converged
            final_window = signal[-int(time_steps / 10):]  # Last 10% of simulation
            mean_final = np.mean(final_window)
            std_final = np.std(final_window)

            # Calculate if monotonic
            is_increasing = np.all(np.diff(signal) >= -1e-10)
            is_decreasing = np.all(np.diff(signal) <= 1e-10)
            is_monotonic = is_increasing or is_decreasing

            # Check for oscillations
            fft = np.fft.fft(signal - np.mean(signal))
            power = np.abs(fft) ** 2
            freqs = np.fft.fftfreq(len(signal), dt)

            # Exclude DC component (first element)
            power = power[1:]
            freqs = freqs[1:]

            # Find dominant frequencies (if any)
            threshold = 0.1 * np.max(power) if np.max(power) > 0 else 0
            dominant_indices = np.where(power > threshold)[0]
            dominant_freqs = []

            if len(dominant_indices) > 0:
                for idx in dominant_indices:
                    if freqs[idx] > 0:  # Only positive frequencies
                        dominant_freqs.append(freqs[idx])

            oscillation_period = 1 / dominant_freqs[0] if dominant_freqs else 0
            has_oscillations = len(dominant_freqs) > 0 and oscillation_period > 0

            # Calculate convergence statistics
            epsilon = 0.01 * (np.max(signal) - np.min(signal)) if np.max(signal) > np.min(signal) else 0.01

            # Time to reach within epsilon of final value
            diffs = np.abs(signal - mean_final)
            convergence_idx = np.argmax(diffs <= epsilon)

            if convergence_idx == 0 and diffs[0] > epsilon:
                convergence_idx = time_steps  # Did not converge

            convergence_time = convergence_idx * dt if convergence_idx < time_steps else float('inf')

            # Check stability at end
            is_stable = std_final < 0.01 * np.abs(mean_final) if mean_final != 0 else std_final < epsilon

            convergence_stats[metric] = {
                'converged': convergence_idx < time_steps,
                'convergence_time': convergence_time,
                'final_value': mean_final,
                'final_std': std_final,
                'is_monotonic': is_monotonic,
                'is_stable': is_stable,
                'has_oscillations': has_oscillations,
                'oscillation_period': oscillation_period if has_oscillations else 0,
                'dominant_frequencies': dominant_freqs
            }

        results = {
            'time_series': results,
            'convergence_stats': convergence_stats
        }

        self.validation_results['convergence'] = results
        return results

    def visualize_dependency_graph(self, figsize=(12, 10)):
        """
        Visualize the function dependency graph colored by hierarchy level.

        Args:
            figsize: Size of the figure

        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        if self.dependency_graph is None:
            self.build_dependency_graph()

        try:
            import matplotlib.pyplot as plt
            from matplotlib.cm import get_cmap
        except ImportError:
            print("matplotlib is required for visualization.")
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Get unique levels
        levels = sorted(set(data['level'] for _, data in self.dependency_graph.nodes(data=True)))

        # Create color map
        cmap = get_cmap('tab10', len(levels))
        color_map = {level: cmap(i) for i, level in enumerate(levels)}

        # Assign colors to nodes based on level
        node_colors = [color_map[self.dependency_graph.nodes[n]['level']] for n in self.dependency_graph.nodes()]

        # Create node labels (shortened names)
        node_labels = {node: node.replace('_', '\n') for node in self.dependency_graph.nodes()}

        # Create node sizes based on out-degree (how many other functions this calls)
        node_sizes = [300 + 100 * self.dependency_graph.out_degree(n) for n in self.dependency_graph.nodes()]

        # Draw graph
        pos = nx.spring_layout(self.dependency_graph, seed=42)

        nx.draw_networkx_nodes(self.dependency_graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
        nx.draw_networkx_edges(self.dependency_graph, pos, edge_color='gray', alpha=0.5, arrows=True)
        nx.draw_networkx_labels(self.dependency_graph, pos, labels=node_labels, font_size=10)

        # Add legend for levels
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[level], markersize=10)
                for level in levels]
        ax.legend(handles, levels, title='Hierarchy Level', loc='upper left', bbox_to_anchor=(1, 1))

        plt.title('Equation Hierarchy Dependency Graph')
        plt.axis('off')
        plt.tight_layout()

        return fig

    def visualize_cross_level_impact(self, figsize=(10, 8)):
        """
        Visualize cross-level impact as a heatmap.

        Args:
            figsize: Size of the figure

        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        if 'cross_level_impact' not in self.validation_results:
            print("No cross-level impact results found. Run analyze_cross_level_impact first.")
            return None

        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("matplotlib is required for visualization.")
            return None

        # Get level sensitivities
        level_sensitivities = self.validation_results['cross_level_impact']['level_sensitivities']

        # Get unique levels
        param_levels = set(k[0] for k in level_sensitivities.keys())
        metric_levels = set(k[1] for k in level_sensitivities.keys())

        all_levels = sorted(set(param_levels).union(metric_levels))

        # Create sensitivity matrix
        sensitivity_matrix = np.zeros((len(all_levels), len(all_levels)))

        for i, param_level in enumerate(all_levels):
            for j, metric_level in enumerate(all_levels):
                if (param_level, metric_level) in level_sensitivities:
                    sensitivity_matrix[i, j] = level_sensitivities[(param_level, metric_level)]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        im = ax.imshow(sensitivity_matrix, cmap='viridis')

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Average Sensitivity', rotation=-90, va="bottom")

        # Add labels
        ax.set_xticks(np.arange(len(all_levels)))
        ax.set_yticks(np.arange(len(all_levels)))
        ax.set_xticklabels(all_levels)
        ax.set_yticklabels(all_levels)

        # Rotate tick labels and set alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add annotations
        for i in range(len(all_levels)):
            for j in range(len(all_levels)):
                text = ax.text(j, i, f"{sensitivity_matrix[i, j]:.2f}",
                            ha="center", va="center", color="w" if sensitivity_matrix[i, j] > 0.5 else "black")

        ax.set_title("Cross-Level Impact Sensitivity")
        ax.set_xlabel("Output Level")
        ax.set_ylabel("Input Level")

        fig.tight_layout()

        return fig

    def visualize_signal_propagation(self, figsize=(12, 8)):
        """
        Visualize signal propagation across levels.

        Args:
            figsize: Size of the figure

        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        if 'signal_propagation' not in self.validation_results:
            print("No signal propagation results found. Run run_signal_propagation_test first.")
            return None

        try:
            import matplotlib.pyplot as plt
            from matplotlib.cm import get_cmap
        except ImportError:
            print("matplotlib is required for visualization.")
            return None

        # Get results
        results = self.validation_results['signal_propagation']
        time = results['time_series']['time']
        metrics = results['time_series']['metrics']
        stats = results['statistics']

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create color map
        cmap = get_cmap('tab10', len(metrics))

        # Plot each metric
        for i, (metric, values) in enumerate(metrics.items()):
            ax.plot(time, values, label=metric, color=cmap(i))

            # Mark key points
            first_response = stats['first_response'][metric]
            peak_response = stats['peak_response'][metric]
            settling_time = stats['settling_time'][metric]

            if first_response < len(time):
                ax.scatter(time[first_response], values[first_response], marker='o', color=cmap(i))
                ax.axvline(x=time[first_response], linestyle='--', alpha=0.3, color=cmap(i))

            if peak_response < len(time):
                ax.scatter(time[peak_response], values[peak_response], marker='^', color=cmap(i))

            if settling_time < len(time):
                ax.scatter(time[settling_time], values[settling_time], marker='s', color=cmap(i))
                ax.axvline(x=time[settling_time], linestyle=':', alpha=0.3, color=cmap(i))

        ax.axhline(y=0, linestyle='-', alpha=0.2, color='black')

        ax.set_title("Signal Propagation Across Levels")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Signal Amplitude")
        ax.legend()

        fig.tight_layout()

        return fig

    def generate_validation_report(self, output_dir):
        """
        Generate a comprehensive validation report in HTML format.

        Args:
            output_dir: Directory to save the report
        """
        try:
            import os
            import base64
            from io import BytesIO
            from pathlib import Path
        except ImportError:
            print("Required modules missing for report generation.")
            return

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Generate plots
        plots = {}

        # Dependency graph
        fig = self.visualize_dependency_graph()
        if fig:
            buf = BytesIO()
            fig.savefig(buf, format='png')
            plt.close(fig)
            plots['dependency_graph'] = base64.b64encode(buf.getbuffer()).decode('ascii')

        # Cross-level impact
        if 'cross_level_impact' in self.validation_results:
            fig = self.visualize_cross_level_impact()
            if fig:
                buf = BytesIO()
                fig.savefig(buf, format='png')
                plt.close(fig)
                plots['cross_level_impact'] = base64.b64encode(buf.getbuffer()).decode('ascii')

        # Signal propagation
        if 'signal_propagation' in self.validation_results:
            fig = self.visualize_signal_propagation()
            if fig:
                buf = BytesIO()
                fig.savefig(buf, format='png')
                plt.close(fig)
                plots['signal_propagation'] = base64.b64encode(buf.getbuffer()).decode('ascii')

        # Create HTML content
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cross-Level Validation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                h1, h2, h3 { color: #333; }
                .container { max-width: 1200px; margin: 0 auto; }
                .plot-container { margin: 20px 0; text-align: center; }
                .plot { max-width: 100%; height: auto; border: 1px solid #ddd; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .section { margin-bottom: 40px; border-bottom: 1px solid #eee; padding-bottom: 20px; }
                .summary { background-color: #f8f8f8; padding: 15px; border-radius: 5px; margin: 20px 0; }
                .warning { background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin: 10px 0; }
                .success { background-color: #d4edda; border-left: 4px solid #28a745; padding: 10px; margin: 10px 0; }
                .error { background-color: #f8d7da; border-left: 4px solid #dc3545; padding: 10px; margin: 10px 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Cross-Level Validation Report</h1>

                <div class="summary">
                    <h2>Validation Summary</h2>
                    <p>This report analyzes the consistency and interactions between different levels of the equation hierarchy.</p>
        """

        # Add overall validation status
        all_validations_pass = True

        if 'hierarchy_consistency' in self.validation_results:
            consistency = self.validation_results['hierarchy_consistency']
            all_validations_pass = all_validations_pass and consistency['is_consistent']

        if 'level_dependencies' in self.validation_results:
            dependencies = self.validation_results['level_dependencies']
            all_validations_pass = all_validations_pass and dependencies['is_valid']

        feedback_loops = self.validation_results.get('feedback_loops', [])
        cross_level_feedback_loops = [loop for loop in feedback_loops if loop.get('is_cross_level', False)]

        html_content += f"""
                    <div class="{'success' if all_validations_pass else 'error'}">
                        <strong>Overall Validation Status:</strong> {'PASS' if all_validations_pass else 'FAIL'}
                    </div>
                    <ul>
                        <li><strong>Hierarchy Consistency:</strong> {'PASS' if 'hierarchy_consistency' in self.validation_results and self.validation_results['hierarchy_consistency']['is_consistent'] else 'FAIL'}</li>
                        <li><strong>Level Dependencies:</strong> {'PASS' if 'level_dependencies' in self.validation_results and self.validation_results['level_dependencies']['is_valid'] else 'FAIL'}</li>
                        <li><strong>Cross-Level Feedback Loops:</strong> {len(cross_level_feedback_loops)} found</li>
                    </ul>
                </div>
        """

        # Save HTML report
        with open(os.path.join(output_dir, 'cross_level_validation_report.html'), 'w') as f:
            f.write(html_content)

        print(f"Report generated in {output_dir}")