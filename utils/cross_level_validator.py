import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import ast
import inspect
import re
from pathlib import Path


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
        Build a directed graph of function dependencies across hierarchy levels
        using AST parsing for more accurate dependency detection.

        Returns:
            networkx.DiGraph: Dependency graph
        """
        try:
            import networkx as nx
            import inspect
            import ast
        except ImportError:
            print("networkx, inspect, and ast are required for dependency graph analysis.")
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

        # Create a mapping of function objects to names
        func_name_map = {}
        for name, func in self.equation_functions.items():
            func_name_map[func] = name

        # Analyze function calls to determine edges using AST
        for caller_name, caller_func in self.equation_functions.items():
            # Get function source code
            try:
                source = inspect.getsource(caller_func)
                self._analyze_dependencies(G, caller_name, source, func_name_map)
            except (TypeError, OSError) as e:
                print(f"Error getting source for {caller_name}: {e}")
                # Try a fallback approach for built-in functions
                self._analyze_dependencies_fallback(G, caller_name, caller_func, func_name_map)

        self.dependency_graph = G
        return G

    def get_function_levels(self):
        """
        Returns a dictionary mapping function names to their hierarchy levels.

        Returns:
            dict: Mapping of function names to their hierarchy levels
        """
        result = {}
        for level, funcs in self.hierarchy_levels.items():
            for func_name in funcs:
                result[func_name] = level
        return result

    def _analyze_dependencies(self, graph, caller_name, source, func_name_map):
        """
        Analyze function dependencies using AST parsing.

        Args:
            graph: NetworkX graph to add edges to
            caller_name: Name of the calling function
            source: Source code of the calling function
            func_name_map: Mapping of function objects to names
        """
        try:
            # Parse the source code into an AST
            tree = ast.parse(source)

            # Create a visitor to find function calls
            visitor = FunctionCallVisitor(func_name_map.keys(), self.equation_functions.keys())
            visitor.visit(tree)

            # Add edges for function calls
            for callee_name in visitor.function_calls:
                if callee_name in self.equation_functions and callee_name != caller_name:
                    graph.add_edge(caller_name, callee_name)
        except SyntaxError as e:
            print(f"Syntax error in {caller_name}: {e}")
            # Fall back to simpler string matching
            self._analyze_dependencies_fallback(graph, caller_name, source, func_name_map)

    def _analyze_dependencies_fallback(self, graph, caller_name, caller_func_or_source, func_name_map):
        """
        Fallback method for analyzing dependencies using string matching.

        Args:
            graph: NetworkX graph to add edges to
            caller_name: Name of the calling function
            caller_func_or_source: Either the function object or its source code
            func_name_map: Mapping of function objects to names
        """
        # Determine if we have source code or a function object
        if isinstance(caller_func_or_source, str):
            source = caller_func_or_source
        else:
            try:
                # Try to get docstring or function repr
                source = caller_func_or_source.__doc__ or str(caller_func_or_source)
            except:
                source = str(caller_func_or_source)

        # Check for explicit calls to other functions
        for callee_name in self.equation_functions:
            if callee_name != caller_name:
                # Look for exact function name followed by opening parenthesis
                pattern = r'\b' + re.escape(callee_name) + r'\s*\('
                if re.search(pattern, source):
                    graph.add_edge(caller_name, callee_name)

        # Try to detect dependencies in default arguments or closures
        if not isinstance(caller_func_or_source, str):
            try:
                # Inspect function's defaults and closure
                try:
                    defaults = caller_func_or_source.__defaults__ or ()
                except:
                    defaults = ()

                # Check if any default value is a function we know
                for default in defaults:
                    if default in func_name_map:
                        callee_name = func_name_map[default]
                        if callee_name != caller_name:
                            graph.add_edge(caller_name, callee_name)
            except:
                pass

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

                <div class="section">
                    <h2>Equation Hierarchy Dependency Graph</h2>
        """

        # Add dependency graph plot
        if 'dependency_graph' in plots:
            html_content += f"""
                    <div class="plot-container">
                        <img class="plot" src="data:image/png;base64,{plots['dependency_graph']}" alt="Dependency Graph">
                    </div>
            """
        else:
            html_content += """
                    <p>No dependency graph available. There might be no dependencies between functions.</p>
            """

        html_content += """
                </div>

                <div class="section">
                    <h2>Level Dependency Analysis</h2>
        """

        # Add level dependency results
        if 'level_dependencies' in self.validation_results:
            dependencies = self.validation_results['level_dependencies']

            # Add dependency table if any dependencies exist
            if dependencies['level_dependencies']:
                html_content += """
                    <h3>Level-to-Level Dependencies</h3>
                    <table>
                        <tr>
                            <th>From Level</th>
                            <th>To Level</th>
                            <th>Dependencies</th>
                        </tr>
                """

                for (from_level, to_level), deps in dependencies['level_dependencies'].items():
                    html_content += f"""
                        <tr>
                            <td>{from_level}</td>
                            <td>{to_level}</td>
                            <td>{len(deps)}</td>
                        </tr>
                    """

                html_content += """
                    </table>
                """
            else:
                html_content += """
                    <div class="warning">
                        No dependencies detected between functions. This might indicate:
                        <ul>
                            <li>Functions are self-contained and don't call each other</li>
                            <li>Dependencies exist but weren't detected</li>
                            <li>The code structure uses alternative means of interaction</li>
                        </ul>
                    </div>
                """

            # Add violations if any
            if dependencies['violations']:
                html_content += """
                    <h3>Dependency Violations</h3>
                    <div class="error">
                        The following dependencies violate the expected hierarchy structure:
                    </div>
                    <table>
                        <tr>
                            <th>From Function</th>
                            <th>From Level</th>
                            <th>To Function</th>
                            <th>To Level</th>
                        </tr>
                """

                for violation in dependencies['violations']:
                    html_content += f"""
                        <tr>
                            <td>{violation['from_function']}</td>
                            <td>{violation['from_level']}</td>
                            <td>{violation['to_function']}</td>
                            <td>{violation['to_level']}</td>
                        </tr>
                    """

                html_content += """
                    </table>
                """
            else:
                html_content += """
                    <div class="success">
                        No dependency violations found. All dependencies follow the expected hierarchy structure.
                    </div>
                """

        html_content += """
                </div>

                <div class="section">
                    <h2>Feedback Loop Analysis</h2>
        """

        # Add feedback loop results
        if feedback_loops:
            # Count loops by type
            cross_level_count = len(cross_level_feedback_loops)
            single_level_count = len(feedback_loops) - cross_level_count

            html_content += f"""
                    <div class="summary">
                        <p>Found {len(feedback_loops)} feedback loops in total:</p>
                        <ul>
                            <li><strong>Cross-Level Loops:</strong> {cross_level_count}</li>
                            <li><strong>Single-Level Loops:</strong> {single_level_count}</li>
                        </ul>
                    </div>
            """

            # Add table of loops
            html_content += """
                    <h3>Detected Feedback Loops</h3>
                    <table>
                        <tr>
                            <th>Loop</th>
                            <th>Length</th>
                            <th>Levels Involved</th>
                            <th>Cross-Level</th>
                        </tr>
            """

            for i, loop in enumerate(feedback_loops):
                html_content += f"""
                        <tr>
                            <td>{' → '.join(loop['functions'])}</td>
                            <td>{loop['length']}</td>
                            <td>{', '.join(loop['unique_levels'])}</td>
                            <td>{'Yes' if loop['is_cross_level'] else 'No'}</td>
                        </tr>
                """

            html_content += """
                    </table>
            """

            # Add warning if there are cross-level loops
            if cross_level_feedback_loops:
                html_content += """
                    <div class="warning">
                        <strong>Warning:</strong> Cross-level feedback loops may cause unexpected behaviors and stability issues.
                        Consider reviewing these loops to ensure they're intentional and well-controlled.
                    </div>
                """
        else:
            html_content += """
                    <div class="success">
                        No feedback loops detected in the equation hierarchy.
                    </div>
            """

        html_content += """
                </div>

                <div class="section">
                    <h2>Recommendations</h2>
                    <ul>
        """

        # Generate recommendations
        if 'level_dependencies' in self.validation_results and len(
                self.validation_results['level_dependencies']['level_dependencies']) == 0:
            html_content += """
                        <li>
                            <strong>Consider function dependency analysis:</strong> No function dependencies were detected. If functions are intended to call each other, consider making these dependencies explicit in the code.
                        </li>
            """

        html_content += """
                        <li>
                            <strong>Document cross-level interactions:</strong> Create explicit documentation about how each hierarchy level interacts with others, even if dependencies aren't directly visible in the code.
                        </li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """

        # Save HTML report
        with open(os.path.join(output_dir, 'cross_level_validation_report.html'), 'w') as f:
            f.write(html_content)

        print(f"Report generated in {output_dir}")


class FunctionCallVisitor(ast.NodeVisitor):
    """
    AST visitor to detect function calls within source code.
    """

    def __init__(self, func_objects=None, func_names=None):
        """
        Initialize with lists of function objects and names to check.

        Args:
            func_objects: List of function objects to identify
            func_names: List of function names to check for
        """
        self.func_objects = func_objects or []
        self.func_names = set(func_names or [])
        self.function_calls = set()

    def visit_Call(self, node):
        """Visit a function call node in the AST."""
        # Check if this is a direct function name call
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.func_names:
                self.function_calls.add(func_name)

        # Check for attribute access (like module.function)
        elif isinstance(node.func, ast.Attribute):
            # Try to get the full name
            try:
                if isinstance(node.func.value, ast.Name):
                    module_name = node.func.value.id
                    func_name = node.func.attr
                    full_name = f"{module_name}.{func_name}"

                    # Check if this matches any known functions
                    if func_name in self.func_names:
                        self.function_calls.add(func_name)
            except:
                pass

        # Continue visiting child nodes
        self.generic_visit(node)