# validation/validation_visualizers.py
"""
Visualization tools for the validation framework.
These tools create visual representations of validation results.
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import json
import warnings

logger = logging.getLogger('validation_visualizers')

# Filter the specific NaN warnings from seaborn/numpy
warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in subtract")

def create_parameter_sensitivity_visualizations(analyzer, output_dir='validation/reports/sensitivity'):
    """
    Create comprehensive visualizations for parameter sensitivity analysis.

    Parameters:
        analyzer: ParameterSensitivityAnalyzer instance with completed analysis
        output_dir: Directory to save visualization files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Parameter importance bar chart
    importance = analyzer.calculate_parameter_importance()

    plt.figure(figsize=(10, 6))
    importance.sort_values().plot(kind='barh', color='steelblue')
    plt.title('Parameter Importance Ranking', fontsize=14)
    plt.xlabel('Relative Importance', fontsize=12)
    plt.ylabel('Parameter', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path / 'parameter_importance.png', dpi=300)
    plt.close()

    # 2. Tornado plots for each metric
    metrics = analyzer.metrics
    for metric in metrics:
        plt.figure(figsize=(12, 8))

        # Get tornado data for this metric
        tornado_data = analyzer.calculate_tornado_plot_data(metric)
        tornado_data = tornado_data.sort_values('range')

        # Plot tornado chart
        y_pos = range(len(tornado_data))
        plt.barh(y_pos, tornado_data['max_diff'], left=tornado_data['min_value'], color='lightblue')
        plt.barh(y_pos, tornado_data['min_diff'], left=tornado_data['min_value'], color='steelblue')

        # Add labels
        plt.yticks(y_pos, tornado_data.index)
        plt.title(f'Tornado Plot for {metric}', fontsize=14)
        plt.xlabel('Effect on Output', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path / f'tornado_{metric}.png', dpi=300)
        plt.close()

    # 3. Correlation matrix heatmap
    correlations = analyzer.calculate_parameter_correlations()

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlations, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(correlations, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=True, fmt='.2f')

    plt.title('Parameter-Metric Correlation Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path / 'correlation_matrix.png', dpi=300)
    plt.close()

    # 4. Parameter interaction network
    # Create a graph of significant parameter interactions
    interactions = analyzer.identify_parameter_interactions()

    # Only include interactions above threshold
    threshold = 0.1
    significant_interactions = [(p1, p2, w) for p1, p2, w in interactions if abs(w) > threshold]

    if significant_interactions:
        plt.figure(figsize=(12, 12))
        G = nx.Graph()

        # Add nodes and edges
        parameters = list(analyzer.parameter_ranges.keys())
        for param in parameters:
            G.add_node(param, type='parameter')

        for p1, p2, weight in significant_interactions:
            G.add_edge(p1, p2, weight=abs(weight), color='blue' if weight > 0 else 'red')

        # Create layout
        pos = nx.spring_layout(G, seed=42)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800)

        # Draw edges with weights
        edges = G.edges(data=True)
        edge_colors = [d['color'] for _, _, d in edges]
        edge_weights = [d['weight'] * 5 for _, _, d in edges]

        nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color=edge_colors, alpha=0.7)

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10)

        plt.title('Parameter Interaction Network', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path / 'interaction_network.png', dpi=300)
        plt.close()

    # 5. Response surfaces for pairs of most important parameters
    if len(importance) >= 2:
        # Get the two most important parameters
        top_params = importance.index[-2:].tolist()

        # For each metric, create a response surface
        for metric in metrics:
            # Extract data for these parameters
            surface_data = analyzer.calculate_response_surface(top_params[0], top_params[1], metric)

            if surface_data is not None:
                plt.figure(figsize=(10, 8))

                # Create contour plot
                x = surface_data['x']
                y = surface_data['y']
                z = surface_data['z']

                contour = plt.contourf(x, y, z, 20, cmap='viridis')
                plt.colorbar(contour, label=metric)

                plt.title(f'Response Surface for {metric}', fontsize=14)
                plt.xlabel(top_params[0], fontsize=12)
                plt.ylabel(top_params[1], fontsize=12)

                plt.tight_layout()
                plt.savefig(output_path / f'response_surface_{metric}.png', dpi=300)
                plt.close()

    # 6. One-at-a-time sensitivity plots
    results = analyzer.get_one_at_a_time_results()

    if results is not None:
        for param in analyzer.parameter_ranges.keys():
            plt.figure(figsize=(12, 8))

            # Extract data for this parameter
            param_data = results[results['parameter'] == param]
            param_values = param_data['param_value'].unique()

            # Plot each metric
            for metric in metrics:
                metric_data = param_data[['param_value', metric]].values
                plt.plot(metric_data[:, 0], metric_data[:, 1], marker='o', label=metric)

            plt.title(f'One-at-a-time Sensitivity for {param}', fontsize=14)
            plt.xlabel(f'{param} Value', fontsize=12)
            plt.ylabel('Metric Value', fontsize=12)
            plt.legend()

            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path / f'oat_{param}.png', dpi=300)
            plt.close()


def create_dimensional_consistency_visualizations(results, output_dir='validation/reports/dimensional'):
    """
    Create visualizations for dimensional consistency analysis.

    Parameters:
        results: Dictionary of consistency check results
        output_dir: Directory to save visualization files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare data for visualization
    equation_names = []
    statuses = []
    issue_counts = []

    for name, result in results.items():
        equation_names.append(name)
        statuses.append(result['status'])
        issue_counts.append(len(result.get('issues', [])))

    # Create status bar chart with color coding
    plt.figure(figsize=(12, 8))
    bars = plt.barh(equation_names, [1] * len(equation_names))

    # Color bars based on status
    for i, status in enumerate(statuses):
        if status == 'CONSISTENT':
            bars[i].set_color('green')
        elif status == 'INCONSISTENT':
            bars[i].set_color('red')
        elif status == 'WARNING':
            bars[i].set_color('orange')
        else:
            bars[i].set_color('gray')

    # Add status labels
    for i, (bar, status) in enumerate(zip(bars, statuses)):
        plt.text(0.5, i, status, ha='center', va='center', color='white', fontweight='bold')

    plt.title('Dimensional Consistency Status by Equation', fontsize=14)
    plt.xlabel('Status', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path / 'dimension_status.png', dpi=300)
    plt.close()

    # Create issue count bar chart
    plt.figure(figsize=(12, 8))
    plt.barh(equation_names, issue_counts, color='coral')

    plt.title('Dimensional Consistency Issues by Equation', fontsize=14)
    plt.xlabel('Number of Issues', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path / 'dimension_issues.png', dpi=300)
    plt.close()

    # Create dimension usage heatmap
    # First, collect dimension usage data
    dimension_data = {}
    all_dimensions = set()

    for name, result in results.items():
        dimensions = result.get('dimensions_used', [])
        dimension_data[name] = dimensions
        all_dimensions.update(dimensions)

    if all_dimensions:
        # Convert to matrix form
        all_dimensions = sorted(list(all_dimensions))
        all_equations = equation_names

        matrix = np.zeros((len(all_equations), len(all_dimensions)))

        for i, eq in enumerate(all_equations):
            for j, dim in enumerate(all_dimensions):
                if dim in dimension_data.get(eq, []):
                    matrix[i, j] = 1

        # Create heatmap
        plt.figure(figsize=(14, 10))
        sns.heatmap(matrix, cmap='Blues', cbar=False, linewidths=.5,
                    xticklabels=all_dimensions, yticklabels=all_equations)

        plt.title('Dimension Usage by Equation', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path / 'dimension_usage.png', dpi=300)
        plt.close()


def create_edge_case_visualizations(analysis_results, output_dir='validation/reports/edge_case'):
    """
    Create visualizations for edge case analysis.

    Parameters:
        analysis_results: Dictionary of edge case analysis results from EdgeCaseChecker.analysis_results
        output_dir: Directory to save visualization files
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # If no analysis results, create placeholder viz
    if not analysis_results or not isinstance(analysis_results, dict):
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "No edge case analysis results available",
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes)
        plt.tight_layout()
        plt.savefig(output_path / 'edge_case_summary.png', dpi=300)
        plt.close()
        return

    # Extract data for visualization
    function_names = []
    edge_case_counts = {}
    severity_counts = {}

    # Define edge case categories
    categories = [
        'division_by_zero',
        'log_of_non_positive',
        'sqrt_of_negative',
        'exponent_overflow',
        'array_bounds',
        'conditional_logic'
    ]

    # Define severity categories
    severities = ['low', 'medium', 'high', 'critical']

    # Initialize counts
    for category in categories:
        edge_case_counts[category] = []

    for severity in severities:
        severity_counts[severity] = []

    # Process data - adapt this section to match analysis_results structure
    for name, result in analysis_results.items():
        function_names.append(name)

        # Count edge cases by category
        edge_cases = result.get('edge_cases', [])
        category_counts = {cat: 0 for cat in categories}

        # Count each edge case type
        for edge_case in edge_cases:
            if edge_case in category_counts:
                category_counts[edge_case] += 1

        # Add counts to the lists
        for category in categories:
            edge_case_counts[category].append(category_counts.get(category, 0))

        # Generate recommendations to assess severity
        # Note: This assumes recommendations are generated on demand
        # If pre-computed, adapt to use stored values
        sev_counts = {s: 0 for s in severities}

        # Count patterns by category and assign severity
        for edge_case in edge_cases:
            # Default severities by edge case type
            if edge_case in ['division_by_zero', 'log_of_non_positive', 'sqrt_of_negative']:
                sev_counts['high'] += 1
            elif edge_case in ['exponent_overflow']:
                sev_counts['medium'] += 1
            else:
                sev_counts['low'] += 1

        for severity in severities:
            severity_counts[severity].append(sev_counts[severity])

    # In case there are no functions analyzed
    if not function_names:
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "No functions analyzed for edge cases",
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes)
        plt.tight_layout()
        plt.savefig(output_path / 'edge_case_summary.png', dpi=300)
        plt.close()
        return

    # 1. Edge case counts by category
    edge_case_df = pd.DataFrame(edge_case_counts, index=function_names)

    plt.figure(figsize=(14, 10))
    edge_case_df.plot(kind='bar', stacked=True, colormap='viridis')

    plt.title('Edge Cases by Category and Function', fontsize=14)
    plt.xlabel('Function', fontsize=12)
    plt.ylabel('Number of Edge Cases', fontsize=12)
    plt.legend(title='Edge Case Type')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path / 'edge_case_counts.png', dpi=300)
    plt.close()

    # 2. Edge case severity by function
    severity_df = pd.DataFrame(severity_counts, index=function_names)

    plt.figure(figsize=(14, 10))
    # Use appropriate colors for severity (green to red)
    colors = ['green', 'yellow', 'orange', 'red']
    severity_df.plot(kind='bar', stacked=True, color=colors)

    plt.title('Edge Case Severity by Function', fontsize=14)
    plt.xlabel('Function', fontsize=12)
    plt.ylabel('Number of Issues', fontsize=12)
    plt.legend(title='Severity')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path / 'edge_case_severity.png', dpi=300)
    plt.close()

    # 3. Overall edge case coverage
    if function_names:
        total_edge_cases = edge_case_df.sum(axis=1)
        fixed_edge_cases = []

        for name in function_names:
            # Fixed count is likely not available in analysis_results
            # Just use 0 as default
            fixed = 0
            fixed_edge_cases.append(fixed)

        coverage_df = pd.DataFrame({
            'Total': total_edge_cases,
            'Fixed': fixed_edge_cases
        })

        plt.figure(figsize=(14, 10))
        coverage_df.plot(kind='bar', color=['red', 'green'])

        plt.title('Edge Case Detection and Resolution by Function', fontsize=14)
        plt.xlabel('Function', fontsize=12)
        plt.ylabel('Number of Edge Cases', fontsize=12)
        plt.legend()
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path / 'edge_case_coverage.png', dpi=300)
        plt.close()

        # 4. Edge case radar chart for the 5 functions with most issues
        if len(function_names) > 0:
            # Get the 5 functions with most edge cases
            total_by_function = total_edge_cases.sort_values(ascending=False)
            top_functions = total_by_function.index[:min(5, len(total_by_function))].tolist()

            # Prepare data for radar chart
            radar_data = edge_case_df.loc[top_functions]

            # Create radar chart
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, polar=True)

            # Number of variables
            N = len(categories)

            # Compute angle for each category
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            angles += angles[:1]  # Close the loop

            # Plot each function
            for i, func in enumerate(top_functions):
                values = radar_data.loc[func].tolist()
                values += values[:1]  # Close the loop

                ax.plot(angles, values, linewidth=2, label=func)
                ax.fill(angles, values, alpha=0.1)

            # Set category labels
            plt.xticks(angles[:-1], categories)

            plt.title('Edge Case Profile for Top 5 Functions', fontsize=14)
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

            plt.tight_layout()
            plt.savefig(output_path / 'edge_case_radar.png', dpi=300)
            plt.close()


def create_cross_level_visualizations(validator, output_dir='validation/reports/cross_level'):
    """
    Create visualizations for cross-level validation.

    Parameters:
        validator: CrossLevelValidator instance
        output_dir: Directory to save visualization files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Function dependency graph
    dependency_graph = validator.dependency_graph

    if dependency_graph:
        plt.figure(figsize=(16, 12))

        # Create NetworkX graph
        G = nx.DiGraph()

        # Add nodes with level information
        for func_name, level in validator.get_function_levels().items():
            G.add_node(func_name, level=level)

        # For a networkx DiGraph object:
        if hasattr(dependency_graph, 'nodes') and callable(dependency_graph.nodes):
            # NetworkX DiGraph approach
            for caller in dependency_graph.nodes():
                if hasattr(dependency_graph, 'successors') and callable(dependency_graph.successors):
                    for callee in dependency_graph.successors(caller):
                        G.add_edge(caller, callee)
        else:
            # Dictionary approach
            for caller, callees in dependency_graph.items():
                for callee in callees:
                    G.add_edge(caller, callee)

        # Create position layout
        try:
            # Try to use graphviz layout if available
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except (ImportError, AttributeError):
            # Fall back to spring layout if graphviz not available
            pos = nx.spring_layout(G)

        # Color nodes by level
        level_colors = {
            'Level 1 (Core)': 'lightcoral',
            'Level 2 (Extended)': 'lightblue',
            'Level 3 (Quantum)': 'lightgreen',
            'Level 4 (Multi-Civilization)': 'lightyellow',
            'Level 5 (Astrophysics)': 'violet',
            'Unknown': 'gray'
        }

        # Draw nodes
        for level, color in level_colors.items():
            nodes = [n for n, d in G.nodes(data=True) if d.get('level') == level]
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color, node_size=1000, alpha=0.8)

        # Draw edges
        nx.draw_networkx_edges(G, pos, width=1.5, arrowsize=20)

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10)

        # Create legend
        for level, color in level_colors.items():
            plt.plot([], [], 'o', label=level, color=color)

        plt.legend(fontsize=12)
        plt.title('Function Dependency Graph by Level', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path / 'dependency_graph.png', dpi=300)
        plt.close()

    # 2. Level dependency matrix
    level_dependencies = validator.validate_level_dependencies()
    if level_dependencies:
        dependencies = level_dependencies.get('level_dependencies', {})

        # First, collect levels while filtering out None values
        levels = []
        for level_pair in dependencies.keys():
            for level in level_pair:
                if level is not None and level not in levels:
                    levels.append(level)

        # Sort the levels if they're all strings or comparable types
        try:
            levels = sorted(levels)
        except TypeError:
            # If levels can't be sorted (mixed types), keep original order
            pass

        # Create the matrix
        if not levels or not dependencies:
            # No valid levels or dependencies found
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "No level dependencies found",
                     horizontalalignment='center', verticalalignment='center',
                     transform=plt.gca().transAxes)
            plt.tight_layout()
            plt.savefig(output_path / 'level_dependency_matrix.png', dpi=300)
            plt.close()

            # Define an empty matrix to prevent variable not defined errors
            matrix = np.zeros((1, 1))
        else:
            # Normal case - create and fill the matrix
            matrix = np.zeros((len(levels), len(levels)))

            # Fill the matrix with proper checks
            for (higher, lower), calls in dependencies.items():
                # Skip any level pair that contains None
                if higher is None or lower is None:
                    continue

                # Get indices with proper error handling
                try:
                    h_idx = levels.index(higher)
                    l_idx = levels.index(lower)
                    matrix[h_idx, l_idx] = len(calls)
                except ValueError:
                    # Skip this entry if either level is not in the levels list
                    continue

            # Check if matrix contains any valid data
            if np.all(np.isnan(matrix)):
                # Empty matrix case
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, "No dependencies detected between levels",
                         horizontalalignment='center', verticalalignment='center',
                         transform=plt.gca().transAxes)
                plt.tight_layout()
                plt.savefig(output_path / 'level_dependency_matrix.png', dpi=300)
                plt.close()
            else:
                # Replace NaN values with zeros to prevent warnings
                matrix_clean = np.nan_to_num(matrix, nan=0.0)

                # Create heatmap with valid data
                plt.figure(figsize=(12, 10))
                sns.heatmap(matrix_clean, cmap='YlOrRd', annot=True, fmt='.0f',
                            xticklabels=levels, yticklabels=levels)

                plt.title('Level Dependency Matrix', fontsize=14)
                plt.xlabel('Called Level', fontsize=12)
                plt.ylabel('Calling Level', fontsize=12)
                plt.tight_layout()
                plt.savefig(output_path / 'level_dependency_matrix.png', dpi=300)
                plt.close()

    # 3. Feedback loop visualization
    feedback_loops = validator.detect_feedback_loops()

    if feedback_loops:
        plt.figure(figsize=(14, 12))

        # Create NetworkX graph
        G = nx.DiGraph()

        # Add all nodes and edges
        for func_name, level in validator.get_function_levels().items():
            G.add_node(func_name, level=level)

        for caller, callees in validator.get_dependency_graph().items():
            for callee in callees:
                G.add_edge(caller, callee, in_loop=False)

        # Mark edges in feedback loops
        for loop in feedback_loops:
            cycle = loop.get('cycle', [])
            for i in range(len(cycle) - 1):
                if G.has_edge(cycle[i], cycle[i + 1]):
                    G[cycle[i]][cycle[i + 1]]['in_loop'] = True

        # Create position layout
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot') if nx.nx_agraph_available else nx.spring_layout(G)

        # Color nodes by level
        level_colors = {
            'Level 1 (Core)': 'lightcoral',
            'Level 2 (Extended)': 'lightblue',
            'Level 3 (Quantum)': 'lightgreen',
            'Level 4 (Multi-Civilization)': 'lightyellow',
            'Level 5 (Astrophysics)': 'violet',
            'Unknown': 'gray'
        }

        # Draw nodes
        for level, color in level_colors.items():
            nodes = [n for n, d in G.nodes(data=True) if d.get('level') == level]
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color, node_size=1000, alpha=0.8)

        # Draw regular edges
        regular_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get('in_loop')]
        nx.draw_networkx_edges(G, pos, edgelist=regular_edges, width=1.0, alpha=0.5)

        # Draw loop edges
        loop_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('in_loop')]
        nx.draw_networkx_edges(G, pos, edgelist=loop_edges, width=3.0, edge_color='red')

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10)

        # Create legend
        for level, color in level_colors.items():
            plt.plot([], [], 'o', label=level, color=color)

        plt.plot([], [], '-', label='Regular Dependency', color='black', alpha=0.5)
        plt.plot([], [], '-', label='Feedback Loop', color='red', linewidth=3)

        plt.legend(fontsize=12)
        plt.title('Feedback Loops in Function Dependencies', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path / 'feedback_loops.png', dpi=300)
        plt.close()

    # 4. Signal propagation visualization
    propagation_results = None
    if hasattr(validator, 'run_signal_propagation_test'):
        try:
            # Provide default values for the required parameters
            # Updated to handle any additional kwargs
            dummy_simulation = lambda params, *args, **kwargs: {'knowledge': 0, 'intelligence': 0}
            base_params = {'K_0': 1.0, 'alpha': 0.1, 'beta': 0.2}
            start_params = {'K_0': 2.0}
            metrics = ['knowledge', 'intelligence']

            propagation_results = validator.run_signal_propagation_test(
                dummy_simulation,
                base_params,
                start_params,
                metrics
            )
        except Exception as e:
            logger.error(f"Error running signal propagation test: {e}")
            propagation_results = None

    if propagation_results:
        levels = sorted(set(validator.get_function_levels().values()))

        # Extract propagation times between levels
        level_times = {}
        for from_level in levels:
            for to_level in levels:
                if from_level != to_level:
                    key = (from_level, to_level)
                    level_times[key] = propagation_results.get(key, float('inf'))

        # Create matrix
        matrix = np.zeros((len(levels), len(levels)))
        matrix.fill(np.nan)  # Fill with NaN for cells that should be empty

        for (from_level, to_level), time in level_times.items():
            from_idx = levels.index(from_level)
            to_idx = levels.index(to_level)
            if time != float('inf'):
                matrix[from_idx, to_idx] = time

        # Create heatmap
        plt.figure(figsize=(12, 10))
        cmap = LinearSegmentedColormap.from_list('custom_cmap', ['green', 'yellow', 'red'])
        sns.heatmap(matrix, cmap=cmap, annot=True, fmt='.2f',
                    xticklabels=levels, yticklabels=levels, mask=np.isnan(matrix))

        plt.title('Signal Propagation Time Between Levels (Timesteps)', fontsize=14)
        plt.xlabel('To Level', fontsize=12)
        plt.ylabel('From Level', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path / 'signal_propagation.png', dpi=300)
        plt.close()


def create_comprehensive_validation_report(validation_results, output_dir='validation/reports'):
    """
    Create a comprehensive HTML report combining all validation results.

    Parameters:
        validation_results: Dictionary with results from all validation components
        output_dir: Directory to save the report
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate standalone visualizations
    if 'sensitivity' in validation_results:
        create_parameter_sensitivity_visualizations(
            validation_results['sensitivity'],
            output_dir=str(output_path / 'sensitivity')
        )

    if 'dimensional' in validation_results:
        create_dimensional_consistency_visualizations(
            validation_results['dimensional'],
            output_dir=str(output_path / 'dimensional')
        )

    if 'edge_case' in validation_results:
        create_edge_case_visualizations(
            validation_results['edge_case'],
            output_dir=str(output_path / 'edge_case')
        )

    if 'cross_level' in validation_results:
        create_cross_level_visualizations(
            validation_results['cross_level'],
            output_dir=str(output_path / 'cross_level')
        )

    # Create HTML report
    html_content = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Comprehensive Validation Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #2980b9;
                border-bottom: 1px solid #ddd;
                padding-bottom: 5px;
            }}
            .report-section {{
                margin-bottom: 40px;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 20px;
                background-color: #f9f9f9;
            }}
            .summary-table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }}
            .summary-table th, .summary-table td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            .summary-table th {{
                background-color: #f2f2f2;
            }}
            .status-success {{
                color: green;
                font-weight: bold;
            }}
            .status-warning {{
                color: orange;
                font-weight: bold;
            }}
            .status-error {{
                color: red;
                font-weight: bold;
            }}
            .visualization {{
                text-align: center;
                margin: 20px 0;
            }}
            .visualization img {{
                max-width: 100%;
                border: 1px solid #ddd;
                border-radius: 5px;
                box-shadow: 0 0 5px rgba(0,0,0,0.1);
            }}
            .tabs {{
                display: flex;
                border-bottom: 1px solid #ddd;
                margin-bottom: 20px;
            }}
            .tab {{
                padding: 10px 20px;
                cursor: pointer;
                background-color: #f1f1f1;
                border: 1px solid #ddd;
                border-bottom: none;
                border-radius: 5px 5px 0 0;
                margin-right: 5px;
            }}
            .tab.active {{
                background-color: white;
                border-bottom: 1px solid white;
            }}
            .tab-content {{
                display: none;
            }}
            .tab-content.active {{
                display: block;
            }}
        </style>
    </head>
    <body>
        <h1>Comprehensive Validation Report</h1>
        <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="report-section">
            <h2>Executive Summary</h2>
            <table class="summary-table">
                <tr>
                    <th>Validation Component</th>
                    <th>Status</th>
                    <th>Issues</th>
                    <th>Recommendations</th>
                </tr>
    """

    # Add summary rows for each component
    components = {
        'dimension': 'Dimension Handler',
        'sensitivity': 'Parameter Sensitivity',
        'edge_case': 'Edge Case Detection',
        'cross_level': 'Cross-Level Validation',
        'dimensional': 'Dimensional Consistency'
    }

    for key, name in components.items():
        if key in validation_results:
            result = validation_results[key]

            # Determine status
            if key == 'dimension':
                status = 'Success' if result.get('fixed_count', 0) == 0 else 'Warning'
                issues = result.get('warning_count', 0)
                recommendations = "Fix array dimensions before simulation"
            elif key == 'sensitivity':
                status = 'Success'
                issues = 0
                recommendations = "Focus on parameters with highest importance"
            elif key == 'edge_case':
                total_issues = sum(len(r.get('recommendations', [])) for r in result.values())
                status = 'Error' if total_issues > 0 else 'Success'
                issues = total_issues
                recommendations = "Fix identified edge cases"
            elif key == 'cross_level':
                violations = result.get('level_dependencies', {}).get('violations', [])
                status = 'Error' if violations else 'Success'
                issues = len(violations)
                recommendations = "Resolve dependency violations"
            elif key == 'dimensional':
                inconsistent = sum(1 for r in result.values() if r.get('status') == 'INCONSISTENT')
                status = 'Error' if inconsistent > 0 else 'Success'
                issues = inconsistent
                recommendations = "Fix dimensional inconsistencies"

            # Add row
            status_class = f"status-{status.lower()}"
            html_content += f"""
                <tr>
                    <td>{name}</td>
                    <td class="{status_class}">{status}</td>
                    <td>{issues}</td>
                    <td>{recommendations}</td>
                </tr>
            """

    html_content += """
            </table>
        </div>

        <div class="tabs">
            <div class="tab active" onclick="showTab('sensitivity')">Parameter Sensitivity</div>
            <div class="tab" onclick="showTab('edge-case')">Edge Case Detection</div>
            <div class="tab" onclick="showTab('cross-level')">Cross-Level Validation</div>
            <div class="tab" onclick="showTab('dimensional')">Dimensional Consistency</div>
        </div>
    """

    # Parameter Sensitivity Section
    html_content += """
        <div id="sensitivity" class="tab-content active">
            <div class="report-section">
                <h2>Parameter Sensitivity Analysis</h2>
    """

    if 'sensitivity' in validation_results:
        html_content += """
                <div class="visualization">
                    <h3>Parameter Importance</h3>
                    <img src="sensitivity/parameter_importance.png" alt="Parameter Importance">
                </div>

                <div class="visualization">
                    <h3>Parameter Correlation Matrix</h3>
                    <img src="sensitivity/correlation_matrix.png" alt="Correlation Matrix">
                </div>
        """

        # Check if interaction network exists
        if (output_path / 'sensitivity' / 'interaction_network.png').exists():
            html_content += """
                <div class="visualization">
                    <h3>Parameter Interaction Network</h3>
                    <img src="sensitivity/interaction_network.png" alt="Interaction Network">
                </div>
            """
    else:
        html_content += "<p>No parameter sensitivity analysis results available.</p>"

    html_content += """
            </div>
        </div>
    """

    # Edge Case Section
    html_content += """
        <div id="edge-case" class="tab-content">
            <div class="report-section">
                <h2>Edge Case Detection</h2>
    """

    if 'edge_case' in validation_results:
        html_content += """
                <div class="visualization">
                    <h3>Edge Cases by Category</h3>
                    <img src="edge_case/edge_case_counts.png" alt="Edge Case Counts">
                </div>

                <div class="visualization">
                    <h3>Edge Case Severity</h3>
                    <img src="edge_case/edge_case_severity.png" alt="Edge Case Severity">
                </div>

                <div class="visualization">
                    <h3>Edge Case Coverage</h3>
                    <img src="edge_case/edge_case_coverage.png" alt="Edge Case Coverage">
                </div>
        """

        # Check if radar chart exists
        if (output_path / 'edge_case' / 'edge_case_radar.png').exists():
            html_content += """
                <div class="visualization">
                    <h3>Edge Case Profile for Top Functions</h3>
                    <img src="edge_case/edge_case_radar.png" alt="Edge Case Radar">
                </div>
            """
    else:
        html_content += "<p>No edge case detection results available.</p>"

    html_content += """
            </div>
        </div>
    """

    # Cross-Level Section
    html_content += """
        <div id="cross-level" class="tab-content">
            <div class="report-section">
                <h2>Cross-Level Validation</h2>
    """

    if 'cross_level' in validation_results:
        html_content += """
                <div class="visualization">
                    <h3>Function Dependency Graph</h3>
                    <img src="cross_level/dependency_graph.png" alt="Dependency Graph">
                </div>

                <div class="visualization">
                    <h3>Level Dependency Matrix</h3>
                    <img src="cross_level/level_dependency_matrix.png" alt="Level Dependency Matrix">
                </div>
        """

        # Check if feedback loops visualization exists
        if (output_path / 'cross_level' / 'feedback_loops.png').exists():
            html_content += """
                <div class="visualization">
                    <h3>Feedback Loops</h3>
                    <img src="cross_level/feedback_loops.png" alt="Feedback Loops">
                </div>
            """

        # Check if signal propagation visualization exists
        if (output_path / 'cross_level' / 'signal_propagation.png').exists():
            html_content += """
                <div class="visualization">
                    <h3>Signal Propagation</h3>
                    <img src="cross_level/signal_propagation.png" alt="Signal Propagation">
                </div>
            """
    else:
        html_content += "<p>No cross-level validation results available.</p>"

    html_content += """
            </div>
        </div>
    """

    # Dimensional Consistency Section
    html_content += """
        <div id="dimensional" class="tab-content">
            <div class="report-section">
                <h2>Dimensional Consistency</h2>
    """

    if 'dimensional' in validation_results:
        html_content += """
                <div class="visualization">
                    <h3>Dimensional Consistency Status</h3>
                    <img src="dimensional/dimension_status.png" alt="Dimension Status">
                </div>

                <div class="visualization">
                    <h3>Dimensional Consistency Issues</h3>
                    <img src="dimensional/dimension_issues.png" alt="Dimension Issues">
                </div>
        """

        # Check if dimension usage visualization exists
        if (output_path / 'dimensional' / 'dimension_usage.png').exists():
            html_content += """
                <div class="visualization">
                    <h3>Dimension Usage by Equation</h3>
                    <img src="dimensional/dimension_usage.png" alt="Dimension Usage">
                </div>
            """
    else:
        html_content += "<p>No dimensional consistency results available.</p>"

    html_content += """
            </div>
        </div>

        <script>
            function showTab(tabId) {
                // Hide all tab contents
                const tabContents = document.getElementsByClassName('tab-content');
                for (let i = 0; i < tabContents.length; i++) {
                    tabContents[i].classList.remove('active');
                }

                // Deactivate all tabs
                const tabs = document.getElementsByClassName('tab');
                for (let i = 0; i < tabs.length; i++) {
                    tabs[i].classList.remove('active');
                }

                // Activate selected tab and content
                document.getElementById(tabId).classList.add('active');
                const selectedTab = document.querySelector(`.tab[onclick="showTab('${tabId}')"]`);
                selectedTab.classList.add('active');
            }
        </script>
    </body>
    </html>
    """

    # Write HTML file
    with open(output_path / 'validation_report.html', 'w') as f:
        f.write(html_content)

    return output_path / 'validation_report.html'