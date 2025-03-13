"""
Cross-Scale Interaction Validator
This tool validates interactions between equations operating at different scales,
ensuring proper integration from quantum to cosmic scales.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import importlib
import inspect
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CrossScaleValidator")


class CrossScaleValidator:
    """
    Validates interactions between equations operating at different scales.
    """

    def __init__(self):
        """Initialize the cross-scale validator."""
        self.scale_levels = [
            "quantum",
            "agent",
            "group",
            "civilization",
            "multi_civilization",
            "cosmic"
        ]

        self.key_transitions = [
            ("quantum_tunneling_probability", "truth_adoption"),
            ("quantum_entanglement_correlation", "knowledge_field_influence"),
            ("knowledge_field_influence", "knowledge_diffusion"),
            ("truth_adoption", "civilization_lifecycle_phase"),
            ("suppression_feedback", "suppression_event_horizon"),
            ("knowledge_field_gradient", "galactic_structure_model")
        ]

        # Define expected parameter transformations between scales
        self.parameter_mappings = {
            "quantum": {
                "agent": {
                    "barrier_height": "suppression",
                    "energy_level": "knowledge",
                    "tunneling_constant": "wisdom"
                }
            },
            "agent": {
                "group": {
                    "knowledge": "K_i",
                    "suppression": "S_i",
                    "wisdom": "W_i"
                }
            },
            "group": {
                "civilization": {
                    "K_i": "knowledge_array",
                    "S_i": "suppression_array",
                    "agent_positions": "positions"
                }
            },
            "civilization": {
                "multi_civilization": {
                    "age": "ages",
                    "intensity": "influence_array",
                    "phase_thresholds": "civilization characteristics"
                }
            },
            "multi_civilization": {
                "cosmic": {
                    "interaction_strength": "galactic_influence",
                    "civilizations": "cosmic_structure"
                }
            }
        }

    def evaluate_cross_scale_interactions(self, levels=None, key_transitions=None,
                                          source_modules=None):
        """
        Evaluate the quality of interactions between equations operating at different scales.

        Args:
            levels: List of scale levels to evaluate
            key_transitions: List of key transitions to evaluate
            source_modules: List of module names containing equations

        Returns:
            Dictionary containing evaluation results
        """
        if levels is None:
            levels = self.scale_levels

        if key_transitions is None:
            key_transitions = self.key_transitions

        if source_modules is None:
            source_modules = ["equations", "astrophysics_extensions",
                              "quantum_em_extensions", "multi_civilization_extensions"]

        logger.info(f"Evaluating cross-scale interactions across {len(levels)} levels")

        # Load all equations
        all_equations = self._load_equations(source_modules)

        # Build dependency graph
        dependency_graph = self._build_dependency_graph(all_equations)

        # Evaluate scale transitions
        scale_transitions = self._evaluate_scale_transitions(all_equations, levels)

        # Evaluate key transition quality
        transition_quality = self._evaluate_transition_quality(all_equations, key_transitions)

        # Simulate signal propagation
        signal_propagation = self._simulate_signal_propagation(dependency_graph, levels)

        return {
            "dependency_graph": dependency_graph,
            "scale_transitions": scale_transitions,
            "transition_quality": transition_quality,
            "signal_propagation": signal_propagation,
            "overall_integration_score": self._calculate_overall_score(scale_transitions, transition_quality)
        }

    def _load_equations(self, module_names):
        """
        Load equation functions from specified modules.

        Args:
            module_names: List of module names to load

        Returns:
            Dictionary of equations by function name
        """
        all_equations = {}

        for module_name in module_names:
            try:
                # Import the module
                full_module_name = f"config.{module_name}"
                module = importlib.import_module(full_module_name)

                # Get all functions from the module
                for name, func in inspect.getmembers(module, inspect.isfunction):
                    # Skip private functions
                    if name.startswith('_'):
                        continue

                    # Extract function information
                    source = inspect.getsource(func)
                    signature = str(inspect.signature(func))
                    docstring = inspect.getdoc(func) or ""
                    parameter_names = list(inspect.signature(func).parameters.keys())

                    # Determine scale level
                    scale = self._determine_scale_level(name, docstring, source)

                    all_equations[name] = {
                        "name": name,
                        "module": module_name,
                        "signature": signature,
                        "docstring": docstring,
                        "parameters": parameter_names,
                        "scale_level": scale,
                        "source": source
                    }

                logger.info(f"Loaded equations from {module_name}")

            except ImportError as e:
                logger.error(f"Could not import module {module_name}: {str(e)}")
            except Exception as e:
                logger.error(f"Error processing module {module_name}: {str(e)}")

        return all_equations

    def _determine_scale_level(self, name, docstring, source):
        """
        Determine the scale level for an equation.

        Args:
            name: Function name
            docstring: Function docstring
            source: Function source code

        Returns:
            String representing the scale level
        """
        combined_text = f"{name} {docstring}".lower()

        scale_keywords = {
            "quantum": ["quantum", "tunneling", "entanglement"],
            "agent": ["agent", "individual", "person", "free will"],
            "group": ["group", "network", "multiple agents"],
            "civilization": ["civilization", "society", "culture"],
            "multi_civilization": ["multi", "interaction", "diffusion"],
            "cosmic": ["cosmic", "universal", "galactic", "lifecycle"]
        }

        # Name-based mappings
        name_mappings = {
            "intelligence_growth": "agent",
            "free_will_decision": "agent",
            "truth_adoption": "agent",
            "wisdom_field": "agent",
            "resistance_resurgence": "agent",
            "suppression_feedback": "agent",
            "quantum_tunneling_probability": "quantum",
            "quantum_entanglement_correlation": "quantum",
            "knowledge_field_influence": "group",
            "knowledge_field_gradient": "group",
            "build_entanglement_network": "group",
            "civilization_lifecycle_phase": "civilization",
            "suppression_event_horizon": "civilization",
            "knowledge_gravitational_lensing": "civilization",
            "knowledge_diffusion": "multi_civilization",
            "cultural_influence": "multi_civilization",
            "resource_competition": "multi_civilization",
            "civilization_movement": "multi_civilization",
            "galactic_structure_model": "cosmic",
            "cosmic_background_knowledge": "cosmic",
            "dark_energy_knowledge_acceleration": "cosmic"
        }

        # Check direct mapping first
        if name in name_mappings:
            return name_mappings[name]

        # Otherwise, look for keywords
        for scale, keywords in scale_keywords.items():
            for keyword in keywords:
                if keyword in combined_text:
                    return scale

        return "unknown"

    def _build_dependency_graph(self, all_equations):
        """
        Build a dependency graph showing how equations at different scales interact.

        Args:
            all_equations: Dictionary of all equations

        Returns:
            Dictionary containing dependency graph information
        """
        # Initialize the graph
        G = nx.DiGraph()

        # Add nodes (equations) to the graph
        for name, info in all_equations.items():
            scale = info["scale_level"]
            G.add_node(name, scale=scale, module=info["module"])

        # Add edges based on parameter sharing and explicit references
        for name1, info1 in all_equations.items():
            for name2, info2 in all_equations.items():
                if name1 != name2:
                    # Check for parameter sharing
                    shared_params = self._find_shared_parameters(info1, info2)

                    # Check for explicit references in source code
                    if name2 in info1["source"]:
                        G.add_edge(name1, name2, type="explicit_reference", weight=2.0)
                    # Add edge for parameter sharing (weaker connection)
                    elif shared_params:
                        G.add_edge(name1, name2, type="parameter_sharing",
                                   shared_params=shared_params, weight=1.0)

        # Identify cross-scale connections
        cross_scale_edges = []
        for u, v, data in G.edges(data=True):
            scale_u = G.nodes[u]["scale"]
            scale_v = G.nodes[v]["scale"]
            if scale_u != scale_v and scale_u != "unknown" and scale_v != "unknown":
                cross_scale_edges.append((u, v, scale_u, scale_v))

        # Calculate metrics
        total_edges = G.number_of_edges()
        cross_scale_count = len(cross_scale_edges)
        cross_scale_ratio = cross_scale_count / max(1, total_edges)

        # Build adjacency matrix between scales
        scale_adjacency = np.zeros((len(self.scale_levels), len(self.scale_levels)))
        for u, v, scale_u, scale_v in cross_scale_edges:
            if scale_u in self.scale_levels and scale_v in self.scale_levels:
                i = self.scale_levels.index(scale_u)
                j = self.scale_levels.index(scale_v)
                scale_adjacency[i, j] += 1

        return {
            "graph": G,
            "cross_scale_edges": cross_scale_edges,
            "cross_scale_count": cross_scale_count,
            "cross_scale_ratio": cross_scale_ratio,
            "scale_adjacency": scale_adjacency
        }

    def _find_shared_parameters(self, info1, info2):
        """
        Find shared parameters between two equations.

        Args:
            info1: Information about first equation
            info2: Information about second equation

        Returns:
            List of shared parameter names
        """
        shared = []
        param1 = info1["parameters"]
        param2 = info2["parameters"]

        # Look for exact parameter name matches
        for p1 in param1:
            if p1 in param2:
                shared.append(p1)

        # Look for parameter mappings
        scale1 = info1["scale_level"]
        scale2 = info2["scale_level"]

        if scale1 in self.parameter_mappings and scale2 in self.parameter_mappings[scale1]:
            mappings = self.parameter_mappings[scale1][scale2]
            for p1, p2 in mappings.items():
                if p1 in param1 and p2 in param2:
                    shared.append(f"{p1} → {p2}")

        return shared

    def _evaluate_scale_transitions(self, all_equations, levels):
        """
        Evaluate the quality of transitions between adjacent scale levels.

        Args:
            all_equations: Dictionary of all equations
            levels: List of scale levels to evaluate

        Returns:
            Dictionary containing transition quality metrics
        """
        transitions = {}

        # Evaluate transitions between adjacent levels
        for i in range(len(levels) - 1):
            level1 = levels[i]
            level2 = levels[i + 1]

            # Find equations that bridge these levels
            bridge_equations = []
            for name, info in all_equations.items():
                # Check if equation explicitly mentions both levels
                combined_text = f"{info['docstring']} {info['source']}".lower()
                if (level1.lower() in combined_text and level2.lower() in combined_text):
                    bridge_equations.append(name)

            # Evaluate quality based on number of bridges and their completeness
            quality = 0.0
            if bridge_equations:
                # More bridge equations = better quality
                quality += min(1.0, len(bridge_equations) / 3.0) * 0.6

                # Check parameter mappings between levels
                expected_mappings = self.parameter_mappings.get(level1, {}).get(level2, {})
                if expected_mappings:
                    coverage = 0.0
                    for name in bridge_equations:
                        param_coverage = self._check_parameter_mapping_coverage(
                            all_equations[name], expected_mappings)
                        coverage = max(coverage, param_coverage)

                    quality += coverage * 0.4

            transitions[(level1, level2)] = {
                "bridge_equations": bridge_equations,
                "quality": quality
            }

        # Calculate overall transition quality
        avg_quality = sum(t["quality"] for t in transitions.values()) / max(1, len(transitions))

        return {
            "transitions": transitions,
            "average_quality": avg_quality
        }

    def _check_parameter_mapping_coverage(self, equation_info, expected_mappings):
        """
        Check how well an equation covers expected parameter mappings.

        Args:
            equation_info: Information about the equation
            expected_mappings: Dictionary of expected parameter mappings

        Returns:
            Float between 0 and 1 indicating mapping coverage
        """
        parameters = equation_info["parameters"]
        source = equation_info["source"].lower()
        docstring = equation_info["docstring"].lower()
        combined = source + " " + docstring

        covered = 0
        for param1, param2 in expected_mappings.items():
            # Check if both parameters are mentioned
            if (param1.lower() in combined and param2.lower() in combined):
                covered += 1

        return covered / max(1, len(expected_mappings))

    def _evaluate_transition_quality(self, all_equations, key_transitions):
        """
        Evaluate the quality of key transitions between specific equations.

        Args:
            all_equations: Dictionary of all equations
            key_transitions: List of key transitions to evaluate

        Returns:
            Dictionary containing transition quality metrics
        """
        quality = {}

        for eq1_name, eq2_name in key_transitions:
            # Skip if either equation is missing
            if eq1_name not in all_equations or eq2_name not in all_equations:
                quality[(eq1_name, eq2_name)] = 0.0
                continue

            eq1 = all_equations[eq1_name]
            eq2 = all_equations[eq2_name]

            # Skip if scales are the same (not a cross-scale transition)
            if eq1["scale_level"] == eq2["scale_level"]:
                quality[(eq1_name, eq2_name)] = 1.0
                continue

            # Evaluate transition quality
            transition_score = 0.0

            # 1. Check for explicit references
            if eq2_name in eq1["source"] or eq1_name in eq2["source"]:
                transition_score += 0.4

            # 2. Check for parameter mappings
            shared_params = self._find_shared_parameters(eq1, eq2)
            if shared_params:
                transition_score += 0.3 * min(1.0, len(shared_params) / 3.0)

            # 3. Check for conceptual integration in docstrings
            eq1_scale = eq1["scale_level"]
            eq2_scale = eq2["scale_level"]
            if eq1_scale in eq2["docstring"] or eq2_scale in eq1["docstring"]:
                transition_score += 0.3

            quality[(eq1_name, eq2_name)] = transition_score

        return quality

    def _simulate_signal_propagation(self, dependency_graph, levels):
        """
        Simulate how signals propagate across scales in the dependency graph.

        Args:
            dependency_graph: Dictionary containing dependency graph
            levels: List of scale levels

        Returns:
            Dictionary containing signal propagation metrics
        """
        G = dependency_graph["graph"]

        propagation_metrics = {}

        # For each scale, propagate a signal and see how it reaches other scales
        for source_scale in levels:
            # Find all nodes at this scale
            source_nodes = [n for n, data in G.nodes(data=True)
                            if data.get("scale") == source_scale]

            if not source_nodes:
                continue

            # Initialize signal strength at each node
            signal = {n: 0.0 for n in G.nodes()}
            for n in source_nodes:
                signal[n] = 1.0

            # Propagate for a few steps
            steps = 3
            for _ in range(steps):
                new_signal = signal.copy()
                for n in G.nodes():
                    # Get signal from incoming edges
                    incoming = 0.0
                    for pred in G.predecessors(n):
                        weight = G.edges[pred, n].get("weight", 1.0)
                        incoming += signal[pred] * weight

                    # Update node signal (with some decay)
                    if incoming > 0:
                        new_signal[n] = max(signal[n], 0.8 * incoming)

                signal = new_signal

            # Measure signal strength at each scale
            scale_signal = {scale: 0.0 for scale in levels}
            for n, strength in signal.items():
                node_scale = G.nodes[n].get("scale")
                if node_scale in scale_signal:
                    scale_signal[node_scale] = max(scale_signal[node_scale], strength)

            propagation_metrics[source_scale] = {
                "signal_by_scale": scale_signal,
                "average_signal": sum(scale_signal.values()) / len(scale_signal),
                "max_signal": max(scale_signal.values())
            }

        # Calculate propagation efficiency
        total_efficiency = 0.0
        for metrics in propagation_metrics.values():
            total_efficiency += metrics["average_signal"]

        average_efficiency = total_efficiency / max(1, len(propagation_metrics))

        return {
            "by_source_scale": propagation_metrics,
            "average_efficiency": average_efficiency
        }

    def _calculate_overall_score(self, scale_transitions, transition_quality):
        """
        Calculate an overall integration score.

        Args:
            scale_transitions: Dictionary of scale transition metrics
            transition_quality: Dictionary of transition quality metrics

        Returns:
            Float between 0 and 1 indicating overall integration quality
        """
        # Average scale transition quality (40%)
        scale_score = scale_transitions["average_quality"]

        # Average key transition quality (60%)
        key_score = sum(transition_quality.values()) / max(1, len(transition_quality))

        return 0.4 * scale_score + 0.6 * key_score

    def visualize_dependency_graph(self, dependency_graph, output_file=None):
        """
        Visualize the equation dependency graph.

        Args:
            dependency_graph: Dictionary containing dependency graph
            output_file: Path to save the visualization

        Returns:
            matplotlib figure
        """
        G = dependency_graph["graph"]

        # Define node colors by scale
        scale_colors = {
            "quantum": "purple",
            "agent": "blue",
            "group": "green",
            "civilization": "orange",
            "multi_civilization": "red",
            "cosmic": "brown",
            "unknown": "gray"
        }

        node_colors = [scale_colors.get(G.nodes[n].get("scale"), "gray") for n in G.nodes()]

        # Define edge colors by type
        edge_colors = []
        for u, v, data in G.edges(data=True):
            if data.get("type") == "explicit_reference":
                edge_colors.append("black")
            else:
                edge_colors.append("lightgray")

        # Create figure
        plt.figure(figsize=(12, 10))

        # Use spring layout for better visualization
        pos = nx.spring_layout(G, seed=42)

        # Draw graph
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, alpha=0.8)
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=1.0, alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=8)

        # Add legend
        handles = []
        labels = []
        for scale, color in scale_colors.items():
            if scale != "unknown":
                handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=color, markersize=10))
                labels.append(scale)

        plt.legend(handles, labels, loc='upper right', title="Scale Levels")

        # Add title
        plt.title("Equation Dependency Graph Across Scales")

        # Remove axis
        plt.axis('off')

        # Save figure if output file provided
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Dependency graph saved to {output_file}")

        return plt.gcf()

    def visualize_scale_adjacency(self, dependency_graph, output_file=None):
        """
        Visualize the adjacency matrix between scales.

        Args:
            dependency_graph: Dictionary containing dependency graph
            output_file: Path to save the visualization

        Returns:
            matplotlib figure
        """
        # Get scale adjacency matrix
        scale_adjacency = dependency_graph["scale_adjacency"]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot heatmap
        im = ax.imshow(scale_adjacency, cmap='Blues')

        # Add labels
        ax.set_xticks(np.arange(len(self.scale_levels)))
        ax.set_yticks(np.arange(len(self.scale_levels)))
        ax.set_xticklabels(self.scale_levels)
        ax.set_yticklabels(self.scale_levels)

        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Number of Connections")

        # Add annotations
        for i in range(len(self.scale_levels)):
            for j in range(len(self.scale_levels)):
                text = ax.text(j, i, int(scale_adjacency[i, j]),
                               ha="center", va="center", color="black" if scale_adjacency[i, j] < 3 else "white")

        # Add title
        ax.set_title("Cross-Scale Connections Between Equation Levels")

        # Add axis labels
        ax.set_xlabel("To Scale")
        ax.set_ylabel("From Scale")

        # Adjust layout
        plt.tight_layout()

        # Save figure if output file provided
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Scale adjacency matrix saved to {output_file}")

        return fig

    def visualize_signal_propagation(self, signal_propagation, output_file=None):
        """
        Visualize signal propagation across scales.

        Args:
            signal_propagation: Dictionary containing signal propagation metrics
            output_file: Path to save the visualization

        Returns:
            matplotlib figure
        """
        by_source = signal_propagation["by_source_scale"]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Initialize matrix for heatmap
        levels = sorted(list(by_source.keys()))
        matrix = np.zeros((len(levels), len(levels)))

        # Fill matrix
        for i, source in enumerate(levels):
            signals = by_source[source]["signal_by_scale"]
            for j, target in enumerate(levels):
                matrix[i, j] = signals.get(target, 0.0)

        # Plot heatmap
        im = ax.imshow(matrix, cmap='YlOrRd', vmin=0.0, vmax=1.0)

        # Add labels
        ax.set_xticks(np.arange(len(levels)))
        ax.set_yticks(np.arange(len(levels)))
        ax.set_xticklabels(levels)
        ax.set_yticklabels(levels)

        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Signal Strength")

        # Add annotations
        for i in range(len(levels)):
            for j in range(len(levels)):
                text = ax.text(j, i, f"{matrix[i, j]:.2f}",
                               ha="center", va="center", color="black" if matrix[i, j] < 0.5 else "white")

        # Add title
        ax.set_title("Signal Propagation Across Scales")

        # Add axis labels
        ax.set_xlabel("Target Scale")
        ax.set_ylabel("Source Scale")

        # Adjust layout
        plt.tight_layout()

        # Save figure if output file provided
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Signal propagation visualization saved to {output_file}")

        return fig


# Example usage
if __name__ == "__main__":
    validator = CrossScaleValidator()

    # Evaluate cross-scale interactions
    results = validator.evaluate_cross_scale_interactions()

    # Visualize dependency graph
    validator.visualize_dependency_graph(results["dependency_graph"],
                                         "validation/reports/unified/dependency_graph.png")

    # Visualize scale adjacency matrix
    validator.visualize_scale_adjacency(results["dependency_graph"],
                                        "validation/reports/unified/scale_adjacency.png")

    # Visualize signal propagation
    validator.visualize_signal_propagation(results["signal_propagation"],
                                           "validation/reports/unified/signal_propagation.png")

    # Print summary
    print(f"Overall integration score: {results['overall_integration_score']:.2f}")
    print(f"Cross-scale connections: {results['dependency_graph']['cross_scale_count']}")
    print(f"Average scale transition quality: {results['scale_transitions']['average_quality']:.2f}")
    print(f"Signal propagation efficiency: {results['signal_propagation']['average_efficiency']:.2f}")