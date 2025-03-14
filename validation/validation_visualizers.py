
# validation/validation_visualizers.py
"""
Visualization utilities for validation results.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationVisualizer:
    """
    Creates visualizations for validation results.
    """

    def __init__(self):
        """Initialize the visualizer."""
        pass

    def plot_equation_coverage(self, coverage_data, output_path=None):
        """Plot equation coverage data."""
        try:
            plt.figure(figsize=(10, 6))
            plt.title("Equation Coverage")
            plt.savefig(output_path)
            plt.close()
            return True
        except Exception as e:
            logger.error(f"Error plotting equation coverage: {e}")
            return False

    def plot_cross_scale_interactions(self, cross_scale_data, output_path=None):
        """Plot cross-scale interactions."""
        try:
            plt.figure(figsize=(10, 6))
            plt.title("Cross-Scale Interactions")
            plt.savefig(output_path)
            plt.close()
            return True
        except Exception as e:
            logger.error(f"Error plotting cross-scale interactions: {e}")
            return False

    def plot_sensitivity_heatmap(self, sensitivity_data, output_path=None):
        """Plot sensitivity heatmap."""
        try:
            plt.figure(figsize=(10, 6))
            plt.title("Sensitivity Heatmap")
            plt.savefig(output_path)
            plt.close()
            return True
        except Exception as e:
            logger.error(f"Error plotting sensitivity heatmap: {e}")
            return False

    def plot_simulation_comparison(self, comparison_data, output_path=None):
        """Plot simulation comparison."""
        try:
            plt.figure(figsize=(10, 6))
            plt.title("Simulation Comparison")
            plt.savefig(output_path)
            plt.close()
            return True
        except Exception as e:
            logger.error(f"Error plotting simulation comparison: {e}")
            return False
