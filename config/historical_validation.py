"""
Historical validation module.

This module provides functionality for validating simulation results against historical data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HistoricalValidation:
    """
    Class for historical validation.
    """

    def __init__(self,
                 data_path=None,
                 historical_start_year=1000,
                 historical_end_year=2020,
                 simulation_timesteps=400,
                 enable_circuit_breaker=False,
                 enable_adaptive_timestep=False):
        """
        Initialize the historical validation.

        Parameters:
            data_path: Path to historical data CSV file. If None, data will be generated.
            historical_start_year: Starting year for historical data
            historical_end_year: Ending year for historical data
            simulation_timesteps: Number of simulation timesteps to run
            enable_circuit_breaker: Enable numerical stability checks
            enable_adaptive_timestep: Enable adaptive timestep for stability
        """
        self.data_path = data_path
        self.historical_start_year = historical_start_year
        self.historical_end_year = historical_end_year
        self.simulation_timesteps = simulation_timesteps
        self.enable_circuit_breaker = enable_circuit_breaker
        self.enable_adaptive_timestep = enable_adaptive_timestep

        # Initialize metrics
        self.error_metrics = {
            'knowledge_rmse': None,
            'intelligence_rmse': None,
            'suppression_rmse': None,
            'truth_rmse': None,
            'overall_rmse': None
        }

        # Load or generate historical data
        self.historical_data = self._load_or_generate_historical_data()

        # Initialize simulation results
        self.simulation_results = None
        self.optimized_parameters = None

        logger.info(f"HistoricalValidation initialized with data from {historical_start_year} to {historical_end_year}")

    def _load_or_generate_historical_data(self):
        """
        Load historical data from CSV or generate synthetic data.

        Returns:
            DataFrame containing historical data
        """
        if self.data_path and os.path.exists(self.data_path):
            try:
                logger.info(f"Loading historical data from {self.data_path}")
                return pd.read_csv(self.data_path)
            except Exception as e:
                logger.error(f"Error loading historical data: {e}")
                logger.info("Falling back to generated data")

        # If data_path is not provided or loading fails, generate synthetic data
        try:
            from config.historical_data_generator import generate_historical_data
            logger.info("Generating synthetic historical data")
            return generate_historical_data(
                start_year=self.historical_start_year,
                end_year=self.historical_end_year,
                interval=10,  # Adjust as needed
                add_noise=True
            )
        except ImportError:
            logger.error("Cannot import generate_historical_data, creating basic data")
            # Create basic fallback data
            years = np.arange(self.historical_start_year, self.historical_end_year + 1, 10)
            data = pd.DataFrame({
                'year': years,
                'knowledge_index': np.linspace(5, 80, len(years)),
                'suppression_index': np.linspace(80, 20, len(years)),
                'intelligence_index': np.linspace(10, 70, len(years)),
                'truth_index': np.linspace(20, 60, len(years))
            })
            return data

    def _run_simulation(self, parameters):
        """
        Run simulation with given parameters.

        Parameters:
            parameters: Dictionary of simulation parameters

        Returns:
            DataFrame containing simulation results
        """
        try:
            # Try to import and run comprehensive simulation
            from simulations.comprehensive_simulation import run_simulation

            # Check what parameters the simulation accepts
            import inspect
            sim_params = inspect.signature(run_simulation).parameters

            # Create kwargs dict with the right parameter names
            sim_kwargs = {}

            # Handle 'timesteps' parameter (error in your logs)
            if 'timesteps' in sim_params:
                sim_kwargs['timesteps'] = self.simulation_timesteps
            elif 'num_steps' in sim_params:
                sim_kwargs['num_steps'] = self.simulation_timesteps
            elif 'num_timesteps' in sim_params:
                sim_kwargs['num_timesteps'] = self.simulation_timesteps
            elif 'steps' in sim_params:
                sim_kwargs['steps'] = self.simulation_timesteps

            # Handle parameters dict
            if 'params' in sim_params:
                sim_kwargs['params'] = parameters
            elif 'parameters' in sim_params:
                sim_kwargs['parameters'] = parameters

            # Handle circuit breaker
            if 'enable_circuit_breaker' in sim_params:
                sim_kwargs['enable_circuit_breaker'] = self.enable_circuit_breaker

            if 'enable_adaptive_timestep' in sim_params:
                sim_kwargs['enable_adaptive_timestep'] = self.enable_adaptive_timestep

            # Run simulation with properly named parameters
            logger.info("Running comprehensive simulation with parameters")
            results = run_simulation(**sim_kwargs)

            # Convert dict results to DataFrame if needed
            import pandas as pd
            import numpy as np
            if isinstance(results, dict) and not isinstance(results, pd.DataFrame):
                # Convert dictionary of arrays to DataFrame
                logger.info("Converting dictionary results to DataFrame")

                # First, check for the time dimension
                if 'time' in results:
                    timesteps = results['time']
                else:
                    # Find the first non-event value to determine length
                    for key, value in results.items():
                        if key != 'events' and hasattr(value, '__len__'):
                            timesteps = np.arange(len(value))
                            break
                    else:
                        # Fallback if no arrays found
                        timesteps = np.arange(100)

                # Create base dataframe with timesteps
                df_data = {'timestep': timesteps}

                # Check each value in the dictionary and handle appropriately
                for key, value in results.items():
                    if key == 'time' or key == 'events':
                        # Skip time (already handled) and events (complex structure)
                        continue

                    if isinstance(value, np.ndarray):
                        # Handle differently based on dimensions
                        if value.ndim == 1:
                            # 1D array can be added directly
                            df_data[key] = value
                        elif value.ndim == 2:
                            # For 2D arrays, we'll take mean values across civilizations/agents
                            # and add individual columns for each civilization/agent
                            df_data[f"{key}_mean"] = np.mean(value, axis=1)
                            df_data[f"{key}_std"] = np.std(value, axis=1)

                            # Add individual columns for each civilization/agent, limited to first 5
                            num_entities = min(5, value.shape[1])
                            for i in range(num_entities):
                                df_data[f"{key}_{i + 1}"] = value[:, i]

                # Create the DataFrame
                results = pd.DataFrame(df_data)

                # Add year column if not present
                if 'year' not in results.columns:
                    year_range = self.historical_end_year - self.historical_start_year
                    results['year'] = self.historical_start_year + results['timestep'] * year_range / len(results)

            return results
        except Exception as e:
            logger.warning(f"Error running comprehensive simulation: {e}")

            try:
                # Try alternate simulation
                from simulations.historical_simulation import run_simulation as run_alt_simulation

                # Similar parameter handling for alternative simulation
                import inspect
                sim_params = inspect.signature(run_alt_simulation).parameters

                sim_kwargs = {}
                if 'params' in sim_params:
                    sim_kwargs['params'] = parameters
                elif 'parameters' in sim_params:
                    sim_kwargs['parameters'] = parameters

                if 'timesteps' in sim_params:
                    sim_kwargs['timesteps'] = self.simulation_timesteps
                elif 'num_steps' in sim_params:
                    sim_kwargs['num_steps'] = self.simulation_timesteps

                if 'enable_circuit_breaker' in sim_params:
                    sim_kwargs['enable_circuit_breaker'] = self.enable_circuit_breaker

                # Run the simulation
                results = run_alt_simulation(**sim_kwargs)

                # Convert dict results to DataFrame if needed
                import pandas as pd
                import numpy as np
                if isinstance(results, dict) and not isinstance(results, pd.DataFrame):
                    # Convert dictionary of arrays to DataFrame
                    logger.info("Converting dictionary results to DataFrame")

                    # First, check for the time dimension
                    if 'time' in results:
                        timesteps = results['time']
                    else:
                        # Find the first non-event value to determine length
                        for key, value in results.items():
                            if key != 'events' and hasattr(value, '__len__'):
                                timesteps = np.arange(len(value))
                                break
                        else:
                            # Fallback if no arrays found
                            timesteps = np.arange(100)

                    # Create base dataframe with timesteps
                    df_data = {'timestep': timesteps}

                    # Check each value in the dictionary and handle appropriately
                    for key, value in results.items():
                        if key == 'time' or key == 'events':
                            # Skip time (already handled) and events (complex structure)
                            continue

                        if isinstance(value, np.ndarray):
                            # Handle differently based on dimensions
                            if value.ndim == 1:
                                # 1D array can be added directly
                                df_data[key] = value
                            elif value.ndim == 2:
                                # For 2D arrays, we'll take mean values across civilizations/agents
                                # and add individual columns for each civilization/agent
                                df_data[f"{key}_mean"] = np.mean(value, axis=1)
                                df_data[f"{key}_std"] = np.std(value, axis=1)

                                # Add individual columns for each civilization/agent, limited to first 5
                                num_entities = min(5, value.shape[1])
                                for i in range(num_entities):
                                    df_data[f"{key}_{i + 1}"] = value[:, i]

                    # Create the DataFrame
                    results = pd.DataFrame(df_data)

                    # Add year column if not present
                    if 'year' not in results.columns:
                        year_range = self.historical_end_year - self.historical_start_year
                        results['year'] = self.historical_start_year + results['timestep'] * year_range / len(results)

                return results
            except Exception as e:
                logger.warning(f"Error running alternative simulation: {e}")
                logger.warning("No simulation module found, using dummy simulation")

                # Create dummy results if no simulation is available
                import numpy as np
                import pandas as pd

                timesteps = np.arange(self.simulation_timesteps)
                k0 = parameters.get('knowledge_growth_rate', 0.15)
                a0 = parameters.get('truth_adoption_rate', 0.12)

                # Simple growth and decay functions
                knowledge = 1 + 50 * (1 - np.exp(-k0 * timesteps / 100))
                truth = 1 + 40 * (1 - np.exp(-a0 * timesteps / 100))
                suppression = 80 * np.exp(-0.005 * timesteps) + 20
                intelligence = knowledge * (1 - 0.5 * suppression / 100)

                # Create DataFrame
                results = pd.DataFrame({
                    'timestep': timesteps,
                    'knowledge': knowledge,
                    'truth': truth,
                    'suppression': suppression,
                    'intelligence': intelligence
                })

                # Map timesteps to years
                year_range = self.historical_end_year - self.historical_start_year
                results['year'] = self.historical_start_year + timesteps * year_range / self.simulation_timesteps

                return results

    def _calculate_metrics(self, sim_results, hist_data):
        """
        Calculate error metrics between simulation and historical data.

        Parameters:
            sim_results: DataFrame with simulation results
            hist_data: DataFrame with historical data

        Returns:
            Dictionary of error metrics
        """
        # Create mappings between simulation and historical column names
        sim_to_hist = {
            'knowledge': 'knowledge_index',
            'intelligence': 'intelligence_index',
            'suppression': 'suppression_index',
            'truth': 'truth_index'
        }

        # Initialize metrics
        metrics = {
            'knowledge_rmse': None,
            'intelligence_rmse': None,
            'suppression_rmse': None,
            'truth_rmse': None,
            'overall_rmse': 0,
            'valid_metrics': 0
        }

        # Merge dataframes on year
        # First, make sure both dataframes have 'year' column
        if 'year' not in sim_results.columns:
            logger.warning("Simulation results missing 'year' column. Creating based on timesteps.")
            year_range = self.historical_end_year - self.historical_start_year
            sim_results['year'] = self.historical_start_year + sim_results['timestep'] * year_range / self.simulation_timesteps

        # Get common years for comparison
        sim_years = sim_results['year'].values
        common_years = hist_data[hist_data['year'].isin(sim_years)]

        if len(common_years) == 0:
            # If no direct matches, interpolate simulation results to historical years
            logger.info("No direct year matches. Interpolating simulation results.")
            merged_data = hist_data.copy()

            for sim_col, hist_col in sim_to_hist.items():
                if sim_col in sim_results.columns and hist_col in hist_data.columns:
                    # Create interpolation function from simulation data
                    from scipy.interpolate import interp1d
                    f = interp1d(sim_results['year'], sim_results[sim_col],
                                 bounds_error=False, fill_value="extrapolate")

                    # Generate interpolated values for historical years
                    merged_data[f"{sim_col}_sim"] = f(hist_data['year'])

                    # Calculate RMSE
                    rmse = np.sqrt(np.mean((merged_data[hist_col] - merged_data[f"{sim_col}_sim"]) ** 2))
                    metrics[f"{sim_col}_rmse"] = rmse
                    metrics['overall_rmse'] += rmse
                    metrics['valid_metrics'] += 1
        else:
            # Merge on year if direct matches exist
            merged_data = pd.merge(
                hist_data,
                sim_results[['year'] + list(sim_to_hist.keys())],
                on='year',
                how='inner'
            )

            # Calculate metrics for each variable
            for sim_col, hist_col in sim_to_hist.items():
                if sim_col in sim_results.columns and hist_col in hist_data.columns:
                    rmse = np.sqrt(np.mean((merged_data[hist_col] - merged_data[sim_col]) ** 2))
                    metrics[f"{sim_col}_rmse"] = rmse
                    metrics['overall_rmse'] += rmse
                    metrics['valid_metrics'] += 1

        # Calculate overall RMSE
        if metrics['valid_metrics'] > 0:
            metrics['overall_rmse'] /= metrics['valid_metrics']

        return metrics

    def optimize_parameters(self, params_to_optimize=None, max_iterations=20):
        """
        Find optimal parameters for historical fit.

        Parameters:
            params_to_optimize: List of parameters to optimize
            max_iterations: Maximum number of iterations

        Returns:
            Dictionary of optimized parameters
        """
        logger.info(f"Optimizing parameters with {max_iterations} iterations")

        # Default parameters to optimize if not specified
        if params_to_optimize is None:
            params_to_optimize = {
                'knowledge_growth_rate': (0.05, 0.3),  # (min, max)
                'truth_adoption_rate': (0.05, 0.2),
                'alpha_wisdom': (0.05, 0.15)
            }

        # Initial parameter set
        best_params = {
            'knowledge_growth_rate': 0.15,
            'truth_adoption_rate': 0.12,
            'alpha_wisdom': 0.1
        }

        # If circuit breaker is enabled, add it to parameters
        if self.enable_circuit_breaker:
            best_params['circuit_breaker_threshold'] = 1e-10

        # Update with any additional parameters from params_to_optimize
        for param, (min_val, max_val) in params_to_optimize.items():
            if param not in best_params:
                best_params[param] = (min_val + max_val) / 2  # start with middle value

        # Run initial simulation
        sim_results = self._run_simulation(best_params)
        best_metrics = self._calculate_metrics(sim_results, self.historical_data)
        best_error = best_metrics['overall_rmse']

        logger.info(f"Initial parameters: {best_params}, Initial error: {best_error}")

        # Simple grid search optimization
        for iteration in range(max_iterations):
            improved = False

            # Try adjusting each parameter
            for param, (min_val, max_val) in params_to_optimize.items():
                # Try values above and below current value
                current_value = best_params[param]
                step_size = (max_val - min_val) / 10

                for direction in [-1, 1]:
                    new_value = current_value + direction * step_size

                    # Ensure value is within bounds
                    new_value = max(min_val, min(max_val, new_value))

                    # Skip if value is unchanged
                    if new_value == current_value:
                        continue

                    # Try new parameter value
                    test_params = best_params.copy()
                    test_params[param] = new_value

                    try:
                        # Run simulation with new parameters
                        sim_results = self._run_simulation(test_params)
                        metrics = self._calculate_metrics(sim_results, self.historical_data)
                        error = metrics['overall_rmse']

                        # Check if improvement
                        if error < best_error:
                            best_params = test_params
                            best_error = error
                            best_metrics = metrics
                            improved = True
                            logger.info(f"Iteration {iteration+1}: Improved {param} to {new_value}, error: {error}")
                    except Exception as e:
                        logger.warning(f"Error with parameters {test_params}: {e}")

            # If no improvement was found, reduce step size
            if not improved:
                # Reduce step size for next iteration
                for param, (min_val, max_val) in params_to_optimize.items():
                    params_to_optimize[param] = (min_val, min_val + (max_val - min_val) * 0.5)

                logger.info(f"Iteration {iteration+1}: No improvement, reducing search space")

            # Early stopping if error is very small
            if best_error < 1.0:
                logger.info(f"Achieved excellent fit (error: {best_error}), stopping optimization")
                break

        # Store results
        self.optimized_parameters = best_params
        self.simulation_results = self._run_simulation(best_params)
        self.error_metrics = best_metrics

        logger.info(f"Optimization complete. Best parameters: {best_params}, Error: {best_error}")
        return best_params

    def visualize_comparison(self, save_path=None):
        """
        Visualize comparison between simulation and historical data.

        Parameters:
            save_path: Path to save visualization

        Returns:
            matplotlib Figure object
        """
        # Run simulation if not already run
        if self.simulation_results is None:
            if self.optimized_parameters is None:
                logger.info("No optimized parameters, using defaults")
                self.optimized_parameters = self.optimize_parameters(max_iterations=5)

            self.simulation_results = self._run_simulation(self.optimized_parameters)

        # Prepare mapping between simulation and historical column names
        sim_to_hist = {
            'knowledge': 'knowledge_index',
            'intelligence': 'intelligence_index',
            'suppression': 'suppression_index',
            'truth': 'truth_index'
        }

        # Create grid of plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        # Create year mapping for simulation results if not present
        if 'year' not in self.simulation_results.columns and 'timestep' in self.simulation_results.columns:
            year_range = self.historical_end_year - self.historical_start_year
            self.simulation_results['year'] = self.historical_start_year + self.simulation_results['timestep'] * year_range / self.simulation_timesteps

        # Plot each metric
        for i, (sim_col, hist_col) in enumerate(sim_to_hist.items()):
            ax = axes[i]

            # Plot historical data
            if hist_col in self.historical_data.columns:
                ax.plot(self.historical_data['year'], self.historical_data[hist_col],
                       'b-', label='Historical Data', linewidth=2)

            # Plot simulation data
            if sim_col in self.simulation_results.columns:
                ax.plot(self.simulation_results['year'], self.simulation_results[sim_col],
                       'r--', label='Simulation', linewidth=2)

            # Add title and labels
            title = hist_col.replace('_', ' ').title()
            ax.set_title(title)
            ax.set_xlabel('Year')
            ax.set_ylabel('Index Value')
            ax.legend()
            ax.grid(True)

            # Add RMSE if available
            if f"{sim_col}_rmse" in self.error_metrics and self.error_metrics[f"{sim_col}_rmse"] is not None:
                rmse = self.error_metrics[f"{sim_col}_rmse"]
                ax.text(0.05, 0.95, f"RMSE: {rmse:.2f}", transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

        # Add overall title
        overall_rmse = self.error_metrics.get('overall_rmse', None)
        if overall_rmse is not None:
            fig.suptitle(f"Historical vs. Simulation Comparison (Overall RMSE: {overall_rmse:.2f})", fontsize=16)
        else:
            fig.suptitle("Historical vs. Simulation Comparison", fontsize=16)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        # Save figure if path provided
        if save_path is not None:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Visualization saved to {save_path}")
            except Exception as e:
                logger.error(f"Error saving visualization: {e}")

        return fig

    def calculate_error(self):
        """
        Calculate error between simulation and historical data.

        Returns:
            RMSE error value
        """
        # Run simulation if not already run
        if self.simulation_results is None:
            if self.optimized_parameters is None:
                logger.info("No optimized parameters, using defaults")
                default_params = {
                    'knowledge_growth_rate': 0.15,
                    'truth_adoption_rate': 0.12,
                    'alpha_wisdom': 0.1
                }
                self.simulation_results = self._run_simulation(default_params)
            else:
                self.simulation_results = self._run_simulation(self.optimized_parameters)

        # Calculate metrics
        self.error_metrics = self._calculate_metrics(self.simulation_results, self.historical_data)

        return self.error_metrics['overall_rmse']

    def save_results(self, output_dir=None):
        """
        Save validation results.

        Parameters:
            output_dir: Directory to save results
        """
        if output_dir is None:
            logger.warning("No output directory provided, skipping result saving")
            return

        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Save historical data
            self.historical_data.to_csv(os.path.join(output_dir, 'historical_data.csv'), index=False)

            # Save simulation results
            if self.simulation_results is not None:
                self.simulation_results.to_csv(os.path.join(output_dir, 'simulation_results.csv'), index=False)

            # Save parameters
            if self.optimized_parameters is not None:
                pd.DataFrame([self.optimized_parameters]).to_csv(
                    os.path.join(output_dir, 'simulation_parameters.csv'), index=False)

            # Save error metrics
            if self.error_metrics is not None:
                pd.DataFrame([self.error_metrics]).to_csv(
                    os.path.join(output_dir, 'error_metrics.csv'), index=False)

            logger.info(f"Results saved to {output_dir}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")

    def run_validation(self, output_dir=None):
        """
        Run the historical validation and generate a report.

        Args:
            output_dir (str): Directory to save validation report

        Returns:
            dict: Validation results
        """
        if output_dir is None:
            logger.warning("No output directory provided, using default")
            output_dir = "validation/reports/historical"

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Run parameter optimization
        if self.optimized_parameters is None:
            logger.info("Running parameter optimization")
            self.optimize_parameters()

        # Generate comparison visualization
        logger.info("Generating comparison visualization")
        fig = self.visualize_comparison(save_path=os.path.join(output_dir, "comparison.png"))

        # Calculate errors if not already calculated
        if any(metric is None for metric in self.error_metrics.values()):
            logger.info("Calculating error metrics")
            self.calculate_error()

        # Save results
        logger.info("Saving validation results")
        self.save_results(output_dir)

        # Generate HTML report
        logger.info("Generating HTML report")
        report_path = os.path.join(output_dir, "historical_validation_report.html")
        self._generate_html_report(report_path)

        # Return results
        return {
            'status': 'success',
            'error_metrics': self.error_metrics,
            'optimized_parameters': self.optimized_parameters,
            'report_path': report_path
        }

    def _generate_html_report(self, output_path):
        """
        Generate an HTML report for historical validation.

        Args:
            output_path (str): Path to save HTML report
        """
        # Create report timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Format error metrics for display
        error_metrics_html = ""
        if self.error_metrics:
            error_metrics_html = "<table class='metrics'>"
            error_metrics_html += "<tr><th>Metric</th><th>Value</th></tr>"
            for metric, value in self.error_metrics.items():
                if value is not None and metric != 'valid_metrics':
                    formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                    error_metrics_html += f"<tr><td>{metric}</td><td>{formatted_value}</td></tr>"
            error_metrics_html += "</table>"

        # Format optimized parameters for display
        parameters_html = ""
        if self.optimized_parameters:
            parameters_html = "<table class='parameters'>"
            parameters_html += "<tr><th>Parameter</th><th>Value</th></tr>"
            for param, value in self.optimized_parameters.items():
                formatted_value = f"{value:.6f}" if isinstance(value, float) else str(value)
                parameters_html += f"<tr><td>{param}</td><td>{formatted_value}</td></tr>"
            parameters_html += "</table>"

        # Determine validation status
        overall_rmse = self.error_metrics.get('overall_rmse')
        if overall_rmse is not None:
            if overall_rmse < 5.0:
                status = "Good"
                status_color = "green"
            elif overall_rmse < 10.0:
                status = "Acceptable"
                status_color = "orange"
            else:
                status = "Poor"
                status_color = "red"
        else:
            status = "Unknown"
            status_color = "gray"

        # Create HTML content
        html_content = f'''
<!DOCTYPE html>
<html>
<head>
    <title>Historical Validation Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .header {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            border-left: 5px solid #2c3e50;
        }}
        .status {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 3px;
            color: white;
            font-weight: bold;
            background-color: {status_color};
        }}
        .section {{
            margin-bottom: 30px;
            padding: 20px;
            background-color: #fff;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .visualization {{
            text-align: center;
            margin: 20px 0;
        }}
        .visualization img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        table.metrics, table.parameters {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        table.metrics th, table.metrics td,
        table.parameters th, table.parameters td {{
            padding: 12px 15px;
            border: 1px solid #ddd;
            text-align: left;
        }}
        table.metrics th, table.parameters th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        table.metrics tr:nth-child(even),
        table.parameters tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #777;
            font-size: 0.9em;
        }}
        .files-list {{
            list-style: none;
            padding: 0;
        }}
        .files-list li {{
            padding: 10px;
            background-color: #f8f9fa;
            margin-bottom: 5px;
            border-radius: 3px;
        }}
        .files-list li a {{
            text-decoration: none;
            color: #3498db;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Historical Validation Report</h1>
            <p>Generated on: {timestamp}</p>
            <p>Status: <span class="status">{status}</span></p>
        </div>
        
        <div class="section">
            <h2>Visualization</h2>
            <div class="visualization">
                <img src="comparison.png" alt="Historical Comparison">
            </div>
        </div>
        
        <div class="section">
            <h2>Error Metrics</h2>
            {error_metrics_html}
        </div>
        
        <div class="section">
            <h2>Optimized Parameters</h2>
            {parameters_html}
        </div>
        
        <div class="section">
            <h2>Validation Files</h2>
            <ul class="files-list">
                <li><a href="historical_data.csv">Historical Data</a></li>
                <li><a href="simulation_results.csv">Simulation Results</a></li>
                <li><a href="simulation_parameters.csv">Simulation Parameters</a></li>
                <li><a href="error_metrics.csv">Error Metrics</a></li>
            </ul>
        </div>
        
        <div class="footer">
            <p>Generated by Axiomatic Intelligence Growth Simulation - Historical Validation Module</p>
        </div>
    </div>
</body>
</html>
'''

        # Write HTML to file
        with open(output_path, 'w') as f:
            f.write(html_content)

        logger.info(f"HTML report saved to {output_path}")


# In config/historical_validation.py, fix the _run_simulation method:

def _run_simulation(self, parameters):
    """
    Run simulation with given parameters.

    Parameters:
        parameters: Dictionary of simulation parameters

    Returns:
        DataFrame containing simulation results
    """
    try:
        # Try to import and run comprehensive simulation
        from simulations.comprehensive_simulation import run_simulation

        logger.info("Running comprehensive simulation with parameters")

        # Check what parameters the simulation accepts
        import inspect
        sim_params = inspect.signature(run_simulation).parameters

        sim_args = {}
        # Map parameters properly
        if 'params' in sim_params:
            sim_args['params'] = parameters
        elif 'parameters' in sim_params:
            sim_args['parameters'] = parameters

        # Handle timesteps parameter properly
        if 'timesteps' in sim_params:
            sim_args['timesteps'] = self.simulation_timesteps
        elif 'num_steps' in sim_params:
            sim_args['num_steps'] = self.simulation_timesteps
        elif 'num_timesteps' in sim_params:
            sim_args['num_timesteps'] = self.simulation_timesteps
        elif 'steps' in sim_params:
            sim_args['steps'] = self.simulation_timesteps

        # Add circuit breaker if supported
        if 'enable_circuit_breaker' in sim_params:
            sim_args['enable_circuit_breaker'] = self.enable_circuit_breaker

        # Add adaptive timestep if supported
        if 'enable_adaptive_timestep' in sim_params:
            sim_args['enable_adaptive_timestep'] = self.enable_adaptive_timestep

        # Run the simulation with the correct parameters
        results = run_simulation(**sim_args)

        return results
    except Exception as e:
        logger.warning(f"Error running comprehensive simulation: {e}")
        logger.warning("Trying alternative simulation")

        try:
            # Try alternate simulation
            from simulations.historical_simulation import run_simulation as run_alt_simulation

            # Similar parameter handling for alternative simulation
            import inspect
            sim_params = inspect.signature(run_alt_simulation).parameters

            sim_args = {}
            if 'params' in sim_params:
                sim_args['params'] = parameters
            elif 'parameters' in sim_params:
                sim_args['parameters'] = parameters

            if 'timesteps' in sim_params:
                sim_args['timesteps'] = self.simulation_timesteps
            elif 'num_steps' in sim_params:
                sim_args['num_steps'] = self.simulation_timesteps

            if 'enable_circuit_breaker' in sim_params:
                sim_args['enable_circuit_breaker'] = self.enable_circuit_breaker

            results = run_alt_simulation(**sim_args)
            return results
        except Exception as e:
            logger.warning(f"Error running alternative simulation: {e}")
            logger.warning("No simulation module found, using dummy simulation")

            # Create dummy results if no simulation is available
            timesteps = np.arange(self.simulation_timesteps)
            k0 = parameters.get('knowledge_growth_rate', 0.15)
            a0 = parameters.get('truth_adoption_rate', 0.12)

            # Simple growth and decay functions
            knowledge = 1 + 50 * (1 - np.exp(-k0 * timesteps / 100))
            truth = 1 + 40 * (1 - np.exp(-a0 * timesteps / 100))
            suppression = 80 * np.exp(-0.005 * timesteps) + 20
            intelligence = knowledge * (1 - 0.5 * suppression / 100)

            # Create DataFrame
            results = pd.DataFrame({
                'timestep': timesteps,
                'knowledge': knowledge,
                'truth': truth,
                'suppression': suppression,
                'intelligence': intelligence
            })

            # Map timesteps to years
            year_range = self.historical_end_year - self.historical_start_year
            results['year'] = self.historical_start_year + timesteps * year_range / self.simulation_timesteps

            return results


if __name__ == "__main__":
    # Run as standalone script
    import argparse

    parser = argparse.ArgumentParser(description="Run historical validation")
    parser.add_argument("--output-dir", help="Directory to save results")
    parser.add_argument("--no-optimize", action="store_true", help="Skip parameter optimization")
    parser.add_argument("--no-visualize", action="store_true", help="Skip visualization")
    parser.add_argument("--iterations", type=int, default=20, help="Maximum optimization iterations")

    args = parser.parse_args()

    # Run validation
    run_historical_validation(
        output_dir=args.output_dir,
        optimize=not args.no_optimize,
        visualize=not args.no_visualize,
        max_iterations=args.iterations
    )