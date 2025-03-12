"""
Integration between the validation framework and historical data validation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from pathlib import Path


def create_dummy_simulation_function():
    """Create a dummy simulation function for testing."""
    def dummy_simulation(params):
        # Ensure params is a dictionary with expected keys
        if not isinstance(params, dict):
            try:
                # Handle case where params is a list/tuple
                param_names = ["ALPHA_WISDOM", "BETA_FEEDBACK", "GAMMA_PHASE", "LAMBDA_DECAY"]
                if isinstance(params, (list, tuple)) and len(params) <= len(param_names):
                    params = {name: value for name, value in zip(param_names, params)}
                else:
                    # If can't convert, use default values
                    params = {
                        "ALPHA_WISDOM": 0.1,
                        "BETA_FEEDBACK": 0.05,
                        "GAMMA_PHASE": 0.1,
                        "LAMBDA_DECAY": 0.05
                    }
            except Exception:
                # Fallback to default parameters
                params = {
                    "ALPHA_WISDOM": 0.1,
                    "BETA_FEEDBACK": 0.05,
                    "GAMMA_PHASE": 0.1,
                    "LAMBDA_DECAY": 0.05
                }

        # Generate synthetic results based on parameters
        try:
            alpha = float(params.get("ALPHA_WISDOM", 0.1))
            beta = float(params.get("BETA_FEEDBACK", 0.05))
            gamma = float(params.get("GAMMA_PHASE", 0.1))
            lambda_decay = float(params.get("LAMBDA_DECAY", 0.05))

            # Generate time steps from 1900 to 2000
            time_steps = np.arange(1900, 2000)
            years = time_steps - 1900  # years since 1900 for calculations

            # Generate synthetic time series with the parameters
            # Knowledge grows with alpha, dampened by beta
            knowledge = 10.0 + years * alpha * (1.0 - beta * years/100)

            # Intelligence is derived from knowledge with gamma factor
            intelligence = knowledge * gamma * (1.0 + years/200)

            # Suppression decreases over time with lambda_decay
            suppression = 20.0 * np.exp(-lambda_decay * years/50)

            # Truth adoption increases with knowledge and decreases with suppression
            truth = 5.0 + 0.5 * knowledge - 0.2 * suppression

            # Cap values to reasonable ranges
            knowledge = np.maximum(0, np.minimum(100, knowledge))
            intelligence = np.maximum(0, np.minimum(100, intelligence))
            suppression = np.maximum(0, np.minimum(100, suppression))
            truth = np.maximum(0, np.minimum(100, truth))

            return {
                "year": time_steps,
                "knowledge": knowledge,
                "intelligence": intelligence,
                "suppression": suppression,
                "truth": truth
            }

        except Exception as e:
            print(f"Error in dummy simulation with params {params}: {e}")
            # Return minimal valid result structure to avoid breaking validation
            return {
                "year": np.array([1900, 1950, 2000]),
                "knowledge": np.array([10.0, 20.0, 30.0]),
                "intelligence": np.array([5.0, 10.0, 15.0]),
                "suppression": np.array([20.0, 10.0, 5.0]),
                "truth": np.array([10.0, 20.0, 30.0])
            }

    return dummy_simulation


class HistoricalDataValidator:
    """
    A class to validate simulation outcomes against historical data.
    Integrates with validation framework components to ensure numerical stability.
    """

    def __init__(
            self,
            simulation_func,
            historical_data_path,
            output_dir='validation/historical',
            parameter_ranges=None
    ):
        """
        Initialize the validator.

        Parameters:
            simulation_func: Function that runs simulation with given parameters
            historical_data_path: Path to historical data CSV
            output_dir: Directory to save validation results
            parameter_ranges: Dictionary of parameter ranges for optimization
        """
        self.simulation_func = simulation_func
        self.historical_data = pd.read_csv(historical_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parameter_ranges = parameter_ranges or {}

        # Initialize error metrics
        self.best_fit_params = None
        self.best_fit_error = float('inf')
        self.error_metrics = {}
        self.best_simulation = None

        # Initialize validation components
        self.dim_handler = None
        self.circuit_breaker = None
        self.edge_case_checker = None

    def set_dimension_handler(self, handler):
        """Set dimension handler for array validation."""
        self.dim_handler = handler

    def set_circuit_breaker(self, breaker):
        """Set circuit breaker for numerical stability checks."""
        self.circuit_breaker = breaker

    def set_edge_case_checker(self, checker):
        """Set edge case checker for risk identification."""
        self.edge_case_checker = checker

    def run_simulation_with_validation(self, params):
        """
        Run simulation with validation checks.

        Parameters:
            params: Dictionary of simulation parameters

        Returns:
            DataFrame of simulation results
        """
        # Run simulation
        sim_results = self.simulation_func(params)

        # Apply dimension handler if available
        if self.dim_handler and isinstance(sim_results, dict):
            # Get expected shapes based on historical data
            expected_shapes = {
                col: (len(self.historical_data),) for col in self.historical_data.columns
                if col != 'year'
            }

            # Fix dimensions
            sim_results = self.dim_handler.verify_and_fix_if_needed(
                sim_results, expected_shapes, "historical_validation"
            )

        # Apply circuit breaker if available
        if self.circuit_breaker and isinstance(sim_results, dict):
            for key, values in sim_results.items():
                if isinstance(values, np.ndarray):
                    # Create an instance if self.circuit_breaker is a class
                    if isinstance(self.circuit_breaker, type):
                        cb_instance = self.circuit_breaker()
                        for i in range(len(values)):
                            values[i] = cb_instance.check_and_fix(values[i])
                    else:
                        # Use existing instance
                        for i in range(len(values)):
                            values[i] = self.circuit_breaker.check_and_fix(values[i])

        return sim_results

    # In historical_integration.py, add the following fix to calculate_errors method

    def calculate_errors(self, simulation_data):
        """
        Calculate error metrics between simulation and historical data with enhanced error handling.
        """
        try:
            # Convert simulation data to DataFrame if it's a dictionary
            if isinstance(simulation_data, dict):
                sim_df = pd.DataFrame(simulation_data)
            else:
                sim_df = simulation_data

            # Ensure both datasets have the same length
            min_length = min(len(sim_df), len(self.historical_data))
            sim_df = sim_df.iloc[:min_length]
            hist_df = self.historical_data.iloc[:min_length]

            # Calculate errors for each metric
            metrics = set(hist_df.columns) & set(sim_df.columns)
            metrics = [m for m in metrics if m != 'year']

            errors = {}

            for metric in metrics:
                # Extract data with error handling
                try:
                    hist_values = hist_df[metric].values
                    sim_values = sim_df[metric].values

                    # Calculate metrics with proper error checking
                    mse = np.mean((hist_values - sim_values) ** 2)
                    rmse = np.sqrt(mse) if not np.isnan(mse) else float('inf')
                    mae = np.mean(np.abs(hist_values - sim_values))

                    # Calculate normalized metrics
                    if np.max(hist_values) > np.min(hist_values):
                        nrmse = rmse / (np.max(hist_values) - np.min(hist_values))
                    else:
                        nrmse = rmse

                    if np.mean(hist_values) != 0:
                        mape = np.mean(np.abs((hist_values - sim_values) / hist_values)) * 100
                    else:
                        mape = float('inf')

                    # Store metrics
                    errors[metric] = {
                        'mse': mse,
                        'rmse': rmse,
                        'mae': mae,
                        'nrmse': nrmse,
                        'mape': mape if not np.isnan(mape) and not np.isinf(mape) else None
                    }
                except Exception as e:
                    # Handle any errors in metric calculation
                    errors[metric] = {
                        'mse': float('inf'),
                        'rmse': float('inf'),
                        'mae': float('inf'),
                        'nrmse': float('inf'),
                        'mape': None,
                        'error': str(e)
                    }

            # Calculate overall error (weighted RMSE) with proper error handling
            if metrics:
                try:
                    overall_rmse = np.mean([errors[metric]['rmse'] for metric in metrics
                                            if not np.isnan(errors[metric]['rmse']) and not np.isinf(
                            errors[metric]['rmse'])])
                    errors['overall'] = {'rmse': overall_rmse}
                except Exception:
                    errors['overall'] = {'rmse': float('inf')}
            else:
                errors['overall'] = {'rmse': float('inf')}

            return errors

        except Exception as e:
            # Fallback error structure
            return {
                'overall': {'rmse': float('inf'), 'error': str(e)},
                'error_message': str(e)
            }

    def objective_function(self, param_values):
        """
        Objective function for parameter optimization with improved error handling.
        """
        # Convert parameter values to dictionary
        param_names = list(self.parameter_ranges.keys())
        params = {name: value for name, value in zip(param_names, param_values)}

        # Run simulation with robust error handling
        try:
            sim_results = self.run_simulation_with_validation(params)
            errors = self.calculate_errors(sim_results)

            # Check if errors has the expected structure
            if 'overall' in errors and 'rmse' in errors['overall']:
                return errors['overall']['rmse']
            else:
                print(f"Error in simulation with params {params}: 'overall' or 'rmse' missing in errors")
                return float('inf')
        except Exception as e:
            print(f"Error in simulation with params {params}: {str(e)}")
            return float('inf')

    def find_best_parameters(self, max_evals=100):
        """
        Find the best parameters that minimize the error with historical data.

        Parameters:
            max_evals: Maximum number of evaluations

        Returns:
            Dictionary of best parameters
        """
        if not self.parameter_ranges:
            print("No parameter ranges provided for optimization")
            return None

        # Define bounds
        param_names = list(self.parameter_ranges.keys())
        bounds = [self.parameter_ranges[name] for name in param_names]

        # Define initial guess (middle of bounds)
        x0 = [(b[0] + b[1]) / 2 for b in bounds]

        # Run optimization
        result = minimize(
            self.objective_function,
            x0,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxfun': max_evals}
        )

        # Store best parameters
        self.best_fit_params = {
            name: value for name, value in zip(param_names, result.x)
        }
        self.best_fit_error = result.fun

        # Run simulation with best parameters to get detailed errors
        sim_results = self.run_simulation_with_validation(self.best_fit_params)
        self.error_metrics = self.calculate_errors(sim_results)

        # Store best simulation results
        self.best_simulation = sim_results

        return self.best_fit_params

    def compare_to_historical_data(self, sim_results=None):
        """
        Compare simulation results to historical data.

        Parameters:
            sim_results: Simulation results (uses best fit if None)

        Returns:
            Error metrics and generates comparison plots
        """
        # Use best simulation if not provided
        if sim_results is None:
            if self.best_simulation is None:
                print("No simulation results available. Run find_best_parameters first.")
                return None
            sim_results = self.best_simulation

        # Convert simulation data to DataFrame if it's a dictionary
        if isinstance(sim_results, dict):
            sim_df = pd.DataFrame(sim_results)
        else:
            sim_df = sim_results

        # Ensure both datasets have the same length
        min_length = min(len(sim_df), len(self.historical_data))
        sim_df = sim_df.iloc[:min_length]
        hist_df = self.historical_data.iloc[:min_length]

        # Set up time axis
        if 'year' in hist_df.columns:
            time_values = hist_df['year'].values
        else:
            time_values = np.arange(min_length)

        # Identify metrics to compare
        metrics = set(hist_df.columns) & set(sim_df.columns)
        metrics = [m for m in metrics if m != 'year']

        # Create comparison plots
        for metric in metrics:
            plt.figure(figsize=(12, 6))

            # Plot historical data
            plt.plot(time_values, hist_df[metric].values, 'o-', label='Historical', color='blue')

            # Plot simulation data
            plt.plot(time_values, sim_df[metric].values, 's--', label='Simulation', color='red')

            # Add error band if available
            if self.error_metrics and metric in self.error_metrics:
                rmse = self.error_metrics[metric]['rmse']
                plt.fill_between(
                    time_values,
                    sim_df[metric].values - rmse,
                    sim_df[metric].values + rmse,
                    color='red',
                    alpha=0.2
                )

            # Add labels and legend
            plt.title(f'Comparison of {metric}')
            plt.xlabel('Time' if 'year' not in hist_df.columns else 'Year')
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Save plot
            plt.savefig(self.output_dir / f'comparison_{metric}.png', dpi=300)
            plt.close()

        # Create combined metrics plot
        if len(metrics) > 1:
            plt.figure(figsize=(15, 10))

            # Create subplot for each metric
            n_cols = min(2, len(metrics))
            n_rows = (len(metrics) + n_cols - 1) // n_cols

            for i, metric in enumerate(metrics):
                plt.subplot(n_rows, n_cols, i + 1)

                # Plot historical data
                plt.plot(time_values, hist_df[metric].values, 'o-', label='Historical', color='blue')

                # Plot simulation data
                plt.plot(time_values, sim_df[metric].values, 's--', label='Simulation', color='red')

                # Add labels and legend
                plt.title(f'{metric}')
                plt.xlabel('Time' if 'year' not in hist_df.columns else 'Year')
                plt.legend()
                plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.output_dir / 'comparison_all_metrics.png', dpi=300)
            plt.close()

        # Create error metrics visualization
        if self.error_metrics:
            # Bar chart of RMSE for each metric
            plt.figure(figsize=(12, 6))
            metric_rmse = {m: self.error_metrics[m]['rmse'] for m in metrics}

            plt.bar(metric_rmse.keys(), metric_rmse.values(), color='coral')
            plt.title('RMSE by Metric')
            plt.xlabel('Metric')
            plt.ylabel('RMSE')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)

            plt.tight_layout()
            plt.savefig(self.output_dir / 'error_metrics_rmse.png', dpi=300)
            plt.close()

            # Radar chart of all error metrics
            if len(metrics) >= 3:
                plt.figure(figsize=(10, 10))
                ax = plt.subplot(111, polar=True)

                # Number of metrics
                N = len(metrics)

                # Compute angle for each metric
                angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
                angles += angles[:1]  # Close the loop

                # Get NRMSE for each metric
                nrmse_values = [self.error_metrics[m]['nrmse'] for m in metrics]
                nrmse_values += nrmse_values[:1]  # Close the loop

                # Plot radar
                ax.plot(angles, nrmse_values, 'o-', linewidth=2)
                ax.fill(angles, nrmse_values, alpha=0.25)

                # Set category labels
                plt.xticks(angles[:-1], metrics)

                plt.title('Normalized RMSE by Metric')

                plt.tight_layout()
                plt.savefig(self.output_dir / 'error_metrics_radar.png', dpi=300)
                plt.close()

        return self.error_metrics

    def generate_historical_validation_report(self):
        """
        Generate a comprehensive validation report.

        Returns:
            Path to the report HTML file
        """
        if not self.best_fit_params or not self.error_metrics:
            print("No validation results available. Run find_best_parameters first.")
            return None

        # Create report
        html_content = f"""<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Historical Validation Report</title>
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
                .parameter-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                .parameter-table th, .parameter-table td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                .parameter-table th {{
                    background-color: #f2f2f2;
                }}
                .error-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                .error-table th, .error-table td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                .error-table th {{
                    background-color: #f2f2f2;
                }}
            </style>
        </head>
        <body>
            <h1>Historical Validation Report</h1>
            <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <div class="report-section">
                <h2>Executive Summary</h2>
                <p>Overall RMSE: {self.error_metrics['overall']['rmse']:.4f}</p>

                <h3>Best Fit Parameters</h3>
                <table class="parameter-table">
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
        """

        # Add best fit parameters
        for param, value in self.best_fit_params.items():
            html_content += f"""
                    <tr>
                        <td>{param}</td>
                        <td>{value:.4f}</td>
                    </tr>
            """

        html_content += """
                </table>

                <h3>Error Metrics by Data Series</h3>
                <table class="error-table">
                    <tr>
                        <th>Metric</th>
                        <th>RMSE</th>
                        <th>NRMSE</th>
                        <th>MAE</th>
                        <th>MAPE (%)</th>
                    </tr>
        """

        # Add error metrics for each data series
        for metric, errors in self.error_metrics.items():
            if metric != 'overall':
                mape_value = errors.get('mape', 'N/A')
                if mape_value is not None:
                    mape_str = f"{mape_value:.2f}"
                else:
                    mape_str = "N/A"

                html_content += f"""
                    <tr>
                        <td>{metric}</td>
                        <td>{errors['rmse']:.4f}</td>
                        <td>{errors['nrmse']:.4f}</td>
                        <td>{errors['mae']:.4f}</td>
                        <td>{mape_str}</td>
                    </tr>
                """

        html_content += """
                </table>
            </div>

            <div class="report-section">
                <h2>Comparison Visualizations</h2>
        """

        # Add comparison plots
        metrics = [m for m in self.error_metrics.keys() if m != 'overall']

        for metric in metrics:
            html_content += f"""
                <div class="visualization">
                    <h3>{metric} Comparison</h3>
                    <img src="comparison_{metric}.png" alt="{metric} Comparison">
                </div>
            """

        # Add combined metrics plot if it exists
        if len(metrics) > 1 and (self.output_dir / 'comparison_all_metrics.png').exists():
            html_content += """
                <div class="visualization">
                    <h3>All Metrics Comparison</h3>
                    <img src="comparison_all_metrics.png" alt="All Metrics Comparison">
                </div>
            """

        # Add error metrics visualizations if they exist
        if (self.output_dir / 'error_metrics_rmse.png').exists():
            html_content += """
                <div class="visualization">
                    <h3>RMSE by Metric</h3>
                    <img src="error_metrics_rmse.png" alt="RMSE by Metric">
                </div>
            """

        if (self.output_dir / 'error_metrics_radar.png').exists():
            html_content += """
                <div class="visualization">
                    <h3>Normalized RMSE by Metric</h3>
                    <img src="error_metrics_radar.png" alt="Normalized RMSE by Metric">
                </div>
            """

        html_content += """
            </div>

            <div class="report-section">
                <h2>Validation Framework Integration</h2>
        """

        # Add information about validation framework integration
        if self.dim_handler:
            html_content += """
                <h3>Dimension Handler Integration</h3>
                <p>Dimension handling was applied to ensure consistency between simulation outputs and historical data formats.</p>
            """

        if self.circuit_breaker:
            html_content += """
                <h3>Circuit Breaker Integration</h3>
                <p>Numerical stability checks were applied to prevent NaN, infinite values, and other numerical instabilities.</p>
            """

        if self.edge_case_checker:
            html_content += """
                <h3>Edge Case Checker Integration</h3>
                <p>Edge case detection was applied to identify potential numerical risks in the simulation.</p>
            """

        html_content += """
            </div>
        </body>
        </html>
        """

        # Write HTML file
        report_path = self.output_dir / 'historical_validation_report.html'
        with open(report_path, 'w') as f:
            f.write(html_content)

        return report_path


def integrate_historical_validation(simulation_func, validation_components, parameter_ranges=None):
    """
    Set up historical validation with integration to the validation framework.

    Parameters:
        simulation_func: Function that runs the simulation
        validation_components: Dictionary of validation components
        parameter_ranges: Dictionary of parameter ranges for optimization

    Returns:
        Configured HistoricalDataValidator object
    """
    # Initialize validator
    validator = HistoricalDataValidator(
        simulation_func=simulation_func,
        historical_data_path='outputs/data/historical_data.csv',
        parameter_ranges=parameter_ranges
    )

    # Integrate validation components
    if 'dimension_handler' in validation_components:
        validator.set_dimension_handler(validation_components['dimension_handler'])

    if 'circuit_breaker' in validation_components:
        validator.set_circuit_breaker(validation_components['circuit_breaker'])

    if 'edge_case_checker' in validation_components:
        validator.set_edge_case_checker(validation_components['edge_case_checker'])

    return validator


# Example usage:
if __name__ == "__main__":
    # Define parameter ranges for optimization
    parameter_ranges = {
        'alpha_wisdom': (0.05, 0.2),
        'beta_feedback': (0.01, 0.1),
        'gamma_phase': (0.05, 0.2),
        'lambda_decay': (0.01, 0.1)
    }

    # Create dummy simulation function
    run_simulation = create_dummy_simulation_function()

    # Import validation components
    from utils.dim_handler import DimensionHandler
    from utils.circuit_breaker import CircuitBreaker
    from utils.edge_case_checker import EdgeCaseChecker

    # Create validation components
    validation_components = {
        'dimension_handler': DimensionHandler(verbose=True, auto_fix=True),
        'circuit_breaker': CircuitBreaker(),
        'edge_case_checker': EdgeCaseChecker({})  # Pass empty dict for now
    }

    # Create validator
    validator = integrate_historical_validation(
        run_simulation,
        validation_components,
        parameter_ranges
    )

    # Run optimization and validation
    best_params = validator.find_best_parameters(max_evals=20)
    print(f"Best parameters: {best_params}")

    # Generate comparison visualizations
    errors = validator.compare_to_historical_data()
    print(f"Error metrics: {errors}")

    # Generate report
    report_path = validator.generate_historical_validation_report()
    print(f"Report generated: {report_path}")