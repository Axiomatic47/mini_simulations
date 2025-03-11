import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pathlib import Path
import sys

# Add parent directory to path to find modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import core simulation functions
from config.equations import (
    intelligence_growth, truth_adoption, wisdom_field,
    resistance_resurgence, suppression_feedback
)

# Import astrophysics extensions if they're available
try:
    from config.astrophysics_extensions import (
        civilization_lifecycle_phase, suppression_event_horizon
    )
except ImportError:
    print("Warning: Astrophysics extensions not found. Some functionality will be limited.")


class HistoricalValidation:
    """
    A class for validating the Axiomatic Intelligence Growth Simulation against historical data.
    """

    def __init__(self, data_source=None, start_year=1000, end_year=2020, interval=10):
        """
        Initialize the historical validation model.

        Parameters:
            data_source (str): Path to CSV file with historical data, or None to use synthetic data
            start_year (int): First year to include in validation
            end_year (int): Last year to include in validation
            interval (int): Year interval for data points
        """
        self.start_year = start_year
        self.end_year = end_year
        self.interval = interval
        self.years = np.arange(start_year, end_year + 1, interval)
        self.num_years = len(self.years)

        # Historical metrics to track
        self.metrics = [
            "knowledge_index",
            "suppression_index",
            "intelligence_index",
            "truth_index"
        ]

        # Load or generate historical data
        if data_source is not None:
            self.historical_data = self._load_historical_data(data_source)
        else:
            self.historical_data = self._generate_synthetic_data()

        # Initialize simulation parameters
        self.default_params = self._get_default_parameters()
        self.current_params = self.default_params.copy()

        # Store simulation results
        self.simulation_results = None

    def _load_historical_data(self, data_source):
        """Load historical data from CSV file."""
        try:
            data = pd.read_csv(data_source)
            # Check if all needed columns exist
            required_columns = ["year"] + self.metrics
            for col in required_columns:
                if col not in data.columns:
                    raise ValueError(f"Required column '{col}' not found in data")

            # Filter to relevant years and ensure data is sorted
            data = data[data["year"].isin(self.years)].sort_values("year")

            # Check if we have all the years
            if len(data) != self.num_years:
                # Interpolate missing years if necessary
                data = self._interpolate_missing_years(data)

            return data

        except Exception as e:
            print(f"Error loading historical data: {e}")
            print("Falling back to synthetic data generation.")
            return self._generate_synthetic_data()

    def _interpolate_missing_years(self, data):
        """Interpolate missing years in historical data."""
        # Create a complete dataframe with all years
        complete_df = pd.DataFrame({"year": self.years})

        # Merge with existing data
        merged = pd.merge(complete_df, data, on="year", how="left")

        # Interpolate missing values
        for metric in self.metrics:
            merged[metric] = merged[metric].interpolate(method='linear')

        return merged

    def _generate_synthetic_data(self):
        """Generate synthetic historical data based on known patterns."""
        data = pd.DataFrame({"year": self.years})

        # Generate historical knowledge index with key periods
        # (e.g., Renaissance, Enlightenment, Industrial Revolution, Information Age)
        knowledge = np.zeros(self.num_years)

        # Base exponential growth with periods of acceleration
        time_scale = (self.years - self.start_year) / (self.end_year - self.start_year)

        # Basic growth component (slow early growth, accelerating in modern era)
        knowledge = 10 * (np.exp(3 * time_scale) - 1) / (np.exp(3) - 1)

        # Add Renaissance effect (1400-1600)
        renaissance_mask = (self.years >= 1400) & (self.years <= 1600)
        knowledge[renaissance_mask] += 3 * np.sin(np.pi * (self.years[renaissance_mask] - 1400) / 200)

        # Add Enlightenment effect (1650-1800)
        enlightenment_mask = (self.years >= 1650) & (self.years <= 1800)
        knowledge[enlightenment_mask] += 5 * np.sin(np.pi * (self.years[enlightenment_mask] - 1650) / 150)

        # Add Industrial Revolution effect (1760-1900)
        industrial_mask = (self.years >= 1760) & (self.years <= 1900)
        knowledge[industrial_mask] += 7 * np.sin(np.pi * (self.years[industrial_mask] - 1760) / 140)

        # Add Information Age effect (1950-present)
        info_age_mask = (self.years >= 1950)
        knowledge[info_age_mask] += 15 * (1 - np.exp(-(self.years[info_age_mask] - 1950) / 30))

        # Add some noise
        knowledge += np.random.normal(0, 0.5, self.num_years)

        # Ensure knowledge is positive and normalized to 0-100 scale
        knowledge = np.maximum(0, knowledge)
        knowledge = 100 * knowledge / knowledge[-1]

        # Generate suppression index
        suppression = np.zeros(self.num_years)

        # Base suppression starts high and generally decreases
        suppression = 80 * np.exp(-2 * time_scale) + 20

        # Add Dark Ages effect (500-1300)
        dark_ages_mask = (self.years >= self.start_year) & (self.years <= 1300)
        if np.any(dark_ages_mask):
            suppression[dark_ages_mask] += 10 * np.exp(-(self.years[dark_ages_mask] - self.start_year) / 300)

        # Add religious persecution effects (1450-1750)
        persecution_mask = (self.years >= 1450) & (self.years <= 1750)
        if np.any(persecution_mask):
            suppression[persecution_mask] += 15 * np.sin(np.pi * (self.years[persecution_mask] - 1450) / 300)

        # Add World Wars effects
        ww1_mask = (self.years >= 1914) & (self.years <= 1918)
        ww2_mask = (self.years >= 1939) & (self.years <= 1945)
        cold_war_mask = (self.years >= 1947) & (self.years <= 1991)

        if np.any(ww1_mask):
            suppression[ww1_mask] += 20
        if np.any(ww2_mask):
            suppression[ww2_mask] += 25
        if np.any(cold_war_mask):
            suppression[cold_war_mask] += 15 * np.exp(-(self.years[cold_war_mask] - 1947) / 44)

        # Add some noise
        suppression += np.random.normal(0, 1.0, self.num_years)

        # Ensure suppression is in 0-100 range
        suppression = np.maximum(0, suppression)
        suppression = np.minimum(100, suppression)

        # Generate intelligence index (related to knowledge and inverse of suppression)
        intelligence = knowledge * (1 - 0.5 * suppression / 100)
        intelligence = 100 * intelligence / intelligence[-1]  # Normalize to 0-100

        # Generate truth index (related to knowledge but with different dynamics)
        truth = 20 + 80 * (1 - np.exp(-3 * time_scale))  # Starts at 20, approaches 100

        # Add some noise
        truth += np.random.normal(0, 1.0, self.num_years)
        truth = np.clip(truth, 0, 100)  # Keep in 0-100 range

        # Combine into dataframe
        data["knowledge_index"] = knowledge
        data["suppression_index"] = suppression
        data["intelligence_index"] = intelligence
        data["truth_index"] = truth

        return data

    def _get_default_parameters(self):
        """Set default parameters for simulation."""
        return {
            # Basic parameters
            "K_0": 1.0,  # Initial knowledge
            "S_0": 10.0,  # Initial suppression
            "I_0": 5.0,  # Initial intelligence
            "T_0": 1.0,  # Initial truth

            # Growth and decay rates
            "knowledge_growth_rate": 0.05,  # Base knowledge growth rate
            "truth_adoption_rate": 0.5,  # Truth adoption acceleration
            "truth_max": 40.0,  # Maximum theoretical truth
            "suppression_decay": 0.05,  # Suppression decay rate

            # Feedback parameters
            "alpha_feedback": 0.1,  # Suppression reinforcement coefficient
            "beta_feedback": 0.05,  # Knowledge disruption coefficient
            "alpha_wisdom": 0.1,  # Wisdom scaling with suppression

            # Resistance and resurgence
            "resistance": 2.0,  # Base resistance level
            "alpha_resurge": 5.0,  # Resurgence intensity
            "mu_resurge": 0.05,  # Resurgence decay rate

            # Phase transition parameters
            "gamma_phase": 0.1,  # Phase transition sharpness
            "t_crit_phase": 20.0,  # Critical threshold for transition
        }

    def run_simulation(self, params=None, return_arrays=False):
        """
        Run a simulation with given parameters.

        Parameters:
            params (dict): Parameters to use, or None to use current parameters
            return_arrays (bool): Whether to return raw arrays instead of dataframe

        Returns:
            DataFrame or dict: Simulation results
        """
        if params is not None:
            self.current_params = params

        # Setup simulation
        timesteps = self.num_years
        dt = 1.0

        # Set up arrays
        K = np.zeros(timesteps)
        S = np.zeros(timesteps)
        I = np.zeros(timesteps)
        T = np.zeros(timesteps)

        # Initial conditions
        K[0] = self.current_params["K_0"]
        S[0] = self.current_params["S_0"]
        I[0] = self.current_params["I_0"]
        T[0] = self.current_params["T_0"]

        # Extract parameters
        knowledge_growth_rate = self.current_params["knowledge_growth_rate"]
        truth_adoption_rate = self.current_params["truth_adoption_rate"]
        truth_max = self.current_params["truth_max"]
        suppression_decay = self.current_params["suppression_decay"]
        alpha_feedback = self.current_params["alpha_feedback"]
        beta_feedback = self.current_params["beta_feedback"]
        alpha_wisdom = self.current_params["alpha_wisdom"]
        resistance = self.current_params["resistance"]
        alpha_resurge = self.current_params["alpha_resurge"]
        mu_resurge = self.current_params["mu_resurge"]
        gamma_phase = self.current_params["gamma_phase"]
        t_crit_phase = self.current_params["t_crit_phase"]

        # Simulation loop
        for t in range(1, timesteps):
            # Calculate wisdom field
            W = wisdom_field(1.0, alpha_wisdom, S[t - 1], resistance, K[t - 1])

            # Update truth adoption
            T[t] = T[t - 1] + truth_adoption(T[t - 1], truth_adoption_rate, truth_max) * dt

            # Update knowledge with phase transition
            # Simplified from full phase transition equation
            growth_term = knowledge_growth_rate * K[t - 1] * (1 + gamma_phase * max(0, T[t - 1] - t_crit_phase))
            K[t] = K[t - 1] + growth_term * dt

            # Update suppression with resurgence
            base_suppression = S[0] * np.exp(-suppression_decay * t)
            if t > int(timesteps / 3):  # Resurgence in middle period
                resurgence = alpha_resurge * np.exp(-mu_resurge * (t - int(timesteps / 3)))
            else:
                resurgence = 0

            suppression_fb = suppression_feedback(alpha_feedback, S[t - 1], beta_feedback, K[t - 1])
            S[t] = base_suppression + resurgence + suppression_fb * dt

            # Ensure non-negative suppression
            S[t] = max(0.1, S[t])

            # Update intelligence
            I[t] = I[t - 1] + intelligence_growth(K[t - 1], W, resistance, S[t - 1], 1.5) * dt

            # Ensure non-negative intelligence
            I[t] = max(0.1, I[t])

        # Scale results to match historical data scale (0-100)
        K_scaled = 100 * K / max(1e-10, K[-1])
        S_scaled = 100 * S / max(1e-10, S[0])  # Higher at beginning
        I_scaled = 100 * I / max(1e-10, I[-1])
        T_scaled = 100 * T / max(1e-10, T[-1])

        if return_arrays:
            return {
                "knowledge": K_scaled,
                "suppression": S_scaled,
                "intelligence": I_scaled,
                "truth": T_scaled,
                "raw_knowledge": K,
                "raw_suppression": S,
                "raw_intelligence": I,
                "raw_truth": T
            }

        # Create dataframe
        sim_data = pd.DataFrame({
            "year": self.years,
            "knowledge_index": K_scaled,
            "suppression_index": S_scaled,
            "intelligence_index": I_scaled,
            "truth_index": T_scaled
        })

        self.simulation_results = sim_data
        return sim_data

    def calculate_error(self, params=None, weighted=True):
        """
        Calculate error between simulation and historical data.

        Parameters:
            params (dict): Parameters to use, or None to use current parameters
            weighted (bool): Whether to use weighted error metrics

        Returns:
            float: Total error metric
        """
        if params is not None:
            sim_data = self.run_simulation(params)
        else:
            if self.simulation_results is None:
                sim_data = self.run_simulation()
            else:
                sim_data = self.simulation_results

        # Merge simulation and historical data
        merged = pd.merge(sim_data, self.historical_data, on="year", suffixes=("_sim", ""))

        # Calculate errors for each metric
        errors = {}
        for metric in self.metrics:
            sim_col = f"{metric}_sim"
            hist_col = metric

            # Calculate mean squared error
            mse = ((merged[sim_col] - merged[hist_col]) ** 2).mean()
            errors[metric] = mse

        # Weighted total error (prioritize knowledge and intelligence)
        if weighted:
            weights = {
                "knowledge_index": 0.4,
                "intelligence_index": 0.3,
                "suppression_index": 0.2,
                "truth_index": 0.1
            }
            total_error = sum(errors[m] * weights[m] for m in self.metrics)
        else:
            total_error = sum(errors.values())

        return total_error

    def optimize_parameters(self, params_to_optimize=None, bounds=None, method='SLSQP'):
        """
        Optimize parameters to minimize error with historical data.

        Parameters:
            params_to_optimize (list): List of parameter names to optimize, or None for all
            bounds (dict): Dictionary of parameter bounds, or None for default bounds
            method (str): Optimization method (SLSQP, L-BFGS-B, etc.)

        Returns:
            dict: Optimized parameters
        """
        if params_to_optimize is None:
            params_to_optimize = list(self.default_params.keys())

        # Setup parameter bounds
        if bounds is None:
            bounds = {}
            for param in params_to_optimize:
                default_val = self.default_params[param]
                # Set bounds to reasonable ranges around default values
                if param in ["K_0", "I_0", "T_0"]:
                    bounds[param] = (0.1, 10.0)
                elif param == "S_0":
                    bounds[param] = (1.0, 50.0)
                elif "rate" in param or "decay" in param:
                    bounds[param] = (0.001, 0.5)
                elif param == "truth_max":
                    bounds[param] = (10.0, 100.0)
                elif param in ["alpha_feedback", "beta_feedback", "alpha_wisdom"]:
                    bounds[param] = (0.01, 0.5)
                elif param == "resistance":
                    bounds[param] = (0.1, 10.0)
                elif param in ["alpha_resurge", "gamma_phase"]:
                    bounds[param] = (0.01, 10.0)
                elif param == "mu_resurge":
                    bounds[param] = (0.001, 0.2)
                elif param == "t_crit_phase":
                    bounds[param] = (5.0, 50.0)
                else:
                    # Default bounds if not specified
                    bounds[param] = (0.1 * default_val, 10.0 * default_val)

        # Prepare initial values and bounds for scipy optimizer
        initial_params = [self.current_params[p] for p in params_to_optimize]
        param_bounds = [bounds[p] for p in params_to_optimize]

        # Define optimization function
        def objective(x):
            # Map parameter values back to dictionary
            param_dict = self.current_params.copy()
            for i, param_name in enumerate(params_to_optimize):
                param_dict[param_name] = x[i]

            # Calculate error
            return self.calculate_error(param_dict)

        # Run optimization
        print(f"Starting parameter optimization for {len(params_to_optimize)} parameters...")
        print(f"Initial error: {self.calculate_error():.4f}")

        result = minimize(
            objective,
            initial_params,
            method=method,
            bounds=param_bounds,
            options={'disp': True}
        )

        # Update parameters with optimized values
        optimized_params = self.current_params.copy()
        for i, param_name in enumerate(params_to_optimize):
            optimized_params[param_name] = result.x[i]

        self.current_params = optimized_params

        # Run final simulation with optimized parameters
        self.run_simulation()

        print(f"Optimization complete. Final error: {self.calculate_error():.4f}")
        print("Optimized parameters:")
        for param in params_to_optimize:
            print(f"  {param}: {self.current_params[param]:.6f}")

        return optimized_params

    def visualize_comparison(self, figsize=(15, 12), save_path=None):
        """
        Visualize comparison between historical data and simulation.

        Parameters:
            figsize (tuple): Figure size
            save_path (str): Path to save figure, or None to display only

        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if self.simulation_results is None:
            self.run_simulation()

        # Merge simulation and historical data
        merged = pd.merge(
            self.simulation_results,
            self.historical_data,
            on="year",
            suffixes=("_sim", "")
        )

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        # Plot metrics
        for i, metric in enumerate(self.metrics):
            ax = axes[i]
            sim_col = f"{metric}_sim"
            hist_col = metric

            # Plot historical data
            ax.plot(merged["year"], merged[hist_col], 'b-', linewidth=2, label="Historical")

            # Plot simulation data
            ax.plot(merged["year"], merged[sim_col], 'r--', linewidth=2, label="Simulation")

            # Calculate error
            mse = ((merged[sim_col] - merged[hist_col]) ** 2).mean()
            rmse = np.sqrt(mse)

            # Add title and labels
            metric_name = metric.replace("_", " ").title()
            ax.set_title(f"{metric_name} (RMSE: {rmse:.2f})")
            ax.set_xlabel("Year")
            ax.set_ylabel("Index Value")
            ax.grid(True)
            ax.legend()

        # Add overall title with parameters
        param_text = (
            f"K₀={self.current_params['K_0']:.2f}, "
            f"S₀={self.current_params['S_0']:.2f}, "
            f"kᵣ={self.current_params['knowledge_growth_rate']:.3f}, "
            f"Tᵣ={self.current_params['truth_adoption_rate']:.2f}"
        )
        fig.suptitle(f"Historical Validation\n{param_text}", fontsize=16)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        # Save figure if path provided
        if save_path is not None:
            plt.savefig(save_path)
            print(f"Figure saved to {save_path}")

        return fig

    def visualize_key_periods(self, periods=None, figsize=(15, 12), save_path=None):
        """
        Visualize comparison during key historical periods.

        Parameters:
            periods (list): List of (name, start_year, end_year) tuples, or None for defaults
            figsize (tuple): Figure size
            save_path (str): Path to save figure, or None to display only

        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if periods is None:
            periods = [
                ("Renaissance", 1400, 1600),
                ("Enlightenment", 1650, 1800),
                ("Industrial Revolution", 1760, 1900),
                ("Modern Era", 1900, self.end_year)
            ]

        if self.simulation_results is None:
            self.run_simulation()

        # Create figure
        fig, axes = plt.subplots(len(periods), len(self.metrics), figsize=figsize)

        # Merge simulation and historical data
        merged = pd.merge(
            self.simulation_results,
            self.historical_data,
            on="year",
            suffixes=("_sim", "")
        )

        # Plot each period and metric
        for i, (period_name, start_year, end_year) in enumerate(periods):
            period_data = merged[(merged["year"] >= start_year) & (merged["year"] <= end_year)]

            for j, metric in enumerate(self.metrics):
                ax = axes[i, j]
                sim_col = f"{metric}_sim"
                hist_col = metric

                # Plot historical data
                ax.plot(period_data["year"], period_data[hist_col], 'b-', linewidth=2, label="Historical")

                # Plot simulation data
                ax.plot(period_data["year"], period_data[sim_col], 'r--', linewidth=2, label="Simulation")

                # Calculate error for this period
                mse = ((period_data[sim_col] - period_data[hist_col]) ** 2).mean()
                rmse = np.sqrt(mse)

                # Add title and labels
                metric_name = metric.replace("_", " ").title()
                if i == 0:
                    ax.set_title(metric_name)
                if j == 0:
                    ax.set_ylabel(f"{period_name}\n({start_year}-{end_year})")
                ax.grid(True)

                # Add error text
                ax.text(0.05, 0.95, f"RMSE: {rmse:.2f}", transform=ax.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

                # Add legend only to top row
                if i == 0:
                    ax.legend()

        # Add overall title
        fig.suptitle("Historical Validation by Period", fontsize=16)

        plt.tight_layout()
        plt.subplots_adjust(top=0.95, hspace=0.3, wspace=0.3)

        # Save figure if path provided
        if save_path is not None:
            plt.savefig(save_path)
            print(f"Figure saved to {save_path}")

        return fig

    def export_to_csv(self, filepath, include_historical=True):
        """
        Export simulation results to CSV.

        Parameters:
            filepath (str): Path to save CSV file
            include_historical (bool): Whether to include historical data in export

        Returns:
            None
        """
        if self.simulation_results is None:
            self.run_simulation()

        if include_historical:
            # Merge simulation and historical data
            export_data = pd.merge(
                self.simulation_results,
                self.historical_data,
                on="year",
                suffixes=("_sim", "_hist")
            )
        else:
            export_data = self.simulation_results

        # Export to CSV
        export_data.to_csv(filepath, index=False)
        print(f"Data exported to {filepath}")


if __name__ == "__main__":
    # Example usage

    # Create historical validation object
    validator = HistoricalValidation()

    # Run simulation with default parameters
    print("Running simulation with default parameters...")
    validator.run_simulation()

    # Calculate error
    error = validator.calculate_error()
    print(f"Error with default parameters: {error:.4f}")

    # Optimize a subset of parameters
    params_to_optimize = ["K_0", "S_0", "knowledge_growth_rate", "truth_adoption_rate"]
    print(f"Optimizing parameters: {params_to_optimize}")
    validator.optimize_parameters(params_to_optimize=params_to_optimize)

    # Visualize comparison
    print("Generating visualizations...")
    validator.visualize_comparison(save_path="outputs/plots/historical_validation_comparison.png")
    validator.visualize_key_periods(save_path="outputs/plots/historical_validation_periods.png")

    # Export results
    print("Exporting results...")
    validator.export_to_csv("outputs/data/historical_validation_results.csv")

    print("Done!")