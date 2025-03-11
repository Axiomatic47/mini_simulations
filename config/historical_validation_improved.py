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


class HistoricalValidationImproved:
    """
    An improved class for validating the Axiomatic Intelligence Growth Simulation against historical data.
    Implements event-driven perturbations, phase-specific parameters, cultural transfer functions,
    and improved suppression dynamics.
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

        # Define historical periods and events
        self.historical_periods = [
            {"name": "Medieval", "start_year": 1000, "end_year": 1400},
            {"name": "Renaissance", "start_year": 1400, "end_year": 1600},
            {"name": "Enlightenment", "start_year": 1600, "end_year": 1800},
            {"name": "Industrial", "start_year": 1800, "end_year": 1900},
            {"name": "Modern", "start_year": 1900, "end_year": 2020}
        ]

        self.historical_events = [
            {"name": "Black Death", "year": 1350, "duration": 10,
             "effects": {"knowledge": -0.2, "suppression": 0.3, "intelligence": -0.2, "truth": -0.1}},
            {"name": "Printing Press", "year": 1440, "duration": 20,
             "effects": {"knowledge": 0.5, "suppression": -0.2, "intelligence": 0.3, "truth": 0.2}},
            {"name": "Religious Wars", "year": 1550, "duration": 80,
             "effects": {"knowledge": -0.1, "suppression": 0.4, "intelligence": -0.1, "truth": -0.2}},
            {"name": "Scientific Revolution", "year": 1600, "duration": 80,
             "effects": {"knowledge": 0.4, "suppression": -0.3, "intelligence": 0.3, "truth": 0.6}},
            {"name": "Industrial Revolution", "year": 1760, "duration": 80,
             "effects": {"knowledge": 0.6, "suppression": -0.2, "intelligence": 0.5, "truth": 0.3}},
            {"name": "World War I", "year": 1914, "duration": 4,
             "effects": {"knowledge": 0.3, "suppression": 0.7, "intelligence": -0.1, "truth": -0.2}},
            {"name": "Great Depression", "year": 1929, "duration": 10,
             "effects": {"knowledge": -0.1, "suppression": 0.3, "intelligence": -0.2, "truth": -0.1}},
            {"name": "World War II", "year": 1939, "duration": 6,
             "effects": {"knowledge": 0.4, "suppression": 0.8, "intelligence": -0.1, "truth": -0.3}},
            {"name": "Information Age", "year": 1970, "duration": 50,
             "effects": {"knowledge": 0.8, "suppression": -0.4, "intelligence": 0.7, "truth": 0.5}}
        ]

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

        # Add Scientific Revolution effect (sudden jump around 1600)
        scientific_revolution_mask = (self.years >= 1600)
        if np.any(scientific_revolution_mask):
            truth[scientific_revolution_mask] += 20

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
            "K_0": 10.0,  # Initial knowledge
            "S_0": 18.0,  # Initial suppression
            "I_0": 5.0,  # Initial intelligence
            "T_0": 1.0,  # Initial truth

            # Growth and decay rates
            "knowledge_growth_rate": 0.027,  # Base knowledge growth rate
            "truth_adoption_rate": 0.34,  # Truth adoption acceleration
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

            # NEW: Period-specific parameters
            "medieval_knowledge_mult": 0.5,  # Knowledge growth multiplier for medieval period
            "renaissance_knowledge_mult": 1.2,  # Knowledge growth multiplier for renaissance
            "enlightenment_knowledge_mult": 1.5,  # Knowledge growth multiplier for enlightenment
            "industrial_knowledge_mult": 2.0,  # Knowledge growth multiplier for industrial rev
            "modern_knowledge_mult": 3.0,  # Knowledge growth multiplier for modern period

            # NEW: Cultural transfer parameters
            "scientific_revolution_effect": 0.6,  # Effect of scientific revolution on truth
            "cultural_diffusion_rate": 0.05,  # Rate of cultural diffusion of knowledge
            "truth_knowledge_synergy": 0.2,  # Synergistic effect between truth and knowledge

            # NEW: Suppression parameters
            "suppression_resurgence_strength": 0.8,  # Strength of suppression resurgence
            "suppression_recovery_rate": 0.1,  # Rate of recovery from suppression spikes
            "war_suppression_multiplier": 2.0,  # Multiplier for war effects on suppression
        }

    def _get_period_multipliers(self, params, year):
        """Get period-specific multipliers for a given year."""
        # Default multiplier is 1.0
        multipliers = {"knowledge": 1.0, "truth": 1.0, "suppression": 1.0, "intelligence": 1.0}

        # Apply period-specific multipliers
        if year <= 1400:  # Medieval
            multipliers["knowledge"] = params["medieval_knowledge_mult"]
        elif year <= 1600:  # Renaissance
            multipliers["knowledge"] = params["renaissance_knowledge_mult"]
        elif year <= 1800:  # Enlightenment
            multipliers["knowledge"] = params["enlightenment_knowledge_mult"]
        elif year <= 1900:  # Industrial
            multipliers["knowledge"] = params["industrial_knowledge_mult"]
        else:  # Modern
            multipliers["knowledge"] = params["modern_knowledge_mult"]

        return multipliers

    def _get_event_effects(self, year):
        """Calculate event effects for a given year."""
        effects = {"knowledge": 0.0, "suppression": 0.0, "intelligence": 0.0, "truth": 0.0}

        for event in self.historical_events:
            event_start = event["year"]
            event_end = event["year"] + event["duration"]

            if event_start <= year <= event_end:
                # Calculate effect strength based on position within event duration
                # Strongest at the beginning, tapering off toward the end
                position = (year - event_start) / max(1, event["duration"])
                strength = 1.0 - position ** 2  # Quadratic decay

                # Apply event effects with decay
                for metric, effect in event["effects"].items():
                    effects[metric] += effect * strength

        return effects

    def _apply_cultural_transfer(self, K, T, params):
        """Apply cultural transfer effects between knowledge and truth."""
        # Synergistic effect - knowledge and truth enhance each other
        k_enhancement = params["truth_knowledge_synergy"] * T * K / 100.0
        t_enhancement = params["cultural_diffusion_rate"] * K

        # Add scientific revolution effect if applicable
        if T > 20:  # Only apply after a certain truth threshold
            t_enhancement += params["scientific_revolution_effect"] * K / 20.0

        return k_enhancement, t_enhancement

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
        war_suppression_multiplier = self.current_params["war_suppression_multiplier"]

        # Simulation loop
        for t in range(1, timesteps):
            current_year = self.years[t]

            # Get period-specific multipliers
            period_mults = self._get_period_multipliers(self.current_params, current_year)

            # Get event effects
            event_effects = self._get_event_effects(current_year)

            # Apply cultural transfer effects
            k_enhancement, t_enhancement = self._apply_cultural_transfer(K[t - 1], T[t - 1], self.current_params)

            # Calculate wisdom field
            W = wisdom_field(1.0, alpha_wisdom, S[t - 1], resistance, K[t - 1])

            # Update truth adoption with cultural effects
            truth_change = truth_adoption(T[t - 1], truth_adoption_rate, truth_max) + t_enhancement
            truth_change *= (1.0 + event_effects["truth"])  # Apply event effects
            T[t] = T[t - 1] + truth_change * dt

            # Update knowledge with phase transition and period multipliers
            growth_term = knowledge_growth_rate * K[t - 1] * (1 + gamma_phase * max(0, T[t - 1] - t_crit_phase))
            growth_term *= period_mults["knowledge"]  # Apply period multiplier
            growth_term += k_enhancement  # Add cultural enhancement
            growth_term *= (1.0 + event_effects["knowledge"])  # Apply event effects
            K[t] = K[t - 1] + growth_term * dt

            # Update suppression with resurgence and event effects
            # Base suppression decay
            base_suppression = S[0] * np.exp(-suppression_decay * t)

            # Event-driven suppression
            if event_effects["suppression"] > 0:
                # Amplify suppression during wars and other high-suppression events
                suppression_event = event_effects["suppression"] * S[t - 1] * war_suppression_multiplier
            else:
                suppression_event = event_effects["suppression"] * S[t - 1]

            # Feedback and recovery dynamics
            suppression_fb = suppression_feedback(alpha_feedback, S[t - 1], beta_feedback, K[t - 1])

            # Combine all suppression effects
            S[t] = base_suppression + suppression_fb * dt + suppression_event

            # Ensure non-negative suppression
            S[t] = max(0.1, S[t])

            # Update intelligence with event effects
            intel_growth = intelligence_growth(K[t - 1], W, resistance, S[t - 1], 1.5)
            intel_growth *= (1.0 + event_effects["intelligence"])  # Apply event effects
            I[t] = max(0.1, I[t - 1] + intel_growth * dt)  # Ensure positive intelligence

        # Scale results to match historical data scale (0-100)
        K_scaled = 100 * K / max(1e-10, K[-1])
        S_scaled = S.copy()  # Suppression is already on a 0-100 scale
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
                    bounds[param] = (0.1, 20.0)
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
                elif "mult" in param:
                    bounds[param] = (0.1, 5.0)
                elif "effect" in param:
                    bounds[param] = (0.0, 1.0)
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

        # Define historical periods for shading
        periods = [
            {"name": "Medieval", "start": 1000, "end": 1400, "color": "gray", "alpha": 0.1},
            {"name": "Renaissance", "start": 1400, "end": 1600, "color": "gold", "alpha": 0.1},
            {"name": "Enlightenment", "start": 1600, "end": 1800, "color": "lightblue", "alpha": 0.1},
            {"name": "Industrial", "start": 1800, "end": 1900, "color": "lightgreen", "alpha": 0.1},
            {"name": "Modern", "start": 1900, "end": 2020, "color": "salmon", "alpha": 0.1}
        ]

        # Define events to mark
        events = [
            {"name": "Printing Press", "year": 1440, "color": "green"},
            {"name": "Scientific Revolution", "year": 1600, "color": "blue"},
            {"name": "Industrial Revolution", "year": 1760, "color": "brown"},
            {"name": "World Wars", "year": 1914, "color": "red"},
            {"name": "Information Age", "year": 1970, "color": "purple"}
        ]

        # Plot metrics
        for i, metric in enumerate(self.metrics):
            ax = axes[i]
            sim_col = f"{metric}_sim"
            hist_col = metric

            # Add period shading
            for period in periods:
                if period["start"] >= min(merged["year"]) and period["end"] <= max(merged["year"]):
                    ax.axvspan(period["start"], period["end"],
                               alpha=period["alpha"], color=period["color"])

            # Plot historical data
            ax.plot(merged["year"], merged[hist_col], 'b-', linewidth=2, label="Historical")

            # Plot simulation data
            ax.plot(merged["year"], merged[sim_col], 'r--', linewidth=2, label="Simulation")

            # Add event markers
            for event in events:
                if event["year"] >= min(merged["year"]) and event["year"] <= max(merged["year"]):
                    ax.axvline(x=event["year"], color=event["color"], linestyle=':', alpha=0.7)
                    # Add text annotation slightly above
                    y_pos = 0.9 * ax.get_ylim()[1]
                    ax.text(event["year"], y_pos, event["name"],
                            rotation=90, color=event["color"], ha='right', fontsize=8)

            # Calculate RMSE for this metric
            rmse = np.sqrt(((merged[sim_col] - merged[hist_col]) ** 2).mean())

            # Set title with RMSE
            metric_name = metric.replace("_index", "").title()
            ax.set_title(f"{metric_name} Over Time (RMSE: {rmse:.2f})")

            ax.set_xlabel("Year")
            ax.set_ylabel(metric_name)
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Add overall title with parameter info
        param_str = f"K₀={self.current_params['K_0']:.2f}, S₀={self.current_params['S_0']:.2f}, "
        param_str += f"kᵣ={self.current_params['knowledge_growth_rate']:.3f}, "
        param_str += f"Tᵣ={self.current_params['truth_adoption_rate']:.2f}"

        plt.suptitle(f"Historical Validation - Simulation vs. Historical Data\n{param_str}",
                     fontsize=14)

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle

        # Save figure if requested
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {save_path}")

        return fig

    def visualize_periods(self, figsize=(15, 16), save_path=None):
        """
        Visualize comparison between historical data and simulation for each historical period.

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

        # Create figure with one row per period
        fig, axes = plt.subplots(len(self.historical_periods), len(self.metrics), figsize=figsize)

        # Plot each period and metric
        for i, period in enumerate(self.historical_periods):
            # Filter data for this period
            period_data = merged[(merged["year"] >= period["start_year"]) &
                                 (merged["year"] <= period["end_year"])]

            if len(period_data) == 0:
                continue

            for j, metric in enumerate(self.metrics):
                ax = axes[i, j]
                sim_col = f"{metric}_sim"
                hist_col = metric

                # Plot historical data
                ax.plot(period_data["year"], period_data[hist_col], 'b-', linewidth=2,
                        label="Historical")

                # Plot simulation data
                ax.plot(period_data["year"], period_data[sim_col], 'r--', linewidth=2,
                        label="Simulation")

                # Calculate RMSE for this period and metric
                rmse = np.sqrt(((period_data[sim_col] - period_data[hist_col]) ** 2).mean())

                # Set title
                metric_name = metric.replace("_index", "").title()
                if i == 0:  # Only set column titles on the first row
                    ax.set_title(metric_name)

                # Set labels
                if j == 0:  # Only set period labels on the first column
                    ax.set_ylabel(f"{period['name']}\n({period['start_year']}-{period['end_year']})")

                if i == len(self.historical_periods) - 1:  # Only set x labels on the last row
                    ax.set_xlabel("Year")

                # Add RMSE text
                ax.text(0.05, 0.95, f"RMSE: {rmse:.2f}", transform=ax.transAxes,
                        fontsize=9, va='top', bbox=dict(facecolor='white', alpha=0.5))

                # Only add legend to the first plot
                if i == 0 and j == 0:
                    ax.legend()

                ax.grid(True, alpha=0.3)

        plt.suptitle("Historical Validation by Period", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.98])  # Make room for the suptitle

        # Save figure if requested
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {save_path}")

        return fig

    def visualize_events(self, figsize=(15, 15), save_path=None):
        """
        Visualize the impact of key historical events on model metrics.

        Parameters:
            figsize (tuple): Figure size
            save_path (str): Path to save figure, or None to display only

        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if self.simulation_results is None:
            self.run_simulation()

        # Select key events to visualize (those with significant effects)
        key_events = [
            {"name": "Black Death", "year": 1350, "window": 50},
            {"name": "Printing Press", "year": 1440, "window": 60},
            {"name": "Scientific Revolution", "year": 1600, "window": 100},
            {"name": "Industrial Revolution", "year": 1760, "window": 100},
            {"name": "World Wars", "year": 1914, "window": 40}
        ]

        # Create figure
        fig, axes = plt.subplots(len(key_events), len(self.metrics), figsize=figsize)

        # Merge simulation and historical data
        merged = pd.merge(
            self.simulation_results,
            self.historical_data,
            on="year",
            suffixes=("_sim", "")
        )

        # Plot each event and metric
        for i, event in enumerate(key_events):
            # Extract data around the event
            event_year = event["year"]
            window = event["window"]
            event_data = merged[(merged["year"] >= event_year - window // 2) &
                                (merged["year"] <= event_year + window // 2)]

            if len(event_data) == 0:
                continue

            for j, metric in enumerate(self.metrics):
                ax = axes[i, j]
                sim_col = f"{metric}_sim"
                hist_col = metric

                # Plot historical data
                ax.plot(event_data["year"], event_data[hist_col], 'b-', linewidth=2,
                        label="Historical")

                # Plot simulation data
                ax.plot(event_data["year"], event_data[sim_col], 'r--', linewidth=2,
                        label="Simulation")

                # Add vertical line at event year
                ax.axvline(x=event_year, color='k', linestyle=':', alpha=0.7)

                # Set title
                metric_name = metric.replace("_index", "").title()
                if i == 0:  # Only set column titles on the first row
                    ax.set_title(metric_name)

                # Set labels
                if j == 0:  # Only set event labels on the first column
                    ax.set_ylabel(event["name"])

                if i == len(key_events) - 1:  # Only set x labels on the last row
                    ax.set_xlabel("Year")

                # Calculate event impact (difference from pre-event to post-event)
                pre_event = event_data[event_data["year"] < event_year][hist_col].mean()
                post_event = event_data[event_data["year"] > event_year][hist_col].mean()
                impact = post_event - pre_event

                # Add impact text
                impact_text = f"Impact: {impact:.1f}"
                ax.text(0.05, 0.95, impact_text, transform=ax.transAxes,
                        fontsize=9, va='top', bbox=dict(facecolor='white', alpha=0.5))

                # Only add legend to the first plot
                if i == 0 and j == 0:
                    ax.legend()

                ax.grid(True, alpha=0.3)

        plt.suptitle("Impact of Historical Events", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.98])  # Make room for the suptitle

        # Save figure if requested
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {save_path}")

        return fig

    def save_results(self, output_dir="outputs"):
        """
        Save all simulation results and parameters to files.

        Parameters:
            output_dir (str): Directory to save results
        """
        # Make sure we have results
        if self.simulation_results is None:
            self.run_simulation()

        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Save simulation results
        sim_path = Path(output_dir) / "simulation_results.csv"
        self.simulation_results.to_csv(sim_path, index=False)
        print(f"Saved simulation results to {sim_path}")

        # Save historical data
        hist_path = Path(output_dir) / "historical_data.csv"
        self.historical_data.to_csv(hist_path, index=False)
        print(f"Saved historical data to {hist_path}")

        # Save parameters
        param_path = Path(output_dir) / "simulation_parameters.csv"
        param_df = pd.DataFrame({"parameter": list(self.current_params.keys()),
                                 "value": list(self.current_params.values())})
        param_df.to_csv(param_path, index=False)
        print(f"Saved simulation parameters to {param_path}")

        # Save error metrics
        error_metrics = {}
        for metric in self.metrics:
            sim_values = self.simulation_results[f"{metric}"]
            hist_values = self.historical_data[metric]
            mse = ((sim_values - hist_values) ** 2).mean()
            rmse = np.sqrt(mse)
            error_metrics[f"{metric}_mse"] = mse
            error_metrics[f"{metric}_rmse"] = rmse

        error_metrics["total_error"] = self.calculate_error()
        error_metrics["unweighted_error"] = self.calculate_error(weighted=False)

        error_path = Path(output_dir) / "error_metrics.csv"
        error_df = pd.DataFrame({"metric": list(error_metrics.keys()),
                                 "value": list(error_metrics.values())})
        error_df.to_csv(error_path, index=False)
        print(f"Saved error metrics to {error_path}")

    def run_comprehensive_analysis(self, output_dir="outputs", optimize=True):
        """
        Run a comprehensive analysis including optimization, visualizations, and result saving.

        Parameters:
            output_dir (str): Directory to save results
            optimize (bool): Whether to optimize parameters

        Returns:
            dict: Dictionary with analysis results
        """
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Run initial simulation with default parameters
        print("Running initial simulation with default parameters...")
        self.run_simulation()

        # Save initial results
        self.save_results(output_dir=f"{output_dir}/default")

        # Create initial visualizations
        print("Creating visualizations for default parameters...")
        fig1 = self.visualize_comparison(save_path=f"{output_dir}/default/comparison.png")
        fig2 = self.visualize_periods(save_path=f"{output_dir}/default/periods.png")
        fig3 = self.visualize_events(save_path=f"{output_dir}/default/events.png")

        # Optimize parameters if requested
        if optimize:
            print("\nOptimizing parameters...")
            # Focus on key parameters for optimization
            params_to_optimize = [
                "K_0", "S_0", "knowledge_growth_rate", "truth_adoption_rate",
                "suppression_decay", "medieval_knowledge_mult", "renaissance_knowledge_mult",
                "enlightenment_knowledge_mult", "industrial_knowledge_mult",
                "modern_knowledge_mult", "scientific_revolution_effect",
                "cultural_diffusion_rate", "truth_knowledge_synergy",
                "war_suppression_multiplier"
            ]
            optimized_params = self.optimize_parameters(params_to_optimize=params_to_optimize)

            # Save optimized results
            self.save_results(output_dir=f"{output_dir}/optimized")

            # Create optimized visualizations
            print("Creating visualizations for optimized parameters...")
            fig4 = self.visualize_comparison(save_path=f"{output_dir}/optimized/comparison.png")
            fig5 = self.visualize_periods(save_path=f"{output_dir}/optimized/periods.png")
            fig6 = self.visualize_events(save_path=f"{output_dir}/optimized/events.png")

        # Prepare summary
        initial_error = self.calculate_error(self.default_params)
        final_error = self.calculate_error()

        results = {
            "initial_error": initial_error,
            "final_error": final_error,
            "improvement": (initial_error - final_error) / initial_error * 100 if initial_error > 0 else 0,
            "default_parameters": self.default_params,
            "current_parameters": self.current_params
        }

        # Print summary
        print("\n" + "=" * 50)
        print("Comprehensive Analysis Summary")
        print("=" * 50)
        print(f"Initial error: {initial_error:.4f}")
        print(f"Final error: {final_error:.4f}")
        print(f"Improvement: {results['improvement']:.2f}%")
        print("=" * 50)

        return results