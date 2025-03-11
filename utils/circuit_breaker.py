"""
Circuit breaker utility for detecting and handling numerical instabilities
in simulations.
"""

import numpy as np

class CircuitBreaker:
    """
    A utility class that detects numerical instabilities in simulations
    and provides mechanisms to recover from them.
    """

    def __init__(self, threshold=1e-6, max_value=1e6, min_value=-1e6, max_rate_of_change=1e3):
        """
        Initialize the circuit breaker with thresholds.

        Args:
            threshold (float): Sensitivity threshold for detecting instabilities
            max_value (float): Maximum allowed value before triggering
            min_value (float): Minimum allowed value before triggering
            max_rate_of_change (float): Maximum allowed rate of change before triggering
        """
        self.threshold = threshold
        self.max_value = max_value
        self.min_value = min_value
        self.max_rate_of_change = max_rate_of_change
        self.trigger_count = 0
        self.was_triggered = False
        self.energy_history = []
        self.last_trigger_reason = None
        self.timestep_recommendations = []

    def check_value_stability(self, value):
        """
        Check if a value indicates numerical instability.

        Args:
            value: The value to check

        Returns:
            bool: True if the value indicates instability, False otherwise
        """
        # Check for NaN or infinity
        if np.isnan(value) or np.isinf(value):
            self.trigger_count += 1
            self.was_triggered = True
            self.last_trigger_reason = "NaN or Infinity detected"
            return True

        # Check for values outside allowed range
        if value > self.max_value or value < self.min_value:
            self.trigger_count += 1
            self.was_triggered = True
            self.last_trigger_reason = f"Value {value} outside allowed range [{self.min_value}, {self.max_value}]"
            return True

        return False

    def check_array_stability(self, array):
        """
        Check if an array contains any values indicating numerical instability.

        Args:
            array: The array to check

        Returns:
            bool: True if the array indicates instability, False otherwise
        """
        # Check for NaN or infinity
        if np.isnan(array).any() or np.isinf(array).any():
            self.trigger_count += 1
            self.was_triggered = True
            self.last_trigger_reason = "NaN or Infinity detected in array"
            return True

        # Check for values outside allowed range
        if (array > self.max_value).any() or (array < self.min_value).any():
            self.trigger_count += 1
            self.was_triggered = True
            self.last_trigger_reason = f"Array values outside allowed range [{self.min_value}, {self.max_value}]"
            return True

        return False

    def check_rate_of_change(self, current, previous):
        """
        Check if the rate of change between two values indicates instability.

        Args:
            current: The current value
            previous: The previous value

        Returns:
            bool: True if the rate of change indicates instability, False otherwise
        """
        if previous == 0:
            rate = current
        else:
            rate = abs(current - previous) / (abs(previous) + self.threshold)

        if rate > self.max_rate_of_change:
            self.trigger_count += 1
            self.was_triggered = True
            self.last_trigger_reason = f"Rate of change {rate} exceeds maximum {self.max_rate_of_change}"

            # Recommend timestep adjustment based on excessive rate of change
            if len(self.timestep_recommendations) < 10:  # Limit number of recommendations
                recommended_dt = 1.0 / (rate / self.max_rate_of_change)
                self.timestep_recommendations.append(recommended_dt)

            return True

        return False

    def check_energy_conservation(self, energy):
        """
        Monitor system energy to detect potential instabilities.

        Args:
            energy: The current system energy

        Returns:
            bool: True if energy indicates instability, False otherwise
        """
        self.energy_history.append(energy)

        # Need at least 3 points to detect instability
        if len(self.energy_history) < 3:
            return False

        # Check for sudden energy spikes (more than 50% increase)
        if energy > 1.5 * self.energy_history[-2]:
            self.trigger_count += 1
            self.was_triggered = True
            self.last_trigger_reason = f"Energy spike detected: {energy} > 1.5 * {self.energy_history[-2]}"
            return True

        return False

    def safe_exp(self, x, max_result=1e10):
        """
        Safe exponential function to prevent overflow.

        Args:
            x: The exponent
            max_result: Maximum allowed result

        Returns:
            float: Bounded exponential result
        """
        # Limit the exponent to avoid overflow
        x = np.clip(x, -50.0, 50.0)
        result = np.clip(np.exp(x), 0.0, max_result)

        # Check if clipping was applied
        if x < -50.0 or x > 50.0 or result == max_result:
            self.trigger_count += 1
            self.was_triggered = True
            self.last_trigger_reason = "Exponential overflow prevented"

        return result

    def safe_div(self, x, y, default=0.0):
        """
        Safe division to prevent division by zero.

        Args:
            x: Numerator
            y: Denominator
            default: Value to return if denominator is near zero

        Returns:
            float: Division result or default
        """
        if abs(y) < self.threshold:
            self.trigger_count += 1
            self.was_triggered = True
            self.last_trigger_reason = "Division by zero prevented"
            return default
        return x / y

    def safe_sqrt(self, x, default=0.0):
        """
        Safe square root function to prevent domain errors.

        Args:
            x: The input value
            default: Value to return if input is negative

        Returns:
            float: Square root result or default
        """
        if x < 0:
            self.trigger_count += 1
            self.was_triggered = True
            self.last_trigger_reason = "Negative square root prevented"
            return default
        return np.sqrt(x)

    def safe_log(self, x, default=0.0):
        """
        Safe logarithm function to prevent domain errors.

        Args:
            x: The input value
            default: Value to return if input is negative or zero

        Returns:
            float: Logarithm result or default
        """
        if x <= 0:
            self.trigger_count += 1
            self.was_triggered = True
            self.last_trigger_reason = "Invalid logarithm input prevented"
            return default
        return np.log(x)

    def recommend_timestep(self, current_dt):
        """
        Recommend a timestep based on collected stability information.

        Args:
            current_dt: Current timestep being used

        Returns:
            float: Recommended timestep for stability
        """
        if not self.timestep_recommendations:
            return current_dt

        # Average the recommendations, but ensure it's not too small
        avg_recommendation = np.mean(self.timestep_recommendations)
        recommended_dt = max(0.01 * current_dt, min(avg_recommendation, current_dt))

        # Clear recommendations after providing advice
        self.timestep_recommendations = []

        return recommended_dt

    def check_gradients(self, gradients, threshold=10.0):
        """
        Check for excessively steep gradients that might cause instability.

        Args:
            gradients: Array of gradient values
            threshold: Maximum allowed gradient magnitude

        Returns:
            bool: True if gradients exceed threshold, False otherwise
        """
        if np.any(np.abs(gradients) > threshold):
            self.trigger_count += 1
            self.was_triggered = True
            self.last_trigger_reason = f"Gradient magnitude exceeds threshold {threshold}"
            return True
        return False

    def get_status_report(self):
        """
        Get a report of the circuit breaker's status.

        Returns:
            dict: Status report with trigger count, last reason, etc.
        """
        return {
            "triggers": self.trigger_count,
            "was_triggered": self.was_triggered,
            "last_reason": self.last_trigger_reason,
            "energy_stability": len(self.energy_history) > 2 and
                               abs(self.energy_history[-1] - self.energy_history[-2]) < 0.1 * self.energy_history[-2]
        }

    def reset(self):
        """Reset the circuit breaker state."""
        self.trigger_count = 0
        self.was_triggered = False
        self.energy_history = []
        self.last_trigger_reason = None
        self.timestep_recommendations = []