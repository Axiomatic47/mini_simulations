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
        self.last_fixed_value = None
        self.last_prev_value = None

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

        # Check for rate of change if we have a previous value
        if self.last_prev_value is not None:
            if abs(value - self.last_prev_value) > self.max_rate_of_change * abs(self.last_prev_value):
                self.was_triggered = True
                self.trigger_count += 1
                self.last_trigger_reason = "Rate of change exceeds threshold"
                return True

        # Update state
        self.last_prev_value = value
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

    def check_and_fix(self, value, min_val=None, max_val=None, default=None):
        """
        Check if a value is stable and fix it if not.

        Args:
            value: Value to check
            min_val (float): Minimum allowed value (default: self.min_value)
            max_val (float): Maximum allowed value (default: self.max_value)
            default: Default value to use if fixing fails

        Returns:
            Value with stability issues fixed
        """
        # Use instance defaults if not provided
        min_val = self.min_value if min_val is None else min_val
        max_val = self.max_value if max_val is None else max_val

        # Use current value as default if not provided
        if default is None:
            if self.last_fixed_value is not None:
                default = self.last_fixed_value
            else:
                default = 0.0

        # Check for array values
        if isinstance(value, np.ndarray):
            return self._fix_array(value, min_val, max_val, default)

        # Try to convert to float if not already numeric
        try:
            value = float(value)
        except (TypeError, ValueError):
            self.was_triggered = True
            self.trigger_count += 1
            self.last_trigger_reason = "Non-numeric value replaced"
            self.last_fixed_value = default
            return default

        # Check for non-finite values
        if not np.isfinite(value):
            self.was_triggered = True
            self.trigger_count += 1
            self.last_trigger_reason = "Non-finite value replaced"
            self.last_fixed_value = default
            return default

        # Check for extreme values
        if abs(value) > max_val:
            self.was_triggered = True
            self.trigger_count += 1
            self.last_trigger_reason = "Extreme value clipped"
            self.last_fixed_value = np.sign(value) * max_val
            return self.last_fixed_value

        # Check for very small non-zero values
        if value != 0 and abs(value) < abs(min_val):
            self.was_triggered = True
            self.trigger_count += 1
            self.last_trigger_reason = "Very small value replaced"
            self.last_fixed_value = np.sign(value) * abs(min_val)
            return self.last_fixed_value

        # Value is stable
        self.last_fixed_value = value
        return value

    def _fix_array(self, array, min_val, max_val, default):
        """Fix stability issues in an array."""
        try:
            array = np.array(array, dtype=float)
        except:
            self.was_triggered = True
            self.trigger_count += 1
            self.last_trigger_reason = "Non-numeric array replaced"
            if isinstance(default, np.ndarray) and default.shape == array.shape:
                return default
            return np.zeros_like(array, dtype=float)

        # Replace non-finite values
        if not np.all(np.isfinite(array)):
            self.was_triggered = True
            self.trigger_count += 1
            self.last_trigger_reason = "Non-finite array values replaced"
            array = np.nan_to_num(array, nan=0.0, posinf=max_val, neginf=-max_val)

        # Clip extreme values
        if np.any(np.abs(array) > max_val):
            self.was_triggered = True
            self.trigger_count += 1
            self.last_trigger_reason = "Extreme array values clipped"
            array = np.clip(array, -max_val, max_val)

        # Replace very small non-zero values
        small_mask = (array != 0) & (np.abs(array) < abs(min_val))
        if np.any(small_mask):
            self.was_triggered = True
            self.trigger_count += 1
            self.last_trigger_reason = "Very small array values replaced"
            array[small_mask] = np.sign(array[small_mask]) * abs(min_val)

        self.last_fixed_value = array.copy()
        return array

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

    def get_trigger_count(self):
        """Get the number of times the circuit breaker was triggered."""
        return self.trigger_count

    def reset(self):
        """Reset the circuit breaker state."""
        self.trigger_count = 0
        self.was_triggered = False
        self.energy_history = []
        self.last_trigger_reason = None
        self.timestep_recommendations = []
        self.last_fixed_value = None
        self.last_prev_value = None