#!/bin/bash

# Change to the project root directory
cd "$(dirname "$0")"

# Create necessary directories if they don't exist
mkdir -p tests
mkdir -p config
mkdir -p outputs/data
mkdir -p outputs/plots
mkdir -p simulations
mkdir -p utils  # Make sure utils directory exists for circuit breaker

# Set up Python environment
if [ -d "venv" ]; then
    echo "Using existing virtual environment..."
    source venv/bin/activate
else
    echo "Creating new virtual environment..."
    python3 -m venv venv
    source venv/bin/activate

    # Check if pip install succeeds
    if ! pip install -r requirements.txt; then
        echo "Error installing requirements. Please check requirements.txt and your internet connection."
        exit 1
    fi
    echo "Virtual environment created and dependencies installed."
fi

# Create __init__.py files for proper package structure
touch config/__init__.py
touch tests/__init__.py
touch simulations/__init__.py
touch utils/__init__.py

# Make sure circuit breaker exists
if [ ! -f "utils/circuit_breaker.py" ]; then
    echo "Creating circuit_breaker.py utility for numerical stability..."
    cat > utils/circuit_breaker.py << 'EOF'
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
            return True

        # Check for values outside allowed range
        if value > self.max_value or value < self.min_value:
            self.trigger_count += 1
            self.was_triggered = True
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
            return True

        # Check for values outside allowed range
        if (array > self.max_value).any() or (array < self.min_value).any():
            self.trigger_count += 1
            self.was_triggered = True
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
        return np.clip(np.exp(x), 0.0, max_result)

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
            return default
        return x / y

    def reset(self):
        """Reset the circuit breaker state."""
        self.trigger_count = 0
        self.was_triggered = False
        self.energy_history = []
EOF

    echo "Circuit breaker utility created."
fi

# Check if specific test is provided
if [ $# -eq 1 ]; then
    test_file=$1

    # Check if the file exists
    if [ ! -f "$test_file" ]; then
        echo "Error: Test file $test_file does not exist"
        exit 1
    fi

    # Check for stability flag
    if [ "$2" == "--check-stability" ]; then
        echo "Running specific test with stability checks: $test_file"
        python -m unittest $test_file --check-stability
    else
        echo "Running specific test: $test_file"
        python -m unittest $test_file
    fi
    exit_code=$?
else
    # Run all tests using standard unittest module
    if [ "$1" == "--check-stability" ]; then
        echo "Running all tests with stability checks..."
        python -m unittest discover -s tests --check-stability
    else
        echo "Running all tests..."
        python -m unittest discover -s tests
    fi
    exit_code=$?
fi

# Print success/failure message
if [ $exit_code -eq 0 ]; then
    echo -e "\n\033[0;32m✅ All tests passed successfully!\033[0m"
else
    echo -e "\n\033[0;31m❌ Some tests failed. Please check the output above for details.\033[0m"
fi

# Exit with the same status as the Python script
exit $exit_code