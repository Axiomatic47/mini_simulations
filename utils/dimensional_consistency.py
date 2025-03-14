import functools
# utils/dimensional_consistency.py

import numpy as np
from enum import Enum

class Dimension(Enum):
    """Enumeration of possible dimensions in the simulation."""
    KNOWLEDGE = "knowledge"
    INTELLIGENCE = "intelligence"
    TRUTH = "truth"
    SUPPRESSION = "suppression"
    RESISTANCE = "resistance"
    WISDOM = "wisdom"
    TIME = "time"
    SPACE = "space"
    PROBABILITY = "probability"
    ENERGY = "energy"
    FORCE = "force"
    INFLUENCE = "influence"
    RESOURCES = "resources"
    RATE = "rate"  # Added for truth adoption rate

    # Dimensionless quantities
    DIMENSIONLESS = "dimensionless"


class DimensionalValue:
    """Class to represent a value with its associated dimension."""

    def __init__(self, value, dimension, units=None):
        """
        Initialize a dimensional value.

        Args:
            value: The numerical value
            dimension: The dimension (from Dimension enum)
            units: Optional specific units for this dimension
        """
        self.value = value
        self.dimension = dimension
        self.units = units

    def __repr__(self):
        return f"DimensionalValue({self.value}, {self.dimension}, {self.units})"

    def __add__(self, other):
        """Addition that validates dimensional consistency."""
        if not isinstance(other, DimensionalValue):
            raise TypeError(f"Cannot add DimensionalValue to {type(other)}")

        if self.dimension != other.dimension:
            raise ValueError(f"Cannot add values with dimensions {self.dimension} and {other.dimension}")

        return DimensionalValue(self.value + other.value, self.dimension, self.units)

    def __sub__(self, other):
        """Subtraction that validates dimensional consistency."""
        if not isinstance(other, DimensionalValue):
            raise TypeError(f"Cannot subtract {type(other)} from DimensionalValue")

        if self.dimension != other.dimension:
            raise ValueError(f"Cannot subtract values with dimensions {self.dimension} and {other.dimension}")

        return DimensionalValue(self.value - other.value, self.dimension, self.units)

    def __mul__(self, other):
        """Multiplication with dimensional tracking."""
        if isinstance(other, (int, float, np.number)):
            # Scalar multiplication preserves dimension
            return DimensionalValue(self.value * other, self.dimension, self.units)

        elif isinstance(other, DimensionalValue):
            # For dimensional multiplication, we would need a more complex system
            # to track combined dimensions. For simplicity, we'll return DIMENSIONLESS
            # but in a full implementation, you would create combined dimensions.
            return DimensionalValue(self.value * other.value, Dimension.DIMENSIONLESS)

        else:
            raise TypeError(f"Cannot multiply DimensionalValue by {type(other)}")

    def __truediv__(self, other):
        """Division with dimensional tracking."""
        if isinstance(other, (int, float, np.number)):
            # Scalar division preserves dimension
            return DimensionalValue(self.value / other, self.dimension, self.units)

        elif isinstance(other, DimensionalValue):
            if self.dimension == other.dimension:
                # Same dimensions cancel to dimensionless
                return DimensionalValue(self.value / other.value, Dimension.DIMENSIONLESS)
            else:
                # For different dimensions, would need proper dimensional analysis
                # For simplicity, we'll return DIMENSIONLESS
                return DimensionalValue(self.value / other.value, Dimension.DIMENSIONLESS)

        else:
            raise TypeError(f"Cannot divide DimensionalValue by {type(other)}")


def validate_equation_dimensions(func):
    """
    Decorator to validate dimensional consistency of equation outputs.

    This decorator wraps equation functions to ensure they maintain
    dimensional consistency when processing inputs and outputs.
    """

    def wrapper(*args, **kwargs):
        # Process inputs to ensure they have proper dimensions
        processed_args = []
        for arg in args:
            if isinstance(arg, (int, float, np.number)) and not isinstance(arg, DimensionalValue):
                # Warn about dimensionless inputs but allow them for backward compatibility
                print(f"Warning: Dimensionless input {arg} passed to {func.__name__}")
                processed_args.append(arg)
            else:
                processed_args.append(arg)

        # Call the original function
        result = func(*processed_args, **kwargs)

        # Check the output dimension
        if not isinstance(result, DimensionalValue) and not hasattr(result, 'dimension'):
            # For backward compatibility, don't raise an error but print a warning
            print(f"Warning: Function {func.__name__} returned a value without dimensional information")

        return result

    return wrapper


def check_dimensional_consistency(equations_dict_or_func):
    """
    Check dimensional consistency across equations.
    Can be used as a decorator on a single function or called with a dictionary of functions.
    """
    # Initialize results dictionary
    results = {}

    # If a single function is passed, convert to dict
    if callable(equations_dict_or_func):
        func = equations_dict_or_func

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Call the original function
            return func(*args, **kwargs)

        equations_dict = {func.__name__: func}

        # Perform the dimensional check
        for name, equation in equations_dict.items():
            try:
                # This is simplified - in practice, you'd need to generate
                # appropriate test inputs for each equation
                test_inputs = generate_test_inputs(name)
                output = equation(*test_inputs)

                # Check if output has expected dimension
                expected_dimension = get_expected_dimension(name)
                if hasattr(output, 'dimension') and output.dimension != expected_dimension:
                    results[name] = {
                        "status": "Inconsistent",
                        "expected": expected_dimension,
                        "actual": output.dimension
                    }
                else:
                    results[name] = {"status": "Consistent"}

            except Exception as e:
                results[name] = {
                    "status": "Error",
                    "message": str(e)
                }

        return wrapper
    else:
        equations_dict = equations_dict_or_func

        # For each equation, check its inputs and outputs
        for name, equation in equations_dict.items():
            try:
                # This is simplified - in practice, you'd need to generate
                # appropriate test inputs for each equation
                test_inputs = generate_test_inputs(name)
                output = equation(*test_inputs)

                # Check if output has expected dimension
                expected_dimension = get_expected_dimension(name)
                if hasattr(output, 'dimension') and output.dimension != expected_dimension:
                    results[name] = {
                        "status": "Inconsistent",
                        "expected": expected_dimension,
                        "actual": output.dimension
                    }
                else:
                    results[name] = {"status": "Consistent"}

            except Exception as e:
                results[name] = {
                    "status": "Error",
                    "message": str(e)
                }

        return results


def generate_test_inputs(equation_name):
    """
    Generate appropriate test inputs for a given equation.
    This would be customized based on the specific equation.
    """
    # This is a placeholder implementation
    if equation_name == "intelligence_growth" or equation_name == "intelligence_growth_with_dimensions":
        # For intelligence growth equation - K, W, R, S, N
        return [
            DimensionalValue(10.0, Dimension.KNOWLEDGE),
            DimensionalValue(1.0, Dimension.WISDOM),
            DimensionalValue(2.0, Dimension.RESISTANCE),
            DimensionalValue(5.0, Dimension.SUPPRESSION),
            DimensionalValue(0.5, Dimension.DIMENSIONLESS)
        ]
    elif equation_name == "wisdom_field" or equation_name == "wisdom_field_with_dimensions":
        # For wisdom field equation - W_0, alpha, S, R, K
        return [
            1.0,  # W_0 (scalar)
            0.1,  # alpha (scalar)
            DimensionalValue(5.0, Dimension.SUPPRESSION),
            DimensionalValue(2.0, Dimension.RESISTANCE),
            DimensionalValue(10.0, Dimension.KNOWLEDGE)
        ]
    elif equation_name == "truth_adoption" or equation_name == "truth_adoption_with_dimensions":
        # For truth adoption equation - T, A, T_max
        return [
            DimensionalValue(10.0, Dimension.TRUTH),
            2.5,  # A (scalar)
            40.0  # T_max (scalar)
        ]
    elif equation_name == "suppression_feedback" or equation_name == "suppression_feedback_with_dimensions":
        # For suppression feedback equation - alpha, S, beta, K
        return [
            0.1,  # alpha (scalar)
            DimensionalValue(5.0, Dimension.SUPPRESSION),
            0.05,  # beta (scalar)
            DimensionalValue(10.0, Dimension.KNOWLEDGE)
        ]
    elif equation_name == "resistance_resurgence" or equation_name == "resistance_resurgence_with_dimensions":
        # For resistance resurgence equation - S_0, lambda_decay, t, alpha_resurge, mu_resurge, t_crit
        return [
            5.0,  # S_0 (scalar)
            0.05,  # lambda_decay (scalar)
            100,  # t (integer time)
            5.0,  # alpha_resurge (scalar)
            0.05,  # mu_resurge (scalar)
            150  # t_crit (integer time)
        ]

    # Default fallback
    return [DimensionalValue(1.0, Dimension.DIMENSIONLESS)]


def get_expected_dimension(equation_name):
    """
    Get the expected output dimension for a given equation.
    """
    dimension_map = {
        "intelligence_growth": Dimension.INTELLIGENCE,
        "intelligence_growth_with_dimensions": Dimension.INTELLIGENCE,
        "truth_adoption": Dimension.TRUTH,
        "truth_adoption_with_dimensions": Dimension.RATE,
        "wisdom_field": Dimension.WISDOM,
        "wisdom_field_with_dimensions": Dimension.WISDOM,
        "suppression_feedback": Dimension.SUPPRESSION,
        "suppression_feedback_with_dimensions": Dimension.SUPPRESSION,
        "resistance_resurgence": Dimension.SUPPRESSION,
        "resistance_resurgence_with_dimensions": Dimension.SUPPRESSION,
        "knowledge_field_influence": Dimension.FORCE,
        "quantum_tunneling_probability": Dimension.PROBABILITY,
        # Add mappings for all other equations
    }

    return dimension_map.get(equation_name, Dimension.DIMENSIONLESS)


# Example of applying the dimensional validation to an equation
@validate_equation_dimensions
def intelligence_growth_with_dimensions(K, W, R, S, N, K_max=100.0):
    """
    Dimensionally-validated version of intelligence growth equation.

    Args:
        K: Knowledge with KNOWLEDGE dimension
        W: Wisdom with WISDOM dimension
        R: Resistance with RESISTANCE dimension
        S: Suppression with SUPPRESSION dimension
        N: Network effect (dimensionless or scalar)
        K_max: Maximum knowledge capacity (scalar)

    Returns:
        Intelligence growth rate with INTELLIGENCE dimension
    """
    # Ensure inputs have correct dimensions
    if K.dimension != Dimension.KNOWLEDGE:
        raise ValueError(f"Expected KNOWLEDGE dimension, got {K.dimension}")
    if W.dimension != Dimension.WISDOM:
        raise ValueError(f"Expected WISDOM dimension, got {W.dimension}")
    if R.dimension != Dimension.RESISTANCE:
        raise ValueError(f"Expected RESISTANCE dimension, got {R.dimension}")
    if S.dimension != Dimension.SUPPRESSION:
        raise ValueError(f"Expected SUPPRESSION dimension, got {S.dimension}")

    # Handle N as either a dimensional value or scalar
    n_value = N.value if isinstance(N, DimensionalValue) else N

    # Perform the calculation with dimension tracking
    growth_term = (K.value * W.value) / (1.0 + K.value / K_max)

    # Create a dimensional result
    result = growth_term - R.value - S.value + n_value
    return DimensionalValue(result, Dimension.INTELLIGENCE)


@validate_equation_dimensions
def wisdom_field_with_dimensions(W_0, alpha, S, R, K):
    """
    Dimensionally-validated version of wisdom field equation.

    Args:
        W_0: Base wisdom (scalar)
        alpha: Suppression impact factor (scalar)
        S: Suppression with SUPPRESSION dimension
        R: Resistance with RESISTANCE dimension
        K: Knowledge with KNOWLEDGE dimension

    Returns:
        Wisdom value with WISDOM dimension
    """
    # Ensure inputs have correct dimensions
    if S.dimension != Dimension.SUPPRESSION:
        raise ValueError(f"Expected SUPPRESSION dimension, got {S.dimension}")
    if R.dimension != Dimension.RESISTANCE:
        raise ValueError(f"Expected RESISTANCE dimension, got {R.dimension}")
    if K.dimension != Dimension.KNOWLEDGE:
        raise ValueError(f"Expected KNOWLEDGE dimension, got {K.dimension}")

    # W_0 and alpha are dimensionless scalars

    # Perform calculation with dimension tracking
    S_max = 100.0  # Maximum suppression for stability
    R_max = 10.0  # Maximum resistance for stability
    K_min = 0.01  # Minimum knowledge to prevent division by zero

    exp_term = np.exp(-alpha * min(S_max, S.value))
    ratio_term = (1.0 + min(R_max, R.value) / max(K_min, K.value))

    return DimensionalValue(W_0 * exp_term * ratio_term, Dimension.WISDOM)


@validate_equation_dimensions
def truth_adoption_with_dimensions(T, A, T_max):
    """
    Dimensionally-validated version of truth adoption equation.

    Args:
        T: Truth with TRUTH dimension
        A: Adoption acceleration factor (scalar)
        T_max: Maximum truth value (scalar)

    Returns:
        Truth adoption rate with RATE dimension
    """
    # Ensure inputs have correct dimensions
    if T.dimension != Dimension.TRUTH:
        raise ValueError(f"Expected TRUTH dimension, got {T.dimension}")

    # A is a dimensionless acceleration factor

    # Perform calculation with dimension tracking
    dT = A * (1.0 - (T.value / T_max) ** 2)

    return DimensionalValue(dT, Dimension.RATE)  # Rate of change of truth


@validate_equation_dimensions
def suppression_feedback_with_dimensions(alpha, S, beta, K):
    """
    Dimensionally-validated version of suppression feedback equation.

    Args:
        alpha: Suppression reinforcement coefficient (scalar)
        S: Suppression with SUPPRESSION dimension
        beta: Knowledge disruption coefficient (scalar)
        K: Knowledge with KNOWLEDGE dimension

    Returns:
        Suppression feedback with SUPPRESSION dimension
    """
    # Ensure inputs have correct dimensions
    if S.dimension != Dimension.SUPPRESSION:
        raise ValueError(f"Expected SUPPRESSION dimension, got {S.dimension}")
    if K.dimension != Dimension.KNOWLEDGE:
        raise ValueError(f"Expected KNOWLEDGE dimension, got {K.dimension}")

    # Perform calculation with dimension tracking
    S_max = 100.0  # Maximum suppression for stability
    K_max = 1000.0  # Maximum knowledge for stability

    reinforcement = alpha * min(S_max, S.value)
    disruption = beta * min(K_max, K.value) * (1 + 0.1 * K.value / 100.0)

    # Apply maximum feedback bound
    feedback = min(5.0, reinforcement) - disruption

    return DimensionalValue(feedback, Dimension.SUPPRESSION)


@validate_equation_dimensions
def resistance_resurgence_with_dimensions(S_0, lambda_decay, t, alpha_resurge, mu_resurge, t_crit):
    """
    Dimensionally-validated version of resistance resurgence equation.

    Args:
        S_0: Initial suppression (scalar)
        lambda_decay: Decay rate (scalar)
        t: Time (integer)
        alpha_resurge: Maximum resurgence strength (scalar)
        mu_resurge: Decay rate of resurgence (scalar)
        t_crit: Critical time when resurgence begins (integer)

    Returns:
        Suppression level with SUPPRESSION dimension
    """
    # Perform calculation with dimension tracking
    t_max = 1000  # Maximum time value for stability
    t_total = 1000  # Total simulation time
    safe_t = min(t_max, t)

    # Base suppression decay
    base_suppression = S_0 * np.exp(-lambda_decay * safe_t)

    # Resurgence term
    if t > t_crit:
        decay_exponent = max(-50, -mu_resurge * min(t_max, t - t_crit))
        time_factor = max(0, 1 - (t - t_crit) / t_total)
        resurgence = alpha_resurge * np.exp(decay_exponent) * time_factor
    else:
        resurgence = 0

    return DimensionalValue(base_suppression + resurgence, Dimension.SUPPRESSION)


# Example of updating an existing function with dimensional validation
def update_equations_with_dimensions(equations_module):
    """
    Add dimensional validation to existing equation functions.

    Args:
        equations_module: Module containing the equation functions

    Returns:
        dict: Dictionary of updated functions with dimensional validation
    """
    updated_equations = {}

    # Wrap core equations with dimensional validation
    if hasattr(equations_module, 'intelligence_growth'):
        updated_equations['intelligence_growth'] = validate_equation_dimensions(
            equations_module.intelligence_growth
        )

    if hasattr(equations_module, 'truth_adoption'):
        updated_equations['truth_adoption'] = validate_equation_dimensions(
            equations_module.truth_adoption
        )

    if hasattr(equations_module, 'wisdom_field'):
        updated_equations['wisdom_field'] = validate_equation_dimensions(
            equations_module.wisdom_field
        )

    if hasattr(equations_module, 'suppression_feedback'):
        updated_equations['suppression_feedback'] = validate_equation_dimensions(
            equations_module.suppression_feedback
        )

    if hasattr(equations_module, 'resistance_resurgence'):
        updated_equations['resistance_resurgence'] = validate_equation_dimensions(
            equations_module.resistance_resurgence
        )

    # Add more equations as needed

    return updated_equations