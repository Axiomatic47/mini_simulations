"""
Simple direct test for smooth transitions in the equations.
"""

import os
import sys
import numpy as np

# Add the project root to Python's path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Now try to import the equations from config directory
try:
    from config.equations import (
        suppression_feedback, 
        quantum_tunneling_probability,
        resistance_resurgence,
        wisdom_field
    )
    print("Successfully imported equations module from config directory!")
except ImportError as e:
    print(f"Error importing equations from config: {e}")
    
    # Create config directory if needed
    config_dir = os.path.join(current_dir, 'config')
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
        print(f"Created config directory at {config_dir}")
    
    # Check if equations.py exists in the root directory
    if os.path.exists(os.path.join(current_dir, 'equations.py')):
        print("Found equations.py in root directory, will copy to config/")
        import shutil
        shutil.copy(
            os.path.join(current_dir, 'equations.py'),
            os.path.join(config_dir, 'equations.py')
        )
        print("Copied equations.py to config directory")
    else:
        print("Could not find equations.py in root directory")
        sys.exit(1)
    
    # Create __init__.py in config directory
    with open(os.path.join(config_dir, '__init__.py'), 'w') as f:
        f.write("# Config package\n")
    
    # Try importing again
    try:
        from config.equations import (
            suppression_feedback, 
            quantum_tunneling_probability,
            resistance_resurgence,
            wisdom_field
        )
        print("Successfully imported equations after setup!")
    except ImportError as e:
        print(f"Still failed to import equations: {e}")
        sys.exit(1)

# Create a CircuitBreaker stub if needed
try:
    from utils.circuit_breaker import CircuitBreaker
    print("Successfully imported CircuitBreaker!")
except ImportError:
    print("Creating circuit_breaker stub...")
    
    # Create utils directory if needed
    utils_dir = os.path.join(current_dir, 'utils')
    if not os.path.exists(utils_dir):
        os.makedirs(utils_dir)
        print(f"Created utils directory at {utils_dir}")
    
    # Create __init__.py in utils directory
    with open(os.path.join(utils_dir, '__init__.py'), 'w') as f:
        f.write("# Utils package\n")
    
    # Create circuit_breaker.py with stub implementation
    with open(os.path.join(utils_dir, 'circuit_breaker.py'), 'w') as f:
        f.write("""
# Circuit breaker stub for testing
import numpy as np

class CircuitBreaker:
    \"\"\"
    A circuit breaker utility to detect and handle numerical instabilities.
    \"\"\"
    
    def __init__(self, threshold=1e-10, max_value=1e10, min_value=1e-10, max_rate_of_change=1e3):
        self.threshold = threshold
        self.max_value = max_value
        self.min_value = min_value
        self.max_rate_of_change = max_rate_of_change
        self.trigger_count = 0
        self.was_triggered = False
        self.last_trigger_reason = None
    
    def check_value_stability(self, value):
        \"\"\"Check if a value is stable.\"\"\"
        if not isinstance(value, (int, float)):
            return False
        
        if not np.isfinite(value):
            self.trigger_count += 1
            self.was_triggered = True
            self.last_trigger_reason = "Value is not finite"
            return False
        
        if abs(value) > self.max_value:
            self.trigger_count += 1
            self.was_triggered = True
            self.last_trigger_reason = "Value exceeds maximum"
            return False
        
        return True
    
    def check_and_fix(self, value, min_val=None, max_val=None):
        \"\"\"Check a value and fix it if necessary.\"\"\"
        if min_val is None:
            min_val = -self.max_value
        
        if max_val is None:
            max_val = self.max_value
        
        if not self.check_value_stability(value):
            value = 0.0  # Default value for unstable values
        
        # Apply bounds
        value = max(min_val, min(max_val, value))
        
        return value
    
    def safe_div(self, numerator, denominator, default=0.0):
        \"\"\"Safe division that handles division by zero.\"\"\"
        if abs(denominator) < self.threshold:
            self.trigger_count += 1
            self.was_triggered = True
            self.last_trigger_reason = "Division by very small value"
            return default
        
        return numerator / denominator
    
    def safe_exp(self, value, max_exp=100.0):
        \"\"\"Safe exponential that prevents overflow.\"\"\"
        if value > max_exp:
            self.trigger_count += 1
            self.was_triggered = True
            self.last_trigger_reason = "Exponential overflow prevented"
            return self.max_value
        
        return np.exp(value)
    
    def safe_sqrt(self, value, default=0.0):
        \"\"\"Safe square root that handles negative values.\"\"\"
        if value < 0:
            self.trigger_count += 1
            self.was_triggered = True
            self.last_trigger_reason = "Square root of negative value"
            return default
        
        return np.sqrt(value)
    
    def safe_log(self, value, default=0.0):
        \"\"\"Safe logarithm that handles non-positive values.\"\"\"
        if value <= 0:
            self.trigger_count += 1
            self.was_triggered = True
            self.last_trigger_reason = "Logarithm of non-positive value"
            return default
        
        return np.log(value)
""")
    print("Created CircuitBreaker stub in utils/circuit_breaker.py")
    
    # Import the new stub
    try:
        from utils.circuit_breaker import CircuitBreaker
        print("Successfully imported CircuitBreaker stub!")
    except ImportError as e:
        print(f"Failed to import CircuitBreaker stub: {e}")
        sys.exit(1)

def test_suppression_feedback():
    """Test smooth transitions in suppression_feedback."""
    print("\n===== Testing suppression_feedback smooth transitions =====")
    # Test specific points in the transition zone
    testpoints = [19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0]
    results = [suppression_feedback(0.1, 10.0, 0.2, K) for K in testpoints]
    
    # Print results to see the transition
    for k, r in zip(testpoints, results):
        print(f"K = {k:.1f} -> feedback = {r:.4f}")
    
    # Check if the transition is smooth
    is_smooth = all(results[i] < results[i-1] for i in range(1, len(results)))
    print(f"Transition is smooth: {is_smooth}")
    return is_smooth

def test_quantum_tunneling():
    """Test smooth transitions in quantum_tunneling_probability."""
    print("\n===== Testing quantum_tunneling_probability smooth transitions =====")
    # Test transition near barrier
    barrier_height = 10.0
    barrier_width = 1.0
    
    # Points near the barrier height
    testpoints = [9.0, 9.3, 9.5, 9.7, 9.9, 10.0]
    results = [quantum_tunneling_probability(barrier_height, barrier_width, E) for E in testpoints]
    
    # Print results to see the transition
    for e, p in zip(testpoints, results):
        print(f"Energy = {e:.1f} -> probability = {p:.4f}")
    
    # Check if the transition is smooth (increasing)
    is_smooth = all(results[i] > results[i-1] for i in range(1, len(results)))
    print(f"Transition is smooth: {is_smooth}")
    return is_smooth

def test_resistance_resurgence():
    """Test smooth transitions in resistance_resurgence."""
    print("\n===== Testing resistance_resurgence smooth transitions =====")
    # Test transition near critical time
    S_0 = 10.0
    lambda_decay = 0.1
    t_crit = 20.0
    alpha_resurge = 5.0
    mu_resurge = 0.05
    
    # Test points before and after critical time
    testpoints = [15.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0]
    results = [resistance_resurgence(S_0, lambda_decay, t, alpha_resurge, mu_resurge, t_crit) 
               for t in testpoints]
    
    # Print results to see the transition
    for t, r in zip(testpoints, results):
        print(f"Time = {t:.1f} -> resurgence = {r:.4f}")
    
    # Calculate expected decay-only values
    decay_only = [S_0 * np.exp(-lambda_decay * t) for t in testpoints]
    
    # Print comparison to show the pre-critical ramp-up
    for t, actual, decay in zip(testpoints[:4], results[:4], decay_only[:4]):
        diff = actual - decay
        print(f"Time = {t:.1f} -> Difference from decay-only: {diff:.6f}")
    
    # Pre-critical gradual onset should show non-zero positive differences
    pre_critical_differences = [results[i] - decay_only[i] for i in range(4)]
    has_pre_critical_onset = any(diff > 1e-6 for diff in pre_critical_differences)
    print(f"Has pre-critical onset: {has_pre_critical_onset}")
    
    # Check if there's a smooth transition around critical time
    critical_transition = results[4] > results[3]
    print(f"Smooth transition at critical time: {critical_transition}")
    
    return has_pre_critical_onset or critical_transition

def test_wisdom_field():
    """Test smooth transitions in wisdom_field."""
    print("\n===== Testing wisdom_field smooth transitions =====")
    W_0 = 1.0
    alpha = 0.1
    R = 1.0
    K = 1.0
    
    # Test points around the high suppression threshold
    testpoints = [20.0, 24.0, 25.0, 26.0, 30.0, 40.0]
    results = [wisdom_field(W_0, alpha, S, R, K) for S in testpoints]
    
    # Print results to see the transition
    for s, w in zip(testpoints, results):
        print(f"Suppression = {s:.1f} -> wisdom = {w:.6f}")
    
    # Calculate the decay rates without transition
    no_transition = [W_0 * np.exp(-alpha * S) * (1 + R/K) for S in testpoints]
    
    # Compare the differences
    print("\nComparing with vs. without smooth transition:")
    for s, with_trans, no_trans in zip(testpoints, results, no_transition):
        diff = with_trans - no_trans
        ratio = with_trans / no_trans if no_trans != 0 else float('inf')
        print(f"S = {s:.1f}: With transition = {with_trans:.6f}, Without = {no_trans:.6f}, Ratio = {ratio:.4f}")
    
    # The dampening should be more noticeable at higher suppression values
    # Let's check the ratio of wisdom at S=40 vs S=20 with and without transition
    with_trans_ratio = results[5] / results[0]
    no_trans_ratio = no_transition[5] / no_transition[0]
    
    # With smooth transition, the ratio should be higher (less extreme decay)
    is_dampened = with_trans_ratio > no_trans_ratio
    print(f"High suppression dampening effect: {is_dampened}")
    print(f"With transition ratio (S=40/S=20): {with_trans_ratio:.6f}")
    print(f"Without transition ratio (S=40/S=20): {no_trans_ratio:.6f}")
    
    return is_dampened

def run_all_tests():
    """Run all tests and report results."""
    print("RUNNING SMOOTH TRANSITION TESTS")
    print("===============================")
    
    results = {
        "Suppression Feedback": test_suppression_feedback(),
        "Quantum Tunneling": test_quantum_tunneling(),
        "Resistance Resurgence": test_resistance_resurgence(),
        "Wisdom Field": test_wisdom_field()
    }
    
    print("\n===============================")
    print("SUMMARY OF RESULTS:")
    all_passed = True
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{name}: {status}")
        all_passed = all_passed and passed
    
    print(f"\nOVERALL RESULT: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed

if __name__ == "__main__":
    run_all_tests()
