"""
Simple direct test for smooth transitions in the equations.
"""

import numpy as np
from equations import (
    suppression_feedback, 
    quantum_tunneling_probability,
    resistance_resurgence,
    wisdom_field
)

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
    has_pre_critical_onset = all(diff > 0 for diff in pre_critical_differences)
    print(f"Has pre-critical onset: {has_pre_critical_onset}")
    
    # Check if there's a smooth transition around critical time
    critical_transition = results[4] > results[3]
    print(f"Smooth transition at critical time: {critical_transition}")
    
    return has_pre_critical_onset and critical_transition

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
        print(f"{name}: {'PASSED' if passed else 'FAILED'}")
        all_passed = all_passed and passed
    
    print("\nOVERALL RESULT: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed

if __name__ == "__main__":
    run_all_tests()
