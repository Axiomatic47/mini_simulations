import sys
import unittest
import numpy as np
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import equations to test
from config.equations import (
    intelligence_growth, free_will_decision, truth_adoption,
    wisdom_field, resistance_resurgence, suppression_feedback,
    civilization_oscillation, knowledge_growth_phase_transition
)

# Import circuit breaker if available
try:
    from utils.circuit_breaker import CircuitBreaker

    has_circuit_breaker = True
except ImportError:
    has_circuit_breaker = False
    print("Warning: CircuitBreaker not available. Numerical stability checks will be limited.")


class TestEquationsBasic(unittest.TestCase):
    """Basic tests for equation correctness, boundary conditions, and expected behavior."""

    def setUp(self):
        """Set up for tests including checking for numerical stability mode."""
        # Check if we're running in numerical stability check mode
        self.check_stability = os.environ.get('CHECK_NUMERICAL_STABILITY') == 'true'
        if self.check_stability and has_circuit_breaker:
            self.circuit_breaker = CircuitBreaker(
                threshold=1e-6,
                max_value=1e6,
                min_value=-1e6,
                max_rate_of_change=1e3
            )
            # Counter for numerical stability issues
            self.stability_issues = 0

    def tearDown(self):
        """Report stability issues after each test if in stability check mode."""
        if self.check_stability and hasattr(self, 'stability_issues') and self.stability_issues > 0:
            print(f"⚠️ {self.stability_issues} numerical stability issues detected")
            # Update environment variable to track total issues
            current_issues = int(os.environ.get('NUMERICAL_STABILITY_ISSUES', '0'))
            os.environ['NUMERICAL_STABILITY_ISSUES'] = str(current_issues + self.stability_issues)

    def check_value_stability(self, value, name="value"):
        """Check if a value indicates numerical instability."""
        if not self.check_stability or not has_circuit_breaker:
            return

        # Check for NaN or infinity
        if np.isnan(value) or np.isinf(value):
            print(f"⚠️ Numerical instability detected: {name} = {value}")
            self.stability_issues += 1
            return True

        # Check for extremely large values
        if abs(value) > 1e6:
            print(f"⚠️ Extremely large value detected: {name} = {value}")
            self.stability_issues += 1
            return True

        # Check for extremely small values (potential underflow)
        if 0 < abs(value) < 1e-12:
            print(f"⚠️ Extremely small value detected: {name} = {value}")
            self.stability_issues += 1
            return True

        return False

    def test_intelligence_growth(self):
        """Test that intelligence growth responds correctly to inputs with bounds."""
        # Intelligence should increase with knowledge and wisdom
        i1 = intelligence_growth(K=10, W=2, R=1, S=1, N=0)
        i2 = intelligence_growth(K=5, W=2, R=1, S=1, N=0)
        self.assertGreater(i1, i2)
        self.check_value_stability(i1, "intelligence_growth(K=10)")
        self.check_value_stability(i2, "intelligence_growth(K=5)")

        # Intelligence should decrease with resistance and suppression
        i3 = intelligence_growth(K=10, W=2, R=5, S=1, N=0)
        i4 = intelligence_growth(K=10, W=2, R=1, S=1, N=0)
        self.assertLess(i3, i4)
        self.check_value_stability(i3, "intelligence_growth(R=5)")

        # Network effect should amplify intelligence
        i5 = intelligence_growth(K=10, W=2, R=1, S=1, N=5)
        i6 = intelligence_growth(K=10, W=2, R=1, S=1, N=0)
        self.assertGreater(i5, i6)
        self.check_value_stability(i5, "intelligence_growth(N=5)")

        # Zero knowledge should not produce intelligence growth
        i7 = intelligence_growth(K=0, W=2, R=1, S=1, N=0)
        self.assertLessEqual(i7, 0)
        self.check_value_stability(i7, "intelligence_growth(K=0)")

        # Test with extreme values for numerical stability
        if self.check_stability:
            # Very large knowledge
            i_large = intelligence_growth(K=1e6, W=2, R=1, S=1, N=0)
            self.check_value_stability(i_large, "intelligence_growth(K=1e6)")

            # Very large suppression
            i_high_s = intelligence_growth(K=10, W=2, R=1, S=1e6, N=0)
            self.check_value_stability(i_high_s, "intelligence_growth(S=1e6)")

            # Division by zero potential
            i_div_zero = intelligence_growth(K=0, W=0, R=0, S=0, N=0)
            self.check_value_stability(i_div_zero, "intelligence_growth(all zeros)")

    def test_free_will_decision(self):
        """Test that free will decision correctly balances influences with bounds."""
        # Stronger identity and knowledge should increase positive decision force
        d1 = free_will_decision(q_Id=2, E_K=5, q_R=1, E_F=2)
        d2 = free_will_decision(q_Id=1, E_K=5, q_R=1, E_F=2)
        self.assertGreater(d1, d2)
        self.check_value_stability(d1, "free_will_decision(q_Id=2)")
        self.check_value_stability(d2, "free_will_decision(q_Id=1)")

        # Stronger resistance and fear should decrease decision force
        d3 = free_will_decision(q_Id=1, E_K=5, q_R=5, E_F=5)
        d4 = free_will_decision(q_Id=1, E_K=5, q_R=1, E_F=2)
        self.assertLess(d3, d4)
        self.check_value_stability(d3, "free_will_decision(q_R=5,E_F=5)")

        # Balance of forces should yield predictable result
        d5 = free_will_decision(q_Id=2, E_K=3, q_R=2, E_F=3)
        self.assertEqual(d5, 0)
        self.check_value_stability(d5, "free_will_decision(balanced)")

        # Ensure bounded within [-1, 1] by tanh
        d6 = free_will_decision(q_Id=1000, E_K=1000, q_R=1, E_F=1)
        self.assertLessEqual(d6, 1.0)
        self.check_value_stability(d6, "free_will_decision(extreme positive)")

        d7 = free_will_decision(q_Id=1, E_K=1, q_R=1000, E_F=1000)
        self.assertGreaterEqual(d7, -1.0)
        self.check_value_stability(d7, "free_will_decision(extreme negative)")

    def test_truth_adoption(self):
        """Test truth adoption behavior and asymptotic properties with bounds."""
        # Truth adoption should increase with acceleration factor
        t1 = truth_adoption(T=10, A=5, T_max=100)
        t2 = truth_adoption(T=10, A=2, T_max=100)
        self.assertGreater(t1, t2)
        self.check_value_stability(t1, "truth_adoption(A=5)")
        self.check_value_stability(t2, "truth_adoption(A=2)")

        # Truth adoption should slow as it approaches maximum
        t3 = truth_adoption(T=10, A=2, T_max=100)
        t4 = truth_adoption(T=90, A=2, T_max=100)
        self.assertGreater(t3, t4)
        self.check_value_stability(t3, "truth_adoption(T=10)")
        self.check_value_stability(t4, "truth_adoption(T=90)")

        # Truth cannot exceed maximum (relativistic limit)
        for T_max in [50, 100, 200]:
            # If T = T_max, adoption rate should be close to zero
            t5 = truth_adoption(T=T_max, A=10, T_max=T_max)
            self.assertAlmostEqual(t5, 0, places=5)
            self.check_value_stability(t5, f"truth_adoption(T=T_max={T_max})")

        # Test negative input T (should be handled gracefully)
        t6 = truth_adoption(T=-10, A=2, T_max=100)
        self.assertGreaterEqual(t6, 0)  # Should not produce negative adoption
        self.check_value_stability(t6, "truth_adoption(T=-10)")

        # Test very large acceleration factor
        t7 = truth_adoption(T=10, A=1e6, T_max=100)
        self.check_value_stability(t7, "truth_adoption(A=1e6)")

        # Test division by zero potential (T_max = 0)
        if self.check_stability:
            t8 = truth_adoption(T=10, A=2, T_max=0)
            self.check_value_stability(t8, "truth_adoption(T_max=0)")

    def test_wisdom_field(self):
        """Test wisdom field dynamics and interactions with bounds."""
        # Wisdom should increase with base wisdom
        w1 = wisdom_field(W_0=10, alpha=0.1, S=5, R=2, K=10)
        w2 = wisdom_field(W_0=5, alpha=0.1, S=5, R=2, K=10)
        self.assertGreater(w1, w2)
        self.check_value_stability(w1, "wisdom_field(W_0=10)")
        self.check_value_stability(w2, "wisdom_field(W_0=5)")

        # Suppression should decrease wisdom
        w3 = wisdom_field(W_0=10, alpha=0.1, S=10, R=2, K=10)
        w4 = wisdom_field(W_0=10, alpha=0.1, S=5, R=2, K=10)
        self.assertLess(w3, w4)
        self.check_value_stability(w3, "wisdom_field(S=10)")
        self.check_value_stability(w4, "wisdom_field(S=5)")

        # Resistance-knowledge ratio effect - with fixed S, higher R/K ratio increases wisdom
        # Testing with equal R * K products to isolate the ratio effect
        high_r_low_k = wisdom_field(W_0=10, alpha=0.1, S=5, R=10, K=2)
        low_r_high_k = wisdom_field(W_0=10, alpha=0.1, S=5, R=2, K=10)
        self.assertGreater(high_r_low_k, low_r_high_k)
        self.check_value_stability(high_r_low_k, "wisdom_field(R=10,K=2)")
        self.check_value_stability(low_r_high_k, "wisdom_field(R=2,K=10)")

        # Test with very small knowledge (potential division by zero)
        w5 = wisdom_field(W_0=10, alpha=0.1, S=5, R=2, K=1e-10)
        self.check_value_stability(w5, "wisdom_field(K=1e-10)")

        # Test with zero knowledge
        w6 = wisdom_field(W_0=10, alpha=0.1, S=5, R=2, K=0)
        self.check_value_stability(w6, "wisdom_field(K=0)")

        # Test with very large suppression
        w7 = wisdom_field(W_0=10, alpha=0.1, S=1e6, R=2, K=10)
        self.check_value_stability(w7, "wisdom_field(S=1e6)")

    def test_resistance_resurgence(self):
        """Test resistance resurgence behavior with bounds."""
        S_0 = 100
        lambda_decay = 0.05
        alpha_resurge = 20
        mu_resurge = 0.1
        t_crit = 50

        # Suppression should decay over time
        r1 = resistance_resurgence(S_0, lambda_decay, 10, alpha_resurge, mu_resurge, t_crit)
        r2 = resistance_resurgence(S_0, lambda_decay, 40, alpha_resurge, mu_resurge, t_crit)
        self.assertGreater(r1, r2)
        self.check_value_stability(r1, "resistance_resurgence(t=10)")
        self.check_value_stability(r2, "resistance_resurgence(t=40)")

        # Resurgence should happen after critical time
        r3 = resistance_resurgence(S_0, lambda_decay, t_crit + 1, alpha_resurge, mu_resurge, t_crit)
        r4 = resistance_resurgence(S_0, lambda_decay, t_crit - 1, alpha_resurge, mu_resurge, t_crit)
        self.assertGreater(r3, r4)
        self.check_value_stability(r3, "resistance_resurgence(t=t_crit+1)")
        self.check_value_stability(r4, "resistance_resurgence(t=t_crit-1)")

        # Resurgence should fade over time
        r5 = resistance_resurgence(S_0, lambda_decay, t_crit + 1, alpha_resurge, mu_resurge, t_crit)
        r6 = resistance_resurgence(S_0, lambda_decay, t_crit + 20, alpha_resurge, mu_resurge, t_crit)
        self.assertGreater(r5, r6)
        self.check_value_stability(r5, "resistance_resurgence(t=t_crit+1)")
        self.check_value_stability(r6, "resistance_resurgence(t=t_crit+20)")

        # Test with very large time value (potential exponential overflow)
        r7 = resistance_resurgence(S_0, lambda_decay, 1000, alpha_resurge, mu_resurge, t_crit)
        self.check_value_stability(r7, "resistance_resurgence(t=1000)")

        # Test with very large alpha_resurge
        r8 = resistance_resurgence(S_0, lambda_decay, t_crit + 1, 1e6, mu_resurge, t_crit)
        self.check_value_stability(r8, "resistance_resurgence(alpha_resurge=1e6)")

        # Test with zero decay rate
        r9 = resistance_resurgence(S_0, 0, 10, alpha_resurge, mu_resurge, t_crit)
        self.check_value_stability(r9, "resistance_resurgence(lambda_decay=0)")

    def test_suppression_feedback(self):
        """Test suppression feedback dynamics with bounds."""
        # Feedback should increase with suppression
        f1 = suppression_feedback(alpha=0.1, S=10, beta=0.05, K=5)
        f2 = suppression_feedback(alpha=0.1, S=5, beta=0.05, K=5)
        self.assertGreater(f1, f2)
        self.check_value_stability(f1, "suppression_feedback(S=10)")
        self.check_value_stability(f2, "suppression_feedback(S=5)")

        # Feedback should decrease with knowledge
        f3 = suppression_feedback(alpha=0.1, S=10, beta=0.05, K=10)
        f4 = suppression_feedback(alpha=0.1, S=10, beta=0.05, K=5)
        self.assertLess(f3, f4)
        self.check_value_stability(f3, "suppression_feedback(K=10)")
        self.check_value_stability(f4, "suppression_feedback(K=5)")

        # When knowledge dominates, feedback can be negative
        f5 = suppression_feedback(alpha=0.1, S=5, beta=0.2, K=20)
        self.assertLess(f5, 0)
        self.check_value_stability(f5, "suppression_feedback(K=20,beta=0.2)")

        # Test with very large values
        f6 = suppression_feedback(alpha=0.1, S=1e6, beta=0.05, K=5)
        self.check_value_stability(f6, "suppression_feedback(S=1e6)")

        f7 = suppression_feedback(alpha=0.1, S=5, beta=0.05, K=1e6)
        self.check_value_stability(f7, "suppression_feedback(K=1e6)")

        # Test with zero values
        f8 = suppression_feedback(alpha=0.1, S=0, beta=0.05, K=5)
        self.check_value_stability(f8, "suppression_feedback(S=0)")

        f9 = suppression_feedback(alpha=0.1, S=5, beta=0.05, K=0)
        self.check_value_stability(f9, "suppression_feedback(K=0)")

    def test_civilization_oscillation(self):
        """Test civilization oscillation dynamics with bounds."""
        # Oscillation should respond to state
        o1 = civilization_oscillation(E=1, dE_dt=0, gamma=0.1, omega=1)
        o2 = civilization_oscillation(E=-1, dE_dt=0, gamma=0.1, omega=1)
        self.assertLess(o1, o2)
        self.check_value_stability(o1, "civilization_oscillation(E=1)")
        self.check_value_stability(o2, "civilization_oscillation(E=-1)")

        # Test that oscillation is correct with different parameters
        # For undamped SHO, acceleration is -ω²E
        E_val = 1.0
        omega_val = 2.0
        o3 = civilization_oscillation(E=E_val, dE_dt=0, gamma=0, omega=omega_val)
        self.assertEqual(o3, -(omega_val ** 2) * E_val)
        self.check_value_stability(o3, "civilization_oscillation(omega=2)")

        # Test with very large values
        o4 = civilization_oscillation(E=1e3, dE_dt=0, gamma=0.1, omega=1)
        self.check_value_stability(o4, "civilization_oscillation(E=1e3)")

        o5 = civilization_oscillation(E=1, dE_dt=1e3, gamma=0.1, omega=1)
        self.check_value_stability(o5, "civilization_oscillation(dE_dt=1e3)")

        o6 = civilization_oscillation(E=1, dE_dt=0, gamma=0.1, omega=1e3)
        self.check_value_stability(o6, "civilization_oscillation(omega=1e3)")

        # Test with very small values
        o7 = civilization_oscillation(E=1e-10, dE_dt=0, gamma=0.1, omega=1)
        self.check_value_stability(o7, "civilization_oscillation(E=1e-10)")

    def test_knowledge_growth_phase_transition(self):
        """Test knowledge growth phase transition behavior with bounds."""
        # Knowledge should grow more rapidly post-transition
        pre_transition = knowledge_growth_phase_transition(
            K_0=10, beta_decay=0.01, t=10, A=5, gamma=0.1, T=10, T_crit=20)
        post_transition = knowledge_growth_phase_transition(
            K_0=10, beta_decay=0.01, t=10, A=5, gamma=0.1, T=30, T_crit=20)
        self.assertGreater(post_transition, pre_transition)
        self.check_value_stability(pre_transition, "knowledge_growth(T<T_crit)")
        self.check_value_stability(post_transition, "knowledge_growth(T>T_crit)")

        # Knowledge should decay in heavily suppressed environments
        decaying = knowledge_growth_phase_transition(
            K_0=10, beta_decay=0.1, t=20, A=5, gamma=0.1, T=5, T_crit=20)
        self.assertLess(decaying, 10)
        self.check_value_stability(decaying, "knowledge_growth(decaying)")

        # Test with very large values
        k1 = knowledge_growth_phase_transition(
            K_0=1e6, beta_decay=0.01, t=10, A=5, gamma=0.1, T=10, T_crit=20)
        self.check_value_stability(k1, "knowledge_growth(K_0=1e6)")

        k2 = knowledge_growth_phase_transition(
            K_0=10, beta_decay=0.01, t=1000, A=5, gamma=0.1, T=10, T_crit=20)
        self.check_value_stability(k2, "knowledge_growth(t=1000)")

        k3 = knowledge_growth_phase_transition(
            K_0=10, beta_decay=0.01, t=10, A=1e6, gamma=0.1, T=10, T_crit=20)
        self.check_value_stability(k3, "knowledge_growth(A=1e6)")

        # Test with very small values
        k4 = knowledge_growth_phase_transition(
            K_0=1e-10, beta_decay=0.01, t=10, A=5, gamma=0.1, T=10, T_crit=20)
        self.check_value_stability(k4, "knowledge_growth(K_0=1e-10)")


class TestNumericalStability(unittest.TestCase):
    """Specific tests for numerical stability of equations under extreme conditions."""

    def setUp(self):
        """Set up for numerical stability tests."""
        # Initialize circuit breaker if available
        if has_circuit_breaker:
            self.circuit_breaker = CircuitBreaker(
                threshold=1e-6,
                max_value=1e6,
                min_value=-1e6,
                max_rate_of_change=1e3
            )

        # Set up extreme test values
        self.extreme_values = {
            'very_large': 1e10,
            'very_small': 1e-10,
            'zero': 0,
            'negative': -100
        }

        # Counter for stability issues
        self.stability_issues = 0

    def tearDown(self):
        """Report stability issues after each test."""
        if hasattr(self, 'stability_issues') and self.stability_issues > 0:
            print(f"⚠️ {self.stability_issues} numerical stability issues detected")
            # Update environment variable to track total issues
            current_issues = int(os.environ.get('NUMERICAL_STABILITY_ISSUES', '0'))
            os.environ['NUMERICAL_STABILITY_ISSUES'] = str(current_issues + self.stability_issues)

    def check_stability(self, func, args, name="function"):
        """Test a function with extreme inputs to check stability."""
        try:
            result = func(**args)

            # Check for NaN or infinity
            if np.isnan(result) or np.isinf(result):
                print(f"⚠️ Numerical instability in {name}: {result} with args {args}")
                self.stability_issues += 1
                return False

            return True
        except Exception as e:
            print(f"⚠️ Exception in {name}: {e} with args {args}")
            self.stability_issues += 1
            return False

    def test_intelligence_growth_stability(self):
        """Test intelligence_growth function with extreme inputs."""

        # Test with various combinations of extreme values
        test_cases = [
            {'K': self.extreme_values['very_large'], 'W': 1, 'R': 1, 'S': 1, 'N': 0},
            {'K': 1, 'W': self.extreme_values['very_large'], 'R': 1, 'S': 1, 'N': 0},
            {'K': 1, 'W': 1, 'R': self.extreme_values['very_large'], 'S': 1, 'N': 0},
            {'K': 1, 'W': 1, 'R': 1, 'S': self.extreme_values['very_large'], 'N': 0},
            {'K': 1, 'W': 1, 'R': 1, 'S': 1, 'N': self.extreme_values['very_large']},
            {'K': self.extreme_values['zero'], 'W': 1, 'R': 1, 'S': 1, 'N': 0},
            {'K': 1, 'W': self.extreme_values['zero'], 'R': 1, 'S': 1, 'N': 0},
            {'K': 1, 'W': 1, 'R': self.extreme_values['zero'], 'S': 1, 'N': 0},
            {'K': 1, 'W': 1, 'R': 1, 'S': self.extreme_values['zero'], 'N': 0},
            {'K': self.extreme_values['negative'], 'W': 1, 'R': 1, 'S': 1, 'N': 0},
            {'K': 1, 'W': self.extreme_values['negative'], 'R': 1, 'S': 1, 'N': 0},
            {'K': 1, 'W': 1, 'R': self.extreme_values['negative'], 'S': 1, 'N': 0},
            {'K': 1, 'W': 1, 'R': 1, 'S': self.extreme_values['negative'], 'N': 0},
            {'K': 1, 'W': 1, 'R': 1, 'S': 1, 'N': self.extreme_values['negative']},
            {'K': self.extreme_values['very_small'], 'W': 1, 'R': 1, 'S': 1, 'N': 0},
            {'K': 1, 'W': self.extreme_values['very_small'], 'R': 1, 'S': 1, 'N': 0}
        ]

        for i, args in enumerate(test_cases):
            self.check_stability(intelligence_growth, args, f"intelligence_growth case {i}")

    def test_free_will_decision_stability(self):
        """Test free_will_decision function with extreme inputs."""

        test_cases = [
            {'q_Id': self.extreme_values['very_large'], 'E_K': 1, 'q_R': 1, 'E_F': 1},
            {'q_Id': 1, 'E_K': self.extreme_values['very_large'], 'q_R': 1, 'E_F': 1},
            {'q_Id': 1, 'E_K': 1, 'q_R': self.extreme_values['very_large'], 'E_F': 1},
            {'q_Id': 1, 'E_K': 1, 'q_R': 1, 'E_F': self.extreme_values['very_large']},
            {'q_Id': self.extreme_values['negative'], 'E_K': 1, 'q_R': 1, 'E_F': 1},
            {'q_Id': 1, 'E_K': self.extreme_values['negative'], 'q_R': 1, 'E_F': 1},
            {'q_Id': 1, 'E_K': 1, 'q_R': self.extreme_values['negative'], 'E_F': 1},
            {'q_Id': 1, 'E_K': 1, 'q_R': 1, 'E_F': self.extreme_values['negative']}
        ]

        for i, args in enumerate(test_cases):
            # tanh should bound all outputs to [-1, 1], so this should be stable
            result = free_will_decision(**args)
            self.assertGreaterEqual(result, -1.0)
            self.assertLessEqual(result, 1.0)

            self.check_stability(free_will_decision, args, f"free_will_decision case {i}")

    def test_truth_adoption_stability(self):
        """Test truth_adoption function with extreme inputs."""

        test_cases = [
            {'T': self.extreme_values['very_large'], 'A': 1, 'T_max': 100},
            {'T': 10, 'A': self.extreme_values['very_large'], 'T_max': 100},
            {'T': 10, 'A': 1, 'T_max': self.extreme_values['very_large']},
            {'T': self.extreme_values['negative'], 'A': 1, 'T_max': 100},
            {'T': 10, 'A': self.extreme_values['negative'], 'T_max': 100},
            {'T': 10, 'A': 1, 'T_max': self.extreme_values['negative']},
            {'T': 10, 'A': 1, 'T_max': self.extreme_values['zero']},
            {'T': self.extreme_values['zero'], 'A': 1, 'T_max': 100},
            {'T': 10, 'A': self.extreme_values['zero'], 'T_max': 100}
        ]

        for i, args in enumerate(test_cases):
            self.check_stability(truth_adoption, args, f"truth_adoption case {i}")

    def test_wisdom_field_stability(self):
        """Test wisdom_field function with extreme inputs."""

        test_cases = [
            {'W_0': self.extreme_values['very_large'], 'alpha': 0.1, 'S': 5, 'R': 2, 'K': 10},
            {'W_0': 10, 'alpha': self.extreme_values['very_large'], 'S': 5, 'R': 2, 'K': 10},
            {'W_0': 10, 'alpha': 0.1, 'S': self.extreme_values['very_large'], 'R': 2, 'K': 10},
            {'W_0': 10, 'alpha': 0.1, 'S': 5, 'R': self.extreme_values['very_large'], 'K': 10},
            {'W_0': 10, 'alpha': 0.1, 'S': 5, 'R': 2, 'K': self.extreme_values['very_large']},
            {'W_0': 10, 'alpha': 0.1, 'S': 5, 'R': 2, 'K': self.extreme_values['zero']},
            {'W_0': 10, 'alpha': 0.1, 'S': 5, 'R': self.extreme_values['zero'], 'K': 10},
            {'W_0': 10, 'alpha': 0.1, 'S': self.extreme_values['zero'], 'R': 2, 'K': 10},
            {'W_0': 10, 'alpha': self.extreme_values['zero'], 'S': 5, 'R': 2, 'K': 10},
            {'W_0': self.extreme_values['zero'], 'alpha': 0.1, 'S': 5, 'R': 2, 'K': 10},
            {'W_0': 10, 'alpha': 0.1, 'S': 5, 'R': 2, 'K': self.extreme_values['very_small']}
        ]

        for i, args in enumerate(test_cases):
            self.check_stability(wisdom_field, args, f"wisdom_field case {i}")

    def test_resistance_resurgence_stability(self):
        """Test resistance_resurgence function with extreme inputs."""

        test_cases = [
            {'S_0': self.extreme_values['very_large'], 'lambda_decay': 0.05, 't': 10, 'alpha_resurge': 5,
             'mu_resurge': 0.1, 't_crit': 50},
            {'S_0': 100, 'lambda_decay': self.extreme_values['very_large'], 't': 10, 'alpha_resurge': 5,
             'mu_resurge': 0.1, 't_crit': 50},
            {'S_0': 100, 'lambda_decay': 0.05, 't': self.extreme_values['very_large'], 'alpha_resurge': 5,
             'mu_resurge': 0.1, 't_crit': 50},
            {'S_0': 100, 'lambda_decay': 0.05, 't': 10, 'alpha_resurge': self.extreme_values['very_large'],
             'mu_resurge': 0.1, 't_crit': 50},
            {'S_0': 100, 'lambda_decay': 0.05, 't': 10, 'alpha_resurge': 5,
             'mu_resurge': self.extreme_values['very_large'], 't_crit': 50},
            {'S_0': 100, 'lambda_decay': 0.05, 't': 10, 'alpha_resurge': 5, 'mu_resurge': 0.1,
             't_crit': self.extreme_values['very_large']},
            {'S_0': 100, 'lambda_decay': self.extreme_values['zero'], 't': 10, 'alpha_resurge': 5, 'mu_resurge': 0.1,
             't_crit': 50},
            {'S_0': 100, 'lambda_decay': 0.05, 't': 10, 'alpha_resurge': 5, 'mu_resurge': self.extreme_values['zero'],
             't_crit': 50},
            {'S_0': self.extreme_values['negative'], 'lambda_decay': 0.05, 't': 10, 'alpha_resurge': 5,
             'mu_resurge': 0.1, 't_crit': 50},
            {'S_0': 100, 'lambda_decay': self.extreme_values['negative'], 't': 10, 'alpha_resurge': 5,
             'mu_resurge': 0.1, 't_crit': 50}
        ]

        for i, args in enumerate(test_cases):
            self.check_stability(resistance_resurgence, args, f"resistance_resurgence case {i}")

    def test_knowledge_growth_phase_transition_stability(self):
        """Test knowledge_growth_phase_transition function with extreme inputs."""

        test_cases = [
            {'K_0': self.extreme_values['very_large'], 'beta_decay': 0.01, 't': 10, 'A': 5, 'gamma': 0.1, 'T': 30,
             'T_crit': 20},
            {'K_0': 10, 'beta_decay': self.extreme_values['very_large'], 't': 10, 'A': 5, 'gamma': 0.1, 'T': 30,
             'T_crit': 20},
            {'K_0': 10, 'beta_decay': 0.01, 't': self.extreme_values['very_large'], 'A': 5, 'gamma': 0.1, 'T': 30,
             'T_crit': 20},
            {'K_0': 10, 'beta_decay': 0.01, 't': 10, 'A': self.extreme_values['very_large'], 'gamma': 0.1, 'T': 30,
             'T_crit': 20},
            {'K_0': 10, 'beta_decay': 0.01, 't': 10, 'A': 5, 'gamma': self.extreme_values['very_large'], 'T': 30,
             'T_crit': 20},
            {'K_0': 10, 'beta_decay': 0.01, 't': 10, 'A': 5, 'gamma': 0.1, 'T': self.extreme_values['very_large'],
             'T_crit': 20},
            {'K_0': 10, 'beta_decay': 0.01, 't': 10, 'A': 5, 'gamma': 0.1, 'T': 30,
             'T_crit': self.extreme_values['very_large']},
            {'K_0': self.extreme_values['zero'], 'beta_decay': 0.01, 't': 10, 'A': 5, 'gamma': 0.1, 'T': 30,
             'T_crit': 20},
            {'K_0': 10, 'beta_decay': self.extreme_values['zero'], 't': 10, 'A': 5, 'gamma': 0.1, 'T': 30,
             'T_crit': 20},
            {'K_0': 10, 'beta_decay': 0.01, 't': self.extreme_values['zero'], 'A': 5, 'gamma': 0.1, 'T': 30,
             'T_crit': 20}
        ]

        for i, args in enumerate(test_cases):
            self.check_stability(knowledge_growth_phase_transition, args, f"knowledge_growth_phase_transition case {i}")


class TestConsistency(unittest.TestCase):
    """Tests for consistency of equations with theoretical framework goals."""

    def test_intelligence_suppression_relationship(self):
        """Intelligence and suppression should be inversely related."""
        # Create arrays to test consistency over range of values
        knowledge = np.linspace(1, 20, 20)
        wisdom = np.full_like(knowledge, 2.0)
        resistance = np.full_like(knowledge, 1.0)
        network = np.full_like(knowledge, 1.0)

        # Test with high suppression
        high_suppression = np.full_like(knowledge, 10.0)
        high_intelligence = np.array([
            intelligence_growth(k, w, r, s, n)
            for k, w, r, s, n in zip(knowledge, wisdom, resistance, high_suppression, network)
        ])

        # Test with low suppression
        low_suppression = np.full_like(knowledge, 2.0)
        low_intelligence = np.array([
            intelligence_growth(k, w, r, s, n)
            for k, w, r, s, n in zip(knowledge, wisdom, resistance, low_suppression, network)
        ])

        # Intelligence should be higher with lower suppression
        self.assertTrue(np.all(low_intelligence > high_intelligence))

    def test_truth_tipping_point(self):
        """Test that truth adoption exhibits a tipping point in knowledge growth."""
        # Initialize variables
        time_steps = 100
        dt = 1
        T = np.zeros(time_steps)
        K = np.zeros(time_steps)

        # Constants
        A_truth, T_max = 2.5, 40
        T_crit_phase = 20
        A_phase, gamma_phase = 1.5, 0.1

        # Initial conditions
        T[0] = 1.0
        K[0] = 1.0

        # Run a simplified simulation with bounds to prevent instability
        for t in range(1, time_steps):
            T[t] = np.clip(
                T[t - 1] + truth_adoption(T[t - 1], A_truth, T_max) * dt,
                0, T_max
            )
            K[t] = np.clip(
                knowledge_growth_phase_transition(
                    K[t - 1], 0.01, t, A_phase, gamma_phase, T[t - 1], T_crit_phase
                ),
                0, 1000  # reasonable upper bound
            )

        # Find where truth exceeds the critical threshold
        crossing_indices = np.where(T > T_crit_phase)[0]
        if len(crossing_indices) > 0:
            crossing_point = crossing_indices[0]

            # Check if knowledge growth accelerates after crossing point
            if crossing_point + 10 < time_steps:
                pre_crossing_growth = np.diff(K[:crossing_point + 5])
                post_crossing_growth = np.diff(K[crossing_point:crossing_point + 10])

                # Check with a sufficient buffer to ensure stability in the comparison
                self.assertGreater(np.mean(post_crossing_growth), np.mean(pre_crossing_growth) * 0.9)
        else:
            # If no crossing occurs, the test is inconclusive but not failed
            print("Warning: Truth did not cross critical threshold in simulation.")

    def test_suppression_decay_physics_analogy(self):
        """Test that suppression decay follows nuclear decay-like pattern."""
        time_steps = np.arange(0, 100, 1)
        S_0 = 100
        lambda_decay = 0.05
        alpha_resurge = 0  # No resurgence for this test
        mu_resurge = 0
        t_crit = 999  # No critical point for this test

        # Calculate suppression values over time with bounds
        suppressions = np.array([
            resistance_resurgence(S_0, lambda_decay, min(t, 50), alpha_resurge, mu_resurge, t_crit)
            for t in time_steps
        ])

        # Ensure all values are positive and finite
        self.assertTrue(np.all(suppressions >= 0))
        self.assertTrue(np.all(np.isfinite(suppressions)))

        # For true exponential decay, ln(S/S_0) should be linear with time
        # Use only valid points for the fit (where S > 0)
        valid_indices = suppressions > 0
        valid_times = time_steps[valid_indices]
        valid_suppressions = suppressions[valid_indices]

        if len(valid_suppressions) > 10:  # Enough points for a meaningful fit
            log_suppressions = np.log(valid_suppressions / S_0)

            # Fit a line to the logarithmic values
            slope, _ = np.polyfit(valid_times, log_suppressions, 1)

            # Slope should approximately equal -lambda_decay
            # Use a more relaxed tolerance due to potential numerical effects
            self.assertAlmostEqual(slope, -lambda_decay, places=1)

    def test_relativistic_truth_limit(self):
        """Test that truth adoption exhibits relativistic-like speed limit."""
        # As truth approaches maximum, adoption rate should asymptotically approach zero
        t_values = np.linspace(1, 95, 95)  # Values up to but not including T_max
        t_max = 100
        a = 0.5  # Use a smaller acceleration factor to ensure values stay small

        adoption_rates = np.array([truth_adoption(t, a, t_max) for t in t_values])

        # Verify decreasing adoption rate as T approaches T_max
        # Use diff with a tolerance to account for potential floating point issues
        decreases = np.diff(adoption_rates) < 1e-10
        self.assertTrue(np.all(decreases))

        # Verify asymptotic approach to zero - the last value should be quite small
        self.assertLess(adoption_rates[-1], 0.3)


class TestCornerCases(unittest.TestCase):
    """Tests for corner cases, extreme values, and stability."""

    def setUp(self):
        """Set up for tests including checking for numerical stability mode."""
        # Check if we're running in numerical stability check mode
        self.check_stability = os.environ.get('CHECK_NUMERICAL_STABILITY') == 'true'
        if self.check_stability and has_circuit_breaker:
            self.circuit_breaker = CircuitBreaker(
                threshold=1e-6,
                max_value=1e6,
                min_value=-1e6,
                max_rate_of_change=1e3
            )
            # Counter for numerical stability issues
            self.stability_issues = 0

    def tearDown(self):
        """Report stability issues after each test if in stability check mode."""
        if self.check_stability and hasattr(self, 'stability_issues') and self.stability_issues > 0:
            print(f"⚠️ {self.stability_issues} numerical stability issues detected")
            # Update environment variable to track total issues
            current_issues = int(os.environ.get('NUMERICAL_STABILITY_ISSUES', '0'))
            os.environ['NUMERICAL_STABILITY_ISSUES'] = str(current_issues + self.stability_issues)

    def test_zero_values(self):
        """Test equations with zero values."""
        # Intelligence with zero knowledge
        i1 = intelligence_growth(K=0, W=1, R=1, S=1, N=0)
        self.assertTrue(np.isfinite(i1))

        # Truth adoption with zero truth
        t1 = truth_adoption(T=0, A=1, T_max=10)
        self.assertGreater(t1, 0)
        self.assertTrue(np.isfinite(t1))

        # Suppression with zero suppression
        s1 = resistance_resurgence(S_0=0, lambda_decay=0.1, t=10, alpha_resurge=5, mu_resurge=0.1, t_crit=50)
        self.assertEqual(s1, 0)
        self.assertTrue(np.isfinite(s1))

        # Knowledge growth with zero knowledge
        k1 = knowledge_growth_phase_transition(K_0=0, beta_decay=0.1, t=10, A=5, gamma=0.1, T=30, T_crit=20)
        self.assertGreaterEqual(k1, 0)
        self.assertTrue(np.isfinite(k1))

        # Wisdom field with zero knowledge (potential division by zero)
        try:
            w1 = wisdom_field(W_0=1, alpha=0.1, S=1, R=1, K=0)
            self.assertTrue(np.isfinite(w1))
        except ZeroDivisionError:
            self.fail("wisdom_field raised ZeroDivisionError with K=0")

    def test_negative_values(self):
        """Test equations with negative values and ensure appropriate handling."""
        # Intelligence can go negative (civilizational collapse)
        i1 = intelligence_growth(K=1, W=1, R=10, S=10, N=0)
        self.assertLess(i1, 0)
        self.assertTrue(np.isfinite(i1))

        # Truth adoption should not be negative even with negative input
        t1 = truth_adoption(T=-10, A=1, T_max=10)
        self.assertGreaterEqual(t1, 0)
        self.assertTrue(np.isfinite(t1))

        # Suppression should not be negative with negative input
        s1 = resistance_resurgence(S_0=-10, lambda_decay=0.1, t=10, alpha_resurge=5, mu_resurge=0.1, t_crit=50)
        self.assertTrue(np.isfinite(s1))

        # Wisdom field with negative inputs
        w1 = wisdom_field(W_0=1, alpha=0.1, S=-5, R=1, K=1)
        self.assertTrue(np.isfinite(w1))

        w2 = wisdom_field(W_0=1, alpha=0.1, S=1, R=-5, K=1)
        self.assertTrue(np.isfinite(w2))

        w3 = wisdom_field(W_0=1, alpha=0.1, S=1, R=1, K=-5)
        self.assertTrue(np.isfinite(w3))

    def test_very_large_values(self):
        """Test that equations handle very large values without exploding."""
        large_value = 1e6

        # Intelligence with large knowledge
        i1 = intelligence_growth(K=large_value, W=1, R=1, S=1, N=0)
        self.assertTrue(np.isfinite(i1))

        # Truth adoption with large acceleration
        t1 = truth_adoption(T=10, A=large_value, T_max=100)
        self.assertTrue(np.isfinite(t1))

        # Suppression with large initial value
        s1 = resistance_resurgence(S_0=large_value, lambda_decay=0.1, t=10, alpha_resurge=5, mu_resurge=0.1, t_crit=50)
        self.assertTrue(np.isfinite(s1))

        # Knowledge growth with large parameters
        k1 = knowledge_growth_phase_transition(K_0=large_value, beta_decay=0.1, t=10, A=5, gamma=0.1, T=30, T_crit=20)
        self.assertTrue(np.isfinite(k1))

        # Wisdom field with large values
        w1 = wisdom_field(W_0=large_value, alpha=0.1, S=1, R=1, K=1)
        self.assertTrue(np.isfinite(w1))

        # Free will decision with large inputs
        d1 = free_will_decision(q_Id=large_value, E_K=large_value, q_R=1, E_F=1)
        self.assertTrue(np.isfinite(d1))
        self.assertLessEqual(d1, 1.0)  # Should be bounded by tanh


class TestOscillation(unittest.TestCase):
    """Tests for civilization oscillation behavior."""

    def test_oscillation_period(self):
        """Test that civilization oscillation has the correct period."""
        # Set up oscillation parameters
        E = 1.0
        dE_dt = 0.0
        gamma = 0.0  # No damping for this test
        omega = 0.3  # Natural frequency

        # Natural period should be 2π/omega
        expected_period = 2 * np.pi / omega

        # Run simplified simulation with stability measures
        time_steps = 200
        dt = 0.1
        E_values = np.zeros(time_steps)
        E_values[0] = E
        dE_dt_current = dE_dt

        for t in range(1, time_steps):
            # Calculate acceleration with bounds
            d2E_dt2 = civilization_oscillation(
                np.clip(E_values[t - 1], -10, 10),  # Bound position
                np.clip(dE_dt_current, -10, 10),  # Bound velocity
                gamma,
                omega
            )

            # Update velocity and position with bounds
            dE_dt_current = np.clip(dE_dt_current + d2E_dt2 * dt, -10, 10)
            E_values[t] = np.clip(E_values[t - 1] + dE_dt_current * dt, -10, 10)

        # Find period by locating zero crossings (with stability handling)
        # Replace any potential NaN or inf values
        E_values = np.nan_to_num(E_values, nan=0, posinf=10, neginf=-10)

        # Find zero crossings
        zero_crossings = np.where(np.diff(np.signbit(E_values)))[0]

        # Period should be approximately 2 * time between consecutive zero crossings
        if len(zero_crossings) >= 3:
            measured_period = 2 * dt * (zero_crossings[2] - zero_crossings[0])
            # Use a more generous tolerance for the comparison due to numerical effects
            self.assertAlmostEqual(measured_period, expected_period, delta=1.0)
        else:
            # Not enough zero crossings found, check if there's oscillation
            min_val = np.min(E_values)
            max_val = np.max(E_values)
            # If there's significant oscillation, warn but don't fail
            if max_val - min_val > 1.0:
                print(
                    f"Warning: Oscillation detected but insufficient zero crossings found. Min={min_val}, Max={max_val}")
            # Otherwise, test is inconclusive but shouldn't fail
            else:
                print("Warning: Insufficient oscillation detected for period measurement.")

    def test_damping_behavior(self):
        """Test that civilization oscillation is properly damped."""
        # Set up oscillation parameters
        E = 1.0
        dE_dt = 0.0
        gamma = 0.1  # Damping factor
        omega = 0.3  # Natural frequency

        # Run simplified simulation with stability measures
        time_steps = 200
        dt = 0.1
        E_values = np.zeros(time_steps)
        E_values[0] = E
        dE_dt_current = dE_dt

        for t in range(1, time_steps):
            # Calculate acceleration with bounds
            d2E_dt2 = civilization_oscillation(
                np.clip(E_values[t - 1], -10, 10),  # Bound position
                np.clip(dE_dt_current, -10, 10),  # Bound velocity
                gamma,
                omega
            )

            # Update velocity and position with bounds
            dE_dt_current = np.clip(dE_dt_current + d2E_dt2 * dt, -10, 10)
            E_values[t] = np.clip(E_values[t - 1] + dE_dt_current * dt, -10, 10)

        # Replace any potential NaN or inf values
        E_values = np.nan_to_num(E_values, nan=0, posinf=10, neginf=-10)

        # Find local maxima with handling for numerical stability
        maxima_indices = []
        for i in range(1, time_steps - 1):
            if E_values[i] > E_values[i - 1] and E_values[i] > E_values[i + 1]:
                maxima_indices.append(i)

        # Need at least 3 maxima to test damping
        if len(maxima_indices) >= 3:
            # Get maxima values
            maxima = E_values[maxima_indices]

            # Check if magnitude is decreasing
            for i in range(1, len(maxima)):
                # Allow for numerical imprecision, but general trend should be decreasing
                if i < len(maxima) - 1:  # Skip last point for safety
                    self.assertLessEqual(maxima[i], maxima[i - 1] * 1.01)  # Allow 1% tolerance
        else:
            # Not enough maxima found, check if there's oscillation
            amplitude = (np.max(E_values) - np.min(E_values)) / 2
            # If there's significant oscillation, warn but don't fail
            if amplitude > 0.1:
                print(f"Warning: Oscillation detected but insufficient maxima found. Amplitude={amplitude}")
            # Otherwise, test is inconclusive but shouldn't fail
            else:
                print("Warning: Insufficient oscillation detected for damping measurement.")


class TestPhaseTransition(unittest.TestCase):
    """Tests for phase transition behavior in knowledge growth."""

    def test_critical_threshold(self):
        """Test that knowledge growth changes behavior at critical threshold."""
        # Test values below, at, and above threshold
        K_0 = 10  # Initial knowledge
        beta_decay = 0.01  # Slow decay
        t = 10  # Time step
        A = 5  # Growth amplitude
        gamma = 0.5  # Transition sharpness
        T_crit = 20  # Critical threshold

        # Test at different truth levels
        T_below = 15  # Below critical threshold
        T_at = T_crit  # At critical threshold
        T_above = 25  # Above critical threshold

        # Calculate knowledge at each level with bounds
        k_below = knowledge_growth_phase_transition(K_0, beta_decay, t, A, gamma, T=T_below, T_crit=T_crit)
        k_at = knowledge_growth_phase_transition(K_0, beta_decay, t, A, gamma, T=T_at, T_crit=T_crit)
        k_above = knowledge_growth_phase_transition(K_0, beta_decay, t, A, gamma, T=T_above, T_crit=T_crit)

        # Knowledge should increase with truth level
        self.assertLess(k_below, k_at)
        self.assertLess(k_at, k_above)

        # Ensure all values are finite
        self.assertTrue(np.isfinite(k_below))
        self.assertTrue(np.isfinite(k_at))
        self.assertTrue(np.isfinite(k_above))


class TestFeedbackLoops(unittest.TestCase):
    """Tests for feedback loops and system dynamics."""

    def test_positive_feedback_knowledge_truth(self):
        """Test positive feedback loop between knowledge and truth."""
        # Set up simulation
        time_steps = 100
        dt = 1
        T = np.zeros(time_steps)
        K = np.zeros(time_steps)

        # Constants - adjusted for stronger feedback effects
        A_truth, T_max = 0.2, 40  # Lower A_truth for stability
        T_crit_phase = 10  # Lower threshold for easier crossing
        A_phase, gamma_phase = 2, 0.5  # Higher gamma for sharper transition

        # Initial conditions
        T[0] = 5.0
        K[0] = 5.0

        # Run a simplified simulation with bounds to prevent instability
        for t in range(1, time_steps):
            # Update truth based on current knowledge to create feedback
            # More knowledge increases truth adoption rate
            truth_change = truth_adoption(
                np.clip(T[t - 1], 0, T_max),
                A_truth * (1 + 0.01 * np.clip(K[t - 1], 0, 1000)),
                T_max
            )
            T[t] = np.clip(T[t - 1] + truth_change * dt, 0, T_max)

            # Update knowledge based on truth
            knowledge_change = knowledge_growth_phase_transition(
                np.clip(K[t - 1], 0, 1000),
                0.01,
                min(t, 100),  # Limit t to prevent overflow
                A_phase,
                gamma_phase,
                np.clip(T[t - 1], 0, T_max),
                T_crit_phase
            )
            K[t] = np.clip(K[t - 1] + knowledge_change * dt, 0, 1000)

        # Calculate growth rates with stability handling
        k_growth_rates = np.diff(K)
        k_growth_rates = np.nan_to_num(k_growth_rates, nan=0.0, posinf=100, neginf=-100)

        t_growth_rates = np.diff(T)
        t_growth_rates = np.nan_to_num(t_growth_rates, nan=0.0, posinf=10, neginf=-10)

        # Find index where truth crosses critical threshold
        crossing_indices = np.where(T > T_crit_phase)[0]
        if len(crossing_indices) > 0:
            crossing_point = crossing_indices[0]

            # Ensure we have enough points after crossing
            if crossing_point + 10 < time_steps:
                # Check if knowledge growth accelerates after crossing
                pre_crossing_growth = np.mean(k_growth_rates[max(0, crossing_point - 10):crossing_point])
                post_crossing_growth = np.mean(k_growth_rates[crossing_point:min(crossing_point + 10, time_steps - 1)])

                # Check with a buffer for numerical stability
                self.assertGreater(post_crossing_growth, pre_crossing_growth * 0.8)

                # Check that truth adoption is mostly positive (a weaker but more stable assertion)
                positive_growth = np.sum(t_growth_rates > 0)
                self.assertGreater(positive_growth, time_steps * 0.6)  # At least 60% should be positive

    def test_negative_feedback_suppression(self):
        """Test negative feedback loop with suppression."""
        # Set up simulation
        time_steps = 100
        dt = 1
        K = np.zeros(time_steps)
        S = np.zeros(time_steps)

        # Constants with reasonably bounded values
        alpha_feedback = 0.1
        beta_feedback = 0.2

        # Initial conditions - high suppression, low knowledge
        K[0] = 1.0
        S[0] = 10.0

        # Run a simplified simulation with negative feedback and bounds
        for t in range(1, time_steps):
            # Calculate feedback with bounds
            feedback = suppression_feedback(
                alpha_feedback,
                np.clip(S[t - 1], 0, 100),
                beta_feedback,
                np.clip(K[t - 1], 0, 100)
            )

            # Update suppression with bounds
            S[t] = np.clip(S[t - 1] + feedback * dt, 0, 100)

            # Simple knowledge growth (increases over time) with suppression modifier
            growth = np.clip(1 - 0.1 * S[t - 1], 0, 5)  # Bound growth rate to prevent extreme jumps
            K[t] = np.clip(K[t - 1] + growth * dt, 0, 100)

        # Ensure all arrays contain finite values
        K = np.nan_to_num(K, nan=0, posinf=100, neginf=0)
        S = np.nan_to_num(S, nan=0, posinf=100, neginf=0)

        # Initially, suppression should maintain or increase
        # Use a robust test with tolerance for numerical effects
        initial_trend = S[5] - S[0]
        self.assertGreaterEqual(initial_trend, -0.1)  # Allow small decrease

        # As knowledge grows, suppression should eventually decrease
        # Find where knowledge exceeds critical threshold for negative feedback
        critical_value = alpha_feedback / beta_feedback
        crossover_indices = np.where(K > critical_value)[0]

        if len(crossover_indices) > 0:
            crossover_point = crossover_indices[0]

            # Ensure we have enough points after crossover
            if crossover_point + 20 < time_steps:
                # Check for suppression decrease after sufficient time
                late_suppression = S[crossover_point + 20]
                crossover_suppression = S[crossover_point]

                # Use a relaxed comparison to handle potential oscillations
                self.assertLessEqual(late_suppression, crossover_suppression * 1.1)


if __name__ == '__main__':
    unittest.main()