import sys
import unittest
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import quantum and EM extensions to test
from config.quantum_em_extensions import (
    knowledge_field_influence, quantum_entanglement_correlation,
    knowledge_field_gradient, build_entanglement_network,
    quantum_tunneling_probability
)


class TestElectromagneticAnalogies(unittest.TestCase):
    """Tests for electromagnetic field analogy functions."""

    def test_knowledge_field_influence(self):
        """Test knowledge field influence between agents."""
        # Field influence should follow inverse square law
        K_i = 10
        K_j = 10
        kappa = 0.05

        # Test at different distances
        influence_1 = knowledge_field_influence(K_i, K_j, r_ij=1, kappa=kappa)
        influence_2 = knowledge_field_influence(K_i, K_j, r_ij=2, kappa=kappa)
        influence_4 = knowledge_field_influence(K_i, K_j, r_ij=4, kappa=kappa)

        # Verify inverse square relationship (ratio should be 4:1 for double distance)
        self.assertAlmostEqual(influence_1 / influence_2, 4.0, places=5)
        self.assertAlmostEqual(influence_2 / influence_4, 4.0, places=5)

        # Influence should be proportional to both knowledge levels
        self.assertEqual(
            knowledge_field_influence(K_i=2, K_j=3, r_ij=1, kappa=kappa),
            knowledge_field_influence(K_i=3, K_j=2, r_ij=1, kappa=kappa)
        )

        # Test that zero knowledge has no influence
        self.assertEqual(knowledge_field_influence(K_i=0, K_j=10, r_ij=1, kappa=kappa), 0)

        # Test minimum distance protection
        very_close = knowledge_field_influence(K_i=10, K_j=10, r_ij=0.001, kappa=kappa)
        self.assertTrue(np.isfinite(very_close))

    def test_knowledge_field_gradient(self):
        """Test knowledge field gradients."""
        # Create a simple agent system
        agent_knowledge = np.array([10, 5, 2])
        agent_positions = np.array([
            [0.0, 0.0],  # Agent 0 at origin
            [1.0, 0.0],  # Agent 1 at (1,0)
            [0.0, 1.0]  # Agent 2 at (0,1)
        ], dtype=float)  # Ensure float type for agent positions

        # Calculate field gradients
        gradients = knowledge_field_gradient(agent_knowledge, agent_positions)

        # Gradients should point from lower to higher knowledge
        # Agent 0 has highest knowledge, so others should point toward it
        self.assertTrue(gradients[1][0] < 0)  # Agent 1 should move left (toward Agent 0)
        self.assertTrue(gradients[2][1] < 0)  # Agent 2 should move down (toward Agent 0)

        # Agent 0 should move based on net influence, magnitude should be smaller than others
        self.assertLess(np.linalg.norm(gradients[0]), np.linalg.norm(gradients[1]))

        # Test uniform knowledge (no gradients)
        uniform_knowledge = np.array([5, 5, 5])
        uniform_gradients = knowledge_field_gradient(uniform_knowledge, agent_positions)

        # All gradient magnitudes should be small
        for gradient in uniform_gradients:
            self.assertLess(np.linalg.norm(gradient), 1e-5)


class TestQuantumAnalogies(unittest.TestCase):
    """Tests for quantum mechanical analogy functions."""

    def test_quantum_entanglement_correlation(self):
        """Test quantum entanglement correlation between knowledge states."""
        # Identical knowledge states should have maximum entanglement
        max_corr = quantum_entanglement_correlation(K_i=10, K_j=10, rho=0.1, sigma=0.05)
        self.assertEqual(max_corr, 0.1)  # rho is the maximum

        # Correlation should decay with knowledge difference
        corr_diff_1 = quantum_entanglement_correlation(K_i=10, K_j=9, rho=0.1, sigma=0.05)
        corr_diff_2 = quantum_entanglement_correlation(K_i=10, K_j=8, rho=0.1, sigma=0.05)

        # Larger difference means lower correlation
        self.assertGreater(corr_diff_1, corr_diff_2)

        # Correlation should be symmetric
        self.assertEqual(
            quantum_entanglement_correlation(K_i=5, K_j=10, rho=0.1, sigma=0.05),
            quantum_entanglement_correlation(K_i=10, K_j=5, rho=0.1, sigma=0.05)
        )

        # Very different knowledge states should have minimal correlation
        min_corr = quantum_entanglement_correlation(K_i=1, K_j=100, rho=0.1, sigma=0.05)
        self.assertLess(min_corr, 0.01)

    def test_entanglement_network(self):
        """Test entanglement network construction."""
        # Create simple agent knowledge states
        agent_knowledge = np.array([10, 8, 5, 2])
        num_agents = len(agent_knowledge)

        # Build entanglement matrix
        max_entanglement = 0.2
        decay_rate = 0.1
        entanglement_matrix = build_entanglement_network(
            agent_knowledge, max_entanglement, decay_rate)

        # Check matrix dimensions
        self.assertEqual(entanglement_matrix.shape, (num_agents, num_agents))

        # Diagonal should be 1 (self-entanglement)
        for i in range(num_agents):
            self.assertEqual(entanglement_matrix[i, i], 1.0)

        # Matrix should be symmetric
        for i in range(num_agents):
            for j in range(num_agents):
                self.assertEqual(entanglement_matrix[i, j], entanglement_matrix[j, i])

        # Similar agents should have higher entanglement
        # Agents 0 and 1 are more similar than Agents 0 and 3
        self.assertGreater(entanglement_matrix[0, 1], entanglement_matrix[0, 3])

        # Verify min entanglement is found at correct location
        # Find the minimum non-diagonal entanglement value
        min_val = float('inf')
        min_i, min_j = -1, -1
        for i in range(num_agents):
            for j in range(num_agents):
                if i != j and entanglement_matrix[i, j] < min_val:
                    min_val = entanglement_matrix[i, j]
                    min_i, min_j = i, j

        # The minimum should be between most different agents (0 and 3)
        self.assertTrue((min_i == 0 and min_j == 3) or (min_i == 3 and min_j == 0))

    def test_quantum_tunneling_probability(self):
        """Test quantum tunneling probability calculation."""
        # Energy above barrier should always tunnel
        self.assertEqual(
            quantum_tunneling_probability(barrier_height=5, barrier_width=1, energy_level=10), 1.0)

        # Energy equal to barrier should tunnel
        self.assertEqual(
            quantum_tunneling_probability(barrier_height=5, barrier_width=1, energy_level=5), 1.0)

        # Energy below barrier should have probability between 0 and 1
        prob_below = quantum_tunneling_probability(barrier_height=10, barrier_width=1, energy_level=5)
        self.assertTrue(0 < prob_below < 1)

        # Tunneling probability should decrease with barrier width
        prob_narrow = quantum_tunneling_probability(barrier_height=10, barrier_width=1, energy_level=5)
        prob_wide = quantum_tunneling_probability(barrier_height=10, barrier_width=2, energy_level=5)
        self.assertGreater(prob_narrow, prob_wide)

        # Tunneling probability should decrease with barrier height
        prob_low = quantum_tunneling_probability(barrier_height=10, barrier_width=1, energy_level=5)
        prob_high = quantum_tunneling_probability(barrier_height=20, barrier_width=1, energy_level=5)
        self.assertGreater(prob_low, prob_high)

        # Tunneling probability should increase with energy level
        prob_low_energy = quantum_tunneling_probability(barrier_height=10, barrier_width=1, energy_level=2)
        prob_high_energy = quantum_tunneling_probability(barrier_height=10, barrier_width=1, energy_level=8)
        self.assertGreater(prob_high_energy, prob_low_energy)


class TestIntegration(unittest.TestCase):
    """Tests for integration of EM and quantum effects with core equations."""

    def test_field_entanglement_interaction(self):
        """Test interaction between field influence and entanglement."""
        # Create simple system - agents with varying knowledge
        num_agents = 5
        agent_knowledge = np.linspace(1, 10, num_agents)
        agent_positions = np.random.rand(num_agents, 2)  # Random positions in 2D space

        # Calculate field influences
        field_influences = np.zeros((num_agents, num_agents))
        for i in range(num_agents):
            for j in range(num_agents):
                if i != j:
                    r_ij = np.linalg.norm(agent_positions[i] - agent_positions[j])
                    field_influences[i, j] = knowledge_field_influence(
                        agent_knowledge[i], agent_knowledge[j], r_ij)

        # Calculate entanglement correlations
        entanglement_matrix = build_entanglement_network(agent_knowledge)

        # Field effects should be stronger for nearby agents,
        # while entanglement can be strong even for distant agents
        # Find most distant agent pair
        max_dist_i, max_dist_j = 0, 0
        max_dist = 0
        for i in range(num_agents):
            for j in range(num_agents):
                if i != j:
                    dist = np.linalg.norm(agent_positions[i] - agent_positions[j])
                    if dist > max_dist:
                        max_dist = dist
                        max_dist_i, max_dist_j = i, j

        # For most distant pair, entanglement should be more significant relative to field effect
        # We compare the ratio to the maximum value of each effect
        if np.max(field_influences) > 0:  # Prevent division by zero
            field_ratio = field_influences[max_dist_i, max_dist_j] / np.max(field_influences)
            entanglement_ratio = (entanglement_matrix[max_dist_i, max_dist_j] - 1.0) / np.max(
                entanglement_matrix - np.eye(num_agents))

            # Distance should affect field influence more than entanglement
            if entanglement_ratio > 0 and field_ratio > 0:
                self.assertGreaterEqual(entanglement_ratio / field_ratio, 1.0)

    def test_tunneling_breakthrough(self):
        """Test tunneling breakthrough across suppression barriers."""
        # Set up a scenario with high suppression barrier
        barrier_height = 10.0
        barrier_width = 1.0

        # Test how energy levels affect tunneling
        energy_levels = np.linspace(1, 9, 9)  # All below barrier

        # Higher energy should provide higher tunneling probability
        tunneling_probs = [
            quantum_tunneling_probability(barrier_height, barrier_width, e)
            for e in energy_levels
        ]

        # Probabilities should increase monotonically with energy
        self.assertTrue(np.all(np.diff(tunneling_probs) >= 0))

        # Calculate critical energy where tunneling probability exceeds 50%
        critical_energy = None
        for energy in np.linspace(1, 9, 81):  # Finer grid
            prob = quantum_tunneling_probability(barrier_height, barrier_width, energy)
            if prob > 0.5:
                critical_energy = energy
                break

        if critical_energy is not None:
            # Verify that energy levels below critical have probability < 50%
            self.assertLess(
                quantum_tunneling_probability(barrier_height, barrier_width, critical_energy - 0.5),
                0.5
            )

            # Verify that energy levels above critical have probability > 50%
            self.assertGreater(
                quantum_tunneling_probability(barrier_height, barrier_width, critical_energy + 0.5),
                0.5
            )


class TestSystemicEffects(unittest.TestCase):
    """Tests for systemic effects of EM and quantum principles."""

    def test_field_diffusion(self):
        """Test knowledge field diffusion through system."""
        # Create a chain of agents with knowledge decreasing along the chain
        num_agents = 5
        agent_knowledge = np.array([10, 5, 2, 1, 0.5])
        positions = np.array([[float(i), 0.0] for i in range(num_agents)])  # 1D chain with explicit float type

        # Calculate field gradients
        gradients = knowledge_field_gradient(agent_knowledge, positions)

        # Knowledge should flow from higher to lower
        # Agent 0 should move right (positive x)
        self.assertGreater(gradients[0, 0], 0)

        # Agent 4 should move left (negative x)
        self.assertLess(gradients[4, 0], 0)

        # Middle agents should feel both pulls
        # Their movement depends on the field strength, but there should be gradients
        self.assertNotEqual(gradients[2, 0], 0)

    def test_entanglement_clusters(self):
        """Test entanglement cluster formation."""
        # Create two clusters of agents with similar knowledge
        cluster1_knowledge = np.random.normal(10, 1, 5)  # Cluster around 10
        cluster2_knowledge = np.random.normal(2, 0.5, 5)  # Cluster around 2
        all_knowledge = np.concatenate([cluster1_knowledge, cluster2_knowledge])

        # Build entanglement network
        entanglement_matrix = build_entanglement_network(all_knowledge)

        # Calculate average entanglement within and between clusters
        within_cluster1 = 0
        count1 = 0
        for i in range(5):
            for j in range(5):
                if i != j:
                    within_cluster1 += entanglement_matrix[i, j]
                    count1 += 1
        avg_within_cluster1 = within_cluster1 / count1

        within_cluster2 = 0
        count2 = 0
        for i in range(5, 10):
            for j in range(5, 10):
                if i != j:
                    within_cluster2 += entanglement_matrix[i, j]
                    count2 += 1
        avg_within_cluster2 = within_cluster2 / count2

        between_clusters = 0
        count_between = 0
        for i in range(5):
            for j in range(5, 10):
                between_clusters += entanglement_matrix[i, j]
                count_between += 1
        avg_between_clusters = between_clusters / count_between

        # Entanglement should be higher within clusters than between clusters
        self.assertGreater(avg_within_cluster1, avg_between_clusters)
        self.assertGreater(avg_within_cluster2, avg_between_clusters)


if __name__ == '__main__':
    unittest.main()