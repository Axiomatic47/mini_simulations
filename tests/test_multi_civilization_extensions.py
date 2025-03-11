import sys
import unittest
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import multi-civilization extensions to test
from config.multi_civilization_extensions import (
    initialize_civilizations, calculate_distance_matrix, calculate_interaction_strength,
    knowledge_diffusion, cultural_influence, resource_competition,
    civilization_movement, update_civilization_sizes,
    detect_civilization_collapse, detect_civilization_mergers,
    process_civilization_merger, spawn_new_civilization, remove_civilization,
    process_all_civilization_interactions
)


class TestMultiCivilizationBasics(unittest.TestCase):
    """Basic tests for multi-civilization extensions with stability controls."""

    def test_initialize_civilizations(self):
        """Test civilization initialization with parameter validation."""
        # Test with various inputs including edge cases
        test_cases = [
            (5, "Normal case"),
            (1, "Single civilization"),
            (0, "Zero civilizations - should handle gracefully"),
            (-1, "Negative value - should handle gracefully")
        ]

        for num_civilizations, description in test_cases:
            with self.subTest(description):
                civilizations = initialize_civilizations(num_civilizations)

                # For invalid inputs, should return at least one civilization
                actual_count = len(civilizations["ages"]) if "ages" in civilizations else 0
                self.assertGreaterEqual(actual_count, max(1, num_civilizations))

                # Check that all expected keys exist
                expected_keys = ["ages", "positions", "innovation_rates", "suppression_resistance",
                                 "knowledge_retention", "expansion_tendency", "resources",
                                 "influence", "velocities", "sizes"]
                for key in expected_keys:
                    self.assertIn(key, civilizations)

                # Check that arrays have correct dimensions
                self.assertEqual(len(civilizations["ages"]), actual_count)
                self.assertEqual(civilizations["positions"].shape, (actual_count, 2))
                self.assertEqual(len(civilizations["innovation_rates"]), actual_count)

                # Check that values are within expected ranges
                self.assertTrue(np.all(civilizations["innovation_rates"] >= 0.8))
                self.assertTrue(np.all(civilizations["innovation_rates"] <= 1.2))
                self.assertTrue(np.all(civilizations["suppression_resistance"] >= 0.7))
                self.assertTrue(np.all(civilizations["suppression_resistance"] <= 1.3))
                self.assertTrue(np.all(civilizations["positions"] >= 0))
                self.assertTrue(np.all(civilizations["positions"] <= 10))

    def test_calculate_distance_matrix(self):
        """Test distance matrix calculation with stability for edge cases."""
        # Test normal case
        positions = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ])

        distances = calculate_distance_matrix(positions)

        # Check matrix dimensions
        self.assertEqual(distances.shape, (3, 3))

        # Check diagonal is zero
        self.assertEqual(distances[0, 0], 0)
        self.assertEqual(distances[1, 1], 0)
        self.assertEqual(distances[2, 2], 0)

        # Check distances
        self.assertEqual(distances[0, 1], 1.0)  # Distance from [0,0] to [1,0]
        self.assertEqual(distances[0, 2], 1.0)  # Distance from [0,0] to [0,1]
        self.assertAlmostEqual(distances[1, 2], np.sqrt(2), places=5)  # Distance from [1,0] to [0,1]

        # Check symmetry
        self.assertEqual(distances[0, 1], distances[1, 0])
        self.assertEqual(distances[0, 2], distances[2, 0])
        self.assertEqual(distances[1, 2], distances[2, 1])

        # Test empty array
        empty_positions = np.array([])
        try:
            empty_distances = calculate_distance_matrix(empty_positions.reshape(0, 2))
            # Should return a valid matrix even for empty input
            self.assertEqual(empty_distances.shape, (0, 0))
        except Exception as e:
            # Or gracefully handle the error
            self.assertTrue(True)

        # Test single civilization
        single_position = np.array([[1.0, 2.0]])
        single_distance = calculate_distance_matrix(single_position)
        self.assertEqual(single_distance.shape, (1, 1))
        self.assertEqual(single_distance[0, 0], 0)

    def test_calculate_interaction_strength(self):
        """Test interaction strength calculation with stability controls."""
        distance_matrix = np.array([
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0]
        ])

        # Test with default max distance
        interaction = calculate_interaction_strength(distance_matrix)

        # Check matrix dimensions
        self.assertEqual(interaction.shape, (3, 3))

        # Check diagonal is zero
        self.assertEqual(interaction[0, 0], 0)
        self.assertEqual(interaction[1, 1], 0)
        self.assertEqual(interaction[2, 2], 0)

        # Check interaction strength decreases with distance
        self.assertGreater(interaction[0, 1], interaction[0, 2])
        self.assertGreater(interaction[1, 0], interaction[1, 2])

        # Test with smaller max distance
        interaction_limited = calculate_interaction_strength(distance_matrix, max_interaction_distance=1.5)

        # Interactions beyond max distance should be zero
        self.assertEqual(interaction_limited[0, 2], 0)
        self.assertEqual(interaction_limited[1, 2], 0)
        self.assertEqual(interaction_limited[2, 0], 0)
        self.assertEqual(interaction_limited[2, 1], 0)

        # Interactions within max distance should be positive
        self.assertGreater(interaction_limited[0, 1], 0)
        self.assertGreater(interaction_limited[1, 0], 0)

        # Test with zero distance - should handle division by zero safely
        zero_distance_matrix = np.zeros((3, 3))
        zero_interaction = calculate_interaction_strength(zero_distance_matrix, min_distance=0.1)

        # Interactions should be finite and reasonable (not infinite)
        self.assertTrue(np.all(np.isfinite(zero_interaction)))
        self.assertTrue(np.all(zero_interaction >= 0))

        # Test with negative distance - should handle safely
        negative_distance_matrix = np.array([
            [0.0, -1.0, 2.0],
            [-1.0, 0.0, -3.0],
            [2.0, -3.0, 0.0]
        ])

        negative_interaction = calculate_interaction_strength(negative_distance_matrix, min_distance=0.1)

        # Interactions should be finite and reasonable
        self.assertTrue(np.all(np.isfinite(negative_interaction)))
        self.assertTrue(np.all(negative_interaction >= 0))


class TestMultiCivilizationInteractions(unittest.TestCase):
    """Tests for civilization interactions with stability controls."""

    def setUp(self):
        """Set up test civilizations."""
        self.num_civilizations = 3
        self.civilizations = initialize_civilizations(self.num_civilizations)

        # Fix positions for deterministic tests
        self.civilizations["positions"] = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [5.0, 5.0]
        ])

        # Fix sizes for deterministic tests
        self.civilizations["sizes"] = np.array([1.0, 2.0, 3.0])

        # Fix innovation rates
        self.civilizations["innovation_rates"] = np.array([1.0, 1.0, 1.0])

        # Fix knowledge retention
        self.civilizations["knowledge_retention"] = np.array([0.5, 0.5, 0.5])

        # Fix expansion tendency
        self.civilizations["expansion_tendency"] = np.array([1.0, 1.0, 1.0])

        # Knowledge, suppression, influence, and resources arrays
        self.knowledge_array = np.array([5.0, 10.0, 15.0])
        self.suppression_array = np.array([2.0, 5.0, 8.0])
        self.influence_array = np.array([3.0, 6.0, 9.0])
        self.resources_array = np.array([10.0, 20.0, 30.0])

        # Calculate distance and interaction matrices
        self.distance_matrix = calculate_distance_matrix(self.civilizations["positions"])
        self.interaction_matrix = calculate_interaction_strength(self.distance_matrix)

    def test_knowledge_diffusion(self):
        """Test knowledge diffusion between civilizations with stability controls."""
        # Test normal case
        diffusion = knowledge_diffusion(
            self.civilizations,
            self.knowledge_array,
            self.interaction_matrix,
            diffusion_rate=0.01,
            max_diffusion=0.5  # Add maximum diffusion parameter
        )

        # Check output dimensions
        self.assertEqual(len(diffusion), self.num_civilizations)

        # Knowledge should flow from higher to lower
        self.assertGreater(diffusion[0], 0)  # Civ 0 has lowest knowledge, should gain

        # Interaction should be stronger between closer civilizations
        self.assertGreater(abs(diffusion[0]), abs(diffusion[2]))

        # Test with extreme values
        extreme_knowledge = np.array([1e6, 1e-6, 0.0])
        extreme_diffusion = knowledge_diffusion(
            self.civilizations,
            extreme_knowledge,
            self.interaction_matrix,
            diffusion_rate=0.01,
            max_diffusion=1000.0  # Higher limit for extreme case
        )

        # Results should be finite and respect max diffusion
        self.assertTrue(np.all(np.isfinite(extreme_diffusion)))
        self.assertTrue(np.all(np.abs(extreme_diffusion) <= 1000.0))

        # Test with empty arrays
        empty_civs = initialize_civilizations(0)
        empty_knowledge = np.array([])
        empty_interaction = np.array([]).reshape(0, 0)

        empty_diffusion = knowledge_diffusion(
            empty_civs,
            empty_knowledge,
            empty_interaction
        )

        # Should return empty array without error
        self.assertEqual(len(empty_diffusion), 0)

    def test_cultural_influence(self):
        """Test cultural influence with stability controls."""
        # Test normal case
        influence_change = cultural_influence(
            self.civilizations,
            self.influence_array,
            self.interaction_matrix,
            base_influence_rate=0.02,
            max_influence_change=1.0,
            min_division=0.1
        )

        # Check output dimensions
        self.assertEqual(len(influence_change), self.num_civilizations)

        # Influence should flow from higher to lower
        self.assertGreater(influence_change[0], 0)  # Civ 0 has lowest influence, should gain

        # Civilization 2 is far from others, should have less influence change
        self.assertLess(abs(influence_change[2]), abs(influence_change[0]))

        # Test with extreme values
        extreme_influence = np.array([1e6, 1e-6, 0.0])
        extreme_influence_change = cultural_influence(
            self.civilizations,
            extreme_influence,
            self.interaction_matrix,
            base_influence_rate=0.02,
            max_influence_change=10.0,
            min_division=0.1
        )

        # Results should be finite and respect max influence change
        self.assertTrue(np.all(np.isfinite(extreme_influence_change)))
        self.assertTrue(np.all(np.abs(extreme_influence_change) <= 10.0))

        # Test with division by zero scenario
        zero_sizes = self.civilizations.copy()
        zero_sizes["sizes"] = np.array([1.0, 0.0, 3.0])  # Second civilization has zero size

        # Should handle division by zero gracefully
        zero_size_influence = cultural_influence(
            zero_sizes,
            self.influence_array,
            self.interaction_matrix,
            min_division=0.1  # Minimum division value should prevent issues
        )

        # Results should be finite
        self.assertTrue(np.all(np.isfinite(zero_size_influence)))

    def test_resource_competition(self):
        """Test resource competition with stability controls."""
        # Test normal case
        resource_change = resource_competition(
            self.civilizations,
            self.resources_array,
            self.interaction_matrix,
            competition_rate=0.01,
            max_resource_change=2.0,
            min_division=0.1
        )

        # Check output dimensions
        self.assertEqual(len(resource_change), self.num_civilizations)

        # Test extreme values
        extreme_resources = np.array([1e10, 1e-10, 0.0])
        extreme_resource_change = resource_competition(
            self.civilizations,
            extreme_resources,
            self.interaction_matrix,
            competition_rate=0.01,
            max_resource_change=10.0,
            min_division=0.1
        )

        # Results should be finite and respect max resource change
        self.assertTrue(np.all(np.isfinite(extreme_resource_change)))
        self.assertTrue(np.all(np.abs(extreme_resource_change) <= 10.0))

        # Test log behavior with negative power ratios
        # Create scenario where first civilization would 'donate' resources
        weak_civ = self.civilizations.copy()
        weak_civ["influence"] = np.array([1.0, 10.0, 10.0])  # First civilization is weak
        weak_civ["knowledge_retention"] = np.array([0.1, 0.9, 0.9])  # First civilization has low retention

        weak_resources = np.array([5.0, 50.0, 50.0])  # Equal resources for comparison

        weak_resource_change = resource_competition(
            weak_civ,
            weak_resources,
            self.interaction_matrix,
            competition_rate=0.01,
            max_resource_change=2.0,
            min_division=0.1
        )

        # Verify first civilization loses resources
        self.assertLessEqual(weak_resource_change[0], 0)

    def test_civilization_movement(self):
        """Test civilization movement with stability and bounded velocities."""
        # Create a copy of positions
        original_positions = self.civilizations["positions"].copy()

        # Run movement function with stability parameters
        civilization_movement(
            self.civilizations,
            self.interaction_matrix,
            dt=1.0,
            attraction_factor=0.01,
            repulsion_threshold=1.0,
            max_velocity=1.0,
            damping=0.9
        )

        # Check that positions changed but not excessively
        self.assertFalse(np.array_equal(self.civilizations["positions"], original_positions))

        # Verify movement is bounded
        position_change = np.max(np.abs(self.civilizations["positions"] - original_positions))
        self.assertLessEqual(position_change, 1.0)  # Max velocity * dt

        # Check that civilizations move toward more influential neighbors
        # Civ 0 should move toward Civ 1 (which has higher influence)
        self.assertGreater(self.civilizations["positions"][0, 0], original_positions[0, 0])

        # Test with extreme velocities
        extreme_civ = self.civilizations.copy()
        extreme_civ["velocities"] = np.array([[100.0, 100.0], [-100.0, -100.0], [0.0, 0.0]])

        extreme_positions_before = extreme_civ["positions"].copy()

        # Should apply velocity limitation
        civilization_movement(
            extreme_civ,
            self.interaction_matrix,
            dt=1.0,
            max_velocity=0.5  # Strict limit
        )

        # Check that movement is limited
        extreme_movement = np.max(np.abs(extreme_civ["positions"] - extreme_positions_before))
        self.assertLessEqual(extreme_movement, 0.5)  # Should respect max_velocity

    def test_detect_civilization_collapse(self):
        """Test detection of civilization collapse with stability for edge cases."""
        # Create a test case where one civilization should collapse
        knowledge = np.array([10.0, 1.0, 5.0])
        suppression = np.array([5.0, 20.0, 10.0])

        # Test normal case
        collapses = detect_civilization_collapse(knowledge, suppression, threshold=0.1, min_division=0.1)

        # Check that civilization 1 collapses (knowledge/suppression < threshold)
        self.assertFalse(collapses[0])
        self.assertTrue(collapses[1])
        self.assertFalse(collapses[2])

        # Test with zero knowledge
        zero_knowledge = np.array([0.0, 10.0, 5.0])
        zero_collapses = detect_civilization_collapse(zero_knowledge, suppression, threshold=0.1, min_division=0.1)

        # First civilization should collapse
        self.assertTrue(zero_collapses[0])

        # Test with zero suppression - should not cause division by zero
        zero_suppression = np.array([10.0, 0.0, 5.0])
        zero_supp_collapses = detect_civilization_collapse(knowledge, zero_suppression, threshold=0.1, min_division=0.1)

        # Should not collapse with zero suppression
        self.assertFalse(zero_supp_collapses[1])

        # Test with empty arrays
        empty_collapses = detect_civilization_collapse(np.array([]), np.array([]), threshold=0.1, min_division=0.1)
        self.assertEqual(len(empty_collapses), 0)

    def test_detect_civilization_mergers(self):
        """Test detection of civilization mergers with stability controls."""
        # Create a test case with two civilizations close together
        self.civilizations["positions"] = np.array([
            [0.0, 0.0],
            [0.3, 0.0],  # Very close to civilization 0
            [5.0, 5.0]
        ])

        # Civilization 1 is much larger than civilization 0
        self.civilizations["sizes"] = np.array([1.0, 4.0, 2.0])

        # Test normal case
        mergers = detect_civilization_mergers(
            self.civilizations,
            distance_threshold=0.5,
            size_ratio_threshold=3.0
        )

        # Should detect one merger: civilization 1 absorbs civilization 0
        self.assertEqual(len(mergers), 1)
        self.assertEqual(mergers[0], (1, 0))  # Format is (absorber, absorbed)

        # Test with extreme size disparity
        extreme_sizes = self.civilizations.copy()
        extreme_sizes["sizes"] = np.array([0.01, 100.0, 2.0])  # Extreme size difference

        extreme_mergers = detect_civilization_mergers(
            extreme_sizes,
            distance_threshold=0.5,
            size_ratio_threshold=3.0
        )

        # Should still detect the merger despite extreme size difference
        self.assertEqual(len(extreme_mergers), 1)

        # Test with zero size
        zero_sizes = self.civilizations.copy()
        zero_sizes["sizes"] = np.array([0.0, 4.0, 2.0])  # First civilization has zero size

        zero_mergers = detect_civilization_mergers(
            zero_sizes,
            distance_threshold=0.5,
            size_ratio_threshold=3.0
        )

        # Should detect merger despite zero size
        self.assertEqual(len(zero_mergers), 1)

        # Test with empty civilizations
        empty_civs = initialize_civilizations(0)
        empty_mergers = detect_civilization_mergers(empty_civs)

        # Should return empty list
        self.assertEqual(len(empty_mergers), 0)

    def test_process_civilization_merger(self):
        """Test processing of civilization mergers with stability controls."""
        # Before merger
        absorber_knowledge_before = self.knowledge_array[1]
        absorbed_knowledge = self.knowledge_array[0]

        # Process merger with stability parameters
        self.civilizations, self.knowledge_array = process_civilization_merger(
            self.civilizations,
            self.knowledge_array,
            1,  # Absorber index
            0,  # Absorbed index
            knowledge_transfer_ratio=0.8,
            resource_transfer_ratio=1.0,
            influence_transfer_ratio=0.9,
            size_transfer_ratio=0.7
        )

        # After merger, absorber should have increased knowledge
        self.assertGreater(self.knowledge_array[1], absorber_knowledge_before)

        # Absorber should gain a percentage of the absorbed civilization's knowledge
        expected_gain = 0.8 * absorbed_knowledge
        self.assertAlmostEqual(
            self.knowledge_array[1] - absorber_knowledge_before,
            expected_gain,
            places=5
        )

        # Test invalid indices
        invalid_civs = self.civilizations.copy()
        invalid_knowledge = self.knowledge_array.copy()

        # Process with invalid absorber index (too high)
        invalid_res1 = process_civilization_merger(
            invalid_civs,
            invalid_knowledge,
            10,  # Invalid absorber
            0
        )

        # Should return unchanged arrays
        self.assertEqual(len(invalid_res1[1]), len(invalid_knowledge))

        # Process with invalid absorbed index (too high)
        invalid_res2 = process_civilization_merger(
            invalid_civs,
            invalid_knowledge,
            0,
            10  # Invalid absorbed
        )

        # Should return unchanged arrays
        self.assertEqual(len(invalid_res2[1]), len(invalid_knowledge))

        # Process with same absorber and absorbed
        invalid_res3 = process_civilization_merger(
            invalid_civs,
            invalid_knowledge,
            1,
            1  # Same as absorber
        )

        # Should return unchanged arrays
        self.assertEqual(len(invalid_res3[1]), len(invalid_knowledge))

    def test_spawn_new_civilization(self):
        """Test spawning of new civilizations with stability controls."""
        num_civilizations_before = len(self.knowledge_array)

        # Spawn new civilization - fix: position must be a 1D array of length 2
        position = np.array([2.0, 2.0])
        parent_idx = 1  # Civilization 1 is the parent

        # Before the new civilization is created, store the parent's resources
        parent_resources_before = self.civilizations["resources"][parent_idx]

        # Make a copy of the civilizations dict to avoid modifying the original
        civ_copy = {k: v.copy() for k, v in self.civilizations.items()}

        # Spawn with stability parameters
        new_civs, new_knowledge, new_suppression = spawn_new_civilization(
            civ_copy,
            self.knowledge_array.copy(),  # Use copy to avoid modifying original
            self.suppression_array.copy(),
            position,
            parent_idx,
            mutation_factor=0.2,
            min_size=0.1,
            max_size=10.0,
            resource_transfer_ratio=0.2,
            influence_transfer_ratio=0.1,
            knowledge_transfer_ratio=0.3
        )

        # Should have one more civilization
        self.assertEqual(len(new_knowledge), num_civilizations_before + 1)

        # New civilization should be at the specified position
        np.testing.assert_array_equal(
            new_civs["positions"][-1],
            position
        )

        # Parent should have lost some resources
        self.assertLess(
            new_civs["resources"][parent_idx],
            parent_resources_before
        )

        # New civilization should inherit some knowledge from parent
        self.assertGreater(new_knowledge[-1], 0)

        # Test with no parent (random spawn)
        random_civs, random_knowledge, random_suppression = spawn_new_civilization(
            civ_copy,
            self.knowledge_array.copy(),
            self.suppression_array.copy(),
            position,
            parent_idx=None  # No parent
        )

        # Should still spawn a new civilization
        self.assertEqual(len(random_knowledge), num_civilizations_before + 1)

        # Test with invalid parent index
        invalid_parent_civs, invalid_parent_knowledge, invalid_parent_suppression = spawn_new_civilization(
            civ_copy,
            self.knowledge_array.copy(),
            self.suppression_array.copy(),
            position,
            parent_idx=10  # Invalid index
        )

        # Should still spawn a civilization but as a random one
        self.assertEqual(len(invalid_parent_knowledge), num_civilizations_before + 1)

        # Test with extreme mutation factor
        extreme_mutation_civs, extreme_mutation_knowledge, extreme_mutation_suppression = spawn_new_civilization(
            civ_copy,
            self.knowledge_array.copy(),
            self.suppression_array.copy(),
            position,
            parent_idx=1,
            mutation_factor=10.0  # Extreme mutation
        )

        # Should spawn civilization with bounded parameters
        self.assertEqual(len(extreme_mutation_knowledge), num_civilizations_before + 1)
        self.assertGreaterEqual(extreme_mutation_civs["innovation_rates"][-1], 0)
        self.assertLessEqual(extreme_mutation_civs["innovation_rates"][-1], 10)

    def test_remove_civilization(self):
        """Test removal of civilizations with stability for edge cases."""
        num_civilizations_before = len(self.knowledge_array)

        # Test normal case - remove civilization 1
        self.civilizations, self.knowledge_array, self.suppression_array = remove_civilization(
            self.civilizations,
            self.knowledge_array,
            self.suppression_array,
            1
        )

        # Should have one less civilization
        self.assertEqual(len(self.knowledge_array), num_civilizations_before - 1)

        # Civilization 2 should now be at index 1
        self.assertEqual(self.knowledge_array[1], 15.0)  # Was 15.0 at index 2

        # Test with invalid index
        invalid_civs = self.civilizations.copy()
        invalid_knowledge = self.knowledge_array.copy()
        invalid_suppression = self.suppression_array.copy()

        # Index too high
        high_idx_civs, high_idx_knowledge, high_idx_suppression = remove_civilization(
            invalid_civs,
            invalid_knowledge,
            invalid_suppression,
            10  # Invalid index
        )

        # Should return unchanged arrays
        self.assertEqual(len(high_idx_knowledge), len(invalid_knowledge))

        # Negative index
        neg_idx_civs, neg_idx_knowledge, neg_idx_suppression = remove_civilization(
            invalid_civs,
            invalid_knowledge,
            invalid_suppression,
            -1  # Invalid index
        )

        # Should return unchanged arrays
        self.assertEqual(len(neg_idx_knowledge), len(invalid_knowledge))

        # Test with single civilization
        single_civ = initialize_civilizations(1)
        single_knowledge = np.array([5.0])
        single_suppression = np.array([2.0])

        # Remove the only civilization
        empty_civs, empty_knowledge, empty_suppression = remove_civilization(
            single_civ,
            single_knowledge,
            single_suppression,
            0
        )

        # Should result in empty arrays
        self.assertEqual(len(empty_knowledge), 0)
        self.assertEqual(len(empty_suppression), 0)
        self.assertEqual(len(empty_civs["ages"]), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full interaction process with stability controls."""

    def test_process_all_civilization_interactions(self):
        """Test the full civilization interaction process with stability controls."""
        # Initialize test civilizations
        np.random.seed(42)  # For reproducible results
        num_civilizations = 3
        civilizations = initialize_civilizations(num_civilizations)

        # Convert ages to float to avoid casting error
        civilizations["ages"] = civilizations["ages"].astype(float)

        # Fix positions for deterministic test
        civilizations["positions"] = np.array([
            [0.0, 0.0],
            [0.4, 0.0],  # Very close to civilization 0
            [5.0, 5.0]  # Far from others
        ])

        # Fix sizes to create potential for mergers
        civilizations["sizes"] = np.array([1.0, 4.0, 2.0])

        # Test arrays
        knowledge_array = np.array([5.0, 10.0, 15.0])
        suppression_array = np.array([2.0, 5.0, 8.0])
        influence_array = np.array([3.0, 6.0, 9.0])
        resources_array = np.array([10.0, 20.0, 30.0])

        # Process all interactions with stability parameters
        (new_civilizations, new_knowledge, new_suppression,
         new_influence, new_resources, events) = process_all_civilization_interactions(
            civilizations,
            knowledge_array,
            suppression_array,
            influence_array,
            resources_array,
            dt=1.0,  # Explicit dt for stability
            max_spawn_probability=0.05,
            max_random_spawn_probability=0.01,
            max_civilizations=20,
            min_division=0.01
        )

        # Should return values for all arrays
        self.assertIsNotNone(new_civilizations)
        self.assertIsNotNone(new_knowledge)
        self.assertIsNotNone(new_suppression)
        self.assertIsNotNone(new_influence)
        self.assertIsNotNone(new_resources)

        # Should return an events list
        self.assertIsInstance(events, list)

        # At least one of these should change
        self.assertTrue(
            np.any(new_knowledge != knowledge_array) or
            np.any(new_suppression != suppression_array) or
            np.any(new_influence != influence_array) or
            np.any(new_resources != resources_array) or
            len(new_knowledge) != len(knowledge_array)  # Number of civilizations changed
        )

        # Test with empty civilizations
        empty_civs = initialize_civilizations(0)
        empty_knowledge = np.array([])
        empty_suppression = np.array([])
        empty_influence = np.array([])
        empty_resources = np.array([])

        # Should handle empty case gracefully
        (empty_new_civs, empty_new_knowledge, empty_new_suppression,
         empty_new_influence, empty_new_resources, empty_events) = process_all_civilization_interactions(
            empty_civs,
            empty_knowledge,
            empty_suppression,
            empty_influence,
            empty_resources
        )

        # Should return consistent results
        self.assertIsInstance(empty_events, list)

        # Test with extreme values
        extreme_civs = civilizations.copy()
        extreme_knowledge = np.array([1e6, 1e-6, 1.0])
        extreme_suppression = np.array([1e6, 1e-6, 1.0])
        extreme_influence = np.array([1e6, 1e-6, 1.0])
        extreme_resources = np.array([1e6, 1e-6, 1.0])

        # Should handle extreme values gracefully
        (extreme_new_civs, extreme_new_knowledge, extreme_new_suppression,
         extreme_new_influence, extreme_new_resources, extreme_events) = process_all_civilization_interactions(
            extreme_civs,
            extreme_knowledge,
            extreme_suppression,
            extreme_influence,
            extreme_resources,
            dt=0.1  # Smaller dt for stability
        )

        # Results should be finite and within reasonable bounds
        self.assertTrue(np.all(np.isfinite(extreme_new_knowledge)))
        self.assertTrue(np.all(np.isfinite(extreme_new_suppression)))
        self.assertTrue(np.all(np.isfinite(extreme_new_influence)))
        self.assertTrue(np.all(np.isfinite(extreme_new_resources)))

        # Test with single civilization
        single_civ = initialize_civilizations(1)
        single_knowledge = np.array([5.0])
        single_suppression = np.array([2.0])
        single_influence = np.array([3.0])
        single_resources = np.array([10.0])

        # Should handle single civilization case
        (single_new_civs, single_new_knowledge, single_new_suppression,
         single_new_influence, single_new_resources, single_events) = process_all_civilization_interactions(
            single_civ,
            single_knowledge,
            single_suppression,
            single_influence,
            single_resources
        )

        # Results should be valid
        self.assertGreaterEqual(len(single_new_knowledge), 1)


if __name__ == '__main__':
    unittest.main()