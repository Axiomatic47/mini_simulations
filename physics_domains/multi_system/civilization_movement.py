"""
Update positions of civilizations based on attractive and repulsive forces.
"""

import numpy as np


def civilization_movement(civilizations, interaction_strength, dt=1.0,
                          attraction_factor=0.01, repulsion_threshold=1.0,
                          max_velocity=1.0, damping=0.9):
    """
    Update positions of civilizations based on attractive and repulsive forces.

    Physics Domain: multi_system
    Scale Level: multi_civilization
    Application Domains: civilization

    Parameters:
        civilizations (dict): Civilization parameters
        interaction_strength (array): Matrix of interaction strengths
        dt (float): Time step
        attraction_factor (float): Strength of attraction between civilizations
        repulsion_threshold (float): Distance threshold for repulsion
        max_velocity (float): Maximum velocity for stability
        damping (float): Velocity damping factor (0-1)

    Returns:
        array: Updated positions for all civilizations
    """
    # Apply parameter bounds
    dt = max(0.01, min(2.0, dt))
    damping = max(0, min(1, damping))

    num_civilizations = len(civilizations["positions"])

    # Handle empty case
    if num_civilizations == 0:
        return civilizations["positions"]

    # Calculate forces between civilizations
    forces = np.zeros((num_civilizations, 2))

    for i in range(num_civilizations):
        for j in range(num_civilizations):
            if i != j:
                # Calculate direction vector
                direction = civilizations["positions"][j] - civilizations["positions"][i]
                distance = np.linalg.norm(direction)

                # Normalize direction
                if distance > 0:
                    direction = direction / distance

                # Calculate attractive force based on influence and knowledge
                attraction = (attraction_factor *
                              interaction_strength[i, j] *
                              civilizations["influence"][j] *
                              civilizations["expansion_tendency"][i])

                # Calculate repulsive force if civilizations are too close
                repulsion = 0
                if distance < repulsion_threshold:
                    repulsion = -attraction_factor * (repulsion_threshold - distance) * 5

                # Combined force
                net_force = (attraction + repulsion) * direction
                forces[i] += net_force

    # Update velocities (with damping)
    civilizations["velocities"] = damping * civilizations["velocities"] + forces * dt

    # Apply velocity limits for stability
    velocity_magnitudes = np.linalg.norm(civilizations["velocities"], axis=1)
    for i in range(num_civilizations):
        if velocity_magnitudes[i] > max_velocity:
            civilizations["velocities"][i] = (civilizations["velocities"][i] /
                                              velocity_magnitudes[i] * max_velocity)

    # Update positions
    civilizations["positions"] += civilizations["velocities"] * dt

    return civilizations["positions"]