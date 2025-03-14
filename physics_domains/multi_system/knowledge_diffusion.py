"""
Model knowledge diffusion between civilizations with numerical stability safeguards.
"""

import numpy as np


def knowledge_diffusion(civilizations, knowledge_array, interaction_strength,
                        diffusion_rate=0.01, max_diffusion=0.5, min_division=1e-10):
    """
    Model knowledge diffusion between civilizations with numerical stability safeguards.

    Physics Domain: multi_system
    Scale Level: multi_civilization
    Application Domains: knowledge

    Parameters:
        civilizations (dict): Civilization parameters
        knowledge_array (array): Current knowledge levels for all civilizations
        interaction_strength (array): Matrix of interaction strengths
        diffusion_rate (float): Base rate of knowledge diffusion
        max_diffusion (float): Maximum diffusion amount per time step
        min_division (float): Minimum value to prevent division by zero

    Returns:
        array: Knowledge change due to diffusion
    """
    # Apply parameter bounds
    diffusion_rate = max(0, min(1, diffusion_rate))

    # Get dimensions and ensure they match
    num_civilizations = len(knowledge_array)
    knowledge_change = np.zeros(num_civilizations)

    # Handle empty case
    if num_civilizations == 0:
        return knowledge_change

    # Safety check for interaction_strength dimensions
    if interaction_strength.shape[0] != num_civilizations or interaction_strength.shape[1] != num_civilizations:
        # Resize interaction matrix if dimensions don't match
        temp_interaction = np.zeros((num_civilizations, num_civilizations))
        # Copy available values
        for i in range(min(interaction_strength.shape[0], num_civilizations)):
            for j in range(min(interaction_strength.shape[1], num_civilizations)):
                temp_interaction[i, j] = interaction_strength[i, j]
        interaction_strength = temp_interaction
        print(f"Warning: Interaction strength matrix resized to match {num_civilizations} civilizations")

    # Ensure civilization parameters arrays exist and have correct length
    for key in ["innovation_rates", "knowledge_retention"]:
        if key not in civilizations or len(civilizations[key]) != num_civilizations:
            # Create or resize array with default values
            default_val = 0.5  # middle value as default
            civilizations[key] = np.full(num_civilizations, default_val)
            print(f"Warning: {key} array resized with default values")

    # Calculate knowledge diffusion for each civilization
    for i in range(num_civilizations):
        for j in range(num_civilizations):
            # Skip self-interaction or negative/zero interaction strength
            if i == j or interaction_strength[i, j] <= 0:
                continue

            try:
                # Knowledge flows from higher to lower levels
                knowledge_diff = knowledge_array[j] - knowledge_array[i]

                # Apply diffusion with appropriate bounds
                if knowledge_diff > 0:
                    # Receiving knowledge
                    # Ensure innovation rate is within bounds
                    innovation_rate = np.clip(civilizations["innovation_rates"][i], 0.01, 10.0)

                    # Calculate diffusion amount with bounds
                    diffusion_amount = (diffusion_rate *
                                        interaction_strength[i, j] *
                                        knowledge_diff *
                                        innovation_rate)

                    # Check for NaN or Inf
                    if np.isnan(diffusion_amount) or np.isinf(diffusion_amount):
                        diffusion_amount = max_diffusion * 0.01  # Safe default

                    # Apply maximum diffusion bound
                    knowledge_change[i] += np.clip(diffusion_amount, 0, max_diffusion)
                else:
                    # Giving knowledge - reduced outflow based on knowledge retention
                    # Ensure knowledge retention is within bounds
                    knowledge_retention = np.clip(civilizations["knowledge_retention"][i], 0.0, 1.0)

                    # Calculate diffusion amount with bounds
                    diffusion_amount = (diffusion_rate *
                                        interaction_strength[i, j] *
                                        knowledge_diff *
                                        (1 - knowledge_retention))

                    # Check for NaN or Inf
                    if np.isnan(diffusion_amount) or np.isinf(diffusion_amount):
                        diffusion_amount = -max_diffusion * 0.01  # Safe default

                    # Apply maximum diffusion bound
                    knowledge_change[i] += np.clip(diffusion_amount, -max_diffusion, 0)
            except Exception as e:
                print(f"Warning: Error in knowledge diffusion calculation: {e}")
                # Skip this interaction pair
                continue

    # Final check for NaN or Inf values
    knowledge_change = np.nan_to_num(knowledge_change, nan=0.0, posinf=max_diffusion, neginf=-max_diffusion)

    return knowledge_change