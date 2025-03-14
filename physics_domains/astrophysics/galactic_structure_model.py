import numpy as np

def galactic_structure_model(num_agents, core_influence=2.0, arm_strength=0.5,
                             max_influence=5.0, min_influence=0.0):
    """Models societal structure similar to galactic formation,
with core and peripheral agents.

Parameters:
    num_agents (int): Number of agents in society
    core_influence (float): Strength of core knowledge influence
    arm_strength (float): Strength of "spiral arm" connections
    max_influence (float): Maximum influence value for stability
    min_influence (float): Minimum influence value

Returns:
    array: Influence matrix between agents

Physics Domain: Astrophysics
Scale Level: Civilization
Application Domains: Intelligence, Knowledge
    """
    # Apply parameter bounds
    num_agents = max(1, int(num_agents))
    core_influence = max(min_influence, min(max_influence, core_influence))
    arm_strength = max(min_influence, min(max_influence, arm_strength))

    # Create an influence matrix
    influence_matrix = np.zeros((num_agents, num_agents))

    # Core-periphery structure
    core_size = max(1, int(num_agents * 0.2))  # Core is ~20% of agents

    # Core agents have stronger mutual influence
    for i in range(core_size):
        for j in range(core_size):
            if i != j:
                influence_matrix[i, j] = core_influence

    # Spiral arm structure - influenced by nearest neighbors and core
    for i in range(core_size, num_agents):
        # Connect to nearest neighbors in a ring, ensuring indices stay in valid range
        prev_neighbor = core_size + ((i - core_size - 1) % max(1, (num_agents - core_size)))
        next_neighbor = core_size + ((i - core_size + 1) % max(1, (num_agents - core_size)))

        influence_matrix[i, prev_neighbor] = arm_strength
        influence_matrix[i, next_neighbor] = arm_strength

        # Connect to a random core agent
        core_connection = np.random.randint(0, core_size)
        influence_matrix[i, core_connection] = arm_strength * 2
        influence_matrix[core_connection, i] = arm_strength

    # Ensure all values are within bounds
    influence_matrix = np.clip(influence_matrix, min_influence, max_influence)

    return influence_matrix