import numpy as np


# Electromagnetism Analogy: Knowledge Field Influence
def knowledge_field_influence(K_i, K_j, r_ij, kappa=0.05):
    """
    Calculates the electromagnetic-like influence of knowledge fields between agents.

    Parameters:
        K_i (float): Knowledge state of agent i
        K_j (float): Knowledge state of agent j
        r_ij (float): Relational or conceptual distance between agents
        kappa (float): Knowledge permeability constant (analogous to electromagnetic constant)

    Returns:
        float: Knowledge field influence (analogous to electromagnetic force)
    """
    # Prevent division by zero
    if r_ij < 0.01:
        r_ij = 0.01

    # Coulomb's Law analog for knowledge field influence
    return kappa * K_i * K_j / (r_ij ** 2)


# Quantum Entanglement Analogy: Knowledge State Correlation
def quantum_entanglement_correlation(K_i, K_j, rho=0.1, sigma=0.05):
    """
    Models quantum entanglement-like correlations between agent knowledge states.

    Parameters:
        K_i (float): Knowledge state of agent i
        K_j (float): Knowledge state of agent j
        rho (float): Maximum entanglement strength
        sigma (float): Entanglement decay rate based on knowledge difference

    Returns:
        float: Entanglement correlation factor (nonlocal connection strength)
    """
    # Calculate knowledge difference
    K_diff = abs(K_i - K_j)

    # Exponential decay of entanglement with increasing difference
    return rho * np.exp(-sigma * K_diff)


# Field Gradient: Directional Knowledge Flow
def knowledge_field_gradient(agent_knowledge, agent_positions, field_strength=0.1):
    """
    Calculates knowledge field gradients in conceptual space.

    Parameters:
        agent_knowledge (array): Knowledge values for all agents
        agent_positions (array): Conceptual positions of agents in knowledge space
        field_strength (float): Baseline field strength

    Returns:
        array: Gradient vector indicating direction and strength of knowledge flow
    """
    num_agents = len(agent_knowledge)
    gradients = np.zeros_like(agent_positions)

    for i in range(num_agents):
        for j in range(num_agents):
            if i != j:
                # Calculate direction vector
                direction = agent_positions[j] - agent_positions[i]
                distance = max(np.linalg.norm(direction), 0.01)
                # Normalize
                direction = direction / distance
                # Knowledge difference determines strength and sign
                k_diff = agent_knowledge[j] - agent_knowledge[i]
                # Apply inverse square law
                gradients[i] += direction * field_strength * k_diff / (distance ** 2)

    return gradients


# Entanglement Network: Multi-agent Correlation Matrix
def build_entanglement_network(agent_knowledge, max_entanglement=0.2, decay_rate=0.1):
    """
    Builds a correlation matrix representing quantum-like entanglement between agents.

    Parameters:
        agent_knowledge (array): Knowledge values for all agents
        max_entanglement (float): Maximum possible entanglement strength
        decay_rate (float): Rate at which entanglement decays with knowledge difference

    Returns:
        array: Matrix of entanglement strengths between all agent pairs
    """
    num_agents = len(agent_knowledge)
    entanglement_matrix = np.zeros((num_agents, num_agents))

    for i in range(num_agents):
        for j in range(num_agents):
            if i != j:
                K_diff = abs(agent_knowledge[i] - agent_knowledge[j])
                entanglement_matrix[i, j] = max_entanglement * np.exp(-decay_rate * K_diff)
            else:
                entanglement_matrix[i, j] = 1.0  # Self-entanglement is always 1

    return entanglement_matrix


# Quantum Tunneling: Breakthrough Probability
def quantum_tunneling_probability(barrier_height, barrier_width, energy_level):
    """
    Calculates the probability of breakthrough in knowledge/truth adoption.

    Parameters:
        barrier_height (float): Height of suppression barrier
        barrier_width (float): Width of suppression barrier (resistance over time)
        energy_level (float): Current knowledge/intelligence energy level

    Returns:
        float: Probability of tunneling through suppression barrier
    """
    # Simple quantum tunneling formula (approximated)
    if energy_level >= barrier_height:
        return 1.0  # No tunneling needed if energy exceeds barrier

    # Simplified tunneling probability formula
    tunneling_constant = 0.5  # Adjustable constant
    exponent = -tunneling_constant * barrier_width * np.sqrt(barrier_height - energy_level)

    return np.exp(exponent)