import numpy as np

def identity_binding(agent_knowledge, agent_identity_strength, binding_factor=0.5, max_binding=5.0):
    """
    Models how agent identity binds to knowledge through strong nuclear-like forces.
    
    Physics Domain: strong_nuclear
    Scale Level: agent
    Application Domains: knowledge, identity
    
    Parameters:
        agent_knowledge (float): Current agent knowledge level
        agent_identity_strength (float): Strength of agent's identity
        binding_factor (float): Factor controlling binding strength
        max_binding (float): Maximum binding strength
        
    Returns:
        float: Binding strength between knowledge and identity
    """
    # Apply safe bounds to parameters
    agent_knowledge = max(0.0, agent_knowledge)
    agent_identity_strength = max(0.0, agent_identity_strength)
    binding_factor = max(0.0, min(1.0, binding_factor))
    
    # Calculate binding strength using strong force analogy
    # Strong nuclear force increases with distance up to a point, then drops off rapidly
    relative_strength = agent_knowledge / max(0.1, agent_identity_strength)
    
    if relative_strength < 1.0:
        # Distance within effective range - binding increases with separation
        binding_strength = binding_factor * relative_strength * agent_identity_strength
    else:
        # Distance beyond effective range - binding drops exponentially
        binding_strength = binding_factor * agent_identity_strength * np.exp(1.0 - relative_strength)
    
    # Apply maximum binding cap
    return min(max_binding, binding_strength)
