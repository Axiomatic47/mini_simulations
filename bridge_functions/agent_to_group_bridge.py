import numpy as np

def agent_to_group_bridge(agent_knowledge_array, agent_weights=None):
    """Bridges agent-level knowledge to group-level collective knowledge.

Physics Domain: electromagnetism
Scale Levels: agent, group
Application Domains: knowledge, intelligence

Parameters:
    agent_knowledge_array (array): Knowledge levels of individual agents
    agent_weights (array): Optional weights for each agent's contribution
    
Returns:
    float: Group-level collective knowledge
Scale Level: Agent, Group"""
    # Validate inputs
    if agent_knowledge_array is None or len(agent_knowledge_array) == 0:
        return 0.0
    
    # Apply equal weights if none provided
    if agent_weights is None:
        agent_weights = np.ones(len(agent_knowledge_array)) / len(agent_knowledge_array)
    else:
        # Normalize weights to sum to 1
        agent_weights = np.array(agent_weights) / max(1e-10, np.sum(agent_weights))
        
    # Calculate weighted knowledge sum
    collective_knowledge = np.sum(agent_knowledge_array * agent_weights)
    
    # Apply emergent properties - group knowledge can exceed sum of parts
    emergent_factor = 1.0 + 0.1 * np.log1p(len(agent_knowledge_array))
    
    return collective_knowledge * emergent_factor
