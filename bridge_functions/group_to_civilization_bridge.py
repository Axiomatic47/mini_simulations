import numpy as np

def group_to_civilization_bridge(group_knowledge_array, group_interaction_matrix=None):
    """Bridges group-level knowledge to civilization-level collective intelligence.

Physics Domain: weak_nuclear
Scale Levels: group, civilization
Application Domains: knowledge, intelligence

Parameters:
    group_knowledge_array (array): Knowledge levels of different groups
    group_interaction_matrix (array): Interaction strengths between groups
    
Returns:
    float: Civilization-level collective intelligence
Scale Level: Group, Civilization"""
    # Validate inputs
    if group_knowledge_array is None or len(group_knowledge_array) == 0:
        return 0.0
    
    # If no interaction matrix provided, use default
    if group_interaction_matrix is None:
        n_groups = len(group_knowledge_array)
        group_interaction_matrix = np.ones((n_groups, n_groups)) - np.eye(n_groups)
    
    # Calculate base knowledge contribution
    base_knowledge = np.mean(group_knowledge_array)
    
    # Calculate interaction effect
    interaction_effect = 0.0
    for i in range(len(group_knowledge_array)):
        for j in range(len(group_knowledge_array)):
            if i != j:
                # Groups with similar knowledge levels enhance each other
                knowledge_similarity = 1.0 / (1.0 + np.abs(group_knowledge_array[i] - group_knowledge_array[j]))
                interaction_effect += group_interaction_matrix[i, j] * knowledge_similarity
    
    # Normalize interaction effect
    n_groups = len(group_knowledge_array)
    if n_groups > 1:
        interaction_effect /= (n_groups * (n_groups - 1))
    
    # Combine base knowledge with interaction bonus
    civilization_intelligence = base_knowledge * (1.0 + 0.2 * interaction_effect)
    
    return civilization_intelligence
