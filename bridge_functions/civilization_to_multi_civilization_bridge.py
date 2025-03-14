import numpy as np

def civilization_to_multi_civilization_bridge(civ_knowledge_array, civ_positions, interaction_range=10.0):
    """Bridges civilization-level dynamics to multi-civilization scale interactions.

Physics Domain: astrophysics
Scale Levels: civilization, multi_civilization
Application Domains: knowledge, civilization

Parameters:
    civ_knowledge_array (array): Knowledge levels of different civilizations
    civ_positions (array): Spatial positions of civilizations
    interaction_range (float): Maximum distance for civilization interactions
    
Returns:
    dict: Multi-civilization system properties including knowledge distribution
Scale Level: Civilization, Multi Civilization"""
    # Validate inputs
    if civ_knowledge_array is None or len(civ_knowledge_array) == 0:
        return {"total_knowledge": 0.0, "knowledge_distribution": [], "interaction_matrix": []}
    
    n_civs = len(civ_knowledge_array)
    
    # Calculate distances between civilizations
    interaction_matrix = np.zeros((n_civs, n_civs))
    for i in range(n_civs):
        for j in range(n_civs):
            if i != j:
                distance = np.linalg.norm(np.array(civ_positions[i]) - np.array(civ_positions[j]))
                # Interaction strength decreases with distance
                if distance < interaction_range:
                    interaction_matrix[i, j] = 1.0 / (1.0 + distance)
    
    # Calculate knowledge distribution metrics
    total_knowledge = np.sum(civ_knowledge_array)
    max_knowledge = np.max(civ_knowledge_array) if len(civ_knowledge_array) > 0 else 0
    knowledge_inequality = np.std(civ_knowledge_array) / max(0.001, np.mean(civ_knowledge_array)) if len(civ_knowledge_array) > 0 else 0
    
    # Calculate properties of multi-civilization system
    return {
        "total_knowledge": total_knowledge,
        "max_knowledge": max_knowledge,
        "knowledge_inequality": knowledge_inequality,
        "interaction_matrix": interaction_matrix,
        "knowledge_distribution": civ_knowledge_array
    }
