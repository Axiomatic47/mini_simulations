import numpy as np

def multi_civilization_to_cosmic_bridge(multi_civ_systems, cosmic_time):
    """Bridges multi-civilization dynamics to cosmic scale phenomena.

Physics Domain: astrophysics
Scale Levels: multi_civilization, cosmic
Application Domains: knowledge, civilization

Parameters:
    multi_civ_systems (list): List of multi-civilization system properties
    cosmic_time (float): Current universal time scale
    
Returns:
    dict: Cosmic-level properties including universal knowledge metrics
Scale Level: Multi Civilization, Cosmic"""
    # Validate inputs
    if multi_civ_systems is None or len(multi_civ_systems) == 0:
        return {"universal_knowledge": 0.0, "knowledge_density": 0.0}
    
    # Extract knowledge properties from each multi-civilization system
    total_knowledge_values = [system["total_knowledge"] for system in multi_civ_systems]
    max_knowledge_values = [system["max_knowledge"] for system in multi_civ_systems]
    inequality_values = [system["knowledge_inequality"] for system in multi_civ_systems]
    
    # Calculate universal metrics
    universal_knowledge = np.sum(total_knowledge_values)
    universal_max_knowledge = np.max(max_knowledge_values)
    average_inequality = np.mean(inequality_values)
    
    # Calculate knowledge density - increases with cosmic time
    # (early universe has lower knowledge density)
    cosmic_age_factor = min(1.0, cosmic_time / 13.8)  # Normalized to cosmic age in billions of years
    knowledge_density = universal_knowledge * cosmic_age_factor / 100.0
    
    # Calculate cosmic-level emergence effects
    emergence_factor = 1.0 + 0.1 * np.log1p(len(multi_civ_systems))
    
    return {
        "universal_knowledge": universal_knowledge * emergence_factor,
        "universal_max_knowledge": universal_max_knowledge,
        "knowledge_density": knowledge_density,
        "average_inequality": average_inequality,
        "number_of_systems": len(multi_civ_systems)
    }
