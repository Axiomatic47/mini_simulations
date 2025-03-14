import numpy as np

def civilization_oscillation(age, base_intensity, oscillation_frequency=0.05, 
                             damping_factor=0.01, amplitude=1.0):
    """
    Models oscillation patterns in civilization development analogous to strong nuclear interactions.
    
    Physics Domain: strong_nuclear
    Scale Level: civilization
    Application Domains: civilization
    
    Parameters:
        age (float): Current age of the civilization
        base_intensity (float): Base intensity of civilization's activity
        oscillation_frequency (float): Frequency of oscillation cycles
        damping_factor (float): How quickly oscillations dampen with time
        amplitude (float): Initial amplitude of oscillations
        
    Returns:
        float: Oscillation intensity at current age
    """
    # Apply bounds to parameters
    age = max(0.0, age)
    base_intensity = max(0.0, base_intensity)
    oscillation_frequency = max(0.001, oscillation_frequency)
    damping_factor = max(0.0, min(1.0, damping_factor))
    amplitude = max(0.0, amplitude)
    
    # Calculate damping envelope
    damping = np.exp(-damping_factor * age)
    
    # Calculate oscillation
    oscillation = amplitude * damping * np.sin(oscillation_frequency * age)
    
    # Apply to base intensity (ensuring result remains positive)
    result = base_intensity * (1.0 + oscillation)
    
    return max(0.1, result)
