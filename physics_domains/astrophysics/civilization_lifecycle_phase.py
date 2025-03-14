import numpy as np

def civilization_lifecycle_phase(age, intensity, phase_thresholds, phase_intensities,
                                 max_intensity=10.0, min_intensity=0.1):
    """
    Models civilization phases similar to stellar evolution.

    Parameters:
        age (float): Current age of civilization
        intensity (float): Base intensity factor for civilization
        phase_thresholds (array): Age thresholds for phase transitions [formation, main_sequence, peak, decline, collapse]
        phase_intensities (array): Intensity modifiers for each phase
        max_intensity (float): Maximum intensity value for stability
        min_intensity (float): Minimum intensity value for stability

    Returns:
        float: Civilization intensity at current age
        int: Current phase of civilization (0-5)

    Physics Domain: astrophysics
    Scale Level: civilization
    Application Domains: civilization, intelligence
    """
    # Apply parameter bounds
    age = max(0, age)
    intensity = max(min_intensity, min(max_intensity, intensity))

    # Ensure phase_thresholds is an array with at least 5 elements
    if len(phase_thresholds) < 5:
        phase_thresholds = np.append(phase_thresholds, [1000] * (5 - len(phase_thresholds)))

    # Ensure phase_intensities is an array with at least 6 elements (for 6 phases)
    if len(phase_intensities) < 6:
        phase_intensities = np.append(phase_intensities, [1.0] * (6 - len(phase_intensities)))

    # Determine current phase based on age
    if age < phase_thresholds[0]:
        # Proto-civilization / Formation (analogous to stellar nebula)
        phase = 0
        progress = age / max(0.1, phase_thresholds[0])
        # Bound progress to prevent overflow
        progress = min(10, progress)
        phase_intensity = phase_intensities[0] * (1 - np.exp(-3 * min(5, progress)))

    elif age < phase_thresholds[1]:
        # Early civilization (analogous to main sequence star)
        phase = 1
        progress = (age - phase_thresholds[0]) / max(0.1, phase_thresholds[1] - phase_thresholds[0])
        # Bound progress to prevent overflow or invalid values
        progress = max(0, min(1, progress))
        phase_intensity = phase_intensities[1] * (1 + 0.5 * np.sin(progress * np.pi))

        # Smooth transition at threshold
        if age >= phase_thresholds[0] - 1 and age <= phase_thresholds[0] + 1:
            # Blend between phase 0 and phase 1 near the threshold
            blend_factor = (age - (phase_thresholds[0] - 1)) / 2.0
            blend_factor = max(0, min(1, blend_factor))  # Ensure between 0 and 1

            phase_0_intensity = phase_intensities[0] * (1 - np.exp(-3))  # Fully developed phase 0
            phase_intensity = (1 - blend_factor) * phase_0_intensity + blend_factor * phase_intensity

    elif age < phase_thresholds[2]:
        # Peak civilization (analogous to giant phase)
        phase = 2
        progress = (age - phase_thresholds[1]) / max(0.1, phase_thresholds[2] - phase_thresholds[1])
        # Bound progress to prevent overflow
        progress = max(0, min(1, progress))
        phase_intensity = phase_intensities[2] * (1 + progress * (1 - progress) * 4)

        # Smooth transition at threshold
        if age >= phase_thresholds[1] - 1 and age <= phase_thresholds[1] + 1:
            blend_factor = (age - (phase_thresholds[1] - 1)) / 2.0
            blend_factor = max(0, min(1, blend_factor))

            # Calculate phase 1 intensity at full development
            phase_1_progress = 1.0  # End of phase 1
            phase_1_intensity = phase_intensities[1] * (1 + 0.5 * np.sin(phase_1_progress * np.pi))

            phase_intensity = (1 - blend_factor) * phase_1_intensity + blend_factor * phase_intensity

    elif age < phase_thresholds[3]:
        # Declining civilization (analogous to stellar contraction)
        phase = 3
        progress = (age - phase_thresholds[2]) / max(0.1, phase_thresholds[3] - phase_thresholds[2])
        # Bound progress to prevent overflow
        progress = max(0, min(1, progress))
        phase_intensity = phase_intensities[3] * (1 - progress ** 2)

        # Smooth transition
        if age >= phase_thresholds[2] - 1 and age <= phase_thresholds[2] + 1:
            blend_factor = (age - (phase_thresholds[2] - 1)) / 2.0
            blend_factor = max(0, min(1, blend_factor))

            # Calculate phase 2 intensity at full development
            phase_2_intensity = phase_intensities[2] * (1 + 1.0 * (1 - 1.0) * 4)  # At peak

            phase_intensity = (1 - blend_factor) * phase_2_intensity + blend_factor * phase_intensity

    elif age < phase_thresholds[4]:
        # Collapse/Transformation (analogous to supernova)
        phase = 4
        progress = (age - phase_thresholds[3]) / max(0.1, phase_thresholds[4] - phase_thresholds[3])
        # Bound progress to prevent overflow
        progress = max(0, min(1, progress))

        # Brief intensity spike followed by collapse
        if progress < 0.2:
            # Supernova-like flash
            phase_intensity = phase_intensities[4] * (1 + 5 * progress)
        else:
            # Rapid decline
            phase_intensity = phase_intensities[4] * (1 - (progress - 0.2) / max(0.1, 0.8))

        # Smooth transition
        if age >= phase_thresholds[3] - 1 and age <= phase_thresholds[3] + 1:
            blend_factor = (age - (phase_thresholds[3] - 1)) / 2.0
            blend_factor = max(0, min(1, blend_factor))

            # Calculate phase 3 intensity at end
            phase_3_intensity = phase_intensities[3] * (1 - 1.0 ** 2)  # At end of phase 3

            phase_intensity = (1 - blend_factor) * phase_3_intensity + blend_factor * phase_intensity

    else:
        # Remnant/Rebirth (analogous to neutron star/black hole or stellar remnant)
        phase = 5
        time_since_collapse = max(0, age - phase_thresholds[4])
        # Cap exponential input to prevent overflow
        exp_input = min(10, -0.05 * time_since_collapse)
        # Gradual rebirth possibility after sufficient time
        phase_intensity = phase_intensities[5] * (0.1 + 0.9 * (1 - np.exp(exp_input)))

        # Smooth transition
        if age >= phase_thresholds[4] - 1 and age <= phase_thresholds[4] + 1:
            blend_factor = (age - (phase_thresholds[4] - 1)) / 2.0
            blend_factor = max(0, min(1, blend_factor))

            # Calculate phase 4 intensity at end (after complete collapse)
            phase_4_intensity = phase_intensities[4] * (1 - (1.0 - 0.2) / 0.8)  # At end of phase 4

            phase_intensity = (1 - blend_factor) * phase_4_intensity + blend_factor * phase_intensity

    # Apply final bounds to intensity
    phase_intensity = max(min_intensity, min(max_intensity, phase_intensity))

    return intensity * phase_intensity, phase