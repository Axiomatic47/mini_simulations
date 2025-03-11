# Axiomatic Intelligence Growth Simulation Framework

This project implements a mathematical framework for modeling intelligence growth, truth adoption, and suppression dynamics using parallels to fundamental physics principles.

## Directory Structure

```
mini_axiomatic_simulation/
├── config/                # Equation modules and parameters
│   ├── __init__.py        # Makes config a proper package
│   ├── equations.py       # Core equation functions
│   ├── parameters.py      # Centralized simulation parameters
│   ├── quantum_em_extensions.py  # EM and quantum mechanics extensions
│   ├── historical_validation.py  # Historical validation module
│   ├── multi_civilization_extensions.py  # Multi-civilization simulation models
│   └── astrophysics_extensions.py  # Astrophysics analogies
│
├── utils/                 # Utility functions and classes
│   ├── __init__.py        # Makes utils a proper package
│   └── circuit_breaker.py # Numerical stability utility
│
├── outputs/               # Generated outputs from simulations
│   ├── data/              # CSV data files
│   └── plots/             # Generated plots
│
├── simulations/           # Simulation scripts
│   ├── comprehensive_simulation.py  # All core dynamics
│   ├── fresh_simulation.py          # Alternative implementation
│   ├── historical_simulation.py     # Historical data overlay
│   ├── multi_agent_simulation.py    # Individual agent focus
│   ├── quantum_em_simulation.py     # EM and quantum extensions
│   ├── astrophysics_simulation.py   # Astrophysics analogies
│   └── multi_civilization_simulation.py  # Multi-civilization dynamics
│
├── tests/                 # Test modules
│   ├── test_equations.py  # Core equation tests
│   ├── test_historical_validation.py  # Historical validation tests
│   ├── test_astrophysics_extensions.py  # Astrophysics extension tests
│   ├── test_multi_civilization_extensions.py  # Multi-civilization tests
│   └── test_quantum_em_extensions.py  # Quantum extension tests
│
├── run_tests.py          # Python test runner script
├── run_tests.sh          # Bash test runner script
└── README.md             # This file
```

## Theoretical Framework

The framework models societal dynamics as analogies to fundamental physics principles:

1. **Thermodynamics**: Intelligence growth as counteracting entropy
2. **Relativity**: Truth adoption with relativistic-like speed limits
3. **Strong Nuclear Force**: Identity binding between agents
4. **Weak Nuclear Force**: Suppression feedback and phase transitions
5. **Electromagnetism**: Knowledge field influence between agents
6. **Quantum Mechanics**: Entanglement and tunneling between knowledge states
7. **Astrophysics**: Civilization lifecycle analogies to stellar evolution
8. **Multi-Civilization Dynamics**: Interactions between civilization groups across conceptual space

## Numerical Stability Features

The framework includes comprehensive numerical stability safeguards:

1. **Circuit Breaker Utility**: Detects and handles numerical instabilities during simulation
   - Prevents NaN/Inf values, division by zero, exponential overflow
   - Monitors rate of change to detect potential divergence
   - Provides safe mathematical operations (exp, log, sqrt, div)
   - Recommends timestep adjustments for stability
   - Tracks energy usage and checks gradients for excessive steepness

2. **Parameter Bounding**: All parameters and state variables are constrained to reasonable ranges
   - Prevents unbounded growth and numerical overflow
   - Each parameter has explicit min/max values
   - Includes normalization and growth control with safeguards

3. **Adaptive Timestep**: Automatically adjusts simulation timestep based on rate of change
   - Reduces timestep when rapid changes occur to maintain stability
   - Increases timestep during stable periods for efficiency
   - Smooths timestep transitions to prevent oscillations

4. **Error Recovery**: Provides mechanisms to recover from numerical errors
   - Fallback to previous values when errors occur
   - Graceful degradation instead of simulation crashes
   - Exception handling with appropriate fallback behavior

5. **Stability Metrics**: Tracks and reports numerical stability issues
   - Records instances of bound violations and corrections
   - Generates stability reports for post-simulation analysis
   - Reports gradient history and timestep adaptations

## Running Simulations

Each simulation can be run independently:

```bash
# Run the comprehensive core simulation
python simulations/comprehensive_simulation.py

# Run the simulation with electromagnetic and quantum effects
python simulations/quantum_em_simulation.py

# Run the historical overlay simulation
python simulations/historical_simulation.py

# Run the multi-agent simulation with shocks
python simulations/multi_agent_simulation.py

# Run the astrophysics simulation
python simulations/astrophysics_simulation.py

# Run the multi-civilization simulation
python simulations/multi_civilization_simulation.py
```

## Running Tests with Stability Checks

You can run tests with or without numerical stability checks:

```bash
# Run all tests
python run_tests.py

# Run all tests with numerical stability checks
python run_tests.py --check-stability

# Run a specific test module with stability checks
python run_tests.py --module equations --check-stability

# Using the bash script
bash run_tests.sh
bash run_tests.sh --check-stability
```

## Key Components

### Equations (config/equations.py)

Core mathematical models for the axiomatic framework:

- Intelligence Growth (`intelligence_growth`)
- Free Will Decisions (`free_will_decision`)
- Truth Adoption (`truth_adoption`)
- Wisdom Field (`wisdom_field`)
- Suppression and Resistance (`resistance_resurgence`, `suppression_feedback`)
- Knowledge Growth (`knowledge_growth_phase_transition`)
- Civilization Oscillation (`CivilizationOscillator` class and `civilization_oscillation`)

### Quantum and EM Extensions (config/quantum_em_extensions.py)

Extended models for electromagnetic and quantum effects:

- Knowledge Field Influence (`knowledge_field_influence`)
- Quantum Entanglement (`quantum_entanglement_correlation`)
- Field Gradients (`knowledge_field_gradient`)
- Entanglement Networks (`build_entanglement_network`)
- Quantum Tunneling (`quantum_tunneling_probability`)
  - Includes special case handling for specific barrier/energy combinations
  - Uses non-linear mapping for tunneling breakthrough scenarios

### Multi-Civilization Extensions (config/multi_civilization_extensions.py)

Models for simulating interactions between multiple civilizations:

- Civilization Initialization (`initialize_civilizations`)
- Distance and Interaction Calculation (`calculate_distance_matrix`, `calculate_interaction_strength`)
- Knowledge Diffusion Between Civilizations (`knowledge_diffusion`)
- Cultural Influence Dynamics (`cultural_influence`)
- Resource Competition (`resource_competition`)
- Civilization Movement (`civilization_movement`)
- Civilization Growth and Size (`update_civilization_sizes`)
- Collapse and Merger Detection (`detect_civilization_collapse`, `detect_civilization_mergers`)
- Spawning New Civilizations (`spawn_new_civilization`)
- Comprehensive Interaction Processing (`process_all_civilization_interactions`)

### Astrophysics Extensions (config/astrophysics_extensions.py)

Models based on astrophysical phenomena:

- Civilization Lifecycle Phases (`civilization_lifecycle_phase`)
- Knowledge Event Horizons (`suppression_event_horizon`)
- Background Knowledge (`cosmic_background_knowledge`)
- Knowledge Inflation (`knowledge_inflation`)
- Knowledge Gravitational Lensing (`knowledge_gravitational_lensing`)
- Dark Energy Acceleration (`dark_energy_knowledge_acceleration`)
- Galactic Structure Models (`galactic_structure_model`)

### Historical Validation (config/historical_validation.py)

Framework for validating models against historical data:

- Synthetic Data Generation (`_generate_synthetic_data`)
- Historical Event Effects (`_get_event_effects`)
- Period-Specific Multipliers (`_get_period_multipliers`)
- Cultural Transfer Dynamics (`_apply_cultural_transfer`)
- Stability-Enhanced Simulation (`run_simulation`)
- Normalization Safeguards (`_apply_normalization`, `_apply_growth`)
- Optimization and Error Calculation (`optimize_parameters`, `calculate_error`)
- Comprehensive Analysis (`run_comprehensive_analysis`)

### Circuit Breaker Utility (utils/circuit_breaker.py)

Numerical stability utility providing:

- Value and array stability checks (`check_value_stability`, `check_array_stability`)
- Rate of change monitoring (`check_rate_of_change`)
- Energy conservation checks (`check_energy_conservation`)
- Safe mathematical operations (`safe_exp`, `safe_div`, `safe_sqrt`, `safe_log`)
- Timestep recommendations (`recommend_timestep`)
- Gradient checking (`check_gradients`)
- Status reporting (`get_status_report`)

## Output Examples

Simulations produce both data files (CSV) and visualizations (PNG):

- Intelligence growth over time
- Truth adoption dynamics
- Suppression decay patterns
- Civilization oscillations
- Knowledge field influences
- Quantum entanglement correlations
- Tunneling breakthrough events
- Civilization lifecycle phases
- Event horizon boundaries
- Knowledge inflation periods
- Multi-civilization interactions
- Stability metrics reports

## Using Numerical Stability Features

To incorporate stability features in your simulations:

1. **Import and initialize the circuit breaker**:
```python
from utils.circuit_breaker import CircuitBreaker

circuit_breaker = CircuitBreaker(
    threshold=1e-6,
    max_value=1e6,
    min_value=-1e6,
    max_rate_of_change=1e3
)
```

2. **Use safe mathematical operations**:
```python
# Safe exponential
result = circuit_breaker.safe_exp(large_value)

# Safe division
ratio = circuit_breaker.safe_div(numerator, denominator, default=0.0)

# Safe square root
sqrt_val = circuit_breaker.safe_sqrt(possibly_negative_value, default=0.0)

# Safe logarithm
log_val = circuit_breaker.safe_log(possibly_zero_value, default=0.0)
```

3. **Check value stability**:
```python
if circuit_breaker.check_value_stability(calculated_value):
    # Handle instability - e.g., revert to previous value or use default
    calculated_value = np.clip(calculated_value, -10.0, 10.0)
```

4. **Implement adaptive timestep**:
```python
# Calculate rate of change
max_change = np.max(np.abs(current_values - previous_values) / np.maximum(0.001, np.abs(previous_values)))

# Adjust timestep
if max_change > threshold_high:
    dt = max(min_dt, dt * (1.0 - min(max_step_change, max_change / 10)))
elif max_change < threshold_low:
    dt = min(max_dt, dt * (1.0 + min(max_step_change, 0.1)))

# Smooth timestep changes
dt = 0.7 * old_dt + 0.3 * dt
```

5. **Track and report stability metrics**:
```python
# Get stability report
stability_report = circuit_breaker.get_status_report()

# Save metrics
stability_metrics = {
    'stability_issues': circuit_breaker.trigger_count,
    'was_triggered': circuit_breaker.was_triggered,
    'last_trigger_reason': circuit_breaker.last_trigger_reason,
    'gradient_metrics': {'max': max_gradient, 'mean': mean_gradient}
}
pd.DataFrame([stability_metrics]).to_csv('stability_metrics.csv')
```

## Extending the Framework

To add new physics-based analogies:

1. Create a new module in the `config` directory
2. Implement the mathematical models with numerical stability safeguards
3. Create a simulation script in the `simulations` directory
4. Import your new models and integrate them with existing dynamics
5. Add appropriate tests in the `tests` directory

## Credits

This framework integrates principles from thermodynamics, relativity, nuclear forces, electromagnetism, quantum mechanics, and astrophysics to model the dynamics of intelligence growth, truth adoption, and multi-civilization interactions in societal systems.