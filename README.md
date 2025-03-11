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
│   └── historical_validation_improved.py  # Historical validation module
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

## Numerical Stability Features

The framework includes comprehensive numerical stability safeguards:

1. **Circuit Breaker Utility**: Detects and handles numerical instabilities during simulation
   - Prevents NaN/Inf values, division by zero, exponential overflow
   - Monitors rate of change to detect potential divergence
   - Provides safe mathematical operations (exp, log, sqrt, div)
   - Recommends timestep adjustments for stability

2. **Parameter Bounding**: All parameters and state variables are constrained to reasonable ranges
   - Prevents unbounded growth and numerical overflow
   - Each parameter has explicit min/max values

3. **Adaptive Timestep**: Automatically adjusts simulation timestep based on rate of change
   - Reduces timestep when rapid changes occur to maintain stability
   - Increases timestep during stable periods for efficiency

4. **Error Recovery**: Provides mechanisms to recover from numerical errors
   - Fallback to previous values when errors occur
   - Graceful degradation instead of simulation crashes

5. **Stability Metrics**: Tracks and reports numerical stability issues
   - Records instances of bound violations and corrections
   - Generates stability reports for post-simulation analysis

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
- Suppression and Resistance (`resistance_resurgence`, `suppression_feedback`)
- Knowledge Growth (`knowledge_growth_phase_transition`)
- Civilization Oscillation (`civilization_oscillation`)

### Quantum and EM Extensions (config/quantum_em_extensions.py)

Extended models for electromagnetic and quantum effects:

- Knowledge Field Influence (`knowledge_field_influence`)
- Quantum Entanglement (`quantum_entanglement_correlation`)
- Field Gradients (`knowledge_field_gradient`)
- Entanglement Networks (`build_entanglement_network`)
- Quantum Tunneling (`quantum_tunneling_probability`)

### Astrophysics Extensions

Models based on astrophysical phenomena:

- Civilization Lifecycle Phases (`civilization_lifecycle_phase`)
- Knowledge Event Horizons (`suppression_event_horizon`)
- Background Knowledge (`cosmic_background_knowledge`)
- Knowledge Inflation (`knowledge_inflation`)
- Knowledge Gravitational Lensing (`knowledge_gravitational_lensing`)
- Dark Energy Acceleration (`dark_energy_knowledge_acceleration`)

### Circuit Breaker Utility (utils/circuit_breaker.py)

Numerical stability utility providing:

- Value and array stability checks
- Rate of change monitoring
- Energy conservation checks
- Safe mathematical operations
- Timestep recommendations

### Parameters (config/parameters.py)

Centralized parameters for consistent simulation settings across all models.

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
max_change = np.max(np.abs(current_values - previous_values))

# Adjust timestep
if max_change > threshold_high:
    dt = max(min_dt, dt * 0.8)  # Reduce timestep
elif max_change < threshold_low:
    dt = min(max_dt, dt * 1.2)  # Increase timestep
```

5. **Track and report stability metrics**:
```python
# Get stability report
stability_report = circuit_breaker.get_status_report()

# Save metrics
stability_metrics = {
    'stability_issues': circuit_breaker.trigger_count,
    'was_triggered': circuit_breaker.was_triggered,
    'last_trigger_reason': circuit_breaker.last_trigger_reason
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

This framework integrates principles from thermodynamics, relativity, nuclear forces, electromagnetism, and quantum mechanics to model the dynamics of intelligence growth and truth adoption in societal systems.