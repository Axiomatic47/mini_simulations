# Axiomatic Intelligence Growth Simulation Framework

This project implements a mathematical framework for modeling intelligence growth, truth adoption, and suppression dynamics using parallels to fundamental physics principles. The framework includes comprehensive numerical stability safeguards to ensure reliable simulation results and a visualization dashboard to explore the data.

## Directory Structure

```
mini_axiomatic_simulation/
├── config/                # Equation modules and parameters
│   ├── __init__.py        # Makes config a proper package
│   ├── equations.py       # Core equation functions with enhanced stability
│   ├── parameters.py      # Centralized simulation parameters
│   ├── quantum_em_extensions.py  # EM and quantum mechanics extensions
│   ├── historical_validation.py  # Historical validation module
│   ├── multi_civilization_extensions.py  # Multi-civilization simulation models
│   └── astrophysics_extensions.py  # Astrophysics analogies
│
├── utils/                 # Utility functions and classes
│   ├── __init__.py        # Makes utils a proper package
│   ├── circuit_breaker.py # Numerical stability utility
│   ├── dim_handler.py     # Dimension mismatch handler
│   ├── sensitivity_analyzer.py  # Parameter sensitivity analysis
│   ├── cross_level_validator.py # Hierarchical validation
│   ├── edge_case_checker.py     # Edge case detection and fixing
│   └── dimensional_consistency.py # Physical dimension validation
│
├── validation/            # Validation framework components
│   ├── __init__.py         # Makes validation a proper package
│   ├── empirical_validation.py  # Historical data validation
│   └── reports/            # Validation reports directory
│
├── outputs/               # Generated outputs from simulations
│   ├── data/              # CSV data files
│   ├── plots/             # Generated plots
│   └── dashboard/         # Dashboard files
│
├── simulations/           # Simulation scripts
│   ├── comprehensive_simulation.py  # All core dynamics
│   ├── fresh_simulation.py          # Alternative implementation
│   ├── historical_simulation.py     # Historical data overlay
│   ├── multi_agent_simulation.py    # Individual agent focus
│   ├── quantum_em_simulation.py     # EM and quantum extensions
│   ├── astrophysics_simulation.py   # Astrophysics analogies
│   ├── multi_civilization_simulation.py  # Multi-civilization dynamics
│   └── validation_preparation.py    # Runs validation framework
│
├── tests/                 # Test modules
│   ├── test_equations.py  # Core equation tests
│   ├── test_historical_validation.py  # Historical validation tests
│   ├── test_astrophysics_extensions.py  # Astrophysics extension tests
│   ├── test_multi_civilization_extensions.py  # Multi-civilization tests
│   ├── test_quantum_em_extensions.py  # Quantum extension tests
│   └── test_cross_level_coupling.py  # Cross-level validation tests
│
├── dashboard/             # Multi-civilization dashboard
│   ├── minimal_dashboard.py  # Main dashboard application
│   ├── static/            # Static dashboard assets
│   └── templates/         # HTML templates
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

## Multi-Civilization Dashboard

The project includes an interactive visualization dashboard for exploring multi-civilization simulation data. This dashboard provides real-time insights into simulation dynamics including knowledge growth, suppression effects, civilization interactions, and event tracking.

### Dashboard Features

- **Interactive Time Navigation**: Explore simulation data across different time steps
- **Multiple Visualization Views**: Overview, Civilizations, Events, and Stability tabs
- **Real-time Data Updates**: View simulation statistics as they evolve
- **Event Tracking**: Monitor important events like mergers, collisions, and collapses
- **Knowledge & Suppression Analytics**: Visualize how these key metrics interact
- **Stability Metrics**: Track simulation stability and circuit breaker activations

### Running the Dashboard

To launch the dashboard:

```bash
# Navigate to the dashboard directory
cd dashboard

# Run the dashboard application
python minimal_dashboard.py
```

Then access the dashboard in your web browser at: http://127.0.0.1:5000

### Dashboard Architecture

The dashboard uses a Flask backend with a modern frontend built on:
- **Chart.js**: For high-performance data visualization
- **Bootstrap**: For responsive UI components
- **REST API**: For data exchange between the simulation and visualization layers

## Comprehensive Validation Framework

The framework includes an extensive validation system with five key components:

1. **Dimensionality Mismatch Handler**: Automatically detects and fixes array dimension inconsistencies
   - Verifies array dimensions match expected shapes
   - Automatically resizes arrays when needed
   - Provides safe multi-civilization update operations
   - Prevents indexing errors and shape mismatches

2. **Parameter Sensitivity Analysis**: Identifies which parameters most affect outcomes
   - Supports one-at-a-time sensitivity testing
   - Implements global sensitivity analysis with enhanced techniques
   - Calculates parameter correlations and importance rankings
   - Generates comprehensive sensitivity reports

3. **Cross-Level Coupling Validation**: Ensures proper interactions between hierarchy levels
   - Builds and validates dependency graphs
   - Detects potential feedback loops
   - Analyzes signal propagation across levels
   - Validates convergence properties

4. **Edge Case Completion**: Identifies and fixes potential numerical stability issues
   - Automatically detects division by zero, sqrt of negative, etc.
   - Generates recommendations for improving robustness
   - Creates fixed versions of functions with proper safeguards
   - Adds circuit breaker integration to functions

5. **Dimensional Consistency**: Ensures physical dimension consistency in equations
   - Tracks physical dimensions like knowledge, resistance, etc.
   - Validates that equation operations respect dimensions
   - Prevents physically meaningless calculations
   - Enforces dimensional homogeneity

## Enhanced Numerical Stability Features

The framework includes comprehensive numerical stability safeguards:

1. **Circuit Breaker Utility**: Detects and handles numerical instabilities during simulation
   - Prevents NaN/Inf values, division by zero, exponential overflow
   - Monitors rate of change to detect potential divergence
   - Provides safe mathematical operations (exp, log, sqrt, div)
   - Recommends timestep adjustments for stability
   - Tracks energy usage and checks gradients for excessive steepness
   - Records stability incidents for post-simulation analysis

2. **Parameter Bounding**: All parameters and state variables are constrained to reasonable ranges
   - Prevents unbounded growth and numerical overflow
   - Each parameter has explicit min/max values
   - Includes normalization and growth control with safeguards
   - Applies bounds consistently across all equations

3. **Adaptive Timestep**: Automatically adjusts simulation timestep based on rate of change
   - Reduces timestep when rapid changes occur to maintain stability
   - Increases timestep during stable periods for efficiency
   - Smooths timestep transitions to prevent oscillations
   - Adapts based on multiple metrics (knowledge, suppression, etc.)

4. **Error Recovery**: Provides mechanisms to recover from numerical errors
   - Fallback to previous values when errors occur
   - Graceful degradation instead of simulation crashes
   - Exception handling with appropriate fallback behavior
   - Smoothing for abrupt transitions and phase changes

5. **Enhanced Special Case Handling**: Improved handling of edge cases and critical transitions
   - Smooth transitions for critical threshold crossings
   - Epsilon-based comparison for special case detection
   - Gradient damping for extreme parameter values
   - Improved bound checking and enforcement

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

## Running Validation

You can run the complete validation framework or individual components:

```bash
# Run the complete validation framework
python simulations/validation_preparation.py

# Run individual validation components
python simulations/check_dimensions.py
python simulations/run_sensitivity_analysis.py
python simulations/validate_cross_level.py
python simulations/fix_edge_cases.py
python simulations/check_dimensions.py
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

### Enhanced Stabilized Equations (config/equations.py)

Numerically stabilized versions of core equations with circuit breaker integration:

- Intelligence Growth with Circuit Breaker (`intelligence_growth`)
- Free Will Decisions with Bounded Output (`free_will_decision`)
- Truth Adoption with Additional Damping (`truth_adoption`)
- Wisdom Field with Numerical Safeguards (`wisdom_field`, `wisdom_field_enhanced`)
- Suppression and Resistance with Time Bounds (`resistance_resurgence`, `suppression_feedback`)
- Enhanced Suppression Feedback with Smooth Transitions (`suppression_feedback_enhanced`) 
- Quantum Tunneling with Enhanced Stability (`quantum_tunneling_probability`)
- Knowledge Field Influence with Stability Safeguards (`knowledge_field_influence`)

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

### Circuit Breaker Utility (utils/circuit_breaker.py)

Numerical stability utility providing:

- Value and array stability checks (`check_value_stability`, `check_array_stability`)
- Rate of change monitoring (`check_rate_of_change`)
- Energy conservation checks (`check_energy_conservation`)
- Safe mathematical operations (`safe_exp`, `safe_div`, `safe_sqrt`, `safe_log`)
- Timestep recommendations (`recommend_timestep`)
- Gradient checking (`check_gradients`)
- Status reporting (`get_status_report`)
- Comprehensive result validation (`check_and_fix`)

### Dashboard (dashboard/minimal_dashboard.py)

Interactive visualization dashboard with:

- Flask-based web server for data delivery
- Chart.js visualizations for real-time data exploration
- Bootstrap UI for responsive design
- Multiple visualization perspectives on simulation data
- Interactive time navigation controls
- Event tracking and analysis capabilities

## Using Numerical Stability Features

To incorporate stability features in your simulations:

1. **Import and initialize the circuit breaker**:
```python
from utils.circuit_breaker import CircuitBreaker

circuit_breaker = CircuitBreaker(
    threshold=1e-10,
    max_value=1e10,
    min_value=1e-10,
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

3. **Check and fix unstable values**:
```python
# Apply bounds and check stability
safe_value = circuit_breaker.check_and_fix(calculated_value, min_val=-10.0, max_val=10.0)
```

4. **Implement adaptive timestep**:
```python
# Calculate rate of change
max_change = np.max(np.abs(current_values - previous_values) / np.maximum(0.001, np.abs(previous_values)))

# Get recommended timestep from circuit breaker
dt = circuit_breaker.recommend_timestep(dt, max_change)
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

## Using the Validation Framework

To validate your simulations before running:

1. **Check dimension consistency**:
```python
from utils.dim_handler import DimensionHandler

# Create handler
dim_handler = DimensionHandler(verbose=True, auto_fix=True)

# Check array dimensions
arrays = {'positions': positions_array, 'knowledge': knowledge_array}
expected_shapes = {'positions': (n_civs, 2), 'knowledge': (n_civs,)}

# Verify and fix dimensions
fixed_arrays = dim_handler.verify_and_fix_if_needed(arrays, expected_shapes, "simulation")
```

2. **Run parameter sensitivity analysis**:
```python
from utils.sensitivity_analyzer import ParameterSensitivityAnalyzer

# Create analyzer
analyzer = ParameterSensitivityAnalyzer(run_simulation, metrics, base_parameters)

# Define parameter ranges
analyzer.define_parameter_ranges({
    'K_0': (0.1, 5.0, 5),
    'alpha_wisdom': (0.05, 0.2, 5),
    'gamma_phase': (0.05, 0.2, 5)
})

# Run analysis
results = analyzer.run_one_at_a_time_sensitivity()
importance = analyzer.calculate_parameter_importance()
```

3. **Check for edge cases**:
```python
from utils.edge_case_checker import EdgeCaseChecker

# Create checker
checker = EdgeCaseChecker(equation_functions)

# Analyze functions
checker.analyze_all_functions()

# Generate recommendations
recommendations = checker.generate_recommendations()

# Generate fixes
fixed_code = checker.generate_fixes('wisdom_field')
```

4. **Validate dimensional consistency**:
```python
from utils.dimensional_consistency import Dimension, DimensionalValue

# Create dimensional values
K = DimensionalValue(10.0, Dimension.KNOWLEDGE)
W = DimensionalValue(1.0, Dimension.WISDOM)

# Use in dimensionally-validated function
result = intelligence_growth_with_dimensions(K, W, R, S, 1.5)
```

## Extending the Framework

To add new physics-based analogies:

1. Create a new module in the `config` directory
2. Implement the mathematical models with numerical stability safeguards
3. Create a simulation script in the `simulations` directory
4. Import your new models and integrate them with existing dynamics
5. Add appropriate tests in the `tests` directory

## Extending the Dashboard

To enhance the visualization dashboard:

1. **Add new visualizations**:
   - Create new chart configurations in the dashboard.js file
   - Add new HTML elements in the index.html template
   - Ensure proper data transformation in the Python backend

2. **Implement persistent storage**:
   - Add database integration for storing simulation results
   - Create data retrieval endpoints in the Flask application
   - Enable historical data comparison in the visualization

3. **Deploy to production**:
   - Set up proper WSGI server (Gunicorn/uWSGI)
   - Configure for HTTPS using proper certificates
   - Implement user authentication if needed

## Credits

This framework integrates principles from thermodynamics, relativity, nuclear forces, electromagnetism, quantum mechanics, and astrophysics to model the dynamics of intelligence growth, truth adoption, and multi-civilization interactions in societal systems.