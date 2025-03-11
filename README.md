# Axiomatic Intelligence Growth Simulation Framework

This project implements a mathematical framework for modeling intelligence growth, truth adoption, and suppression dynamics using parallels to fundamental physics principles.

## Directory Structure

```
mini_axiomatic_simulation/
├── config/                # Equation modules and parameters
│   ├── __init__.py        # Makes config a proper package
│   ├── equations.py       # Core equation functions
│   ├── parameters.py      # Centralized simulation parameters
│   └── quantum_em_extensions.py  # EM and quantum mechanics extensions
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
│   └── quantum_em_simulation.py     # EM and quantum extensions
│
└── README.md              # This file
```

## Theoretical Framework

The framework models societal dynamics as analogies to fundamental physics principles:

1. **Thermodynamics**: Intelligence growth as counteracting entropy
2. **Relativity**: Truth adoption with relativistic-like speed limits
3. **Strong Nuclear Force**: Identity binding between agents
4. **Weak Nuclear Force**: Suppression feedback and phase transitions
5. **Electromagnetism**: Knowledge field influence between agents
6. **Quantum Mechanics**: Entanglement and tunneling between knowledge states

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

## Extending the Framework

To add new physics-based analogies:

1. Create a new module in the `config` directory
2. Implement the mathematical models
3. Create a simulation script in the `simulations` directory
4. Import your new models and integrate them with existing dynamics

## Credits

This framework integrates principles from thermodynamics, relativity, nuclear forces, electromagnetism, and quantum mechanics to model the dynamics of intelligence growth and truth adoption in societal systems.