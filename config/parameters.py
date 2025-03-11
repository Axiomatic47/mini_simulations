"""
Centralized parameters for axiomatic simulations.
Import from this file for consistent parameter settings across simulations.
Enhanced with numerical stability parameters.
"""

# Standard simulation parameters
TIMESTEPS = 400
DT = 1  # Time step size
NUM_AGENTS = 5

# ===== Core Equation Parameters =====

# Wisdom parameters
W_0 = 1.0  # Base wisdom level
ALPHA_WISDOM = 0.1  # Suppression impact on wisdom

# Resistance and Network parameters
RESISTANCE = 2.0  # Base resistance level
NETWORK_EFFECT = 1.5  # Base network effect strength

# Truth adoption parameters
A_TRUTH = 2.5  # Truth adoption acceleration factor
T_MAX = 40  # Maximum theoretical truth level

# Decay parameters
LAMBDA_DECAY = 0.05  # Suppression decay rate

# Feedback parameters
ALPHA_FEEDBACK = 0.1  # Suppression reinforcement coefficient
BETA_FEEDBACK = 0.05  # Knowledge disruption coefficient

# Resistance resurgence parameters
ALPHA_RESURGE = 5.0  # Resurgence intensity
MU_RESURGE = 0.05  # Resurgence decay rate
T_CRIT_RESURGE = 150  # Critical time for resurgence

# Knowledge phase transition parameters
K_0_PHASE = 1.0  # Initial knowledge
BETA_DECAY_PHASE = 0.02  # Knowledge decay rate
A_PHASE = 1.5  # Knowledge growth amplitude
GAMMA_PHASE = 0.1  # Phase transition sharpness
T_CRIT_PHASE = 20  # Critical threshold for transition

# Civilization oscillation parameters
GAMMA_OSC = 0.005  # Oscillation damping factor
OMEGA_OSC = 0.3  # Natural oscillation frequency

# ===== Quantum & EM Extension Parameters =====

# Electromagnetic field parameters
KAPPA_EM = 0.025  # Knowledge field permeability constant

# Quantum entanglement parameters
RHO_ENTANGLEMENT = 0.08  # Maximum entanglement strength
SIGMA_ENTANGLEMENT = 0.15  # Entanglement decay rate
TUNNELING_THRESHOLD = 0.7  # Threshold for tunneling events

# Multi-agent simulation specific parameters
KNOWLEDGE_CAP = 50  # Maximum knowledge level
SUPPRESSION_FLOOR = 0.1  # Minimum suppression level
MOMENTUM_FACTOR = 0.1  # Decision momentum factor

# External shock parameters
SHOCK_TIMES = [50, 120]  # Times when external shocks are applied
SHOCK_MAGNITUDES = [2.5, 3.0]  # Magnitudes of external shocks

# ===== Numerical Stability Parameters =====

# Knowledge and intelligence bounds
K_MAX = 1000.0  # Maximum knowledge value
I_MAX = 1000.0  # Maximum intelligence value
W_MAX = 10.0    # Maximum wisdom value

# Suppression bounds
S_MAX = 100.0  # Maximum suppression value
S_MIN = 0.1    # Minimum suppression value

# Resistance bounds
R_MAX = 100.0  # Maximum resistance value
R_MIN = 0.0    # Minimum resistance value

# Time bounds for exponential calculations
T_MAX_EXPONENT = 500.0  # Maximum time value for exponential calculations

# Field parameters bounds
FIELD_STRENGTH_MAX = 10.0  # Maximum field strength
GRADIENT_MAX = 10.0       # Maximum gradient magnitude
MIN_DISTANCE = 0.1        # Minimum distance to prevent division by zero

# Entanglement parameters
K_DIFF_MAX = 100.0  # Maximum knowledge difference for entanglement calculation
ENT_MAX = 1.0       # Maximum entanglement value

# Tunneling parameters
P_MIN = 0.0001  # Minimum tunneling probability
P_MAX = 0.99    # Maximum tunneling probability

# Time step adaptation
DT_MIN = 0.01   # Minimum time step
DT_MAX = 1.0    # Maximum time step