# pyedtools-control

A student-friendly Python library for learning classical control systems. Designed for undergraduate intro students with zero friction - be productive in minutes.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import pyed_control as ctrl

# Create a second-order system
sys = ctrl.second_order(wn=2, zeta=0.5)
sys.analyze()  # Comprehensive analysis printout
sys.step()     # Plot step response with annotations

# Use pre-built physical systems
from pyed_control import systems
motor = systems.dc_motor()
motor.bode(show_margins=True)

# Design a PID controller
pid = ctrl.PID.ziegler_nichols(motor, show_work=True)
closed_loop = motor.with_controller(pid)

# Compare open and closed loop
ctrl.compare(motor, closed_loop, labels=['Open Loop', 'With PID'])

# Discrete-time (z-domain) systems
discrete = ctrl.c2d(sys, dt=0.1, method='zoh', show_work=True)
discrete.analyze()
```

## Features

### Simple System Creation

```python
# From transfer function coefficients
sys = ctrl.tf([1], [1, 2, 1])  # G(s) = 1 / (s^2 + 2s + 1)

# Standard forms
sys = ctrl.first_order(tau=0.5, gain=2.0)
sys = ctrl.second_order(wn=2.0, zeta=0.5)
sys = ctrl.integrator()

# From zeros, poles, gain
sys = ctrl.zpk(zeros=[-1], poles=[-2, -3], gain=5)
```

### Pre-Built Physical Systems

```python
from pyed_control import systems

# Mechanical
msd = systems.mass_spring_damper(m=1, k=4, b=0.5)
car = systems.quarter_car_suspension()

# Electrical
rc = systems.rc_circuit(R=1000, C=1e-6)
rlc = systems.rlc_circuit(R=100, L=0.1, C=1e-6)

# Electromechanical
motor = systems.dc_motor()
motor_pos = systems.dc_motor_position()
pendulum = systems.inverted_pendulum()  # Unstable!

# Thermal & Fluid
heater = systems.thermal_mass(C=1000, R=0.1)
tank = systems.tank_level(A=2.0, R=0.5)

# Aerospace
aircraft = systems.aircraft_pitch()
satellite = systems.satellite_attitude(J=100)
quadcopter = systems.quadcopter_attitude()

# Biological/Medical
glucose = systems.glucose_insulin()
drug = systems.drug_concentration()
baroreflex = systems.heart_rate_baroreflex()
```

### Custom System Builder

```python
from pyed_control.systems import SystemBuilder

# Fluent API for building systems
sys = (SystemBuilder()
    .set_name("Custom System")
    .add_complex_poles(wn=10, zeta=0.5)  # Second-order dynamics
    .add_real_pole(time_constant=0.1)    # Fast pole
    .add_zero(-5)                         # Add a zero
    .set_dc_gain(2.0)                     # Set DC gain
    .build())
```

### Comprehensive Analysis

```python
sys.analyze()  # Full printout including:
               # - Transfer function
               # - Poles/zeros with damping info
               # - Stability assessment
               # - Step response characteristics
               # - Frequency response margins

sys.is_stable      # Quick stability check
sys.poles          # Pole locations
sys.zeros          # Zero locations
sys.dc_gain        # DC gain
sys.step_info()    # Rise time, settling time, overshoot, etc.
sys.frequency_info()  # Gain margin, phase margin, bandwidth
```

### Easy Plotting

```python
sys.step()           # Step response with annotations
sys.impulse()        # Impulse response
sys.bode()           # Bode diagram with margins
sys.nyquist()        # Nyquist diagram
sys.root_locus()     # Root locus
sys.pole_zero_map()  # Pole-zero map
sys.all_plots()      # All plots in one figure

# Compare multiple systems
ctrl.compare(sys1, sys2, sys3,
             labels=['Original', 'PID', 'Lead'],
             plot_type='step')  # or 'bode', 'pzmap'
```

### Controller Design

```python
# PID Controllers (multiple tuning methods)
pid = ctrl.PID(kp=1.0, ki=0.5, kd=0.1)
pid = ctrl.PID.ziegler_nichols(plant, show_work=True)  # Classic tuning
pid = ctrl.PID.cohen_coon(plant, show_work=True)       # Better for delay
pid = ctrl.PID.imc(plant, tau_c=0.5, show_work=True)   # Internal Model Control
pid.explain()  # Educational explanation of each term

# Lead/Lag Compensators
lead = ctrl.Lead.for_phase_boost(45, at_frequency=10)
lag = ctrl.Lag.for_steady_state_improvement(plant, factor=10)

# Apply controller
closed_loop = plant.with_controller(pid)
closed_loop.step()
```

### Discrete-Time (Z-Domain) Support

```python
# Convert continuous to discrete
continuous = ctrl.second_order(wn=2, zeta=0.5)
discrete = ctrl.c2d(continuous, dt=0.1, method='zoh', show_work=True)

# Available discretization methods: 'zoh', 'tustin', 'matched', 'euler'

# Analyze discrete system
discrete.analyze()      # Shows poles in z-plane
discrete.step()         # Discrete step response
discrete.pole_zero_map()  # Z-plane pole-zero map

# Convert back to continuous
back_to_continuous = ctrl.d2c(discrete, method='tustin')

# Learn about discretization
ctrl.explain_discretization('tustin')
```

### State-Space Analysis

```python
from pyed_control import analysis

# Controllability and Observability
analysis.is_controllable(sys, show_work=True)
analysis.is_observable(sys, show_work=True)
analysis.state_space_analysis(sys)  # Full analysis

# Get matrices
C = analysis.controllability_matrix(sys)
O = analysis.observability_matrix(sys)
```

### Sensitivity & Robustness Analysis

```python
from pyed_control import analysis

# Sensitivity functions
S, omega, S_mag = analysis.sensitivity_function(plant, pid, show_work=True)
T, omega, T_mag = analysis.complementary_sensitivity(plant, pid)

# Robustness margins
margins = analysis.robustness_margins(plant, pid, show_work=True)
print(f"Sensitivity peak: {margins['sensitivity_peak']}")

# Plot sensitivity functions
analysis.plot_sensitivity(plant, pid)
```

### Steady-State Error Analysis

```python
from pyed_control import analysis

# System type (number of integrators)
sys_type = analysis.system_type(sys, show_work=True)

# Error constants
Kp, Kv, Ka = analysis.error_constants(sys, show_work=True)

# Steady-state error for different inputs
e_step = analysis.steady_state_error(sys, input_type='step')
e_ramp = analysis.steady_state_error(sys, input_type='ramp')
```

### Routh-Hurwitz Stability

```python
from pyed_control import analysis

# Check stability using Routh-Hurwitz
routh_table, is_stable = analysis.routh_hurwitz(sys, show_work=True)
```

### Educational Features

```python
from pyed_control.education import explain, get_hint, check_design

# Concept explanations (20+ topics available!)
explain('poles')
explain('damping_ratio')
explain('phase_margin')
explain('bode')
explain('nyquist')
explain('root_locus')
explain('pid')
explain('controllability')
explain('observability')
explain('sensitivity')
explain('discretization')
explain('steady_state_error')
explain('lead_lag')

# With system context
explain('stability', context=my_system)

# Get hints for improvement
get_hint(sys, issue='overshoot')
get_hint(sys, issue='slow')
get_hint(sys, issue='steady_state_error')

# Check design specifications
specs = {'overshoot': 10, 'settling_time': 2.0, 'phase_margin': 45}
result = check_design(closed_loop, specs)
```

### Jupyter Notebook Integration

```python
# In Jupyter notebooks, systems display beautifully with LaTeX equations
sys = ctrl.second_order(wn=2, zeta=0.5)
sys  # Displays formatted transfer function with stability info
```

### Show Your Work

Many functions have a `show_work=True` parameter that prints step-by-step calculations:

```python
ctrl.PID.ziegler_nichols(plant, show_work=True)
ctrl.Lead.for_phase_boost(45, at_frequency=10, show_work=True)
ctrl.is_stable(sys, show_work=True)
```

## Package Structure

```
pyed_control/
    System              # Central class for continuous-time systems
    DiscreteSystem      # Class for discrete-time (z-domain) systems
    tf(), zpk(), ss()   # Factory functions
    c2d(), d2c()        # Domain conversion
    PID, Lead, Lag      # Controller classes

    core/               # Core system classes
        system.py           # System class
        discrete.py         # DiscreteSystem class

    systems/            # Pre-built physical systems (30+)
        mechanical.py       # mass_spring_damper, quarter_car...
        electrical.py       # rc_circuit, rlc_circuit...
        electromechanical.py # dc_motor, inverted_pendulum...
        aerospace.py        # aircraft, satellite, quadcopter...
        biological.py       # glucose_insulin, baroreflex...
        thermal.py          # thermal_mass...
        fluid.py            # tank_level, two_tank...
        custom.py           # SystemBuilder

    analysis/           # Analysis functions
        stability.py        # poles, zeros, routh_hurwitz, system_type
        time_response.py    # step_info
        frequency.py        # margins
        state_space.py      # controllability, observability
        sensitivity.py      # S(s), T(s), robustness margins

    controllers/        # Controller design
        pid.py              # PID (Ziegler-Nichols, Cohen-Coon, IMC)
        lead_lag.py         # Lead, Lag, LeadLag

    plotting/           # Visualization
    education/          # Educational features (20+ explanations)
```

## Dependencies

- `control>=0.9.0` - Python Control Systems Library
- `numpy` - Numerical computing
- `scipy` - Scientific computing
- `matplotlib` - Plotting

## License

MIT
