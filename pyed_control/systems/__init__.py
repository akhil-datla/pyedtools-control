"""
Pre-built physical systems for control education.

Categories
----------
Mechanical Systems:
    mass_spring_damper, rotational_system, quarter_car_suspension

Electrical Systems:
    rc_circuit, rl_circuit, rlc_circuit

Electromechanical Systems:
    dc_motor, dc_motor_position, servo_system, inverted_pendulum

Aerospace Systems:
    aircraft_pitch, aircraft_altitude, aircraft_heading,
    satellite_attitude, satellite_with_flexibility,
    rocket_hover, quadcopter_altitude, quadcopter_attitude

Biological Systems:
    glucose_insulin, drug_concentration, pupil_light_reflex,
    heart_rate_baroreflex, respiratory_control, muscle_tendon,
    cell_signaling, body_temperature

Thermal Systems:
    thermal_mass, two_room_thermal

Fluid Systems:
    tank_level, two_tank

Custom Systems:
    SystemBuilder - Fluent builder for custom systems
    from_time_constant, from_bandwidth, from_rise_time,
    from_settling_time, from_overshoot
"""

from .mechanical import (
    mass_spring_damper,
    rotational_system,
    quarter_car_suspension,
)
from .electromechanical import (
    dc_motor,
    dc_motor_position,
    inverted_pendulum,
)
from .electrical import (
    rc_circuit,
    rl_circuit,
    rlc_circuit,
)
from .thermal import thermal_mass
from .fluid import tank_level, two_tank
from .aerospace import (
    aircraft_pitch,
    aircraft_altitude,
    aircraft_heading,
    satellite_attitude,
    satellite_with_flexibility,
    rocket_hover,
    quadcopter_altitude,
    quadcopter_attitude,
)
from .biological import (
    glucose_insulin,
    drug_concentration,
    pupil_light_reflex,
    heart_rate_baroreflex,
    respiratory_control,
    muscle_tendon,
    cell_signaling,
    body_temperature,
)
from .custom import (
    SystemBuilder,
    from_time_constant,
    from_bandwidth,
    from_rise_time,
    from_settling_time,
    from_overshoot,
)

__all__ = [
    # Mechanical
    "mass_spring_damper",
    "rotational_system",
    "quarter_car_suspension",
    # Electromechanical
    "dc_motor",
    "dc_motor_position",
    "inverted_pendulum",
    # Electrical
    "rc_circuit",
    "rl_circuit",
    "rlc_circuit",
    # Thermal
    "thermal_mass",
    # Fluid
    "tank_level",
    "two_tank",
    # Aerospace
    "aircraft_pitch",
    "aircraft_altitude",
    "aircraft_heading",
    "satellite_attitude",
    "satellite_with_flexibility",
    "rocket_hover",
    "quadcopter_altitude",
    "quadcopter_attitude",
    # Biological
    "glucose_insulin",
    "drug_concentration",
    "pupil_light_reflex",
    "heart_rate_baroreflex",
    "respiratory_control",
    "muscle_tendon",
    "cell_signaling",
    "body_temperature",
    # Custom
    "SystemBuilder",
    "from_time_constant",
    "from_bandwidth",
    "from_rise_time",
    "from_settling_time",
    "from_overshoot",
]
