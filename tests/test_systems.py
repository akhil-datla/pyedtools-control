"""Tests for pre-built system models."""

import pytest
import numpy as np


def test_mass_spring_damper():
    """Test mass-spring-damper system."""
    from pyed_control.systems import mass_spring_damper

    sys = mass_spring_damper(m=1.0, k=4.0, b=0.5)

    assert sys.order == 2
    assert sys.is_stable
    assert "m" in sys._physical_params
    assert sys._physical_params["m"] == 1.0


def test_mass_spring_damper_undamped():
    """Test undamped mass-spring-damper (oscillator)."""
    from pyed_control.systems import mass_spring_damper

    sys = mass_spring_damper(m=1.0, k=4.0, b=0.0)

    assert sys.order == 2
    # With zero damping, poles should be on imaginary axis
    poles = sys.poles
    assert all(np.isclose(np.real(p), 0, atol=1e-10) for p in poles)


def test_dc_motor():
    """Test DC motor model."""
    from pyed_control.systems import dc_motor

    motor = dc_motor()

    assert motor.order == 2
    assert motor.is_stable
    assert motor._input_unit == "V"
    assert motor._output_unit == "rad/s"


def test_dc_motor_position():
    """Test DC motor position model."""
    from pyed_control.systems import dc_motor_position

    motor = dc_motor_position()

    assert motor.order == 3  # Extra integrator
    # Has pole at origin
    assert any(np.isclose(p, 0, atol=1e-10) for p in motor.poles)


def test_inverted_pendulum_unstable():
    """Test that inverted pendulum is unstable."""
    from pyed_control.systems import inverted_pendulum

    pend = inverted_pendulum()

    assert not pend.is_stable
    # Should have at least one RHP pole
    assert any(np.real(p) > 0 for p in pend.poles)


def test_rc_circuit():
    """Test RC circuit model."""
    from pyed_control.systems import rc_circuit

    rc = rc_circuit(R=1000, C=1e-6)

    assert rc.order == 1
    assert rc.is_stable
    # Time constant should be R*C = 1ms
    assert np.isclose(rc._physical_params["tau"], 0.001, rtol=0.01)


def test_rlc_circuit():
    """Test RLC circuit model."""
    from pyed_control.systems import rlc_circuit

    rlc = rlc_circuit(R=100, L=0.1, C=1e-6)

    assert rlc.order == 2
    assert rlc.is_stable


def test_tank_level():
    """Test tank level system."""
    from pyed_control.systems import tank_level

    tank = tank_level(A=2.0, R=0.5)

    assert tank.order == 1
    assert tank.is_stable


def test_two_tank():
    """Test two-tank system."""
    from pyed_control.systems import two_tank

    tanks = two_tank()

    assert tanks.order == 2
    assert tanks.is_stable


def test_thermal_mass():
    """Test thermal mass system."""
    from pyed_control.systems import thermal_mass

    heater = thermal_mass(C=1000, R=0.1)

    assert heater.order == 1
    assert heater.is_stable


def test_quarter_car_suspension():
    """Test quarter car suspension model."""
    from pyed_control.systems import quarter_car_suspension

    car = quarter_car_suspension()

    assert car.order == 4
    assert car.is_stable
