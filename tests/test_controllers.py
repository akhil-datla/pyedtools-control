"""Tests for controller classes."""

import pytest
import numpy as np


def test_pid_creation():
    """Test basic PID controller creation."""
    from pyed_control import PID

    pid = PID(kp=1.0, ki=0.5, kd=0.1)

    assert pid.kp == 1.0
    assert pid.ki == 0.5
    assert pid.kd == 0.1


def test_pid_as_tf():
    """Test PID transfer function representation."""
    from pyed_control import PID

    pid = PID(kp=1.0, ki=0.5, kd=0.1)
    tf = pid.as_tf()

    # PID should be proper (numerator degree <= denominator degree)
    assert tf is not None


def test_pid_as_system():
    """Test PID as System conversion."""
    from pyed_control import PID

    pid = PID(kp=2.0, ki=1.0, kd=0.5)
    sys = pid.as_system()

    assert sys is not None
    assert "PID" in sys._name


def test_pid_ziegler_nichols():
    """Test Ziegler-Nichols tuning."""
    from pyed_control import PID
    from pyed_control.systems import mass_spring_damper

    plant = mass_spring_damper()
    pid = PID.ziegler_nichols(plant)

    assert pid.kp > 0
    # Z-N typically gives non-zero Ki and Kd for PID
    assert pid.ki >= 0
    assert pid.kd >= 0


def test_pid_repr():
    """Test PID string representation."""
    from pyed_control import PID

    pid = PID(kp=1.0, ki=0.5, kd=0.1)
    repr_str = repr(pid)

    assert "PID" in repr_str
    assert "1.0" in repr_str or "1" in repr_str


def test_lead_compensator():
    """Test lead compensator creation."""
    from pyed_control import Lead

    lead = Lead(zero=1, pole=10)

    assert lead.zero == 1
    assert lead.pole == 10


def test_lead_for_phase_boost():
    """Test lead compensator design for phase boost."""
    from pyed_control import Lead

    lead = Lead.for_phase_boost(45, at_frequency=10)

    assert lead.zero > 0
    assert lead.pole > 0
    assert lead.zero < lead.pole  # Lead condition


def test_lead_invalid_params():
    """Test lead compensator rejects invalid parameters."""
    from pyed_control import Lead

    with pytest.raises(ValueError):
        # zero > pole is not a lead compensator
        Lead(zero=10, pole=1)


def test_lag_compensator():
    """Test lag compensator creation."""
    from pyed_control import Lag

    lag = Lag(zero=10, pole=1)

    assert lag.zero == 10
    assert lag.pole == 1


def test_lag_invalid_params():
    """Test lag compensator rejects invalid parameters."""
    from pyed_control import Lag

    with pytest.raises(ValueError):
        # zero < pole is not a lag compensator
        Lag(zero=1, pole=10)


def test_leadlag_compensator():
    """Test combined lead-lag compensator."""
    from pyed_control import Lead, Lag, LeadLag

    lead = Lead(zero=1, pole=10)
    lag = Lag(zero=10, pole=1)
    leadlag = LeadLag(lead, lag)

    sys = leadlag.as_system()
    assert sys is not None
