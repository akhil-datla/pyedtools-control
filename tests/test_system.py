"""Tests for the core System class."""

import pytest
import numpy as np


def test_system_from_tf():
    """Test creating a system from transfer function coefficients."""
    from pyed_control import System

    sys = System.from_tf([1], [1, 2, 1])
    assert sys.order == 2
    assert len(sys.poles) == 2
    assert sys.is_stable


def test_system_from_zpk():
    """Test creating a system from zeros, poles, and gain."""
    from pyed_control import System

    sys = System.from_zpk(zeros=[-1], poles=[-2, -3], gain=5)
    assert len(sys.zeros) == 1
    assert len(sys.poles) == 2
    assert sys.is_stable


def test_system_first_order():
    """Test first-order system factory method."""
    from pyed_control import System

    sys = System.first_order(tau=0.5, gain=2.0)
    assert sys.order == 1
    assert sys.is_stable
    assert np.isclose(sys.dc_gain, 2.0, rtol=0.01)


def test_system_second_order():
    """Test second-order system factory method."""
    from pyed_control import System

    sys = System.second_order(wn=2.0, zeta=0.5)
    assert sys.order == 2
    assert sys.is_stable

    # Check that wn and zeta are stored
    assert "wn" in sys._physical_params
    assert "zeta" in sys._physical_params
    assert sys._physical_params["wn"] == 2.0
    assert sys._physical_params["zeta"] == 0.5


def test_system_stability_stable():
    """Test stability detection for stable system."""
    from pyed_control import System

    sys = System.from_tf([1], [1, 3, 2])  # Poles at -1, -2
    assert sys.is_stable


def test_system_stability_unstable():
    """Test stability detection for unstable system."""
    from pyed_control import System

    sys = System.from_tf([1], [1, -1, -2])  # Has RHP pole
    assert not sys.is_stable


def test_system_poles():
    """Test pole calculation."""
    from pyed_control import System

    sys = System.from_tf([1], [1, 3, 2])  # (s+1)(s+2)
    poles = sys.poles
    assert len(poles) == 2
    # Poles should be at -1 and -2
    assert np.isclose(sorted(np.real(poles)), [-2, -1]).all()


def test_system_zeros():
    """Test zero calculation."""
    from pyed_control import System

    sys = System.from_tf([1, 5], [1, 3, 2])  # Zero at -5
    zeros = sys.zeros
    assert len(zeros) == 1
    assert np.isclose(np.real(zeros[0]), -5, rtol=0.01)


def test_system_dc_gain():
    """Test DC gain calculation."""
    from pyed_control import System

    # G(s) = 10 / (s + 2) -> DC gain = 10/2 = 5
    sys = System.from_tf([10], [1, 2])
    assert np.isclose(sys.dc_gain, 5.0, rtol=0.01)


def test_system_step_info():
    """Test step response characteristics calculation."""
    from pyed_control import System

    sys = System.second_order(wn=2.0, zeta=0.5)
    info = sys.step_info()

    assert "rise_time" in info
    assert "settling_time" in info
    assert "overshoot" in info
    assert "steady_state" in info

    # For zeta=0.5, expect significant overshoot
    assert info["overshoot"] > 10


def test_system_with_controller():
    """Test closed-loop system creation."""
    from pyed_control import System, PID

    plant = System.second_order(wn=2.0, zeta=0.3)
    pid = PID(kp=1.0, ki=0.5)

    closed_loop = plant.with_controller(pid)
    assert closed_loop is not None
    assert closed_loop.order >= plant.order


def test_system_series():
    """Test series connection."""
    from pyed_control import System

    sys1 = System.from_tf([1], [1, 1])  # 1/(s+1)
    sys2 = System.from_tf([2], [1, 2])  # 2/(s+2)

    combined = sys1.series(sys2)
    assert combined.order == 2


def test_system_feedback():
    """Test feedback connection."""
    from pyed_control import System

    sys = System.from_tf([1], [1, 1])  # 1/(s+1)
    closed = sys.feedback()

    # Closed loop of 1/(s+1) with unity feedback is 1/(s+2)
    assert closed.is_stable


def test_system_repr():
    """Test string representation."""
    from pyed_control import System

    sys = System.second_order(wn=2.0, zeta=0.5)
    repr_str = repr(sys)

    assert "System" in repr_str
    assert "order" in repr_str
    assert "stable" in repr_str


def test_convenience_functions():
    """Test top-level convenience functions."""
    import pyed_control as ctrl

    sys1 = ctrl.tf([1], [1, 1])
    assert sys1.order == 1

    sys2 = ctrl.second_order(wn=2, zeta=0.5)
    assert sys2.order == 2

    sys3 = ctrl.first_order(tau=1.0)
    assert sys3.order == 1

    sys4 = ctrl.integrator()
    assert sys4.order == 1
