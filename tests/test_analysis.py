"""Tests for analysis functions."""

import pytest
import numpy as np


def test_is_stable_function():
    """Test is_stable analysis function."""
    from pyed_control import System
    from pyed_control.analysis import is_stable

    stable_sys = System.from_tf([1], [1, 2, 1])
    unstable_sys = System.from_tf([1], [1, -1])

    assert is_stable(stable_sys)
    assert not is_stable(unstable_sys)


def test_poles_function():
    """Test poles analysis function."""
    from pyed_control import System
    from pyed_control.analysis import poles

    sys = System.from_tf([1], [1, 3, 2])  # Poles at -1, -2
    p = poles(sys)

    assert len(p) == 2


def test_poles_detailed():
    """Test detailed pole analysis."""
    from pyed_control import System
    from pyed_control.analysis import poles

    sys = System.second_order(wn=2.0, zeta=0.5)
    p = poles(sys, detailed=True)

    assert len(p) > 0
    assert "type" in p[0]
    # Complex poles for underdamped system
    assert p[0]["type"] == "complex"
    assert "damping_ratio" in p[0]


def test_zeros_function():
    """Test zeros analysis function."""
    from pyed_control import System
    from pyed_control.analysis import zeros

    sys = System.from_tf([1, 5], [1, 3, 2])  # Zero at -5
    z = zeros(sys)

    assert len(z) == 1


def test_step_info_function():
    """Test step_info analysis function."""
    from pyed_control import System
    from pyed_control.analysis import step_info

    sys = System.second_order(wn=2.0, zeta=0.5)
    info = step_info(sys)

    assert "rise_time" in info
    assert "settling_time" in info
    assert "overshoot" in info
    assert info["overshoot"] > 0  # Underdamped has overshoot


def test_step_info_unstable_raises():
    """Test step_info raises for unstable system."""
    from pyed_control import System
    from pyed_control.analysis import step_info
    from pyed_control.exceptions import UnstableSystemError

    sys = System.from_tf([1], [1, -1])  # Unstable

    with pytest.raises(UnstableSystemError):
        step_info(sys)


def test_margins_function():
    """Test margins analysis function."""
    from pyed_control import System
    from pyed_control.analysis import margins

    sys = System.from_tf([10], [1, 3, 2, 0])  # Type-1 system
    m = margins(sys)

    assert "phase_margin" in m
    assert "gain_margin" in m
