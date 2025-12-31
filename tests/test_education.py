"""Tests for educational features."""

import pytest


def test_explain_function():
    """Test explain function runs without error."""
    from pyed_control.education import explain

    # Should not raise
    explain("poles")
    explain("stability")
    explain("damping_ratio")


def test_explain_with_context():
    """Test explain with system context."""
    from pyed_control import System
    from pyed_control.education import explain

    sys = System.second_order(wn=2, zeta=0.5)
    explain("damping_ratio", context=sys)


def test_concept_function():
    """Test concept function."""
    from pyed_control.education import concept

    result = concept("poles")
    assert "Poles" in result


def test_get_hint_function():
    """Test get_hint function."""
    from pyed_control import System
    from pyed_control.education import get_hint

    sys = System.second_order(wn=2, zeta=0.3)  # Low damping, high overshoot
    get_hint(sys, issue="overshoot")


def test_check_design_function():
    """Test check_design function."""
    from pyed_control import System
    from pyed_control.education import check_design

    sys = System.second_order(wn=2, zeta=0.7)

    specs = {
        "overshoot": 10,
        "settling_time": 5.0,
    }

    result = check_design(sys, specs)

    assert "passed" in result
    assert "failed" in result
    assert "details" in result
