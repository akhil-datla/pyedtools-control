"""
pyedtools-control: Educational Control Systems Library
======================================================

A student-friendly library for learning classical control systems.

Quick Start
-----------
>>> import pyed_control as ctrl

# Create a system
>>> sys = ctrl.second_order(wn=2, zeta=0.5)
>>> sys.analyze()

# Use a pre-built system
>>> motor = ctrl.systems.dc_motor()
>>> motor.step()

# Design a controller
>>> pid = ctrl.PID(kp=1, ki=0.5, kd=0.1)
>>> closed_loop = motor.with_controller(pid)

# Compare systems
>>> ctrl.compare(motor, closed_loop, labels=['Open Loop', 'With PID'])

# Discrete-time (z-domain) systems
>>> discrete = ctrl.c2d(sys, dt=0.1, method='zoh')
>>> discrete.analyze()

Main Classes
------------
System : The central class for representing continuous-time control systems
DiscreteSystem : Class for discrete-time (z-domain) control systems
PID : PID controller with multiple tuning methods
Lead : Lead compensator
Lag : Lag compensator
"""

__version__ = "0.2.0"

# Import the main System class
from .core.system import System

# Import discrete-time support
from .core.discrete import DiscreteSystem, c2d, d2c, explain_discretization

# Import subpackages
from . import systems
from . import controllers
from . import analysis
from . import plotting
from . import education

# Import controller classes at top level for convenience
from .controllers.pid import PID
from .controllers.lead_lag import Lead, Lag, LeadLag

# Import key analysis functions
from .analysis import is_stable, poles, zeros, step_info, margins

# Import comparison function
from .plotting.comparison import compare

# Convenience factory functions at top level
def tf(num, den, name=None):
    """
    Create a system from transfer function coefficients.

    Parameters
    ----------
    num : list
        Numerator coefficients [highest power ... constant]
    den : list
        Denominator coefficients [highest power ... constant]
    name : str, optional
        System name

    Examples
    --------
    >>> # G(s) = 5 / (s^2 + 3s + 2)
    >>> sys = ctrl.tf([5], [1, 3, 2])
    """
    return System.from_tf(num, den, name=name)


def zpk(zeros, poles, gain, name=None):
    """
    Create a system from zeros, poles, and gain.

    Examples
    --------
    >>> sys = ctrl.zpk(zeros=[-1], poles=[-2, -3], gain=5)
    """
    return System.from_zpk(zeros, poles, gain, name=name)


def ss(A, B, C, D, name=None):
    """
    Create a system from state-space matrices.

    Examples
    --------
    >>> sys = ctrl.ss([[-1]], [[1]], [[1]], [[0]])
    """
    return System.from_ss(A, B, C, D, name=name)


def first_order(tau, gain=1.0, name=None):
    """
    Create a first-order system: G(s) = gain / (tau*s + 1)

    Parameters
    ----------
    tau : float
        Time constant (seconds)
    gain : float
        DC gain (default 1.0)

    Examples
    --------
    >>> sys = ctrl.first_order(tau=0.5, gain=2.0)
    """
    return System.first_order(tau, gain, name=name)


def second_order(wn, zeta, gain=1.0, name=None):
    """
    Create a standard second-order system.

    G(s) = gain * wn^2 / (s^2 + 2*zeta*wn*s + wn^2)

    Parameters
    ----------
    wn : float
        Natural frequency (rad/s)
    zeta : float
        Damping ratio (0=undamped, 1=critical, >1=overdamped)
    gain : float
        DC gain (default 1.0)

    Examples
    --------
    >>> sys = ctrl.second_order(wn=2.0, zeta=0.5)
    """
    return System.second_order(wn, zeta, gain, name=name)


def integrator(gain=1.0, name=None):
    """Create a pure integrator: G(s) = gain/s"""
    return System.integrator(gain, name=name)


__all__ = [
    # Main classes
    "System",
    "DiscreteSystem",
    # Subpackages
    "systems",
    "controllers",
    "analysis",
    "plotting",
    "education",
    # Controller classes
    "PID",
    "Lead",
    "Lag",
    "LeadLag",
    # Analysis functions
    "is_stable",
    "poles",
    "zeros",
    "step_info",
    "margins",
    # Plotting
    "compare",
    # Factory functions
    "tf",
    "zpk",
    "ss",
    "first_order",
    "second_order",
    "integrator",
    # Discrete-time functions
    "c2d",
    "d2c",
    "explain_discretization",
]
