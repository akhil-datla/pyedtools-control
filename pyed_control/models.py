"""Pre-built physical models for control education."""

from control import TransferFunction, tf


def mass_spring_damper(m=1.0, k=1.0, b=0.2):
    """Return the transfer function of a mass-spring-damper system.

    The form is ``1/(m s^2 + b s + k)`` where ``m`` is mass, ``k`` is spring
    constant, and ``b`` is damping coefficient.
    """
    num = [1.0]
    den = [m, b, k]
    return TransferFunction(num, den)


def dc_motor(J=0.01, b=0.1, K=0.01, R=1.0, L=0.5):
    """Return the transfer function of a simple DC motor."""
    num = [K]
    den = [L*J, L*b + R*J, R*b + K**2]
    return TransferFunction(num, den)
