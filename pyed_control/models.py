"""Pre-built physical models for control education."""

from control import TransferFunction, tf, StateSpace

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


def inverted_pendulum(M=0.5, m=0.2, b=0.1, l=0.3, I=0.006, g=9.81):
    """Linearized inverted pendulum on a cart.

    Returns a ``StateSpace`` object with state vector
    ``[x, x_dot, theta, theta_dot]`` and input force ``F``.
    Parameters follow common textbook notation: ``M`` cart mass,
    ``m`` pendulum mass, ``b`` cart friction, ``l`` pendulum center of mass
    length, ``I`` pendulum inertia about its center, and ``g`` gravity.
    """
    import numpy as np

    q = (M + m) * (I + m * l**2) - (m * l) ** 2

    A = np.array(
        [
            [0, 1, 0, 0],
            [0, -(I + m * l**2) * b / q, (m**2 * g * l**2) / q, 0],
            [0, 0, 0, 1],
            [0, -(m * l * b) / q, m * g * l * (M + m) / q, 0],
        ]
    )
    B = np.array(
        [
            [0],
            [(I + m * l**2) / q],
            [0],
            [(m * l) / q],
        ]
    )
    C = np.eye(4)
    D = np.zeros((4, 1))
    return StateSpace(A, B, C, D)
