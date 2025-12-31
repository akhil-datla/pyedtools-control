"""
Electromechanical system models.

These systems involve both electrical and mechanical domains,
commonly used in control education.
"""

import numpy as np
from ..core.system import System


def dc_motor(J=0.01, b=0.1, K=0.01, R=1.0, L=0.5, name=None):
    """
    DC motor (armature-controlled).

    Transfer Function (Voltage to Angular Velocity):

                           K
        G(s) = --------------------------
               (Ls + R)(Js + b) + K^2

    Physical Interpretation:
    - Input: Armature voltage V (Volts)
    - Output: Angular velocity omega (rad/s)
    - J: Rotor inertia (kg*m^2)
    - b: Viscous friction (N*m*s/rad)
    - K: Motor constant (V*s/rad = N*m/A)
    - R: Armature resistance (Ohms)
    - L: Armature inductance (Henrys)

    Parameters
    ----------
    J : float
        Rotor inertia in kg*m^2 (default: 0.01)
    b : float
        Viscous friction in N*m*s/rad (default: 0.1)
    K : float
        Motor constant (Kt = Ke = K) in V*s/rad (default: 0.01)
    R : float
        Armature resistance in Ohms (default: 1.0)
    L : float
        Armature inductance in Henrys (default: 0.5)

    Returns
    -------
    System
        Transfer function from armature voltage to angular velocity

    Examples
    --------
    >>> motor = dc_motor()
    >>> motor.step()  # Velocity response to voltage step

    >>> # For position control, add an integrator
    >>> position_motor = dc_motor_position()

    See Also
    --------
    dc_motor_position : DC motor with position output
    """
    if name is None:
        name = "DC Motor (velocity)"

    # G(s) = K / ((Ls + R)(Js + b) + K^2)
    # Expanding: (Ls + R)(Js + b) = LJs^2 + (Lb + RJ)s + Rb
    # So denominator: LJs^2 + (Lb + RJ)s + (Rb + K^2)

    num = [K]
    den = [L * J, L * b + R * J, R * b + K**2]

    sys = System.from_tf(num, den, name=name)
    sys._physical_params = {"J": J, "b": b, "K": K, "R": R, "L": L}
    sys._input_unit = "V"
    sys._output_unit = "rad/s"
    sys._system_type = "electromechanical"

    return sys


def dc_motor_position(J=0.01, b=0.1, K=0.01, R=1.0, L=0.5, name=None):
    """
    DC motor with position output.

    Same as dc_motor() but output is angular position instead of velocity.
    This adds an integrator to the motor transfer function.

    Transfer Function (Voltage to Position):

                             K
        G(s) = -----------------------------
               s * ((Ls + R)(Js + b) + K^2)

    Parameters
    ----------
    J : float
        Rotor inertia in kg*m^2 (default: 0.01)
    b : float
        Viscous friction in N*m*s/rad (default: 0.1)
    K : float
        Motor constant in V*s/rad (default: 0.01)
    R : float
        Armature resistance in Ohms (default: 1.0)
    L : float
        Armature inductance in Henrys (default: 0.5)

    Returns
    -------
    System
        Transfer function from armature voltage to angular position

    Examples
    --------
    >>> motor = dc_motor_position()
    >>> motor.analyze()
    >>> motor.root_locus()  # See root locus for position control
    """
    if name is None:
        name = "DC Motor (position)"

    # Same as dc_motor but with s in denominator for integration
    num = [K]
    den = [L * J, L * b + R * J, R * b + K**2, 0]  # Extra 0 for s term

    sys = System.from_tf(num, den, name=name)
    sys._physical_params = {"J": J, "b": b, "K": K, "R": R, "L": L}
    sys._input_unit = "V"
    sys._output_unit = "rad"
    sys._system_type = "electromechanical"

    return sys


def inverted_pendulum(M=0.5, m=0.2, b=0.1, l=0.3, I=0.006, g=9.81, name=None):
    """
    Linearized inverted pendulum on a cart.

    The classic unstable system used to study:
    - Stabilization of unstable systems
    - State feedback control
    - Observer design

    State Variables:
    - x: Cart position (m)
    - x_dot: Cart velocity (m/s)
    - theta: Pendulum angle from vertical (rad)
    - theta_dot: Angular velocity (rad/s)

    Input: Force on cart (N)
    Output: Pendulum angle (rad)

    Parameters
    ----------
    M : float
        Cart mass in kg (default: 0.5)
    m : float
        Pendulum mass in kg (default: 0.2)
    b : float
        Cart friction coefficient in N*s/m (default: 0.1)
    l : float
        Pendulum length to center of mass in m (default: 0.3)
    I : float
        Pendulum moment of inertia in kg*m^2 (default: 0.006)
    g : float
        Gravitational acceleration in m/s^2 (default: 9.81)
    name : str, optional
        System name

    Returns
    -------
    System
        State-space model of the linearized system (UNSTABLE)

    Examples
    --------
    >>> pend = inverted_pendulum()
    >>> pend.is_stable  # False - system is unstable!
    >>> pend.poles  # See the unstable pole

    Notes
    -----
    This system is UNSTABLE. The step response will grow unboundedly.
    Use with feedback control to stabilize.
    """
    if name is None:
        name = "Inverted Pendulum"

    # Derived parameters
    p = I + m * l**2  # Effective inertia
    denom = (M + m) * p - (m * l) ** 2

    # State-space matrices (linearized about theta=0)
    A = [
        [0, 1, 0, 0],
        [0, -b * p / denom, -(m**2) * g * l**2 / denom, 0],
        [0, 0, 0, 1],
        [0, b * m * l / denom, m * g * l * (M + m) / denom, 0],
    ]

    B = [[0], [p / denom], [0], [-m * l / denom]]

    # Output pendulum angle
    C = [[0, 0, 1, 0]]

    D = [[0]]

    sys = System.from_ss(A, B, C, D, name=name)
    sys._physical_params = {"M": M, "m": m, "b": b, "l": l, "I": I, "g": g}
    sys._input_unit = "N"
    sys._output_unit = "rad"
    sys._system_type = "electromechanical"

    return sys
