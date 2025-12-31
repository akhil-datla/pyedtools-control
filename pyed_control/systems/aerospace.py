"""
Aerospace control systems for education.

Includes simplified models of aircraft and satellite systems commonly
used in control systems textbooks.
"""

import numpy as np
from typing import Optional
from ..core.system import System


def aircraft_pitch(
    tau_theta: float = 0.5,
    tau_q: float = 0.1,
    k_delta: float = 1.0,
    name: Optional[str] = None,
) -> System:
    """
    Simplified aircraft pitch dynamics (short-period approximation).

    Models the relationship between elevator deflection and pitch angle.
    This is a common second-order approximation used in introductory
    flight dynamics courses.

    Transfer function: G(s) = k_delta / (tau_theta * tau_q * s^2 + (tau_theta + tau_q) * s + 1)

    Parameters
    ----------
    tau_theta : float
        Pitch angle time constant (seconds), default 0.5
    tau_q : float
        Pitch rate time constant (seconds), default 0.1
    k_delta : float
        Elevator effectiveness (rad/rad), default 1.0
    name : str, optional
        System name

    Returns
    -------
    System
        Aircraft pitch dynamics system

    Examples
    --------
    >>> from pyed_control import systems
    >>> aircraft = systems.aircraft_pitch()
    >>> aircraft.analyze()

    Notes
    -----
    This is a simplified short-period model. Real aircraft have additional
    modes (phugoid, Dutch roll, spiral) not captured here.
    """
    # Second-order system from two time constants
    a2 = tau_theta * tau_q
    a1 = tau_theta + tau_q
    a0 = 1

    num = [k_delta]
    den = [a2, a1, a0]

    sys = System.from_tf(num, den, name=name or "Aircraft Pitch Dynamics")
    sys._description = (
        "Short-period pitch approximation. "
        f"Pitch time constant: {tau_theta}s, Rate time constant: {tau_q}s"
    )
    sys._physical_params = {
        "tau_theta": tau_theta,
        "tau_q": tau_q,
        "k_delta": k_delta,
    }
    sys._input_unit = "rad (elevator)"
    sys._output_unit = "rad (pitch angle)"
    return sys


def aircraft_altitude(
    V: float = 100.0,
    tau_theta: float = 0.5,
    tau_q: float = 0.1,
    name: Optional[str] = None,
) -> System:
    """
    Aircraft altitude dynamics (simplified).

    Models the relationship between pitch angle and altitude change.
    Includes the pitch dynamics and integration to altitude.

    Parameters
    ----------
    V : float
        Aircraft velocity (m/s), default 100 m/s
    tau_theta : float
        Pitch time constant (seconds), default 0.5
    tau_q : float
        Pitch rate time constant (seconds), default 0.1
    name : str, optional
        System name

    Returns
    -------
    System
        Aircraft altitude system (unstable - requires controller!)

    Notes
    -----
    This system includes an integrator (altitude from vertical velocity)
    so it's marginally stable and requires a controller for practical use.
    """
    # Pitch dynamics (second order)
    a2 = tau_theta * tau_q
    a1 = tau_theta + tau_q
    a0 = 1

    # Add integrator for altitude (h = integral of V*sin(theta) ≈ V*theta for small angles)
    # So G(s) = V * pitch_dynamics / s
    num = [V]
    den = [a2, a1, a0, 0]  # Extra s in denominator for integration

    sys = System.from_tf(num, den, name=name or "Aircraft Altitude Dynamics")
    sys._description = (
        f"Altitude response to pitch command. Velocity: {V} m/s. "
        "Note: Marginally stable - requires feedback control!"
    )
    sys._physical_params = {
        "V": V,
        "tau_theta": tau_theta,
        "tau_q": tau_q,
    }
    return sys


def aircraft_heading(
    tau_r: float = 0.3,
    tau_beta: float = 0.1,
    k_rudder: float = 0.5,
    name: Optional[str] = None,
) -> System:
    """
    Aircraft heading/yaw dynamics (simplified).

    Models the relationship between rudder input and heading angle.
    Simplified Dutch roll approximation.

    Parameters
    ----------
    tau_r : float
        Yaw rate time constant (seconds), default 0.3
    tau_beta : float
        Sideslip time constant (seconds), default 0.1
    k_rudder : float
        Rudder effectiveness, default 0.5
    name : str, optional
        System name

    Returns
    -------
    System
        Aircraft heading dynamics
    """
    a2 = tau_r * tau_beta
    a1 = tau_r + tau_beta
    a0 = 1

    # Heading is integral of yaw rate
    num = [k_rudder]
    den = [a2, a1, a0, 0]  # Integrator for heading

    sys = System.from_tf(num, den, name=name or "Aircraft Heading Dynamics")
    sys._description = "Yaw/heading response to rudder input (simplified Dutch roll)"
    sys._physical_params = {
        "tau_r": tau_r,
        "tau_beta": tau_beta,
        "k_rudder": k_rudder,
    }
    return sys


def satellite_attitude(
    J: float = 100.0,
    name: Optional[str] = None,
) -> System:
    """
    Satellite single-axis attitude dynamics.

    In the absence of atmosphere, satellite attitude is a double
    integrator from torque to angle.

    G(s) = 1 / (J * s^2)

    Parameters
    ----------
    J : float
        Moment of inertia (kg*m^2), default 100.0
    name : str, optional
        System name

    Returns
    -------
    System
        Satellite attitude dynamics (marginally stable - requires control!)

    Examples
    --------
    >>> from pyed_control import systems
    >>> sat = systems.satellite_attitude(J=50)
    >>> sat.analyze()  # Note: marginally stable!

    Notes
    -----
    This is the fundamental double-integrator problem in control theory.
    The system is marginally stable with two poles at the origin.
    Requires feedback control (e.g., PD controller) for stability.
    """
    num = [1.0 / J]
    den = [1, 0, 0]  # s^2

    sys = System.from_tf(num, den, name=name or "Satellite Attitude")
    sys._description = (
        f"Single-axis attitude dynamics. J = {J} kg*m^2. "
        "Double integrator - marginally stable, requires control!"
    )
    sys._physical_params = {"J": J}
    sys._input_unit = "N*m (torque)"
    sys._output_unit = "rad (angle)"
    return sys


def satellite_with_flexibility(
    J: float = 100.0,
    wn_flex: float = 2.0,
    zeta_flex: float = 0.01,
    flex_ratio: float = 0.1,
    name: Optional[str] = None,
) -> System:
    """
    Satellite attitude with flexible appendage dynamics.

    Models a rigid satellite body with a flexible solar array or antenna.
    The flexibility introduces a lightly-damped resonance that can destabilize
    high-bandwidth controllers.

    Parameters
    ----------
    J : float
        Moment of inertia (kg*m^2), default 100.0
    wn_flex : float
        Flexible mode natural frequency (rad/s), default 2.0
    zeta_flex : float
        Flexible mode damping ratio (typically very low), default 0.01
    flex_ratio : float
        Ratio of flexible to rigid response at DC, default 0.1
    name : str, optional
        System name

    Returns
    -------
    System
        Satellite with flexible mode

    Notes
    -----
    The flexible mode creates a notch-like feature in the frequency response
    and phase problems for control design. This is a classic spacecraft
    control challenge.
    """
    # Rigid body: 1/(J*s^2)
    # Plus flexible mode contribution
    # G(s) = (1/J) * (1/s^2 + flex_ratio * wn^2 / (s^2 + 2*zeta*wn*s + wn^2))

    # Combine into single transfer function
    # This requires computing the combined numerator and denominator
    wn2 = wn_flex ** 2

    # Rigid part: 1/(J*s^2) with denominator s^2*(s^2 + 2*zeta*wn*s + wn^2)
    # Flexible part adds: flex_ratio * wn^2 / (s^2 * (s^2 + 2*zeta*wn*s + wn^2))
    # Combined: (s^2 + 2*zeta*wn*s + wn^2 + flex_ratio*wn^2) / (J * s^2 * (s^2 + 2*zeta*wn*s + wn^2))

    # Numerator: s^2 + 2*zeta*wn*s + wn^2*(1 + flex_ratio)
    num = [1.0/J, 2*zeta_flex*wn_flex/J, wn2*(1 + flex_ratio)/J]

    # Denominator: s^4 + 2*zeta*wn*s^3 + wn^2*s^2
    den = [1, 2*zeta_flex*wn_flex, wn2, 0, 0]

    sys = System.from_tf(num, den, name=name or "Satellite with Flexibility")
    sys._description = (
        f"Rigid body + flexible appendage. J = {J} kg*m^2, "
        f"flex mode at {wn_flex} rad/s with zeta = {zeta_flex}"
    )
    sys._physical_params = {
        "J": J,
        "wn_flex": wn_flex,
        "zeta_flex": zeta_flex,
        "flex_ratio": flex_ratio,
    }
    return sys


def rocket_hover(
    m: float = 1000.0,
    g: float = 9.81,
    L: float = 5.0,
    I: float = 10000.0,
    name: Optional[str] = None,
) -> System:
    """
    Rocket hovering dynamics (simplified attitude).

    Models the pitch/tilt dynamics of a hovering rocket (like SpaceX landing).
    This is an inverted pendulum in the gravity field.

    Parameters
    ----------
    m : float
        Mass (kg), default 1000
    g : float
        Gravitational acceleration (m/s^2), default 9.81
    L : float
        Distance from CoM to thrust point (m), default 5.0
    I : float
        Moment of inertia about CoM (kg*m^2), default 10000
    name : str, optional
        System name

    Returns
    -------
    System
        Rocket hover dynamics (UNSTABLE - requires active control!)

    Notes
    -----
    This system is unstable with a pole in the right half-plane.
    Active control (like the Falcon 9 landing system) is essential.
    """
    # Linearized: I * theta_ddot = m*g*L * theta + tau_control
    # G(s) = theta/tau = 1 / (I*s^2 - m*g*L)
    # = 1/I / (s^2 - (m*g*L/I))

    omega_sq = m * g * L / I  # This is the unstable natural frequency squared

    num = [1.0 / I]
    den = [1, 0, -omega_sq]

    sys = System.from_tf(num, den, name=name or "Rocket Hover Dynamics")
    sys._description = (
        f"Hovering rocket attitude. Mass={m}kg, L={L}m. "
        f"UNSTABLE pole at s={np.sqrt(omega_sq):.2f} rad/s!"
    )
    sys._physical_params = {
        "m": m,
        "g": g,
        "L": L,
        "I": I,
    }
    return sys


def quadcopter_altitude(
    m: float = 1.0,
    k_motor: float = 10.0,
    tau_motor: float = 0.05,
    name: Optional[str] = None,
) -> System:
    """
    Quadcopter altitude dynamics.

    Models the relationship between throttle command and altitude.
    Includes motor dynamics and gravity.

    Parameters
    ----------
    m : float
        Quadcopter mass (kg), default 1.0
    k_motor : float
        Motor thrust gain (N per unit command), default 10.0
    tau_motor : float
        Motor time constant (seconds), default 0.05
    name : str, optional
        System name

    Returns
    -------
    System
        Quadcopter altitude dynamics
    """
    # Motor dynamics: K/(tau*s + 1)
    # Then F = thrust, a = (F-mg)/m, h = double integral of a
    # At hover, we linearize around F = mg
    # G(s) = (k_motor/m) / (s^2 * (tau*s + 1))

    num = [k_motor / m]
    den = [tau_motor, 1, 0, 0]  # (tau*s + 1) * s^2

    sys = System.from_tf(num, den, name=name or "Quadcopter Altitude")
    sys._description = (
        f"Altitude response to throttle. Mass={m}kg. "
        "Marginally stable - requires altitude controller."
    )
    sys._physical_params = {
        "m": m,
        "k_motor": k_motor,
        "tau_motor": tau_motor,
    }
    return sys


def quadcopter_attitude(
    I: float = 0.01,
    k_motor: float = 10.0,
    L: float = 0.25,
    tau_motor: float = 0.02,
    name: Optional[str] = None,
) -> System:
    """
    Quadcopter single-axis attitude dynamics (pitch or roll).

    Models the relationship between differential motor command and attitude angle.

    Parameters
    ----------
    I : float
        Moment of inertia (kg*m^2), default 0.01
    k_motor : float
        Motor thrust gain (N per unit command), default 10.0
    L : float
        Arm length (m), default 0.25
    tau_motor : float
        Motor time constant (seconds), default 0.02
    name : str, optional
        System name

    Returns
    -------
    System
        Quadcopter attitude dynamics
    """
    # Torque from differential thrust: tau = 2 * L * F_diff
    # G(s) = (2*L*k_motor/I) / (s^2 * (tau*s + 1))

    k = 2 * L * k_motor / I
    num = [k]
    den = [tau_motor, 1, 0, 0]

    sys = System.from_tf(num, den, name=name or "Quadcopter Attitude")
    sys._description = (
        f"Single-axis attitude (pitch/roll). I={I} kg*m^2, arm={L}m. "
        "Double integrator with motor lag."
    )
    sys._physical_params = {
        "I": I,
        "k_motor": k_motor,
        "L": L,
        "tau_motor": tau_motor,
    }
    return sys
