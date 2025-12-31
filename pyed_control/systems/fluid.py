"""
Fluid system models.

Tank and hydraulic systems for control education.
"""

from ..core.system import System


def tank_level(A=1.0, R=1.0, name=None):
    """
    Single tank liquid level system.

    A first-order system modeling liquid level in a tank
    with an outlet valve.

    Transfer Function (Inlet Flow to Level):

                    R
        G(s) = -----------
               A*R*s + 1

    Physical Interpretation:
    - Input: Inlet volumetric flow rate (m^3/s)
    - Output: Liquid level (m)
    - A: Tank cross-sectional area (m^2)
    - R: Linearized outlet resistance (s/m^2)
    - Time constant: tau = A*R

    Parameters
    ----------
    A : float
        Tank area in m^2 (default: 1.0)
    R : float
        Outlet resistance in s/m^2 (default: 1.0)
    name : str, optional
        System name

    Returns
    -------
    System
        First-order tank level system

    Examples
    --------
    >>> tank = tank_level(A=2.0, R=0.5)
    >>> tank.analyze()
    >>> tank.step()  # Level response to step change in inlet flow
    """
    tau = A * R

    if name is None:
        name = f"Tank Level (tau={tau:.1f} s)"

    sys = System.first_order(tau=tau, gain=R, name=name)
    sys._physical_params = {"A": A, "R": R, "tau": tau}
    sys._input_unit = "m^3/s"
    sys._output_unit = "m"
    sys._system_type = "fluid"

    return sys


def two_tank(A1=1.0, A2=1.0, R1=1.0, R2=1.0, name=None):
    """
    Two tanks in series (interacting tank system).

    A second-order system with two tanks where the outlet of
    the first tank flows into the second tank.

    This is a classic example of a system where the time constants
    of individual tanks combine in an interesting way.

    Parameters
    ----------
    A1 : float
        First tank area in m^2 (default: 1.0)
    A2 : float
        Second tank area in m^2 (default: 1.0)
    R1 : float
        First outlet resistance in s/m^2 (default: 1.0)
    R2 : float
        Second outlet resistance in s/m^2 (default: 1.0)
    name : str, optional
        System name

    Returns
    -------
    System
        Second-order system (inlet flow to second tank level)

    Examples
    --------
    >>> tanks = two_tank()
    >>> tanks.analyze()
    >>> tanks.step()

    Notes
    -----
    This is an overdamped second-order system (two real poles).
    """
    tau1 = A1 * R1
    tau2 = A2 * R2

    if name is None:
        name = f"Two-Tank System"

    # For non-interacting tanks in series:
    # G(s) = R2 / ((tau1*s + 1)(tau2*s + 1))
    # = R2 / (tau1*tau2*s^2 + (tau1+tau2)*s + 1)

    num = [R2]
    den = [tau1 * tau2, tau1 + tau2, 1]

    sys = System.from_tf(num, den, name=name)
    sys._physical_params = {"A1": A1, "A2": A2, "R1": R1, "R2": R2, "tau1": tau1, "tau2": tau2}
    sys._input_unit = "m^3/s"
    sys._output_unit = "m"
    sys._system_type = "fluid"

    return sys
