"""
Thermal system models.

Simple thermal systems for control education.
"""

from ..core.system import System


def thermal_mass(C=1000, R=0.1, name=None):
    """
    Simple thermal mass with heat transfer.

    A first-order thermal system modeling a body with thermal
    capacitance exchanging heat with the environment.

    Transfer Function (Heat Input to Temperature):

                    R
        G(s) = -----------
               R*C*s + 1

    Physical Interpretation:
    - Input: Heat input Q (Watts)
    - Output: Temperature rise above ambient (Kelvin or Celsius)
    - C: Thermal capacitance (J/K) - heat storage capacity
    - R: Thermal resistance (K/W) - resistance to heat flow
    - Time constant: tau = R*C

    This is analogous to an RC electrical circuit.

    Parameters
    ----------
    C : float
        Thermal capacitance in J/K (default: 1000)
    R : float
        Thermal resistance in K/W (default: 0.1)
    name : str, optional
        System name

    Returns
    -------
    System
        First-order thermal system

    Examples
    --------
    >>> heater = thermal_mass(C=1000, R=0.1)  # 100 second time constant
    >>> heater.analyze()
    >>> heater.step()  # Temperature response to step heat input
    """
    tau = R * C

    if name is None:
        name = f"Thermal Mass (tau={tau:.1f} s)"

    sys = System.first_order(tau=tau, gain=R, name=name)
    sys._physical_params = {"C": C, "R": R, "tau": tau}
    sys._input_unit = "W"
    sys._output_unit = "K"
    sys._system_type = "thermal"

    return sys
