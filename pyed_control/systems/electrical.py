"""
Electrical circuit system models.

Classic first and second-order electrical systems.
"""

from ..core.system import System


def rc_circuit(R=1000, C=1e-6, name=None):
    """
    RC low-pass filter circuit.

    A first-order system commonly used in signal filtering.

    Transfer Function (Input Voltage to Output Voltage):

                    1
        G(s) = -----------
               R*C*s + 1

    Physical Interpretation:
    - Input: Input voltage (V)
    - Output: Voltage across capacitor (V)
    - R: Resistance (Ohms)
    - C: Capacitance (Farads)
    - Time constant: tau = R*C

    Parameters
    ----------
    R : float
        Resistance in Ohms (default: 1000)
    C : float
        Capacitance in Farads (default: 1e-6 = 1 uF)
    name : str, optional
        System name

    Returns
    -------
    System
        First-order low-pass filter

    Examples
    --------
    >>> rc = rc_circuit(R=1000, C=1e-6)  # 1 ms time constant
    >>> rc.analyze()
    >>> rc.bode()  # See the low-pass characteristic
    """
    tau = R * C

    if name is None:
        name = f"RC Circuit (tau={tau*1000:.2f} ms)"

    sys = System.first_order(tau=tau, gain=1.0, name=name)
    sys._physical_params = {"R": R, "C": C, "tau": tau}
    sys._input_unit = "V"
    sys._output_unit = "V"
    sys._system_type = "electrical"

    return sys


def rl_circuit(R=10, L=0.01, name=None):
    """
    RL circuit (current response).

    A first-order system modeling inductor current response.

    Transfer Function (Voltage to Current):

                    1/R
        G(s) = -----------
               (L/R)*s + 1

    Physical Interpretation:
    - Input: Applied voltage (V)
    - Output: Current through inductor (A)
    - R: Resistance (Ohms)
    - L: Inductance (Henrys)
    - Time constant: tau = L/R

    Parameters
    ----------
    R : float
        Resistance in Ohms (default: 10)
    L : float
        Inductance in Henrys (default: 0.01 = 10 mH)
    name : str, optional
        System name

    Returns
    -------
    System
        First-order system
    """
    tau = L / R
    gain = 1 / R

    if name is None:
        name = f"RL Circuit (tau={tau*1000:.2f} ms)"

    sys = System.first_order(tau=tau, gain=gain, name=name)
    sys._physical_params = {"R": R, "L": L, "tau": tau}
    sys._input_unit = "V"
    sys._output_unit = "A"
    sys._system_type = "electrical"

    return sys


def rlc_circuit(R=100, L=0.1, C=1e-6, name=None):
    """
    Series RLC circuit.

    A second-order system with voltage across the capacitor as output.

    Transfer Function (Input Voltage to Capacitor Voltage):

                        1/(L*C)
        G(s) = ---------------------------
               s^2 + (R/L)*s + 1/(L*C)

    Physical Interpretation:
    - Input: Applied voltage (V)
    - Output: Voltage across capacitor (V)
    - R: Resistance (Ohms) - determines damping
    - L: Inductance (Henrys) - determines inertia
    - C: Capacitance (Farads) - determines stiffness

    System Characteristics:
    - Natural frequency: wn = 1/sqrt(L*C) rad/s
    - Damping ratio: zeta = R/2 * sqrt(C/L)
    - Critical damping: R_crit = 2*sqrt(L/C)

    Parameters
    ----------
    R : float
        Resistance in Ohms (default: 100)
    L : float
        Inductance in Henrys (default: 0.1)
    C : float
        Capacitance in Farads (default: 1e-6)
    name : str, optional
        System name

    Returns
    -------
    System
        Second-order system

    Examples
    --------
    >>> rlc = rlc_circuit(R=100, L=0.1, C=1e-6)
    >>> rlc.analyze()  # See natural frequency and damping

    >>> # Underdamped case
    >>> rlc_under = rlc_circuit(R=50)  # Lower resistance
    >>> rlc_under.step()  # See oscillations
    """
    import numpy as np

    wn = 1 / np.sqrt(L * C)
    zeta = (R / 2) * np.sqrt(C / L)

    if name is None:
        name = f"RLC Circuit (wn={wn:.1f}, zeta={zeta:.2f})"

    sys = System.second_order(wn=wn, zeta=zeta, gain=1.0, name=name)
    sys._physical_params = {"R": R, "L": L, "C": C, "wn": wn, "zeta": zeta}
    sys._input_unit = "V"
    sys._output_unit = "V"
    sys._system_type = "electrical"

    return sys
