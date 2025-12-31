"""
Sensitivity analysis for control systems.

Provides functions for analyzing closed-loop sensitivity,
robustness, and the effects of parameter variations.
"""

import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Union, List, Tuple


def sensitivity_function(
    plant: "System",
    controller: "System" = None,
    omega: Optional[np.ndarray] = None,
    show_work: bool = False,
) -> Tuple["System", np.ndarray, np.ndarray]:
    """
    Compute the sensitivity function S(s).

    The sensitivity function is S(s) = 1 / (1 + L(s)) where L(s) is the
    loop transfer function (controller * plant for unity feedback).

    S(s) represents:
    - Ratio of closed-loop to open-loop sensitivity to disturbances
    - Transfer function from reference to tracking error
    - Effect of plant variations on output

    Parameters
    ----------
    plant : System
        The plant transfer function
    controller : System, optional
        The controller (if None, assumes unity loop L = plant)
    omega : array-like, optional
        Frequency array for evaluation
    show_work : bool
        If True, print explanation

    Returns
    -------
    S : System
        The sensitivity function as a System object
    omega : ndarray
        Frequency array
    S_mag : ndarray
        Magnitude of S(jw)

    Examples
    --------
    >>> from pyed_control import systems, PID
    >>> plant = systems.dc_motor()
    >>> pid = PID(kp=1, ki=0.5, kd=0.1)
    >>> S, omega, S_mag = sensitivity_function(plant, pid)
    """
    from ..core.system import System

    if omega is None:
        omega = np.logspace(-2, 4, 500)

    # Get loop transfer function
    if controller is None:
        if isinstance(plant, System):
            L_sys = plant._sys
        else:
            L_sys = plant
    else:
        if isinstance(plant, System):
            plant_tf = plant._sys
        else:
            plant_tf = plant

        if isinstance(controller, System):
            ctrl_tf = controller._sys
        elif hasattr(controller, 'as_tf'):
            ctrl_tf = controller.as_tf()
        else:
            ctrl_tf = controller

        L_sys = ctrl.series(ctrl_tf, plant_tf)

    # S = 1 / (1 + L)
    S_sys = ctrl.feedback(1, L_sys)

    # Compute frequency response
    mag, phase, omega_out = ctrl.frequency_response(S_sys, omega)
    S_mag = np.abs(np.array(mag).flatten())

    if show_work:
        print("\n" + "=" * 60)
        print(" Sensitivity Function Analysis")
        print("=" * 60)
        print("\nThe sensitivity function S(s) = 1 / (1 + L(s))")
        print("\nPhysical Interpretation:")
        print("  - S(s) is the transfer function from disturbance to output")
        print("  - |S(jw)| < 1 means disturbances are attenuated at frequency w")
        print("  - |S(jw)| > 1 means disturbances are amplified at frequency w")
        print("\nKey Design Goals:")
        print("  - Low |S(jw)| at low frequencies (good disturbance rejection)")
        print("  - Peak of |S(jw)| should be limited (typically < 2 or 6 dB)")

        # Find peak
        S_peak = np.max(S_mag)
        S_peak_db = 20 * np.log10(S_peak)
        peak_freq = omega[np.argmax(S_mag)]

        print(f"\nResults:")
        print(f"  Peak sensitivity: |S|_max = {S_peak:.3f} ({S_peak_db:.1f} dB)")
        print(f"  Peak frequency: w = {peak_freq:.2f} rad/s")

        # Sensitivity margin
        if S_peak > 1:
            M_s = S_peak
            print(f"\nSensitivity margin M_s = {M_s:.3f}")
            print(f"  Recommended: M_s < 2.0 (6 dB) for good robustness")
        print()

    return System(S_sys, name="Sensitivity Function S(s)"), omega, S_mag


def complementary_sensitivity(
    plant: "System",
    controller: "System" = None,
    omega: Optional[np.ndarray] = None,
    show_work: bool = False,
) -> Tuple["System", np.ndarray, np.ndarray]:
    """
    Compute the complementary sensitivity function T(s).

    The complementary sensitivity is T(s) = L(s) / (1 + L(s)) = 1 - S(s)

    T(s) represents:
    - Closed-loop transfer function from reference to output
    - Effect of measurement noise on output
    - Multiplicative uncertainty tolerance

    Parameters
    ----------
    plant : System
        The plant transfer function
    controller : System, optional
        The controller (if None, assumes unity loop L = plant)
    omega : array-like, optional
        Frequency array for evaluation
    show_work : bool
        If True, print explanation

    Returns
    -------
    T : System
        The complementary sensitivity function as a System object
    omega : ndarray
        Frequency array
    T_mag : ndarray
        Magnitude of T(jw)
    """
    from ..core.system import System

    if omega is None:
        omega = np.logspace(-2, 4, 500)

    # Get loop transfer function
    if controller is None:
        if isinstance(plant, System):
            L_sys = plant._sys
        else:
            L_sys = plant
    else:
        if isinstance(plant, System):
            plant_tf = plant._sys
        else:
            plant_tf = plant

        if isinstance(controller, System):
            ctrl_tf = controller._sys
        elif hasattr(controller, 'as_tf'):
            ctrl_tf = controller.as_tf()
        else:
            ctrl_tf = controller

        L_sys = ctrl.series(ctrl_tf, plant_tf)

    # T = L / (1 + L)
    T_sys = ctrl.feedback(L_sys, 1)

    # Compute frequency response
    mag, phase, omega_out = ctrl.frequency_response(T_sys, omega)
    T_mag = np.abs(np.array(mag).flatten())

    if show_work:
        print("\n" + "=" * 60)
        print(" Complementary Sensitivity Function Analysis")
        print("=" * 60)
        print("\nThe complementary sensitivity T(s) = L(s) / (1 + L(s))")
        print("\nNote: S(s) + T(s) = 1 (fundamental constraint!)")
        print("\nPhysical Interpretation:")
        print("  - T(s) is the closed-loop transfer function (reference to output)")
        print("  - T(s) is also how measurement noise affects output")
        print("  - |T(jw)| should be 1 at low frequencies (good tracking)")
        print("  - |T(jw)| should be small at high frequencies (noise rejection)")
        print("\nKey Design Goals:")
        print("  - Bandwidth: frequency where |T| drops to 0.707 (-3 dB)")
        print("  - Roll-off: how fast T decreases at high frequencies")
        print("  - Peak of |T| should be limited (typically < 1.5)")

        # Find bandwidth
        T_db = 20 * np.log10(T_mag)
        bw_idx = np.where(T_db < -3)[0]
        bandwidth = omega[bw_idx[0]] if len(bw_idx) > 0 else None

        T_peak = np.max(T_mag)
        T_peak_db = 20 * np.log10(T_peak)

        print(f"\nResults:")
        print(f"  Peak: |T|_max = {T_peak:.3f} ({T_peak_db:.1f} dB)")
        if bandwidth:
            print(f"  Bandwidth: {bandwidth:.2f} rad/s")
        else:
            print(f"  Bandwidth: > {omega[-1]:.0f} rad/s (beyond analysis range)")
        print()

    return System(T_sys, name="Complementary Sensitivity T(s)"), omega, T_mag


def loop_transfer_function(
    plant: "System",
    controller: "System",
) -> "System":
    """
    Compute the loop transfer function L(s) = C(s) * G(s).

    Parameters
    ----------
    plant : System
        The plant transfer function G(s)
    controller : System
        The controller C(s)

    Returns
    -------
    System
        The loop transfer function L(s)
    """
    from ..core.system import System

    if isinstance(plant, System):
        plant_tf = plant._sys
    else:
        plant_tf = plant

    if isinstance(controller, System):
        ctrl_tf = controller._sys
    elif hasattr(controller, 'as_tf'):
        ctrl_tf = controller.as_tf()
    else:
        ctrl_tf = controller

    L_sys = ctrl.series(ctrl_tf, plant_tf)
    return System(L_sys, name="Loop Transfer Function L(s)")


def robustness_margins(
    plant: "System",
    controller: "System" = None,
    show_work: bool = False,
) -> Dict[str, Any]:
    """
    Compute robustness margins and stability measures.

    Includes:
    - Gain margin (Gm)
    - Phase margin (Pm)
    - Sensitivity peak (Ms)
    - Complementary sensitivity peak (Mt)
    - Modulus margin

    Parameters
    ----------
    plant : System
        The plant transfer function
    controller : System, optional
        The controller
    show_work : bool
        If True, print detailed analysis

    Returns
    -------
    dict
        Dictionary with all margin values
    """
    from ..core.system import System

    omega = np.logspace(-3, 5, 1000)

    # Get loop transfer function
    if controller is None:
        if isinstance(plant, System):
            L_sys = plant._sys
        else:
            L_sys = plant
    else:
        L = loop_transfer_function(plant, controller)
        L_sys = L._sys

    # Classical margins
    try:
        gm, pm, wcg, wcp = ctrl.margin(L_sys)
        gm_db = 20 * np.log10(gm) if gm and gm > 0 else float('inf')
    except Exception:
        gm, pm, wcg, wcp = float('inf'), float('inf'), None, None
        gm_db = float('inf')

    # Sensitivity peaks
    S, _, S_mag = sensitivity_function(plant, controller, omega)
    T, _, T_mag = complementary_sensitivity(plant, controller, omega)

    Ms = float(np.max(S_mag))
    Mt = float(np.max(T_mag))

    # Modulus margin (minimum distance to -1)
    # This is 1/Ms
    modulus_margin = 1 / Ms if Ms > 0 else float('inf')

    results = {
        "gain_margin": gm,
        "gain_margin_db": gm_db,
        "phase_margin": pm,
        "gain_crossover": wcg,
        "phase_crossover": wcp,
        "sensitivity_peak": Ms,
        "sensitivity_peak_db": 20 * np.log10(Ms),
        "complementary_peak": Mt,
        "complementary_peak_db": 20 * np.log10(Mt),
        "modulus_margin": modulus_margin,
    }

    if show_work:
        print("\n" + "=" * 60)
        print(" Robustness Margins Analysis")
        print("=" * 60)

        print("\n--- Classical Margins ---")
        if np.isinf(gm_db):
            print(f"  Gain Margin: inf dB (always stable regardless of gain)")
        else:
            print(f"  Gain Margin: {gm:.2f} ({gm_db:.1f} dB)")
        if pm is not None:
            print(f"  Phase Margin: {pm:.1f} deg")
        if wcg is not None:
            print(f"  Gain Crossover: {wcg:.2f} rad/s")
        if wcp is not None:
            print(f"  Phase Crossover: {wcp:.2f} rad/s")

        print("\n--- Sensitivity-Based Margins ---")
        print(f"  Peak Sensitivity |S|_max = {Ms:.3f} ({20*np.log10(Ms):.1f} dB)")
        print(f"  Peak Complementary |T|_max = {Mt:.3f} ({20*np.log10(Mt):.1f} dB)")
        print(f"  Modulus Margin = {modulus_margin:.3f}")

        print("\n--- Robustness Assessment ---")
        if gm_db >= 6 and pm >= 45:
            print("  Classical margins: GOOD (Gm >= 6dB, Pm >= 45deg)")
        elif gm_db >= 4 and pm >= 30:
            print("  Classical margins: ACCEPTABLE (Gm >= 4dB, Pm >= 30deg)")
        else:
            print("  Classical margins: POOR - consider redesigning controller")

        if Ms <= 1.5:
            print("  Sensitivity peak: EXCELLENT (Ms <= 1.5)")
        elif Ms <= 2.0:
            print("  Sensitivity peak: GOOD (Ms <= 2.0)")
        else:
            print("  Sensitivity peak: POOR - high sensitivity to disturbances")

        print()

    return results


def plot_sensitivity(
    plant: "System",
    controller: "System" = None,
    omega: Optional[np.ndarray] = None,
    show: bool = True,
) -> None:
    """
    Plot sensitivity and complementary sensitivity functions.

    Creates a plot showing both S(jw) and T(jw) on the same axes,
    illustrating the fundamental trade-off between them.

    Parameters
    ----------
    plant : System
        The plant transfer function
    controller : System, optional
        The controller
    omega : array-like, optional
        Frequency array
    show : bool
        If True, display the plot

    Examples
    --------
    >>> from pyed_control import systems, PID
    >>> plant = systems.dc_motor()
    >>> pid = PID.ziegler_nichols(plant)
    >>> plot_sensitivity(plant, pid)
    """
    if omega is None:
        omega = np.logspace(-2, 4, 500)

    S, _, S_mag = sensitivity_function(plant, controller, omega)
    T, _, T_mag = complementary_sensitivity(plant, controller, omega)

    S_db = 20 * np.log10(S_mag)
    T_db = 20 * np.log10(T_mag)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogx(omega, S_db, 'b-', label='|S(jω)| Sensitivity', linewidth=2)
    ax.semilogx(omega, T_db, 'r-', label='|T(jω)| Complementary', linewidth=2)

    # Reference lines
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=6, color='orange', linestyle='--', alpha=0.5, label='6 dB limit')
    ax.axhline(y=-3, color='green', linestyle=':', alpha=0.5, label='-3 dB (bandwidth)')

    ax.set_xlabel('Frequency (rad/s)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Sensitivity Functions')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best')
    ax.set_ylim([-40, 20])

    plt.tight_layout()
    if show:
        plt.show()


def parameter_sensitivity(
    plant: "System",
    param_name: str,
    param_range: np.ndarray,
    param_nominal: float,
    controller: "System" = None,
    show_work: bool = False,
) -> Dict[str, Any]:
    """
    Analyze sensitivity to a parameter variation.

    Shows how closed-loop performance changes as a plant parameter varies.

    Parameters
    ----------
    plant : System
        The plant (as a function that takes the parameter)
    param_name : str
        Name of the parameter being varied
    param_range : array-like
        Range of parameter values to test
    param_nominal : float
        Nominal parameter value
    controller : System, optional
        The controller
    show_work : bool
        If True, plot results

    Returns
    -------
    dict
        Dictionary with sensitivity analysis results
    """
    # This would require the plant to be parameterized,
    # which is complex. For now, return a stub.
    print("\nParameter sensitivity analysis requires a parameterized plant.")
    print("Consider using the robustness_margins function instead for")
    print("general robustness assessment.")
    return {"not_implemented": True}


def disk_margin(
    plant: "System",
    controller: "System" = None,
    alpha: float = 2.0,
    show_work: bool = False,
) -> Dict[str, float]:
    """
    Compute disk-based stability margins.

    The disk margin gives simultaneous gain and phase margin guarantees.
    A system with disk margin (alpha, sigma) can tolerate:
    - Gain variations from 1/alpha to alpha
    - Phase variations of ±asin((alpha-1/alpha)/(alpha+1/alpha)*sin(sigma))

    Parameters
    ----------
    plant : System
        The plant transfer function
    controller : System, optional
        The controller
    alpha : float
        Disk margin parameter (default 2.0)
    show_work : bool
        If True, print analysis

    Returns
    -------
    dict
        Dictionary with disk margin values
    """
    from ..core.system import System

    # Get sensitivity peak
    omega = np.logspace(-3, 5, 1000)
    S, _, S_mag = sensitivity_function(plant, controller, omega)
    Ms = float(np.max(S_mag))

    # Disk margin from sensitivity peak
    # The modulus margin is 1/Ms
    # This translates to gain margin of Ms/(Ms-1) and phase margin of 2*asin(1/(2*Ms))

    if Ms > 1:
        gain_factor = Ms / (Ms - 1)
        phase_margin_rad = 2 * np.arcsin(1 / (2 * Ms))
        phase_margin_deg = np.degrees(phase_margin_rad)
    else:
        gain_factor = float('inf')
        phase_margin_deg = 90.0

    results = {
        "sensitivity_peak": Ms,
        "gain_margin_lower": 1 / gain_factor if gain_factor != float('inf') else 0,
        "gain_margin_upper": gain_factor,
        "phase_margin": phase_margin_deg,
    }

    if show_work:
        print("\n" + "=" * 60)
        print(" Disk Margin Analysis")
        print("=" * 60)
        print(f"\nSensitivity peak: Ms = {Ms:.3f}")
        print(f"\nSimultaneous margins (guaranteed):")
        print(f"  Gain can vary by factor: [{1/gain_factor:.3f}, {gain_factor:.3f}]")
        print(f"  Phase can vary by: ±{phase_margin_deg:.1f} deg")
        print("\nNote: These are simultaneous margins, meaning the system")
        print("remains stable if gain AND phase vary within these bounds.")
        print()

    return results
