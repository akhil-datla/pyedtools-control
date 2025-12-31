"""
System comparison plotting.
"""

import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.system import System


def compare(
    *systems,
    plot_type: str = "step",
    labels: Optional[List[str]] = None,
    T: Optional[float] = None,
    **kwargs,
):
    """
    Compare multiple systems on the same plot.

    Parameters
    ----------
    systems : System objects
        Systems to compare
    plot_type : str
        Type of plot: 'step', 'impulse', 'bode', 'pzmap'
    labels : list of str, optional
        Labels for legend
    T : float, optional
        Simulation time for time-domain plots

    Examples
    --------
    >>> compare(original, with_pid, with_lead, plot_type='step',
    ...         labels=['Original', 'PID', 'Lead'])

    >>> compare(sys1, sys2, plot_type='bode')
    """
    from ..core.system import System

    if len(systems) == 0:
        raise ValueError("At least one system is required")

    if labels is None:
        labels = []
        for i, s in enumerate(systems):
            if isinstance(s, System) and s._name:
                labels.append(s._name)
            else:
                labels.append(f"System {i+1}")

    if plot_type == "step":
        fig, ax = plt.subplots(figsize=(10, 6))
        for sys, label in zip(systems, labels):
            if isinstance(sys, System):
                s = sys._sys
            else:
                s = sys
            t, y = ctrl.step_response(s, T=T)
            y = np.array(y).flatten()
            ax.plot(t, y, label=label, **kwargs)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Response")
        ax.set_title("Step Response Comparison")
        ax.grid(True, alpha=0.3)
        ax.legend()

    elif plot_type == "impulse":
        fig, ax = plt.subplots(figsize=(10, 6))
        for sys, label in zip(systems, labels):
            if isinstance(sys, System):
                s = sys._sys
            else:
                s = sys
            t, y = ctrl.impulse_response(s, T=T)
            y = np.array(y).flatten()
            ax.plot(t, y, label=label, **kwargs)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Response")
        ax.set_title("Impulse Response Comparison")
        ax.grid(True, alpha=0.3)
        ax.legend()

    elif plot_type == "bode":
        fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        omega = np.logspace(-2, 4, 500)

        for sys, label in zip(systems, labels):
            if isinstance(sys, System):
                s = sys._sys
            else:
                s = sys
            mag, phase, omega_out = ctrl.frequency_response(s, omega)
            mag = np.abs(np.array(mag).flatten())
            phase = np.angle(np.array(phase).flatten(), deg=True)
            mag_db = 20 * np.log10(mag)

            ax_mag.semilogx(omega_out, mag_db, label=label, **kwargs)
            ax_phase.semilogx(omega_out, phase, label=label, **kwargs)

        ax_mag.set_ylabel("Magnitude (dB)")
        ax_mag.set_title("Bode Diagram Comparison")
        ax_mag.grid(True, which="both", alpha=0.3)
        ax_mag.legend()

        ax_phase.set_xlabel("Frequency (rad/s)")
        ax_phase.set_ylabel("Phase (deg)")
        ax_phase.grid(True, which="both", alpha=0.3)
        ax_phase.legend()

    elif plot_type == "pzmap":
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = plt.cm.tab10.colors

        for i, (sys, label) in enumerate(zip(systems, labels)):
            if isinstance(sys, System):
                poles = sys.poles
                zeros = sys.zeros
            else:
                poles = ctrl.poles(sys)
                zeros = ctrl.zeros(sys)

            color = colors[i % len(colors)]

            if len(poles) > 0:
                ax.plot(
                    np.real(poles),
                    np.imag(poles),
                    "x",
                    markersize=10,
                    markeredgewidth=2,
                    color=color,
                    label=f"{label} poles",
                )

            if len(zeros) > 0:
                ax.plot(
                    np.real(zeros),
                    np.imag(zeros),
                    "o",
                    markersize=10,
                    markerfacecolor="none",
                    markeredgewidth=2,
                    color=color,
                    label=f"{label} zeros",
                )

        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        ax.axvline(x=0, color="gray", linestyle="-", alpha=0.3)
        ax.set_xlabel("Real")
        ax.set_ylabel("Imaginary")
        ax.set_title("Pole-Zero Map Comparison")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect("equal")

    else:
        raise ValueError(f"Unknown plot_type: {plot_type}. Use 'step', 'impulse', 'bode', or 'pzmap'")

    plt.tight_layout()
    plt.show()
