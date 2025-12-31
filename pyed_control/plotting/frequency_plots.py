"""
Frequency-domain plotting functions.
"""

import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
from typing import Optional, Union, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.system import System


def bode(
    sys: Union["System", ctrl.TransferFunction, List],
    omega: Optional[np.ndarray] = None,
    show_margins: bool = True,
    labels: Optional[List[str]] = None,
    **kwargs,
):
    """
    Plot Bode diagram of one or more systems.

    Parameters
    ----------
    sys : System, TransferFunction, or list
        System(s) to plot
    omega : array, optional
        Frequency range in rad/s
    show_margins : bool
        If True, mark gain and phase margins
    labels : list of str, optional
        Labels for each system

    Examples
    --------
    >>> bode(sys)
    >>> bode(sys, show_margins=True)
    >>> bode([sys1, sys2], labels=['Open', 'Closed'])
    """
    from ..core.system import System

    if not isinstance(sys, list):
        systems = [sys]
    else:
        systems = sys

    if labels is None:
        labels = []
        for i, s in enumerate(systems):
            if isinstance(s, System) and s._name:
                labels.append(s._name)
            else:
                labels.append(f"System {i+1}")

    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    if omega is None:
        omega = np.logspace(-2, 4, 500)

    for system, label in zip(systems, labels):
        if isinstance(system, System):
            s = system._sys
        else:
            s = system

        mag, phase, omega_out = ctrl.frequency_response(s, omega)
        mag = np.abs(np.array(mag).flatten())
        phase = np.angle(np.array(phase).flatten(), deg=True)
        mag_db = 20 * np.log10(mag)

        ax_mag.semilogx(omega_out, mag_db, label=label, **kwargs)
        ax_phase.semilogx(omega_out, phase, label=label, **kwargs)

    ax_mag.set_ylabel("Magnitude (dB)")
    ax_mag.set_title("Bode Diagram")
    ax_mag.grid(True, which="both", alpha=0.3)
    ax_mag.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
    ax_mag.legend()

    ax_phase.set_xlabel("Frequency (rad/s)")
    ax_phase.set_ylabel("Phase (deg)")
    ax_phase.grid(True, which="both", alpha=0.3)
    ax_phase.axhline(y=-180, color="gray", linestyle="-", alpha=0.5)
    ax_phase.legend()

    if show_margins and len(systems) == 1:
        try:
            from ..analysis.frequency import margins

            m = margins(systems[0])
            if m["gain_crossover"]:
                ax_mag.axvline(
                    x=m["gain_crossover"],
                    color="red",
                    linestyle="--",
                    alpha=0.5,
                    label=f"Crossover: {m['gain_crossover']:.2f} rad/s",
                )
            if m["phase_margin"]:
                ax_phase.annotate(
                    f"PM = {m['phase_margin']:.1f} deg",
                    xy=(m["gain_crossover"], -180 + m["phase_margin"]),
                    fontsize=9,
                )
        except Exception:
            pass

    plt.tight_layout()
    plt.show()


def nyquist(
    sys: Union["System", ctrl.TransferFunction],
    omega: Optional[np.ndarray] = None,
    show_critical: bool = True,
    ax: Optional[plt.Axes] = None,
    **kwargs,
):
    """
    Plot Nyquist diagram.

    Parameters
    ----------
    sys : System or TransferFunction
        The system to plot
    omega : array, optional
        Frequency range
    show_critical : bool
        If True, mark the -1+0j point
    ax : matplotlib axis, optional
        Axis to plot on
    """
    from ..core.system import System

    if isinstance(sys, System):
        s = sys._sys
        name = sys._name
    else:
        s = sys
        name = None

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    ctrl.nyquist_plot(s, omega=omega, ax=ax, **kwargs)

    if show_critical:
        ax.plot(-1, 0, "rx", markersize=10, markeredgewidth=2, label="Critical point (-1, 0)")
        ax.legend()

    title = "Nyquist Diagram"
    if name:
        title += f": {name}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()
