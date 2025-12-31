"""
Time-domain plotting functions.
"""

import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
from typing import Optional, Union, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.system import System


def step(
    sys: Union["System", ctrl.TransferFunction, List],
    T: Optional[float] = None,
    show_info: bool = True,
    labels: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
):
    """
    Plot step response of one or more systems.

    Parameters
    ----------
    sys : System, TransferFunction, or list
        System(s) to plot
    T : float, optional
        Simulation time (auto-calculated if not provided)
    show_info : bool
        If True, annotate with response characteristics
    labels : list of str, optional
        Labels for each system
    ax : matplotlib axis, optional
        Axis to plot on

    Examples
    --------
    >>> step(sys)
    >>> step([sys1, sys2], labels=['Original', 'Controlled'])
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

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    for system, label in zip(systems, labels):
        if isinstance(system, System):
            s = system._sys
        else:
            s = system

        t, y = ctrl.step_response(s, T=T)
        y = np.array(y).flatten()
        ax.plot(t, y, label=label, **kwargs)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Response")
    ax.set_title("Step Response")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if show_info and len(systems) == 1:
        try:
            from ..analysis.time_response import step_info as get_step_info

            info = get_step_info(systems[0])
            textstr = (
                f"Rise time: {info['rise_time']:.3f} s\n"
                f"Settling time: {info['settling_time']:.3f} s\n"
                f"Overshoot: {info['overshoot']:.1f}%"
            )
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            ax.text(
                0.98, 0.98, textstr,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=props,
            )
        except Exception:
            pass

    plt.tight_layout()
    plt.show()


def impulse(
    sys: Union["System", ctrl.TransferFunction, List],
    T: Optional[float] = None,
    labels: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
):
    """
    Plot impulse response of one or more systems.

    Parameters
    ----------
    sys : System, TransferFunction, or list
        System(s) to plot
    T : float, optional
        Simulation time
    labels : list of str, optional
        Labels for each system
    ax : matplotlib axis, optional
        Axis to plot on
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

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    for system, label in zip(systems, labels):
        if isinstance(system, System):
            s = system._sys
        else:
            s = system

        t, y = ctrl.impulse_response(s, T=T)
        y = np.array(y).flatten()
        ax.plot(t, y, label=label, **kwargs)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Response")
    ax.set_title("Impulse Response")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()
