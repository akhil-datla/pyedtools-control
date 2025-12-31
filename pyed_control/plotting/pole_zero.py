"""
Pole-zero map plotting.
"""

import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
from typing import Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.system import System


def pzmap(
    sys: Union["System", ctrl.TransferFunction],
    show_regions: bool = False,
    ax: Optional[plt.Axes] = None,
    **kwargs,
):
    """
    Plot pole-zero map.

    Parameters
    ----------
    sys : System or TransferFunction
        The system to plot
    show_regions : bool
        If True, show stability region shading
    ax : matplotlib axis, optional
        Axis to plot on

    Examples
    --------
    >>> pzmap(sys)
    >>> pzmap(sys, show_regions=True)
    """
    from ..core.system import System

    if isinstance(sys, System):
        poles = sys.poles
        zeros = sys.zeros
        name = sys._name
    else:
        poles = ctrl.poles(sys)
        zeros = ctrl.zeros(sys)
        name = None

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Determine axis limits
    all_points = list(poles) + list(zeros)
    if len(all_points) > 0:
        max_real = max(abs(np.real(p)) for p in all_points) * 1.2
        max_imag = max(abs(np.imag(p)) for p in all_points) * 1.2
        max_val = max(max_real, max_imag, 1)
    else:
        max_val = 1

    if show_regions:
        ax.axvspan(-max_val * 2, 0, alpha=0.1, color="green", label="Stable region")
        ax.axvspan(0, max_val * 2, alpha=0.1, color="red", label="Unstable region")

    # Plot poles
    if len(poles) > 0:
        ax.plot(
            np.real(poles),
            np.imag(poles),
            "x",
            markersize=10,
            markeredgewidth=2,
            color="red",
            label="Poles",
        )

    # Plot zeros
    if len(zeros) > 0:
        ax.plot(
            np.real(zeros),
            np.imag(zeros),
            "o",
            markersize=10,
            markerfacecolor="none",
            markeredgewidth=2,
            color="blue",
            label="Zeros",
        )

    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax.axvline(x=0, color="gray", linestyle="-", alpha=0.3)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")

    title = "Pole-Zero Map"
    if name:
        title += f": {name}"
    ax.set_title(title)

    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()
