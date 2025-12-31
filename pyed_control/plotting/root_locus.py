"""
Root locus plotting.
"""

import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
from typing import Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.system import System


def root_locus(
    sys: Union["System", ctrl.TransferFunction],
    gains: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
):
    """
    Plot root locus diagram.

    Parameters
    ----------
    sys : System or TransferFunction
        The open-loop system
    gains : array, optional
        Gain values to plot
    ax : matplotlib axis, optional
        Axis to plot on

    Examples
    --------
    >>> root_locus(sys)
    """
    from ..core.system import System

    if isinstance(sys, System):
        s = sys._sys
        name = sys._name
    else:
        s = sys
        name = None

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    ctrl.root_locus(s, ax=ax, **kwargs)

    title = "Root Locus"
    if name:
        title += f": {name}"
    ax.set_title(title)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax.axvline(x=0, color="gray", linestyle="-", alpha=0.3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
