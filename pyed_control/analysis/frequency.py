"""
Frequency-domain response analysis.
"""

import numpy as np
import control as ctrl
from typing import Dict, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.system import System


def margins(
    sys: Union["System", ctrl.TransferFunction],
) -> Dict[str, Optional[float]]:
    """
    Get stability margins from frequency response.

    Parameters
    ----------
    sys : System or TransferFunction
        The system to analyze

    Returns
    -------
    dict
        Dictionary with keys:
        - gain_margin: Gain margin in dB
        - phase_margin: Phase margin in degrees
        - gain_crossover: Gain crossover frequency (rad/s)
        - phase_crossover: Phase crossover frequency (rad/s)
        - bandwidth: -3dB bandwidth (rad/s)

    Examples
    --------
    >>> m = margins(sys)
    >>> print(f"Phase margin: {m['phase_margin']:.1f} deg")
    >>> print(f"Gain margin: {m['gain_margin']:.1f} dB")
    """
    from ..core.system import System

    if isinstance(sys, System):
        system = sys._sys
    else:
        system = sys

    try:
        gm, pm, wcg, wcp = ctrl.margin(system)
        gm_db = 20 * np.log10(gm) if gm is not None and gm > 0 else None
    except Exception:
        gm_db, pm, wcg, wcp = None, None, None, None

    # Calculate bandwidth
    try:
        omega = np.logspace(-2, 4, 1000)
        mag, _, _ = ctrl.frequency_response(system, omega)
        mag = np.abs(np.array(mag).flatten())
        dc = mag[0]
        bw_idx = np.where(mag < dc / np.sqrt(2))[0]
        bandwidth = omega[bw_idx[0]] if len(bw_idx) > 0 else None
    except Exception:
        bandwidth = None

    return {
        "gain_margin": gm_db,
        "phase_margin": pm,
        "gain_crossover": wcg,
        "phase_crossover": wcp,
        "bandwidth": bandwidth,
    }
