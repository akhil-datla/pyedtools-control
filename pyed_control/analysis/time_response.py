"""
Time-domain response analysis.
"""

import numpy as np
import control as ctrl
from typing import Dict, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.system import System


def step_info(
    sys: Union["System", ctrl.TransferFunction],
    settling_threshold: float = 0.02,
) -> Dict[str, float]:
    """
    Calculate step response characteristics.

    Parameters
    ----------
    sys : System or TransferFunction
        The system to analyze
    settling_threshold : float
        Threshold for settling time (default: 0.02 = 2%)

    Returns
    -------
    dict
        Dictionary with keys:
        - rise_time: Time from 10% to 90% of final value (s)
        - settling_time: Time to stay within threshold of final value (s)
        - overshoot: Percent overshoot (%)
        - peak_time: Time of first peak (s)
        - peak_value: Value at first peak
        - steady_state: Final value

    Examples
    --------
    >>> info = step_info(sys)
    >>> print(f"Rise time: {info['rise_time']:.3f} s")
    >>> print(f"Overshoot: {info['overshoot']:.1f}%")
    """
    from ..core.system import System
    from ..exceptions import UnstableSystemError

    if isinstance(sys, System):
        if not sys.is_stable:
            raise UnstableSystemError(sys.poles, operation="step_info")
        system = sys._sys
    else:
        system = sys
        if any(np.real(p) > 0 for p in ctrl.poles(system)):
            raise UnstableSystemError(ctrl.poles(system), operation="step_info")

    # Simulate step response
    t, y = ctrl.step_response(system)
    y = np.array(y).flatten()

    # Steady state value
    ss = y[-1]

    # Handle zero or near-zero steady state
    if abs(ss) < 1e-10:
        return {
            "rise_time": float("nan"),
            "settling_time": float("nan"),
            "overshoot": 0.0,
            "peak_time": 0.0,
            "peak_value": 0.0,
            "steady_state": 0.0,
        }

    y_norm = y / ss

    # Rise time (10% to 90%)
    try:
        t_10 = t[np.where(y_norm >= 0.1)[0][0]]
        t_90 = t[np.where(y_norm >= 0.9)[0][0]]
        rise_time = t_90 - t_10
    except IndexError:
        rise_time = float("nan")

    # Settling time
    settled = np.abs(y_norm - 1) <= settling_threshold
    if np.any(settled):
        outside = np.where(~settled)[0]
        if len(outside) > 0:
            settling_time = t[outside[-1]]
        else:
            settling_time = 0.0
    else:
        settling_time = t[-1]

    # Overshoot
    peak_idx = np.argmax(y)
    peak_value = y[peak_idx]
    peak_time = t[peak_idx]
    overshoot = max(0, (peak_value - ss) / abs(ss) * 100)

    return {
        "rise_time": float(rise_time),
        "settling_time": float(settling_time),
        "overshoot": float(overshoot),
        "peak_time": float(peak_time),
        "peak_value": float(peak_value),
        "steady_state": float(ss),
    }
