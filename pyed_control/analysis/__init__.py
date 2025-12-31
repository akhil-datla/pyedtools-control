"""
System analysis module.

Provides functions for analyzing control systems:
- Stability analysis (poles, zeros, Routh-Hurwitz)
- Time-domain response characteristics
- Frequency-domain characteristics
- Steady-state error analysis
- State-space analysis (controllability, observability)
- Sensitivity and robustness analysis
"""

from .stability import (
    is_stable,
    poles,
    zeros,
    routh_hurwitz,
    system_type,
    error_constants,
    steady_state_error,
)
from .time_response import step_info
from .frequency import margins
from .state_space import (
    controllability_matrix,
    observability_matrix,
    is_controllable,
    is_observable,
    state_space_analysis,
    controllable_form,
    observable_form,
)
from .sensitivity import (
    sensitivity_function,
    complementary_sensitivity,
    loop_transfer_function,
    robustness_margins,
    plot_sensitivity,
    disk_margin,
)

__all__ = [
    # Stability
    "is_stable",
    "poles",
    "zeros",
    "routh_hurwitz",
    # Steady-state error
    "system_type",
    "error_constants",
    "steady_state_error",
    # Time response
    "step_info",
    # Frequency response
    "margins",
    # State-space analysis
    "controllability_matrix",
    "observability_matrix",
    "is_controllable",
    "is_observable",
    "state_space_analysis",
    "controllable_form",
    "observable_form",
    # Sensitivity/Robustness
    "sensitivity_function",
    "complementary_sensitivity",
    "loop_transfer_function",
    "robustness_margins",
    "plot_sensitivity",
    "disk_margin",
]
