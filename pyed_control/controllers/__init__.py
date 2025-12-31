"""
Controller design module.

Provides classes for designing various controllers:
- PID controllers with multiple tuning methods
- Lead and lag compensators
"""

from .pid import PID
from .lead_lag import Lead, Lag, LeadLag

__all__ = ["PID", "Lead", "Lag", "LeadLag"]
