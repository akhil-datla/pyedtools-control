"""
Plotting module for control systems visualization.

Provides functions for creating beautiful, educational plots:
- Time-domain: step, impulse
- Frequency-domain: bode, nyquist
- Root locus
- Pole-zero maps
- Comparison plots
"""

from .time_plots import step, impulse
from .frequency_plots import bode, nyquist
from .root_locus import root_locus
from .pole_zero import pzmap
from .comparison import compare

__all__ = [
    "step",
    "impulse",
    "bode",
    "nyquist",
    "root_locus",
    "pzmap",
    "compare",
]
