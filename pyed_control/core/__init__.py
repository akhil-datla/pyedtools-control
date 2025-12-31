"""Core module containing the central System class."""

from .system import System
from .discrete import DiscreteSystem, c2d, d2c, explain_discretization

__all__ = [
    'System',
    'DiscreteSystem',
    'c2d',
    'd2c',
    'explain_discretization',
]
