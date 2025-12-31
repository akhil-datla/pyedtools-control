"""
Educational features module.

Provides functions to help students understand control systems concepts:
- explain(): Get explanations of concepts
- get_hint(): Get hints for improving a system
- check_design(): Verify if a system meets specifications
"""

from .explanations import explain, concept
from .hints import get_hint, check_design

__all__ = [
    "explain",
    "concept",
    "get_hint",
    "check_design",
]
