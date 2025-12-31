"""
Custom exceptions with educational error messages.

All exceptions include:
- Clear explanation of what went wrong
- Suggestions for how to fix it
- Context-specific guidance
"""

import numpy as np


class ControlError(Exception):
    """Base exception for control systems errors."""
    pass


class UnstableSystemError(ControlError):
    """
    Raised when an operation requires a stable system.

    Provides educational context about why the system is unstable
    and suggestions for stabilization.
    """

    def __init__(self, poles, message=None, operation=None):
        self.poles = poles
        self.unstable_poles = [p for p in poles if np.real(p) > 0]

        msg = message or "This operation requires a stable system."

        if operation:
            msg = f"Cannot perform '{operation}' on an unstable system."

        msg += "\n\n"
        msg += "Your system has unstable poles (positive real parts):\n"
        for p in self.unstable_poles:
            if np.isreal(p):
                msg += f"  s = {np.real(p):.4f}\n"
            else:
                msg += f"  s = {np.real(p):.4f} +/- {abs(np.imag(p)):.4f}j\n"

        msg += "\nWhy this matters:\n"
        msg += "  Poles with positive real parts cause the response to grow\n"
        msg += "  unboundedly over time instead of settling to a steady value.\n"

        msg += "\nSuggestions:\n"
        msg += "  - Add feedback control to move poles to the left half-plane\n"
        msg += "  - Check your transfer function coefficients for errors\n"
        msg += "  - Use sys.poles to see all pole locations\n"
        msg += "  - Use sys.pole_zero_map() to visualize pole positions"

        super().__init__(msg)


class ImproperSystemError(ControlError):
    """
    Raised when a system is improper (more zeros than poles).
    """

    def __init__(self, num_order, den_order):
        self.num_order = num_order
        self.den_order = den_order

        msg = f"System is improper: numerator order ({num_order}) > denominator order ({den_order})\n\n"
        msg += "Why this matters:\n"
        msg += "  Improper systems are not physically realizable.\n"
        msg += "  They would require infinite gain at high frequencies.\n"
        msg += "\nSuggestions:\n"
        msg += "  - Check your transfer function coefficients\n"
        msg += "  - Add more poles (increase denominator order)\n"
        msg += "  - For derivative action, use a filtered derivative:\n"
        msg += "    Instead of: s\n"
        msg += "    Use: s / (tau*s + 1)  where tau is small"

        super().__init__(msg)


class InvalidParameterError(ControlError):
    """
    Raised when a parameter has an invalid value.
    """

    def __init__(self, param_name, value, requirement, suggestion=None, context=None):
        self.param_name = param_name
        self.value = value
        self.requirement = requirement

        msg = f"Invalid parameter: {param_name} = {value}\n"
        msg += f"Requirement: {requirement}\n"

        if context:
            msg += f"\nContext: {context}\n"

        if suggestion:
            msg += f"\nSuggestion: {suggestion}"

        super().__init__(msg)


class NoSolutionError(ControlError):
    """
    Raised when a design problem has no solution.
    """

    def __init__(self, message, specs=None, limitations=None, suggestions=None):
        msg = message

        if specs:
            msg += "\n\nRequested specifications:\n"
            for spec, value in specs.items():
                msg += f"  - {spec}: {value}\n"

        if limitations:
            msg += "\nSystem limitations:\n"
            for lim, value in limitations.items():
                msg += f"  - {lim}: {value}\n"

        if suggestions:
            msg += "\nSuggestions:\n"
            for s in suggestions:
                msg += f"  - {s}\n"

        super().__init__(msg)


class ConvergenceError(ControlError):
    """
    Raised when an iterative algorithm fails to converge.
    """

    def __init__(self, algorithm, max_iterations, message=None):
        msg = message or f"Algorithm '{algorithm}' did not converge after {max_iterations} iterations."
        msg += "\n\nThis may indicate:\n"
        msg += "  - The problem has no solution\n"
        msg += "  - Initial conditions are too far from solution\n"
        msg += "  - Numerical precision issues\n"
        msg += "\nSuggestions:\n"
        msg += "  - Try different initial parameter values\n"
        msg += "  - Check if the design specifications are achievable\n"
        msg += "  - Simplify the problem or relax constraints"

        super().__init__(msg)


def validate_positive(value, name, allow_zero=False):
    """Validate that a value is positive (or non-negative if allow_zero)."""
    if allow_zero:
        if value < 0:
            raise InvalidParameterError(
                name, value,
                f"{name} must be non-negative",
                f"Use a value >= 0 for {name}"
            )
    else:
        if value <= 0:
            raise InvalidParameterError(
                name, value,
                f"{name} must be positive",
                f"Use a value > 0 for {name}"
            )


def validate_damping_ratio(zeta, name="zeta"):
    """Validate a damping ratio value."""
    if zeta < 0:
        raise InvalidParameterError(
            name, zeta,
            "Damping ratio must be non-negative",
            "Use zeta >= 0. Common values: 0.3-0.8 for underdamped, 1.0 for critical"
        )


def validate_transfer_function(num, den):
    """Validate transfer function coefficients."""
    if len(den) == 0:
        raise InvalidParameterError(
            "den", den,
            "Denominator cannot be empty",
            "Provide at least one coefficient for the denominator"
        )

    if den[0] == 0:
        raise InvalidParameterError(
            "den", den,
            "Leading denominator coefficient cannot be zero",
            "The highest-order coefficient must be non-zero"
        )

    num_order = len(num) - 1 if len(num) > 0 else 0
    den_order = len(den) - 1

    if num_order > den_order:
        raise ImproperSystemError(num_order, den_order)
