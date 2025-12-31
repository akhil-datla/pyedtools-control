"""
Stability analysis functions.
"""

import numpy as np
import control as ctrl
from typing import Union, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.system import System


def is_stable(
    sys: Union["System", ctrl.TransferFunction],
    show_work: bool = False,
) -> bool:
    """
    Check if a system is stable.

    A system is stable if all poles have negative real parts
    (all poles in the left half-plane).

    Parameters
    ----------
    sys : System or TransferFunction
        The system to analyze
    show_work : bool
        If True, print step-by-step analysis

    Returns
    -------
    bool
        True if the system is stable

    Examples
    --------
    >>> sys = ctrl.second_order(wn=2, zeta=0.5)
    >>> is_stable(sys)
    True

    >>> is_stable(sys, show_work=True)
    # Prints detailed pole analysis
    """
    from ..core.system import System

    if isinstance(sys, System):
        system_poles = sys.poles
        sys_obj = sys._sys
    else:
        system_poles = ctrl.poles(sys)
        sys_obj = sys

    stable = all(np.real(p) < 0 for p in system_poles)

    if show_work:
        print("\nStability Analysis (Pole Location Method)")
        print("=" * 45)

        # Show transfer function
        if hasattr(sys_obj, "num") and hasattr(sys_obj, "den"):
            print("\nTransfer function denominator determines poles.")

        print("\nStep 1: Find poles (roots of denominator)")
        for i, p in enumerate(system_poles):
            if np.isreal(p):
                real_p = np.real(p)
                status = "< 0 OK" if real_p < 0 else "> 0 UNSTABLE" if real_p > 0 else "= 0 MARGINAL"
                print(f"  Pole {i+1}: s = {real_p:.4f}  (Real part {status})")
            else:
                real_p = np.real(p)
                imag_p = np.imag(p)
                status = "< 0 OK" if real_p < 0 else "> 0 UNSTABLE" if real_p > 0 else "= 0 MARGINAL"
                print(f"  Pole {i+1}: s = {real_p:.4f} + {imag_p:.4f}j  (Real part {status})")

        print("\nStep 2: Check pole locations")
        n_unstable = sum(1 for p in system_poles if np.real(p) > 0)
        n_marginal = sum(1 for p in system_poles if np.real(p) == 0)

        if n_unstable > 0:
            print(f"  {n_unstable} pole(s) in right half-plane (RHP)")
        if n_marginal > 0:
            print(f"  {n_marginal} pole(s) on imaginary axis")
        if n_unstable == 0 and n_marginal == 0:
            print("  All poles in left half-plane (LHP)")

        print("\nConclusion:", end=" ")
        if stable:
            print("The system is STABLE.")
        elif n_marginal > 0 and n_unstable == 0:
            print("The system is MARGINALLY STABLE.")
        else:
            print("The system is UNSTABLE.")

    return stable


def poles(
    sys: Union["System", ctrl.TransferFunction],
    detailed: bool = False,
) -> Union[np.ndarray, List[dict]]:
    """
    Get system poles with optional detailed analysis.

    Parameters
    ----------
    sys : System or TransferFunction
        The system to analyze
    detailed : bool
        If True, return list of dicts with pole properties

    Returns
    -------
    ndarray or list of dict
        Pole locations, or detailed info if detailed=True

    Examples
    --------
    >>> poles(sys)
    array([-1.+2.j, -1.-2.j])

    >>> poles(sys, detailed=True)
    [{'location': (-1+2j), 'damping_ratio': 0.447, 'natural_freq': 2.236, ...}, ...]
    """
    from ..core.system import System

    if isinstance(sys, System):
        system_poles = sys.poles
    else:
        system_poles = ctrl.poles(sys)

    if not detailed:
        return system_poles

    result = []
    processed = set()

    for i, p in enumerate(system_poles):
        if i in processed:
            continue

        info = {
            "location": complex(p),
            "real_part": float(np.real(p)),
            "imag_part": float(np.imag(p)),
            "stable": np.real(p) < 0,
        }

        if np.isreal(p):
            info["type"] = "real"
            info["time_constant"] = -1 / np.real(p) if np.real(p) != 0 else float("inf")
        else:
            # Complex pole - find conjugate
            for j, p2 in enumerate(system_poles):
                if j > i and np.isclose(p, np.conj(p2)):
                    processed.add(j)
                    break

            sigma = np.real(p)
            omega_d = abs(np.imag(p))
            wn = np.sqrt(sigma**2 + omega_d**2)
            zeta = -sigma / wn if wn != 0 else 0

            info["type"] = "complex"
            info["natural_frequency"] = float(wn)
            info["damping_ratio"] = float(zeta)
            info["damped_frequency"] = float(omega_d)

        result.append(info)

    return result


def zeros(
    sys: Union["System", ctrl.TransferFunction],
) -> np.ndarray:
    """
    Get system zeros.

    Parameters
    ----------
    sys : System or TransferFunction
        The system to analyze

    Returns
    -------
    ndarray
        Zero locations
    """
    from ..core.system import System

    if isinstance(sys, System):
        return sys.zeros
    else:
        return ctrl.zeros(sys)


def routh_hurwitz(
    sys: Union["System", ctrl.TransferFunction],
    show_work: bool = True,
) -> dict:
    """
    Apply the Routh-Hurwitz stability criterion.

    This is an algebraic method to determine stability without
    computing the actual pole locations. Useful for systems with
    symbolic or parametric coefficients.

    Parameters
    ----------
    sys : System or TransferFunction
        The system to analyze
    show_work : bool
        If True, print the Routh array construction (default: True)

    Returns
    -------
    dict
        Dictionary with keys:
        - 'is_stable': bool
        - 'routh_array': 2D numpy array
        - 'first_column': First column of Routh array
        - 'sign_changes': Number of sign changes in first column
        - 'rhp_poles': Number of right-half-plane poles

    Examples
    --------
    >>> result = routh_hurwitz(sys, show_work=True)

    Routh-Hurwitz Stability Analysis
    =================================

    Characteristic polynomial: s^3 + 6s^2 + 11s + 6

    Constructing Routh array:

    s^3 |  1    11
    s^2 |  6     6
    s^1 | 10     0
    s^0 |  6     0

    First column: [1, 6, 10, 6]
    Sign changes: 0

    Conclusion: No sign changes = No poles in RHP
                The system is STABLE.
    """
    from ..core.system import System

    if isinstance(sys, System):
        den = sys.den
    else:
        den = np.array(sys.den[0][0])

    # Ensure coefficients are in descending order of power
    coeffs = np.array(den).flatten()
    n = len(coeffs)

    if n < 2:
        return {
            'is_stable': True,
            'routh_array': np.array([[coeffs[0]]]),
            'first_column': [coeffs[0]],
            'sign_changes': 0,
            'rhp_poles': 0,
        }

    # Build Routh array
    # Number of rows = n (order + 1)
    # Number of columns = ceil(n/2)
    n_cols = (n + 1) // 2
    routh = np.zeros((n, n_cols))

    # Fill first two rows with coefficients
    routh[0, :len(coeffs[0::2])] = coeffs[0::2]  # Even indices
    routh[1, :len(coeffs[1::2])] = coeffs[1::2]  # Odd indices

    # Fill remaining rows
    for i in range(2, n):
        for j in range(n_cols - 1):
            if routh[i-1, 0] == 0:
                # Handle zero in first column - replace with small epsilon
                routh[i-1, 0] = 1e-10

            a = routh[i-2, 0]
            b = routh[i-2, j+1] if j+1 < n_cols else 0
            c = routh[i-1, 0]
            d = routh[i-1, j+1] if j+1 < n_cols else 0

            routh[i, j] = (c * b - a * d) / c if c != 0 else 0

    # Count sign changes in first column
    first_col = routh[:, 0]
    sign_changes = 0
    for i in range(1, len(first_col)):
        if first_col[i-1] * first_col[i] < 0:
            sign_changes += 1

    is_stable = sign_changes == 0 and all(first_col > 0)

    if show_work:
        print("\nRouth-Hurwitz Stability Analysis")
        print("=" * 40)

        # Format polynomial
        poly_str = _format_polynomial(coeffs)
        print(f"\nCharacteristic polynomial: {poly_str}")

        print("\nConstructing Routh array:\n")

        # Print Routh array
        for i in range(n):
            power = n - 1 - i
            row_str = f"s^{power} |"
            for j in range(n_cols):
                if abs(routh[i, j]) > 1e-9:
                    row_str += f"  {routh[i, j]:8.4g}"
                else:
                    row_str += f"  {'0':>8}"
            print(row_str)

        print(f"\nFirst column: [{', '.join(f'{x:.4g}' for x in first_col)}]")
        print(f"Sign changes: {sign_changes}")

        print("\nConclusion:", end=" ")
        if is_stable:
            print("No sign changes = No poles in RHP")
            print("             The system is STABLE.")
        else:
            print(f"{sign_changes} sign change(s) = {sign_changes} pole(s) in RHP")
            print("             The system is UNSTABLE.")

    return {
        'is_stable': is_stable,
        'routh_array': routh,
        'first_column': list(first_col),
        'sign_changes': sign_changes,
        'rhp_poles': sign_changes,
    }


def _format_polynomial(coeffs: np.ndarray) -> str:
    """Format polynomial coefficients as a string."""
    n = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        power = n - i
        if abs(c) < 1e-10:
            continue
        if power == 0:
            terms.append(f"{c:.4g}")
        elif power == 1:
            if c == 1:
                terms.append("s")
            else:
                terms.append(f"{c:.4g}s")
        else:
            if c == 1:
                terms.append(f"s^{power}")
            else:
                terms.append(f"{c:.4g}s^{power}")
    return " + ".join(terms) if terms else "0"


def system_type(sys: Union["System", ctrl.TransferFunction]) -> int:
    """
    Determine the system type (number of integrators in open loop).

    The system type determines the steady-state error characteristics:
    - Type 0: Finite error to step input
    - Type 1: Zero error to step, finite error to ramp
    - Type 2: Zero error to step and ramp, finite error to parabola

    Parameters
    ----------
    sys : System or TransferFunction
        The system to analyze

    Returns
    -------
    int
        System type (0, 1, 2, ...)

    Examples
    --------
    >>> system_type(sys)
    1  # Type-1 system (one integrator)
    """
    from ..core.system import System

    if isinstance(sys, System):
        system_poles = sys.poles
    else:
        system_poles = ctrl.poles(sys)

    # Count poles at origin
    n_integrators = sum(1 for p in system_poles if np.isclose(p, 0, atol=1e-10))
    return n_integrators


def error_constants(
    sys: Union["System", ctrl.TransferFunction],
    show_work: bool = False,
) -> dict:
    """
    Calculate the static error constants Kp, Kv, Ka.

    These constants determine the steady-state error for different
    input types:
    - Kp (position constant): e_ss = 1/(1 + Kp) for step input
    - Kv (velocity constant): e_ss = 1/Kv for ramp input
    - Ka (acceleration constant): e_ss = 1/Ka for parabolic input

    Parameters
    ----------
    sys : System or TransferFunction
        The open-loop system
    show_work : bool
        If True, print the calculations

    Returns
    -------
    dict
        Dictionary with keys: 'Kp', 'Kv', 'Ka', 'type'

    Examples
    --------
    >>> constants = error_constants(sys, show_work=True)

    Static Error Constants
    ======================

    System Type: 1 (one integrator)

    Calculating error constants using limits:
      Kp = lim(s->0) G(s) = inf
      Kv = lim(s->0) s*G(s) = 10
      Ka = lim(s->0) s^2*G(s) = 0

    Steady-State Errors:
      Step input:     e_ss = 1/(1+Kp) = 0
      Ramp input:     e_ss = 1/Kv = 0.1
      Parabola input: e_ss = 1/Ka = inf
    """
    from ..core.system import System

    if isinstance(sys, System):
        tf_sys = sys._sys
    else:
        tf_sys = sys

    sys_type = system_type(sys)

    # Calculate error constants by evaluating limits
    # Kp = lim(s->0) G(s)
    # Kv = lim(s->0) s*G(s)
    # Ka = lim(s->0) s^2*G(s)

    try:
        dc = float(ctrl.dcgain(tf_sys))
        if np.isinf(dc) or np.isnan(dc):
            Kp = float('inf')
        else:
            Kp = dc
    except Exception:
        Kp = float('inf') if sys_type >= 1 else 0

    # For Kv, we need lim(s->0) s*G(s)
    if sys_type >= 2:
        Kv = float('inf')
    elif sys_type == 1:
        # Evaluate s*G(s) at s=0
        # This is the DC gain of the system with one integrator removed
        try:
            # Get transfer function coefficients
            if isinstance(sys, System):
                num = sys.num
                den = sys.den
            else:
                num = np.array(tf_sys.num[0][0])
                den = np.array(tf_sys.den[0][0])

            # s*G(s) removes one s from denominator
            # Evaluate at s=0: num(0) / den'(0) where den' has one less s factor
            num_val = num[-1] if len(num) > 0 else 0
            # Remove one zero from denominator (divide by s)
            if den[-1] == 0:
                den_reduced = den[:-1]
                den_val = den_reduced[-1] if len(den_reduced) > 0 else 1
            else:
                den_val = den[-1]
            Kv = num_val / den_val if den_val != 0 else float('inf')
        except Exception:
            Kv = float('inf')
    else:
        Kv = 0

    # For Ka, we need lim(s->0) s^2*G(s)
    if sys_type >= 3:
        Ka = float('inf')
    elif sys_type == 2:
        try:
            if isinstance(sys, System):
                num = sys.num
                den = sys.den
            else:
                num = np.array(tf_sys.num[0][0])
                den = np.array(tf_sys.den[0][0])

            num_val = num[-1] if len(num) > 0 else 0
            # Remove two zeros from denominator
            den_reduced = den
            for _ in range(2):
                if len(den_reduced) > 0 and den_reduced[-1] == 0:
                    den_reduced = den_reduced[:-1]
            den_val = den_reduced[-1] if len(den_reduced) > 0 else 1
            Ka = num_val / den_val if den_val != 0 else float('inf')
        except Exception:
            Ka = float('inf')
    else:
        Ka = 0

    result = {
        'Kp': Kp,
        'Kv': Kv,
        'Ka': Ka,
        'type': sys_type,
    }

    if show_work:
        print("\nStatic Error Constants")
        print("=" * 40)

        type_names = {0: "zero", 1: "one", 2: "two", 3: "three"}
        type_name = type_names.get(sys_type, str(sys_type))
        integrator_word = "integrator" if sys_type == 1 else "integrators"
        print(f"\nSystem Type: {sys_type} ({type_name} {integrator_word})")

        print("\nCalculating error constants using limits:")
        print(f"  Kp = lim(s->0) G(s) = {Kp if not np.isinf(Kp) else 'inf'}")
        print(f"  Kv = lim(s->0) s*G(s) = {Kv if not np.isinf(Kv) else 'inf'}")
        print(f"  Ka = lim(s->0) s^2*G(s) = {Ka if not np.isinf(Ka) else 'inf'}")

        print("\nSteady-State Errors:")
        # Step error
        if np.isinf(Kp):
            step_err = 0
            print(f"  Step input:     e_ss = 1/(1+Kp) = 0")
        else:
            step_err = 1 / (1 + Kp) if Kp >= 0 else float('inf')
            print(f"  Step input:     e_ss = 1/(1+Kp) = {step_err:.4g}")

        # Ramp error
        if np.isinf(Kv):
            ramp_err = 0
            print(f"  Ramp input:     e_ss = 1/Kv = 0")
        elif Kv > 0:
            ramp_err = 1 / Kv
            print(f"  Ramp input:     e_ss = 1/Kv = {ramp_err:.4g}")
        else:
            print(f"  Ramp input:     e_ss = 1/Kv = inf (unbounded)")

        # Parabola error
        if np.isinf(Ka):
            para_err = 0
            print(f"  Parabola input: e_ss = 1/Ka = 0")
        elif Ka > 0:
            para_err = 1 / Ka
            print(f"  Parabola input: e_ss = 1/Ka = {para_err:.4g}")
        else:
            print(f"  Parabola input: e_ss = 1/Ka = inf (unbounded)")

    return result


def steady_state_error(
    sys: Union["System", ctrl.TransferFunction],
    input_type: str = "step",
    input_magnitude: float = 1.0,
    show_work: bool = False,
) -> float:
    """
    Calculate the steady-state error for a given input type.

    Uses the Final Value Theorem for stable systems.

    Parameters
    ----------
    sys : System or TransferFunction
        The closed-loop system
    input_type : str
        Type of input: 'step', 'ramp', or 'parabola'
    input_magnitude : float
        Magnitude of the input (default: 1.0)
    show_work : bool
        If True, print the calculation steps

    Returns
    -------
    float
        Steady-state error

    Examples
    --------
    >>> e_ss = steady_state_error(closed_loop, input_type='step', show_work=True)

    Steady-State Error Calculation (Step Input)
    ============================================

    Using Final Value Theorem: e_ss = lim(s->0) s * E(s)

    For unit step input R(s) = 1/s:
      E(s) = R(s) - Y(s) = R(s) * (1 - T(s))
      where T(s) is the closed-loop transfer function

      e_ss = lim(s->0) s * (1/s) * (1 - T(s))
           = lim(s->0) (1 - T(s))
           = 1 - T(0)
           = 1 - 0.833
           = 0.167 (16.7%)
    """
    from ..core.system import System
    from ..exceptions import UnstableSystemError

    if isinstance(sys, System):
        if not sys.is_stable:
            raise UnstableSystemError(sys.poles, operation="steady_state_error")
        tf_sys = sys._sys
    else:
        tf_sys = sys

    # Get DC gain (closed-loop gain at s=0)
    try:
        T0 = float(ctrl.dcgain(tf_sys))
    except Exception:
        T0 = 0

    if input_type.lower() == 'step':
        e_ss = (1 - T0) * input_magnitude
    elif input_type.lower() == 'ramp':
        # For ramp, need the velocity error constant of the closed-loop
        # This is more complex - approximate by simulation or return inf
        e_ss = float('inf')  # Simplified - would need open-loop analysis
    elif input_type.lower() == 'parabola':
        e_ss = float('inf')  # Simplified
    else:
        raise ValueError(f"Unknown input type: {input_type}. Use 'step', 'ramp', or 'parabola'")

    if show_work:
        print(f"\nSteady-State Error Calculation ({input_type.title()} Input)")
        print("=" * 50)

        if input_type.lower() == 'step':
            print("\nUsing Final Value Theorem: e_ss = lim(s->0) s * E(s)")
            print(f"\nFor step input R(s) = {input_magnitude}/s:")
            print("  E(s) = R(s) - Y(s) = R(s) * (1 - T(s))")
            print("  where T(s) is the closed-loop transfer function")
            print("")
            print("  e_ss = lim(s->0) s * (1/s) * (1 - T(s))")
            print("       = lim(s->0) (1 - T(s))")
            print(f"       = 1 - T(0)")
            print(f"       = 1 - {T0:.4g}")
            print(f"       = {e_ss:.4g}")
            if abs(e_ss) > 0.001:
                print(f"       = {abs(e_ss/input_magnitude)*100:.1f}% of input")
        else:
            print(f"\nFor {input_type} input, steady-state error requires")
            print("open-loop error constant analysis.")

    return e_ss
