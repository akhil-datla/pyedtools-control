"""
State-space analysis functions.

Provides controllability, observability, and related state-space
analysis tools for control systems education.
"""

import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
import control as ctrl


def controllability_matrix(
    sys: "System",
    show_work: bool = False,
) -> np.ndarray:
    """
    Compute the controllability matrix.

    The controllability matrix is: C = [B, A*B, A^2*B, ..., A^(n-1)*B]

    A system is controllable if and only if the controllability matrix
    has full row rank (rank = n, where n is the system order).

    Parameters
    ----------
    sys : System
        The system to analyze
    show_work : bool
        If True, print the construction process

    Returns
    -------
    np.ndarray
        The controllability matrix

    Examples
    --------
    >>> from pyed_control import second_order
    >>> from pyed_control.analysis import controllability_matrix
    >>> sys = second_order(wn=2, zeta=0.5)
    >>> C = controllability_matrix(sys, show_work=True)
    """
    from ..core.system import System

    # Get state-space representation
    if isinstance(sys, System):
        ss = ctrl.tf2ss(sys._sys)
    else:
        ss = sys

    A = np.array(ss.A)
    B = np.array(ss.B)
    n = A.shape[0]

    if show_work:
        print("\n" + "=" * 60)
        print(" Controllability Matrix Construction")
        print("=" * 60)
        print(f"\nSystem order: n = {n}")
        print("\nState matrix A:")
        print(_format_matrix(A))
        print("\nInput matrix B:")
        print(_format_matrix(B))
        print("\nControllability matrix: C = [B | A*B | A^2*B | ... | A^(n-1)*B]")
        print()

    # Build controllability matrix
    C_cols = [B]
    A_power = A.copy()

    for i in range(1, n):
        C_cols.append(A_power @ B)
        if show_work:
            if i == 1:
                print(f"A*B =")
            else:
                print(f"A^{i}*B =")
            print(_format_matrix(C_cols[-1]))
        A_power = A_power @ A

    C = np.hstack(C_cols)

    if show_work:
        print("\nComplete controllability matrix:")
        print(_format_matrix(C))
        print(f"\nRank of C: {np.linalg.matrix_rank(C)}")
        print(f"Required rank for controllability: {n}")

    return C


def observability_matrix(
    sys: "System",
    show_work: bool = False,
) -> np.ndarray:
    """
    Compute the observability matrix.

    The observability matrix is: O = [C; C*A; C*A^2; ...; C*A^(n-1)]

    A system is observable if and only if the observability matrix
    has full column rank (rank = n, where n is the system order).

    Parameters
    ----------
    sys : System
        The system to analyze
    show_work : bool
        If True, print the construction process

    Returns
    -------
    np.ndarray
        The observability matrix

    Examples
    --------
    >>> from pyed_control import second_order
    >>> from pyed_control.analysis import observability_matrix
    >>> sys = second_order(wn=2, zeta=0.5)
    >>> O = observability_matrix(sys, show_work=True)
    """
    from ..core.system import System

    if isinstance(sys, System):
        ss = ctrl.tf2ss(sys._sys)
    else:
        ss = sys

    A = np.array(ss.A)
    C = np.array(ss.C)
    n = A.shape[0]

    if show_work:
        print("\n" + "=" * 60)
        print(" Observability Matrix Construction")
        print("=" * 60)
        print(f"\nSystem order: n = {n}")
        print("\nState matrix A:")
        print(_format_matrix(A))
        print("\nOutput matrix C:")
        print(_format_matrix(C))
        print("\nObservability matrix: O = [C; C*A; C*A^2; ...; C*A^(n-1)]^T")
        print()

    # Build observability matrix
    O_rows = [C]
    A_power = A.copy()

    for i in range(1, n):
        O_rows.append(C @ A_power)
        if show_work:
            if i == 1:
                print(f"C*A =")
            else:
                print(f"C*A^{i} =")
            print(_format_matrix(O_rows[-1]))
        A_power = A_power @ A

    O = np.vstack(O_rows)

    if show_work:
        print("\nComplete observability matrix:")
        print(_format_matrix(O))
        print(f"\nRank of O: {np.linalg.matrix_rank(O)}")
        print(f"Required rank for observability: {n}")

    return O


def is_controllable(
    sys: "System",
    show_work: bool = False,
) -> bool:
    """
    Check if the system is controllable.

    A system is controllable if and only if the controllability matrix
    has full rank.

    Parameters
    ----------
    sys : System
        The system to analyze
    show_work : bool
        If True, print the analysis process

    Returns
    -------
    bool
        True if the system is controllable

    Examples
    --------
    >>> from pyed_control import systems
    >>> motor = systems.dc_motor()
    >>> is_controllable(motor, show_work=True)
    """
    from ..core.system import System

    if isinstance(sys, System):
        ss = ctrl.tf2ss(sys._sys)
    else:
        ss = sys

    n = ss.A.shape[0]
    C = controllability_matrix(sys, show_work=show_work)
    rank = np.linalg.matrix_rank(C)

    controllable = rank == n

    if show_work:
        print()
        if controllable:
            print("CONTROLLABLE: Rank equals system order.")
            print("  -> All states can be driven to any desired value")
            print("     using appropriate input.")
        else:
            print(f"NOT CONTROLLABLE: Rank ({rank}) < System order ({n})")
            print("  -> Some states cannot be influenced by the input.")
            print("  -> State feedback control cannot place all poles.")

    return controllable


def is_observable(
    sys: "System",
    show_work: bool = False,
) -> bool:
    """
    Check if the system is observable.

    A system is observable if and only if the observability matrix
    has full rank.

    Parameters
    ----------
    sys : System
        The system to analyze
    show_work : bool
        If True, print the analysis process

    Returns
    -------
    bool
        True if the system is observable

    Examples
    --------
    >>> from pyed_control import systems
    >>> motor = systems.dc_motor()
    >>> is_observable(motor, show_work=True)
    """
    from ..core.system import System

    if isinstance(sys, System):
        ss = ctrl.tf2ss(sys._sys)
    else:
        ss = sys

    n = ss.A.shape[0]
    O = observability_matrix(sys, show_work=show_work)
    rank = np.linalg.matrix_rank(O)

    observable = rank == n

    if show_work:
        print()
        if observable:
            print("OBSERVABLE: Rank equals system order.")
            print("  -> All states can be determined from the output history.")
            print("  -> State estimation (observer design) is possible.")
        else:
            print(f"NOT OBSERVABLE: Rank ({rank}) < System order ({n})")
            print("  -> Some states cannot be determined from output.")
            print("  -> Full-state observer design is not possible.")

    return observable


def state_space_analysis(
    sys: "System",
    show_work: bool = True,
) -> Dict[str, Any]:
    """
    Comprehensive state-space analysis.

    Computes controllability, observability, and related properties.

    Parameters
    ----------
    sys : System
        The system to analyze
    show_work : bool
        If True, print detailed analysis

    Returns
    -------
    dict
        Dictionary with analysis results

    Examples
    --------
    >>> from pyed_control import systems
    >>> motor = systems.dc_motor()
    >>> results = state_space_analysis(motor)
    """
    from ..core.system import System

    if isinstance(sys, System):
        ss = ctrl.tf2ss(sys._sys)
    else:
        ss = sys

    A = np.array(ss.A)
    B = np.array(ss.B)
    C = np.array(ss.C)
    D = np.array(ss.D)
    n = A.shape[0]

    if show_work:
        print("\n" + "=" * 60)
        print(" State-Space Analysis")
        if hasattr(sys, '_name') and sys._name:
            print(f" System: {sys._name}")
        print("=" * 60)

        print(f"\nSystem order: n = {n}")
        print(f"Number of inputs: m = {B.shape[1]}")
        print(f"Number of outputs: p = {C.shape[0]}")

        print("\n--- State-Space Matrices ---")
        print("\nA (state matrix):")
        print(_format_matrix(A))
        print("\nB (input matrix):")
        print(_format_matrix(B))
        print("\nC (output matrix):")
        print(_format_matrix(C))
        print("\nD (feedthrough matrix):")
        print(_format_matrix(D))

    # Compute controllability
    C_mat = controllability_matrix(sys, show_work=False)
    c_rank = np.linalg.matrix_rank(C_mat)
    controllable = c_rank == n

    # Compute observability
    O_mat = observability_matrix(sys, show_work=False)
    o_rank = np.linalg.matrix_rank(O_mat)
    observable = o_rank == n

    # Eigenvalues (poles)
    eigenvalues = np.linalg.eigvals(A)

    if show_work:
        print("\n--- Controllability ---")
        print(f"Controllability matrix rank: {c_rank} / {n}")
        if controllable:
            print("System is CONTROLLABLE")
        else:
            print("System is NOT CONTROLLABLE")

        print("\n--- Observability ---")
        print(f"Observability matrix rank: {o_rank} / {n}")
        if observable:
            print("System is OBSERVABLE")
        else:
            print("System is NOT OBSERVABLE")

        print("\n--- Eigenvalues (Poles) ---")
        for eig in eigenvalues:
            if np.isreal(eig):
                print(f"  lambda = {np.real(eig):.4f}")
            else:
                print(f"  lambda = {np.real(eig):.4f} ± {abs(np.imag(eig)):.4f}j")

        print("\n--- Summary ---")
        stable = all(np.real(eig) < 0 for eig in eigenvalues)
        print(f"Stable: {stable}")
        print(f"Controllable: {controllable}")
        print(f"Observable: {observable}")

        if controllable and observable:
            print("\nThe system is both controllable and observable.")
            print("This means:")
            print("  - State feedback control can place all poles")
            print("  - Full-state observer can estimate all states")
            print("  - Output feedback can achieve arbitrary pole placement")
        elif controllable:
            print("\nThe system is controllable but not observable.")
            print("State feedback can work, but observer design is limited.")
        elif observable:
            print("\nThe system is observable but not controllable.")
            print("States can be estimated, but some cannot be controlled.")
        else:
            print("\nThe system is neither controllable nor observable.")
            print("Consider if this model correctly represents the physical system.")

        print()

    return {
        "order": n,
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "controllability_matrix": C_mat,
        "controllability_rank": c_rank,
        "is_controllable": controllable,
        "observability_matrix": O_mat,
        "observability_rank": o_rank,
        "is_observable": observable,
        "eigenvalues": eigenvalues,
        "is_stable": all(np.real(eig) < 0 for eig in eigenvalues),
    }


def controllable_form(sys: "System") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert system to controllable canonical form.

    The controllable canonical form has a specific structure that makes
    controller design straightforward.

    Parameters
    ----------
    sys : System
        The system to convert

    Returns
    -------
    tuple
        (A_c, B_c, C_c, D_c) matrices in controllable form

    Raises
    ------
    ValueError
        If the system is not controllable
    """
    from ..core.system import System

    if not is_controllable(sys):
        raise ValueError("System must be controllable to convert to controllable form")

    if isinstance(sys, System):
        ss = ctrl.tf2ss(sys._sys)
    else:
        ss = sys

    # Use python-control's canonical form conversion
    try:
        ss_ctrl = ctrl.canonical_form(ss, form='reachable')
        return ss_ctrl.A, ss_ctrl.B, ss_ctrl.C, ss_ctrl.D
    except Exception:
        # Fallback: compute manually from transfer function
        from ..core.system import System
        if isinstance(sys, System):
            num = sys.num
            den = sys.den
        else:
            tf = ctrl.tf(sys)
            num = np.array(tf.num[0][0])
            den = np.array(tf.den[0][0])

        # Normalize
        num = num / den[0]
        den = den / den[0]

        n = len(den) - 1

        # Build controllable form
        A_c = np.zeros((n, n))
        for i in range(n - 1):
            A_c[i, i + 1] = 1
        A_c[-1, :] = -den[1:][::-1]

        B_c = np.zeros((n, 1))
        B_c[-1, 0] = 1

        C_c = np.zeros((1, n))
        if len(num) <= n:
            C_c[0, :len(num)] = num[::-1]
        else:
            C_c[0, :] = num[1:n+1][::-1]

        D_c = np.array([[num[0]]]) if len(num) > n else np.array([[0]])

        return A_c, B_c, C_c, D_c


def observable_form(sys: "System") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert system to observable canonical form.

    The observable canonical form has a specific structure that makes
    observer design straightforward.

    Parameters
    ----------
    sys : System
        The system to convert

    Returns
    -------
    tuple
        (A_o, B_o, C_o, D_o) matrices in observable form

    Raises
    ------
    ValueError
        If the system is not observable
    """
    if not is_observable(sys):
        raise ValueError("System must be observable to convert to observable form")

    # Observable form is the dual of controllable form
    A_c, B_c, C_c, D_c = controllable_form(sys)
    return A_c.T, C_c.T, B_c.T, D_c.T


def _format_matrix(M: np.ndarray) -> str:
    """Format a matrix for display."""
    if M.ndim == 1:
        M = M.reshape(-1, 1)

    rows = []
    for row in M:
        row_str = "  [" + "  ".join(f"{x:8.4f}" for x in row) + "]"
        rows.append(row_str)
    return "\n".join(rows)
