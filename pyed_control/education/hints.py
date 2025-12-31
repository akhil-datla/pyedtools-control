"""
Hints and design checking for control systems education.
"""

from typing import Optional, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.system import System


def get_hint(sys: "System", issue: Optional[str] = None):
    """
    Get hints for improving a system or fixing issues.

    Parameters
    ----------
    sys : System
        The system to analyze
    issue : str, optional
        Specific issue to address:
        - 'unstable': Help stabilizing
        - 'slow': Help speeding up response
        - 'overshoot': Help reducing overshoot
        - 'steady_state_error': Help reducing SSE
        If not provided, analyzes system and suggests improvements

    Examples
    --------
    >>> get_hint(closed_loop_sys, issue='overshoot')
    """
    print("\n" + "=" * 50)
    if issue:
        print(f"Hints for: {issue}")
    else:
        print("System Analysis and Hints")
    print("=" * 50)

    if issue == "unstable" or not sys.is_stable:
        _hint_unstable(sys)
    elif issue == "overshoot":
        _hint_overshoot(sys)
    elif issue == "slow":
        _hint_slow(sys)
    elif issue == "steady_state_error":
        _hint_sse(sys)
    else:
        _general_hints(sys)


def _hint_unstable(sys):
    """Provide hints for stabilizing an unstable system."""
    print("\nYour system is UNSTABLE.")
    print("\nUnstable poles:")
    for p in sys.poles:
        if p.real > 0:
            print(f"  s = {p}")

    print("\nTo stabilize:")
    print("  1. Add feedback control (negative feedback)")
    print("  2. Use a controller with appropriate gain")
    print("  3. Consider PID control:")
    print("     - Proportional: Moves poles left")
    print("     - Derivative: Adds damping")
    print("  4. For systems with RHP zeros, be careful - bandwidth is limited")

    print("\nQuick fix to try:")
    print("  >>> pid = ctrl.PID(kp=1)  # Start with proportional only")
    print("  >>> closed = plant.with_controller(pid)")
    print("  >>> closed.is_stable  # Check if now stable")


def _hint_overshoot(sys):
    """Provide hints for reducing overshoot."""
    try:
        info = sys.step_info()
        overshoot = info["overshoot"]
        print(f"\nCurrent overshoot: {overshoot:.1f}%")
    except Exception:
        overshoot = None
        print("\nCould not calculate overshoot (system may be unstable)")
        return

    print("\nTo reduce overshoot:")
    print("  1. Increase damping ratio:")
    print("     - For PID: Increase Kd (derivative gain)")
    print("     - For lead compensator: Increase phase boost")
    print("")
    print("  2. Decrease proportional gain:")
    print("     - Reduces responsiveness but also overshoot")
    print("     - Try reducing Kp by 20-30%")
    print("")
    print("  3. Add a zero closer to origin:")
    print("     - Adds phase lead")
    print("     - Lead compensator can help")

    if overshoot > 30:
        print("\nFor very high overshoot (>30%):")
        print("  - Your system is likely significantly underdamped")
        print("  - Consider aggressive derivative action")
        print("  - Or redesign with higher target damping ratio")


def _hint_slow(sys):
    """Provide hints for speeding up response."""
    try:
        info = sys.step_info()
        rise_time = info["rise_time"]
        settling_time = info["settling_time"]
        print(f"\nCurrent rise time: {rise_time:.3f} s")
        print(f"Current settling time: {settling_time:.3f} s")
    except Exception:
        print("\nCould not calculate time metrics")
        return

    print("\nTo speed up response:")
    print("  1. Increase bandwidth:")
    print("     - Increase proportional gain Kp")
    print("     - Add lead compensation")
    print("")
    print("  2. Increase natural frequency:")
    print("     - Higher wn = faster response")
    print("     - But watch for stability!")
    print("")
    print("  3. Careful with integral action:")
    print("     - Too much Ki can slow things down")
    print("     - May need to rebalance gains")

    print("\nRule of thumb:")
    print("  Rise time ~ 1.8 / wn")
    print("  Settling time ~ 4 / (zeta * wn)")


def _hint_sse(sys):
    """Provide hints for reducing steady-state error."""
    try:
        dc_gain = sys.dc_gain
        print(f"\nDC gain: {dc_gain:.4f}")
    except Exception:
        dc_gain = None

    print("\nTo reduce steady-state error:")
    print("  1. Add integral action (Ki):")
    print("     - Integral action forces error to zero")
    print("     - Essential for eliminating SSE to step input")
    print("")
    print("  2. Add lag compensation:")
    print("     - Increases low-frequency gain")
    print("     - Minimal effect on transient if designed properly")
    print("")
    print("  3. Increase forward path gain:")
    print("     - Higher gain = lower SSE")
    print("     - But watch stability margins!")

    print("\nFor different input types:")
    print("  - Step input: Need type-1 system (1 integrator) for zero SSE")
    print("  - Ramp input: Need type-2 system (2 integrators) for zero SSE")


def _general_hints(sys):
    """Provide general analysis and hints."""
    print(f"\nSystem: {sys._name or 'Unnamed'}")
    print(f"Order: {sys.order}")
    print(f"Stable: {sys.is_stable}")

    if not sys.is_stable:
        _hint_unstable(sys)
        return

    try:
        info = sys.step_info()
        print(f"\nStep Response:")
        print(f"  Rise time: {info['rise_time']:.3f} s")
        print(f"  Settling time: {info['settling_time']:.3f} s")
        print(f"  Overshoot: {info['overshoot']:.1f}%")

        issues = []
        if info["overshoot"] > 20:
            issues.append("high overshoot")
        if info["settling_time"] > 5:
            issues.append("slow settling")

        if issues:
            print(f"\nPotential issues: {', '.join(issues)}")
            print("Use get_hint(sys, issue='...') for specific advice")
    except Exception:
        pass

    try:
        freq_info = sys.frequency_info()
        pm = freq_info.get("phase_margin")
        gm = freq_info.get("gain_margin")
        if pm is not None:
            print(f"\nFrequency Response:")
            print(f"  Phase margin: {pm:.1f} deg")
            if pm < 45:
                print("  Warning: Low phase margin - consider adding phase lead")
        if gm is not None:
            print(f"  Gain margin: {gm:.1f} dB")
    except Exception:
        pass


def check_design(sys: "System", specs: Dict[str, float]) -> dict:
    """
    Check if a system meets design specifications.

    Parameters
    ----------
    sys : System
        The system to check
    specs : dict
        Specifications to check:
        - 'overshoot': max percent overshoot
        - 'settling_time': max settling time (s)
        - 'rise_time': max rise time (s)
        - 'phase_margin': min phase margin (deg)
        - 'gain_margin': min gain margin (dB)

    Returns
    -------
    dict
        Results with 'passed', 'failed' lists and details

    Examples
    --------
    >>> specs = {'overshoot': 10, 'settling_time': 2.0, 'phase_margin': 45}
    >>> result = check_design(closed_loop, specs)
    """
    print("\n" + "=" * 50)
    print("Design Specification Check")
    print("=" * 50)

    results = {"passed": [], "failed": [], "details": {}}

    # Get step info if needed
    step_specs = ["overshoot", "settling_time", "rise_time"]
    if any(s in specs for s in step_specs):
        try:
            info = sys.step_info()
        except Exception as e:
            info = {}
            print(f"\nWarning: Could not get step info: {e}")

    # Get frequency info if needed
    freq_specs = ["phase_margin", "gain_margin"]
    if any(s in specs for s in freq_specs):
        try:
            freq_info = sys.frequency_info()
        except Exception:
            freq_info = {}

    # Check each specification
    for spec_name, target in specs.items():
        actual = None
        passed = False

        if spec_name == "overshoot":
            actual = info.get("overshoot")
            if actual is not None:
                passed = actual <= target

        elif spec_name == "settling_time":
            actual = info.get("settling_time")
            if actual is not None:
                passed = actual <= target

        elif spec_name == "rise_time":
            actual = info.get("rise_time")
            if actual is not None:
                passed = actual <= target

        elif spec_name == "phase_margin":
            actual = freq_info.get("phase_margin")
            if actual is not None:
                passed = actual >= target

        elif spec_name == "gain_margin":
            actual = freq_info.get("gain_margin")
            if actual is not None:
                passed = actual >= target

        # Record result
        results["details"][spec_name] = {
            "target": target,
            "actual": actual,
            "passed": passed,
        }

        if passed:
            results["passed"].append(spec_name)
        else:
            results["failed"].append(spec_name)

        # Print result
        status = "PASS" if passed else "FAIL"
        if actual is not None:
            if spec_name in ["overshoot"]:
                print(f"\n{spec_name}: {actual:.1f}%  {status} (spec: <{target}%)")
            elif spec_name in ["settling_time", "rise_time"]:
                print(f"\n{spec_name}: {actual:.3f} s  {status} (spec: <{target} s)")
            elif spec_name == "phase_margin":
                print(f"\n{spec_name}: {actual:.1f} deg  {status} (spec: >{target} deg)")
            elif spec_name == "gain_margin":
                print(f"\n{spec_name}: {actual:.1f} dB  {status} (spec: >{target} dB)")
        else:
            print(f"\n{spec_name}: Could not measure  UNKNOWN")

        # Add suggestion if failed
        if not passed and actual is not None:
            if spec_name == "overshoot":
                print("  Hint: Try increasing Kd or reducing Kp")
            elif spec_name in ["settling_time", "rise_time"]:
                print("  Hint: Try increasing Kp or adding lead compensation")
            elif spec_name == "phase_margin":
                print("  Hint: Add lead compensator for more phase")
            elif spec_name == "gain_margin":
                print("  Hint: Reduce overall gain or add lag compensation")

    # Summary
    n_passed = len(results["passed"])
    n_total = len(specs)
    print(f"\n{'=' * 50}")
    print(f"Overall: {n_passed}/{n_total} specifications met")

    return results
