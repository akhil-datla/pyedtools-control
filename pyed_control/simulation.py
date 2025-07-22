"""Simulation helpers built on python-control."""

from control import feedback, step_response, impulse_response, forced_response


def feedback_loop(plant, controller=1):
    """Return closed-loop transfer function with unity feedback."""
    return feedback(controller*plant, 1)


def step_info(sys):
    """Return basic step response characteristics."""
    import numpy as np
    t, y = step_response(sys)
    steady_state = y[-1]
    overshoot = (y.max() - steady_state) / steady_state * 100 if steady_state else 0
    settling_idx = np.where(abs(y - steady_state) <= 0.02 * abs(steady_state))[0]
    settling_time = t[settling_idx[0]] if len(settling_idx) else t[-1]
    return dict(overshoot=overshoot, settling_time=settling_time)


def impulse(sys):
    """Return time and response of impulse."""
    t, y = impulse_response(sys)
    return t, y


def simulate(sys, T, U):
    """Run forced response simulation.

    Returns ``t, y`` for transfer functions or ``t, y, x`` for state-space
    systems. If ``forced_response`` returns only two values, ``x`` will be
    ``None``.
    """
    result = forced_response(sys, T, U)
    if len(result) == 3:
        t, y, x = result
    else:
        t, y = result
        x = None
    return t, y, x
