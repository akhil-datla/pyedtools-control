"""Controller design helpers."""

from control import TransferFunction, lqr, lqe

def make_pid(kp, ki=0.0, kd=0.0):
    """Return a PID controller as a TransferFunction."""
    num = [kd, kp, ki]
    den = [1, 0]
    return TransferFunction(num, den)


def pid_ziegler_nichols(plant):
    """Rudimentary Ziegler–Nichols PID tuning.

    This implementation finds the ultimate gain Ku and period Tu by applying
    a simple relay feedback approximation. Here we approximate using the
    frequency response where the phase is -180 degrees.
    """
    import numpy as np
    from control import bode

    omega = np.logspace(-2, 4, 1000)
    mag, phase, _ = bode(plant, omega, plot=False)
    # Find frequency where phase crosses -180 degrees
    idx = (phase <= -np.pi + 1e-3).nonzero()[0]
    if len(idx) == 0:
        raise ValueError("Plant does not cross -180 degrees")
    w180 = omega[idx[0]]
    Ku = 1.0 / mag[idx[0]]
    Tu = 2 * np.pi / w180
    kp = 0.6 * Ku
    ki = 1.2 * Ku / Tu
    kd = 3 * Ku * Tu / 40
    return kp, ki, kd

def lead_compensator(z, p, k=1.0):
    """Return a lead compensator ``k*(s+z)/(s+p)``."""
    num = [k, k * z]
    den = [1, p]
    return TransferFunction(num, den)


def lag_compensator(z, p, k=1.0):
    """Return a lag compensator ``k*(s+z)/(s+p)``."""
    num = [k, k * z]
    den = [1, p]
    return TransferFunction(num, den)


def state_feedback_lqr(A, B, Q, R):
    """Compute LQR gain ``K`` using ``control.lqr``."""
    K, S, E = lqr(A, B, Q, R)
    return K, S, E


def observer_gain(A, C, Q, R):
    """Compute observer gain ``L`` using ``control.lqe``."""
    L, P, E = lqe(A, None, C, Q, R)
    return L, P, E
