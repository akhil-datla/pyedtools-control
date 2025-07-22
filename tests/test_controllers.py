import numpy as np
from pyed_control import controllers, models


def test_make_pid_tf():
    C = controllers.make_pid(1.0, 0.5, 0.1)
    assert np.allclose(C.num[0][0], [0.1, 1.0, 0.5])
    assert np.allclose(C.den[0][0], [1.0, 0.0])


def test_pid_ziegler_nichols_returns_tuple():
    P = models.mass_spring_damper()
    kp, ki, kd = controllers.pid_ziegler_nichols(P)
    assert kp > 0 and ki > 0 and kd > 0


def test_lead_compensator_tf():
    C = controllers.lead_compensator(1, 10, k=2)
    assert np.allclose(C.num[0][0], [2, 2])
    assert np.allclose(C.den[0][0], [1, 10])


def test_lqr_gain_shape():
    sys = models.inverted_pendulum()
    Q = np.eye(4)
    R = np.array([[1]])
    K, S, E = controllers.state_feedback_lqr(sys.A, sys.B, Q, R)
    assert K.shape == (1, 4)
