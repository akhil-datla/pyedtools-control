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
