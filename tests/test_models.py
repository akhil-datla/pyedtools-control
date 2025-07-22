import numpy as np
from pyed_control import models


def test_mass_spring_damper_tf():
    sys = models.mass_spring_damper(m=2.0, k=4.0, b=1.0)
    assert np.allclose(sys.den[0][0], [2.0, 1.0, 4.0])


def test_inverted_pendulum_shape():
    sys = models.inverted_pendulum()
    assert sys.A.shape == (4, 4) and sys.B.shape == (4, 1)
