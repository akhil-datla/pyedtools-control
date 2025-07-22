from pyed_control import models, controllers, simulation


def test_feedback_loop_type():
    P = models.mass_spring_damper()
    C = controllers.make_pid(1, 1, 1)
    T = simulation.feedback_loop(P, C)
    assert T.ninputs == 1 and T.noutputs == 1

def test_impulse_returns_arrays():
    P = models.mass_spring_damper()
    t, y = simulation.impulse(P)
    assert len(t) == len(y)

def test_forced_response_shapes():
    P = models.mass_spring_damper()
    T = [0, 1, 2]
    U = [0, 0, 1]
    t, y, x = simulation.simulate(P, T, U)
    assert len(t) == len(y) == len(U)
    assert x is None

