from pyed_control import models, controllers, simulation


def test_feedback_loop_type():
    P = models.mass_spring_damper()
    C = controllers.make_pid(1, 1, 1)
    T = simulation.feedback_loop(P, C)
    assert T.ninputs == 1 and T.noutputs == 1
