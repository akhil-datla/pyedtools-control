"""
Mechanical system models.

These are classic second-order systems commonly used in control education.
"""

from ..core.system import System


def mass_spring_damper(m=1.0, k=1.0, b=0.2, name=None):
    """
    Mass-spring-damper system.

    A classic second-order mechanical system where a mass is connected
    to a fixed wall by a spring and damper in parallel.

    Transfer Function (Force to Position):

                        1
        G(s) = ---------------------
               m*s^2 + b*s + k

    Physical Interpretation:
    - Input: Applied force F (Newtons)
    - Output: Position x (meters)
    - m: Mass (kg) - resistance to acceleration
    - k: Spring constant (N/m) - stiffness
    - b: Damping coefficient (N*s/m) - friction

    System Characteristics:
    - Natural frequency: wn = sqrt(k/m) rad/s
    - Damping ratio: zeta = b / (2*sqrt(k*m))
    - DC gain: 1/k (static deflection per unit force)

    Parameters
    ----------
    m : float
        Mass in kg (default: 1.0)
    k : float
        Spring constant in N/m (default: 1.0)
    b : float
        Damping coefficient in N*s/m (default: 0.2)
    name : str, optional
        System name (default: auto-generated)

    Returns
    -------
    System
        The mass-spring-damper system

    Examples
    --------
    >>> # Default system
    >>> sys = mass_spring_damper()
    >>> sys.analyze()

    >>> # Stiffer spring, more damping
    >>> sys = mass_spring_damper(m=2.0, k=8.0, b=4.0)

    >>> # Undamped system (will oscillate forever)
    >>> sys = mass_spring_damper(b=0)
    >>> sys.step()  # See sustained oscillations

    See Also
    --------
    rotational_system : Rotational analog (inertia-spring-damper)
    quarter_car_suspension : Automotive suspension model
    """
    import numpy as np

    if name is None:
        name = f"Mass-Spring-Damper (m={m}, k={k}, b={b})"

    num = [1.0]
    den = [m, b, k]

    sys = System.from_tf(num, den, name=name)

    # Add physical metadata
    sys._physical_params = {"m": m, "k": k, "b": b}
    sys._input_unit = "N"
    sys._output_unit = "m"
    sys._system_type = "mechanical"

    # Add derived parameters
    wn = np.sqrt(k / m)
    zeta = b / (2 * np.sqrt(k * m))
    sys._physical_params["wn"] = wn
    sys._physical_params["zeta"] = zeta

    return sys


def rotational_system(J=1.0, k=1.0, b=0.2, name=None):
    """
    Rotational mechanical system (inertia-spring-damper).

    The rotational analog of mass-spring-damper.

    Transfer Function (Torque to Angle):

                        1
        G(s) = ---------------------
               J*s^2 + b*s + k

    Physical Interpretation:
    - Input: Applied torque T (N*m)
    - Output: Angular position theta (rad)
    - J: Moment of inertia (kg*m^2)
    - k: Torsional spring constant (N*m/rad)
    - b: Rotational damping coefficient (N*m*s/rad)

    Parameters
    ----------
    J : float
        Moment of inertia in kg*m^2 (default: 1.0)
    k : float
        Torsional spring constant in N*m/rad (default: 1.0)
    b : float
        Damping coefficient in N*m*s/rad (default: 0.2)
    name : str, optional
        System name

    Returns
    -------
    System
        The rotational system
    """
    import numpy as np

    if name is None:
        name = f"Rotational System (J={J}, k={k}, b={b})"

    num = [1.0]
    den = [J, b, k]

    sys = System.from_tf(num, den, name=name)
    sys._physical_params = {"J": J, "k": k, "b": b}
    sys._input_unit = "N*m"
    sys._output_unit = "rad"
    sys._system_type = "mechanical"

    return sys


def quarter_car_suspension(ms=300, mu=50, ks=15000, kt=150000, bs=1000, name=None):
    """
    Quarter-car suspension model.

    Models one corner of a vehicle suspension system.
    This is a 2-DOF system excellent for studying:
    - Trade-offs between ride comfort and road holding
    - Effect of damping on multiple modes
    - Frequency response to road disturbances

    Parameters
    ----------
    ms : float
        Sprung mass (body) in kg (default: 300)
    mu : float
        Unsprung mass (wheel) in kg (default: 50)
    ks : float
        Suspension spring constant in N/m (default: 15000)
    kt : float
        Tire spring constant in N/m (default: 150000)
    bs : float
        Suspension damping in N*s/m (default: 1000)
    name : str, optional
        System name

    Returns
    -------
    System
        State-space model (road input to body displacement)
    """
    import numpy as np

    if name is None:
        name = "Quarter Car Suspension"

    # State: [zs, zs_dot, zu, zu_dot]
    # zs = sprung mass displacement, zu = unsprung mass displacement
    A = [
        [0, 1, 0, 0],
        [-ks / ms, -bs / ms, ks / ms, bs / ms],
        [0, 0, 0, 1],
        [ks / mu, bs / mu, -(ks + kt) / mu, -bs / mu],
    ]

    B = [[0], [0], [0], [kt / mu]]

    # Output: sprung mass displacement
    C = [[1, 0, 0, 0]]

    D = [[0]]

    sys = System.from_ss(A, B, C, D, name=name)
    sys._physical_params = {"ms": ms, "mu": mu, "ks": ks, "kt": kt, "bs": bs}
    sys._input_unit = "m"
    sys._output_unit = "m"
    sys._system_type = "mechanical"

    return sys
