"""
Concept explanations for control systems education.
"""

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.system import System

EXPLANATIONS = {
    "poles": """
Poles
=====

Poles are the roots of the denominator of a transfer function.
They determine the fundamental dynamic behavior of the system.

For a transfer function G(s) = N(s) / D(s):
- Poles are the values of s where D(s) = 0
- Each pole contributes a mode to the response

Pole Location and Behavior:
- Real negative pole (s = -a): Exponential decay, time constant = 1/a
- Real positive pole (s = +a): Exponential growth (UNSTABLE)
- Complex poles (s = -a +/- bj): Damped oscillation
- Imaginary poles (s = +/- bj): Sustained oscillation

Stability:
- All poles in Left Half Plane (LHP): System is stable
- Any pole in Right Half Plane (RHP): System is unstable
- Poles on imaginary axis: Marginally stable

Example:
  G(s) = 1 / (s^2 + 3s + 2) = 1 / ((s+1)(s+2))
  Poles at s = -1 and s = -2 (both stable)
""",
    "zeros": """
Zeros
=====

Zeros are the roots of the numerator of a transfer function.
They affect the shape of the response but not stability.

For a transfer function G(s) = N(s) / D(s):
- Zeros are the values of s where N(s) = 0
- At a zero frequency, the output is zero for sinusoidal input

Effects of Zeros:
- Real negative zero: Speeds up response, may increase overshoot
- Real positive zero (RHP): Non-minimum phase, causes initial undershoot
- Complex zeros: Affect frequency response shape

Note:
- Zeros do NOT affect stability (only poles do)
- But zeros affect how the system responds to inputs
""",
    "stability": """
Stability
=========

A system is stable if bounded inputs produce bounded outputs (BIBO stability).

For linear time-invariant (LTI) systems:
- STABLE: All poles have negative real parts (all in LHP)
- UNSTABLE: Any pole has positive real part (any in RHP)
- MARGINALLY STABLE: Poles on imaginary axis, none in RHP

Testing Stability:
1. Pole Location: Find poles, check if Re(pole) < 0 for all
2. Routh-Hurwitz: Algebraic test on characteristic polynomial
3. Nyquist Criterion: Graphical test using frequency response
4. Root Locus: See how poles move with gain

Physical Interpretation:
- Stable: Response decays to zero or constant value
- Unstable: Response grows unboundedly
- Marginally stable: Sustained oscillation (constant amplitude)
""",
    "damping_ratio": """
Damping Ratio (zeta)
====================

The damping ratio determines how oscillatory a second-order system's response is.

For a standard second-order system:
                  wn^2
    G(s) = ---------------------------
           s^2 + 2*zeta*wn*s + wn^2

Values and Behavior:
- zeta = 0: Undamped (sustained oscillation, never settles)
- 0 < zeta < 1: Underdamped (decaying oscillation)
- zeta = 1: Critically damped (fastest non-oscillatory response)
- zeta > 1: Overdamped (slow, no oscillation)

Key Relationships:
- Overshoot = exp(-zeta*pi/sqrt(1-zeta^2)) * 100%
- Settling time (2%) = 4 / (zeta*wn)
- Peak time = pi / (wn*sqrt(1-zeta^2))

Typical Design Values:
- 0.4 to 0.8 for good balance of speed and overshoot
- zeta = 0.707 gives maximum bandwidth without peaking
""",
    "natural_frequency": """
Natural Frequency (wn)
======================

The natural frequency is the frequency at which an undamped
second-order system would oscillate.

For a standard second-order system:
                  wn^2
    G(s) = ---------------------------
           s^2 + 2*zeta*wn*s + wn^2

Key Points:
- wn determines the speed of response
- Higher wn = faster response (but may need more control effort)
- Damped frequency: wd = wn * sqrt(1 - zeta^2)

Relationships:
- For complex poles at s = -a +/- bj:
  wn = sqrt(a^2 + b^2)
  zeta = a / wn

- Rise time is approximately: tr = 1.8 / wn
""",
    "phase_margin": """
Phase Margin
============

Phase margin is the additional phase lag at the gain crossover
frequency that would make the system marginally stable.

Definition:
- Find omega_c where |G(jw_c)| = 1 (0 dB)
- Phase margin = 180 + angle(G(jw_c))

Interpretation:
- Larger PM = more robust stability
- PM = 0: Marginally stable (sustained oscillation)
- PM < 0: Unstable

Design Guidelines:
- PM > 45 deg: Good stability margin
- PM > 60 deg: Excellent (may be sluggish)
- PM = 30-45 deg: Acceptable but less robust

Relationship to Time Domain:
- PM roughly corresponds to damping ratio
- PM = 60 deg ~ zeta = 0.6
- PM = 45 deg ~ zeta = 0.45
""",
    "gain_margin": """
Gain Margin
===========

Gain margin is the factor by which the gain can be increased
before the system becomes unstable.

Definition:
- Find omega_p where angle(G(jw_p)) = -180 deg
- Gain margin (dB) = -20*log10(|G(jw_p)|)
- Or: Gain margin (ratio) = 1 / |G(jw_p)|

Interpretation:
- GM > 1 (positive dB): Stable
- GM = 1 (0 dB): Marginally stable
- GM < 1 (negative dB): Unstable

Design Guidelines:
- GM > 6 dB: Good stability margin
- GM > 10 dB: Excellent robustness
- GM = 3-6 dB: Acceptable but tight

Note:
- Some systems have no phase crossover (no finite GM)
- This typically indicates very robust stability
""",
    "pid": """
PID Controller
==============

The PID controller is the most widely used controller in industry.

Transfer Function:
    C(s) = Kp + Ki/s + Kd*s

Terms:
- Proportional (Kp): Acts on current error
  - Larger Kp = faster response, more overshoot
  - Too large = instability

- Integral (Ki): Acts on accumulated error
  - Eliminates steady-state error
  - Larger Ki = faster error elimination, more overshoot
  - Can cause wind-up with saturation

- Derivative (Kd): Acts on rate of change
  - Provides damping/prediction
  - Reduces overshoot
  - Sensitive to noise

Tuning Methods:
1. Ziegler-Nichols: Classic, often aggressive
2. Cohen-Coon: Good for processes with dead time
3. IMC (Internal Model Control): Smooth, less aggressive

Practical Notes:
- Start with P only, then add I, then D
- Use filtered derivative in practice (noise!)
- Consider anti-windup for integral term
""",
    "bode": """
Bode Plot
=========

A Bode plot shows the frequency response of a system as two graphs:
1. Magnitude (in dB) vs. frequency (log scale)
2. Phase (in degrees) vs. frequency (log scale)

Why Bode Plots?
- Visualize how the system responds to different frequency inputs
- Identify stability margins (gain and phase margins)
- Design controllers for desired frequency response

Reading a Bode Plot:
- Magnitude shows amplification/attenuation at each frequency
- Phase shows the delay introduced at each frequency
- Together they tell you how sinusoids are transformed

Key Features:
- DC gain: Magnitude at very low frequency
- Bandwidth: Frequency where magnitude drops by 3 dB
- Crossover frequencies: Where magnitude = 0 dB or phase = -180 deg
- Roll-off rate: How fast magnitude decreases at high frequencies

Stability from Bode:
- At gain crossover (|G| = 1), check phase margin
- At phase crossover (phase = -180), check gain margin
- Both margins should be positive for stability
""",
    "nyquist": """
Nyquist Plot & Criterion
========================

The Nyquist plot shows G(jw) as a curve in the complex plane
as w varies from -infinity to +infinity.

The Nyquist Stability Criterion:
For stability of the closed-loop system 1/(1+G(s)):
  Z = N + P

Where:
- Z = number of unstable closed-loop poles (want Z = 0)
- P = number of unstable open-loop poles (known)
- N = number of clockwise encirclements of -1

Simple Case (P = 0):
If the open-loop is stable, the closed-loop is stable if and
only if the Nyquist plot does NOT encircle -1.

Why Use Nyquist?
- Works for unstable open-loop systems
- Works for systems with time delay
- Gives insight into robustness (distance to -1)
- Can handle non-minimum phase systems

Distance to -1 Point:
- The closest approach to -1 determines robustness
- This distance relates to sensitivity peak
""",
    "root_locus": """
Root Locus
==========

Root locus shows how closed-loop poles move as gain K varies from 0 to infinity.

For closed-loop: 1 + K*G(s) = 0
Poles move from open-loop poles (K=0) toward zeros (K=infinity).

Key Rules:
1. Root locus starts at open-loop poles
2. Root locus ends at zeros (or at infinity)
3. Number of branches = number of poles
4. Locus is symmetric about real axis
5. Real axis segments have odd number of poles+zeros to the right

Using Root Locus:
- Find gain K for desired pole locations
- See when system becomes unstable (crosses imaginary axis)
- Understand trade-off between speed and stability

Design with Root Locus:
- Add poles/zeros to reshape the locus
- Lead compensator: Pulls locus left (more stable)
- Lag compensator: Doesn't change locus shape, changes K values
""",
    "transfer_function": """
Transfer Function
=================

The transfer function G(s) relates the Laplace transform of the
output to the input for a linear time-invariant (LTI) system.

    G(s) = Y(s) / U(s) = N(s) / D(s)

Where:
- s is the complex frequency variable
- N(s) is the numerator polynomial
- D(s) is the denominator polynomial (characteristic polynomial)

Key Properties:
- Poles: Roots of D(s) - determine stability and natural response
- Zeros: Roots of N(s) - affect response shape
- Order: Degree of D(s) - system complexity
- DC Gain: G(0) - steady-state response to step

Standard Forms:
- First order: G(s) = K / (tau*s + 1)
- Second order: G(s) = K*wn^2 / (s^2 + 2*zeta*wn*s + wn^2)

From Transfer Function:
- Impulse response: L^{-1}{G(s)}
- Step response: L^{-1}{G(s)/s}
- Frequency response: G(jw) (substitute s = jw)
""",
    "feedback": """
Feedback Control
================

Feedback is the fundamental principle of control systems.
It uses measurement of output to adjust the input.

Basic Feedback Loop:
        +
  R --> O --> C(s) --> G(s) --> Y
        ^                  |
        |                  |
        +------ H(s) <-----+

Transfer Functions:
- Open-loop: L(s) = C(s)*G(s)*H(s)
- Closed-loop: T(s) = C(s)*G(s) / (1 + L(s))
- Sensitivity: S(s) = 1 / (1 + L(s))
- Note: S(s) + T(s) = 1

Benefits of Feedback:
1. Reduces sensitivity to plant variations
2. Rejects disturbances
3. Can stabilize unstable plants
4. Improves transient response

The Fundamental Trade-off:
- Cannot have |S| small AND |T| small at same frequency
- Low |S| at low freq: Good disturbance rejection
- Low |T| at high freq: Good noise rejection
""",
    "controllability": """
Controllability
===============

A system is controllable if we can drive it from any initial state
to any final state in finite time using appropriate inputs.

Test for Controllability:
For x_dot = Ax + Bu, the controllability matrix is:
  C = [B | AB | A^2*B | ... | A^(n-1)*B]

The system is controllable if and only if rank(C) = n

Physical Meaning:
- All state variables can be influenced by the input
- State feedback can place poles anywhere
- Can achieve arbitrary control objectives

Examples of Non-Controllability:
- Decoupled states with no input connection
- Redundant states
- Physical constraints (e.g., can only push, not pull)

Why It Matters:
- If not controllable, some states can't be changed
- Pole placement by state feedback may be limited
- Need to check before designing control law
""",
    "observability": """
Observability
=============

A system is observable if we can determine all state variables
from the output history in finite time.

Test for Observability:
For x_dot = Ax + Bu, y = Cx, the observability matrix is:
  O = [C; CA; CA^2; ...; CA^(n-1)]

The system is observable if and only if rank(O) = n

Physical Meaning:
- All states affect the output (directly or indirectly)
- Can reconstruct states from measurements
- Observer/estimator design is possible

Examples of Non-Observability:
- Internal states that don't affect output
- Hidden dynamics
- Sensor placement issues

Why It Matters:
- If not observable, can't estimate some states
- Observer design may not work for all states
- Need to check before designing state estimator
""",
    "sensitivity": """
Sensitivity & Robustness
========================

Sensitivity tells us how much the closed-loop changes when
the plant changes.

Sensitivity Function: S(s) = 1 / (1 + L(s))
- Transfer from disturbance at output to output
- Also: relative change in closed-loop / relative change in plant

Complementary Sensitivity: T(s) = L(s) / (1 + L(s))
- Closed-loop transfer function
- Also: effect of measurement noise on output

The Fundamental Constraint: S(s) + T(s) = 1
- Cannot make both small at any frequency
- Design is about choosing the trade-off

Robustness Measures:
- Peak of |S|: Want Ms < 2 (6 dB)
- Peak of |T|: Want Mt < 1.5
- Gain margin: Factor of gain increase to instability
- Phase margin: Additional phase lag to instability
""",
    "discretization": """
Discretization (Continuous to Discrete)
=======================================

Digital controllers operate on sampled signals, requiring
conversion from continuous to discrete time.

Common Methods:
1. Zero-Order Hold (ZOH)
   - Input held constant between samples
   - Most common for real implementation
   - Exact for step inputs

2. Tustin (Bilinear Transform)
   - s = (2/T)(z-1)/(z+1)
   - Preserves stability
   - Good frequency matching at low frequencies
   - Causes frequency warping

3. Matched Pole-Zero
   - z = exp(s*T)
   - Preserves pole/zero locations
   - Good for systems with distinct poles

Key Considerations:
- Sample time T must be small enough (Nyquist: fs > 2*bandwidth)
- Rule of thumb: 10-20 samples per rise time
- Aliasing if sample rate too low
""",
    "steady_state_error": """
Steady-State Error
==================

Steady-state error is the difference between desired and actual
output after transients have died out.

For unity feedback with error E = R - Y:
  e_ss = lim(s->0) s * E(s)

System Type (number of integrators in loop):
- Type 0: Finite error to step, infinite to ramp
- Type 1: Zero error to step, finite to ramp
- Type 2: Zero error to step and ramp, finite to parabola

Error Constants:
- Position constant Kp = lim(s->0) G(s)
- Velocity constant Kv = lim(s->0) s*G(s)
- Acceleration constant Ka = lim(s->0) s^2*G(s)

Steady-State Errors:
- Step: e_ss = 1 / (1 + Kp)
- Ramp: e_ss = 1 / Kv
- Parabola: e_ss = 1 / Ka

Reducing Steady-State Error:
- Add integrator (increases type)
- Increase gain (but may reduce stability)
- Use integral action in controller
""",
    "lead_lag": """
Lead and Lag Compensators
=========================

Lead and lag compensators are simple but powerful controllers.

Lead Compensator: C(s) = K * (s + z) / (s + p), where z < p
- Adds phase at frequencies between z and p
- Speeds up response (increases bandwidth)
- Improves stability margins
- Like adding damping

Lag Compensator: C(s) = K * (s + z) / (s + p), where z > p
- Reduces phase at frequencies between p and z
- Reduces bandwidth
- Improves steady-state accuracy (adds gain at low freq)
- Does not significantly change stability

Lead-Lag Design:
- Lead: Use when you need more phase margin
- Lag: Use when you need less steady-state error
- Can combine both for overall improvement

Design Guidelines:
- Lead center frequency at crossover
- Lead phase boost = atan((p-z)/(2*sqrt(p*z)))
- Lag ratio determines steady-state improvement
""",
}

CONCEPTS = {
    "poles": "Poles: Roots of denominator, determine stability and natural response.",
    "zeros": "Zeros: Roots of numerator, affect response shape but not stability.",
    "stability": "Stability: System is stable if all poles have negative real parts.",
    "damping_ratio": "Damping ratio (zeta): 0=oscillatory, 1=critical, >1=overdamped.",
    "natural_frequency": "Natural frequency (wn): Speed of response in rad/s.",
    "phase_margin": "Phase margin: Extra phase lag needed to destabilize. Target: >45 deg.",
    "gain_margin": "Gain margin: Factor by which gain can increase before instability. Target: >6 dB.",
    "pid": "PID: Proportional-Integral-Derivative controller. Most common in industry.",
    "overshoot": "Overshoot: How much response exceeds final value (%). Lower zeta = higher overshoot.",
    "settling_time": "Settling time: Time to stay within 2% of final value. Approx: 4/(zeta*wn).",
    "rise_time": "Rise time: Time from 10% to 90% of final value. Approx: 1.8/wn.",
    "bandwidth": "Bandwidth: Frequency range where gain > -3dB. Higher = faster response.",
    "bode": "Bode plot: Magnitude and phase vs frequency on log scale.",
    "nyquist": "Nyquist: Complex plane plot of G(jw). Encirclements of -1 determine stability.",
    "root_locus": "Root locus: How poles move as gain varies. Starts at poles, ends at zeros.",
    "transfer_function": "Transfer function: G(s) = Y(s)/U(s), relates Laplace of output to input.",
    "feedback": "Feedback: Using output measurement to adjust input. Fundamental control principle.",
    "controllability": "Controllability: Can we drive system to any state? Rank of [B AB A^2B...] = n.",
    "observability": "Observability: Can we determine all states from output? Rank of [C CA CA^2...]^T = n.",
    "sensitivity": "Sensitivity S(s): How disturbances affect output. S + T = 1 is fundamental.",
    "discretization": "Discretization: Converting continuous to discrete-time for digital control.",
    "steady_state_error": "Steady-state error: Error after transients die. Type = number of integrators.",
    "lead_lag": "Lead/Lag: Lead adds phase (improves PM), Lag adds low-freq gain (reduces error).",
    "time_constant": "Time constant (tau): Time to reach 63.2% of final value for first-order system.",
    "crossover": "Crossover frequency: Where |G(jw)| = 1. Determines closed-loop bandwidth.",
}


def explain(topic: str, context: Optional["System"] = None):
    """
    Get an explanation of a control systems concept.

    Parameters
    ----------
    topic : str
        Concept to explain. Options include:
        'poles', 'zeros', 'stability', 'damping_ratio',
        'natural_frequency', 'phase_margin', 'gain_margin', 'pid'
    context : System, optional
        If provided, explain in context of this specific system

    Examples
    --------
    >>> explain('damping_ratio')
    >>> explain('stability', context=my_system)
    """
    topic_lower = topic.lower().replace(" ", "_").replace("-", "_")

    if topic_lower not in EXPLANATIONS:
        available = ", ".join(sorted(EXPLANATIONS.keys()))
        print(f"Topic '{topic}' not found.")
        print(f"Available topics: {available}")
        return

    print(EXPLANATIONS[topic_lower])

    if context is not None:
        print("\n" + "=" * 50)
        print(f"For your system: {context._name or 'Unnamed'}")
        print("=" * 50)

        if topic_lower in ["poles", "stability"]:
            print(f"\nPoles: {context.poles}")
            print(f"Stable: {context.is_stable}")

        if topic_lower in ["zeros"]:
            print(f"\nZeros: {context.zeros}")

        if topic_lower == "damping_ratio":
            if context.order == 2:
                params = context._physical_params
                if "zeta" in params:
                    zeta = params["zeta"]
                    print(f"\nDamping ratio: zeta = {zeta:.4f}")
                    if zeta < 1:
                        print("Your system is UNDERDAMPED (will oscillate)")
                    elif zeta == 1:
                        print("Your system is CRITICALLY DAMPED")
                    else:
                        print("Your system is OVERDAMPED")


def concept(name: str) -> str:
    """
    Get a one-line definition of a concept.

    Parameters
    ----------
    name : str
        Concept name

    Returns
    -------
    str
        One-line definition

    Examples
    --------
    >>> print(concept('phase_margin'))
    """
    name_lower = name.lower().replace(" ", "_").replace("-", "_")

    if name_lower in CONCEPTS:
        definition = CONCEPTS[name_lower]
        print(definition)
        return definition
    else:
        available = ", ".join(sorted(CONCEPTS.keys()))
        msg = f"Concept '{name}' not found. Available: {available}"
        print(msg)
        return msg
