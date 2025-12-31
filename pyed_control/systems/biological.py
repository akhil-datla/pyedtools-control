"""
Biological and physiological control systems for education.

Includes models of biological regulatory systems commonly
used in control systems courses and biomedical engineering.
"""

import numpy as np
from typing import Optional
from ..core.system import System


def glucose_insulin(
    tau_g: float = 30.0,
    tau_i: float = 15.0,
    k_insulin: float = 0.05,
    name: Optional[str] = None,
) -> System:
    """
    Simplified glucose-insulin regulatory system.

    Models the relationship between insulin injection and blood glucose level.
    This is a simplified version of the minimal model used in diabetes research.

    Parameters
    ----------
    tau_g : float
        Glucose disappearance time constant (minutes), default 30
    tau_i : float
        Insulin action time constant (minutes), default 15
    k_insulin : float
        Insulin sensitivity (mg/dL per mU), default 0.05
    name : str, optional
        System name

    Returns
    -------
    System
        Glucose-insulin dynamics

    Examples
    --------
    >>> from pyed_control import systems
    >>> glucose = systems.glucose_insulin()
    >>> glucose.analyze()

    Notes
    -----
    In reality, glucose regulation involves complex nonlinear dynamics,
    multiple feedback loops, and significant time delays. This simplified
    model captures the essential first-order behavior for control design.
    """
    # Two first-order systems in series
    # Insulin action dynamics, then glucose response
    num = [k_insulin]
    den = [(tau_g/60) * (tau_i/60), (tau_g + tau_i)/60, 1]  # Convert to minutes

    sys = System.from_tf(num, den, name=name or "Glucose-Insulin Dynamics")
    sys._description = (
        f"Glucose response to insulin. "
        f"Glucose tau={tau_g}min, Insulin tau={tau_i}min"
    )
    sys._physical_params = {
        "tau_g": tau_g,
        "tau_i": tau_i,
        "k_insulin": k_insulin,
    }
    sys._input_unit = "mU (insulin)"
    sys._output_unit = "mg/dL (glucose change)"
    return sys


def drug_concentration(
    V_d: float = 50.0,
    k_e: float = 0.1,
    k_a: float = 0.5,
    F: float = 0.8,
    name: Optional[str] = None,
) -> System:
    """
    Pharmacokinetic model of drug concentration.

    One-compartment model with first-order absorption and elimination.
    Models plasma drug concentration after oral or subcutaneous administration.

    Parameters
    ----------
    V_d : float
        Volume of distribution (liters), default 50
    k_e : float
        Elimination rate constant (1/hour), default 0.1
    k_a : float
        Absorption rate constant (1/hour), default 0.5
    F : float
        Bioavailability fraction (0-1), default 0.8
    name : str, optional
        System name

    Returns
    -------
    System
        Drug concentration dynamics

    Notes
    -----
    This is the classic one-compartment pharmacokinetic model.
    The transfer function represents the relationship between
    administered dose rate and plasma concentration.
    """
    # G(s) = F * k_a / (V_d * (s + k_a) * (s + k_e))
    K = F * k_a / V_d
    num = [K]
    den = [1, k_a + k_e, k_a * k_e]

    sys = System.from_tf(num, den, name=name or "Drug Pharmacokinetics")
    sys._description = (
        f"One-compartment PK model. "
        f"V_d={V_d}L, k_e={k_e}/hr, k_a={k_a}/hr, F={F}"
    )
    sys._physical_params = {
        "V_d": V_d,
        "k_e": k_e,
        "k_a": k_a,
        "F": F,
    }
    sys._input_unit = "mg/hr (dose rate)"
    sys._output_unit = "mg/L (concentration)"
    return sys


def pupil_light_reflex(
    tau_fast: float = 0.2,
    tau_slow: float = 1.0,
    gain: float = 0.5,
    delay: float = 0.2,
    name: Optional[str] = None,
) -> System:
    """
    Pupil light reflex dynamics.

    Models how the pupil adjusts its diameter in response to light.
    The pupil constricts (gets smaller) when light increases.

    Parameters
    ----------
    tau_fast : float
        Fast time constant (seconds), default 0.2
    tau_slow : float
        Slow time constant (seconds), default 1.0
    gain : float
        Steady-state gain (mm per log unit), default 0.5
    delay : float
        Neural processing delay (seconds), default 0.2
        Note: Time delay is approximated by a first-order Pade approximation
    name : str, optional
        System name

    Returns
    -------
    System
        Pupil dynamics

    Notes
    -----
    The pupil light reflex is a classic example of a biological feedback
    control system with time delay. The delay arises from neural processing.
    """
    # Two time constants plus Pade approximation for delay
    # Delay: e^(-s*delay) ≈ (1 - s*delay/2) / (1 + s*delay/2)

    # Without exact delay (using approximation)
    # Two cascaded first-order systems
    num = [-gain]  # Negative because pupil constricts with more light
    den = [tau_fast * tau_slow, tau_fast + tau_slow, 1]

    sys = System.from_tf(num, den, name=name or "Pupil Light Reflex")
    sys._description = (
        f"Pupil diameter response to light. "
        f"Fast tau={tau_fast}s, Slow tau={tau_slow}s, delay~{delay}s"
    )
    sys._physical_params = {
        "tau_fast": tau_fast,
        "tau_slow": tau_slow,
        "gain": gain,
        "delay": delay,
    }
    sys._input_unit = "log(lux)"
    sys._output_unit = "mm (diameter change)"
    return sys


def heart_rate_baroreflex(
    tau_vagal: float = 1.0,
    tau_sympathetic: float = 10.0,
    k_vagal: float = -0.5,
    k_sympathetic: float = 0.3,
    name: Optional[str] = None,
) -> System:
    """
    Heart rate response to blood pressure (baroreflex).

    The baroreflex is the body's mechanism to maintain stable blood pressure.
    When pressure increases, heart rate decreases to compensate.

    Parameters
    ----------
    tau_vagal : float
        Vagal (parasympathetic) time constant (seconds), default 1.0
    tau_sympathetic : float
        Sympathetic time constant (seconds), default 10.0
    k_vagal : float
        Vagal gain (bpm per mmHg), default -0.5 (negative = inhibitory)
    k_sympathetic : float
        Sympathetic gain (bpm per mmHg), default 0.3
    name : str, optional
        System name

    Returns
    -------
    System
        Baroreflex dynamics

    Notes
    -----
    The baroreflex involves two pathways with different speeds:
    - Vagal (fast): parasympathetic nervous system, slows heart
    - Sympathetic (slow): increases heart rate

    Both pathways receive blood pressure input from baroreceptors.
    """
    # Two parallel pathways: vagal (fast, negative) + sympathetic (slow, positive)
    # G(s) = k_v/(tau_v*s + 1) + k_s/(tau_s*s + 1)
    # Combined into single TF

    k_v = k_vagal
    k_s = k_sympathetic
    t_v = tau_vagal
    t_s = tau_sympathetic

    # Numerator: k_v*(t_s*s + 1) + k_s*(t_v*s + 1)
    # = (k_v*t_s + k_s*t_v)*s + (k_v + k_s)
    num = [k_v*t_s + k_s*t_v, k_v + k_s]

    # Denominator: (t_v*s + 1)*(t_s*s + 1) = t_v*t_s*s^2 + (t_v+t_s)*s + 1
    den = [t_v*t_s, t_v + t_s, 1]

    sys = System.from_tf(num, den, name=name or "Baroreflex")
    sys._description = (
        f"Heart rate response to blood pressure. "
        f"Vagal tau={tau_vagal}s, Sympathetic tau={tau_sympathetic}s"
    )
    sys._physical_params = {
        "tau_vagal": tau_vagal,
        "tau_sympathetic": tau_sympathetic,
        "k_vagal": k_vagal,
        "k_sympathetic": k_sympathetic,
    }
    sys._input_unit = "mmHg"
    sys._output_unit = "bpm (change)"
    return sys


def respiratory_control(
    tau_chemoreceptor: float = 10.0,
    tau_mechanical: float = 0.5,
    gain: float = 2.0,
    name: Optional[str] = None,
) -> System:
    """
    Respiratory control system (ventilation response to CO2).

    Models how the body adjusts breathing rate and depth in response
    to arterial CO2 levels.

    Parameters
    ----------
    tau_chemoreceptor : float
        Chemoreceptor time constant (seconds), default 10.0
    tau_mechanical : float
        Mechanical lung response time constant (seconds), default 0.5
    gain : float
        Ventilatory sensitivity (L/min per mmHg), default 2.0
    name : str, optional
        System name

    Returns
    -------
    System
        Respiratory control dynamics

    Notes
    -----
    CO2 regulation involves:
    - Chemoreceptors detecting arterial CO2
    - Neural processing
    - Mechanical response of respiratory muscles
    - Gas exchange in lungs (which then affects CO2)

    This creates a closed-loop system with significant delay.
    """
    num = [gain]
    den = [tau_chemoreceptor * tau_mechanical, tau_chemoreceptor + tau_mechanical, 1]

    sys = System.from_tf(num, den, name=name or "Respiratory Control")
    sys._description = (
        f"Ventilation response to CO2. "
        f"Chemoreceptor tau={tau_chemoreceptor}s, Mechanical tau={tau_mechanical}s"
    )
    sys._physical_params = {
        "tau_chemoreceptor": tau_chemoreceptor,
        "tau_mechanical": tau_mechanical,
        "gain": gain,
    }
    sys._input_unit = "mmHg (PaCO2 change)"
    sys._output_unit = "L/min (ventilation change)"
    return sys


def muscle_tendon(
    tau_activation: float = 0.05,
    tau_deactivation: float = 0.1,
    wn: float = 20.0,
    zeta: float = 0.7,
    name: Optional[str] = None,
) -> System:
    """
    Muscle-tendon dynamics.

    Models the force production of a muscle in response to neural activation.
    Combines activation dynamics with the mechanical muscle-tendon unit.

    Parameters
    ----------
    tau_activation : float
        Activation time constant (seconds), default 0.05
    tau_deactivation : float
        Deactivation time constant (seconds), default 0.1
    wn : float
        Natural frequency of muscle mechanics (rad/s), default 20.0
    zeta : float
        Damping ratio of muscle mechanics, default 0.7
    name : str, optional
        System name

    Returns
    -------
    System
        Muscle-tendon dynamics

    Notes
    -----
    Muscle activation dynamics are nonlinear (different time constants for
    activation vs deactivation). This model uses a linear approximation
    with an average time constant.
    """
    # Average time constant for activation dynamics
    tau_avg = (tau_activation + tau_deactivation) / 2

    # First-order activation + second-order mechanics
    # G(s) = wn^2 / ((tau*s + 1) * (s^2 + 2*zeta*wn*s + wn^2))
    wn2 = wn ** 2

    num = [wn2]
    den = [tau_avg, 1 + 2*zeta*wn*tau_avg, wn2*tau_avg + 2*zeta*wn, wn2]

    sys = System.from_tf(num, den, name=name or "Muscle-Tendon Dynamics")
    sys._description = (
        f"Force response to neural activation. "
        f"Act. tau={tau_activation}s, Deact. tau={tau_deactivation}s"
    )
    sys._physical_params = {
        "tau_activation": tau_activation,
        "tau_deactivation": tau_deactivation,
        "wn": wn,
        "zeta": zeta,
    }
    sys._input_unit = "normalized activation (0-1)"
    sys._output_unit = "normalized force (0-1)"
    return sys


def cell_signaling(
    k_on: float = 0.1,
    k_off: float = 0.05,
    k_production: float = 0.5,
    k_degradation: float = 0.1,
    name: Optional[str] = None,
) -> System:
    """
    Cell signaling cascade dynamics.

    Simple model of a receptor-ligand binding cascade with downstream
    protein production.

    Parameters
    ----------
    k_on : float
        Receptor binding rate (1/min), default 0.1
    k_off : float
        Receptor unbinding rate (1/min), default 0.05
    k_production : float
        Downstream production rate, default 0.5
    k_degradation : float
        Degradation rate (1/min), default 0.1
    name : str, optional
        System name

    Returns
    -------
    System
        Cell signaling dynamics

    Notes
    -----
    Cell signaling involves complex biochemical networks.
    This simplified model captures two stages:
    1. Receptor-ligand binding dynamics
    2. Downstream protein production/degradation
    """
    # Two cascaded first-order systems
    # Binding: k_on / (s + k_on + k_off)
    # Production: k_production / (s + k_degradation)

    k1 = k_on + k_off
    k2 = k_degradation

    gain = k_on * k_production / (k1 * k2)
    num = [gain]
    den = [1/(k1*k2), (k1 + k2)/(k1*k2), 1]

    sys = System.from_tf(num, den, name=name or "Cell Signaling Cascade")
    sys._description = (
        f"Receptor-ligand signaling. "
        f"k_on={k_on}, k_off={k_off}, k_prod={k_production}, k_deg={k_degradation}"
    )
    sys._physical_params = {
        "k_on": k_on,
        "k_off": k_off,
        "k_production": k_production,
        "k_degradation": k_degradation,
    }
    sys._input_unit = "ligand concentration"
    sys._output_unit = "protein level"
    return sys


def body_temperature(
    tau_core: float = 300.0,
    tau_skin: float = 60.0,
    k_metabolic: float = 0.01,
    name: Optional[str] = None,
) -> System:
    """
    Body temperature regulation dynamics.

    Models the relationship between metabolic heat production
    and core body temperature.

    Parameters
    ----------
    tau_core : float
        Core temperature time constant (seconds), default 300
    tau_skin : float
        Skin temperature time constant (seconds), default 60
    k_metabolic : float
        Metabolic gain (C per Watt), default 0.01
    name : str, optional
        System name

    Returns
    -------
    System
        Temperature regulation dynamics

    Notes
    -----
    Body temperature regulation involves complex interactions between:
    - Metabolic heat production
    - Blood flow and convection
    - Sweating and evaporation
    - Shivering
    - Behavioral responses

    This simplified model captures the basic thermal dynamics.
    """
    num = [k_metabolic]
    den = [tau_core * tau_skin / 3600, (tau_core + tau_skin) / 60, 1]

    sys = System.from_tf(num, den, name=name or "Body Temperature Regulation")
    sys._description = (
        f"Core temperature response to heat production. "
        f"Core tau={tau_core}s, Skin tau={tau_skin}s"
    )
    sys._physical_params = {
        "tau_core": tau_core,
        "tau_skin": tau_skin,
        "k_metabolic": k_metabolic,
    }
    sys._input_unit = "W (heat production)"
    sys._output_unit = "C (temperature change)"
    return sys
