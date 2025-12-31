"""
Lead and Lag compensators.

These are classic frequency-domain compensation techniques
for improving system performance.
"""

import numpy as np
import control as ctrl
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.system import System


class Lead:
    """
    Lead compensator for improving transient response.

    Transfer Function:
                    s + z
        C(s) = Kc * -----
                    s + p

    where |z| < |p| (zero closer to origin than pole)

    Effects:
    - Increases phase margin (adds phase lead)
    - Speeds up response (higher bandwidth)
    - Increases high-frequency gain

    Examples
    --------
    >>> # Design by specifying zero and pole
    >>> lead = Lead(zero=1, pole=10)

    >>> # Design for phase boost at specific frequency
    >>> lead = Lead.for_phase_boost(45, at_frequency=5)

    >>> # Apply to plant
    >>> compensated = plant.series(lead.as_system()).feedback()
    """

    def __init__(self, zero: float, pole: float, gain: float = 1.0):
        """
        Create a lead compensator.

        Parameters
        ----------
        zero : float
            Zero location (positive value = zero at -zero)
        pole : float
            Pole location (positive value = pole at -pole)
        gain : float
            Overall gain (default: 1.0)
        """
        self.zero = abs(zero)
        self.pole = abs(pole)
        self.gain = gain

        if self.zero >= self.pole:
            raise ValueError(
                f"For a lead compensator, |zero| must be less than |pole|.\n"
                f"You provided zero={self.zero}, pole={self.pole}.\n"
                f"This would create a lag compensator instead."
            )

    @classmethod
    def for_phase_boost(
        cls,
        phase_boost: float,
        at_frequency: float,
        show_work: bool = False,
    ) -> "Lead":
        """
        Design lead compensator for specific phase boost.

        Parameters
        ----------
        phase_boost : float
            Desired phase boost in degrees (typically 30-60 degrees)
        at_frequency : float
            Frequency at which to achieve maximum phase boost (rad/s)
        show_work : bool
            If True, show the design process

        Returns
        -------
        Lead
            Lead compensator achieving the specified phase boost

        Examples
        --------
        >>> lead = Lead.for_phase_boost(45, at_frequency=10, show_work=True)
        """
        if phase_boost >= 90:
            raise ValueError(
                "A single lead compensator cannot achieve >= 90 degrees phase boost.\n"
                "Consider using multiple lead stages or a different approach."
            )

        phi_rad = np.radians(phase_boost)

        if show_work:
            print("\nLead Compensator Design")
            print("=" * 50)
            print(f"\nGoal: Achieve {phase_boost} deg phase boost at w = {at_frequency} rad/s")

        # Calculate alpha
        # sin(phi_max) = (1 - alpha) / (1 + alpha)
        sin_phi = np.sin(phi_rad)
        alpha = (1 - sin_phi) / (1 + sin_phi)

        if show_work:
            print("\nStep 1: Calculate alpha")
            print(f"  sin(phi_max) = (1 - alpha) / (1 + alpha)")
            print(f"  sin({phase_boost} deg) = {sin_phi:.4f}")
            print(f"  Solving: alpha = {alpha:.4f}")

        # Place zero and pole around omega_max
        # zero = omega_max * sqrt(alpha)
        # pole = omega_max / sqrt(alpha)
        sqrt_alpha = np.sqrt(alpha)
        zero = at_frequency * sqrt_alpha
        pole = at_frequency / sqrt_alpha

        if show_work:
            print("\nStep 2: Place zero and pole around w_max")
            print(f"  Zero: z = w_max * sqrt(alpha) = {at_frequency} * {sqrt_alpha:.4f} = {zero:.3f}")
            print(f"  Pole: p = w_max / sqrt(alpha) = {at_frequency} / {sqrt_alpha:.4f} = {pole:.3f}")

        # Calculate gain for unity at low frequencies
        # Or adjust so |C(jw_max)| = 1/sqrt(alpha) to compensate for mag change
        gain = 1.0

        if show_work:
            print(f"\nStep 3: Set gain")
            print(f"  Gain: Kc = {gain}")
            print(f"\nResult: Lead(zero={zero:.3f}, pole={pole:.3f}, gain={gain})")
            print(f"\nNote: The compensator adds {phase_boost} deg at w = {at_frequency} rad/s")
            print(f"      and increases gain at high frequencies by factor {1/alpha:.2f}")

        return cls(zero=zero, pole=pole, gain=gain)

    @classmethod
    def for_phase_margin(
        cls,
        plant: "System",
        target_pm: float,
        show_work: bool = False,
    ) -> "Lead":
        """
        Design lead compensator to achieve target phase margin.

        Parameters
        ----------
        plant : System
            The plant to compensate
        target_pm : float
            Desired phase margin in degrees
        show_work : bool
            If True, show the design process

        Returns
        -------
        Lead
            Lead compensator achieving the target phase margin
        """
        from ..core.system import System

        if isinstance(plant, System):
            freq_info = plant.frequency_info()
        else:
            # Try to get margin info
            gm, pm, wcg, wcp = ctrl.margin(plant)
            freq_info = {"phase_margin": pm, "gain_crossover": wcg}

        current_pm = freq_info.get("phase_margin", 0) or 0
        needed_boost = target_pm - current_pm + 10  # Add 10 deg margin

        if needed_boost <= 0:
            if show_work:
                print(f"Current phase margin ({current_pm:.1f} deg) already meets target.")
            return cls(zero=1, pole=10, gain=1)

        # Design frequency is around current crossover
        wc = freq_info.get("gain_crossover") or 1.0

        if show_work:
            print(f"\nDesigning lead for {target_pm} deg phase margin")
            print(f"Current PM: {current_pm:.1f} deg")
            print(f"Needed boost: {needed_boost:.1f} deg")
            print(f"Design frequency: {wc:.2f} rad/s")

        return cls.for_phase_boost(min(needed_boost, 70), at_frequency=wc, show_work=show_work)

    def as_system(self) -> "System":
        """Convert to System object."""
        from ..core.system import System

        tf = self.as_tf()
        return System(tf, name=f"Lead (z={self.zero:.2f}, p={self.pole:.2f})")

    def as_tf(self) -> ctrl.TransferFunction:
        """Get transfer function representation."""
        num = [self.gain, self.gain * self.zero]
        den = [1, self.pole]
        return ctrl.TransferFunction(num, den)

    def __repr__(self) -> str:
        return f"Lead(zero={self.zero}, pole={self.pole}, gain={self.gain})"


class Lag:
    """
    Lag compensator for improving steady-state accuracy.

    Transfer Function:
                    s + z
        C(s) = Kc * -----
                    s + p

    where |z| > |p| (pole closer to origin than zero)

    Effects:
    - Improves steady-state accuracy (increases low-freq gain)
    - Decreases phase margin slightly
    - Minimal effect on transient if designed properly

    Examples
    --------
    >>> # Design by specifying zero and pole
    >>> lag = Lag(zero=10, pole=1)

    >>> # Design for steady-state improvement
    >>> lag = Lag.for_steady_state_improvement(plant, factor=10)
    """

    def __init__(self, zero: float, pole: float, gain: float = 1.0):
        """
        Create a lag compensator.

        Parameters
        ----------
        zero : float
            Zero location (positive value = zero at -zero)
        pole : float
            Pole location (positive value = pole at -pole)
        gain : float
            Overall gain (default: 1.0)
        """
        self.zero = abs(zero)
        self.pole = abs(pole)
        self.gain = gain

        if self.zero <= self.pole:
            raise ValueError(
                f"For a lag compensator, |zero| must be greater than |pole|.\n"
                f"You provided zero={self.zero}, pole={self.pole}.\n"
                f"This would create a lead compensator instead."
            )

    @classmethod
    def for_steady_state_improvement(
        cls,
        plant: "System",
        factor: float,
        show_work: bool = False,
    ) -> "Lag":
        """
        Design lag compensator to improve steady-state accuracy.

        Parameters
        ----------
        plant : System
            The plant to compensate
        factor : float
            Factor by which to reduce steady-state error
            (e.g., 10 means 10x improvement)
        show_work : bool
            If True, show the design process

        Returns
        -------
        Lag
            Lag compensator achieving the improvement
        """
        from ..core.system import System

        # Get current crossover frequency
        if isinstance(plant, System):
            freq_info = plant.frequency_info()
        else:
            gm, pm, wcg, wcp = ctrl.margin(plant)
            freq_info = {"gain_crossover": wcg}

        wc = freq_info.get("gain_crossover") or 1.0

        # Place lag well below crossover (1/10 of wc or less)
        zero = wc / 10
        pole = zero / factor

        if show_work:
            print(f"\nLag Compensator Design")
            print(f"=" * 50)
            print(f"\nGoal: Improve steady-state accuracy by factor of {factor}")
            print(f"\nCurrent crossover frequency: {wc:.2f} rad/s")
            print(f"Place zero at: z = wc/10 = {zero:.3f} rad/s")
            print(f"Place pole at: p = z/{factor} = {pole:.4f} rad/s")
            print(f"DC gain increase: {factor}")
            print(f"\nResult: Lag(zero={zero:.3f}, pole={pole:.4f})")

        return cls(zero=zero, pole=pole, gain=1.0)

    def as_system(self) -> "System":
        """Convert to System object."""
        from ..core.system import System

        tf = self.as_tf()
        return System(tf, name=f"Lag (z={self.zero:.2f}, p={self.pole:.4f})")

    def as_tf(self) -> ctrl.TransferFunction:
        """Get transfer function representation."""
        num = [1, self.zero]
        den = [1, self.pole]
        return ctrl.TransferFunction(num, den)

    def __repr__(self) -> str:
        return f"Lag(zero={self.zero}, pole={self.pole}, gain={self.gain})"


class LeadLag:
    """
    Combined lead-lag compensator.

    Uses lead section for transient improvement and lag section
    for steady-state improvement.

    Examples
    --------
    >>> # Create from individual components
    >>> lead = Lead(zero=1, pole=10)
    >>> lag = Lag(zero=10, pole=1)
    >>> leadlag = LeadLag(lead, lag)

    >>> # Or specify directly
    >>> leadlag = LeadLag.design(plant, phase_margin=50, ss_improvement=10)
    """

    def __init__(self, lead: Lead, lag: Lag):
        """
        Create a lead-lag compensator.

        Parameters
        ----------
        lead : Lead
            Lead compensator section
        lag : Lag
            Lag compensator section
        """
        self.lead = lead
        self.lag = lag

    @classmethod
    def design(
        cls,
        plant: "System",
        phase_margin: float = 50,
        ss_improvement: float = 10,
        show_work: bool = False,
    ) -> "LeadLag":
        """
        Design a combined lead-lag compensator.

        Parameters
        ----------
        plant : System
            The plant to compensate
        phase_margin : float
            Target phase margin in degrees
        ss_improvement : float
            Factor for steady-state improvement

        Returns
        -------
        LeadLag
            Combined lead-lag compensator
        """
        lead = Lead.for_phase_margin(plant, phase_margin, show_work=show_work)
        lag = Lag.for_steady_state_improvement(plant, ss_improvement, show_work=show_work)
        return cls(lead, lag)

    def as_system(self) -> "System":
        """Convert to System object."""
        from ..core.system import System

        lead_sys = self.lead.as_system()
        lag_sys = self.lag.as_system()
        combined = lead_sys.series(lag_sys)
        combined._name = "Lead-Lag Compensator"
        return combined

    def as_tf(self) -> ctrl.TransferFunction:
        """Get transfer function representation."""
        return ctrl.series(self.lead.as_tf(), self.lag.as_tf())

    def __repr__(self) -> str:
        return f"LeadLag({self.lead}, {self.lag})"
