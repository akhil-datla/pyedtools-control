"""
Custom system creation tools.

Provides fluent builders and utilities for creating custom systems
without dealing with transfer function coefficients directly.
"""

import numpy as np
from typing import Optional, List, Union
from ..core.system import System


class SystemBuilder:
    """
    Fluent builder for creating custom systems step-by-step.

    Build systems by adding poles, zeros, and setting gain without
    needing to compute transfer function coefficients manually.

    Examples
    --------
    >>> # Build a system with two real poles and one zero
    >>> sys = (SystemBuilder()
    ...     .set_name("My Custom System")
    ...     .add_pole(-1)
    ...     .add_pole(-2)
    ...     .add_zero(-0.5)
    ...     .set_dc_gain(5.0)
    ...     .build())

    >>> # Build a second-order system with complex poles
    >>> sys = (SystemBuilder()
    ...     .add_complex_poles(wn=10, zeta=0.5)
    ...     .set_dc_gain(1.0)
    ...     .build())

    >>> # Build a type-2 system (two integrators)
    >>> sys = (SystemBuilder()
    ...     .add_integrators(2)
    ...     .add_real_pole(time_constant=0.1)
    ...     .set_gain(10)
    ...     .build())
    """

    def __init__(self):
        self._zeros: List[complex] = []
        self._poles: List[complex] = []
        self._gain: float = 1.0
        self._name: Optional[str] = None
        self._description: Optional[str] = None

    def set_name(self, name: str) -> "SystemBuilder":
        """Set the system name."""
        self._name = name
        return self

    def set_description(self, description: str) -> "SystemBuilder":
        """Set the system description."""
        self._description = description
        return self

    def add_pole(self, pole: complex) -> "SystemBuilder":
        """
        Add a pole to the system.

        For complex poles, the conjugate is automatically added.

        Parameters
        ----------
        pole : complex or float
            Pole location (use negative real part for stability)

        Examples
        --------
        >>> builder.add_pole(-1)        # Real pole at s = -1
        >>> builder.add_pole(-1 + 2j)   # Complex pair at -1 +/- 2j
        """
        self._poles.append(pole)
        # Automatically add conjugate for complex poles
        if np.imag(pole) != 0:
            self._poles.append(np.conj(pole))
        return self

    def add_poles(self, *poles: complex) -> "SystemBuilder":
        """Add multiple poles at once."""
        for pole in poles:
            self.add_pole(pole)
        return self

    def add_zero(self, zero: complex) -> "SystemBuilder":
        """
        Add a zero to the system.

        For complex zeros, the conjugate is automatically added.

        Parameters
        ----------
        zero : complex or float
            Zero location
        """
        self._zeros.append(zero)
        # Automatically add conjugate for complex zeros
        if np.imag(zero) != 0:
            self._zeros.append(np.conj(zero))
        return self

    def add_zeros(self, *zeros: complex) -> "SystemBuilder":
        """Add multiple zeros at once."""
        for zero in zeros:
            self.add_zero(zero)
        return self

    def add_real_pole(
        self,
        location: Optional[float] = None,
        time_constant: Optional[float] = None,
    ) -> "SystemBuilder":
        """
        Add a real pole by location or time constant.

        Parameters
        ----------
        location : float, optional
            Pole location (e.g., -2 for pole at s = -2)
        time_constant : float, optional
            Time constant tau (pole at s = -1/tau)

        Examples
        --------
        >>> builder.add_real_pole(location=-5)
        >>> builder.add_real_pole(time_constant=0.1)  # Pole at s = -10
        """
        if location is not None:
            self._poles.append(location)
        elif time_constant is not None:
            self._poles.append(-1.0 / time_constant)
        else:
            raise ValueError("Must specify either location or time_constant")
        return self

    def add_complex_poles(self, wn: float, zeta: float) -> "SystemBuilder":
        """
        Add a complex conjugate pole pair from natural frequency and damping.

        Parameters
        ----------
        wn : float
            Natural frequency (rad/s)
        zeta : float
            Damping ratio (0 < zeta < 1 for complex poles)

        Examples
        --------
        >>> builder.add_complex_poles(wn=10, zeta=0.5)
        """
        if zeta >= 1:
            # Overdamped - two real poles
            s1 = -zeta * wn + wn * np.sqrt(zeta**2 - 1)
            s2 = -zeta * wn - wn * np.sqrt(zeta**2 - 1)
            self._poles.append(s1)
            self._poles.append(s2)
        else:
            # Underdamped - complex conjugate pair
            sigma = -zeta * wn
            omega_d = wn * np.sqrt(1 - zeta**2)
            self._poles.append(complex(sigma, omega_d))
            self._poles.append(complex(sigma, -omega_d))
        return self

    def add_complex_zeros(self, wn: float, zeta: float) -> "SystemBuilder":
        """
        Add a complex conjugate zero pair from natural frequency and damping.

        Parameters
        ----------
        wn : float
            Natural frequency (rad/s)
        zeta : float
            Damping ratio
        """
        if zeta >= 1:
            s1 = -zeta * wn + wn * np.sqrt(zeta**2 - 1)
            s2 = -zeta * wn - wn * np.sqrt(zeta**2 - 1)
            self._zeros.append(s1)
            self._zeros.append(s2)
        else:
            sigma = -zeta * wn
            omega_d = wn * np.sqrt(1 - zeta**2)
            self._zeros.append(complex(sigma, omega_d))
            self._zeros.append(complex(sigma, -omega_d))
        return self

    def add_integrators(self, n: int = 1) -> "SystemBuilder":
        """
        Add n integrators (poles at s = 0).

        Parameters
        ----------
        n : int
            Number of integrators to add (default: 1)

        Examples
        --------
        >>> builder.add_integrators(2)  # Type-2 system
        """
        for _ in range(n):
            self._poles.append(0.0)
        return self

    def add_differentiator(self) -> "SystemBuilder":
        """Add a differentiator (zero at s = 0)."""
        self._zeros.append(0.0)
        return self

    def set_gain(self, gain: float) -> "SystemBuilder":
        """
        Set the overall system gain (multiplier).

        Parameters
        ----------
        gain : float
            System gain
        """
        self._gain = gain
        return self

    def set_dc_gain(self, dc_gain: float) -> "SystemBuilder":
        """
        Set the DC gain (gain at s = 0).

        Automatically calculates the required system gain to achieve
        the specified DC gain.

        Parameters
        ----------
        dc_gain : float
            Desired DC gain

        Notes
        -----
        If the system has poles at the origin (integrators), the DC gain
        is infinite and this method will raise an error.
        """
        # Check for integrators
        if any(np.isclose(p, 0) for p in self._poles):
            raise ValueError(
                "Cannot set DC gain for systems with integrators (poles at origin). "
                "Use set_gain() instead."
            )

        # Calculate what gain we need
        # DC gain = gain * prod(zeros) / prod(poles) evaluated at s=0
        # For zero/pole at -a: (s + a) at s=0 gives a
        zero_product = 1.0
        for z in self._zeros:
            zero_product *= abs(z) if np.isreal(z) else abs(z)

        pole_product = 1.0
        for p in self._poles:
            pole_product *= abs(p) if np.isreal(p) else abs(p)

        if pole_product == 0:
            raise ValueError("Cannot compute DC gain with pole at origin")

        self._gain = dc_gain * pole_product / zero_product if zero_product != 0 else dc_gain * pole_product
        return self

    def build(self) -> System:
        """
        Build and return the System.

        Returns
        -------
        System
            The constructed system

        Raises
        ------
        ValueError
            If no poles are defined (system would be improper)
        """
        if len(self._poles) == 0:
            raise ValueError(
                "Cannot build system with no poles. "
                "Add at least one pole using add_pole() or add_real_pole()."
            )

        if len(self._zeros) > len(self._poles):
            raise ValueError(
                f"System would be improper: {len(self._zeros)} zeros > {len(self._poles)} poles. "
                "Add more poles or remove some zeros."
            )

        # Create system from zeros, poles, gain
        sys = System.from_zpk(
            zeros=[complex(z) for z in self._zeros],
            poles=[complex(p) for p in self._poles],
            gain=self._gain,
            name=self._name,
        )

        if self._description:
            sys._description = self._description

        return sys

    def preview(self) -> str:
        """
        Preview the system configuration before building.

        Returns
        -------
        str
            Description of the system to be built
        """
        lines = ["System Builder Preview", "=" * 30]

        if self._name:
            lines.append(f"Name: {self._name}")

        lines.append(f"\nPoles ({len(self._poles)}):")
        if not self._poles:
            lines.append("  (none)")
        for p in self._poles:
            if np.isreal(p):
                lines.append(f"  s = {np.real(p):.4f}")
            else:
                lines.append(f"  s = {np.real(p):.4f} + {np.imag(p):.4f}j")

        lines.append(f"\nZeros ({len(self._zeros)}):")
        if not self._zeros:
            lines.append("  (none)")
        for z in self._zeros:
            if np.isreal(z):
                lines.append(f"  s = {np.real(z):.4f}")
            else:
                lines.append(f"  s = {np.real(z):.4f} + {np.imag(z):.4f}j")

        lines.append(f"\nGain: {self._gain}")

        result = "\n".join(lines)
        print(result)
        return result

    def __repr__(self) -> str:
        return f"SystemBuilder(poles={len(self._poles)}, zeros={len(self._zeros)}, gain={self._gain})"


def from_time_constant(tau: float, gain: float = 1.0, name: Optional[str] = None) -> System:
    """
    Create a first-order system from time constant.

    G(s) = gain / (tau*s + 1)

    Parameters
    ----------
    tau : float
        Time constant in seconds
    gain : float
        DC gain (default: 1.0)
    name : str, optional
        System name
    """
    return System.first_order(tau=tau, gain=gain, name=name)


def from_bandwidth(bandwidth: float, gain: float = 1.0, name: Optional[str] = None) -> System:
    """
    Create a first-order system from bandwidth.

    Parameters
    ----------
    bandwidth : float
        -3dB bandwidth in rad/s
    gain : float
        DC gain (default: 1.0)
    name : str, optional
        System name
    """
    tau = 1.0 / bandwidth
    return System.first_order(tau=tau, gain=gain, name=name)


def from_rise_time(rise_time: float, gain: float = 1.0, name: Optional[str] = None) -> System:
    """
    Create a first-order system with approximate rise time.

    Uses the approximation: rise_time ≈ 2.2 * tau

    Parameters
    ----------
    rise_time : float
        Desired rise time in seconds
    gain : float
        DC gain (default: 1.0)
    name : str, optional
        System name
    """
    tau = rise_time / 2.2
    return System.first_order(tau=tau, gain=gain, name=name)


def from_settling_time(
    settling_time: float,
    zeta: float = 0.7,
    gain: float = 1.0,
    name: Optional[str] = None,
) -> System:
    """
    Create a second-order system with specified settling time.

    Uses the approximation: settling_time ≈ 4 / (zeta * wn)

    Parameters
    ----------
    settling_time : float
        Desired settling time in seconds
    zeta : float
        Damping ratio (default: 0.7)
    gain : float
        DC gain (default: 1.0)
    name : str, optional
        System name
    """
    wn = 4.0 / (zeta * settling_time)
    return System.second_order(wn=wn, zeta=zeta, gain=gain, name=name)


def from_overshoot(
    overshoot: float,
    settling_time: float = 1.0,
    gain: float = 1.0,
    name: Optional[str] = None,
) -> System:
    """
    Create a second-order system with specified overshoot.

    Parameters
    ----------
    overshoot : float
        Desired percent overshoot (e.g., 10 for 10%)
    settling_time : float
        Desired settling time in seconds (default: 1.0)
    gain : float
        DC gain (default: 1.0)
    name : str, optional
        System name
    """
    # Calculate zeta from overshoot
    # overshoot = exp(-zeta*pi/sqrt(1-zeta^2)) * 100
    if overshoot <= 0:
        zeta = 1.0  # Critically damped, no overshoot
    else:
        # Solve for zeta
        ln_os = np.log(overshoot / 100)
        zeta = -ln_os / np.sqrt(np.pi**2 + ln_os**2)

    wn = 4.0 / (zeta * settling_time)
    return System.second_order(wn=wn, zeta=zeta, gain=gain, name=name)
