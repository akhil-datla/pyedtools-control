"""
Discrete-time (z-domain) control systems support.

This module provides discrete-time system creation, conversion between
continuous and discrete domains, and z-domain analysis tools.
"""

import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Dict, Any


class DiscreteSystem:
    """
    Discrete-time (z-domain) control system.

    Represents a system in the z-domain with transfer function H(z).
    Supports conversion to/from continuous-time systems.

    Examples
    --------
    Create from transfer function:
    >>> sys = DiscreteSystem.from_tf([1], [1, -0.5], dt=0.1)

    Convert from continuous-time:
    >>> from pyed_control import first_order
    >>> continuous = first_order(tau=0.5)
    >>> discrete = DiscreteSystem.from_continuous(continuous, dt=0.1)

    Analyze:
    >>> sys.analyze()
    >>> sys.step()
    """

    def __init__(
        self,
        sys: ctrl.TransferFunction,
        dt: float,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """
        Create a DiscreteSystem from a python-control discrete transfer function.

        Parameters
        ----------
        sys : TransferFunction
            The underlying discrete system (must have dt set)
        dt : float
            Sample time in seconds
        name : str, optional
            Human-readable name
        description : str, optional
            Description of the system
        """
        self._sys = sys
        self._dt = dt
        self._name = name
        self._description = description

    # ========== Factory Methods ==========

    @classmethod
    def from_tf(
        cls,
        num: List[float],
        den: List[float],
        dt: float,
        name: Optional[str] = None,
    ) -> "DiscreteSystem":
        """
        Create discrete system from transfer function coefficients.

        Parameters
        ----------
        num : list
            Numerator coefficients [highest power of z ... constant]
        den : list
            Denominator coefficients [highest power of z ... constant]
        dt : float
            Sample time in seconds
        name : str, optional
            System name

        Examples
        --------
        >>> # H(z) = 0.1 / (z - 0.9) with Ts = 0.1s
        >>> sys = DiscreteSystem.from_tf([0.1], [1, -0.9], dt=0.1)
        """
        tf = ctrl.TransferFunction(num, den, dt)
        return cls(tf, dt=dt, name=name)

    @classmethod
    def from_zpk(
        cls,
        zeros: List[complex],
        poles: List[complex],
        gain: float,
        dt: float,
        name: Optional[str] = None,
    ) -> "DiscreteSystem":
        """
        Create discrete system from zeros, poles, and gain.

        Parameters
        ----------
        zeros : list
            Zero locations in z-plane
        poles : list
            Pole locations in z-plane (inside unit circle for stability)
        gain : float
            System gain
        dt : float
            Sample time in seconds

        Examples
        --------
        >>> # Pole at z=0.8, gain=0.2
        >>> sys = DiscreteSystem.from_zpk(zeros=[], poles=[0.8], gain=0.2, dt=0.1)
        """
        tf = ctrl.TransferFunction(
            ctrl.zpk(zeros, poles, gain, dt=dt)
        )
        return cls(tf, dt=dt, name=name)

    @classmethod
    def from_continuous(
        cls,
        continuous_sys: "System",
        dt: float,
        method: str = "zoh",
        name: Optional[str] = None,
        show_work: bool = False,
    ) -> "DiscreteSystem":
        """
        Convert a continuous-time system to discrete-time.

        Parameters
        ----------
        continuous_sys : System
            The continuous-time system to convert
        dt : float
            Sample time in seconds
        method : str
            Discretization method:
            - 'zoh' : Zero-order hold (default, most common)
            - 'tustin' : Bilinear transformation (Tustin's method)
            - 'matched' : Matched pole-zero method
            - 'euler' : Forward Euler method
            - 'backward_euler' : Backward Euler method
        name : str, optional
            Name for the discrete system
        show_work : bool
            If True, print the discretization process

        Returns
        -------
        DiscreteSystem
            The discretized system

        Examples
        --------
        >>> from pyed_control import first_order
        >>> continuous = first_order(tau=0.5, gain=2.0)
        >>> discrete = DiscreteSystem.from_continuous(continuous, dt=0.1, method='zoh')
        """
        # Import here to avoid circular imports
        from .system import System

        if isinstance(continuous_sys, System):
            cont_sys = continuous_sys._sys
        else:
            cont_sys = continuous_sys

        if show_work:
            print("\n" + "=" * 60)
            print(" Continuous-to-Discrete Conversion")
            print("=" * 60)
            print(f"\nSample time: Ts = {dt} s")
            print(f"Sample frequency: fs = {1/dt:.2f} Hz")
            print(f"Discretization method: {method.upper()}")
            print()

        # Perform conversion based on method
        if method.lower() == 'zoh':
            discrete_sys = ctrl.c2d(cont_sys, dt, method='zoh')
            if show_work:
                print("Zero-Order Hold (ZOH) Method:")
                print("  - Assumes input is held constant between samples")
                print("  - Most common method for sampled-data systems")
                print("  - Preserves DC gain exactly")

        elif method.lower() == 'tustin':
            discrete_sys = ctrl.c2d(cont_sys, dt, method='tustin')
            if show_work:
                print("Tustin (Bilinear) Method:")
                print("  - Maps s-plane to z-plane via s = (2/T)*(z-1)/(z+1)")
                print("  - Preserves stability (LHP -> inside unit circle)")
                print("  - Causes frequency warping at high frequencies")
                print(f"  - Warping factor at Nyquist: {np.tan(np.pi/2):.2f}")

        elif method.lower() == 'matched':
            discrete_sys = ctrl.c2d(cont_sys, dt, method='matched')
            if show_work:
                print("Matched Pole-Zero Method:")
                print("  - Maps poles/zeros via z = e^(s*T)")
                print("  - Preserves frequency response at DC and Nyquist")
                print("  - Good for systems with real poles/zeros")

        elif method.lower() == 'euler' or method.lower() == 'forward_euler':
            discrete_sys = ctrl.c2d(cont_sys, dt, method='euler')
            if show_work:
                print("Forward Euler Method:")
                print("  - Simplest method: s ≈ (z-1)/T")
                print("  - May cause instability for fast poles")
                print("  - Requires small sample time for accuracy")

        elif method.lower() == 'backward_euler':
            discrete_sys = ctrl.c2d(cont_sys, dt, method='backward_diff')
            if show_work:
                print("Backward Euler Method:")
                print("  - Uses s ≈ (z-1)/(T*z)")
                print("  - More stable than forward Euler")
                print("  - Introduces additional damping")

        else:
            raise ValueError(
                f"Unknown discretization method: {method}. "
                f"Use 'zoh', 'tustin', 'matched', 'euler', or 'backward_euler'."
            )

        if show_work:
            print("\nOriginal continuous-time poles:")
            cont_poles = ctrl.poles(cont_sys)
            for p in cont_poles:
                if np.isreal(p):
                    print(f"  s = {np.real(p):.4f}")
                else:
                    print(f"  s = {np.real(p):.4f} ± {abs(np.imag(p)):.4f}j")

            print("\nDiscrete-time poles:")
            disc_poles = ctrl.poles(discrete_sys)
            for p in disc_poles:
                mag = abs(p)
                phase = np.angle(p, deg=True)
                if np.isreal(p):
                    print(f"  z = {np.real(p):.4f} (|z| = {mag:.4f})")
                else:
                    print(f"  z = {np.real(p):.4f} ± {abs(np.imag(p)):.4f}j (|z| = {mag:.4f})")

            print(f"\nStability: {'STABLE' if all(abs(p) < 1 for p in disc_poles) else 'UNSTABLE'}")
            print()

        if name is None and hasattr(continuous_sys, '_name') and continuous_sys._name:
            name = f"{continuous_sys._name} (discrete, Ts={dt}s)"

        return cls(discrete_sys, dt=dt, name=name)

    # ========== Properties ==========

    @property
    def name(self) -> Optional[str]:
        """System name."""
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def dt(self) -> float:
        """Sample time in seconds."""
        return self._dt

    @property
    def sample_rate(self) -> float:
        """Sample rate in Hz."""
        return 1.0 / self._dt

    @property
    def nyquist_frequency(self) -> float:
        """Nyquist frequency in Hz."""
        return self.sample_rate / 2

    @property
    def is_stable(self) -> bool:
        """True if all poles are inside the unit circle."""
        return all(abs(p) < 1 for p in self.poles)

    @property
    def poles(self) -> np.ndarray:
        """Array of system poles in z-plane."""
        return ctrl.poles(self._sys)

    @property
    def zeros(self) -> np.ndarray:
        """Array of system zeros in z-plane."""
        return ctrl.zeros(self._sys)

    @property
    def order(self) -> int:
        """System order (number of poles)."""
        return len(self.poles)

    @property
    def dc_gain(self) -> float:
        """DC gain (gain at z=1)."""
        return float(ctrl.dcgain(self._sys))

    @property
    def num(self) -> np.ndarray:
        """Numerator coefficients."""
        return np.array(self._sys.num[0][0])

    @property
    def den(self) -> np.ndarray:
        """Denominator coefficients."""
        return np.array(self._sys.den[0][0])

    # ========== Analysis Methods ==========

    def analyze(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive discrete system analysis.

        Parameters
        ----------
        verbose : bool
            If True, print the report

        Returns
        -------
        dict
            Dictionary with analysis results
        """
        results = {
            "name": self._name,
            "dt": self._dt,
            "sample_rate": self.sample_rate,
            "order": self.order,
            "poles": self.poles,
            "zeros": self.zeros,
            "is_stable": self.is_stable,
            "dc_gain": self.dc_gain,
        }

        if verbose:
            self._print_analysis(results)

        return results

    def _print_analysis(self, results: Dict[str, Any]):
        """Print formatted analysis report."""
        print()
        header = " Discrete System Analysis"
        if self._name:
            header += f": {self._name}"
        print("=" * 60)
        print(header)
        print("=" * 60)

        # Sampling info
        print(f"\nSampling:")
        print(f"  Sample time:       Ts = {results['dt']:.4f} s")
        print(f"  Sample rate:       fs = {results['sample_rate']:.2f} Hz")
        print(f"  Nyquist frequency: fN = {self.nyquist_frequency:.2f} Hz")

        # Transfer function
        print("\nTransfer Function H(z):")
        print(self._format_tf())

        # Poles
        print("\nPoles (z-plane):")
        poles = results["poles"]
        if len(poles) == 0:
            print("  None (static gain)")
        else:
            for p in poles:
                mag = abs(p)
                phase = np.angle(p, deg=True)
                stable = "stable" if mag < 1 else "unstable" if mag > 1 else "marginal"
                if np.isreal(p):
                    print(f"  z = {np.real(p):.4f}")
                    print(f"      |z| = {mag:.4f}, angle = {phase:.1f}° ({stable})")
                else:
                    print(f"  z = {np.real(p):.4f} ± {abs(np.imag(p)):.4f}j")
                    print(f"      |z| = {mag:.4f}, angle = ±{abs(phase):.1f}° ({stable})")

        # Zeros
        print("\nZeros (z-plane):")
        zeros = results["zeros"]
        if len(zeros) == 0:
            print("  None")
        else:
            for z in zeros:
                if np.isreal(z):
                    print(f"  z = {np.real(z):.4f}")
                else:
                    print(f"  z = {np.real(z):.4f} ± {abs(np.imag(z)):.4f}j")

        # Stability
        print("\nStability:")
        if results["is_stable"]:
            print("  STABLE (all poles inside unit circle)")
        else:
            unstable = [p for p in poles if abs(p) > 1]
            marginal = [p for p in poles if abs(p) == 1]
            if len(unstable) > 0:
                print(f"  UNSTABLE ({len(unstable)} pole(s) outside unit circle)")
            elif len(marginal) > 0:
                print(f"  MARGINALLY STABLE ({len(marginal)} pole(s) on unit circle)")

        # DC Gain
        print(f"\nDC Gain: {results['dc_gain']:.4f}")
        print()

    def _format_tf(self) -> str:
        """Format transfer function for display."""
        num = self.num
        den = self.den

        def poly_str(coeffs, var="z"):
            terms = []
            order = len(coeffs) - 1
            for i, c in enumerate(coeffs):
                power = order - i
                if abs(c) < 1e-10:
                    continue
                if power == 0:
                    terms.append(f"{c:.4g}")
                elif power == 1:
                    if c == 1:
                        terms.append(var)
                    else:
                        terms.append(f"{c:.4g}{var}")
                else:
                    if c == 1:
                        terms.append(f"{var}^{power}")
                    else:
                        terms.append(f"{c:.4g}{var}^{power}")

            return " + ".join(terms) if terms else "0"

        num_str = poly_str(num)
        den_str = poly_str(den)

        width = max(len(num_str), len(den_str)) + 4
        lines = [
            "         " + num_str.center(width),
            "  H(z) = " + "-" * width,
            "         " + den_str.center(width),
        ]
        return "\n".join(lines)

    def step_info(self) -> Dict[str, float]:
        """
        Calculate discrete step response characteristics.

        Returns
        -------
        dict
            Dictionary with step response characteristics
        """
        if not self.is_stable:
            raise ValueError("Cannot compute step info for unstable system")

        # Simulate step response
        t, y = ctrl.step_response(self._sys)
        y = np.array(y).flatten()
        t = np.array(t).flatten()

        # Steady state
        ss = y[-1]

        if abs(ss) < 1e-10:
            return {
                "rise_time": float("nan"),
                "settling_time": float("nan"),
                "overshoot": 0.0,
                "steady_state": 0.0,
                "num_samples_to_settle": 0,
            }

        y_norm = y / ss

        # Rise time (10% to 90%)
        try:
            idx_10 = np.where(y_norm >= 0.1)[0][0]
            idx_90 = np.where(y_norm >= 0.9)[0][0]
            rise_time = t[idx_90] - t[idx_10]
            rise_samples = idx_90 - idx_10
        except IndexError:
            rise_time = float("nan")
            rise_samples = 0

        # Settling time (2%)
        settled = np.abs(y_norm - 1) <= 0.02
        if np.any(settled):
            outside = np.where(~settled)[0]
            if len(outside) > 0:
                settling_idx = outside[-1]
                settling_time = t[settling_idx]
                settling_samples = settling_idx
            else:
                settling_time = 0.0
                settling_samples = 0
        else:
            settling_time = t[-1]
            settling_samples = len(t) - 1

        # Overshoot
        peak_idx = np.argmax(y)
        overshoot = max(0, (y[peak_idx] - ss) / abs(ss) * 100)

        return {
            "rise_time": float(rise_time),
            "settling_time": float(settling_time),
            "overshoot": float(overshoot),
            "steady_state": float(ss),
            "num_samples_to_settle": int(settling_samples),
        }

    # ========== Plotting Methods ==========

    def step(
        self,
        N: Optional[int] = None,
        show_samples: bool = True,
        ax: Optional[plt.Axes] = None,
        **kwargs,
    ) -> "DiscreteSystem":
        """
        Plot step response.

        Parameters
        ----------
        N : int, optional
            Number of samples (auto-calculated if not provided)
        show_samples : bool
            If True, show sample points as markers
        ax : matplotlib axis, optional
            Axis to plot on

        Returns
        -------
        DiscreteSystem
            Returns self for chaining
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        t, y = ctrl.step_response(self._sys, T=N)
        y = np.array(y).flatten()

        label = kwargs.pop("label", self._name or "System")

        if show_samples:
            ax.step(t, y, where='post', label=label, **kwargs)
            ax.plot(t, y, 'o', markersize=4, color=ax.lines[-1].get_color())
        else:
            ax.step(t, y, where='post', label=label, **kwargs)

        # Add sample time annotation
        ax.text(
            0.02, 0.98,
            f"Ts = {self._dt:.4f} s",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Response")
        ax.set_title(f"Discrete Step Response{': ' + self._name if self._name else ''}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")

        plt.tight_layout()
        plt.show()

        return self

    def impulse(
        self,
        N: Optional[int] = None,
        show_samples: bool = True,
        ax: Optional[plt.Axes] = None,
        **kwargs,
    ) -> "DiscreteSystem":
        """
        Plot impulse response.

        Parameters
        ----------
        N : int, optional
            Number of samples
        show_samples : bool
            If True, show sample points as markers
        ax : matplotlib axis, optional
            Axis to plot on

        Returns
        -------
        DiscreteSystem
            Returns self for chaining
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        t, y = ctrl.impulse_response(self._sys, T=N)
        y = np.array(y).flatten()

        label = kwargs.pop("label", self._name or "System")

        if show_samples:
            ax.stem(t, y, label=label, **kwargs)
        else:
            ax.plot(t, y, label=label, **kwargs)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Response")
        ax.set_title(f"Discrete Impulse Response{': ' + self._name if self._name else ''}")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.show()

        return self

    def bode(
        self,
        omega: Optional[np.ndarray] = None,
        show_nyquist_freq: bool = True,
        ax: Optional[tuple] = None,
        **kwargs,
    ) -> "DiscreteSystem":
        """
        Plot Bode diagram.

        Parameters
        ----------
        omega : array-like, optional
            Frequency range (up to Nyquist by default)
        show_nyquist_freq : bool
            If True, mark the Nyquist frequency
        ax : tuple of axes, optional
            (mag_ax, phase_ax) to plot on

        Returns
        -------
        DiscreteSystem
            Returns self for chaining
        """
        if ax is None:
            fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        else:
            ax_mag, ax_phase = ax

        # Default to frequencies up to Nyquist
        omega_nyq = np.pi / self._dt
        if omega is None:
            omega = np.logspace(-2, np.log10(omega_nyq), 500)

        mag, phase, omega_out = ctrl.frequency_response(self._sys, omega)
        mag = np.abs(np.array(mag).flatten())
        phase = np.angle(np.array(phase).flatten(), deg=True)
        mag_db = 20 * np.log10(mag)

        ax_mag.semilogx(omega_out, mag_db, **kwargs)
        ax_mag.set_ylabel("Magnitude (dB)")
        ax_mag.set_title(f"Discrete Bode Diagram{': ' + self._name if self._name else ''}")
        ax_mag.grid(True, which="both", alpha=0.3)
        ax_mag.axhline(y=0, color="gray", linestyle="-", alpha=0.5)

        ax_phase.semilogx(omega_out, phase, **kwargs)
        ax_phase.set_xlabel("Frequency (rad/s)")
        ax_phase.set_ylabel("Phase (deg)")
        ax_phase.grid(True, which="both", alpha=0.3)
        ax_phase.axhline(y=-180, color="gray", linestyle="-", alpha=0.5)

        if show_nyquist_freq:
            ax_mag.axvline(x=omega_nyq, color="red", linestyle="--", alpha=0.7,
                          label=f"Nyquist: {omega_nyq:.2f} rad/s")
            ax_phase.axvline(x=omega_nyq, color="red", linestyle="--", alpha=0.7)
            ax_mag.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        plt.show()

        return self

    def pole_zero_map(
        self,
        ax: Optional[plt.Axes] = None,
        show_unit_circle: bool = True,
        **kwargs,
    ) -> "DiscreteSystem":
        """
        Plot pole-zero map in z-plane.

        Parameters
        ----------
        ax : matplotlib axis, optional
            Axis to plot on
        show_unit_circle : bool
            If True, draw the unit circle (stability boundary)

        Returns
        -------
        DiscreteSystem
            Returns self for chaining
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        poles = self.poles
        zeros = self.zeros

        # Draw unit circle
        if show_unit_circle:
            theta = np.linspace(0, 2 * np.pi, 100)
            ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, label='Unit circle')

            # Shade stability region
            circle = plt.Circle((0, 0), 1, fill=True, color='green', alpha=0.1)
            ax.add_patch(circle)

        # Plot poles
        if len(poles) > 0:
            ax.plot(np.real(poles), np.imag(poles), "x", markersize=10,
                   markeredgewidth=2, color="red", label="Poles")

        # Plot zeros
        if len(zeros) > 0:
            ax.plot(np.real(zeros), np.imag(zeros), "o", markersize=10,
                   markerfacecolor="none", markeredgewidth=2, color="blue", label="Zeros")

        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        ax.axvline(x=0, color="gray", linestyle="-", alpha=0.3)
        ax.set_xlabel("Real")
        ax.set_ylabel("Imaginary")
        ax.set_title(f"Z-Plane Pole-Zero Map{': ' + self._name if self._name else ''}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect("equal")

        # Set appropriate limits
        all_points = list(poles) + list(zeros) + [1, -1, 1j, -1j]
        max_abs = max(abs(p) for p in all_points) * 1.2
        ax.set_xlim(-max_abs, max_abs)
        ax.set_ylim(-max_abs, max_abs)

        plt.tight_layout()
        plt.show()

        return self

    # ========== Conversion ==========

    def to_continuous(
        self,
        method: str = "tustin",
        show_work: bool = False,
    ) -> "System":
        """
        Convert to continuous-time system.

        Parameters
        ----------
        method : str
            Conversion method ('tustin' recommended)
        show_work : bool
            If True, print the conversion process

        Returns
        -------
        System
            Continuous-time equivalent
        """
        from .system import System

        if show_work:
            print("\n" + "=" * 60)
            print(" Discrete-to-Continuous Conversion")
            print("=" * 60)
            print(f"\nSample time: Ts = {self._dt} s")
            print(f"Conversion method: {method.upper()}")
            print()

        continuous_sys = ctrl.d2c(self._sys, method=method)

        if show_work:
            print("Discrete poles (z-plane):")
            for p in self.poles:
                print(f"  z = {p:.4f}, |z| = {abs(p):.4f}")

            cont_poles = ctrl.poles(continuous_sys)
            print("\nContinuous poles (s-plane):")
            for p in cont_poles:
                if np.isreal(p):
                    print(f"  s = {np.real(p):.4f}")
                else:
                    print(f"  s = {np.real(p):.4f} ± {abs(np.imag(p)):.4f}j")
            print()

        name = None
        if self._name:
            name = f"{self._name} (continuous)"

        return System(continuous_sys, name=name)

    # ========== Operations ==========

    def series(self, other: "DiscreteSystem") -> "DiscreteSystem":
        """Connect another discrete system in series."""
        if self._dt != other._dt:
            raise ValueError(
                f"Sample times must match: {self._dt} != {other._dt}"
            )
        combined = ctrl.series(self._sys, other._sys)
        return DiscreteSystem(combined, dt=self._dt)

    def parallel(self, other: "DiscreteSystem") -> "DiscreteSystem":
        """Connect another discrete system in parallel."""
        if self._dt != other._dt:
            raise ValueError(
                f"Sample times must match: {self._dt} != {other._dt}"
            )
        combined = ctrl.parallel(self._sys, other._sys)
        return DiscreteSystem(combined, dt=self._dt)

    def feedback(self, H: float = 1) -> "DiscreteSystem":
        """Apply feedback with gain H."""
        closed = ctrl.feedback(self._sys, H)
        return DiscreteSystem(closed, dt=self._dt)

    def __repr__(self) -> str:
        """String representation."""
        info = f"DiscreteSystem("
        if self._name:
            info += f"'{self._name}', "
        info += f"order={self.order}, "
        info += f"dt={self._dt}, "
        info += f"stable={self.is_stable}"
        info += ")"
        return info

    def _repr_html_(self) -> str:
        """
        HTML representation for Jupyter notebooks.

        Renders the transfer function using MathJax for beautiful equations.
        """
        latex = self._format_tf_latex()

        if self.is_stable:
            stability = '<span style="color: green;">Stable</span>'
        else:
            stability = '<span style="color: red;">Unstable</span>'

        html = f'''
        <div style="font-family: sans-serif; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background: #f0f8ff;">
            <div style="font-weight: bold; margin-bottom: 10px;">
                {self._name or 'Discrete System'} (order={self.order}, {stability})
            </div>
            <div style="font-size: 0.9em; color: #666; margin-bottom: 10px;">
                Sample time: Ts = {self._dt}s | Sample rate: {self.sample_rate:.2f} Hz
            </div>
            <div style="font-size: 1.2em; margin: 15px 0;">
                $${latex}$$
            </div>
            <div style="font-size: 0.9em; color: #666;">
                Poles: {self._format_poles_html()} | DC Gain: {self.dc_gain:.4g}
            </div>
        </div>
        '''
        return html

    def _repr_latex_(self) -> str:
        """LaTeX representation for Jupyter notebooks."""
        return f"${self._format_tf_latex()}$"

    def _format_tf_latex(self) -> str:
        """Format transfer function as LaTeX."""
        num = self.num
        den = self.den

        def poly_latex(coeffs, var="z"):
            terms = []
            order = len(coeffs) - 1
            for i, c in enumerate(coeffs):
                power = order - i
                if abs(c) < 1e-10:
                    continue

                if abs(c - round(c)) < 1e-10:
                    c_str = str(int(round(c)))
                elif abs(c) < 0.001 or abs(c) > 1000:
                    c_str = f"{c:.2e}"
                else:
                    c_str = f"{c:.4g}"

                if power == 0:
                    terms.append(c_str)
                elif power == 1:
                    if c == 1:
                        terms.append(var)
                    elif c == -1:
                        terms.append(f"-{var}")
                    else:
                        terms.append(f"{c_str}{var}")
                else:
                    if c == 1:
                        terms.append(f"{var}^{{{power}}}")
                    elif c == -1:
                        terms.append(f"-{var}^{{{power}}}")
                    else:
                        terms.append(f"{c_str}{var}^{{{power}}}")

            if not terms:
                return "0"

            result = terms[0]
            for term in terms[1:]:
                if term.startswith("-"):
                    result += f" - {term[1:]}"
                else:
                    result += f" + {term}"
            return result

        num_latex = poly_latex(num)
        den_latex = poly_latex(den)
        return f"H(z) = \\frac{{{num_latex}}}{{{den_latex}}}"

    def _format_poles_html(self) -> str:
        """Format poles for HTML display."""
        poles = self.poles
        if len(poles) == 0:
            return "None"

        pole_strs = []
        for p in poles:
            if np.isreal(p):
                pole_strs.append(f"{np.real(p):.3g}")
            else:
                pole_strs.append(f"{np.real(p):.3g}±{abs(np.imag(p)):.3g}j")

        return ", ".join(pole_strs)


# ========== Convenience Functions ==========


def c2d(
    continuous_sys: "System",
    dt: float,
    method: str = "zoh",
    show_work: bool = False,
) -> DiscreteSystem:
    """
    Convert continuous-time system to discrete-time.

    Convenience function for DiscreteSystem.from_continuous().

    Parameters
    ----------
    continuous_sys : System
        Continuous-time system to convert
    dt : float
        Sample time in seconds
    method : str
        Discretization method: 'zoh', 'tustin', 'matched', 'euler', 'backward_euler'
    show_work : bool
        If True, print the discretization process

    Returns
    -------
    DiscreteSystem
        The discretized system

    Examples
    --------
    >>> from pyed_control import first_order
    >>> from pyed_control.core.discrete import c2d
    >>> continuous = first_order(tau=0.5)
    >>> discrete = c2d(continuous, dt=0.1)
    """
    return DiscreteSystem.from_continuous(
        continuous_sys, dt=dt, method=method, show_work=show_work
    )


def d2c(
    discrete_sys: DiscreteSystem,
    method: str = "tustin",
    show_work: bool = False,
) -> "System":
    """
    Convert discrete-time system to continuous-time.

    Parameters
    ----------
    discrete_sys : DiscreteSystem
        Discrete-time system to convert
    method : str
        Conversion method (default: 'tustin')
    show_work : bool
        If True, print the conversion process

    Returns
    -------
    System
        Continuous-time equivalent
    """
    return discrete_sys.to_continuous(method=method, show_work=show_work)


def explain_discretization(method: str = None):
    """
    Explain discretization methods and their trade-offs.

    Parameters
    ----------
    method : str, optional
        Specific method to explain, or None for overview
    """
    if method is None:
        print("""
========================================
 Discretization Methods Overview
========================================

When converting a continuous-time system G(s) to discrete-time H(z),
different methods offer different trade-offs:

1. ZERO-ORDER HOLD (ZOH)
   - Most common method for sampled-data systems
   - Assumes input is held constant between samples
   - Exact for step inputs
   - Preserves DC gain exactly

2. TUSTIN (Bilinear Transform)
   - Maps s-plane to z-plane: s = (2/T)*(z-1)/(z+1)
   - Preserves stability (LHP -> inside unit circle)
   - Good frequency response matching at low frequencies
   - Causes "frequency warping" at high frequencies

3. MATCHED POLE-ZERO
   - Maps poles and zeros directly: z = e^(sT)
   - Preserves system structure
   - Good for systems with well-defined poles/zeros
   - May not preserve frequency response exactly

4. FORWARD EULER
   - Simplest method: s ≈ (z-1)/T
   - Can cause instability for fast poles
   - Requires small sample time
   - Mostly for educational purposes

5. BACKWARD EULER
   - Uses s ≈ (z-1)/(T*z)
   - More stable than forward Euler
   - Adds damping to the system
   - Used when stability is critical

CHOOSING A METHOD:
- General purpose: ZOH (default)
- Frequency response matching: Tustin
- System with distinct poles/zeros: Matched
- Real-time control with stability concerns: Backward Euler
- Educational/simple analysis: Forward Euler
""")
    elif method.lower() == 'zoh':
        print("""
Zero-Order Hold (ZOH) Method
============================

The ZOH method models how real digital-to-analog converters (DACs) work.
The input is held constant during each sample period.

Mathematical basis:
  H(z) = (1 - z^(-1)) * Z{L^(-1){G(s)/s}}

where Z{} is the Z-transform and L^(-1){} is inverse Laplace.

Properties:
- Exact for step inputs
- Preserves DC gain
- Introduces a half-sample delay
- Most common for sampled-data control

Use when:
- Implementing controllers on microcontrollers
- System has step-like inputs
- DC gain must be preserved
""")
    elif method.lower() == 'tustin':
        print("""
Tustin (Bilinear Transform) Method
==================================

The Tustin method uses a bilinear mapping between s and z domains.

Transformation:
  s = (2/T) * (z - 1) / (z + 1)

Or equivalently:
  z = (1 + sT/2) / (1 - sT/2)

Properties:
- Maps left half s-plane to inside unit circle (stability preserved)
- Maps imaginary axis to unit circle
- Causes frequency warping: w_d = (2/T) * tan(w_c * T/2)
- No aliasing (frequencies map one-to-one)

Use when:
- Frequency response must be preserved
- Stability is critical
- Working with filters
- Need one-to-one frequency mapping

Note: For critical frequencies, use pre-warping to compensate.
""")
    else:
        print(f"Unknown method: {method}")
        print("Available methods: 'zoh', 'tustin', 'matched', 'euler', 'backward_euler'")
