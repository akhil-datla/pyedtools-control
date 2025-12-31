"""
The System class - the central abstraction for control systems education.

This class wraps transfer functions and state-space models, providing
a unified, student-friendly interface for analysis and design.
"""

import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Dict, Any

from ..exceptions import (
    UnstableSystemError,
    ImproperSystemError,
    InvalidParameterError,
    validate_transfer_function,
    validate_positive,
    validate_damping_ratio,
)


class System:
    """
    The central class for control systems education.

    A System wraps transfer functions or state-space models and provides
    a unified, student-friendly interface for analysis and design.

    Examples
    --------
    Create from transfer function:
    >>> sys = System.from_tf([1], [1, 2, 1])

    Create a standard second-order system:
    >>> sys = System.second_order(wn=2.0, zeta=0.5)

    Quick analysis:
    >>> sys.analyze()          # Print comprehensive analysis
    >>> sys.step()             # Plot step response
    >>> sys.bode()             # Plot Bode diagram
    >>> sys.is_stable          # Check stability

    Chained operations:
    >>> closed_loop = plant.with_controller(pid)
    >>> closed_loop.step()
    """

    def __init__(
        self,
        sys: Union[ctrl.TransferFunction, ctrl.StateSpace],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """
        Create a System from a python-control object.

        Parameters
        ----------
        sys : TransferFunction or StateSpace
            The underlying system representation
        name : str, optional
            Human-readable name for the system
        description : str, optional
            Description of what the system represents
        """
        self._sys = sys
        self._name = name
        self._description = description
        self._physical_params = {}
        self._input_unit = None
        self._output_unit = None
        self._system_type = None

    # ========== Factory Methods ==========

    @classmethod
    def from_tf(
        cls,
        num: List[float],
        den: List[float],
        name: Optional[str] = None,
    ) -> "System":
        """
        Create system from transfer function coefficients.

        Parameters
        ----------
        num : list
            Numerator coefficients [highest power ... constant]
        den : list
            Denominator coefficients [highest power ... constant]
        name : str, optional
            Name for the system

        Examples
        --------
        >>> # G(s) = 5 / (s^2 + 3s + 2)
        >>> sys = System.from_tf([5], [1, 3, 2])

        >>> # G(s) = (s + 1) / (s^2 + 2s + 5)
        >>> sys = System.from_tf([1, 1], [1, 2, 5])
        """
        validate_transfer_function(num, den)
        tf = ctrl.TransferFunction(num, den)
        return cls(tf, name=name)

    @classmethod
    def from_zpk(
        cls,
        zeros: List[complex],
        poles: List[complex],
        gain: float,
        name: Optional[str] = None,
    ) -> "System":
        """
        Create system from zeros, poles, and gain.

        Parameters
        ----------
        zeros : list
            Zero locations (can be complex)
        poles : list
            Pole locations (can be complex)
        gain : float
            Overall system gain

        Examples
        --------
        >>> # Zero at s=-1, poles at s=-2,-3, gain=5
        >>> sys = System.from_zpk(zeros=[-1], poles=[-2, -3], gain=5)
        """
        tf = ctrl.TransferFunction(
            ctrl.zpk(zeros, poles, gain)
        )
        return cls(tf, name=name)

    @classmethod
    def from_ss(
        cls,
        A: List[List[float]],
        B: List[List[float]],
        C: List[List[float]],
        D: List[List[float]],
        name: Optional[str] = None,
    ) -> "System":
        """
        Create system from state-space matrices.

        Parameters
        ----------
        A : array-like
            State matrix (n x n)
        B : array-like
            Input matrix (n x m)
        C : array-like
            Output matrix (p x n)
        D : array-like
            Feedthrough matrix (p x m)

        Examples
        --------
        >>> A = [[-1, 0], [0, -2]]
        >>> B = [[1], [1]]
        >>> C = [[1, 0]]
        >>> D = [[0]]
        >>> sys = System.from_ss(A, B, C, D)
        """
        ss = ctrl.StateSpace(A, B, C, D)
        return cls(ss, name=name)

    @classmethod
    def first_order(
        cls,
        tau: float,
        gain: float = 1.0,
        name: Optional[str] = None,
    ) -> "System":
        """
        Create a first-order system: G(s) = gain / (tau*s + 1)

        Parameters
        ----------
        tau : float
            Time constant (seconds)
        gain : float
            DC gain (default 1.0)

        Examples
        --------
        >>> sys = System.first_order(tau=0.5, gain=2.0)
        """
        validate_positive(tau, "tau")
        num = [gain]
        den = [tau, 1]
        sys = cls.from_tf(num, den, name=name or f"First Order (tau={tau}, K={gain})")
        sys._physical_params = {"tau": tau, "gain": gain}
        return sys

    @classmethod
    def second_order(
        cls,
        wn: float,
        zeta: float,
        gain: float = 1.0,
        name: Optional[str] = None,
    ) -> "System":
        """
        Create a standard second-order system.

        G(s) = gain * wn^2 / (s^2 + 2*zeta*wn*s + wn^2)

        Parameters
        ----------
        wn : float
            Natural frequency (rad/s)
        zeta : float
            Damping ratio (0=undamped, 1=critical, >1=overdamped)
        gain : float
            DC gain (default 1.0)

        Examples
        --------
        >>> sys = System.second_order(wn=2.0, zeta=0.5)  # Underdamped
        >>> sys = System.second_order(wn=2.0, zeta=1.0)  # Critically damped
        """
        validate_positive(wn, "wn")
        validate_damping_ratio(zeta)

        num = [gain * wn**2]
        den = [1, 2 * zeta * wn, wn**2]

        if name is None:
            if zeta < 1:
                damping_type = "underdamped"
            elif zeta == 1:
                damping_type = "critically damped"
            else:
                damping_type = "overdamped"
            name = f"Second Order ({damping_type}, wn={wn}, zeta={zeta})"

        sys = cls.from_tf(num, den, name=name)
        sys._physical_params = {"wn": wn, "zeta": zeta, "gain": gain}
        return sys

    @classmethod
    def integrator(cls, gain: float = 1.0, name: Optional[str] = None) -> "System":
        """Create a pure integrator: G(s) = gain/s"""
        num = [gain]
        den = [1, 0]
        return cls.from_tf(num, den, name=name or f"Integrator (K={gain})")

    @classmethod
    def differentiator(
        cls,
        gain: float = 1.0,
        tau: float = 0.01,
        name: Optional[str] = None,
    ) -> "System":
        """
        Create a filtered differentiator: G(s) = gain*s / (tau*s + 1)

        Uses a first-order filter to make the differentiator proper.

        Parameters
        ----------
        gain : float
            Gain (default 1.0)
        tau : float
            Filter time constant (default 0.01, should be small)
        """
        validate_positive(tau, "tau")
        num = [gain, 0]
        den = [tau, 1]
        return cls.from_tf(num, den, name=name or f"Differentiator (K={gain})")

    # ========== Properties ==========

    @property
    def name(self) -> Optional[str]:
        """System name."""
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def is_stable(self) -> bool:
        """True if all poles have negative real parts."""
        return all(np.real(p) < 0 for p in self.poles)

    @property
    def poles(self) -> np.ndarray:
        """Array of system poles."""
        return ctrl.poles(self._sys)

    @property
    def zeros(self) -> np.ndarray:
        """Array of system zeros."""
        return ctrl.zeros(self._sys)

    @property
    def order(self) -> int:
        """System order (number of poles)."""
        return len(self.poles)

    @property
    def dc_gain(self) -> float:
        """DC gain (gain at s=0)."""
        return float(ctrl.dcgain(self._sys))

    @property
    def is_proper(self) -> bool:
        """True if system is proper (deg(num) <= deg(den))."""
        return len(self.zeros) <= len(self.poles)

    @property
    def is_strictly_proper(self) -> bool:
        """True if strictly proper (deg(num) < deg(den))."""
        return len(self.zeros) < len(self.poles)

    @property
    def is_siso(self) -> bool:
        """True if single-input single-output."""
        return self._sys.ninputs == 1 and self._sys.noutputs == 1

    @property
    def num(self) -> np.ndarray:
        """Numerator coefficients (for transfer functions)."""
        if isinstance(self._sys, ctrl.TransferFunction):
            return np.array(self._sys.num[0][0])
        else:
            tf = ctrl.tf(self._sys)
            return np.array(tf.num[0][0])

    @property
    def den(self) -> np.ndarray:
        """Denominator coefficients (for transfer functions)."""
        if isinstance(self._sys, ctrl.TransferFunction):
            return np.array(self._sys.den[0][0])
        else:
            tf = ctrl.tf(self._sys)
            return np.array(tf.den[0][0])

    # ========== Analysis Methods ==========

    def analyze(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive system analysis.

        Prints a formatted report including:
        - Transfer function display
        - Poles and zeros (with damping/frequency for complex)
        - Stability assessment
        - DC gain
        - Time-domain characteristics (if stable)
        - Frequency-domain margins (if stable)

        Parameters
        ----------
        verbose : bool
            If True, print the report. If False, just return it.

        Returns
        -------
        dict
            Dictionary containing all analysis results
        """
        results = {
            "name": self._name,
            "order": self.order,
            "poles": self.poles,
            "zeros": self.zeros,
            "is_stable": self.is_stable,
            "dc_gain": self.dc_gain if self.is_stable or 0 not in self.poles else None,
        }

        if verbose:
            self._print_analysis(results)

        return results

    def _print_analysis(self, results: Dict[str, Any]):
        """Print formatted analysis report."""
        print()
        header = f" System Analysis"
        if self._name:
            header += f": {self._name}"
        print("=" * 60)
        print(header)
        print("=" * 60)

        # Transfer function
        print("\nTransfer Function:")
        print(self._format_tf())

        # Poles
        print("\nPoles:")
        poles = results["poles"]
        if len(poles) == 0:
            print("  None (static gain)")
        else:
            self._print_pole_info(poles)

        # Zeros
        print("\nZeros:")
        zeros = results["zeros"]
        if len(zeros) == 0:
            print("  None")
        else:
            for z in zeros:
                if np.isreal(z):
                    print(f"  s = {np.real(z):.4f}")
                else:
                    print(f"  s = {np.real(z):.4f} + {np.imag(z):.4f}j")

        # Stability
        print("\nStability:")
        if results["is_stable"]:
            print("  STABLE (all poles in left half-plane)")
        else:
            unstable = [p for p in poles if np.real(p) > 0]
            marginal = [p for p in poles if np.real(p) == 0]
            if len(unstable) > 0:
                print(f"  UNSTABLE ({len(unstable)} pole(s) in right half-plane)")
            elif len(marginal) > 0:
                print(f"  MARGINALLY STABLE ({len(marginal)} pole(s) on imaginary axis)")

        # DC Gain
        if results["dc_gain"] is not None:
            print(f"\nDC Gain: {results['dc_gain']:.4f}")
        else:
            print("\nDC Gain: Undefined (pole at origin or unstable)")

        # Step response info (if stable)
        if results["is_stable"]:
            try:
                step_data = self.step_info()
                print("\nStep Response Characteristics:")
                print(f"  Rise time:     {step_data['rise_time']:.4f} s")
                print(f"  Settling time: {step_data['settling_time']:.4f} s")
                print(f"  Overshoot:     {step_data['overshoot']:.1f}%")
                print(f"  Steady-state:  {step_data['steady_state']:.4f}")
            except Exception:
                pass

            try:
                freq_data = self.frequency_info()
                print("\nFrequency Response Characteristics:")
                if freq_data["gain_margin"] is not None:
                    if np.isinf(freq_data["gain_margin"]):
                        print("  Gain margin:   inf dB")
                    else:
                        print(f"  Gain margin:   {freq_data['gain_margin']:.1f} dB")
                if freq_data["phase_margin"] is not None:
                    print(f"  Phase margin:  {freq_data['phase_margin']:.1f} deg")
                if freq_data["bandwidth"] is not None:
                    print(f"  Bandwidth:     {freq_data['bandwidth']:.2f} rad/s")
            except Exception:
                pass

        print()

    def _print_pole_info(self, poles: np.ndarray):
        """Print detailed pole information."""
        # Group complex conjugate pairs
        processed = set()
        for i, p in enumerate(poles):
            if i in processed:
                continue

            if np.isreal(p):
                real_p = np.real(p)
                tau = -1 / real_p if real_p != 0 else float("inf")
                status = "stable" if real_p < 0 else "unstable" if real_p > 0 else "marginal"
                print(f"  s = {real_p:.4f}")
                print(f"      Real pole, time constant = {abs(tau):.4f} s ({status})")
            else:
                # Find conjugate
                for j, p2 in enumerate(poles):
                    if j > i and np.isclose(p, np.conj(p2)):
                        processed.add(j)
                        break

                sigma = np.real(p)
                omega_d = abs(np.imag(p))
                wn = np.sqrt(sigma**2 + omega_d**2)
                zeta = -sigma / wn if wn != 0 else 0

                status = "stable" if sigma < 0 else "unstable" if sigma > 0 else "marginal"
                print(f"  s = {sigma:.4f} +/- {omega_d:.4f}j")
                print(f"      Complex pair: wn = {wn:.4f} rad/s, zeta = {zeta:.4f} ({status})")

    def _format_tf(self) -> str:
        """Format transfer function for display."""
        num = self.num
        den = self.den

        def poly_str(coeffs):
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
                        terms.append("s")
                    else:
                        terms.append(f"{c:.4g}s")
                else:
                    if c == 1:
                        terms.append(f"s^{power}")
                    else:
                        terms.append(f"{c:.4g}s^{power}")

            return " + ".join(terms) if terms else "0"

        num_str = poly_str(num)
        den_str = poly_str(den)

        # Simple ASCII representation
        width = max(len(num_str), len(den_str)) + 4
        lines = [
            "         " + num_str.center(width),
            "  G(s) = " + "-" * width,
            "         " + den_str.center(width),
        ]
        return "\n".join(lines)

    def step_info(self, settling_threshold: float = 0.02) -> Dict[str, float]:
        """
        Calculate step response characteristics.

        Parameters
        ----------
        settling_threshold : float
            Threshold for settling time (default: 0.02 = 2%)

        Returns
        -------
        dict
            Dictionary with keys: rise_time, settling_time, overshoot,
            peak_time, peak_value, steady_state
        """
        if not self.is_stable:
            raise UnstableSystemError(
                self.poles,
                operation="step_info",
            )

        # Simulate step response
        t, y = ctrl.step_response(self._sys)
        y = np.array(y).flatten()

        # Steady state value
        ss = y[-1]

        # Normalize for calculations
        if abs(ss) < 1e-10:
            return {
                "rise_time": float("nan"),
                "settling_time": float("nan"),
                "overshoot": 0.0,
                "peak_time": 0.0,
                "peak_value": 0.0,
                "steady_state": 0.0,
            }

        y_norm = y / ss

        # Rise time (10% to 90%)
        try:
            t_10 = t[np.where(y_norm >= 0.1)[0][0]]
            t_90 = t[np.where(y_norm >= 0.9)[0][0]]
            rise_time = t_90 - t_10
        except IndexError:
            rise_time = float("nan")

        # Settling time
        settled = np.abs(y_norm - 1) <= settling_threshold
        if np.any(settled):
            # Find last time it was outside the settling band
            outside = np.where(~settled)[0]
            if len(outside) > 0:
                settling_time = t[outside[-1]]
            else:
                settling_time = 0.0
        else:
            settling_time = t[-1]

        # Overshoot
        peak_idx = np.argmax(y)
        peak_value = y[peak_idx]
        peak_time = t[peak_idx]
        overshoot = max(0, (peak_value - ss) / abs(ss) * 100)

        return {
            "rise_time": float(rise_time),
            "settling_time": float(settling_time),
            "overshoot": float(overshoot),
            "peak_time": float(peak_time),
            "peak_value": float(peak_value),
            "steady_state": float(ss),
        }

    def frequency_info(self) -> Dict[str, Optional[float]]:
        """
        Get frequency response characteristics.

        Returns
        -------
        dict
            Dictionary with keys: gain_margin, phase_margin,
            gain_crossover, phase_crossover, bandwidth
        """
        try:
            gm, pm, wcg, wcp = ctrl.margin(self._sys)
            gm_db = 20 * np.log10(gm) if gm is not None and gm > 0 else None
        except Exception:
            gm_db, pm, wcg, wcp = None, None, None, None

        # Calculate bandwidth (-3dB frequency)
        try:
            omega = np.logspace(-2, 4, 1000)
            mag, _, _ = ctrl.frequency_response(self._sys, omega)
            mag = np.abs(np.array(mag).flatten())
            dc = mag[0]
            bw_idx = np.where(mag < dc / np.sqrt(2))[0]
            bandwidth = omega[bw_idx[0]] if len(bw_idx) > 0 else None
        except Exception:
            bandwidth = None

        return {
            "gain_margin": gm_db,
            "phase_margin": pm,
            "gain_crossover": wcg,
            "phase_crossover": wcp,
            "bandwidth": bandwidth,
        }

    # ========== Plotting Methods ==========

    def step(
        self,
        T: Optional[float] = None,
        show_info: bool = True,
        ax: Optional[plt.Axes] = None,
        **kwargs,
    ) -> "System":
        """
        Plot step response.

        Parameters
        ----------
        T : float, optional
            Simulation time (auto-calculated if not provided)
        show_info : bool
            If True, annotate plot with rise time, settling time, etc.
        ax : matplotlib axis, optional
            Axis to plot on (creates new figure if not provided)

        Returns
        -------
        System
            Returns self for chaining
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        t, y = ctrl.step_response(self._sys, T=T)
        y = np.array(y).flatten()

        label = kwargs.pop("label", self._name or "System")
        ax.plot(t, y, label=label, **kwargs)

        if show_info and self.is_stable:
            try:
                info = self.step_info()
                ss = info["steady_state"]

                # Draw steady-state line
                ax.axhline(y=ss, color="gray", linestyle="--", alpha=0.5, label=f"Steady state = {ss:.3f}")

                # Annotate settling time
                ax.axvline(x=info["settling_time"], color="green", linestyle=":", alpha=0.5)

                # Add text box with info
                textstr = (
                    f"Rise time: {info['rise_time']:.3f} s\n"
                    f"Settling time: {info['settling_time']:.3f} s\n"
                    f"Overshoot: {info['overshoot']:.1f}%"
                )
                props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
                ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                        verticalalignment="top", horizontalalignment="right", bbox=props)
            except Exception:
                pass

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Response")
        ax.set_title(f"Step Response{': ' + self._name if self._name else ''}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")

        plt.tight_layout()
        plt.show()

        return self

    def impulse(
        self,
        T: Optional[float] = None,
        ax: Optional[plt.Axes] = None,
        **kwargs,
    ) -> "System":
        """
        Plot impulse response.

        Parameters
        ----------
        T : float, optional
            Simulation time (auto-calculated if not provided)
        ax : matplotlib axis, optional
            Axis to plot on

        Returns
        -------
        System
            Returns self for chaining
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        t, y = ctrl.impulse_response(self._sys, T=T)
        y = np.array(y).flatten()

        label = kwargs.pop("label", self._name or "System")
        ax.plot(t, y, label=label, **kwargs)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Response")
        ax.set_title(f"Impulse Response{': ' + self._name if self._name else ''}")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.show()

        return self

    def bode(
        self,
        omega: Optional[np.ndarray] = None,
        show_margins: bool = True,
        ax: Optional[tuple] = None,
        **kwargs,
    ) -> "System":
        """
        Plot Bode diagram.

        Parameters
        ----------
        omega : array-like, optional
            Frequency range (auto-calculated if not provided)
        show_margins : bool
            If True, mark gain and phase margins on plot
        ax : tuple of axes, optional
            (mag_ax, phase_ax) to plot on

        Returns
        -------
        System
            Returns self for chaining
        """
        if ax is None:
            fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        else:
            ax_mag, ax_phase = ax

        if omega is None:
            omega = np.logspace(-2, 4, 500)

        mag, phase, omega_out = ctrl.frequency_response(self._sys, omega)
        mag = np.abs(np.array(mag).flatten())
        phase = np.angle(np.array(phase).flatten(), deg=True)

        # Convert magnitude to dB
        mag_db = 20 * np.log10(mag)

        ax_mag.semilogx(omega_out, mag_db, **kwargs)
        ax_mag.set_ylabel("Magnitude (dB)")
        ax_mag.set_title(f"Bode Diagram{': ' + self._name if self._name else ''}")
        ax_mag.grid(True, which="both", alpha=0.3)
        ax_mag.axhline(y=0, color="gray", linestyle="-", alpha=0.5)

        ax_phase.semilogx(omega_out, phase, **kwargs)
        ax_phase.set_xlabel("Frequency (rad/s)")
        ax_phase.set_ylabel("Phase (deg)")
        ax_phase.grid(True, which="both", alpha=0.3)
        ax_phase.axhline(y=-180, color="gray", linestyle="-", alpha=0.5)

        if show_margins:
            try:
                freq_info = self.frequency_info()
                if freq_info["gain_crossover"]:
                    wc = freq_info["gain_crossover"]
                    ax_mag.axvline(x=wc, color="red", linestyle="--", alpha=0.5, label=f"Gain crossover: {wc:.2f} rad/s")
                    if freq_info["phase_margin"]:
                        ax_phase.annotate(
                            f"PM = {freq_info['phase_margin']:.1f}°",
                            xy=(wc, -180 + freq_info["phase_margin"]),
                            fontsize=9,
                        )

                if freq_info["phase_crossover"]:
                    wp = freq_info["phase_crossover"]
                    ax_phase.axvline(x=wp, color="blue", linestyle="--", alpha=0.5, label=f"Phase crossover: {wp:.2f} rad/s")
                    if freq_info["gain_margin"]:
                        ax_mag.annotate(
                            f"GM = {freq_info['gain_margin']:.1f} dB",
                            xy=(wp, 0),
                            fontsize=9,
                        )

                ax_mag.legend(loc="upper right", fontsize=8)
                ax_phase.legend(loc="upper right", fontsize=8)
            except Exception:
                pass

        plt.tight_layout()
        plt.show()

        return self

    def nyquist(
        self,
        omega: Optional[np.ndarray] = None,
        show_critical: bool = True,
        ax: Optional[plt.Axes] = None,
        **kwargs,
    ) -> "System":
        """
        Plot Nyquist diagram.

        Parameters
        ----------
        omega : array-like, optional
            Frequency range
        show_critical : bool
            If True, mark the -1+0j point
        ax : matplotlib axis, optional
            Axis to plot on

        Returns
        -------
        System
            Returns self for chaining
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        ctrl.nyquist_plot(self._sys, omega=omega, ax=ax, **kwargs)

        if show_critical:
            ax.plot(-1, 0, "rx", markersize=10, markeredgewidth=2, label="Critical point (-1, 0)")
            ax.legend()

        ax.set_title(f"Nyquist Diagram{': ' + self._name if self._name else ''}")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        plt.tight_layout()
        plt.show()

        return self

    def root_locus(
        self,
        gains: Optional[np.ndarray] = None,
        ax: Optional[plt.Axes] = None,
        **kwargs,
    ) -> "System":
        """
        Plot root locus diagram.

        Parameters
        ----------
        gains : array-like, optional
            Gain values to plot
        ax : matplotlib axis, optional
            Axis to plot on

        Returns
        -------
        System
            Returns self for chaining
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        ctrl.root_locus(self._sys, ax=ax, **kwargs)

        ax.set_title(f"Root Locus{': ' + self._name if self._name else ''}")
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        ax.axvline(x=0, color="gray", linestyle="-", alpha=0.3)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return self

    def pole_zero_map(
        self,
        ax: Optional[plt.Axes] = None,
        show_regions: bool = False,
        **kwargs,
    ) -> "System":
        """
        Plot pole-zero map.

        Parameters
        ----------
        ax : matplotlib axis, optional
            Axis to plot on
        show_regions : bool
            If True, show stability region shading

        Returns
        -------
        System
            Returns self for chaining
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        poles = self.poles
        zeros = self.zeros

        # Plot stability region
        if show_regions:
            ax.axvspan(-10, 0, alpha=0.1, color="green", label="Stable region")
            ax.axvspan(0, 10, alpha=0.1, color="red", label="Unstable region")

        # Plot poles
        if len(poles) > 0:
            ax.plot(np.real(poles), np.imag(poles), "x", markersize=10, markeredgewidth=2, color="red", label="Poles")

        # Plot zeros
        if len(zeros) > 0:
            ax.plot(np.real(zeros), np.imag(zeros), "o", markersize=10, markerfacecolor="none", markeredgewidth=2, color="blue", label="Zeros")

        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        ax.axvline(x=0, color="gray", linestyle="-", alpha=0.3)
        ax.set_xlabel("Real")
        ax.set_ylabel("Imaginary")
        ax.set_title(f"Pole-Zero Map{': ' + self._name if self._name else ''}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect("equal")

        plt.tight_layout()
        plt.show()

        return self

    def all_plots(self) -> "System":
        """
        Show all standard plots in a single figure.

        Creates a 2x2 subplot with:
        - Step response (top-left)
        - Bode plot (top-right, spanning 2 rows)
        - Pole-zero map (bottom-left)

        Returns
        -------
        System
            Returns self for chaining
        """
        fig = plt.figure(figsize=(14, 10))

        # Step response
        ax1 = fig.add_subplot(2, 2, 1)
        t, y = ctrl.step_response(self._sys)
        y = np.array(y).flatten()
        ax1.plot(t, y)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Response")
        ax1.set_title("Step Response")
        ax1.grid(True, alpha=0.3)

        # Bode plot
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 4)
        omega = np.logspace(-2, 4, 500)
        mag, phase, omega_out = ctrl.frequency_response(self._sys, omega)
        mag = np.abs(np.array(mag).flatten())
        phase = np.angle(np.array(phase).flatten(), deg=True)
        mag_db = 20 * np.log10(mag)

        ax2.semilogx(omega_out, mag_db)
        ax2.set_ylabel("Magnitude (dB)")
        ax2.set_title("Bode Diagram")
        ax2.grid(True, which="both", alpha=0.3)

        ax3.semilogx(omega_out, phase)
        ax3.set_xlabel("Frequency (rad/s)")
        ax3.set_ylabel("Phase (deg)")
        ax3.grid(True, which="both", alpha=0.3)

        # Pole-zero map
        ax4 = fig.add_subplot(2, 2, 3)
        poles = self.poles
        zeros = self.zeros
        if len(poles) > 0:
            ax4.plot(np.real(poles), np.imag(poles), "rx", markersize=10, markeredgewidth=2, label="Poles")
        if len(zeros) > 0:
            ax4.plot(np.real(zeros), np.imag(zeros), "bo", markersize=10, markerfacecolor="none", markeredgewidth=2, label="Zeros")
        ax4.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        ax4.axvline(x=0, color="gray", linestyle="-", alpha=0.3)
        ax4.set_xlabel("Real")
        ax4.set_ylabel("Imaginary")
        ax4.set_title("Pole-Zero Map")
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        if self._name:
            fig.suptitle(self._name, fontsize=14)

        plt.tight_layout()
        plt.show()

        return self

    # ========== System Operations ==========

    def with_controller(
        self,
        controller: Union["System", ctrl.TransferFunction],
        feedback: float = 1,
    ) -> "System":
        """
        Create closed-loop system with a controller.

        Parameters
        ----------
        controller : System or TransferFunction
            The controller to use
        feedback : float
            Feedback gain (default: 1 for unity feedback)

        Returns
        -------
        System
            The closed-loop system
        """
        # Handle different controller types
        if isinstance(controller, System):
            ctrl_tf = controller._sys
        elif hasattr(controller, 'as_tf'):
            # PID and other controllers with as_tf method
            ctrl_tf = controller.as_tf()
        else:
            ctrl_tf = controller

        open_loop = ctrl.series(ctrl_tf, self._sys)
        closed_loop = ctrl.feedback(open_loop, feedback)

        name = None
        if self._name:
            name = f"{self._name} (closed-loop)"

        return System(closed_loop, name=name)

    def series(self, other: Union["System", ctrl.TransferFunction]) -> "System":
        """Connect another system in series (cascade)."""
        if isinstance(other, System):
            other_sys = other._sys
        else:
            other_sys = other

        combined = ctrl.series(self._sys, other_sys)
        return System(combined)

    def parallel(self, other: Union["System", ctrl.TransferFunction]) -> "System":
        """Connect another system in parallel."""
        if isinstance(other, System):
            other_sys = other._sys
        else:
            other_sys = other

        combined = ctrl.parallel(self._sys, other_sys)
        return System(combined)

    def feedback(self, H: float = 1) -> "System":
        """Apply feedback with gain H."""
        closed = ctrl.feedback(self._sys, H)
        return System(closed)

    def __mul__(self, other: Union["System", float]) -> "System":
        """Multiply (series connection or gain)."""
        if isinstance(other, (int, float)):
            return System(other * self._sys, name=self._name)
        return self.series(other)

    def __rmul__(self, other: float) -> "System":
        """Right multiply (gain)."""
        return self.__mul__(other)

    def __add__(self, other: "System") -> "System":
        """Add (parallel connection)."""
        return self.parallel(other)

    def __neg__(self) -> "System":
        """Negate (multiply by -1)."""
        return System(-1 * self._sys, name=self._name)

    def __truediv__(self, other: float) -> "System":
        """Divide by scalar."""
        return System(self._sys / other, name=self._name)

    # ========== Comparison ==========

    def compare_with(self, *others: "System", plot: str = "step", labels: Optional[List[str]] = None):
        """
        Compare this system with others on the same plot.

        Parameters
        ----------
        others : System objects
            Systems to compare against
        plot : str
            Type of plot: 'step', 'bode', 'impulse'
        labels : list of str, optional
            Labels for each system
        """
        all_systems = [self] + list(others)

        if labels is None:
            labels = [s._name or f"System {i+1}" for i, s in enumerate(all_systems)]

        if plot == "step":
            fig, ax = plt.subplots(figsize=(10, 6))
            for sys, label in zip(all_systems, labels):
                t, y = ctrl.step_response(sys._sys)
                y = np.array(y).flatten()
                ax.plot(t, y, label=label)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Response")
            ax.set_title("Step Response Comparison")
            ax.grid(True, alpha=0.3)
            ax.legend()

        elif plot == "impulse":
            fig, ax = plt.subplots(figsize=(10, 6))
            for sys, label in zip(all_systems, labels):
                t, y = ctrl.impulse_response(sys._sys)
                y = np.array(y).flatten()
                ax.plot(t, y, label=label)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Response")
            ax.set_title("Impulse Response Comparison")
            ax.grid(True, alpha=0.3)
            ax.legend()

        elif plot == "bode":
            fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            omega = np.logspace(-2, 4, 500)
            for sys, label in zip(all_systems, labels):
                mag, phase, omega_out = ctrl.frequency_response(sys._sys, omega)
                mag = np.abs(np.array(mag).flatten())
                phase = np.angle(np.array(phase).flatten(), deg=True)
                mag_db = 20 * np.log10(mag)
                ax_mag.semilogx(omega_out, mag_db, label=label)
                ax_phase.semilogx(omega_out, phase, label=label)

            ax_mag.set_ylabel("Magnitude (dB)")
            ax_mag.set_title("Bode Diagram Comparison")
            ax_mag.grid(True, which="both", alpha=0.3)
            ax_mag.legend()

            ax_phase.set_xlabel("Frequency (rad/s)")
            ax_phase.set_ylabel("Phase (deg)")
            ax_phase.grid(True, which="both", alpha=0.3)
            ax_phase.legend()

        plt.tight_layout()
        plt.show()

    # ========== Display ==========

    def __repr__(self) -> str:
        """Informative string representation."""
        info = f"System("
        if self._name:
            info += f"'{self._name}', "
        info += f"order={self.order}, "
        info += f"stable={self.is_stable}"
        info += ")"
        return info

    def __str__(self) -> str:
        """String representation with transfer function."""
        lines = [repr(self)]
        lines.append(self._format_tf())
        return "\n".join(lines)

    def show_equation(self, format: str = "ascii") -> str:
        """
        Display the transfer function equation.

        Parameters
        ----------
        format : str
            'ascii' for terminal display
        """
        tf_str = self._format_tf()
        print(tf_str)
        return tf_str

    def _repr_html_(self) -> str:
        """
        HTML representation for Jupyter notebooks.

        Renders the transfer function using MathJax for beautiful equations.
        """
        # Build LaTeX for the transfer function
        latex = self._format_tf_latex()

        # Stability indicator
        if self.is_stable:
            stability = '<span style="color: green;">Stable</span>'
        else:
            stability = '<span style="color: red;">Unstable</span>'

        # Build HTML output
        html = f'''
        <div style="font-family: sans-serif; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background: #f9f9f9;">
            <div style="font-weight: bold; margin-bottom: 10px;">
                {self._name or 'System'} (order={self.order}, {stability})
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
        """
        LaTeX representation for Jupyter notebooks.

        Returns a LaTeX string of the transfer function.
        """
        return f"${self._format_tf_latex()}$"

    def _format_tf_latex(self) -> str:
        """Format transfer function as LaTeX."""
        num = self.num
        den = self.den

        def poly_latex(coeffs, var="s"):
            terms = []
            order = len(coeffs) - 1
            for i, c in enumerate(coeffs):
                power = order - i
                if abs(c) < 1e-10:
                    continue

                # Format coefficient
                if abs(c - round(c)) < 1e-10:
                    c_str = str(int(round(c)))
                elif abs(c) < 0.001 or abs(c) > 1000:
                    c_str = f"{c:.2e}"
                else:
                    c_str = f"{c:.4g}"

                # Build term
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

            # Join with proper signs
            result = terms[0]
            for term in terms[1:]:
                if term.startswith("-"):
                    result += f" - {term[1:]}"
                else:
                    result += f" + {term}"
            return result

        num_latex = poly_latex(num)
        den_latex = poly_latex(den)

        return f"G(s) = \\frac{{{num_latex}}}{{{den_latex}}}"

    def _format_poles_html(self) -> str:
        """Format poles for HTML display."""
        poles = self.poles
        if len(poles) == 0:
            return "None"

        pole_strs = []
        processed = set()

        for i, p in enumerate(poles):
            if i in processed:
                continue

            if np.isreal(p):
                pole_strs.append(f"{np.real(p):.3g}")
            else:
                # Find conjugate pair
                for j, p2 in enumerate(poles):
                    if j > i and np.isclose(p, np.conj(p2)):
                        processed.add(j)
                        break
                pole_strs.append(f"{np.real(p):.3g}±{abs(np.imag(p)):.3g}j")

        return ", ".join(pole_strs)
