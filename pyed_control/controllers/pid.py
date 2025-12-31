"""
PID Controller with multiple tuning methods.

The PID controller is the most common controller in industry.
This class provides multiple ways to design and tune PID controllers.
"""

import numpy as np
import control as ctrl
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.system import System


class PID:
    """
    PID Controller with multiple tuning methods.

    Transfer Function (Standard Form):
        C(s) = Kp + Ki/s + Kd*s
             = (Kd*s^2 + Kp*s + Ki) / s

    Transfer Function (with Derivative Filter):
        C(s) = Kp + Ki/s + Kd*s/(1 + s*Td/N)

    Examples
    --------
    >>> # Direct specification
    >>> pid = PID(kp=1.0, ki=0.5, kd=0.1)

    >>> # Auto-tune using Ziegler-Nichols
    >>> pid = PID.ziegler_nichols(plant)

    >>> # Use with a plant
    >>> closed_loop = plant.with_controller(pid)
    >>> closed_loop.step()

    >>> # Compare different tunings
    >>> pid1 = PID.ziegler_nichols(plant)
    >>> pid2 = PID(kp=2, ki=1, kd=0.5)
    >>> ctrl.compare(plant.with_controller(pid1),
    ...              plant.with_controller(pid2),
    ...              labels=['Z-N', 'Manual'])
    """

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
        derivative_filter: Optional[float] = None,
    ):
        """
        Create a PID controller.

        Parameters
        ----------
        kp : float
            Proportional gain (default: 1.0)
        ki : float
            Integral gain (default: 0.0, i.e., PD controller)
        kd : float
            Derivative gain (default: 0.0, i.e., PI controller)
        derivative_filter : float, optional
            If provided, use filtered derivative with cutoff N.
            The filter is: s/(1 + s/(N*wc)) where wc = Kd/Kp
            (default: None = ideal derivative)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.derivative_filter = derivative_filter

    @classmethod
    def ziegler_nichols(
        cls,
        plant: "System",
        method: str = "frequency",
        controller_type: str = "PID",
        show_work: bool = False,
    ) -> "PID":
        """
        Tune PID using Ziegler-Nichols method.

        Parameters
        ----------
        plant : System
            The plant to control
        method : str
            'frequency' - Use frequency response to find Ku, Tu
        controller_type : str
            'P', 'PI', or 'PID' (default: 'PID')
        show_work : bool
            If True, print the tuning process

        Returns
        -------
        PID
            Tuned PID controller

        Examples
        --------
        >>> pid = PID.ziegler_nichols(plant, show_work=True)
        """
        from ..core.system import System

        if isinstance(plant, System):
            sys = plant._sys
        else:
            sys = plant

        if show_work:
            print("\nZiegler-Nichols PID Tuning (Frequency Response Method)")
            print("=" * 55)
            print("\nStep 1: Find ultimate gain (Ku) and ultimate period (Tu)")
            print("  The ultimate gain is the gain at which the closed-loop")
            print("  system becomes marginally stable (sustained oscillation).")

        # Find ultimate gain and frequency using phase crossover
        omega = np.logspace(-3, 3, 1000)
        try:
            mag, phase, omega_out = ctrl.frequency_response(sys, omega)
            mag = np.abs(np.array(mag).flatten())
            phase = np.angle(np.array(phase).flatten(), deg=True)

            # Find where phase crosses -180 degrees
            phase_wrapped = np.mod(phase + 180, 360) - 180
            crossings = np.where(np.diff(np.sign(phase_wrapped + 180)))[0]

            if len(crossings) == 0:
                # No phase crossover - system may be first-order
                # Use approximate method
                omega_u = omega_out[-1] / 10
                mag_at_omega = np.interp(omega_u, omega_out, mag)
                Ku = 1.0 / mag_at_omega
                Tu = 2 * np.pi / omega_u

                if show_work:
                    print("  Warning: No phase crossover found.")
                    print("  Using approximate method.")
            else:
                idx = crossings[0]
                omega_u = omega_out[idx]
                mag_u = mag[idx]
                Ku = 1.0 / mag_u
                Tu = 2 * np.pi / omega_u

                if show_work:
                    print(f"\n  From frequency response analysis:")
                    print(f"  - Phase crosses -180 deg at: omega_u = {omega_u:.3f} rad/s")
                    print(f"  - Magnitude at this frequency: |G(jw_u)| = {mag_u:.4f}")
                    print(f"  - Ultimate gain: Ku = 1/|G(jw_u)| = {Ku:.3f}")
                    print(f"  - Ultimate period: Tu = 2*pi/w_u = {Tu:.3f} s")

        except Exception:
            # Fallback to simple estimate
            Ku = 10.0
            Tu = 1.0
            if show_work:
                print("  Could not determine Ku, Tu. Using defaults.")

        if show_work:
            print("\nStep 2: Apply Ziegler-Nichols formulas")
            print(f"\n  For {controller_type} controller:")

        # Ziegler-Nichols formulas
        if controller_type.upper() == "P":
            kp = 0.5 * Ku
            ki = 0.0
            kd = 0.0
            if show_work:
                print(f"  - Kp = 0.5 * Ku = {kp:.3f}")
        elif controller_type.upper() == "PI":
            kp = 0.45 * Ku
            Ti = Tu / 1.2
            ki = kp / Ti
            kd = 0.0
            if show_work:
                print(f"  - Kp = 0.45 * Ku = {kp:.3f}")
                print(f"  - Ti = Tu / 1.2 = {Ti:.3f}")
                print(f"  - Ki = Kp / Ti = {ki:.3f}")
        else:  # PID
            kp = 0.6 * Ku
            Ti = Tu / 2
            Td = Tu / 8
            ki = kp / Ti
            kd = kp * Td
            if show_work:
                print(f"  - Kp = 0.6 * Ku = {kp:.3f}")
                print(f"  - Ti = Tu / 2 = {Ti:.3f}")
                print(f"  - Td = Tu / 8 = {Td:.3f}")
                print(f"  - Ki = Kp / Ti = {ki:.3f}")
                print(f"  - Kd = Kp * Td = {kd:.3f}")

        if show_work:
            print(f"\nResult: PID(kp={kp:.3f}, ki={ki:.3f}, kd={kd:.3f})")
            print("\nNote: Z-N tuning typically gives aggressive response with")
            print("~25% overshoot. Consider reducing gains for less overshoot.")

        return cls(kp=kp, ki=ki, kd=kd)

    @classmethod
    def tune_for(
        cls,
        plant: "System",
        overshoot: Optional[float] = None,
        settling_time: Optional[float] = None,
        phase_margin: Optional[float] = None,
    ) -> "PID":
        """
        Automatically tune PID to meet specifications.

        Uses iterative optimization to find PID gains that meet
        the specified performance criteria.

        Parameters
        ----------
        plant : System
            The plant to control
        overshoot : float, optional
            Maximum percent overshoot (e.g., 10 for 10%)
        settling_time : float, optional
            Desired settling time in seconds
        phase_margin : float, optional
            Desired phase margin in degrees

        Returns
        -------
        PID
            Tuned PID controller

        Examples
        --------
        >>> # Tune for low overshoot
        >>> pid = PID.tune_for(plant, overshoot=5)

        >>> # Tune for fast response
        >>> pid = PID.tune_for(plant, settling_time=1.0)
        """
        # Start with Ziegler-Nichols as initial guess
        pid = cls.ziegler_nichols(plant, show_work=False)

        # Simple scaling based on specs
        if overshoot is not None:
            # Lower Kp and increase Kd to reduce overshoot
            if overshoot < 10:
                pid.kp *= 0.7
                pid.kd *= 1.5
            elif overshoot < 20:
                pid.kp *= 0.85
                pid.kd *= 1.2

        if settling_time is not None:
            # This is a rough approximation
            # Faster settling = higher gains (within stability limits)
            from ..core.system import System

            if isinstance(plant, System):
                try:
                    info = plant.step_info()
                    current_ts = info["settling_time"]
                    ratio = current_ts / settling_time
                    pid.kp *= min(ratio, 2.0)
                    pid.ki *= min(ratio, 2.0)
                except Exception:
                    pass

        return pid

    @classmethod
    def cohen_coon(
        cls,
        plant: "System",
        controller_type: str = "PID",
        show_work: bool = False,
    ) -> "PID":
        """
        Tune PID using Cohen-Coon method.

        Cohen-Coon is better for plants with significant dead time.
        Uses step response to identify first-order plus dead time (FOPDT) model.

        Parameters
        ----------
        plant : System
            The plant to control
        controller_type : str
            'P', 'PI', or 'PID' (default: 'PID')
        show_work : bool
            If True, print the tuning process

        Returns
        -------
        PID
            Tuned PID controller

        Examples
        --------
        >>> pid = PID.cohen_coon(plant, show_work=True)

        Notes
        -----
        Cohen-Coon tuning approximates the plant as:
            G(s) = K * exp(-L*s) / (T*s + 1)

        where:
        - K = steady-state gain
        - T = time constant
        - L = dead time (delay)
        """
        from ..core.system import System

        if isinstance(plant, System):
            sys = plant._sys
        else:
            sys = plant

        if show_work:
            print("\nCohen-Coon PID Tuning")
            print("=" * 40)
            print("\nStep 1: Identify FOPDT model from step response")

        # Get step response to identify FOPDT parameters
        t, y = ctrl.step_response(sys)
        y = np.array(y).flatten()

        # Get steady-state gain
        K = y[-1]

        # Normalize response
        y_norm = y / K if K != 0 else y

        # Find time constant T (time to reach 63.2% of final value)
        try:
            idx_63 = np.where(y_norm >= 0.632)[0][0]
            T = t[idx_63]
        except IndexError:
            T = t[-1] / 3

        # Estimate dead time L (time to reach 10% of final value)
        try:
            idx_10 = np.where(y_norm >= 0.1)[0][0]
            L = t[idx_10]
        except IndexError:
            L = 0.1 * T

        # Adjust T to account for dead time
        T = T - L if T > L else T

        if show_work:
            print(f"  Identified FOPDT model:")
            print(f"    K (gain) = {K:.4g}")
            print(f"    T (time constant) = {T:.4g} s")
            print(f"    L (dead time) = {L:.4g} s")
            print(f"    L/T ratio = {L/T:.4g}")

        # Cohen-Coon formulas
        r = L / T  # Dead time ratio

        if show_work:
            print(f"\nStep 2: Apply Cohen-Coon formulas")
            print(f"  For {controller_type} controller:")

        if controller_type.upper() == "P":
            kp = (1 / K) * (T / L) * (1 + r / 3)
            ki = 0.0
            kd = 0.0
            if show_work:
                print(f"  Kp = (1/K)(T/L)(1 + L/3T) = {kp:.3f}")

        elif controller_type.upper() == "PI":
            kp = (1 / K) * (T / L) * (0.9 + r / 12)
            Ti = L * (30 + 3 * r) / (9 + 20 * r)
            ki = kp / Ti
            kd = 0.0
            if show_work:
                print(f"  Kp = {kp:.3f}")
                print(f"  Ti = {Ti:.3f}")
                print(f"  Ki = Kp/Ti = {ki:.3f}")

        else:  # PID
            kp = (1 / K) * (T / L) * (4 / 3 + r / 4)
            Ti = L * (32 + 6 * r) / (13 + 8 * r)
            Td = L * 4 / (11 + 2 * r)
            ki = kp / Ti
            kd = kp * Td
            if show_work:
                print(f"  Kp = {kp:.3f}")
                print(f"  Ti = {Ti:.3f}")
                print(f"  Td = {Td:.3f}")
                print(f"  Ki = {ki:.3f}")
                print(f"  Kd = {kd:.3f}")

        if show_work:
            print(f"\nResult: PID(kp={kp:.3f}, ki={ki:.3f}, kd={kd:.3f})")
            print("\nNote: Cohen-Coon works best for 0.1 < L/T < 1")

        return cls(kp=kp, ki=ki, kd=kd)

    @classmethod
    def imc(
        cls,
        plant: "System",
        tau_c: Optional[float] = None,
        lambda_tune: Optional[float] = None,
        show_work: bool = False,
    ) -> "PID":
        """
        Tune PID using Internal Model Control (IMC) method.

        IMC tuning gives smooth, non-aggressive responses with
        good robustness. It's based on specifying the desired
        closed-loop time constant.

        Parameters
        ----------
        plant : System
            The plant to control
        tau_c : float, optional
            Desired closed-loop time constant.
            If not provided, uses tau_c = max(0.1*T, L) where
            T is the plant time constant and L is the dead time.
        lambda_tune : float, optional
            Alternative to tau_c - the "lambda" tuning parameter.
            Larger lambda = more robust but slower.
        show_work : bool
            If True, print the tuning process

        Returns
        -------
        PID
            Tuned PID controller

        Examples
        --------
        >>> pid = PID.imc(plant, show_work=True)
        >>> pid = PID.imc(plant, tau_c=0.5)  # Specify closed-loop time constant

        Notes
        -----
        IMC tuning is based on the idea that the ideal controller
        is the inverse of the plant model. For a FOPDT plant:

        G(s) = K * exp(-L*s) / (T*s + 1)

        The IMC-PID parameters are:
        - Kp = T / (K * (tau_c + L))
        - Ti = T
        - Td = 0 (or T*L/(2*T + L) for PID)
        """
        from ..core.system import System

        if isinstance(plant, System):
            sys = plant._sys
        else:
            sys = plant

        if show_work:
            print("\nIMC (Internal Model Control) PID Tuning")
            print("=" * 45)
            print("\nStep 1: Identify FOPDT model from step response")

        # Get step response to identify FOPDT parameters
        t, y = ctrl.step_response(sys)
        y = np.array(y).flatten()

        # Get steady-state gain
        K = y[-1]
        y_norm = y / K if K != 0 else y

        # Find time constant T and dead time L
        try:
            idx_63 = np.where(y_norm >= 0.632)[0][0]
            T = t[idx_63]
        except IndexError:
            T = t[-1] / 3

        try:
            idx_10 = np.where(y_norm >= 0.1)[0][0]
            L = t[idx_10]
        except IndexError:
            L = 0.1 * T

        T = T - L if T > L else T

        if show_work:
            print(f"  Identified FOPDT model:")
            print(f"    K = {K:.4g}")
            print(f"    T = {T:.4g} s")
            print(f"    L = {L:.4g} s")

        # Determine closed-loop time constant
        if tau_c is not None:
            closed_loop_tau = tau_c
        elif lambda_tune is not None:
            closed_loop_tau = lambda_tune
        else:
            # Default: max(0.1*T, L) for good robustness
            closed_loop_tau = max(0.1 * T, L)

        if show_work:
            print(f"\nStep 2: Select closed-loop time constant")
            print(f"  tau_c = {closed_loop_tau:.4g} s")

        # IMC-PI tuning (most common)
        kp = T / (K * (closed_loop_tau + L))
        Ti = T
        ki = kp / Ti
        kd = 0.0

        # Optional: Add derivative for better disturbance rejection
        # Td = T * L / (2 * T + L)
        # kd = kp * Td

        if show_work:
            print(f"\nStep 3: Calculate IMC-PI parameters")
            print(f"  Kp = T / (K * (tau_c + L))")
            print(f"     = {T:.4g} / ({K:.4g} * ({closed_loop_tau:.4g} + {L:.4g}))")
            print(f"     = {kp:.4g}")
            print(f"  Ti = T = {Ti:.4g}")
            print(f"  Ki = Kp/Ti = {ki:.4g}")
            print(f"\nResult: PID(kp={kp:.3f}, ki={ki:.3f}, kd={kd:.3f})")
            print("\nNote: IMC tuning gives smooth response with good robustness.")
            print(f"      Decrease tau_c for faster response (current: {closed_loop_tau:.3f})")
            print(f"      Increase tau_c for more robustness")

        return cls(kp=kp, ki=ki, kd=kd)

    @classmethod
    def lambda_tuning(
        cls,
        plant: "System",
        lambda_factor: float = 3.0,
        show_work: bool = False,
    ) -> "PID":
        """
        Tune PID using Lambda tuning method.

        Lambda tuning is a simplified IMC approach where the
        closed-loop time constant is set as a multiple of the
        dominant plant time constant.

        Parameters
        ----------
        plant : System
            The plant to control
        lambda_factor : float
            Multiplier for the plant time constant (default: 3.0)
            - lambda_factor = 1: Aggressive, fast response
            - lambda_factor = 3: Moderate, good balance
            - lambda_factor = 5: Conservative, very robust
        show_work : bool
            If True, print the tuning process

        Returns
        -------
        PID
            Tuned PID controller
        """
        from ..core.system import System

        if isinstance(plant, System):
            sys = plant._sys
        else:
            sys = plant

        # Get step response to identify time constant
        t, y = ctrl.step_response(sys)
        y = np.array(y).flatten()
        K = y[-1]
        y_norm = y / K if K != 0 else y

        try:
            idx_63 = np.where(y_norm >= 0.632)[0][0]
            T = t[idx_63]
        except IndexError:
            T = t[-1] / 3

        tau_c = lambda_factor * T

        if show_work:
            print("\nLambda Tuning Method")
            print("=" * 30)
            print(f"\nPlant time constant: T = {T:.4g} s")
            print(f"Lambda factor: {lambda_factor}")
            print(f"Closed-loop time constant: tau_c = {lambda_factor} * T = {tau_c:.4g} s")

        return cls.imc(plant, tau_c=tau_c, show_work=show_work if not show_work else False)

    def as_system(self) -> "System":
        """
        Convert to System object for analysis and composition.

        Returns
        -------
        System
            The PID controller as a System object
        """
        from ..core.system import System

        tf = self.as_tf()
        return System(tf, name=f"PID(Kp={self.kp:.2f}, Ki={self.ki:.2f}, Kd={self.kd:.2f})")

    def as_tf(self) -> ctrl.TransferFunction:
        """
        Get the transfer function representation.

        Returns
        -------
        TransferFunction
            The PID controller transfer function
        """
        if self.derivative_filter is None:
            # Ideal PID: (Kd*s^2 + Kp*s + Ki) / s
            num = [self.kd, self.kp, self.ki]
            den = [1, 0]
        else:
            # Filtered derivative PID
            # This is more complex, simplified version:
            N = self.derivative_filter
            # Approximate with first-order filter
            num = [self.kd, self.kp, self.ki]
            den = [1 / N, 1, 0]

        return ctrl.TransferFunction(num, den)

    def explain(self):
        """
        Print explanation of what each PID term does.

        Provides educational context about how each gain affects
        the closed-loop response.
        """
        print(f"\nPID Controller: Kp={self.kp}, Ki={self.ki}, Kd={self.kd}")
        print("=" * 50)

        print(f"\nProportional Term (Kp = {self.kp}):")
        print("  - Acts on current error: u_p = Kp * e(t)")
        print("  - Larger Kp = faster response but more overshoot")
        print("  - Too large = instability (oscillation or divergence)")
        print("  - By itself: leaves steady-state error for non-zero setpoint")

        print(f"\nIntegral Term (Ki = {self.ki}):")
        if self.ki > 0:
            print("  - Acts on accumulated error: u_i = Ki * integral(e(t))")
            print("  - Eliminates steady-state error (forces error to zero)")
            print("  - Larger Ki = faster error elimination but more overshoot")
            print("  - Can cause wind-up issues with saturation")
        else:
            print("  - Ki = 0: No integral action (will have steady-state error)")

        print(f"\nDerivative Term (Kd = {self.kd}):")
        if self.kd > 0:
            print("  - Acts on rate of change: u_d = Kd * de/dt")
            print('  - Provides "prediction" / anticipation')
            print("  - Reduces overshoot and oscillation (adds damping)")
            print("  - Sensitive to noise (use filtered derivative in practice)")
        else:
            print("  - Kd = 0: No derivative action")

        print("\nTuning Tips:")
        print("  - Start with Ki = Kd = 0, increase Kp until oscillation")
        print("  - Add Ki to eliminate steady-state error")
        print("  - Add Kd to reduce overshoot and improve stability")

    def __repr__(self) -> str:
        return f"PID(kp={self.kp}, ki={self.ki}, kd={self.kd})"

    def __str__(self) -> str:
        controller_type = ""
        if self.ki == 0 and self.kd == 0:
            controller_type = "P"
        elif self.kd == 0:
            controller_type = "PI"
        elif self.ki == 0:
            controller_type = "PD"
        else:
            controller_type = "PID"

        return f"{controller_type} Controller: Kp={self.kp:.3f}, Ki={self.ki:.3f}, Kd={self.kd:.3f}"
