"""Convenience plotting wrappers."""

import matplotlib.pyplot as plt
import numpy as np
from control import step_response, bode, nyquist_plot, root_locus

def step(sys, title=None):
    """Plot step response."""
    t, y = step_response(sys)
    plt.figure()
    plt.plot(t, y)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    if title:
        plt.title(title)
    plt.grid(True)
    plt.show()


def bode_plot(sys, margins=False):
    """Plot Bode magnitude and phase."""
    mag, phase, omega = bode(sys, dB=True, plot=False)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.semilogx(omega, 20*np.log10(mag))
    ax1.set_ylabel('Magnitude [dB]')
    ax1.grid(True)
    ax2.semilogx(omega, phase * 180 / np.pi)
    ax2.set_ylabel('Phase [deg]')
    ax2.set_xlabel('Frequency [rad/s]')
    ax2.grid(True)
    if margins:
        from control import margin
        gm, pm, wg, wp = margin(sys)
        ax1.axvline(wg, color='r', linestyle='--')
        ax2.axvline(wp, color='r', linestyle='--')
    plt.show()

def nyquist(sys):
    """Plot Nyquist diagram."""
    nyquist_plot(sys)
    plt.show()


def root_locus_plot(sys):
    """Plot root locus diagram."""
    root_locus(sys)
    plt.show()