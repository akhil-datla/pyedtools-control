"""Utilities to convert MATLAB models to python-control."""

import scipy.io
from control import ss, tf


def convert_matlab_ss(matfile, var='sys'):
    """Load a MATLAB .mat file containing A, B, C, D and return StateSpace."""
    data = scipy.io.loadmat(matfile)
    A = data[var]['A'] if isinstance(data[var], dict) else data['A']
    B = data[var]['B'] if isinstance(data[var], dict) else data['B']
    C = data[var]['C'] if isinstance(data[var], dict) else data['C']
    D = data[var]['D'] if isinstance(data[var], dict) else data['D']
    return ss(A, B, C, D)

def convert_matlab_tf(matfile, num_var='num', den_var='den'):
    """Load numerator and denominator arrays from MATLAB .mat file."""
    data = scipy.io.loadmat(matfile)
    num = data[num_var]
    den = data[den_var]
    return tf(num[0][0], den[0][0]) if num.ndim == 2 else tf(num, den)

