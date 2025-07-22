"""Utilities to convert MATLAB models to python-control."""

import scipy.io
from control import ss


def convert_matlab_ss(matfile, var='sys'):
    """Load a MATLAB .mat file containing A, B, C, D and return StateSpace."""
    data = scipy.io.loadmat(matfile)
    A = data[var]['A'] if isinstance(data[var], dict) else data['A']
    B = data[var]['B'] if isinstance(data[var], dict) else data['B']
    C = data[var]['C'] if isinstance(data[var], dict) else data['C']
    D = data[var]['D'] if isinstance(data[var], dict) else data['D']
    return ss(A, B, C, D)
