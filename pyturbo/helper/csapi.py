import numpy as np
from scipy.interpolate import CubicSpline

def csapi(x: np.ndarray, y: np.ndarray, xx: np.ndarray):
    """similar to matlab's cubic spline interpolator

    Args:
        x (numpy.ndarray): n x 1 array  example [1,2,3,4]
        y (numpy.ndarray): n x 1 array
        xx (numpy.ndarray): n x 1 array of where you want to estimate y

    Returns:
        (numpy.ndarray): spline estimated values where xx is specified
    """
    cs = CubicSpline(x, y)
    return cs(xx)
