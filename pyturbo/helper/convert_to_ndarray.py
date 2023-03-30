import numpy as np
import math
def convert_to_ndarray(t):
    """
        converts a scalar or list to numpy array 
    """

    if type(t) is not np.ndarray and type(t) is not list: # Scalar
        t = np.array([t],dtype=float)
    elif (type(t) is list):
        t = np.array(t,dtype=float)
    return t

def cosd(val):
    return np.cos(math.pi/180 * val)

def sind(val):
    return np.sin(math.pi/180 *val)

def tand(y,x):
    return np.tan2(math.pi/180 *y,math.pi/180 *x)