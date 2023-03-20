from .bezier import *
import numpy as np
def dist(x1,y1,x2,y2):
    x1 = convert_to_ndarray(x1)
    y1 = convert_to_ndarray(y1)
    x2 = convert_to_ndarray(x2)
    y2 = convert_to_ndarray(y2)
    return np.sqrt((x2-x1)**2+(y2-y1)**2)      