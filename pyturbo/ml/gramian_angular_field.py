import numpy as np 
import math
from typing import List

def gramian_angular_field(t:List[float],x:List[float]):
    '''

    '''

    x_norm = (x-max(x) + x-min(x))/(max(x)-min(x))
    # Polar coordinates
    phi = np.acos(x)
    r = t/len(t)

    G = np.zeros(shape=(len(x),len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            G[i,j] = x_norm[i]*x_norm[j] - math.sqrt(1-x_norm[i]*x_norm[i])*math.sqrt(1-x_norm[j]*x_norm[j])
    return phi, r,G

def markov():
    pass


