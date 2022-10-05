import numpy as np
               

def exp_ratio(ratio,npoints,maxvalue=1):
    """
        Expansion Ratio
        Inputs:
            ratio - 1 = no epxansion simple linspace 1.2 or 1.4 probably max
            npoints - number of points to use
            max - max value 
    """
    t = np.zeros(npoints)
    s = 0
    for i in range(npoints):
        s = s+ratio**i
    
    dt = 1/s
    t[0] = dt
    for i in range(1,npoints):
       t[i] = t[i-1] + dt*ratio
       dt = dt * ratio
    
    t = maxvalue*t
    return t