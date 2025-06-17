import numpy as np
import numpy.typing as npt 

def exp_ratio(ratio:float,npoints:int,maxvalue:float=1,flip_direction:bool=False) -> npt.NDArray:
    """_summary_

    Args:
        ratio (float): _description_
        npoints (int): _description_
        maxvalue (float, optional): _description_. Defaults to 1.
        flip_direction (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    t = np.zeros(npoints)
    s = 0
    for p in range(npoints-1):
        s+=ratio**(p+1)
    a = 1/s 
    t[0] = 0
    
    if flip_direction:    
        for i in range(1,npoints):
            t[i] = t[i-1]+a*ratio**(npoints-i)
    else:    
        for i in range(1,npoints):
            t[i] = t[i-1] + a*ratio**i
    
    t = maxvalue*t
    return t