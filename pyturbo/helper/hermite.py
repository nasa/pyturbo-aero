import numpy as np 

def hermite01(n:int):
    """Hermite polynomial with grouping at 0 

    Args:
        n (int): number of points 

    Returns:
        (np.ndarray): array of grouping 
    """
    t = np.linspace(0,1,n)
    return -2*t**3 + 3*t**2

def hermite10(n:int):
    """Hermite Polynomial with grouping near 1

    Args:
        n (int): number of points 

    Returns:
        (np.ndarray): array of grouping 
    """
    t = np.linspace(0,1,n)
    return 2*t**3 - 3*t**3 + 1