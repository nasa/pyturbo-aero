import numpy.typing as npt
import numpy as np 

def xr_to_mprime(xr:npt.NDArray):
    """Converts a numpy array n x 2 containing xr to mprime 

    Args:
        xr (npt.NDArray): _description_
    """
    dx = np.diff(xr[:,0])
    dr = np.diff(xr[:,1])
    # mprime
    arc_len = np.cumsum(np.sqrt(dr**2 + dx**2))
    mp = [2/(xr[i,1]+xr[i-1,1])*np.sqrt(dr[i-1]**2 + dx[i-1]**2) for i in range(1,len(xr[:,1]))]
    mp = np.hstack([[0],np.cumsum(mp)])
    arc_len = np.hstack([[0],arc_len])

    return mp,arc_len
