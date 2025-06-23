
from scipy.interpolate import splprep, splev
import numpy as np 
import numpy.typing as npt

def curvature_3d(dx, dy, dz, ddx, ddy, ddz):
    # Compute curvature from first and second derivatives
    cross = np.cross(np.stack((dx, dy, dz), axis=1),
                     np.stack((ddx, ddy, ddz), axis=1))
    num = np.linalg.norm(cross, axis=1)
    denom = (dx**2 + dy**2 + dz**2)**1.5
    with np.errstate(divide='ignore', invalid='ignore'):
        kappa = np.where(denom > 0, num / denom, 0.0)
    return kappa

def resample_curve(curve:npt.NDArray, M:int=100):
    _, idx = np.unique(curve, axis=0, return_index=True)
    if len(idx) < len(curve):
        curve = curve[np.sort(idx)]
    tck, _ = splprep(curve.T, s=0, per=0)
    u_new = np.linspace(0, 1, M)
    return np.stack(splev(u_new, tck), axis=1) # type: ignore

def resample_by_curvature(points: np.ndarray, N: int, smoothing=0.0) -> np.ndarray:
    """
    Resample a 3D curve to concentrate points in high-curvature regions.

    Parameters:
        points: (M, 3) array of input points (x, y, z).
        N: Number of output points to sample.
        smoothing: Smoothing factor for the spline (0 = interpolate).

    Returns:
        (N, 3) array of resampled points.
    """
    if points.shape[1] != 3:
        raise ValueError("Input points must be of shape (M, 3)")
    
    # Fit a parametric spline
    tck, u = splprep(points.T, s=smoothing)

    # Sample finely along the curve
    u_fine = np.linspace(0, 1, 1000)
    x, y, z = splev(u_fine, tck)
    dx, dy, dz = splev(u_fine, tck, der=1)
    ddx, ddy, ddz = splev(u_fine, tck, der=2)

    # Compute curvature at each sampled point
    kappa = curvature_3d(dx, dy, dz, ddx, ddy, ddz)

    # Normalize curvature to a probability distribution
    density = kappa + 1e-3  # add small value to ensure flat regions get some weight
    density /= np.sum(density)

    # Create cumulative distribution function (CDF)
    cdf = np.cumsum(density)
    cdf /= cdf[-1]  # Normalize

    # Invert the CDF to get u values for sampling
    u_resampled = np.interp(np.linspace(0, 1, N), cdf, u_fine)

    # Evaluate spline at resampled u values
    x_new, y_new, z_new = splev(u_resampled, tck)
    # Get the derivative 
    # dx, dy, dz = splev(u_resampled, tck, der=1)
    # ddx, ddy, ddz = splev(u_resampled, tck, der=2)



    return np.vstack((x_new, y_new, z_new)).T
