
from typing import Tuple
from scipy.interpolate import splprep, splev, PchipInterpolator
import numpy as np 
import numpy.typing as npt




def resample_curve(curve:npt.NDArray, M:int=100):
    _, idx = np.unique(curve, axis=0, return_index=True)
    if len(idx) < len(curve):
        curve = curve[np.sort(idx)]
    tck, _ = splprep(curve.T, s=0, per=0)
    u_new = np.linspace(0, 1, M)
    return np.stack(splev(u_new, tck), axis=1) # type: ignore
 
def arc_length_param_strict(points: npt.NDArray) -> Tuple[npt.NDArray,npt.NDArray]:
    deltas = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(deltas, axis=1)
    arc = np.concatenate([[0], np.cumsum(seg_lengths)])
    
    # Remove duplicates from arc to ensure strictly increasing values
    _, unique_idx = np.unique(arc, return_index=True)
    arc = arc[unique_idx]
    points = points[unique_idx]

    arc /= arc[-1]  # Normalize to [0, 1]
    return arc, points

def curvature_2d(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = (dx**2 + dy**2) ** 1.5 + 1e-8
    return numerator / denominator

def curvature_3d(x, y, z):
    dx = np.gradient(x)
    dy = np.gradient(y)
    dz = np.gradient(z)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    ddz = np.gradient(dz)
    cross = np.cross(np.stack([dx, dy, dz], axis=1),
                     np.stack([ddx, ddy, ddz], axis=1))
    num = np.linalg.norm(cross, axis=1)
    denom = (dx**2 + dy**2 + dz**2) ** 1.5 + 1e-8
    return num / denom

def resample_by_curvature(points: npt.NDArray, N: int) -> npt.NDArray:
    """
    Resample 2D or 3D curve using curvature-based importance sampling and PCHIP interpolation.
    """
    _, idx = np.unique(points, axis=0, return_index=True)
    if len(idx) < len(points):
        points = points[np.sort(idx)]
    if points.shape[0] < 2:
        raise ValueError("Need at least 2 unique points.")

    dim = points.shape[1]

    # Interpolate each dimension with PCHIP
    arc, points = arc_length_param_strict(points)
    pchips = [PchipInterpolator(arc, points[:, i]) for i in range(dim)]
    
    u_fine = np.linspace(0, 1, 1000)
    coords_fine = np.stack([p(u_fine) for p in pchips], axis=1)

    # Compute curvature
    if dim == 2:
        kappa = curvature_2d(coords_fine[:, 0], coords_fine[:, 1])
    else:
        kappa = curvature_3d(coords_fine[:, 0], coords_fine[:, 1], coords_fine[:, 2])

    # Normalize curvature into a sampling density
    density = kappa + 1e-3
    density /= np.sum(density)
    cdf = np.cumsum(density)
    cdf /= cdf[-1]

    # Importance sample along CDF
    u_resampled = np.interp(np.linspace(0, 1, N), cdf, u_fine)
    resampled = np.stack([p(u_resampled) for p in pchips], axis=1)
    return resampled