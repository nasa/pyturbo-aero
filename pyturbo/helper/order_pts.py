import numpy as np 
import numpy.typing as npt 
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree # type: ignore

def order_points_nearest_neighbor(points: npt.NDArray) -> npt.NDArray:
    """Order points by nearest neighbor

    Args:
        points (npt.NDArray): Array of points 

    Raises:
        ValueError: input has to be a valid numpy array (N,2) or (N,3)

    Returns:
        npt.NDArray: indices to order the array 
    """
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError("Input must be a NumPy array of shape (N, 2) or (N, 3)")

    N = len(points)
    ordered_indices = [0]  # Start from the first point
    remaining = set(range(1, N))  # All other points

    while remaining:
        last_point = points[ordered_indices[-1]]
        # Get distances from last point to all remaining points
        remaining_list = list(remaining)
        dists = np.linalg.norm(points[remaining_list] - last_point, axis=1)
        nearest_index = remaining_list[np.argmin(dists)]
        ordered_indices.append(nearest_index)
        remaining.remove(nearest_index)

    return np.array(ordered_indices)

def order_points_by_spline_arc_length(points: npt.NDArray, smoothing: float = 0.0) -> npt.NDArray:
    """
    Orders points by their location along the arc length of a spline.

    Parameters:
        points: (N, 2) or (N, 3) array of points.
        smoothing: Smoothing factor for splprep. 0 means interpolation.

    Returns:
        Indices of the input points ordered by arc length along a fitted spline.
    """
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError("points must be of shape (N, 2) or (N, 3)")

    # Fit spline
    tck, u = splprep(points.T, s=smoothing)
    
    # Generate dense points along the spline
    u_dense = np.linspace(0, 1, len(points) * 10)
    spline_pts = np.stack(splev(u_dense, tck), axis=1) # type: ignore

    # Compute cumulative arc length along the spline
    arc_lengths = np.zeros(len(u_dense))
    arc_lengths[1:] = np.cumsum(np.linalg.norm(np.diff(spline_pts, axis=0), axis=1))

    # Find the closest point on the spline for each original point
    tree = cKDTree(spline_pts)
    dists, spline_indices = tree.query(points, k=1)

    # Map those closest spline points to arc length
    point_arc_lengths = arc_lengths[spline_indices]

    # Order original points by their arc length
    ordered_indices = np.argsort(point_arc_lengths)

    return ordered_indices
