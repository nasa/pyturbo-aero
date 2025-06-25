import numpy as np
import matplotlib.pyplot as plt

# Your points (N, 2)
points = np.array([
    [-0.05119767,  0.04      ],
    [-0.0537965 ,  0.04      ],
    [-0.06620976,  0.03350574],
    [-0.06161524,  0.02700707],
    [-0.05530343,  0.020903  ],
    [-0.0417457 ,  0.01459747],
    [-0.04779432,  0.01436815],
    [-0.03713698,  0.01286846],
    [-0.03252826,  0.01113945],
    [-0.02791954,  0.00941044],
    [-0.02331083,  0.00768143],
    [-0.01870211,  0.00595242],
    [-0.01955056,  0.00437341],
    [-0.01409339,  0.0042234 ],
    [-0.00948467,  0.00249439],
    [-0.00487596,  0.00076538],
    [-0.00026724, -0.00096363],
])

# Function to compute angle between two vectors
def angle_between(v1, v2):
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))

# Function to remove sharp turns
def remove_sharp_turns(points, max_angle_deg=30.0):
    max_angle_rad = np.deg2rad(max_angle_deg)
    filtered = [points[0]]
    for i in range(1, len(points)-1):
        p0, p1, p2 = points[i-1], points[i], points[i+1]
        v1 = p1 - p0
        v2 = p2 - p1
        angle = angle_between(v1, v2)
        if angle < max_angle_rad:
            filtered.append(p1)
    filtered.append(points[-1])
    return np.array(filtered)

def keep_significant_turns(points, min_angle_deg=10.0, max_angle_deg=170.0):
    """
    Keep points where the angle between segments is significant, i.e.,
    not nearly straight or overly sharp, preserving key geometry.
    
    Args:
        points (np.ndarray): (N, 2) array of [x, y] points
        min_angle_deg (float): Minimum angle to preserve (sharp turns)
        max_angle_deg (float): Maximum angle beyond which the turn is too flat to keep
    
    Returns:
        np.ndarray: Filtered points
    """
    min_angle_rad = np.deg2rad(min_angle_deg)
    max_angle_rad = np.deg2rad(max_angle_deg)
    
    keep = [points[0]]  # always keep first point
    
    for i in range(1, len(points)-1):
        p0, p1, p2 = points[i-1], points[i], points[i+1]
        v1 = p1 - p0
        v2 = p2 - p1
        angle = angle_between(v1, v2)
        
        # Keep points with significant angle (i.e. sharp or notable bend)
        if angle <= min_angle_rad or angle >= max_angle_rad:
            continue  # too flat — discard
        keep.append(p1)
    
    keep.append(points[-1])  # always keep last point
    return np.array(keep)

# Apply filter
filtered = keep_significant_turns(points, min_angle_deg=5, max_angle_deg=175)

# Plot comparison
plt.figure(figsize=(8,4))
plt.plot(points[:, 0], points[:, 1], 'o--', label='Original')
plt.plot(filtered[:, 0], filtered[:, 1], 'ro-', label='Smoothed')
plt.gca().set_aspect('equal')
plt.title("Removing Sharp Turns Based on Angle")
plt.legend()
plt.grid(True)
plt.show()
print('check')