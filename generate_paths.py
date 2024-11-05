import numpy as np
from scipy.interpolate import interp1d


def generate_bezier_curve(num_equidistant_points=5):
    def bezier_curve(p0, p1, p2, p3, num_samples=1000):
        t = np.linspace(0, 1, num_samples)
        x = (
            (1 - t) ** 3 * p0[0]
            + 3 * (1 - t) ** 2 * t * p1[0]
            + 3 * (1 - t) * t**2 * p2[0]
            + t**3 * p3[0]
        )
        y = (
            (1 - t) ** 3 * p0[1]
            + 3 * (1 - t) ** 2 * t * p1[1]
            + 3 * (1 - t) * t**2 * p2[1]
            + t**3 * p3[1]
        )
        return x, y

    # Define control points (normalized between -1 and 1)
    p0 = (-1.0, 0.0)  # Start point
    p1 = (-0.4, -0.4)  # First control point
    p2 = (0.4, 0.4)  # Second control point
    p3 = (1.0, 0.0)  # End point

    # Generate the curve
    x, y = bezier_curve(p0, p1, p2, p3)

    # Calculate the cumulative distance along the curve
    dx = np.diff(x)
    dy = np.diff(y)
    segment_lengths = np.sqrt(dx**2 + dy**2)
    cumulative_lengths = np.concatenate(([0], np.cumsum(segment_lengths)))

    # Create interpolation functions
    fx = interp1d(
        cumulative_lengths,
        x,
        kind="cubic",
        bounds_error=False,
        fill_value="extrapolate",
    )
    fy = interp1d(
        cumulative_lengths,
        y,
        kind="cubic",
        bounds_error=False,
        fill_value="extrapolate",
    )

    # Generate equidistant points
    equidistant_lengths = np.linspace(0, cumulative_lengths[-1], num_equidistant_points)
    equidistant_points = list(zip(fx(equidistant_lengths), fy(equidistant_lengths)))

    return equidistant_points


import random
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline


def segments_intersect(p1, p2, p3, p4):
    """Check if line segments (p1,p2) and (p3,p4) intersect"""

    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


def is_crossing_path(path, current, new_point):
    """Check if new line segment intersects with any existing segments"""
    if len(path) < 2:
        return False

    # Check intersection with all segments except the last one
    # (which connects to our current point)
    for i in range(len(path) - 2):
        if segments_intersect(path[i], path[i + 1], current, new_point):
            return True
    return False


def generate_random_path(points=10, min_distance=0.1, max_distance=0.3):
    """Generate random path avoiding self-intersections"""
    path = [(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))]

    attempts = 0
    max_attempts = 100  # Prevent infinite loops

    while len(path) < points and attempts < max_attempts:
        current = path[-1]
        valid_point_found = False

        # Try multiple angles to find a valid next point
        for _ in range(36):  # Try every 10 degrees
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(min_distance, max_distance)

            new_x = current[0] + distance * math.cos(angle)
            new_y = current[1] + distance * math.sin(angle)
            new_point = (new_x, new_y)

            if (
                -1 <= new_x <= 1
                and -1 <= new_y <= 1
                and not is_crossing_path(path, current, new_point)
            ):
                path.append(new_point)
                valid_point_found = True
                break

        if not valid_point_found:
            attempts += 1
            if attempts >= max_attempts:
                print(
                    "Warning: Could not generate non-intersecting path, returning partial path"
                )
                break

    return [(round(x, 3), round(y, 3)) for x, y in path]


def calculate_path_distances(path):
    """Calculate cumulative distances along the path"""
    distances = [0]
    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        distance = math.sqrt(dx**2 + dy**2)
        distances.append(distances[-1] + distance)
    return distances


def visualize_smooth_path(path, num_points=100):
    """
    Visualize original path and smoothed path using path length parameterization
    """
    # Calculate cumulative distances along the path
    distances = calculate_path_distances(path)

    # Extract coordinates
    x_coords, y_coords = zip(*path)

    # Create cubic spline interpolation functions
    cs_x = CubicSpline(distances, x_coords)
    cs_y = CubicSpline(distances, y_coords)

    # Generate points for smooth curve
    t_smooth = np.linspace(0, distances[-1], num_points)
    x_smooth = cs_x(t_smooth)
    y_smooth = cs_y(t_smooth)

    # Force the smoothed path to exactly match start and end points
    x_smooth[0] = x_coords[0]
    x_smooth[-1] = x_coords[-1]
    y_smooth[0] = y_coords[0]
    y_smooth[-1] = y_coords[-1]

    # # Plotting
    # plt.figure(figsize=(12, 5))

    # # Plot 1: Original path
    # plt.subplot(121)
    # x_orig, y_orig = zip(*path)
    # plt.plot(x_orig, y_orig, "b-", alpha=0.6, label="Original Path")
    # plt.scatter(x_orig, y_orig, c="red", s=50)
    # plt.grid(True, alpha=0.3)
    # plt.title("Original Path")
    # plt.axis("equal")
    # plt.legend()

    # # Plot 2: Smoothed path
    # plt.subplot(122)
    # plt.plot(x_orig, y_orig, "b-", alpha=0.6, label="Original Path")
    # plt.plot(x_smooth, y_smooth, "r-", alpha=0.6, label="Smoothed Path")
    # plt.scatter(x_orig, y_orig, c="red", s=50)
    # plt.grid(True, alpha=0.3)
    # plt.title("Smoothed Path")
    # plt.axis("equal")
    # plt.legend()

    # plt.tight_layout()
    # plt.show()

    return np.column_stack((x_smooth, y_smooth))
