import numpy as np
from scipy.interpolate import interp1d


def generate_bezier_curve(num_equidistant_points=5, point_distance=3.5):
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

    # Define control points (normalized between 0 and 1)
    p0 = (0, 0.5)  # Start point
    p1 = (0.3, 0.3)  # First control point
    p2 = (0.7, 0.7)  # Second control point
    p3 = (1, 0.5)  # End point

    # Generate the curve
    x, y = bezier_curve(p0, p1, p2, p3)

    # Calculate the cumulative distance along the curve
    dx = np.diff(x)
    dy = np.diff(y)
    segment_lengths = np.sqrt(dx**2 + dy**2)
    cumulative_lengths = np.concatenate(([0], np.cumsum(segment_lengths)))

    # Scale the curve to match the desired point distance
    total_distance = (num_equidistant_points - 1) * point_distance
    scale_factor = total_distance / cumulative_lengths[-1]
    x *= scale_factor
    y *= scale_factor
    cumulative_lengths *= scale_factor

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
    equidistant_lengths = np.linspace(0, total_distance, num_equidistant_points)
    equidistant_points = list(zip(fx(equidistant_lengths), fy(equidistant_lengths)))

    return equidistant_points
