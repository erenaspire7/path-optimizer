import numpy as np


def generate_bezier_curve():
    # Define control points
    p0 = (0, 0.5)  # Start point
    p1 = (0.25, 0)  # First control point
    p2 = (0.75, 1)  # Second control point
    p3 = (1, 0.5)  # End point

    def bezier_curve(p0, p1, p2, p3, num_points=100):
        t = np.linspace(0, 1, num_points)
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
        return list(zip(x, y))

    return bezier_curve(p0, p1, p2, p3)
