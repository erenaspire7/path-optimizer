import math
import numpy as np
from scipy.interpolate import splprep, splev, interp1d
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks

MASS = 0.0005  # kg
MAX_STRENGTH = 255


def detect_curves(path, threshold=0.01):
    curves = []
    current_curve = []

    for i in range(len(path) - 2):
        p1, p2, p3 = path[i], path[i + 1], path[i + 2]

        # Calculate the angle between the two line segments
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p2)
        angle = np.abs(np.arctan2(np.cross(v1, v2), np.dot(v1, v2)))

        if angle > threshold:
            current_curve.extend([p1, p2])
        else:
            if len(current_curve) > 2:
                current_curve.append(p3)
                curves.append(current_curve)
                current_curve = []

    if len(current_curve) > 2:
        curves.append(current_curve)

    return curves


def fit_curve(curve_points):
    x, y = zip(*curve_points)

    try:
        # Attempt to use splprep
        tck, u = splprep([x, y], s=0, k=3)
        return ("spline", tck)
    except Exception as e:
        print(f"Spline fitting failed: {e}")

        # Fallback to linear interpolation
        try:
            f = interp1d(x, y, kind="linear")
            return ("linear", f)
        except Exception as e:
            print(f"Linear interpolation failed: {e}")

            # If all else fails, return the original points
            return ("points", list(zip(x, y)))


def calculate_curvature(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    curvature = np.abs(dx * d2y - dy * d2x) / (dx**2 + dy**2) ** 1.5
    return curvature


def smoothen_path(points, num_interpolated_points=100):
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]

    # Create a cubic spline interpolation
    cs = CubicSpline(x, y)

    # Generate smooth x values
    x_smooth = np.linspace(x.min(), x.max(), num_interpolated_points)

    # Compute corresponding y values
    y_smooth = cs(x_smooth)

    return list(zip(x_smooth, y_smooth))


def find_peak_curvature(curve_data):
    fit_type, data = curve_data

    if fit_type == "spline":
        tck = data
        t = np.linspace(0, 1, 1000)
        x, y = splev(t, tck)
    elif fit_type == "linear":
        f = data
        x = np.linspace(f.x[0], f.x[-1], 1000)
        y = f(x)
    else:  # 'points'
        x, y = zip(*data)

    curvature = calculate_curvature(x, y)
    peak_index = np.argmax(curvature)
    return x[peak_index], y[peak_index]


def coordinate_percentage(start, end, target):
    def calculate_distance(point1, point2):
        return ((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2) ** 0.5

    total_distance = calculate_distance(start, end)
    target_distance = calculate_distance(start, target)

    if total_distance == 0:
        return 0

    percentage = target_distance / total_distance
    return max(0, min(1, percentage))


def generate_magnets(path):
    # retrieve curves in path
    curves = detect_curves(path)

    magnet_attributes = []

    if len(curves) != 0:
        for curve in curves:
            end = curve[-1]
            fitted_curve = fit_curve(curve)
            peak = find_peak_curvature(fitted_curve)

            # percent = coordinate_percentage(start, end, peak)

            magnet_attributes.append((*peak, True))
            magnet_attributes.append((*end, False))

        # add end of path
        magnet_attributes.append((*path[-1], False))

    else:
        pass

    return magnet_attributes


def round_based_on_value(number, base_value):
    if base_value == 0:
        return round(number)

    # Calculate the number of decimal places
    decimal_places = max(0, -int(math.floor(math.log10(abs(base_value)))))

    # Round the number
    return round(number, decimal_places)


def sample_along_path(path, num_points=100):
    distances = np.cumsum(
        [0]
        + [
            np.linalg.norm(np.array(path[i]) - np.array(path[i - 1]))
            for i in range(1, len(path))
        ]
    )
    total_dist = distances[-1]
    new_distances = np.linspace(0, total_dist, num_points)
    sampled_path = []

    for d in new_distances:
        idx = np.searchsorted(distances, d)
        if idx == 0:
            sampled_path.append(path[0])
        else:
            t = (d - distances[idx - 1]) / (distances[idx] - distances[idx - 1])
            point = (1 - t) * np.array(path[idx - 1]) + t * np.array(path[idx])
            sampled_path.append(point)

    return np.array(sampled_path)


def path_similarity(path1, path2):
    if len(path1) != len(path2):
        raise ValueError("Paths must have the same number of points")

    # Convert paths to numpy arrays for efficient computation
    arr1 = np.array(path1)
    arr2 = np.array(path2)

    # Calculate Euclidean distances between corresponding points
    distances = np.linalg.norm(arr1 - arr2, axis=1)

    # Calculate the mean distance
    mean_distance = np.mean(distances)

    # Normalize the similarity score
    max_possible_distance = np.linalg.norm(np.ptp(arr1, axis=0))  # Range of coordinates
    similarity = 1 - (mean_distance / (max_possible_distance + 1e-5))

    return similarity


def curvature_similarity(path1, path2):
    curvature1 = calculate_curvature(*zip(*path1))
    curvature2 = calculate_curvature(*zip(*path2))
    curvature_diff = np.abs(np.array(curvature1) - np.array(curvature2))
    return 1 - np.mean(curvature_diff)


def fitness_function(chromosome, target_path):
    simulated_path = chromosome.simulate_path(target_path)
    simulated_path = sample_along_path(simulated_path)

    path_sim = path_similarity(target_path, simulated_path)
    curve_sim = curvature_similarity(target_path, simulated_path)

    return 0.5 * path_sim + 0.5 * curve_sim


def annotate_magnets(ax, magnets):
    magnet_x, magnet_y = zip(*[(magnet.x, magnet.y) for magnet in magnets])

    ax.scatter(magnet_x, magnet_y, c="y", s=100, label="Magnets")
    for i, (mx, my) in enumerate(zip(magnet_x, magnet_y)):
        ax.annotate(f"M{i+1}", (mx, my), xytext=(5, 5), textcoords="offset points")


def visualize_results(target_path, chromosome, generation):
    simulated_path = chromosome.simulate_path(target_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    x, y = zip(*target_path)

    ax1.plot(x, y, "b-", label="Original Path")
    ax1.plot(x[0], y[0], "go", markersize=10, label="Start")
    ax1.plot(x[-1], y[-1], "ro", markersize=10, label="End")
    annotate_magnets(ax1, chromosome.magnets)
    ax1.set_title("Target Path")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.legend()
    ax1.grid(True)

    x, y = zip(*simulated_path)
    ax2.plot(x, y, "b-", label="Simulated Path")
    ax2.plot(x[0], y[0], "go", markersize=10, label="Start")
    ax2.plot(x[-1], y[-1], "ro", markersize=10, label="End")
    annotate_magnets(ax2, chromosome.magnets)
    ax2.set_title("Simulated Path")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f"results/{generation}.png")
    plt.close()
    print(f"Saved {generation}")
