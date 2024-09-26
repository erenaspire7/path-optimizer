import math
import json
import random
import numpy as np


def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def cumulative_distances(points):
    distances = [0]
    for i in range(1, len(points)):
        distances.append(distances[-1] + distance(points[i - 1], points[i]))
    return distances


def find_equidistant_points(points, num_points):
    cum_distances = cumulative_distances(points)
    total_distance = cum_distances[-1]
    segment_length = total_distance / num_points

    equidistant_points = []
    for i in range(1, num_points + 1):
        target_distance = i * segment_length
        index = np.searchsorted(cum_distances, target_distance)
        if index == len(points):
            equidistant_points.append(points[-1])
        else:
            # Interpolate between points
            d1, d2 = cum_distances[index - 1], cum_distances[index]
            p1, p2 = points[index - 1], points[index]
            t = (target_distance - d1) / (d2 - d1)
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            equidistant_points.append((x, y))

    return equidistant_points


def is_not_in_circle_range(p1, p2, radius):
    return distance(p1, p2) > radius


class Magnet:
    def __init__(self, x, y, duration):
        self.x = x
        self.y = y
        self.active = False
        self.duration = 10
        self.strength = 50  # newtons

    def activate(self):
        self.active = True

    def to_tuple(self):
        return (self.x, self.y)

    def update(self, dt):
        self.duration -= dt

    def deactivate(self):
        self.active = False

    def to_dict(self):
        return {"x": self.x, "y": self.y, "duration": self.duration}


class Chromosome:
    def __init__(self, num_magnets, mass, dt, target_path) -> None:
        self.num_magnets = num_magnets
        self.mass = mass  # kg
        self.dt = dt  # seconds
        self.proximity_radius = random.uniform(0.01, 0.1)
        self.initialize_magnets(target_path)

    def to_dict(self):
        return {
            "num_magnets": self.num_magnets,
            "proximity_radius": self.proximity_radius,
            "magnets": [magnet.to_dict() for magnet in self.magnets],
        }

    def initialize_magnets(self, target_path):
        positions = find_equidistant_points(target_path, self.num_magnets)
        self.magnets = [Magnet(*pos, random.uniform(0.1, 5.0)) for pos in positions]

    def simulate_path(self, target_path):
        start, dest = target_path[0], target_path[-1]

        simulated_path = [start]
        velocity = np.array([0.0, 0.0])

        for magnet in self.magnets:
            magnet.activate()
            ball_pos = np.array(simulated_path[-1])
            magnet_pos = np.array(magnet.to_tuple())

            while magnet.duration > 0:
                direction = magnet_pos - ball_pos
                dist = np.linalg.norm(direction)

                if dist < self.proximity_radius:
                    velocity_reduction_factor = max(
                        0, (dist / self.proximity_radius) ** 2
                    )
                    velocity *= velocity_reduction_factor

                    if dist < self.proximity_radius * 0.1:
                        velocity = np.array([0.0, 0.0])
                        break

                unit_direction = direction / dist
                force_magnitude = magnet.strength / (dist**2)
                force = force_magnitude * unit_direction

                acceleration = force / self.mass
                velocity += acceleration * self.dt

                ball_pos += velocity * self.dt + 0.5 * acceleration * self.dt**2
                simulated_path.append(ball_pos.copy())

                if ball_pos[0] > dest[0] or ball_pos[1] < 0:
                    break

                magnet.update(self.dt)

            magnet.deactivate()

            if np.allclose(ball_pos, dest) or ball_pos[0] > dest[0]:
                break

        return simulated_path
