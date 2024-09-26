import math
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


class Magnet:
    def __init__(self, x, y, duration):
        self.x = x
        self.y = y
        self.active = False
        self.duration = duration
        self.strength = 50  # newtons

    def activate(self):
        self.active = True

    def to_tuple(self):
        return (self.x, self.y)

    def deactivate(self):
        self.active = False

    def to_dict(self):
        return {"x": self.x, "y": self.y, "duration": self.duration}


class Chromosome:
    def __init__(self, num_magnets, mass, time_step, target_path):
        self.num_magnets = num_magnets
        self.mass = mass  # kg
        self.time_step = time_step  # seconds
        self.actual_times = []
        self.initialize_magnets(target_path)

    def to_dict(self):
        return {
            "num_magnets": self.num_magnets,
            "magnets": [magnet.to_dict() for magnet in self.magnets],
        }

    def initialize_magnets(self, target_path):
        positions = find_equidistant_points(target_path, self.num_magnets)
        self.magnets = [Magnet(*pos, random.uniform(0.1, 2.0)) for pos in positions]

    def estimate_time_to_magnet(self, ball_pos, velocity, magnet_pos, acceleration):
        displacement = magnet_pos - ball_pos

        # Quadratic equation coefficients
        a = 0.5 * np.dot(acceleration, acceleration)
        b = np.dot(velocity, acceleration)
        c = np.dot(displacement, displacement)

        # Solve quadratic equation
        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            # Ball won't reach the magnet with current parameters
            return float("inf")

        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t2 = (-b - np.sqrt(discriminant)) / (2 * a)

        # Return the smallest positive solution
        return min(t for t in (t1, t2) if t > 0) if (t1 > 0 or t2 > 0) else float("inf")

    def simulate_path(self, target_path):
        start = target_path[0]
        simulated_path = [start]

        velocity = np.array([0.0, 0.0])
        ball_pos = np.array(start)
        actual_times = []

        for magnet in self.magnets:
            magnet.activate()
            time_elapsed = 0

            while time_elapsed < magnet.duration:
                magnet_pos = np.array(magnet.to_tuple())
                direction = magnet_pos - ball_pos
                dist = np.linalg.norm(direction)

                if dist > 0:
                    acceleration = direction / dist
                else:
                    acceleration = np.array([0.0, 0.0])

                velocity += acceleration * self.time_step
                predicted_pos = ball_pos + velocity * self.time_step

                magnet_direction_current = magnet_pos - ball_pos
                magnet_direction_future = magnet_pos - predicted_pos

                if np.dot(magnet_direction_current, magnet_direction_future) <= 0:
                    # Magnet is between, stop the ball at the magnet
                    distance_to_magnet = np.linalg.norm(magnet_direction_current)
                    time_to_magnet = distance_to_magnet / np.linalg.norm(velocity)
                    time_elapsed += time_to_magnet

                    ball_pos = magnet_pos
                    velocity = np.array([0.0, 0.0])
                    simulated_path.append(ball_pos.copy())

                    break
                else:
                    # Update ball position for the next step
                    ball_pos = predicted_pos
                    simulated_path.append(ball_pos.copy())

                time_elapsed += self.time_step

            actual_times.append(time_elapsed)
            magnet.deactivate()

        self.actual_times = actual_times
        return simulated_path

    def mutate(self, mutation_rate, mutation_magnitude):
        for i, magnet in enumerate(self.magnets):
            if random.random() < mutation_rate:
                actual_time = self.actual_times[i]
                time_difference = actual_time - magnet.duration

                adaptive_mutation = time_difference * random.uniform(
                    mutation_magnitude * 0.1, mutation_magnitude * 0.5
                )
                magnet.duration = max(0.1, magnet.duration + adaptive_mutation)

        return self
