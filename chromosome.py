from utils import MAX_STRENGTH, MASS, coordinate_percentage, round_based_on_value

import numpy as np
from magnet import Magnet


class Chromosome:
    def __init__(self, time_step, magnet_attributes):
        self.time_step = time_step
        self.magnets = [Magnet(*attr) for attr in magnet_attributes]

    def to_dict(self):
        return {
            "magnets": [m.to_dict() for m in self.magnets],
        }

    def calculate_curve_control_point(self, start, end, magnet):
        magnet_pos = np.array(magnet.to_tuple())

        # Calculate the point on the line closest to the magnet
        line_vector = end - start
        t = np.dot(magnet_pos - start, line_vector) / np.dot(line_vector, line_vector)
        t = max(0, min(1, t))  # Clamp t between 0 and 1
        closest_point = start + t * line_vector

        direction_to_magnet = magnet_pos - closest_point
        direction_to_magnet = direction_to_magnet / np.linalg.norm(direction_to_magnet)

        # Adjust curve based on magnet strength and distance
        curve_factor = magnet.magnetic_strength / MAX_STRENGTH
        distance_factor = 1 / (
            1 + np.linalg.norm(magnet_pos - closest_point)
        )  # Inverse distance
        max_curve_distance = np.linalg.norm(end - start) / 2
        curve_distance = curve_factor * distance_factor * max_curve_distance

        # Calculate control point
        control_point = closest_point + direction_to_magnet * curve_distance

        return control_point, t

    def generate_bezier_curve(self, start, control, end, num_points=100):
        curve = []
        for t in np.linspace(0, 1, num_points):
            point = (1 - t) ** 2 * start + 2 * (1 - t) * t * control + t**2 * end
            curve.append(point)
        return curve

    def simulate_path(self, target_path):
        start = target_path[0]
        end = target_path[-1]
        ball_pos = np.array(start)
        simulated_path = [ball_pos]

        velocity = 0
        total_time = 0

        static_magnets = [
            (idx, m) for idx, m in enumerate(self.magnets) if m.curve_point is False
        ]

        for idx, m in static_magnets:
            m.activate(total_time)

            start_pos = simulated_path[-1]

            while m.active:
                pos = np.array(m.to_tuple())

                direction = pos - ball_pos
                dist = np.linalg.norm(direction)

                total_force = np.zeros(2)

                if dist > 0:
                    force_magnitude = (
                        5.12e-9 * m.magnetic_strength / MAX_STRENGTH
                    ) / pow(((dist / 100) + 0.004), 4)
                    force = force_magnitude * direction / dist
                    total_force += force

                acceleration = total_force / MASS
                velocity += acceleration * self.time_step
                predicted_pos = ball_pos + velocity * self.time_step

                # snap to magnet if magnet between cur_pos and predicted_pos
                pos = np.array(m.to_tuple())

                magnet_direction_current = pos - ball_pos
                magnet_direction_future = pos - predicted_pos

                if np.dot(magnet_direction_current, magnet_direction_future) < 0:
                    ball_pos = pos
                    velocity = 0
                    simulated_path.append(ball_pos.copy())

                    m.deactivate(total_time)

                    if self.magnets[idx - 1].curve_point:
                        end_pos = simulated_path[-1]
                        curve_magnet = self.magnets[idx - 1]

                        curve_control, percent = self.calculate_curve_control_point(
                            start_pos, end_pos, curve_magnet
                        )

                        duration = m.end_time - m.start_time
                        curve_end_time = (duration * percent) + m.start_time

                        curved_path = self.generate_bezier_curve(
                            start_pos, curve_control, end_pos
                        )

                        start_idx = np.where((simulated_path == start_pos).all(axis=1))[
                            0
                        ][0]
                        end_idx = np.where((simulated_path == end_pos).all(axis=1))[0][
                            0
                        ]

                        simulated_path[start_idx:end_idx] = curved_path

                        curve_magnet.start_time = m.start_time
                        curve_magnet.end_time = curve_end_time

                    break

                ball_pos = predicted_pos
                total_time += self.time_step
                simulated_path.append(ball_pos.copy())

        return simulated_path

        # for

        # Calculate the bounding box of the target path
        # x_min = min(point[0] for point in target_path)
        # x_max = max(point[0] for point in target_path)
        # y_min = min(point[1] for point in target_path)
        # y_max = max(point[1] for point in target_path)

        # margin = 0.1  # 10% margin
        # x_range = x_max - x_min
        # y_range = y_max - y_min
        # x_min -= margin * x_range
        # x_max += margin * x_range
        # y_min -= margin * y_range
        # y_max += margin * y_range

        #         # Check if the ball has overshot the bounding box
        #         if (
        #             ball_pos[0] < x_min
        #             or ball_pos[0] > x_max
        #             or ball_pos[1] < y_min
        #             or ball_pos[1] > y_max
        #         ):
        #             return simulated_path

        #         if (
        #             np.dot(ball_pos - np.array(end), np.array(end) - np.array(start))
        #             >= 0
        #         ):
        #             return simulated_path
