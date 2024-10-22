from utils import MAX_STRENGTH, MASS, smoothen_path

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

    def generate_arc(self, start, end, magnet, num_points=100):
        magnet_pos = np.array(magnet.to_tuple())
        normalized_strength = magnet.magnetic_strength / MAX_STRENGTH

        line_vector = end - start
        t = np.dot(magnet_pos - start, line_vector) / np.dot(line_vector, line_vector)
        intersection_percent = np.clip(t, 0, 1)

        closest_point = start + intersection_percent * line_vector
        to_magnet = magnet_pos - closest_point
        control_point = closest_point + normalized_strength * to_magnet

        arc_points = []
        for t in np.linspace(0, 1, num_points):
            point = (1 - t) ** 2 * start + 2 * (1 - t) * t * control_point + t**2 * end
            arc_points.append(point)

        return smoothen_path(arc_points), intersection_percent

    def simulate_path(self, target_path):
        start = target_path[0]
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

                        curved_path, percent = self.generate_arc(
                            start_pos, end_pos, curve_magnet
                        )

                        duration = m.end_time - m.start_time
                        curve_end_time = (duration * percent) + m.start_time

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
