import math
import json
import random
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

# from scipy.spatial.distance import euclidean
from scipy.interpolate import interp1d


class Magnet:
    def __init__(self, x, y, offset_factor):
        self.x = x
        self.y = y
        self.offset_factor = offset_factor


class Chromosome:
    def __init__(self, num_magnets):
        self.num_magnets = num_magnets
        self.proportions = np.random.dirichlet(np.ones(num_magnets))
        self.offset_factors = np.random.uniform(-0.1, 0.1, num_magnets)

    def __str__(self) -> str:
        return json.dumps(
            {"num_magnets": self.num_magnets, "proportions": self.proportions.tolist()}
        )

    def initialize_magnets(self, target_path):
        path_length = sum(
            euclidean(p1, p2) for p1, p2 in zip(target_path, target_path[1:])
        )
        magnets = []

        # Calculate the positions for evenly distributed magnets
        magnet_positions = [
            i * path_length / self.num_magnets for i in range(self.num_magnets)
        ]

        current_length = 0
        path_index = 0

        for i, target_length in enumerate(magnet_positions):
            while current_length < target_length and path_index < len(target_path) - 1:
                seg_start, seg_end = (
                    target_path[path_index],
                    target_path[path_index + 1],
                )
                seg_length = euclidean(seg_start, seg_end)

                if current_length + seg_length >= target_length:
                    # Interpolate position within this segment
                    ratio = (target_length - current_length) / seg_length
                    pos = (
                        seg_start[0] + ratio * (seg_end[0] - seg_start[0]),
                        seg_start[1] + ratio * (seg_end[1] - seg_start[1]),
                    )

                    # Apply offset perpendicular to path direction
                    direction = np.array(seg_end) - np.array(seg_start)
                    perpendicular = np.array([-direction[1], direction[0]])
                    perpendicular /= np.linalg.norm(perpendicular)
                    offset_pos = np.array(pos) + self.offset_factors[i] * perpendicular

                    magnets.append(Magnet(*offset_pos, self.offset_factors[i]))
                    break

                current_length += seg_length
                path_index += 1

            if path_index >= len(target_path) - 1:
                # If we've reached the end of the path, place remaining magnets at the end
                magnets.extend(
                    [
                        Magnet(*target_path[-1], 0)
                        for _ in range(self.num_magnets - len(magnets))
                    ]
                )
                break

        return magnets

    def get_path_length(self, target_path: list[tuple[float, float]]):
        return sum(math.dist(p1, p2) for p1, p2 in zip(target_path, target_path[1:]))


class GeneticModel:
    def __init__(
        self,
        num_magnets=5,
        force_magnitude=0.1,
        time_step=0.01,
        mass=1,
        drag_coefficient=0.1,
    ):
        # GA params
        self.population_size = 100
        self.elitism_count = int(self.population_size * 0.5)
        self.crossover_rate = 0.4
        self.mutation_rate = 0.1
        self.mutation_magnitude = 1
        self.completion_bonus = 0.5
        self.partial_credit_factor = 0.5
        self.standard_num_points = 100

        # Physics params
        self.generation = 0
        self.stagnation_limit = 100
        self.num_magnets = num_magnets
        self.force_magnitude = force_magnitude  # N
        self.time_step = time_step  # s
        self.max_simulation_time = 300  # s
        self.mass = mass  # kg
        self.drag_coefficient = drag_coefficient

        self.initialize_population()
        self.initialize_target_paths()

    def initialize_population(self):
        self.population = np.array(
            [Chromosome(self.num_magnets) for _ in range(self.population_size)]
        )

    def log(self, fitness_score, override=False):
        if self.generation == 1 or self.generation % 10 == 0 or override:
            self.errors.append(
                {"generation": self.generation, "fitness": fitness_score}
            )

    def initialize_target_paths(self, size=20, num_points=5):
        target_paths = []

        for _ in range(size):
            start = random.uniform(-1, 1), -1
            end = random.uniform(-1, 1), 1

            x = np.linspace(start[0], end[0], num_points)
            y = np.random.uniform(
                min(start[1], end[1]), max(start[1], end[1]), num_points
            )

            y[0] = start[1]
            y[-1] = end[1]

            path = list(zip(x, y))
            target_paths.append(path)

        self.target_paths = target_paths

    def train(self, generations=10_000):
        self.best_fitness = -np.inf
        self.best_individual = None
        generations_without_improvement = 0

        self.errors = []

        for _ in range(generations):
            self.generation += 1

            fitness_scores = np.array(
                [self.fitness_function(entry) for entry in self.population]
            )

            best_idx = np.argmax(fitness_scores)

            self.log(fitness_scores[best_idx])

            if self.best_fitness < fitness_scores[best_idx]:
                self.best_fitness = fitness_scores[best_idx]
                self.best_individual = self.population[best_idx]
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            if generations_without_improvement >= self.stagnation_limit:
                self.log(fitness_scores[best_idx], True)
                break

            # default to elitism
            indices = np.argsort(fitness_scores)[-self.elitism_count :]
            parents = self.population[indices][::-1]

            new_population = parents.tolist()

            for i in range(0, len(parents), 2):
                parent_1, parent_2 = parents[i], parents[i + 1]
                child_1, child_2 = self.crossover(parent_1, parent_2)
                new_population.extend([self.mutate(child_1), self.mutate(child_2)])

            self.visualize_results()
            self.population = np.array(new_population)

    def save_results(self):
        with open(f"results.json", "w+") as json_file:
            json.dump(
                {
                    "population": [str(ind) for ind in self.population],
                    "best_individual": str(self.best_individual),
                    "best_fitness": self.best_fitness,
                    "errors": self.errors,
                    "params": {
                        "magnets": self.num_magnets,
                        "population_size": self.population_size,
                        "crossover_rate": self.crossover_rate,
                        "mutation_rate": self.mutation_rate,
                        "mutation_magnitude": self.mutation_magnitude,
                        "force_magnitude": self.force_magnitude,
                        "time_step": self.time_step,
                        "max_simulation_time": self.max_simulation_time,
                        "mass": self.mass,
                        "drag_coefficient": self.drag_coefficient,
                    },
                },
                json_file,
            )

    def fitness_function(self, chromosome: Chromosome):
        total_fitness = 0

        for target_path in self.target_paths:
            magnets = chromosome.initialize_magnets(target_path)
            simulated_path, total_time, completion_ratio = self.simulate_path(
                target_path, chromosome, magnets
            )

            path_accuracy = self.calculate_path_accuracy(target_path, simulated_path)
            time_efficiency = 1 - (total_time / self.max_simulation_time)
            magnet_distribution = self.calculate_magnet_distribution(magnets)

            fitness = (
                0.6 * path_accuracy + 0.2 * time_efficiency + 0.2 * magnet_distribution
            ) * completion_ratio

            total_fitness += fitness

        return total_fitness / len(self.target_paths)

    def calculate_path_accuracy(self, target_path, simulated_path):
        target_interp = interp1d(*zip(*target_path), kind="linear")
        simulated_x, simulated_y = zip(*simulated_path)
        try:
            target_y = target_interp(simulated_x)
            mse = np.mean((np.array(simulated_y) - target_y) ** 2)
            return 1 / (1 + mse)
        except ValueError:
            return 0

    def calculate_magnet_distribution(self, magnets: list[Magnet]):
        distances = [
            euclidean((m1.x, m1.y), (m2.x, m2.y))
            for m1, m2 in zip(magnets, magnets[1:])
        ]
        return 1 / (1 + np.std(distances))

    def simulate_path(self, path, chromosome: Chromosome, magnets):
        simulated_path = [path[0]]
        x, y = path[0]
        vx, vy = 0, 0

        total_time = 0

        for magnet, proportion in zip(magnets, chromosome.proportions):
            target_distance = proportion * chromosome.get_path_length(path)
            distance_traveled = 0

            while (
                distance_traveled < target_distance
                and total_time < self.max_simulation_time
            ):
                dist_x = magnet.x - x
                dist_y = magnet.y - y
                dist = math.sqrt(dist_x**2 + dist_y**2)

                if dist > 0:
                    # Calculate force components
                    force_x = self.force_magnitude * dist_x / dist
                    force_y = self.force_magnitude * dist_y / dist

                    # Calculate acceleration (F = ma)
                    ax = force_x / self.mass
                    ay = force_y / self.mass

                    # Apply drag force (proportional to velocity)
                    drag_x = -self.drag_coefficient * vx / self.mass
                    drag_y = -self.drag_coefficient * vy / self.mass

                    ax += drag_x
                    ay += drag_y

                    # Update velocity (v = u + at)
                    vx += ax * self.time_step
                    vy += ay * self.time_step

                    # Update position (s = ut + 0.5at^2)
                    dx = vx * self.time_step + 0.5 * ax * self.time_step**2
                    dy = vy * self.time_step + 0.5 * ay * self.time_step**2

                    # Apply offset (perpendicular to the direction of motion)
                    offset_magnitude = magnet.offset_factor * math.sqrt(dx**2 + dy**2)
                    offset_x = offset_magnitude * (-dy) / math.sqrt(dx**2 + dy**2)
                    offset_y = offset_magnitude * dx / math.sqrt(dx**2 + dy**2)
                    dx += offset_x
                    dy += offset_y

                    x += dx
                    y += dy
                    simulated_path.append((x, y))

                    distance_traveled += math.sqrt(dx**2 + dy**2)
                    total_time += self.time_step

                else:
                    # If the object is exactly on the magnet, add a small random movement
                    dx = random.uniform(-0.01, 0.01)
                    dy = random.uniform(-0.01, 0.01)
                    x += dx
                    y += dy
                    simulated_path.append((x, y))
                    distance_traveled += math.sqrt(dx**2 + dy**2)
                    total_time += self.time_step

            if total_time >= self.max_simulation_time:
                break

        completion_ratio = min(distance_traveled / target_distance, 1.0)
        return simulated_path, total_time, completion_ratio

    def crossover(self, parent1, parent2):
        child1, child2 = Chromosome(self.num_magnets), Chromosome(self.num_magnets)
        alpha = np.random.random(self.num_magnets)

        child1.proportions = (
            alpha * parent1.proportions + (1 - alpha) * parent2.proportions
        )
        child2.proportions = (
            1 - alpha
        ) * parent1.proportions + alpha * parent2.proportions

        child1.proportions /= child1.proportions.sum()
        child2.proportions /= child2.proportions.sum()

        return child1, child2

    def mutate(self, chromosome: Chromosome):
        mutation_mask = (
            np.random.random(chromosome.proportions.shape) < self.mutation_rate
        )
        mutation = np.random.normal(
            0, self.mutation_magnitude, chromosome.proportions.shape
        )
        chromosome.proportions += mutation * mutation_mask
        chromosome.proportions = np.abs(chromosome.proportions)  # Ensure non-negative
        chromosome.proportions /= chromosome.proportions.sum()

        return chromosome

    def visualize_results(self):
        idx = random.randint(0, len(self.target_paths) - 1)

        target_path = self.target_paths[idx]
        target_path = self.upscale(target_path)

        magnets = self.best_individual.initialize_magnets(target_path)
        simulated_path, _, _ = self.simulate_path(
            target_path, self.best_individual, magnets
        )
        simulated_path = self.downscale(simulated_path)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Plot original path
        x, y = zip(*target_path)
        ax1.plot(x, y, "b-", label="Target Path")
        ax1.plot(x[0], y[0], "go", markersize=10, label="Start")
        ax1.plot(x[-1], y[-1], "ro", markersize=10, label="End")
        ax1.set_title("Target Path")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.legend()
        ax1.grid(True)

        # Plot simulated path
        x, y = zip(*simulated_path)
        ax2.plot(x, y, "r-", label="Simulated Path")
        ax2.plot(x[0], y[0], "go", markersize=10, label="Start")
        ax2.plot(x[-1], y[-1], "ro", markersize=10, label="End")
        ax2.set_title("Simulated Path")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.legend()
        ax2.grid(True)

        # Plot magnets
        x, y = zip(*target_path)
        ax3.plot(x, y, "b-", alpha=0.5, label="Target Path")
        magnet_x = [magnet.x for magnet in magnets]
        magnet_y = [magnet.y for magnet in magnets]
        ax3.scatter(magnet_x, magnet_y, c="r", s=100, label="Magnets")
        for i, (mx, my) in enumerate(zip(magnet_x, magnet_y)):
            ax3.annotate(f"M{i+1}", (mx, my), xytext=(5, 5), textcoords="offset points")
        ax3.set_title("Magnet Positions")
        ax3.set_xlabel("X")
        ax3.set_ylabel("Y")
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        plt.savefig(f"results/{self.generation}.png")

        print(f"Saved {self.generation}")

    def upscale(self, path):
        original_path = np.array(path)

        # Calculate the distances between consecutive points
        point_distances = np.sqrt(np.sum(np.diff(original_path, axis=0) ** 2, axis=1))

        # Calculate the cumulative distance along the path
        distances = np.concatenate(([0], np.cumsum(point_distances)))

        # Create evenly spaced points along the path
        equidistant_points = np.linspace(0, distances[-1], self.standard_num_points)

        # Interpolate x and y coordinates separately
        interpolated_path = np.column_stack(
            [
                np.interp(equidistant_points, distances, original_path[:, 0]),
                np.interp(equidistant_points, distances, original_path[:, 1]),
            ]
        )

        return interpolated_path

    def downscale(self, path):
        original_path = np.array(path)

        # Always include start and end points
        downscaled_path = [original_path[0], original_path[-1]]

        # Calculate cumulative distances along the path
        distances = np.cumsum(
            np.sqrt(np.sum(np.diff(original_path, axis=0) ** 2, axis=1))
        )
        total_distance = distances[-1]

        # Calculate target distances for evenly spaced points
        target_distances = np.linspace(0, total_distance, self.standard_num_points)[
            1:-1
        ]

        # Find the indices of points closest to target distances
        indices = np.searchsorted(distances, target_distances)

        # Add these points to our downscaled path
        downscaled_path[1:1] = original_path[indices]

        return np.array(downscaled_path)

    def run(self):
        self.train()
        self.save_results()


if __name__ == "__main__":
    model = GeneticModel(num_magnets=5)
    model.run()
