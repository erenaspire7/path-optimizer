import copy
import json
import random
import numpy as np
from dtaidistance import dtw
import matplotlib.pyplot as plt
from chromosome import Chromosome
from generate_paths import generate_bezier_curve


class GeneticModel:
    def __init__(
        self,
        target_path,
        num_magnets=5,
        force_magnitude=50,
        time_step=0.0001,
        mass=1,
    ):
        # GA params
        self.population_size = 100
        self.elitism_count = int(self.population_size * 0.1)
        self.crossover_rate = 0.4
        self.mutation_rate = 0.1
        self.mutation_magnitude = 1
        self.stagnation_limit = 100
        self.generation = 0

        self.num_magnets = num_magnets
        self.force_magnitude = force_magnitude  # N
        self.time_step = time_step  # s
        self.mass = mass  # kg

        self.target_path = target_path
        self.initialize_population()

        self.initial_magnet_placements = self.population[0]

    def initialize_population(self):
        self.population = np.array(
            [
                Chromosome(
                    self.num_magnets, self.mass, self.time_step, self.target_path
                )
                for _ in range(self.population_size)
            ]
        )

    def log(self, fitness_score, override=False):
        if self.generation == 1 or self.generation % 10 == 0 or override:
            self.performance.append(
                {"generation": self.generation, "fitness": float(fitness_score)}
            )

    def train(self, generations=1):
        self.best_fitness = -np.inf
        self.best_individual: Chromosome = None
        generations_without_improvement = 0

        self.performance = []

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

    def fitness_function(self, chromosome: Chromosome):
        # fitness = 0
        simulated_path = chromosome.simulate_path(self.target_path)

        # Convert paths to numpy arrays if they aren't already
        target_path_array = np.array(self.target_path)
        simulated_path_array = np.array(simulated_path)

        target_x = target_path_array[:, 0]
        target_y = target_path_array[:, 1]

        simulated_x = simulated_path_array[:, 0]
        simulated_y = simulated_path_array[:, 1]

        distance_x, _ = dtw.warping_paths(target_x, simulated_x)
        distance_y, _ = dtw.warping_paths(target_y, simulated_y)

        total_distance = np.sqrt(distance_x**2 + distance_y**2)

        # Normalized to 0 and 1
        max_length = max(len(target_x), len(simulated_x))
        normalized_distance = total_distance / (max_length * np.sqrt(2))

        fitness = 1 - normalized_distance

        return fitness

    def crossover(self, parent_1: Chromosome, parent_2: Chromosome):
        if np.random.random() < self.crossover_rate:
            child1 = Chromosome(
                self.num_magnets, self.mass, self.time_step, self.target_path
            )
            child2 = Chromosome(
                self.num_magnets, self.mass, self.time_step, self.target_path
            )

            crossover_point = np.random.randint(1, len(parent_1.magnets))

            child1.magnets = (
                parent_1.magnets[:crossover_point] + parent_2.magnets[crossover_point:]
            )

            child2.magnets = (
                parent_2.magnets[:crossover_point] + parent_1.magnets[crossover_point:]
            )
        else:
            child1, child2 = copy.deepcopy(parent_1), copy.deepcopy(parent_2)

        return child1, child2

    def mutate(self, child: Chromosome):
        limit = self.mutation_magnitude

        for magnet in child.magnets:
            if random.random() < self.mutation_rate:
                magnet.x += random.uniform(-limit * 0.01, limit * 0.01)
                magnet.y += random.uniform(-limit * 0.01, limit * 0.01)

        return child

    def annotate_magnets(self, ax, magnets):
        magnet_x = [magnet.x for magnet in magnets]
        magnet_y = [magnet.y for magnet in magnets]

        ax.scatter(magnet_x, magnet_y, c="y", s=100, label="Magnets")
        for i, (mx, my) in enumerate(zip(magnet_x, magnet_y)):
            ax.annotate(f"M{i+1}", (mx, my), xytext=(5, 5), textcoords="offset points")

    def visualize_results(self):
        magnets = self.best_individual.magnets
        simulated_path = self.best_individual.simulate_path(self.target_path)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        x, y = zip(*target_path)
        ax1.plot(x, y, "b-", label="Original Path")
        ax1.plot(x[0], y[0], "go", markersize=10, label="Start")
        ax1.plot(x[-1], y[-1], "ro", markersize=10, label="End")
        self.annotate_magnets(ax1, magnets)
        ax1.set_title("Target Path")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.legend()
        ax1.grid(True)

        x, y = zip(*simulated_path)
        ax2.plot(x, y, "b-", label="Simulated Path")
        ax2.plot(x[0], y[0], "go", markersize=10, label="Start")
        ax2.plot(x[-1], y[-1], "ro", markersize=10, label="End")
        self.annotate_magnets(ax2, magnets)
        ax2.set_title("Simulated Path")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(f"results/{self.generation}.png")
        print(f"Saved {self.generation}")

    def save_results(self):
        with open(f"bezier-curve-v1.json", "w+") as json_file:
            json.dump(
                {
                    "best_individual": self.best_individual.to_dict(),
                    "best_fitness": float(self.best_fitness),
                    "errors": self.performance,
                    "params": {
                        "magnets": self.num_magnets,
                        "population_size": self.population_size,
                        "crossover_rate": self.crossover_rate,
                        "mutation_rate": self.mutation_rate,
                        "mutation_magnitude": self.mutation_magnitude,
                        "force_magnitude": self.force_magnitude,
                        "time_step": self.time_step,
                        "mass": self.mass,
                        "elitism_count": self.elitism_count,
                    },
                },
                json_file,
            )

    def run(self):
        self.train()
        self.save_results()


if __name__ == "__main__":
    target_path = generate_bezier_curve()

    model = GeneticModel(target_path)
    model.run()
