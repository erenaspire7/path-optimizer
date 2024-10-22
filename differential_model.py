import copy
import random
import numpy as np
from chromosome import Chromosome
from utils import fitness_function, visualize_results
import json


class DifferentialModel:
    def __init__(self, target_path, magnet_attributes, time_step=0.01):
        self.population_size = 25
        self.F = 0.7  # Differential weight
        self.CR = 0.8  # Crossover rate
        self.stagnation_limit = 10

        self.target_path = target_path

        self.population = [
            Chromosome(time_step, magnet_attributes)
            for _ in range(self.population_size)
        ]
        self.performance = []
        self.best_fitness = -np.inf
        self.best_individual = None
        self.generation = 0

    def log(self, fitness_score, override=False):
        if self.generation == 1 or self.generation % 10 == 0 or override:
            self.performance.append(
                {"generation": self.generation, "fitness": float(fitness_score)}
            )

    def differential_evolution(self, max_generations=100):
        generations_without_improvement = 0

        for _ in range(max_generations):
            self.generation += 1

            next_population = []

            gen_best_fitness = -np.inf
            gen_best_individual = None

            for i in range(self.population_size):
                r1, r2, r3 = random.sample(range(self.population_size), 3)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = random.sample(range(self.population_size), 3)

                mutant = self.mutate(
                    self.population[i],
                    self.population[r1],
                    self.population[r2],
                    self.population[r3],
                )

                trial = self.crossover(self.population[i], mutant)
                next_individual, fitness = self.selection(self.population[i], trial)

                if fitness > gen_best_fitness:
                    gen_best_fitness = fitness
                    gen_best_individual = next_individual

                next_population.append(next_individual)

            self.population = next_population

            if gen_best_fitness > self.best_fitness:
                self.best_fitness = gen_best_fitness
                self.best_individual = copy.deepcopy(gen_best_individual)
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            if generations_without_improvement >= self.stagnation_limit:
                self.log(self.best_fitness, True)
                break

            self.log(self.best_fitness)
            visualize_results(self.target_path, self.best_individual, self.generation)

    def mutate(self, target, r1, r2, r3):
        mutant = copy.deepcopy(target)
        idxs = [idx for idx, m in enumerate(mutant.magnets) if m.curve_point is True]

        for idx in idxs:
            new_strength = r1.magnets[idx].magnetic_strength + self.F * (
                r2.magnets[idx].magnetic_strength - r3.magnets[idx].magnetic_strength
            )

            mutant.magnets[idx].magnetic_strength = max(0, new_strength)
            mutant.magnets[idx].magnetic_strength = min(255, new_strength)

            mutant.magnets[idx].x = r1.magnets[idx].x + self.F * (
                (r2.magnets[idx].x - r3.magnets[idx].x) + random.uniform(-0.1, 0.1)
            )
            mutant.magnets[idx].y = r1.magnets[idx].y + self.F * (
                (r2.magnets[idx].y - r3.magnets[idx].y) + random.uniform(-0.1, 0.1)
            )

        return mutant

    def crossover(self, target, mutant):
        trial = copy.deepcopy(target)

        idxs = [idx for idx, m in enumerate(mutant.magnets) if m.curve_point is True]

        for idx in idxs:
            if random.random() < self.CR:
                trial.magnets[idx].magnetic_strength = mutant.magnets[
                    idx
                ].magnetic_strength

            if random.random() < self.CR:
                trial.magnets[idx].x = mutant.magnets[idx].x

            if random.random() < self.CR:
                trial.magnets[idx].y = mutant.magnets[idx].y

        return trial

    def selection(self, target, trial):
        target_fitness = fitness_function(target, self.target_path)
        trial_fitness = fitness_function(trial, self.target_path)

        if trial_fitness > target_fitness:
            return trial, trial_fitness
        else:
            return target, target_fitness

    def run(self):
        self.differential_evolution()
        self.save_results()

    def save_results(self):
        with open(f"ga-bezier-curve-v1.json", "w+") as json_file:
            json.dump(
                {
                    "best_individual": self.best_individual.to_dict(),
                    "best_fitness": float(self.best_fitness),
                    "performance": self.performance,
                    "params": {
                        "population_size": self.population_size,
                        "differential_weight": self.F,
                        "crossover_rate": self.CR,
                    },
                },
                json_file,
            )
