import itertools
import numpy as np
import pandas as pd
import tqdm, json, time

def rastrigin(position):
    return 10 * len(position) + sum([(x ** 2 - 10 * np.cos(2 * np.pi * x)) for x in position])


def griewank(position):
    return sum([(x ** 2) / 4000 for x in position]) - np.prod(
        [np.cos(x / np.sqrt(i + 1)) for i, x in enumerate(position)]) + 1


def rosenbrock(position):
    suma = 0
    for i in range(len(position) - 1):
        suma += 100 * (position[i + 1] - position[i] ** 2) ** 2 + (1 - position[i]) ** 2

    return suma


def michalewicz(position):
    m = 10
    return -sum([np.sin(x) * np.sin(i * x ** 2 / np.pi) ** (2 * m) for i, x in enumerate(position)])


FUNCTIONS = {"rastrigin": (rastrigin, -5.12, 5.12),
             "griewank": (griewank, -600, 600),
             "rosenbrock": (rosenbrock, -2.048, 2.048),
             "michalewicz": (michalewicz, 0, np.pi)}


class Particle:
    def __init__(self, dimensions, bounds, function, w, c1, c2):
        self.function = function
        self.position = np.random.uniform(bounds[0], bounds[1], dimensions)
        self.velocity_bounds = abs(bounds[1] - bounds[0])
        self.velocity = np.random.uniform(-self.velocity_bounds, self.velocity_bounds, dimensions)
        self.best_position = np.copy(self.position)
        self.best_value = function(self.position)
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def update_velocity(self, global_best_position, w, c1, c2):
        inertia = w * self.velocity
        cognitive = c1 * np.random.rand() * (self.best_position - self.position)
        social = c2 * np.random.rand() * (global_best_position - self.position)
        self.velocity = inertia + cognitive + social

    def update_position(self, bounds):
        self.position += self.velocity
        self.position = np.clip(self.position, bounds[0], bounds[1])

    def evaluate(self):
        value = self.function(self.position)
        if value < self.best_value:
            self.best_value = value
            self.best_position = np.copy(self.position)
        return value


class PSO:
    def __init__(self, dimensions, num_particles, bounds, max_iter, function, w, c1, c2):
        self.dimensions = dimensions
        self.num_particles = num_particles
        self.bounds = bounds
        self.max_iter = max_iter
        self.function = function
        self.particles = [Particle(dimensions, bounds, function, w, c1, c2) for _ in range(num_particles)]
        self.global_best_position = np.random.uniform(bounds[0], bounds[1], dimensions)
        self.global_best_value = float('inf')

    def optimize(self):
        self.last_results = []

        for iteration in range(self.max_iter):

            for particle in self.particles:
                fitness = particle.evaluate()
                if fitness < self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best_position = np.copy(particle.position)

            for particle in self.particles:
                particle_w = particle.w
                particle_c1 = particle.c1
                particle_c2 = particle.c2
                particle.update_velocity(self.global_best_position, particle_w, particle_c1, particle_c2)
                particle.update_position(self.bounds)

            # print(f"Iteration {iteration + 1}/{self.max_iter}, Best Fitness: {self.global_best_value}")
            try:
                if self.last_results[-1][0] != self.global_best_value:
                    self.last_results.append((self.global_best_value, iteration))
            except IndexError:
                self.last_results.append((self.global_best_value, iteration))

        return self.global_best_position, self.global_best_value


def grid_search(max_iter_options, num_particles_options, w_options, c1_options, c2_options, trials, functions):
    # Each run has a different output folder
    out_folder_for_run = f"{OUTPUT_FOLDER}/{time.strftime('%m_%d_%H_%M_%S')}"
    os.mkdir(out_folder_for_run)

    # Collect results
    all_results = []
    best_results = {}

    for fname in functions:
        function, lower_bound, upper_bound = FUNCTIONS[fname]
        for dimensions in DIMS:
            best_config = None
            best_mean_fitness = float('inf')

            for max_iter, num_particles, w, c1, c2 in tqdm.tqdm(itertools.product(
                    max_iter_options, num_particles_options, w_options, c1_options, c2_options
            ), desc="Grid Search", position=0, leave=False, total=len(max_iter_options) * len(num_particles_options) * len(w_options) * len(c1_options) * len(c2_options)):
                trial_fitness = []
                trial_hists = []
                for _ in tqdm.trange(trials, desc=f"{fname}, Dimensions: {dimensions}, Max Iters: {max_iter}, Swarm Size: {num_particles}, w: {w}, c1: {c1}, c2: {c2}", position=1, leave=False):
                    # Initialize and run PSO
                    pso = PSO(dimensions, num_particles, (lower_bound, upper_bound), max_iter, function, w, c1, c2)
                    _, best_fitness = pso.optimize()
                    trial_fitness.append(best_fitness)
                    trial_hists.append(pso.last_results)

                # Calculate statistics
                mean_fitness = np.mean(trial_fitness)
                std_fitness = np.std(trial_fitness)

                # Save the result
                config = {
                    "Function": fname,
                    "Dimensions": dimensions,
                    "Max Iterations": max_iter,
                    "Swarm Size": num_particles,
                    "Momentum (w)": w,
                    "Cognitive Constant (c1)": c1,
                    "Social Constant (c2)": c2,
                    "Mean Fitness": mean_fitness,
                    "Std Fitness": std_fitness,
                    "Histories": trial_hists
                }
                json.dump(config, open(f"{out_folder_for_run}/timeat_{time.strftime('%m_%d_%H_%M_%S')}_config_{fname}_{dimensions}_{max_iter}_{num_particles}_{w}_{c1}_{c2}.json", "w"))
                all_results.append(config)

                # Check if this is the best configuration for this function and dimension
                if mean_fitness < best_mean_fitness:
                    best_mean_fitness = mean_fitness
                    best_config = config

            # Save the best configuration for this function and dimension
            if best_config:
                key = (fname, dimensions)
                best_results[key] = best_config

    # Convert results to DataFrame and save
    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv(out_folder_for_run + "/pso_grid_search_results.csv", index=False)

    # Convert best results to DataFrame and save
    best_results_df = pd.DataFrame(best_results.values())
    best_results_df.to_csv(out_folder_for_run + "/pso_best_configurations.csv", index=False)

    print("Grid search completed. Results saved to:")
    print(" - pso_grid_search_results.csv")
    print(" - pso_best_configurations.csv")

DIMS = [2, 30, 100]

OUTPUT_FOLDER = "pso_results"
import os
try:
    os.mkdir(OUTPUT_FOLDER)
except:
    pass

if __name__ == "__main__":
    # Define parameter ranges for the grid search
    max_iter_options = [3000] # Lasa asa ca oricum converge in mai putine
    num_particles_options = [50, 100, 200]  # 3 variants
    w_options =  [i/10 for i in range(-10, 10, 2)] # 10 variants
    c1_options = [i/10 for i in range(-25, 25, 5)] # 10 variants
    c2_options = [i/10 for i in range(-25, 25, 5)] # 10 variants
    num_trials = 30
    # 3 x 10 x 10 x 10 = 3.000 rulari pt o functie ( * 30min = 25h eh it works)
    grid_search(
        max_iter_options=max_iter_options,
        num_particles_options=num_particles_options,
        w_options=w_options,
        c1_options=c1_options,
        c2_options=c2_options,
        trials=num_trials,
        functions=["rastrigin", "griewank", "rosenbrock", "michalewicz"]
    )
