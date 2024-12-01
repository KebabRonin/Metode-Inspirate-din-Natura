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

    def optimize(self, patience = None, threshold = 0):
        self.last_results = []
        if patience is None:
            patience = self.max_iter
        pat = patience
        pbar = tqdm.trange(self.max_iter, leave=False, position=2)
        for iteration in pbar:
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
                    if self.last_results[-1][0] - self.global_best_value >= threshold:
                        # Reset patience only if improvement >= 0.5
                        pat = patience
                    self.last_results.append((self.global_best_value, iteration))
            except IndexError:
                self.last_results.append((self.global_best_value, iteration))

            pat -= 1
            if pat <= 0:
                print(f"Patience reached at iteration {iteration + 1}/{self.max_iter}, Best Fitness: {self.global_best_value}", file=open('log.txt', 'wt'))
                break
            pbar.set_postfix_str(f"{self.global_best_value:.8f} fitness")

        return self.global_best_position, self.global_best_value


def grid_search(cfgs, trials):
    # Each run has a different output folder
    out_folder_for_run = f"{OUTPUT_FOLDER}/{time.strftime('%m_%d_%H_%M_%S')}"
    os.mkdir(out_folder_for_run)

    # Collect results
    all_results = []

    for fname, dimensions, max_iter, patience, num_particles, w, c1, c2 in tqdm.tqdm(cfgs, desc="Grid Search", position=0, leave=False):
        function, lower_bound, upper_bound = FUNCTIONS[fname]
        trial_fitness = []
        trial_hists = []
        for _ in tqdm.trange(trials, desc=f"{fname}, Dimensions: {dimensions}, Max Iters: {max_iter}, Swarm Size: {num_particles}, w: {w}, c1: {c1}, c2: {c2}", position=1, leave=False):
            # Initialize and run PSO
            pso = PSO(dimensions, num_particles, (lower_bound, upper_bound), max_iter, function, w, c1, c2)
            _, best_fitness = pso.optimize(patience=patience, threshold=0.3)
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

    print("Grid search completed. Results saved to:")

# rosenbrock 10.000iters da 0 pt -0.6, 0, 2.5 (30 dims, 200 particles)

cfgs = [
    # ('rosenbrock', 2, 50, -0.2, -1.0, 2.0), # skip as it is already 0
    # ('rosenbrock', 30, 20_000, 1e10, 200, -0.6, 0.0, 1.5),
    # ('rosenbrock', 30, 3_000, 100, 200, -0.6, 0.0, 2.5), # mine
    # ('rosenbrock', 100, 20_000, 1e10, 200, -0.2, 2.0, 2.0),
    # # ('griewank', 2, 200, 0.6, 1.5, 0.5), # These are close enough to 0
    # ('griewank', 30, 200, -0.2, 2.0, 2.0),
    # ('griewank', 100, 200, -0.2, 2.0, 0.5),
    # ('michalewicz', 2, 20_000, 1e10, 50, -0.6, -1.0, 1.5),
    ('michalewicz', 30, 20_000, 1e10, 200, -0.6, 2.0, -1.0),
    ('michalewicz', 100, 20_000, 1e10, 200, -0.6, 2.0, -1.0),
    ('rosenbrock', 30, 20_000, 1e10, 200, -0.6, 0.0, 2.5), # mine
]
OUTPUT_FOLDER = "pso_results"
import os
try:
    os.mkdir(OUTPUT_FOLDER)
except:
    pass
# maybe this too? S=69 w=-0.4438 c1=-0.2699 c2=3.3950
# grid_search(cfgs, 30)
# exit(0)
if __name__ == "__main__":
    # Define parameter ranges for the grid search
    max_iter_options = [3000] # Lasa asa ca oricum converge in mai putine
    num_particles_options = [200]  # 3 variants
    w_options =  [-0.2] # 8 variants
    c1_options = [-2.0] # 8 variants
    c2_options = [0.5, 1.0, 1.5, 2.0] # 8 variants
    num_trials = 30
    # 3 x 10 x 10 x 10 = 3.000 rulari pt o functie ( * 30min = 25h eh it works)
    grid_search(
        max_iter_options=max_iter_options,
        num_particles_options=num_particles_options,
        w_options=w_options,
        c1_options=c1_options,
        c2_options=c2_options,
        trials=num_trials,
        functions=["griewank"],
        patience=100, # early stopping
        threshold=0.3, # early stopping
    )
