import numpy as np


def rastrigin(position):
    A = 10
    return A * len(position) + sum([(x ** 2 - A * np.cos(2 * np.pi * x)) for x in position])


class Particle:
    def __init__(self, dimensions, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], dimensions)
        self.velocity_bounds = abs(bounds[1] - bounds[0])
        self.velocity = np.random.uniform(-self.velocity_bounds, self.velocity_bounds, dimensions)
        self.best_position = np.copy(self.position)
        self.best_value = rastrigin(self.position)

    def update_velocity(self, global_best_position, w=-0.2089, c1=-0.0787, c2=3.7637):

        inertia = w * self.velocity
        cognitive = c1 * np.random.rand() * (self.best_position - self.position)
        social = c2 * np.random.rand() * (global_best_position - self.position)
        self.velocity = inertia + cognitive + social

    def update_position(self, bounds):
        self.position += self.velocity
        self.position = np.clip(self.position, bounds[0], bounds[1])

    def evaluate(self):
        value = rastrigin(self.position)
        if value < self.best_value:
            self.best_value = value
            self.best_position = np.copy(self.position)
        return value


class PSO:
    def __init__(self, dimensions, num_particles, bounds, max_iter):
        self.dimensions = dimensions
        self.num_particles = num_particles
        self.bounds = bounds
        self.max_iter = max_iter
        self.particles = [Particle(dimensions, bounds) for _ in range(num_particles)]
        self.global_best_position = np.random.uniform(bounds[0], bounds[1], dimensions)
        self.global_best_value = float('inf')

    def optimize(self):

        for iteration in range(self.max_iter):

            for particle in self.particles:
                fitness = particle.evaluate()
                if fitness < self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best_position = np.copy(particle.position)

            for particle in self.particles:
                particle.update_velocity(self.global_best_position)
                particle.update_position(self.bounds)

            print(f"Iteration {iteration + 1}/{self.max_iter}, Best Fitness: {self.global_best_value}")

        return self.global_best_position, self.global_best_value


dimensions_list = [2, 30, 100]
bounds = (-5.12, 5.12)
num_particles = 161
max_iter = 1000

for dimensions in dimensions_list:
    print(f"\nOptimizing Rastrigin's function in {dimensions} dimensions")
    pso = PSO(dimensions=dimensions, num_particles=num_particles, bounds=bounds, max_iter=max_iter)
    best_position, best_value = pso.optimize()
    print(f"Best position: {best_position}")
    print(f"Best value: {best_value}\n")
