import numpy as np


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


functions = {"rastrigin": (rastrigin, -5.12, 5.12),
             "griewank": (griewank, -600, 600),
             "rosenbrock": (rosenbrock, -2.048, 2.048),
             "michalewicz": (michalewicz, 0, np.pi)}


class Particle:
    def __init__(self, dimensions, bounds, function):
        self.function = function
        self.position = np.random.uniform(bounds[0], bounds[1], dimensions)
        self.velocity_bounds = abs(bounds[1] - bounds[0])
        self.velocity = np.random.uniform(-self.velocity_bounds, self.velocity_bounds, dimensions)
        self.best_position = np.copy(self.position)
        self.best_value = function(self.position)

    def update_velocity(self, global_best_position, w=-0.2089, c1=-0.0787, c2=3.7637):
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
    def __init__(self, dimensions, num_particles, bounds, max_iter, function):
        self.dimensions = dimensions
        self.num_particles = num_particles
        self.bounds = bounds
        self.max_iter = max_iter
        self.function = function
        self.particles = [Particle(dimensions, bounds, function) for _ in range(num_particles)]
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


def main():
    for name, (function, lower_bound, upper_bound) in functions.items():
        for dimensions in [2, 30, 100]:
            pso = PSO(dimensions, 100, (lower_bound, upper_bound), 1000, function)
            print(f"Function: {name}, Dimensions: {dimensions}")
            pso.optimize()
            print()


if __name__ == "__main__":
    main()
