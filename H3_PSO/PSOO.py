import numpy as np
import matplotlib.pyplot as plt
import random

class PSOMTSP:
    def __init__(self, num_salesmen, num_particles, iterations, tsp_file, w_start=0.9, w_end=0.4, c1=2, c2=2):
        """
        Particle Swarm Optimization for Multi-Traveling Salesman Problem.

        Parameters:
            num_salesmen (int): Number of salesmen.
            num_particles (int): Number of particles in the swarm.
            iterations (int): Number of iterations.
            tsp_file (str): Path to the TSP file containing city coordinates.
            w_start (float): Initial inertia weight.
            w_end (float): Final inertia weight.
            c1 (float): Cognitive coefficient.
            c2 (float): Social coefficient.
        """
        self.num_salesmen = num_salesmen
        self.num_particles = num_particles
        self.iterations = iterations
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2

        self.distance_matrix = self.create_distance_matrix(tsp_file)
        self.num_cities = len(self.distance_matrix)

        if self.num_cities < self.num_salesmen:
            raise ValueError("Number of cities must be greater than or equal to number of salesmen.")

        self.global_best_position = None
        self.global_best_cost = float('inf')

    @staticmethod
    def create_distance_matrix(tsp_file):
        """Reads TSP file and creates a distance matrix."""
        with open(tsp_file, 'r') as file:
            lines = file.readlines()

        coords = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 3 and parts[0].isdigit():
                coords.append((float(parts[1]), float(parts[2])))

        num_cities = len(coords)
        distance_matrix = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(num_cities):
                distance_matrix[i, j] = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))

        return distance_matrix

    def initialize_particles(self):
        """Initializes particles with random positions and velocities."""
        particles = []
        for _ in range(self.num_particles):
            position = list(range(self.num_cities))
            random.shuffle(position)
            delimiters = sorted(random.sample(range(1, self.num_cities), self.num_salesmen - 1))
            position += delimiters
            velocity = []
            particles.append({'position': position, 'velocity': velocity, 'best_position': position.copy(), 'best_cost': float('inf')})

        return particles

    def calculate_tour_cost(self, position):
        """Calculates the cost of a given particle position."""
        total_cost = 0
        split_positions = self.split_routes(position)
        for route in split_positions:
            route_cost = sum(self.distance_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))
            total_cost += route_cost

        return total_cost

    def split_routes(self, position):
        """Splits a position into routes for each salesman."""
        delimiters = position[-(self.num_salesmen - 1):]
        delimiters = [0] + sorted(delimiters) + [len(position) - self.num_salesmen]
        routes = []
        for i in range(len(delimiters) - 1):
            start, end = delimiters[i], delimiters[i + 1]
            routes.append(position[start:end])

        return routes

    def optimize(self):
        """Runs the PSO algorithm to solve the mTSP."""
        particles = self.initialize_particles()
        fitness_history = []

        for iteration in range(self.iterations):
            inertia_weight = self.w_start - iteration * (self.w_start - self.w_end) / self.iterations

            for particle in particles:
                # Evaluate fitness
                fitness = self.calculate_tour_cost(particle['position'])

                # Update personal best
                if fitness < particle['best_cost']:
                    particle['best_cost'] = fitness
                    particle['best_position'] = particle['position'].copy()

                # Update global best
                if fitness < self.global_best_cost:
                    self.global_best_cost = fitness
                    self.global_best_position = particle['position'].copy()

            # Update velocity and position
            for particle in particles:
                new_velocity = []
                for swap in self._get_valid_swaps(particle):
                    new_velocity.append(swap)

                particle['velocity'] = new_velocity
                particle['position'] = self._apply_swaps(particle['position'], new_velocity)

            fitness_history.append(self.global_best_cost)

            print(f"Iteration {iteration + 1}/{self.iterations}, Best Cost: {self.global_best_cost}")

        self.plot_solution()
        return fitness_history

    def plot_solution(self):
        """Plots the best solution."""
        routes = self.split_routes(self.global_best_position)
        plt.figure(figsize=(10, 6))

        for route in routes:
            x, y = [], []
            for city in route:
                x.append(city[0])
                y.append(city[1])
            plt.plot(x, y, marker='o')

        plt.title("Optimal Routes")
        plt.show()

    def _get_valid_swaps(self, particle):
        """Generates valid swaps for particle velocity updates."""
        swaps = []
        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                if i != j:
                    swaps.append((i, j))

        return swaps

    def _apply_swaps(self, position, swaps):
        """Applies a list of swaps to a particle's position."""
        for i, j in swaps:
            position[i], position[j] = position[j], position[i]
        return position

if __name__ == "__main__":
    tsp_solver = PSOMTSP(
        num_salesmen=2,
        num_particles=200,
        iterations=1000,
        tsp_file=r"D:\Facultate\tsplib\eil51.tsp"
    )
    fitness_history = tsp_solver.optimize()
    plt.plot(fitness_history)
    plt.title("Fitness Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.show()
