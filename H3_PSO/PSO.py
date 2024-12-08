import numpy as np
import math
import random
import matplotlib.pyplot as plt
from typing import List, Tuple


class TSPData:
    def __init__(self, tsp_string: str):
        self.coords = {}
        self.parse_tsp_data(tsp_string)
        self.distances = self.calculate_distances()

    def parse_tsp_data(self, data: str):
        lines = data.strip().split('\n')
        coord_section = False
        for line in lines:
            if line == 'NODE_COORD_SECTION':
                coord_section = True
                continue
            elif line == 'EOF':
                break
            if coord_section:
                node_id, x, y = map(float, line.strip().split())
                self.coords[int(node_id)] = (x, y)

    def calculate_distances(self) -> np.ndarray:
        n = len(self.coords)
        distances = np.zeros((n, n))
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i != j:
                    x1, y1 = self.coords[i]
                    x2, y2 = self.coords[j]
                    distances[i - 1][j - 1] = round(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
        return distances


class MTSPSolver:
    def __init__(self, tsp_data: TSPData, num_salesmen: int, population_size: int = 2000,
                 generations: int = 5000, mutation_rate: float = 0.02, elite_size: int = 50):
        self.distances = tsp_data.distances
        self.coords = tsp_data.coords
        self.num_cities = len(self.distances)
        self.num_salesmen = num_salesmen
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size

    def create_initial_population(self) -> List[List[int]]:
        population = []
        for _ in range(self.population_size):
            route = list(range(2, self.num_cities))  # Exclude depot (1)
            random.shuffle(route)
            population.append(route)
        return population

    def split_route(self, route: List[int]) -> List[List[int]]:
        cities_per_salesman = len(route) // self.num_salesmen
        remainder = len(route) % self.num_salesmen
        splits = []
        start = 0
        for i in range(self.num_salesmen):
            segment_length = cities_per_salesman + (1 if i < remainder else 0)
            tour = [1] + route[start:start + segment_length] + [1]  # Add depot at start and end
            splits.append(tour)
            start += segment_length
        return splits

    def calculate_tour_length(self, tour: List[int]) -> float:
        length = 0
        for i in range(len(tour) - 1):
            length += self.distances[tour[i] - 1][tour[i + 1] - 1]
        return length

    def calculate_fitness(self, route: List[int]) -> float:
        splits = self.split_route(route)
        max_length = max(self.calculate_tour_length(split) for split in splits)
        return 1 / max_length

    def tournament_selection(self, population: List[List[int]], tournament_size: int = 5) -> List[int]:
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=self.calculate_fitness)

    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        size = len(parent1)
        child = [-1] * size
        start, end = sorted(random.sample(range(size), 2))
        child[start:end] = parent1[start:end]
        remaining = [city for city in parent2 if city not in child[start:end]]
        j = 0
        for i in range(size):
            if child[i] == -1:
                child[i] = remaining[j]
                j += 1
        return child

    def mutate(self, route: List[int]) -> List[int]:
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
        return route

    def solve(self) -> Tuple[List[List[int]], float, List[float]]:
        population = self.create_initial_population()
        best_fitness = float('-inf')
        best_route = None

        for _ in range(self.generations):
            population.sort(key=self.calculate_fitness, reverse=True)
            current_best_fitness = self.calculate_fitness(population[0])
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_route = population[0]

            new_population = population[:self.elite_size]
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            population = new_population

        best_splits = self.split_route(best_route)
        tour_lengths = [self.calculate_tour_length(split) for split in best_splits]
        return best_splits, max(tour_lengths), tour_lengths


class MTSPVisualizer:
    def __init__(self, coords, routes):
        self.coords = coords
        self.routes = routes
        self.colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']

    def plot_routes(self):
        plt.figure(figsize=(12, 8))

        # Plot all cities
        x_coords = [coord[0] for coord in self.coords.values()]
        y_coords = [coord[1] for coord in self.coords.values()]
        plt.scatter(x_coords, y_coords, c='gray', s=50)

        # Highlight depot
        depot = self.coords[1]
        plt.scatter([depot[0]], [depot[1]], c='red', s=100, marker='*', label='Depot')

        # Plot each route
        for i, route in enumerate(self.routes):
            route_coords = []
            for city in route:
                route_coords.append(self.coords[city])
            x_coords = [coord[0] for coord in route_coords]
            y_coords = [coord[1] for coord in route_coords]
            plt.plot(x_coords, y_coords, c=self.colors[i % len(self.colors)],
                     label=f'Salesman {i + 1}', linewidth=2)

        plt.legend()
        plt.title('MinMax Multiple TSP Solution')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.show()


def main():
    # Example TSP data (eil51)
    tsp_string = """NAME : eil51
TYPE : TSP
DIMENSION : 51
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_SECTION
1 37 52
2 49 49
3 52 64
4 20 26
5 40 30
6 21 47
7 17 63
8 31 62
9 52 33
10 51 21
11 42 41
12 31 32
13 5 25
14 12 42
15 36 16
16 52 41
17 27 23
18 17 33
19 13 13
20 57 58
21 62 42
22 42 57
23 16 57
24 8 52
25 7 38
26 27 68
27 30 48
28 43 67
29 58 48
30 58 27
31 37 69
32 38 46
33 46 10
34 61 33
35 62 63
36 63 69
37 32 22
38 45 35
39 59 15
40 5 6
41 10 17
42 21 10
43 5 64
44 30 15
45 39 10
46 32 39
47 25 32
48 25 55
49 48 28
50 56 37
51 30 40
EOF"""

    # Create TSP data instance
    tsp_data = TSPData(tsp_string)

    # Create and run solver
    solver = MTSPSolver(tsp_data, num_salesmen=2)
    routes, max_length, tour_lengths = solver.solve()

    # Print results
    print("The tours traveled by the salesmen are:")
    for i, (route, length) in enumerate(zip(routes, tour_lengths)):
        route_str = " ".join(map(str, route))
        print(f"{route_str}   (#{len(route) - 2})  Cost: {length}")

    # Visualize solution
    visualizer = MTSPVisualizer(tsp_data.coords, routes)
    visualizer.plot_routes()


if __name__ == "__main__":
    main()