import numpy as np
from typing import List, Tuple, Dict
import math
import random
import matplotlib.pyplot as plt


def parse_tsp_file(data: str) -> Dict[int, Tuple[float, float]]:
    coordinates = {}
    lines = data.strip().split('\n')
    start_idx = lines.index('NODE_COORD_SECTION')

    for line in lines[start_idx + 1:]:
        if line == 'EOF':
            break
        node_id, x, y = map(float, line.strip().split())
        coordinates[int(node_id)] = (x, y)

    return coordinates


def create_distance_matrix(coordinates: Dict[int, Tuple[float, float]]) -> np.ndarray:
    n = len(coordinates)
    distances = np.zeros((n, n))

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j:
                x1, y1 = coordinates[i]
                x2, y2 = coordinates[j]
                distances[i - 1][j - 1] = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    return distances


class Particle:
    def __init__(self, num_cities: int, num_salesmen: int, distances: np.ndarray):
        self.num_cities = num_cities
        self.num_salesmen = num_salesmen
        self.distances = distances

        # Initialize position (solution encoding)
        self.position = self._initialize_position()
        self.velocity = []  # List of swap operations
        self.personal_best_position = self.position.copy()
        self.personal_best_fitness = float('inf')

    def _initialize_position(self) -> List:
        """Initialize a valid position using balanced route assignment"""
        # Create a random permutation of cities
        cities = list(range(1, self.num_cities + 1))
        random.shuffle(cities)

        # Calculate base route size and remainder
        base_size = self.num_cities // self.num_salesmen
        remainder = self.num_cities % self.num_salesmen

        # Create position with delimiters
        position = []
        start_idx = 0

        for i in range(self.num_salesmen):
            # Calculate route size for this salesman
            route_size = base_size + (1 if i < remainder else 0)
            # Add cities for this route
            position.extend(cities[start_idx:start_idx + route_size])
            start_idx += route_size
            # Add delimiter if not last route
            if i < self.num_salesmen - 1:
                position.append(0)

        return position

    def evaluate_fitness(self) -> float:
        routes = self._get_routes()
        total_distance = 0
        route_distances = []

        for route in routes:
            route_distance = 0
            route_distance += self.distances[0][route[0] - 1]

            for i in range(len(route) - 1):
                route_distance += self.distances[route[i] - 1][route[i + 1] - 1]

            route_distance += self.distances[route[-1] - 1][0]

            total_distance += route_distance
            route_distances.append(route_distance)

        balance_penalty = np.std(route_distances) if route_distances else 0
        return total_distance + 0.5 * balance_penalty

    def _get_routes(self) -> List[List[int]]:
        routes = []
        current_route = []

        for city in self.position:
            if city == 0:
                if current_route:
                    routes.append(current_route)
                current_route = []
            else:
                current_route.append(city)

        if current_route:
            routes.append(current_route)

        return routes

    def update_velocity(self, global_best_position: List, w: float, c1: float, c2: float):
        self.velocity = []

        # Personal influence - with probability c1
        if random.random() < c1:
            swaps = self._get_valid_swaps(self.position, self.personal_best_position)
            if swaps:
                self.velocity.extend(random.sample(swaps, k=min(2, len(swaps))))

        # Global influence - with probability c2
        if random.random() < c2:
            swaps = self._get_valid_swaps(self.position, global_best_position)
            if swaps:
                self.velocity.extend(random.sample(swaps, k=min(2, len(swaps))))

        # Inertia - random valid swaps
        if random.random() < w:
            valid_swaps = self._get_random_valid_swaps()
            if valid_swaps:
                self.velocity.extend(random.sample(valid_swaps, k=min(2, len(valid_swaps))))

    def _get_valid_swaps(self, current: List, target: List) -> List[Tuple[int, int]]:
        """Generate valid swaps that move current solution towards target"""
        valid_swaps = []

        # Find positions where elements differ
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                # Only consider swaps between cities (non-zero elements)
                if current[i] != 0 and current[j] != 0:
                    # Check if swap would move towards target
                    if (current[i] != target[i] and current[j] != target[j] and
                            (current[i] == target[j] or current[j] == target[i])):
                        valid_swaps.append((i, j))

        return valid_swaps

    def _get_random_valid_swaps(self) -> List[Tuple[int, int]]:
        """Generate random valid swaps between cities (non-delimiter elements)"""
        valid_swaps = []

        # Get positions of cities (non-zero elements)
        city_positions = [i for i, x in enumerate(self.position) if x != 0]

        # Generate all possible pairs of city positions
        if len(city_positions) >= 2:
            valid_swaps = [(i, j) for i in city_positions for j in city_positions if i < j]

        return valid_swaps

    def update_position(self):
        """Apply velocity (swap sequence) to current position"""
        for i, j in self.velocity:
            self.position[i], self.position[j] = self.position[j], self.position[i]


class MTSPPSO:
    def __init__(self, num_cities: int, num_salesmen: int, distances: np.ndarray,
                 num_particles: int = 50, max_iterations: int = 1000):
        self.num_cities = num_cities
        self.num_salesmen = num_salesmen
        self.distances = distances
        self.num_particles = num_particles
        self.max_iterations = max_iterations

        self.particles = [Particle(num_cities, num_salesmen, distances)
                          for _ in range(num_particles)]

        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.best_fitness_history = []

    def optimize(self) -> Tuple[List, float]:
        w_start, w_end = 0.6, 0.4
        c1, c2 = 2.5, 2.5

        for iteration in range(self.max_iterations):
            w = w_start - (w_start - w_end) * iteration / self.max_iterations

            for particle in self.particles:
                fitness = particle.evaluate_fitness()

                if fitness < particle.personal_best_fitness:
                    particle.personal_best_position = particle.position.copy()
                    particle.personal_best_fitness = fitness

                if fitness < self.global_best_fitness:
                    self.global_best_position = particle.position.copy()
                    self.global_best_fitness = fitness

            self.best_fitness_history.append(self.global_best_fitness)

            for particle in self.particles:
                particle.update_velocity(self.global_best_position, w, c1, c2)
                particle.update_position()

            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Best fitness = {self.global_best_fitness:.2f}")

        return self.global_best_position, self.global_best_fitness


def plot_routes(coordinates: Dict[int, Tuple[float, float]], routes: List[List[int]]):
    """Plot the routes for each salesman"""
    plt.figure(figsize=(12, 8))

    # Plot depot (city 1) as a special marker
    depot_x, depot_y = coordinates[1]
    plt.plot(depot_x, depot_y, 'k*', markersize=15, label='Depot')

    # Define colors for different salesmen
    colors = ['b', 'r', 'g', 'm', 'c', 'y']

    # Plot routes for each salesman
    for i, route in enumerate(routes):
        color = colors[i % len(colors)]

        # Plot route
        route_coords = [(depot_x, depot_y)]  # Start at depot
        for city in route:
            x, y = coordinates[city]
            route_coords.append((x, y))
        route_coords.append((depot_x, depot_y))  # Return to depot

        # Convert coordinates to arrays for plotting
        x_coords, y_coords = zip(*route_coords)
        plt.plot(x_coords, y_coords, color=color, linewidth=1, alpha=0.7,
                 label=f'Salesman {i + 1}')

        # Plot cities on the route
        for city in route:
            x, y = coordinates[city]
            plt.plot(x, y, 'o', color=color, markersize=8)
            plt.annotate(str(city), (x, y), xytext=(5, 5), textcoords='offset points')

    plt.title('MTSP Routes Visualization')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def solve_eil51_mtsp(tsp_data: str, num_salesmen: int = 2):
    coordinates = parse_tsp_file(tsp_data)
    distances = create_distance_matrix(coordinates)

    pso = MTSPPSO(
        num_cities=51,
        num_salesmen=num_salesmen,
        distances=distances,
        num_particles=200,
        max_iterations=3000
    )

    best_solution, best_fitness = pso.optimize()

    routes = []
    current_route = []
    for city in best_solution:
        if city == 0:
            routes.append(current_route)
            current_route = []
        else:
            current_route.append(city)
    routes.append(current_route)

    print("\nFinal Solution:")
    print(f"Total fitness (distance + balance penalty): {best_fitness:.2f}")
    print("\nRoutes:")
    for i, route in enumerate(routes):
        route_distance = 0
        if route:
            route_distance += distances[0][route[0] - 1]
            for j in range(len(route) - 1):
                route_distance += distances[route[j] - 1][route[j + 1] - 1]
            route_distance += distances[route[-1] - 1][0]

        print(f"Salesman {i + 1}: Depot -> {' -> '.join(map(str, route))} -> Depot")
        print(f"Route distance: {route_distance:.2f}")

    # Plot the routes
    plot_routes(coordinates, routes)

    return best_solution, best_fitness, routes


if __name__ == "__main__":
    tsp_data = """NAME : eil51
COMMENT : 51-city problem (Christofides/Eilon)
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

    best_solution, best_fitness, routes = solve_eil51_mtsp(tsp_data, num_salesmen=2)
