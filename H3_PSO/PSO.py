import numpy as np, tqdm, copy
import math
import random
import matplotlib.pyplot as plt


def create_distance_matrix(tsp_content):
    global COORDS
    coords = []
    lines = tsp_content.strip().split('\n')
    for line in lines:
        if line.strip() == 'NODE_COORD_SECTION':
            continue
        if line.strip() == 'EOF':
            break

        try:
            node_id, x, y = map(float, line.strip().split())
            coords.append((x, y))
        except ValueError:
            continue

    coords = np.array(coords)
    COORDS = coords
    n = len(coords)

    # Create distance matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                dist_matrix[i][j] = np.inf
            else:
                # Euclidean distance
                dx = coords[i][0] - coords[j][0]
                dy = coords[i][1] - coords[j][1]
                dist_matrix[i][j] = np.sqrt(dx*dx + dy*dy)

    return dist_matrix


class Particle:
    def __init__(self, num_salesmen: int, distance_matrix: np.ndarray):
        self.num_cities = distance_matrix.shape[0]
        self.num_salesmen = num_salesmen
        self.distance_matrix = distance_matrix

        self.position = self._initialize_position()
        self.velocity = []  # list of swap operations
        self.personal_best_position = self.position.copy()
        self.personal_best_fitness = self.calculate_fitness()[0]

    def _initialize_position(self) -> list:
        cities = list(range(2, self.num_cities + 1))
        random.shuffle(cities)

        # Calculate base route size and remainder
        base_size = self.num_cities // self.num_salesmen
        remainder = self.num_cities % self.num_salesmen

        # Create position with delimiters
        position = []
        start_idx = 0

        for i in range(self.num_salesmen - 1):
            route_size = base_size + int(i < remainder)
            position.extend(cities[start_idx:start_idx + route_size])
            start_idx += route_size
            position.append(0)
        position.extend(cities[start_idx:])

        return position

    def is_valid(self):
        routes =  self._get_routes()
        if len(routes) != self.num_salesmen:
            return False

        used_cities = []
        for route in routes:
            if route[0] != 1 or route[-1] != 1:
                return False
            used_cities.extend(route[1:-1])

        used_cities.sort()
        expected_cities = sorted(c for c in range(1, self.num_cities + 1) if c != 1)
        return used_cities == expected_cities

    def calculate_tour_cost(self, tour: list[int]) -> float:
        cost = 0
        for i in range(len(tour) - 1):
            cost += self.distance_matrix[tour[i]-1][tour[i + 1]-1]
        return cost

    def calculate_fitness(self) -> float:
        global W1, W2
        """
        w1: weight for total cost
        w2: weight for maximum tour cost
        """
        tour_costs = [self.calculate_tour_cost(route) for route in self._get_routes()]
        total_cost = sum(tour_costs)
        max_cost = max(tour_costs)

        avg_cost = total_cost / self.num_salesmen

        # Weighted sum of objectives
        fitness_score = W1 * avg_cost + W2 * max_cost

        return fitness_score, total_cost, max_cost

    def _get_routes(self) -> list[list[int]]:
        routes = []
        current_route = [1]

        for city in self.position:
            if city == 0:
                if current_route:
                    routes.append(current_route + [1])
                current_route = [1]
            else:
                current_route.append(city)

        if current_route:
            routes.append(current_route + [1])

        return routes

    def _get_valid_swaps(self, current: list, target: list) -> list[tuple[int, int]]:
        """Generate valid swaps that move current solution towards target"""
        valid_swaps = []

        # Find positions where elements differ
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                # # Only consider swaps between cities (non-zero elements)
                if current[i] != 0 and current[j] != 0:
                    # Check if swap would move towards target
                    if (current[i] != target[i] and current[j] != target[j] and
                            (current[i] == target[j] or current[j] == target[i])):
                        valid_swaps.append((i, j))

        return valid_swaps

    def _get_random_valid_swaps(self) -> list[tuple[int, int]]:
        # global SWAAPS
        # if SWAAPS is None:
        #     SWAAPS = [(i, j) for i in range(len(self.position)) for j in range(i+1, len(self.position))]

        # Get positions of cities (non-zero elements)
        city_positions = [i for i, x in enumerate(self.position) if x != 0]

        # Generate all possible pairs of city positions
        if len(city_positions) >= 2:
            valid_swaps = [(i, j) for i in city_positions for j in city_positions if i < j]

        return valid_swaps

    def update_velocity(self, global_best_position: list, w: float, c1: float, c2: float):
        self.velocity = []
        SIGMO = 2

        # Cognitive c1
        swaps = self._get_valid_swaps(self.position, self.personal_best_position)
        swap_count = np.clip(int(random.gauss(mu=c1, sigma=SIGMO)), 0, len(swaps))
        if swaps:
            self.velocity.extend(random.sample(swaps, k=max(len(swaps), swap_count)))

        # Social c2
        swaps = self._get_valid_swaps(self.position, global_best_position)
        swap_count = np.clip(int(random.gauss(mu=c2, sigma=SIGMO)), 0, len(swaps))
        if swaps:
            self.velocity.extend(random.sample(swaps, k=swap_count))

        # Inertia - random swaps
        swap_count = max(0, int(random.gauss(mu=w, sigma=SIGMO)))
        swaps = self._get_random_valid_swaps()
        if swaps:
            self.velocity.extend(random.sample(swaps, k=swap_count))

    def update_position(self):
        """Apply velocity (swap sequence) to current position"""
        for i, j in self.velocity:
            self.position[i], self.position[j] = self.position[j], self.position[i]

SWAAPS = None
class MTSPPSO:
    def __init__(self, num_salesmen: int, distances: np.ndarray,
                 num_particles: int = 50, max_iterations: int = 1000):
        self.num_cities = distances.shape[0]
        self.num_salesmen = num_salesmen
        self.distances = distances
        self.num_particles = num_particles
        self.max_iterations = max_iterations

        self.particles = [Particle(num_salesmen, distances)
                          for _ in range(num_particles)]
        self.global_best_position = None
        self.global_best_fitness = float('inf')

        for particle in self.particles:
            fness = particle.calculate_fitness()[0]
            if fness < self.global_best_fitness:
                self.global_best_fitness = fness
                self.global_best_position = particle.position.copy()
                self.gbest_particle = copy.deepcopy(particle)
        self.best_fitness_history = []
        self.gbest_particle = None

    def optimize(self) -> tuple[Particle, float]:
        w_start, w_end = 10, 5
        c1_start, c1_end = 8, 4
        c2_start, c2_end = 3, 4
        pbar = tqdm.trange(self.max_iterations)
        for iteration in pbar:
            w  =  w_start - ( w_start -  w_end) * iteration / self.max_iterations
            c1 = c1_start - (c1_start - c1_end) * iteration / self.max_iterations
            c2 = c2_start - (c2_start - c2_end) * iteration / self.max_iterations

            for particle in self.particles:
                fitness = particle.calculate_fitness()[0]

                if fitness < particle.personal_best_fitness:
                    particle.personal_best_position = particle.position.copy()
                    particle.personal_best_fitness = fitness

                if fitness < self.global_best_fitness:
                    self.global_best_position = particle.position.copy()
                    self.gbest_particle = copy.deepcopy(particle)
                    plot_sol(self.gbest_particle._get_routes())
                    self.global_best_fitness = fitness

            self.best_fitness_history.append(self.global_best_fitness)

            for particle in self.particles:
                particle.update_velocity(self.global_best_position, w, c1, c2)
                particle.update_position()
                if not particle.is_valid():
                    raise Exception("oops")

            pbar.set_postfix_str(f"Best fitness = {self.global_best_fitness:.5f}")

        return self.gbest_particle, self.global_best_fitness


def plot_sol(best):
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
    plt.cla()
    plt.scatter(list(map(lambda x: x[0], COORDS)), list(map(lambda x: x[1], COORDS)))
    for i, tour in enumerate(best):
        color = colors[i % len(colors)]
        plt.plot(list(map(lambda x: COORDS[x-1][0], tour)), list(map(lambda x: COORDS[x-1][1], tour)),
                 color=color,
                 label=f'Salesman {i + 1}')
    plt.pause(0.05)

def pso_mtsp(distances: str, num_salesmen: int, particles, iterations):
    pso = MTSPPSO(
        num_salesmen=num_salesmen,
        distances=distances,
        num_particles=particles,
        max_iterations=iterations
    )

    best_solution, best_fitness = pso.optimize()

    print("\nFinal Solution:")
    print(f"Total fitness: {best_fitness:.2f}")
    print("\nRoutes:")

    fitness_score, total_cost, max_cost = best_solution.calculate_fitness()
    output = ""
    routes = best_solution._get_routes()
    for tour in routes:
        output += f"  {tour},\n"
    print(f"Fitness Score: {fitness_score:.2f}, Total Cost: {total_cost:.2f}, Max Tour Cost: {max_cost:.2f}||")
    print(" ".join([f"{best_solution.calculate_tour_cost(tour):.4f}" for tour in routes]))
    print(f"[\n{output}]")

    plot_sol(routes)
    plt.show()

    return best_solution, best_fitness, routes


if __name__ == "__main__":
    ## git clone https://github.com/mastqe/tsplib
    INSTANCE = 'eil51'
    SALESMEN = 2
    W1, W2 = 0.8, 0.2 # fitness coefs for total cost (1) and min max cost (2)
    distance_matrix = create_distance_matrix(open(f"H3_GA/tsplib/{INSTANCE}.tsp",'rt').read())
    pso_mtsp(distance_matrix, 
        num_salesmen=SALESMEN, 
        particles=1500, 
        iterations=1500
    )
