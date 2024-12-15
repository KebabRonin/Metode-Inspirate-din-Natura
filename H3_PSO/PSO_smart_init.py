import random

import copy
import matplotlib.pyplot as plt
import numpy as np
import tqdm


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
                dist_matrix[i][j] = np.sqrt(dx * dx + dy * dy)

    return dist_matrix


def get_valid_swaps(source, target):
    if len(source) != len(target) or sorted(source) != sorted(target):
        raise ValueError("Lists must be the same length and contain the same elements")

    n = len(source)

    # Create a mapping of values to their positions in target
    pos_map = {val: idx for idx, val in enumerate(target)}

    # Create a list that represents where each element needs to move
    # For each position i, target_positions[i] tells us where the element at
    # position i in source needs to end up (based on target)
    target_positions = [pos_map[val] for val in source]

    # Keep track of which positions we've already processed
    visited = [False] * n

    # Store the swap operations
    swaps = []

    scopy = copy.deepcopy(source)
    # Process each position
    for start in range(n):
        # Skip if we've already handled this position or
        # if the element is already in the correct spot
        if visited[start] or target_positions[start] == start:
            continue

        # Follow the cycle starting from this position
        current = start
        cycle = []

        while not visited[current]:
            visited[current] = True
            cycle.append(current)
            current = target_positions[current]

        # Generate the swaps for this cycle
        for i in range(len(cycle) - 1):
            pos1 = cycle[i]
            pos2 = cycle[i + 1]
            # Record the values that need to be swapped
            swaps.append((pos1, pos2))

            # Update source to reflect the swap
            scopy[pos1], scopy[pos2] = scopy[pos2], scopy[pos1]

    return swaps


class Particle:
    def __init__(self, num_salesmen: int, distance_matrix: np.ndarray):
        self.num_cities = distance_matrix.shape[0]
        self.num_salesmen = num_salesmen
        self.distance_matrix = distance_matrix

        self.position = self._initialize_position()
        self.velocity = []  # list of swap operations
        self.personal_best_position = self.position.copy()
        self.personal_best_fitness, _, _ = self.calculate_fitness()
        if not self.is_valid():
            raise Exception("Invalid initial position")

    def _initialize_position(self) -> list:
        position = []

        # Sort the cities by angle from the depot
        depot_coords = COORDS[0]
        cities = list(range(1, self.num_cities))
        angles = [(c, (np.pi) + np.arctan2((depot_coords[0] - COORDS[c][0]), (depot_coords[1] - COORDS[c][1]))) for c in cities]
        angles.sort(key=lambda x: x[1])
        # Then divide the cities into n_salesmen groups
        group_start_angle_delims = [x[1] for x in random.sample(angles, self.num_salesmen)]
        group_start_angle_delims.sort()
        for s in range(self.num_salesmen-1):
            #V2: each group gets fixed degrees
            start_angle = group_start_angle_delims[s]
            end_angle = group_start_angle_delims[s + 1]
            group_cities = list(filter(lambda x: start_angle <= x[1] < end_angle, angles))
            random.shuffle(group_cities)
            position.extend([(c[0] + 1) for c in group_cities] + [0])
        start_angle = group_start_angle_delims[-1]
        end_angle = group_start_angle_delims[0]
        group_cities = list(filter(lambda x: not (end_angle <= x[1] < start_angle), angles))
        random.shuffle(group_cities)
        position.extend([(c[0] + 1) for c in group_cities])

        return position

    def is_valid(self):
        routes = self._get_routes()
        if len(routes) != self.num_salesmen:
            return False
        for route in routes:
            if len(route) <= 2:
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
            cost += self.distance_matrix[tour[i] - 1][tour[i + 1] - 1]
        return cost

    def calculate_fitness(self):
        routes = self._get_routes()
        tour_costs = [self.calculate_tour_cost(route) for route in routes]

        # Calculate metrics
        total_cost = sum(tour_costs)
        max_cost = max(tour_costs)  # This is Q in the formulation

        # In the MinMax formulation, our fitness is simply Q (the longest tour)
        # We don't use weighted averages anymore
        fitness_score = max_cost

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

    def _get_random_valid_swaps(self):

        city_positions = [i for i, x in enumerate(self.position) if x != 0]

        # Initialize our results list
        valid_swaps = []

        # Generate swaps only if we have at least 2 cities
        if len(city_positions) >= 2:
            # For each city position
            for i in range(len(city_positions)):
                # Consider swaps with all subsequent city positions
                for j in range(i + 1, len(city_positions)):
                    pos1 = city_positions[i]
                    pos2 = city_positions[j]

                    # Verify that both positions contain actual cities
                    if self.position[pos1] != 0 and self.position[pos2] != 0:
                        valid_swaps.append((pos1, pos2))

        return valid_swaps

    def update_velocity(self, global_best_position, w, c1, c2):

        self.velocity = []

        sigma = 2

        cognitive_swaps = get_valid_swaps(self.position, self.personal_best_position)
        if cognitive_swaps:
            cognitive_count = max(0, min(
                int(random.gauss(mu=c1, sigma=sigma)),
                len(cognitive_swaps)
            ))
            idxs = random.sample(range(len(cognitive_swaps)), k=cognitive_count)
            idxs.sort()
            self.velocity.extend(
                [cognitive_swaps[idx] for idx in idxs]
            )

        # Next, add swaps that move us toward global best (social component)
        social_swaps = get_valid_swaps(self.position, global_best_position)
        if social_swaps:
            social_count = max(0, min(
                int(random.gauss(mu=c2, sigma=sigma)),
                len(social_swaps)
            ))
            idxs = random.sample(range(len(social_swaps)), k=social_count)
            idxs.sort()
            self.velocity.extend(
                [social_swaps[idx] for idx in idxs]
            )

        # Finally, add random swaps for exploration (inertia component)
        random_swaps = self._get_random_valid_swaps()
        if random_swaps:
            # The inertia weight w determines how many random swaps to include
            random_count = max(0, min(
                int(random.gauss(mu=w, sigma=sigma)),
                len(random_swaps)
            ))
            self.velocity.extend(
                random.sample(random_swaps, k=random_count)
            )

    def update_position(self):
        """Apply velocity (swap sequence) to current position"""
        def was_ok_move(i, l):
            if l[i] == 0:
                if i == 0 or i == len(l) - 1 or l[i - 1] == 0 or l[i + 1] == 0:
                    return False
            return True
        # pp = copy.deepcopy(self.position)
        for i, j in self.velocity:
            self.position[i], self.position[j] = self.position[j], self.position[i]
            if not (was_ok_move(i, self.position) and was_ok_move(j, self.position)):
                self.position[i], self.position[j] = self.position[j], self.position[i]


class MTSPPSO:
    def __init__(self, num_salesmen, distances,
                 num_particles=50, max_iterations=1000):
        self.num_cities = distances.shape[0]
        self.num_salesmen = num_salesmen
        self.distances = distances
        self.num_particles = num_particles
        self.max_iterations = max_iterations

        self.particles = [Particle(num_salesmen, distances)
                          for _ in range(num_particles)]
        self.global_best_position = None
        self.gbest_particle = None
        self.global_best_fitness = float('inf')

        for particle in self.particles:
            fness = particle.calculate_fitness()[0]
            if fness < self.global_best_fitness:
                self.global_best_fitness = fness
                self.global_best_position = particle.position.copy()
                self.gbest_particle = copy.deepcopy(particle)
        self.best_fitness_history = []

    def optimize(self):
        w_start, w_end = 3, 1
        c1_start, c1_end = 5, 2
        c2_start, c2_end = 10, 4
        pbar = tqdm.trange(self.max_iterations)
        besst = self.gbest_particle.calculate_fitness()
        fitnesses = [p.calculate_fitness()[0] for p in self.particles]
        pbar.set_postfix_str(f"Fit:{besst[0]:>10.5f}|Total:{besst[1]:>10.5f}|Max:{besst[2]:>10.5f}|Var:{np.var(fitnesses):>10.5f}")
        for iteration in pbar:
            w = w_start - (w_start - w_end) * iteration / self.max_iterations
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
                    besst = self.gbest_particle.calculate_fitness()
            fitnesses = [p.calculate_fitness()[0] for p in self.particles]
            pbar.set_postfix_str(f"Fit:{besst[0]:>10.5f}|Total:{besst[1]:>10.5f}|Max:{besst[2]:>10.5f}|Var:{np.var(fitnesses):>10.5f}")

            self.best_fitness_history.append(self.global_best_fitness)

            for particle in self.particles:
                particle.update_velocity(self.global_best_position, w, c1, c2)
                particle.update_position()

        return self.gbest_particle, self.global_best_fitness


def plot_sol(best):
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
    plt.cla()
    plt.scatter(list(map(lambda x: x[0], COORDS)), list(map(lambda x: x[1], COORDS)))
    for i, tour in enumerate(best):
        color = colors[i % len(colors)]
        plt.plot(list(map(lambda x: COORDS[x - 1][0], tour)), list(map(lambda x: COORDS[x - 1][1], tour)),
                 color=color,
                 label=f'Salesman {i + 1}')
    plt.pause(0.05)


def pso_mtsp(distances, num_salesmen, particles, iterations):
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
    distance_matrix = create_distance_matrix(open(r"H3_GA\tsplib\eil51.tsp", 'rt').read())
    pso_mtsp(distance_matrix,
             num_salesmen=2,
             particles=500,
             iterations=1000
             )
