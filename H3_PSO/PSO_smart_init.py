import random

import copy
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import os
import time
import json
from datetime import datetime

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
        aang = list(set(map(lambda x: x[1], angles)))
        group_start_angle_delims = [x for x in random.sample(aang, self.num_salesmen)]
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
        sigma = 1

        # Get solution length (excluding depot visits)
        solution_length = len(global_best_position)

        # Scale the coefficients based on solution length
        # Using square root to prevent too many swaps for very large problems
        # length_factor = np.sqrt(solution_length) / 4  # Divide by 4 to keep numbers reasonable

        cognitive_swaps = get_valid_swaps(self.position, self.personal_best_position)
        social_swaps = get_valid_swaps(self.position, global_best_position)
        random_swaps = self._get_random_valid_swaps()
        scaled_c1 = c1 * len(cognitive_swaps)
        scaled_c2 = c2 * len(social_swaps)
        scaled_w = w * len(random_swaps)

        # Get cognitive component swaps (toward personal best)
        if cognitive_swaps:
            cognitive_count = np.clip(
                int(random.gauss(mu=scaled_c1, sigma=sigma)), 0, len(cognitive_swaps)
            )
            if cognitive_count > 0:
                idxs = random.sample(range(len(cognitive_swaps)), k=cognitive_count)
                idxs.sort()
                self.velocity.extend(
                    [cognitive_swaps[idx] for idx in idxs]
                )

        # Get social component swaps (toward global best)
        if social_swaps:
            social_count = np.clip(
                int(random.gauss(mu=scaled_c2, sigma=sigma)), 0, len(social_swaps)
            )
            if social_count > 0:
                idxs = random.sample(range(len(social_swaps)), k=social_count)
                idxs.sort()
                self.velocity.extend(
                    [social_swaps[idx] for idx in idxs]
                )

        # Get random swaps for exploration
        if random_swaps:
            random_count = np.clip(
                int(random.gauss(mu=scaled_w, sigma=sigma)), 0, len(random_swaps)
            )
            if random_count > 0:
                self.velocity.extend(
                    random.sample(random_swaps, k=random_count)
                )
        return cognitive_count if cognitive_swaps else None, social_count if social_swaps else None, random_count if random_swaps else None

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
        w_start, w_end = 0.0005, 0.00015
        c1_start, c1_end = 0.3, 0.15
        c2_start, c2_end = 0.15, 0.5
        pbar = tqdm.trange(self.max_iterations, file=None)
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
                c1s, c2s, ws = particle.update_velocity(self.global_best_position, w, c1, c2)
                pbar.set_description_str(f"c1 {f'{c1s}'.rjust(5)}, c2 {f'{c2s}'.rjust(5)}, w {f'{ws}'.rjust(5)}")
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
    # plt.show()

    return best_solution, best_fitness, routes


def run_batch_experiments(file_paths, salesmen_counts=[2, 3, 5, 7], num_trials=10,
                          num_particles=1000, num_iterations=1000):
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"mtsp_results_{timestamp}"
    os.makedirs(base_output_dir, exist_ok=True)

    # Store overall results
    all_results = {}

    for file_path in file_paths:
        problem_name = os.path.basename(file_path).split('.')[0]
        problem_dir = os.path.join(base_output_dir, problem_name)
        os.makedirs(problem_dir, exist_ok=True)

        print(f"\nProcessing {problem_name}")
        print("=" * 50)

        # Read the TSP file content
        with open(file_path, 'r') as f:
            tsp_content = f.read()

        distance_matrix = create_distance_matrix(tsp_content)
        problem_results = {}

        for num_salesmen in salesmen_counts:
            print(f"\nRunning with {num_salesmen} salesmen")
            print("-" * 30)

            # Create directory for this salesmen configuration
            salesmen_dir = os.path.join(problem_dir, f"salesmen_{num_salesmen}")
            os.makedirs(salesmen_dir, exist_ok=True)

            trials_results = []

            for trial in range(num_trials):
                print(f"Trial {trial + 1}/{num_trials}")

                # Create trial directory
                trial_dir = os.path.join(salesmen_dir, f"trial_{trial + 1}")
                os.makedirs(trial_dir, exist_ok=True)

                # Run PSO
                start_time = time.time()
                best_solution, best_fitness, routes = pso_mtsp(
                    distance_matrix,
                    num_salesmen=num_salesmen,
                    particles=num_particles,
                    iterations=num_iterations
                )
                end_time = time.time()

                # Calculate route lengths
                route_lengths = [best_solution.calculate_tour_cost(route) for route in routes]

                # Save plot
                plt.savefig(os.path.join(trial_dir, 'solution_plot.png'))
                plt.close()

                # Prepare trial results
                trial_results = {
                    "trial_number": trial + 1,
                    "max_tour_length": best_fitness,
                    "total_length": sum(route_lengths),
                    "individual_tour_lengths": route_lengths,
                    "routes": routes,
                    "computation_time": end_time - start_time
                }

                # Save trial results to JSON
                with open(os.path.join(trial_dir, 'results.json'), 'w') as f:
                    json.dump(trial_results, f, indent=2)

                trials_results.append(trial_results)

            # Calculate and save summary statistics for this salesmen configuration
            summary_stats = {
                "num_salesmen": num_salesmen,
                "num_trials": num_trials,
                "best_max_length": min(r["max_tour_length"] for r in trials_results),
                "average_max_length": sum(r["max_tour_length"] for r in trials_results) / num_trials,
                "best_total_length": min(r["total_length"] for r in trials_results),
                "average_total_length": sum(r["total_length"] for r in trials_results) / num_trials,
                "average_computation_time": sum(r["computation_time"] for r in trials_results) / num_trials,
                "best_trial": min(range(len(trials_results)),
                                  key=lambda i: trials_results[i]["max_tour_length"]) + 1
            }

            # Save summary statistics for this salesmen configuration
            with open(os.path.join(salesmen_dir, 'summary_stats.json'), 'w') as f:
                json.dump(summary_stats, f, indent=2)

            problem_results[f"salesmen_{num_salesmen}"] = summary_stats

        # Save problem summary
        with open(os.path.join(problem_dir, 'problem_summary.json'), 'w') as f:
            json.dump(problem_results, f, indent=2)

        all_results[problem_name] = problem_results

    # Save overall results
    with open(os.path.join(base_output_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    return base_output_dir


if __name__ == "__main__":
    # Define the file paths
    file_paths = [
        r"D:\Facultate\tsplib\eil51.tsp",
        r"D:\Facultate\tsplib\berlin52.tsp",
        r"D:\Facultate\tsplib\eil76.tsp",
        r"D:\Facultate\tsplib\rat99.tsp"
    ]

    # Run the experiments
    output_dir = run_batch_experiments(file_paths)
    print(f"\nAll results have been saved to: {output_dir}")