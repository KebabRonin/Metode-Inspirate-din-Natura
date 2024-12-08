import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import random


class MTSP_GA:
    def __init__(self, coords: np.ndarray, num_salesmen: int, pop_size: int = 100,
                 max_generations: int = 1000, tournament_size: int = 5,
                 crossover_rate: float = 0.85, mutation_rate: float = 0.1):
        self.coords = coords
        self.num_cities = len(coords)
        self.num_salesmen = num_salesmen
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.distance_matrix = self._compute_distance_matrix()

    def _compute_distance_matrix(self) -> np.ndarray:
        """Compute Euclidean distance matrix."""
        dist_matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                dist = np.sqrt(np.sum((self.coords[i] - self.coords[j]) ** 2))
                dist_matrix[i, j] = dist_matrix[j, i] = dist
        return dist_matrix

    def _generate_initial_population(self) -> List[List[int]]:
        """Generate initial population using group theory approach."""
        population = []
        for _ in range(self.pop_size):
            # Generate a permutation using modular arithmetic
            base_perm = list(range(1, self.num_cities))
            n = len(base_perm)
            k = random.randint(1, n - 1)
            perm = [(i * k) % n + 1 for i in range(n)]
            # Insert depot (city 0) at random positions to create m sub-tours
            positions = sorted(random.sample(range(len(perm) + 1), self.num_salesmen - 1))
            for pos in positions:
                perm.insert(pos, 0)
            perm = [0] + perm + [0]
            population.append(perm)
        return population

    def _calculate_tour_costs(self, chromosome: List[int]) -> Tuple[float, float]:
        """Calculate total cost and maximum individual tour cost."""
        # Split chromosome into tours
        tours = []
        current_tour = []
        for city in chromosome:
            current_tour.append(city)
            if city == 0 and len(current_tour) > 1:  # Complete tour
                tours.append(current_tour)
                current_tour = [0]

        # Calculate costs
        tour_costs = []
        for tour in tours:
            cost = sum(self.distance_matrix[tour[i]][tour[i + 1]]
                       for i in range(len(tour) - 1))
            tour_costs.append(cost)

        return sum(tour_costs), max(tour_costs)

    def _tournament_selection(self, population: List[List[int]],
                              fitness_values: List[Tuple[float, float]]) -> List[int]:
        """Select chromosome using tournament selection."""
        tournament_idx = random.sample(range(len(population)), self.tournament_size)
        tournament_fitness = [fitness_values[i] for i in tournament_idx]
        # Select based on total cost (first objective)
        winner_idx = tournament_idx[tournament_fitness.index(min(tournament_fitness))]
        return population[winner_idx]

    def _crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Implement crossover operator with proper depot handling."""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        # Create offspring
        child1 = [0] * len(parent1)
        child2 = [0] * len(parent2)

        # Keep depot positions fixed
        depot_positions1 = [i for i, x in enumerate(parent1) if x == 0]
        depot_positions2 = [i for i, x in enumerate(parent2) if x == 0]

        for pos in depot_positions1:
            child1[pos] = 0
        for pos in depot_positions2:
            child2[pos] = 0

        # Get non-depot cities from parents
        cities1 = [x for x in parent1 if x != 0]
        cities2 = [x for x in parent2 if x != 0]

        # Create mapping of positions
        non_depot_positions1 = [i for i, x in enumerate(parent1) if x != 0]
        non_depot_positions2 = [i for i, x in enumerate(parent2) if x != 0]

        # Fill children with mapped cities
        for pos1, city in zip(non_depot_positions1, cities2):
            child1[pos1] = city
        for pos2, city in zip(non_depot_positions2, cities1):
            child2[pos2] = city

        return child1, child2

    def _mutation(self, chromosome: List[int]) -> List[int]:
        """Apply in-route mutation operator."""
        if random.random() > self.mutation_rate:
            return chromosome

        # Find sub-tours
        tour_breaks = [i for i, city in enumerate(chromosome) if city == 0]

        # Select random sub-tour
        tour_idx = random.randint(0, len(tour_breaks) - 2)
        start = tour_breaks[tour_idx]
        end = tour_breaks[tour_idx + 1]

        # Select random subsection within sub-tour
        if end - start > 3:  # Only mutate if sub-tour has enough cities
            mut_start = random.randint(start + 1, end - 2)
            mut_end = random.randint(mut_start + 1, end - 1)
            # Invert subsection
            chromosome[mut_start:mut_end + 1] = chromosome[mut_start:mut_end + 1][::-1]

        return chromosome

    def solve(self) -> Tuple[List[int], float, float]:
        """Main GA loop."""
        # Initialize population
        population = self._generate_initial_population()
        best_chromosome = None
        best_total_cost = float('inf')
        best_max_cost = float('inf')

        for generation in range(self.max_generations):
            # Evaluate fitness
            fitness_values = [self._calculate_tour_costs(chrom)
                              for chrom in population]

            # Update best solution
            for chrom, (total_cost, max_cost) in zip(population, fitness_values):
                if total_cost < best_total_cost:
                    best_chromosome = chrom.copy()
                    best_total_cost = total_cost
                    best_max_cost = max_cost

            # Create new population
            new_population = []
            while len(new_population) < self.pop_size:
                # Selection
                parent1 = self._tournament_selection(population, fitness_values)
                parent2 = self._tournament_selection(population, fitness_values)

                # Crossover
                child1, child2 = self._crossover(parent1, parent2)

                # Mutation
                child1 = self._mutation(child1)
                child2 = self._mutation(child2)

                new_population.extend([child1, child2])

            population = new_population[:self.pop_size]

            if generation % 100 == 0:
                print(f"Generation {generation}: Best Total Cost = {best_total_cost:.2f}, "
                      f"Best Max Tour Cost = {best_max_cost:.2f}")

        return best_chromosome, best_total_cost, best_max_cost

    def plot_solution(self, chromosome: List[int]) -> None:
        """Plot the MTSP solution."""
        plt.figure(figsize=(10, 10))

        # Plot cities
        plt.scatter(self.coords[:, 0], self.coords[:, 1], c='red', s=50)

        # Plot depot
        plt.scatter(self.coords[0, 0], self.coords[0, 1], c='green', s=100, label='Depot')

        # Plot tours
        colors = plt.cm.rainbow(np.linspace(0, 1, self.num_salesmen))
        current_tour = []
        color_idx = 0

        for city in chromosome:
            current_tour.append(city)
            if city == 0 and len(current_tour) > 1:  # Complete tour
                tour_coords = self.coords[current_tour]
                plt.plot(tour_coords[:, 0], tour_coords[:, 1],
                         c=colors[color_idx], label=f'Salesman {color_idx + 1}')
                current_tour = [0]
                color_idx += 1

        plt.legend()
        plt.title('MTSP Solution')
        plt.show()


# Parse the provided data
def parse_tsp_data(data_str: str) -> np.ndarray:
    """Parse TSP data string and return coordinates array."""
    lines = data_str.strip().split('\n')
    coords = []
    start_parsing = False

    for line in lines:
        if line.strip() == 'NODE_COORD_SECTION':
            start_parsing = True
            continue
        if line.strip() == 'EOF':
            break
        if start_parsing:
            _, x, y = map(float, line.strip().split())
            coords.append([x, y])

    return np.array(coords)


# Example usage
if __name__ == "__main__":
    # Your provided data string goes here
    data_str = """NAME : eil51
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

    coords = parse_tsp_data(data_str)

    # Create and run GA
    ga = MTSP_GA(coords, num_salesmen=2, pop_size=100, max_generations=1000)
    best_solution, total_cost, max_cost = ga.solve()

    print(f"\nBest Solution Found:")
    print(f"Total Cost: {total_cost:.2f}")
    print(f"Maximum Tour Cost: {max_cost:.2f}")

    # Plot solution
    ga.plot_solution(best_solution)