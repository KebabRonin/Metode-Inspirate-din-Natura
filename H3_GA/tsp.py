
"""For PSO: Use the course permutation repr with the probability one to pick which one to which salesman
	"""
import numpy as np, random, tqdm, copy, math, time
import matplotlib.pyplot as plt
COORDS = None
def create_distance_matrix(tsp_content):
    global COORDS
    coords = []
    lines = tsp_content.strip().split('\n')
    for line in lines:
        if line.strip() == 'NODE_COORD_SECTION':
            parsing_coords = True
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


class MTSPIndividual:
    def __init__(self, distance_matrix: np.ndarray, n_salesmen: int, depot: int = 1):
        self.distance_matrix = distance_matrix
        self.n_cities = len(distance_matrix)
        self.n_salesmen = n_salesmen
        self.depot = depot
        self.chromosomes: list[list[int]] = []

    def initialize_random(self) -> None:
        cities = list(range(1, self.n_cities + 1))
        cities.remove(self.depot)
        available_cities = cities.copy()

        n_cities_for_salesman = len(available_cities) // (self.n_salesmen)
        for i in range(self.n_salesmen):
            selected_cities = random.sample(available_cities, n_cities_for_salesman)
            for city in selected_cities:
                available_cities.remove(city)
            chromosome = [self.depot] + selected_cities + [self.depot]
            self.chromosomes.append(chromosome)
        if len(available_cities) > 0:
            self.chromosomes[-1] = self.chromosomes[-1][:-1] + available_cities + [self.depot]

    def calculate_tour_cost(self, tour: list[int]) -> float:
        cost = 0
        for i in range(len(tour) - 1):
            cost += self.distance_matrix[tour[i]-1][tour[i + 1]-1]
        return cost

    def is_valid(self) -> bool:
        if len(self.chromosomes) != self.n_salesmen:
            return False

        used_cities = []
        for chromosome in self.chromosomes:
            if chromosome[0] != self.depot or chromosome[-1] != self.depot:
                return False
            used_cities.extend(chromosome[1:-1])

        used_cities.sort()
        expected_cities = sorted(c for c in range(1, self.n_cities + 1) if c != self.depot)
        return used_cities == expected_cities

    def calculate_fitness(self, w1=0.5, w2=0.5) -> float:
        """
        w1: weight for total cost
        w2: weight for maximum tour cost
        """
        tour_costs = [self.calculate_tour_cost(tour) for tour in self.chromosomes]
        total_cost = sum(tour_costs)
        max_cost = max(tour_costs)

        avg_cost = total_cost / len(self.chromosomes)

        # Weighted sum of objectives
        fitness_score = w1 * avg_cost + w2 * max_cost

        return fitness_score, total_cost, max_cost

    def calculate_tour_cost(self, tour: list[int]) -> float:
        return sum(self.distance_matrix[tour[i]-1][tour[i+1]-1]
                  for i in range(len(tour)-1))

    def __str__(self) -> str:
        fitness_score, total_cost, max_cost = self.calculate_fitness()
        output = f"Fitness Score: {fitness_score:.2f}, Total Cost: {total_cost:.2f}, Max Tour Cost: {max_cost:.2f}\n"
        for i, tour in enumerate(self.chromosomes):
            output += f"Salesman {i+1}: {tour}\n"
        return output

    def in_route_mutation(self) -> None:
        chromosome_idx = random.randint(0, len(self.chromosomes) - 1)
        chromosome = self.chromosomes[chromosome_idx]

        if len(chromosome) < 4:
            return False
        start = random.randint(1, len(chromosome) - 3)
        leng = random.sample(list(range(2, len(chromosome) - start)), 1)[0]

        # Invert subsection
        self.chromosomes[chromosome_idx][start:start + leng] = \
            self.chromosomes[chromosome_idx][start:start + leng][::-1]
        return True

    def cross_route_mutation(self):
        """Move a city to another salesman"""
        salesmen = list(range(self.n_salesmen))
        s1, s2 = random.sample(salesmen, 2)
        if len(self.chromosomes[s1]) <= 3:
            return False
        cutoff = random.randint(1, len(self.chromosomes[s1][1:-1]))
        # leng = random.sample(list(range(1, len(self.chromosomes[s1]) - cutoff)), 1)[0]
        # insertion_point =random.randint(1, len(self.chromosomes[s2][1:-1]))
        self.chromosomes[s2].insert(-1, self.chromosomes[s1].pop(cutoff))
        # print(self.chromosomes[s1], '\n', self.chromosomes[s2])
        # self.chromosomes[s2] = self.chromosomes[s2][:insertion_point] + self.chromosomes[s1][cutoff:cutoff+leng] + self.chromosomes[s2][insertion_point:]
        # self.chromosomes[s1] = self.chromosomes[s1][:cutoff] + self.chromosomes[s1][cutoff+leng:]
        # print(self.chromosomes[s1], '\n', self.chromosomes[s2])
        return True

    def mutation(self, mutation_rate: float):
        if random.random() > mutation_rate:
            ok = False
            while not ok:
                if random.random() > 0.5:
                    ok = self.in_route_mutation()
                else:
                    ok = self.cross_route_mutation()
                break

class MTSPPopulation:
    def __init__(self, distance_matrix: np.ndarray, n_salesmen: int, pop_size: int, depot: int = 1):
        self.distance_matrix = distance_matrix
        self.n_salesmen = n_salesmen
        self.pop_size = pop_size
        self.depot = depot
        self.solutions: list[MTSPIndividual] = []
        self.initialize_population()

    def initialize_population(self) -> None:
        for _ in range(self.pop_size):
            solution = MTSPIndividual(self.distance_matrix, self.n_salesmen, self.depot)
            solution.initialize_random()
            self.solutions.append(solution)

    def get_best_solution(self) -> MTSPIndividual:
        return min(self.solutions, key=lambda x: x.calculate_fitness()[0])

    def tournament_selection(self, tournament_size: int) -> MTSPIndividual:
        tournament = random.sample(self.solutions, tournament_size)
        return min(tournament, key=lambda x: x.calculate_fitness()[0])

    def replace_solution(self, index: int, new_solution: MTSPIndividual) -> None:
        self.solutions[index] = new_solution

    def get_population_stats(self) -> tuple[float, float, float, float]:
        fitnesses = [sol.calculate_fitness() for sol in self.solutions]
        fitness_scores = [f[0] for f in fitnesses]
        total_costs = [f[1] for f in fitnesses]
        max_costs = [f[2] for f in fitnesses]
        return min(fitness_scores), np.mean(fitness_scores), min(total_costs), np.mean(total_costs), min(max_costs), np.mean(max_costs)

    def crossover(self, parent1: MTSPIndividual, parent2: MTSPIndividual) -> MTSPIndividual:
        """order crossover"""
        offspring = MTSPIndividual(self.distance_matrix, self.n_salesmen, self.depot)
        offspring.chromosomes = []

        for salesman_idx in range(self.n_salesmen):
            p1_route = parent1.chromosomes[salesman_idx]
            p2_route = parent2.chromosomes[salesman_idx]
            route_length = len(p1_route)

            start = random.randint(1, route_length - 4)
            end = random.randint(start, route_length - 2)

            offspring_route = [-1] * route_length
            offspring_route[0] = offspring_route[-1] = self.depot

            offspring_route[start:end + 1] = p1_route[start:end + 1]

            used_cities = set(offspring_route[start:end + 1])
            remaining_cities = [city for city in p2_route
                              if city not in used_cities and city != self.depot]
            try:
                idx = 0
                for i in range(1, start):
                    offspring_route[i] = remaining_cities[idx]
                    idx += 1

                for i in range(end + 1, route_length - 1):
                    offspring_route[i] = remaining_cities[idx]
                    idx += 1
            except IndexError:
                pass

            offspring.chromosomes.append(list(filter(lambda x: x != -1, offspring_route)))

        # Fix the generated offspring
        city_positions = {city: [(salesman_idx, i) for salesman_idx, salesman_tour in enumerate(offspring.chromosomes) for i, tcity in enumerate(salesman_tour) if tcity == city] for city in range(1, len(self.distance_matrix) + 1) if city != self.depot}
        for city in city_positions.keys():
            if len(city_positions[city]) == 1:
                continue
            elif len(city_positions[city]) > 1:
                leave_at = random.sample(city_positions[city], 1)[0]
                for pos in city_positions[city]:
                    if pos != leave_at:
                        offspring.chromosomes[pos[0]].pop(pos[1])
                city_positions = {city: [(salesman_idx, i) for salesman_idx, salesman_tour in enumerate(offspring.chromosomes) for i, tcity in enumerate(salesman_tour) if tcity == city] for city in range(1, len(self.distance_matrix) + 1) if city != self.depot}
            elif len(city_positions[city]) < 1:
                # Change this?
                salesman_idx = random.randint(0, self.n_salesmen - 1)
                city_position = random.randint(1, len(offspring.chromosomes[salesman_idx]) - 2)
                offspring.chromosomes[salesman_idx].insert(city_position, city)
                city_positions = {city: [(salesman_idx, i) for salesman_idx, salesman_tour in enumerate(offspring.chromosomes) for i, tcity in enumerate(salesman_tour) if tcity == city] for city in range(1, len(self.distance_matrix) + 1) if city != self.depot}
        return offspring

    def breed_new_population(self, tournament_size: int, crossover_rate: float) -> list[MTSPIndividual]:
        new_solutions = [self.tournament_selection(tournament_size) for _ in range(int(self.pop_size*crossover_rate))]
        while len(new_solutions) < self.pop_size:
            parent1 = self.tournament_selection(tournament_size)
            parent2 = self.tournament_selection(tournament_size)
            offspring = self.crossover(parent1, parent2)
            if not offspring.is_valid():
                print(set(sum(offspring.chromosomes, [])))
                raise Exception(f"Invalid offspring: {offspring}")
            new_solutions.append(offspring)
        return new_solutions

    def __str__(self) -> str:
        a = ""
        for i, sol in enumerate(self.solutions):
            a += f"* Solution {i+1}: {sol}"
        return a

def plot_sol(best):
    plt.cla()
    plt.scatter(list(map(lambda x: x[0], COORDS)), list(map(lambda x: x[1], COORDS)))
    for tour in best.chromosomes:
        plt.plot(list(map(lambda x: COORDS[x-1][0], tour)), list(map(lambda x: COORDS[x-1][1], tour)))
    plt.pause(0.05)

def run_ag(distance_matrix, n_salesmen, pop_size, max_gens, mutation_rate, crossover_rate):
    population = MTSPPopulation(distance_matrix, n_salesmen=n_salesmen, pop_size=pop_size)
    pbar = tqdm.trange(max_gens, desc="AG generations", position=0)
    hist = []
    mrate = mutation_rate
    crate = crossover_rate
    best = copy.deepcopy(population.get_best_solution())
    liter = 0
    for t in pbar:
        # Crossover + Selection
        population.solutions = population.breed_new_population(tournament_size=10, crossover_rate=crate)
        # Mutation
        [sol.mutation(mrate) for sol in population.solutions]
        if population.get_best_solution().calculate_fitness()[1] < best.calculate_fitness()[1]:
            best = copy.deepcopy(population.get_best_solution())
            plot_sol(best)
        hist.append(population.get_best_solution().calculate_fitness()[1])
        if t - liter > 20 and np.var(hist[-20:]) < 5:
            print(mrate, crate)
            mrate = 1
            crate = 0.9
            liter = t
        else:
            if mrate > mutation_rate:
                mrate *= 0.95
                if mrate < mutation_rate:
                    mrate = mutation_rate
            if crate > crossover_rate:
                crate *= 0.95
                if crate < crossover_rate:
                    crate = crossover_rate
        pbar.set_postfix_str(f"Best solution: {population.get_best_solution().calculate_fitness()[1]:.5f}")
    plt.savefig(f"{INSTANCE}_{time.time()}_{POPSIZE}pop_path.png")
    plt.show()
    plt.plot(hist)
    plt.savefig(f"{time.time()}_hist.png")
    plt.show()
    print(f"Best solution in all:\n{best}")

    min_fitness_scores, avg_fitness_scores, min_total, avg_total, min_max, avg_max = population.get_population_stats()
    print(f"Final Population stats:\nMin total cost: {min_total:.2f}\nAvg total cost: {avg_total:.2f}")
    print(f"Min max cost: {min_max:.2f}\nAvg max cost: {avg_max:.2f}")
    print(f"Min fitness: {min_fitness_scores:.2f}\nAvg fitness: {avg_fitness_scores:.2f}")



## git clone https://github.com/mastqe/tsplib
INSTANCE = 'eil51'
SALESMEN = 2
POPSIZE = 500
GENERATIONS = 1500
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.5
run_ag(
    create_distance_matrix(open(f'H3_GA/tsplib/{INSTANCE}.tsp', 'rt').read()),
    SALESMEN,
    POPSIZE,
    GENERATIONS,
    MUTATION_RATE,
    CROSSOVER_RATE
)