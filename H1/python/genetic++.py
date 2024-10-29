import math
import random

import tqdm

import functions


def decimal(bitstring: list) -> int:
    l = len(bitstring)
    return sum([bitstring[l - 1 - i] * 2 ** i for i in range(l)])


# Get a random bitstring of length n
def get_random_bitstring(n: int) -> list:
    return [random.choice([0, 1]) for _ in range(n)]


# Decoding a bitstring into a real number using the given formula
def decode(bitstring: list, b_min: float, b_max: float, n_bits: int) -> float:
    bit_value = decimal(bitstring)
    return b_min + (bit_value * (b_max - b_min)) / ((2 ** n_bits) - 1)


def decode_chrom(bitstring: list, b_min: float, b_max: float, n_bits) -> list[float]:
    return [decode(bitstring[i * n_bits:(i + 1) * n_bits], b_min, b_max, n_bits) for i in
            range(len(bitstring) // n_bits)]


# Define the value space for the problem
def infer_value_space(func_obj, precision: int, input_dims: int):
    b_min, b_max = func_obj['bounds']
    # Number of intervals between [b_min, b_max]
    N = (b_max - b_min) * (10 ** precision)
    n_bits = math.ceil(math.log2(N))  # Number of bits needed to represent the intervals
    sol_len = input_dims * n_bits  # Total length of the solution bitstring
    return {'sol_len': sol_len, 'n_bits': n_bits, 'b_min': b_min, 'b_max': b_max}


def generate_starting_population(pop_size: int, sol_len: int) -> list:
    return [get_random_bitstring(sol_len) for _ in range(pop_size)]


def evaluate_population(population: list, func, problem_repr: dict) -> list:
    fitness = []

    for individual in population:
        decoded = decode_chrom(individual, problem_repr['b_min'], problem_repr['b_max'], problem_repr['n_bits'])
        rez = func(decoded)
        fitness.append(rez)

    return fitness


def sort_population_by_fitness(population: list, fitness: list) -> list[list[int]]:
    return [x for _, x in sorted(zip(fitness, population), key=lambda pair: pair[0])]


def crossover(parent1: list, parent2: list) -> tuple[list, list]:
    n = len(parent1)
    c = random.randint(1, n - 1)
    return parent1[:c] + parent2[c:], parent2[:c] + parent1[c:]


def crossover_population(population: list) -> list:
    # Shuffle the population, pair up individuals and apply crossover with append
    random.shuffle(population)
    # children = []
    for i in range(0, len(population), 2):
        child1, child2 = crossover(population[i], population[i + 1])
        population.append(child1)
        population.append(child2)

    return population


def mutate(individual: list, mutation_rate: float) -> list:
    n = len(individual)
    for i in range(n):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual


def tournament_selection(population: list, fitness: list, n: int) -> list:
    # Select the best n individuals based on their fitness
    selected = []
    for _ in range(n):
        idx1 = random.randint(0, len(population) - 1)
        idx2 = random.randint(0, len(population) - 1)
        selected.append(population[idx1] if fitness[idx1] < fitness[idx2] and random.random() < 0.8 else population[idx2])

    return selected


def elitism_selection(population: list, fitness: list, n: int) -> list:
    # Select the best n individuals based on their fitness
    return sort_population_by_fitness(population, fitness)[:n]


def mixed_selection(population: list, fitness: list, n: int) -> list:
    # Select the best 10% of the population using elitisim selection and the rest using tournament selection
    n_elitism = n // 10
    n_tournament = n - n_elitism
    selected = elitism_selection(population, fitness, n_elitism)
    selected += tournament_selection(
        list(filter(lambda x: x in selected, population)),
        fitness, n_tournament
    )
    return selected


def ag(problem_repr: dict, pop_size: int, fitness_func, selection_method):
    mutation_rate = 1 / problem_repr['sol_len']
    population = generate_starting_population(pop_size, problem_repr['sol_len'])
    fitness_pop = evaluate_population(population, fitness_func, problem_repr)
    candidate = sorted(list(zip(population, fitness_pop)), key=lambda x: x[1])[0]
    best_solution = candidate
    for t in tqdm.trange(300, desc="AG generations", position=0):
        population = selection_method(population, fitness_pop, pop_size // 2)
        population = crossover_population(population)
        population = [mutate(individual, mutation_rate) for individual in population]
        fitness_pop = evaluate_population(population, fitness_func, problem_repr)
        candidate = sorted(list(zip(population, fitness_pop)), key=lambda x: x[1])[0]
        if candidate[1] < best_solution[1]:
            best_solution = candidate

    return population, best_solution


def main(f):
    func_obj = functions.FUNCTIONS[f]
    pop_size = 1000
    precision = 5
    input_dims = 30
    problem_repr = infer_value_space(func_obj, precision, input_dims)
    fitness_func = func_obj['f']
    selection_method = elitism_selection

    pop, best = ag(problem_repr, pop_size, fitness_func, selection_method)

    best = (best[1], decode_chrom(best[0], problem_repr['b_min'], problem_repr['b_max'], problem_repr['n_bits']))

    # print(*sols, sep='\n')
    print("Best: ", best)


if __name__ == '__main__':
    for f in functions.FUNCTIONS.keys():
        print(f)
        main(f)
