import math
import random

from sympy import floor

import functions
import tqdm


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


def crossover(parent1: list, parent2: list) -> tuple[list, list]:
    n = len(parent1)
    c = random.randint(0, n)
    return parent1[:c] + parent2[c:], parent2[:c] + parent1[c:]


def mutate(individual: list, mutation_rate: float) -> list:
    n = len(individual)
    for i in range(n):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual


# for i:=0 to POP_SIZE
#     uniform random r in (0,1]
#     select for the next generation the individual j
#     for which q[j] < r <= q[j+1]


def wheel_of_fortune(population: list, fitness_pop: list) -> list:
    """
    Evaluate P		for i:=0 to POP_SIZE
                    eval[i] = f( P(i) )
    Total fitness		for i:=0 to POP_SIZE
                    T += eval[i]
    Individual sel.prob.	for i:=0 to POP_SIZE
                    p[i] = eval[i] / T
    Accumulated sel.prob.	q[0] = 0
                for i:=0 to POP_SIZE
                    q[i+1] = q[i] + p[i]
    Selection		for i:=0 to POP_SIZE
                    uniform random r in (0,1]
                    select for the next generation the individual j
                       for which q[j] < r <= q[j+1]
    """
    n = len(population)
    T = sum(fitness_pop) + 1e-10
    p = [fitness_pop[i] / T for i in range(n)]
    q = [0] + [sum(p[:i + 1]) for i in range(n)]
    new_population = []
    for _ in range(n):
        r = random.random()
        for j in range(n):
            if q[j] < r <= q[j + 1]:
                new_population.append(population[j])
                break

    return new_population


# create a function that applies the crossover to the entire population that is sorted by fitness
def crossover_population(population: list, crossover_rate: float) -> list:
    n = len(population)
    population = sorted(population, key=lambda x: random.random())
    for i in range(0, int(floor((crossover_rate * n))), 2):
        population[i], population[i+1] = crossover(population[i], population[i + 1])
    return population


def ag(problem_repr: dict, pop_size: int, fitness_func, selection_method, mutation_rate,
       crossover_rate):
    t = 0
    population = generate_starting_population(pop_size, problem_repr['sol_len'])
    fitness_pop = evaluate_population(population, fitness_func, problem_repr)
    candidate = sorted(list(zip(population, fitness_pop)), key=lambda x: x[1])[0]
    best_solution = candidate
    for t in tqdm.trange(100):
        population = selection_method(population, fitness_pop)
        population = [mutate(individual, mutation_rate) for individual in population]
        population = crossover_population(population, crossover_rate)
        fitness_pop = evaluate_population(population, fitness_func, problem_repr)
        candidate = sols = sorted(list(zip(population, fitness_pop)), key=lambda x: x[1])[0]
        if candidate[1] < best_solution[1]:
            best_solution = candidate

    return population, best_solution


def main():
    func_obj = functions.FUNCTIONS['DeJong']
    pop_size = 100
    precision = 5
    input_dims = 5
    problem_repr = infer_value_space(func_obj, precision, input_dims)
    fitness_func = func_obj['f']
    selection_method = wheel_of_fortune
    mutation_rate = 0.01
    crossover_rate = 0.8

    pop, best = ag(problem_repr, pop_size, fitness_func, selection_method, mutation_rate,
                   crossover_rate)
    rezs = evaluate_population(pop, fitness_func, problem_repr)
    sols = sorted(list(zip(pop, rezs)), key=lambda x: x[1], reverse=True)
    sols = [(decode_chrom(sol[0], problem_repr['b_min'], problem_repr['b_max'], problem_repr['n_bits']), sol[1]) for sol in sols]
    best = (decode_chrom(best[0], problem_repr['b_min'], problem_repr['b_max'], problem_repr['n_bits']), best[1])

    print(*sols, sep='\n')
    print("Best: ", best)


if __name__ == '__main__':
    main()
