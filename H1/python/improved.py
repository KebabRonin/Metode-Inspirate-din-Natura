import math
import random
import tqdm
import functions


# Helper function to convert a bitstring to decimal
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


# Define the value space for the problem
def infer_value_space(func_obj, precision: int, input_dims: int):
    b_min, b_max = func_obj['bounds']
    # Number of intervals between [b_min, b_max]
    N = (b_max - b_min) * (10 ** precision)
    n_bits = math.ceil(math.log2(N))  # Number of bits needed to represent the intervals
    sol_len = input_dims * n_bits  # Total length of the solution bitstring
    return {'sol_len': sol_len, 'n_bits': n_bits, 'b_min': b_min, 'b_max': b_max}


# Get the neighborhood of a bitstring (flip each bit one at a time)
def neighbourhood(vc: list) -> list:
    neighbors = []
    for i in range(len(vc)):
        neighbor = vc.copy()
        neighbor[i] = 1 - neighbor[i]  # Flip the bit
        neighbors.append(neighbor)
    return neighbors


# Improve the current solution by checking its neighborhood
def improve(vc: list, func):
    best_neighbor = None
    best_score = func(vc)

    for i in range(len(vc)):
        vc[i] = 1 - vc[i]  # Flip the bit
        neighbor_score = func(vc)
        if neighbor_score < best_score:
            best_neighbor = i
            best_score = neighbor_score
        elif random.random() < 0.01:  # Small chance of random selection
            best_neighbor = i
            best_score = neighbor_score
        vc[i] = 1 - vc[i]  # Flip it back

    if best_neighbor is not None:
        vc[best_neighbor] = 1 - vc[best_neighbor]  # Apply the best flip
    return vc


# Hill climbing algorithm implementation
def hill_climbing(func_obj, n_dim: int, precision: int, max_iter: int):
    func = func_obj['f']
    problem_repr = infer_value_space(func_obj, precision, n_dim)

    # Generate the initial random solution as a bitstring
    best = get_random_bitstring(problem_repr['sol_len'])

    def apply_eval(x: list) -> float:
        # Decode the bitstring into real values
        decoded_values = [
            decode(x[i * problem_repr['n_bits']:(i + 1) * problem_repr['n_bits']],
                   problem_repr['b_min'], problem_repr['b_max'], problem_repr['n_bits'])
            for i in range(n_dim)
        ]
        return func(decoded_values)

    best_score = apply_eval(best)

    for t in tqdm.trange(max_iter):
        local = False
        vc = get_random_bitstring(problem_repr['sol_len'])
        vc_score = apply_eval(vc)

        # Perform local search using neighborhood improvement
        while not local:
            vn = improve(vc, apply_eval)
            vn_score = apply_eval(vn)

            if vn_score < vc_score:
                vc = vn
                vc_score = vn_score
            else:
                local = True

        if vc_score < best_score:
            best = vc
            best_score = vc_score

    # Decode the best solution bitstring back to real values
    best_decoded = [
        decode(best[i * problem_repr['n_bits']:(i + 1) * problem_repr['n_bits']],
               problem_repr['b_min'], problem_repr['b_max'], problem_repr['n_bits'])
        for i in range(n_dim)
    ]
    return best_decoded, best_score


# Parameters
n_dim = 100
max_iter = 300
precision = 5  # Precision of the interval division
print(f"Running Hill Climbing: {n_dim=}, {max_iter=}, {precision=}")

# Running the hill climbing algorithm for different functions
for k in functions.FUNCTIONS.keys():
    func_obj = functions.FUNCTIONS[k]
    best_solution, best_score = hill_climbing(
        func_obj,
        n_dim=n_dim,
        precision=precision,
        max_iter=max_iter
    )
    print(f"{k}: Best score: {best_score}, Best solution: {best_solution}")
