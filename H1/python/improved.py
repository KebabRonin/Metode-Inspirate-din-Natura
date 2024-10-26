import math
import random
import tqdm
import functions


def decimal(bitstring: list) -> int:
    l = len(bitstring)
    return sum([bitstring[l - 1 - i] * 2 ** i for i in range(l)])


def get_random_bitstring(n: int) -> list:
    return [random.choice([0, 1]) for _ in range(n)]


def decode(bitstring: list, b_min: float, b_max: float, n_bits: int) -> float:
    bit_value = decimal(bitstring)
    return b_min + (bit_value * (b_max - b_min)) / ((2 ** n_bits) - 1)


def infer_value_space(func_obj, precision: int, input_dims: int):
    b_min, b_max = func_obj['bounds']
    N = (b_max - b_min) * (10 ** precision)
    n_bits = math.ceil(math.log2(N))
    sol_len = input_dims * n_bits
    return {'sol_len': sol_len, 'n_bits': n_bits, 'b_min': b_min, 'b_max': b_max}


def neighbourhood(vc: list) -> list:
    neighbors = []
    for i in range(len(vc)):
        neighbor = vc.copy()
        neighbor[i] = 1 - neighbor[i]
        neighbors.append(neighbor)
    return neighbors


def improve(vc: list, func):
    best_neighbor = None
    best_score = func(vc)

    for i in range(len(vc)):
        vc[i] = 1 - vc[i]
        neighbor_score = func(vc)
        if neighbor_score < best_score:
            best_neighbor = i
            best_score = neighbor_score

        vc[i] = 1 - vc[i]

    if best_neighbor is not None:
        vc[best_neighbor] = 1 - vc[best_neighbor]
    return vc


def hill_climbing(func_obj, n_dim: int, precision: int, max_iter: int):
    func = func_obj['f']
    problem_repr = infer_value_space(func_obj, precision, n_dim)

    best = get_random_bitstring(problem_repr['sol_len'])

    def apply_eval(x: list) -> float:
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

    best_decoded = [
        decode(best[i * problem_repr['n_bits']:(i + 1) * problem_repr['n_bits']],
               problem_repr['b_min'], problem_repr['b_max'], problem_repr['n_bits'])
        for i in range(n_dim)
    ]
    return best_decoded, best_score


n_dim = 5
max_iter = 100
precision = 5
print(f"Running Hill Climbing: {n_dim=}, {max_iter=}, {precision=}")

for k in functions.FUNCTIONS.keys():
    func_obj = functions.FUNCTIONS[k]
    best_solution, best_score = hill_climbing(
        func_obj,
        n_dim=n_dim,
        precision=precision,
        max_iter=max_iter
    )
    print(f"{k}: Best score: {best_score}, Best solution: {best_solution}")
