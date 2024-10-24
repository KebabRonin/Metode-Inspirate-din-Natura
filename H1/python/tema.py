"""
t := 0
initialize best
repeat
	local := FALSE
	select a candidate solution (bitstring) vc at random
	evaluate vc
	repeat
		vn := Improve(Neghborhood(vc))
 	if eval(vn) is better than eval(vc)
	then vc := vn
	else local := TRUE
	until local
	t := t + 1
	if vc is better than best
	then best := vc
until t = MAX
"""

import math
import random
import tqdm

import numpy as np

import functions


# x - vector de 5 elem
# interval - [-10, 10]
# precizia - 0.1
# 20 * 10 = 200 -> 8 biti / elemt
# x -> 8 * 5 = 40 biti
# 0010

def decimal(bitstring: np.ndarray):
    l = len(bitstring)
    return sum([int(bitstring[l - 1 - i]) * 2 ** i for i in range(l)])


# def decode(bitstring, b_min, b_max, n):
# 	return b_min + (decimal(bitstring) * (b_max - b_min) / (2 ** n - 1))

def get_random_bitstring(n) -> bool:
    return np.array([random.choice([True, False]) for i in range(n)])


# [-10, 10] -> 200 intervale
# [-10.0, -9.9, .. 9.9, 10.0] - 200 vals
# 200 < 256 = 2**8 -> 8 biti
# x - 5*8 biti solutie

# La grafice scara logaritmica pentru solutii, linear pt timp
# HC ar trebui in < 36 optim la Rastigin (nu cred ca am auzit asta bine: 30 - 40 iters sa ajunga la un optim)
# t = 40-50s pt 30 dimensiuni 100 iters (am belito)
# fitness (max(E) - e [nu 1 - e]) / (max(E) - min(E) + eps[\varepsilon in latex])
# mutatie = 1 / nr_gene_in_solutie (gen 5*8 de mai sus)
# operatori genetici care ajuta la explorarea spatiului (putem sa ii oferim intuitie) - operatii geometrice(rotatie, translatie, etc) pt probleme de geometrie
# ## Hibridizari
# 1. Run GA -> sol prin Sim Annealing ( worst option )
# 2. Hillclimber -> GA (idk)
# 3. Run last generation through SA
# 4. Algo mememic (toata pop prin SA, every time all the time) - v slow (-100x) - mutatie mai mare corecteaza rezultatul (?)
#   * Variant: Eval chromosome based on SA, but don't update the chromosome (use SA just for Fitness calculation)
#     * -> indivizii nu sunt capturati de optime locale, ci sunt evaluati prin 'potentialul' lor
# [mutation -> xover -> HC -> selection] -> HC
# HC with timeout for use in hibridisation
# xover 30% 40% creste timpul de rulare, rezultatele nu prea
# ### Presiunea de selectie
# * higher - exploitation (mutation)
# * lower - exploration (xover)
# fitness in [0,1), augument by (fitness + 1) ^ 2 [same idea as MSE, fitness gets better exponentially -> higher selection pressure]
# eps greedy gen pt scheduling pressure
# Soft greedy: 5% din best sunt dusi automat in generatia noua 

def infer_value_space(func_obj, precision, input_dims):
    b_min = func_obj['bounds'][0]
    b_max = func_obj['bounds'][1]
    n = (b_max - b_min) * (10 ** precision)
    bitstring_len = math.ceil(math.log2(n))
    sol_len = input_dims * bitstring_len
    pow2 = (2 ** bitstring_len) - 1
    interval = (b_max - b_min)

    # Cache 2 ** n
    def decode(bitstring):
        return b_min + (decimal(bitstring) * interval / (pow2))

    return {'sol_len': sol_len, 'bitstring_len': bitstring_len, 'decode': decode}


def neighbourhood(vc: np.ndarray):
    ns = []
    for i in range(len(vc)):
        new_vc = vc.copy()
        new_vc[i] = not (new_vc[i])
        ns.append(new_vc)
    return ns


def improve(vc, func):
    best_neighbor = None
    best_score = func(vc)

    for i in range(len(vc)):
        vc[i] = not (vc[i])
        neighbour_score = func(vc)
        if neighbour_score < best_score:
            best_neighbor = i
            best_score = neighbour_score
        elif random.random() < 0.01:
            best_neighbor = i
            best_score = neighbour_score
        vc[i] = not (vc[i])

    if best_neighbor:
        vc[best_neighbor] = not (vc[best_neighbor])
    return vc


def hill_climbing(func_obj, n_dim, precision, max_iter):
    func = func_obj['f']
    problem_repr = infer_value_space(func_obj, precision, n_dim)

    best = get_random_bitstring(problem_repr['sol_len'])
    best_score = func(best)

    def apply_eval(x):
        return func(np.array(
            [problem_repr['decode'](x[i * problem_repr['bitstring_len']:(i + 1) * problem_repr['bitstring_len']]) for i
             in range(n_dim)]))

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

        # print(f"Iter {t}: curr stop: {x} curr score {vc_score}")

        if vc_score < best_score:
            best = vc
            best_score = vc_score

    best = np.array(
        [problem_repr['decode'](best[i * problem_repr['bitstring_len']:(i + 1) * problem_repr['bitstring_len']]) for i
         in range(n_dim)])
    return best, best_score


n_dim = 30
max_iter = 100
print(f"{n_dim=} {max_iter=}")
for k in functions.FUNCTIONS.keys():
    func_obj = functions.FUNCTIONS[k]
    best_solution, best_score = hill_climbing(func_obj, n_dim, precision=5, max_iter=max_iter)
    print(k, best_score, best_solution)
