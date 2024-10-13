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

import numpy as np
import math, random, tqdm
import functions


# x - vector de 5 elem
# interval - [-10, 10]
# precizia - 0.1
# 20 * 10 = 200 -> 8 biti / elemt
# x -> 8 * 5 = 40 biti
# 0010

def decimal(bitstring: np.ndarray):
	l = len(bitstring)
	return sum([int(bitstring[l - 1 - i]) * 2**i for i in range(l)])

# def decode(bitstring, b_min, b_max, n):
# 	return b_min + (decimal(bitstring) * (b_max - b_min) / (2 ** n - 1))

def get_random_bitstring(n) -> bool:
	return np.array([random.choice([True, False]) for i in range(n)])

# [-10, 10] -> 200 intervale
# [-10.0, -9.9, .. 9.9, 10.0] - 200 vals
# 200 < 256 = 2**8 -> 8 biti
# x - 5*8 biti solutie

def infer_value_space(func_obj, precision, input_dims):
	b_min = func_obj['bounds'][0]
	b_max = func_obj['bounds'][1]
	n = (b_max - b_min) * (10 ** precision)
	bitstring_len = math.ceil(math.log2(n))
	sol_len = input_dims * bitstring_len
	def decode(bitstring):
		return b_min + (decimal(bitstring) * (b_max - b_min) / (2 ** bitstring_len - 1))
	return {'sol_len':sol_len, 'bitstring_len': bitstring_len, 'decode': decode}

def neighbourhood(vc: np.ndarray):
	ns = []
	for i in range(len(vc)):
		new_vc = vc.copy()
		new_vc[i] = not(new_vc[i])
		ns.append(new_vc)
	return ns

def improve(vc, func, neighbours):
	best_neighbor = vc
	best_score = func(vc)

	for neighbour in neighbours:
		neighbour_score = func(neighbour)
		if neighbour_score < best_score:
			best_neighbor = neighbour
			best_score = neighbour_score
		elif random.random() < 0.01:
			best_neighbor = neighbour
			best_score = neighbour_score

	return best_neighbor


def hill_climbing(func_obj, n_dim, precision, max_iter):
	func = func_obj['f']
	problem_repr = infer_value_space(func_obj, precision, n_dim)

	best = get_random_bitstring(problem_repr['sol_len'])
	best_score = func(best)

	def apply_eval(x):
		return func(np.array([problem_repr['decode'](x[i*problem_repr['bitstring_len']:(i+1)*problem_repr['bitstring_len']]) for i in range(n_dim)]))

	for t in tqdm.trange(max_iter):
		local = False
		vc = get_random_bitstring(problem_repr['sol_len'])
		vc_score = apply_eval(vc)

		while not local:
			vn = improve(vc, apply_eval, neighbourhood(vc))
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

	best = np.array([problem_repr['decode'](best[i*problem_repr['bitstring_len']:(i+1)*problem_repr['bitstring_len']]) for i in range(n_dim)])
	return best, best_score


n_dim = 5
max_iter = 150
print(f"{n_dim=} {max_iter=}")
for k in functions.FUNCTIONS.keys():
	func_obj = functions.FUNCTIONS[k]
	best_solution, best_score = hill_climbing(
		func_obj,
		n_dim,
		precision=5,
		max_iter=max_iter
	)
	print(k, best_score, best_solution)