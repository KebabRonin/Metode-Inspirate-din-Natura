import H1.hibrid as hb
import h1
import numpy as np
import H1.functions as ffs

def encode_float(fl, problem_repr):
	actual_nr = int((fl - problem_repr['b_min']) * ((2**problem_repr['n_bits']) - 1) / (problem_repr['b_max'] - problem_repr['b_min']))
	ss = bin(actual_nr)[2:].zfill(problem_repr['n_bits'])
	encoded = [int(s) for s in ss]
	return encoded
# b_min + (bit_value * (b_max - b_min)) / ((2 ** n_bits) - 1)
def encode(sol, problem_repr):
	return sum([encode_float(sol[i], problem_repr) for i in range(len(sol))], start=[])

class HybridParticle(h1.Particle):
    def __init__(self, dimensions, bounds, function, w, c1, c2, precision):
        super().__init__(dimensions, bounds, function, w, c1, c2)
        func_obj = {'f': function, 'bounds': bounds} # For compatibility with H1
        self.problem_repr = hb.infer_value_space(func_obj, precision, dimensions)
        value, position = hb.hc_individual(self.position, self.function, self.problem_repr)
        self.best_value = value
        self.best_position = np.copy(position)
        self.position = np.copy(position)

    def evaluate(self):
        value = self.function(self.position)
        if value < self.best_value:
            # value is the hc from the best position
            value, position = hb.hc_individual(
                encode(self.position, self.problem_repr),
                self.function,
                self.problem_repr
            )
            position = hb.decode_chrom(position, self.problem_repr['b_min'], self.problem_repr['b_max'], self.problem_repr['n_bits'])
            self.best_value = value
            self.best_position = position
            # self.position = position
        return value

class HybridPSO(h1.PSO):
    def __init__(self, dimensions, num_particles, bounds, max_iter, function, w, c1, c2, precision):
        self.dimensions = dimensions
        self.num_particles = num_particles
        self.bounds = bounds
        self.max_iter = max_iter
        self.function = function
        self.particles = [HybridParticle(dimensions, bounds, function, w, c1, c2, precision) for _ in range(num_particles)] # Only change
        self.global_best_position = np.random.uniform(bounds[0], bounds[1], dimensions)
        self.global_best_value = float('inf')

import tqdm
dimensions = 30
num_particles = 200
max_iter = 3_000
fname = 'michalewicz'
function, lower_bound, upper_bound = h1.FUNCTIONS[fname]
w = -0.6
c1 = 2.0
c2 = -1.0
patience=100
precision=5
cfgs = [
	('rosenbrock',  2, 3_000, 50, -0.2, -1.0, 2.0), # skip as it is already 0
	('griewank',    2, 3_000, 200, 0.6, 1.5, 0.5), # These are close enough to 0
	('rastrigin',   2, 3_000, 100, -0.4, -1.5, 2.0),
	('michalewicz', 2, 3_000, 50, -0.6, -1.0, 1.5), # best pe 2

	('rosenbrock',  30, 3_000, 200, -0.6, 0.0, 1.5),
	('griewank',    30, 3_000, 200, -0.2, 2.0, 2.0),
	('rastrigin',   30, 3_000, 200, -0.4, 1.0, 2.0),
	# ('michalewicz', 30, 3_000, 200, ...), # deja facut

	('rosenbrock',  100, 3_000, 200, -0.2, 2.0, 2.0),
	('griewank',    100, 3_000, 200, -0.2, 2.0, 0.5),
	('rastrigin',   100, 3_000, 200, -0.2, 2.0, 2.0),
	('michalewicz', 100, 3_000, 200, -0.6, 2.0, -1.0),
]
for fname, dimensions, max_iter, num_particles, w, c1, c2 in cfgs:
	function, lower_bound, upper_bound = h1.FUNCTIONS[fname]
	trial_fitness = []
	trial_hists = []
	for i in tqdm.trange(30):
		pso = HybridPSO(dimensions, num_particles, (lower_bound, upper_bound), max_iter, function, w, c1, c2, precision)
		_, best_fitness = pso.optimize(patience=patience, threshold=0.3)
		trial_fitness.append(best_fitness)
		trial_hists.append(pso.last_results)
		print(f"{fname}, Dimensions: {dimensions}, Max Iters: {max_iter}, Swarm Size: {num_particles}, w: {w}, c1: {c1}, c2: {c2}, fitness {best_fitness}, hist {pso.last_results}", file=open("log.txt", "wt"))

	# Calculate statistics
	mean_fitness = np.mean(trial_fitness)
	std_fitness = np.std(trial_fitness)

	# Save the result
	config = {
		"Function": fname,
		"Dimensions": dimensions,
		"Max Iterations": max_iter,
		"Swarm Size": num_particles,
		"Momentum (w)": w,
		"Cognitive Constant (c1)": c1,
		"Social Constant (c2)": c2,
		"Mean Fitness": mean_fitness,
		"Std Fitness": std_fitness,
		"Histories": trial_hists
	}
	import json, time
	out_folder_for_run = f"prim/"
	json.dump(config, open(f"{out_folder_for_run}/timeat_{time.strftime('%m_%d_%H_%M_%S')}_config_{fname}_{dimensions}_{max_iter}_{num_particles}_{w}_{c1}_{c2}.json", "w"))


# trial_fitness.append(best_fitness)
# trial_hists.append(pso.last_results)