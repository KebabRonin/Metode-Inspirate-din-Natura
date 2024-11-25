import json, matplotlib.pyplot as plt, math
from matplotlib.collections import PolyCollection
import parse
import os, sys, itertools, tqdm, pickle, numpy as np

## Change these
FUNCTION = 'michalewicz'
DIMS = [100]
max_iter_options = [3000] # Lasa asa ca oricum converge in mai putine
num_particles_options = [200]  # 3 variants
w_options =  [i/10 for i in range(-10, 10+1, 2)] # 10 variants
c1_options = [i/10 for i in range(-20, 20+1, 5)] # 8 variants
c2_options = [i/10 for i in range(-20, 20+1, 5)] # 8 variants


def amass_results():
	filess = list(os.walk(FUNCTION))[0][2]
	pstart = len(f"timeat_xx_xx_xx_xx_xx_config_{FUNCTION}_")
	globalobjs = []

	for i in tqdm.tqdm(itertools.product(DIMS, num_particles_options, w_options, c1_options, c2_options), total=len(DIMS) * len(num_particles_options) * len(w_options) * len(c1_options) * len(c2_options)):
		thisnaem = list(filter(lambda x: (x[pstart:] == f"{i[0]}_3000_{i[1]}_{i[2]}_{i[3]}_{i[4]}.json"), filess))
		print(thisnaem)
		if len(thisnaem) != 1:
			raise Exception(f"{len(thisnaem)} files found for {i}")
			continue
		with open(f"{FUNCTION}/{thisnaem[0]}") as f:
			obj = json.load(f)
			globalobjs.append(obj)

	with open(f'results_{FUNCTION}.pkl', 'wb') as f:
		pickle.dump(globalobjs, f)


def get_stats():
	oo = pickle.load(open(f'results_{FUNCTION}.pkl', 'rb'))
	oop = list(filter(lambda x: x['Dimensions'] == 100, oo))
	shape = (11, 9, 9) #(len(w_options), len(c1_options), len(c2_options))
	results_mean = np.zeros(shape) # w, c1, c2
	results_min = np.zeros(shape) # w, c1, c2
	results_max = np.zeros(shape) # w, c1, c2
	results_std = np.zeros(shape) # w, c1, c2
	for i in range(shape[0]):
		for j in range(shape[1]):
			for k in range(shape[2]):
				c_cfg = list(filter(lambda x: x['Momentum (w)'] == w_options[i] and x['Cognitive Constant (c1)'] == c1_options[j] and x['Social Constant (c2)'] == c2_options[k], oop))[0]
				results_mean[i, j, k] = c_cfg['Mean Fitness']
				results_std[i, j, k] = c_cfg['Std Fitness']
				results_min[i, j, k] = min(list(map(lambda x: x[-1][0], c_cfg['Histories'])))
				results_max[i, j, k] = max(list(map(lambda x: x[-1][0], c_cfg['Histories'])))
	pickle.dump({'min': results_min, 'max': results_max, 'mean': results_mean, 'std': results_std}, open(f"{FUNCTION}_nps.pkl", 'wb'))


# Mean, std, min, max from results.pkl
get_stats()

stats = pickle.load(open(f"{FUNCTION}_nps.pkl", 'rb'))
fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(projection='3d')

xs, ys, zs, cs = list(zip(*list(map(lambda x: (*x[0], x[1]), tuple(np.ndenumerate(stats['mean']))))))

img = ax.scatter(
	xs=list(map(lambda x: w_options[x], xs)),
	ys=list(map(lambda y: c1_options[y], ys)),
	zs=list(map(lambda z: c2_options[z], zs)),
	c=cs,
	cmap='winter',
)
fig.colorbar(img)

ax.set_xlabel('Momentum (w)')
ax.set_ylabel('Cognitive Constant (c1)')
ax.set_zlabel('Social Constant (c2)')
ax.set_title(f"Mean Fitness for {FUNCTION}")
# ax.set(xlim=(-1, 1), ylim=(-2, 2), zlim=(-2, 2), xlabel='w', ylabel=r'c1', zlabel='c2')

plt.show()

exit(0)

# pickle.dump(oo, open('results_{FUNCTION}.pkl', 'wb'))
# for config in oop:
# 	results = []
# 	for h in config['Histories']:
# 		result = h[-1][0]
# 		results.append(result)
# 		ys = list(map(lambda x: x[0], h))
# 		xs = list(map(lambda x: x[1], h))
# 		plt.plot(xs, ys)
# 	print(f"Mean: {np.mean(results)} | Std: {np.std(results)} | Min: {np.min(results)} | Max: {np.max(results)}")
# plt.yscale('log')
# plt.show()
# for i in oo['Histories']:
# 	ys = list(map(lambda x: x[0], i))
# 	xs = list(map(lambda x: x[1], i))
# 	plt.plot(xs, ys)

# plt.show()
# plt.plot(, oo['History'])