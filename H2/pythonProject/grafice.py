import json, matplotlib.pyplot as plt
import os, sys, itertools, tqdm, pickle, numpy as np

## Change these
FUNCTION = 'griewank'
DIMS = [2, 30, 100]
max_iter_options = [3000] # Lasa asa ca oricum converge in mai putine
num_particles_options = [50, 100, 200]  # 3 variants
w_options =  [i/10 for i in range(-10, 10+1, 2)] # 10 variants
c1_options = [i/10 for i in range(-20, 20+1, 5)] # 8 variants
c2_options = [i/10 for i in range(-20, 20+1, 5)] # 8 variants

# import plotly.graph_objects as go
# import numpy as np

# pts = np.loadtxt(np.DataSource().open('https://raw.githubusercontent.com/plotly/datasets/master/mesh_dataset.txt'))
# x, y, z = pts.T

# fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z,
#                    alphahull=5,
#                    opacity=0.4,
#                    color='cyan')])
# fig.show()
# exit(0)


def amass_results(function):
	filess = list(os.walk(function))[0][2]
	pstart = len(f"timeat_xx_xx_xx_xx_xx_config_{function}_")
	globalobjs = []

	for i in tqdm.tqdm(itertools.product(DIMS, num_particles_options, w_options, c1_options, c2_options), total=len(DIMS) * len(num_particles_options) * len(w_options) * len(c1_options) * len(c2_options)):
		thisnaem = list(filter(lambda x: (x[pstart:] == f"{i[0]}_3000_{i[1]}_{i[2]}_{i[3]}_{i[4]}.json"), filess))
		# print(thisnaem)
		if len(thisnaem) > 1:
			print(f"Warning: {len(thisnaem)} files found for {i}")
			for f in thisnaem[:-1]:
				os.rename(f"{function}/{f}", f"duplicates/{f}")
		if len(thisnaem) == 0:
			print(f"Warning: No files found for {i}")
			continue
		try:
			with open(f"{function}/{thisnaem[-1]}", "r") as f: #,encoding='cp850') as f:
				obj = json.load(f)
				globalobjs.append(obj)
		except Exception as e:
			print(f"Failed to load {thisnaem[-1]}", e)

	with open(f'results_{function}.pkl', 'wb') as f:
		pickle.dump(globalobjs, f)


def get_stats(function):
	oo = pickle.load(open(f'results_{function}.pkl', 'rb'))
	oop = oo # list(filter(lambda x: x['Dimensions'] == DIMS[0], oo))
	shape = (3, 3, 11, 9, 9) #(len(dims), len(popsize), len(w_options), len(c1_options), len(c2_options))
	results_mean = np.zeros(shape) # d, pop, w, c1, c2
	results_min = np.zeros(shape) # d, pop, w, c1, c2
	results_max = np.zeros(shape) # d, pop, w, c1, c2
	results_std = np.zeros(shape) # d, pop, w, c1, c2
	for dim in range(shape[0]):
		for pop in range(shape[1]):
			for w in range(shape[2]):
				for c1 in range(shape[3]):
					for c2 in range(shape[4]):
						try:
							c_cfg = list(
							filter(lambda x:
							x['Dimensions'] == DIMS[dim] and
							x['Swarm Size'] == num_particles_options[pop] and
							x['Momentum (w)'] == w_options[w] and
							x['Cognitive Constant (c1)'] == c1_options[c1] and
							x['Social Constant (c2)'] == c2_options[c2]
							, oop))
							c_cfg = c_cfg[0]
							results_mean[dim, pop, w, c1, c2] = c_cfg['Mean Fitness']
							results_std[dim, pop, w, c1, c2] = c_cfg['Std Fitness']
							results_min[dim, pop, w, c1, c2] = min(list(map(lambda x: x[-1][0], c_cfg['Histories'])))
							results_max[dim, pop, w, c1, c2] = max(list(map(lambda x: x[-1][0], c_cfg['Histories'])))
						except Exception as e:
							print(f"Dims {DIMS[dim]}, Pop {num_particles_options[pop]}, w {w_options[w]}, c1 {c1_options[c1]}, c2 {c2_options[c2]}", e)
	pickle.dump({'min': results_min, 'max': results_max, 'mean': results_mean, 'std': results_std}, open(f"{function}_nps.pkl", 'wb'))


# Gather the results
# for f in ['rosenbrock', 'griewank', 'rastrigin', 'michalewicz']:
# 	amass_results(f)
# 	get_stats(f)

import numpy as np
import plotly.graph_objects as go
import pandas as pd
def show_bests(function, sortby):
	stats = pickle.load(open(f"{function}_nps.pkl", 'rb'))
	print(function)
	for i in range(3):
		bb = tuple(np.ndenumerate(stats[sortby][i,...]))[np.argmin(stats[sortby][i,...])]
		rez = bb[1]
		bb = bb[0]
		print(f"best_results: {DIMS[i]} dims, {num_particles_options[bb[0]]} pop, {w_options[bb[1]]} w, {c1_options[bb[2]]} c1, {c2_options[bb[3]]} c2, {stats['mean'][i, *bb]} mean score, {stats['std'][i, *bb]} std, {stats['min'][i, *bb]} min score")

def plot_3d(function):
	stats = pickle.load(open(f"{function}_nps.pkl", 'rb'))
	for pop in range(len(num_particles_options)):
		xs, ys, zs, cs = list(zip(*list(map(lambda x: (*x[0], x[1]), tuple(np.ndenumerate(stats['mean'][1,pop,...]))))))
		cs= np.clip(cs, 0, 600)
		xs = np.array(list(map(lambda x: w_options[x], xs)))
		ys = np.array(list(map(lambda y: c1_options[y], ys)))
		zs = np.array(list(map(lambda z: c2_options[z], zs)))

		fig = plt.figure(figsize=(12,7))
		ax = fig.add_subplot(projection='3d')

		img = ax.scatter(
			xs=xs,
			ys=ys,
			zs=zs,
			c=cs,
			cmap='turbo',
			# norm=matplotlib.colors.LogNorm(),
		)
		fig.colorbar(img)

		ax.set_xlabel('Momentum (w)')
		ax.set_ylabel('Cognitive Constant (c1)')
		ax.set_zlabel('Social Constant (c2)')
		ax.set_title(f"Mean Fitness for {function} {num_particles_options[pop]} particles")

		plt.show()


def plot_3d_volume(function):
	stats = pickle.load(open(f"{function}_nps.pkl", 'rb'))
	for pop in range(len(num_particles_options)):
		xs, ys, zs, cs = list(zip(*list(map(lambda x: (*x[0], x[1]), tuple(np.ndenumerate(stats['mean'][1,pop,...]))))))
		xs = np.array(list(map(lambda x: w_options[x], xs)))
		ys = np.array(list(map(lambda y: c1_options[y], ys)))
		zs = np.array(list(map(lambda z: c2_options[z], zs)))
		cs= np.clip(cs, 0, 600)
		# xs, ys, zs = np.meshgrid(w_options, c1_options, c2_options, indexing='ij')
		# cs = np.array(cs)

		fig = go.Figure(data=go.Volume(
			x=xs, y=ys, z=zs,
			value=cs,
			opacity=0.1,
			opacityscale=[[0, 1], [0.3, 0.2], [1, 0]],
			surface_count=200,
			))
		fig.update_layout(
			title=f"Mean Fitness for {function} {num_particles_options[pop]} particles",
			scene = dict(
				xaxis = dict(
					title='Momentum (w)'
				),
				yaxis = dict(
					title='Cognitive Constant (c1)'
				),
				zaxis = dict(
					title='Social Constant (c2)'
				)
			),
			scene_xaxis_showticklabels=True,
			scene_yaxis_showticklabels=True,
			scene_zaxis_showticklabels=True)
		fig.show()

def plot_hists(fname, dims, pop, w, c1, c2, **kwargs):
	try:
		ALL
	except:
		ALL = dict()
	try:
		ALL[fname]
	except:
		ALL = pickle.load(open(f'results_{fname}.pkl', 'rb'))
	objs = list(filter(lambda x: x['Function'] == fname and x['Dimensions'] == dims and x['Swarm Size'] == pop and x['Momentum (w)'] == w and x['Cognitive Constant (c1)'] == c1 and x['Social Constant (c2)'] == c2, ALL))
	if len(objs) != 1:
		print(f"Warning: {len(objs)} objects found for {fname}, {dims}, {pop}, {w}, {c1}, {c2}")
		return

	obj = objs[0]
	for h in obj['Histories']:
		ys = list(map(lambda x: x[0], h))
		xs = list(map(lambda x: x[1], h))
		plt.plot(xs, ys, **kwargs)

def show_first_bests(function, n, sortby, plot=False):
	stats = pickle.load(open(f"{function}_nps.pkl", 'rb'))
	print("\multirow{"+str(n)+"}{*}{"+function+"}")
	clrss = itertools.cycle(['g', 'y', 'b', 'r', 'c', 'm', 'k'])
	for j in range(3):
		bests = list(np.ndenumerate(stats[sortby][j,...]))
		bests.sort(key=lambda x: x[1])
		atual = 0
		prev = -1e10
		print(" & \multirow{"+str(n)+"}{*}{"+str(DIMS[j])+"}")
		for i in range(n):
			prev = bests[atual][1]
			bb = bests[atual]
			bb = bb[0]
			if plot:
				plot_hists(function, DIMS[j], num_particles_options[bb[0]], w_options[bb[1]], c1_options[bb[2]], c2_options[bb[3]], color = next(clrss))
			skipped = 0
			while f"{bests[atual][1]:.5f}" == f"{prev:.5f}":
				atual += 1
				skipped += 1
			atual += 1
			# if skipped > 0:
			# 	print(f"{skipped} skipped")
			print("\t&"+("&" if i > 0 else "")+f" Swarm Size: {num_particles_options[bb[0]]:3}, $w$: {w_options[bb[1]]:4}, $c1$: {c1_options[bb[2]]:4}, $c2$: {c2_options[bb[3]]:4}" + ("" if skipped == 1 else f"({skipped} configurations)") + f" & {stats['mean'][j, *bb]:.6f} & {stats['std'][j, *bb]:.6f} & {stats['min'][j, *bb]:.6f} \\\\", sep="")
		print("\cmidrule(l){2-6}\n")
		if plot:
			plt.title(f"{function} by {sortby}")
			plt.yscale('log')
			plt.show()
	print("\\midrule\n")
ALL = dict()

# for f in ['rosenbrock', 'griewank', 'rastrigin', 'michalewicz']:
# 	# plot_3d_volume(f)
# 	# show_bests(f)
# 	# stats = pickle.load(open(f"results_{f}.pkl", 'rb'))
# 	# lens = list(map(lambda x: np.mean(list(map(lambda h: h[-1][1], x['Histories']))), stats))
# 	# lens.sort()
# 	# plt.hist(lens, bins=200)
# 	# plt.title(f"{f} ")
# 	# plt.yscale('log')
# 	# plt.xlabel('Iteration of the last improvement')
# 	# plt.ylabel('Number of runs')
# 	# plt.savefig(f"{f}_hist.png", dpi=300, bbox_inches='tight')
# 	# plt.close()
# 	show_first_bests(f, 3, 'mean')
stats = pickle.load(open(f"results_{f}.pkl", 'rb'))
c_cfg = c_cfg[0]
results_mean[dim, pop, w, c1, c2] = c_cfg['Mean Fitness']
results_std[dim, pop, w, c1, c2] = c_cfg['Std Fitness']
results_min[dim, pop, w, c1, c2] = min(list(map(lambda x: x[-1][0], c_cfg['Histories'])))
results_max[dim, pop, w, c1, c2] = max(list(map(lambda x: x[-1][0], c_cfg['Histories'])))