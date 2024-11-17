import json, matplotlib.pyplot as plt

oo = json.load(open('H2/pythonProject/pso_results/11_17_13_57_07\griewank_2_2000_100_1_1_1_1731845906.3786688.json', 'r'))

for i in oo['Histories']:
	ys = list(map(lambda x: x[0], i))
	xs = list(map(lambda x: x[1], i))
	plt.plot(xs, ys)

plt.show()
# plt.plot(, oo['History'])