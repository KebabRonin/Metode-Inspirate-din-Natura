import tqdm, numpy as np, random
import matplotlib.pyplot as plt

COORDS = None
def create_distance_matrix(tsp_content):
    global COORDS
    coords = []
    lines = tsp_content.strip().split('\n')
    for line in lines:
        if line.strip() == 'NODE_COORD_SECTION':
            parsing_coords = True
            continue
        if line.strip() == 'EOF':
            break

        try:
            node_id, x, y = map(float, line.strip().split())
            coords.append((x, y))
        except ValueError:
            continue

    coords = np.array(coords)
    COORDS = coords
    n = len(coords)

    # Create distance matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                dist_matrix[i][j] = np.inf
            else:
                # Euclidean distance
                dx = coords[i][0] - coords[j][0]
                dy = coords[i][1] - coords[j][1]
                dist_matrix[i][j] = np.sqrt(dx*dx + dy*dy)

    return dist_matrix


class Chromosome:
    def __init__(self, n_cities, n_salesmen):
        self.n_cities = n_cities
        self.n_salesmen = n_salesmen
        self.repr = {
            'salesmen': [np.random.choice(n_cities) for _ in range(n_salesmen)],
            'sols': np.random.permutation(n_cities-1)
        }

    def mutate(self, mrate_sales, mrate_order):
        for s in range(len(self.repr['salesmen'])):
            if random.random() > mrate_sales:
                removed_at_idx = random.randint(1, 2)
                inserted_at_idx = random.randint(1, 2)
                self.repr['salesmen'][s]


        for s in range(len(self.repr['sols'])):
            if random.random() > mrate_order:

                self.repr['sols'][s] = 1
class Population:
  pass

def fitness(ch: Chromosome, dmat: np.array):
    first_pos = 0
    tours = []
    for end in ch.repr['salesmen']:
        prev_node = 1
        d = 0
        for node in ch.repr['sols'][first_pos:end]:
            d += dmat[prev_node-1][node-1]
            prev_node = node
        d += dmat[prev_node-1][0]
        tours.append(d)
        first_pos = end
    return np.sum(tours)


import ga
INSTANCE = 'eil51'
individ = ga.MTSPIndividual(create_distance_matrix(open(f'H3_GA/tsplib/{INSTANCE}.tsp', 'rt').read()))
individ.chromosomes = [
  [1, 17, 42, 44, 45, 15, 37, 5, 1],
  [1, 49, 33, 39, 10, 30, 34, 50, 16, 1],
  [1, 2, 3, 36, 35, 20, 29, 21, 9, 38, 11, 1],
  [1, 22, 28, 31, 8, 26, 7, 43, 24, 23, 48, 1],
  [1, 32, 1],
  [1, 46, 12, 19, 40, 41, 4, 47, 51, 1],
  [1, 6, 14, 25, 13, 18, 27, 1],
]

ga.plot_sol(individ)
plt.show()