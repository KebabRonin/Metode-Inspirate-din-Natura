import numpy as np


def create_distance_matrix(tsp_content):
    # Parse coordinates
    coords = []
    lines = tsp_content.strip().split('\n')
    for line in lines:
        if line.strip() == 'NODE_COORD_SECTION':
            parsing_coords = True
            continue
        if line.strip() == 'EOF':
            break

        try:
            # Try to parse as coordinate line
            node_id, x, y = map(float, line.strip().split())
            coords.append((x, y))
        except ValueError:
            continue

    # Convert to numpy array
    coords = np.array(coords)
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
                dist_matrix[i][j] = np.sqrt(dx * dx + dy * dy)

    return dist_matrix



with open('eil51.txt', 'r') as f:
    content = f.read()
distances = create_distance_matrix(content)
print(distances)
