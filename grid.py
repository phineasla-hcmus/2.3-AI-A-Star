# %%
import numpy as np
from dataclasses import dataclass
from PIL import Image, ImageOps, ImageColor

# NW, N, NE, W, E, SW, S, SE
EIGHT_DIR = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))


img = Image.open("./img/map.bmp")
map = np.array(ImageOps.grayscale(img)).astype(int)
y_dim, x_dim = map.shape


def get_neighbors(cur, map, dirs, limit):
    y_dim, x_dim = map.shape
    neighbors = []
    for dir in dirs:
        x = cur[0] + dir[0]
        y = cur[1] + dir[1]
        if (
            (0 <= x < x_dim)
            and (0 <= y < y_dim)
            and (abs(map[cur[1], cur[0]] - map[y, x]) <= limit)
        ):
            neighbors.append((x, y))
    return neighbors


def generate_graph(map: np.ndarray, dirs, limit):
    graph: list[list[tuple[int, int]]] = []
    y_dim, x_dim = map.shape
    for y in range(y_dim):
        row = []
        for x in range(x_dim):
            p = (x, y)
            neighbors = get_neighbors(p, map, dirs, limit)
            node = np.empty(len(neighbors), dtype=object)
            node[:] = neighbors
            row.append(node)
        graph.append(row)
    return graph


graph = np.asarray(generate_graph(map, EIGHT_DIR, 5), object)
# graph = generate_graph(map, EIGHT_DIR, 10)

# %%
graph_img_dim = (x_dim * 2 - 1, y_dim * 2 - 1)
graph_img = Image.new("L", (graph_img_dim))
for y in range(y_dim):
    for x in range(x_dim):
        for neighbor in graph[y, x]:
            pos = (neighbor[0] * 2, neighbor[1] * 2)
            edge = (int((x * 2 + pos[0]) / 2), int((y * 2 + pos[1]) / 2))
            graph_img.putpixel(pos, 255)
            graph_img.putpixel(edge, 255)
graph_img.show()
