# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import re
import math
import numpy as np
from PIL import Image, ImageOps
from queue import PriorityQueue
from typing import Callable, Dict, DefaultDict, Optional
from collections import defaultdict

Point = tuple[int, int]
CostFunc = Callable[[np.ndarray, Point, Point], float]

# NW, N, NE, W, E, SW, S, SE
moveset = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))

with open("input.txt", "r") as f:
    start, goal, [limit] = [
        Point(int(i) for i in re.findall(r"\d+", f.readline())) for j in range(3)
    ]
start, goal, limit

# %%
img = ImageOps.grayscale(Image.open("./img/map.bmp"))
map = np.array(img).astype(int)


# %%
def real_cost(map: np.ndarray, from_pos: Point, to_pos: Point) -> float:
    x1, y1 = from_pos
    x2, y2 = to_pos
    delta = map[to_pos] - map[from_pos]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) + (
        0.5 * np.sign(delta) + 1
    ) * abs(delta)


real_cost(map, start, goal)

# %%
def in_bounds_constraint(obj, cur, neighbor, **kargs) -> bool:
    x, y = neighbor
    x_limit, y_limit = obj.map.shape
    return 0 <= x < x_limit and 0 <= y < y_limit


def limit_constraint(obj, cur, neighbor, **kargs) -> bool:
    limit = kargs["limit"]
    map = obj.map
    return abs(map[neighbor] - map[cur]) <= limit


def grid_neighbors(obj, cur: Point):
    neighbors = []
    for move in obj.moveset:
        neighbor = tuple(i + j for i, j in zip(cur, move))
        if in_bounds_constraint(obj, cur, neighbor) and all(
            f(obj, cur, neighbor, **kwargs) for f, kwargs in obj.constraint
        ):
            neighbors.append(neighbor)
    return neighbors


# %%
class AStar:
    def __init__(
        self,
        map: np.ndarray,
        heuristic: CostFunc,
        real_cost: CostFunc,
        neighbor_finder,
        moveset: tuple[Point],
        custom_constraint,
    ) -> None:
        self.h = heuristic
        self.g = real_cost
        self.map = map
        self.neighbors = neighbor_finder
        self.moveset = moveset
        self.constraint = custom_constraint

    def search(self, start: Point, goal: Point):
        open_set = PriorityQueue()
        came_from: Dict[Point, Optional[Point]] = {start: None}
        g_cost: DefaultDict[Point, float] = defaultdict(lambda: np.inf)
        f_cost: Dict[Point, float] = {}
        open_set.put([0, start])
        g_cost[start] = 0
        f_cost[start] = 0

        counter = 0

        while not open_set.empty():
            cur = open_set.get()[1]
            print(cur)
            if cur == goal:
                return "ADD RECONSTRUCT PATH FUNCTION HERE"
            for next in self.neighbors(self, cur):
                new_g_cost = g_cost[cur] + self.g(self.map, cur, next)
                if new_g_cost < g_cost[next]:
                    g_cost[next] = new_g_cost
                    f_cost[next] = new_g_cost + self.h(self.map, next, goal)
                    came_from[next] = cur  # Set "next neighbor" parent to cur
                    open_set.put([f_cost[next], next])

            # if counter == 200:
            #     return "TIMEOUT"
            # else:
            #     counter += 1

        # Open set is empty but goal was never reached
        return "LOL FAILED"


custom_constraint = [(limit_constraint, {"limit": limit})]
a_star = AStar(
    map,
    real_cost,
    real_cost,
    grid_neighbors,
    moveset,
    custom_constraint,
)

# %%

import time

start_time = time.time()
print(a_star.search(start, goal))
stop_time = time.time()
hours, rem = divmod(stop_time - start_time, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
