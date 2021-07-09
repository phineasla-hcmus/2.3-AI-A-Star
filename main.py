# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import re
import math
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageColor
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
img = Image.open("./img/map.bmp")
grayscale = ImageOps.grayscale(Image.open("./img/map.bmp"))
map = np.array(grayscale).astype(int)
map

# %%
def limit_constraint(obj, cur: Point, neighbor: Point, **kargs) -> bool:
    limit = kargs["limit"]
    map = obj.map
    return abs(map[neighbor] - map[cur]) <= limit
    # if abs(map[neighbor] - map[cur]) <= limit:
    #     return True
    # else:
    #     print(neighbor, cur)
    #     return False


def in_bounds_constraint(obj, neighbor: Point) -> bool:
    x, y = neighbor
    x_limit, y_limit = obj.map.shape
    return 0 <= x < x_limit and 0 <= y < y_limit


def grid_neighbors(obj, cur: Point):
    neighbors = []
    for move in obj.moveset:
        neighbor = tuple(i + j for i, j in zip(cur, move))
        if in_bounds_constraint(obj, neighbor) and all(
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

    def reconstruct_path(self, came_from, start, goal, result_order_from_start):
        path = []
        cur = goal
        while cur != start:
            path.append(cur)
            cur = came_from[cur]
        path.append(start)
        if result_order_from_start:
            path.reverse()
        return path

    def search(self, start: Point, goal: Point, result_order_from_start: bool = False):
        open_set = PriorityQueue()
        came_from: Dict[Point, Optional[Point]] = {start: None}
        g_cost: DefaultDict[Point, float] = defaultdict(lambda: np.inf)
        f_cost: Dict[Point, float] = {}

        open_set.put([0, start])
        g_cost[start] = 0
        f_cost[start] = 0

        while not open_set.empty():
            cur = open_set.get()[1]
            if cur == goal:
                return [
                    self.reconstruct_path(
                        came_from, start, goal, result_order_from_start
                    ),
                    f_cost[goal],
                ]
            for next in self.neighbors(self, cur):
                new_g_cost = g_cost[cur] + self.g(self.map, cur, next)
                if new_g_cost < g_cost[next]:
                    g_cost[next] = new_g_cost
                    f_cost[next] = new_g_cost + self.h(self.map, next, goal)
                    came_from[next] = cur  # Set "next neighbor" parent to cur
                    open_set.put([f_cost[next], next])
        # Open set is empty but goal was never reached
        return "LOL FAILED"


# %%
def real_cost(map: np.ndarray, from_pos: Point, to_pos: Point) -> float:
    x1, y1 = from_pos
    x2, y2 = to_pos
    delta = map[to_pos] - map[from_pos]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) + (
        0.5 * np.sign(delta) + 1
    ) * abs(delta)


def euclid(map: np.ndarray, from_pos: Point, to_pos: Point) -> float:
    x1, y1 = from_pos
    x2, y2 = to_pos
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def lame(map: np.ndarray, from_pos: Point, to_pos: Point) -> float:
    return 0


real_cost(map, start, goal)

#%%
custom_constraint = [(limit_constraint, {"limit": limit})]
a_star = AStar(
    map,
    euclid,
    real_cost,
    grid_neighbors,
    moveset,
    custom_constraint,
)

# %%
from functools import wraps
from time import time


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print("Elapsed: %2.4f sec" % (te - ts))
        return result

    return wrap


# hours, rem = divmod(stop_time - start_time, 3600)
# minutes, seconds = divmod(rem, 60)
# print("Elapsed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
print(f"Search from {start} to {goal} with limit = {limit}")
path, total_cost = timing(a_star.search)(start, goal)
print(f"Total cost: {total_cost}")

# %%
def save_to_img(img: Image.Image, path: list[Point], show=True, save_to_file=None):
    for point in path:
        img.putpixel(point, ImageColor.getrgb("red"))
    if show:
        img.show()
    if save_to_file:
        img.save(save_to_file)


save_to_img(img.copy(), path)

# %%
euclid(map, (0, 0), (1, 1))
