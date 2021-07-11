# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import re
import math
import numpy as np
from PIL import Image, ImageOps, ImageColor
from queue import PriorityQueue
from typing import Callable, Dict, DefaultDict, Optional
from collections import defaultdict

Point = tuple[int, int]
CostFunc = Callable[[np.ndarray, Point, Point], float]

# NW, N, NE, W, E, SW, S, SE
EIGHT_DIR = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))
# N, W, E, S
FOUR_DIR = ((0, -1), (-1, 0), (1, 0), (0, 1))
# %%
def limit_constraint(obj, cur: Point, neighbor: Point, **kargs) -> bool:
    limit = kargs["limit"]
    map = obj.map
    return abs(map[cur] - map[neighbor]) <= limit


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
        neighbor_finder,
        moveset: tuple[Point],
        heuristic: CostFunc,
        real_cost: CostFunc,
        custom_constraint=None,
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
        fringe = PriorityQueue()
        fringe.put([0, start])
        came_from: Dict[Point, Optional[Point]] = {start: None}
        g_cost: DefaultDict[Point, float] = defaultdict(lambda: np.inf)
        g_cost[start] = 0
        f_cost: Dict[Point, float] = {start: 0}

        while not fringe.empty():
            cur = fringe.get()[1]
            # cur_to_goal = np.array([i - j for (i, j) in zip(cur, goal)])
            if cur == goal:
                return [
                    self.reconstruct_path(
                        came_from, start, goal, result_order_from_start
                    ),
                    g_cost,
                    f_cost,
                ]
            for next in self.neighbors(self, cur):
                new_g_cost = g_cost[cur] + self.g(self.map, cur, next)
                if new_g_cost < g_cost[next]:
                    # new_to_goal = np.array([i - j for (i, j) in zip(next, goal)])
                    # cross = abs(np.cross(cur_to_goal, new_to_goal))
                    # f_cost[next] += cross * 0.001
                    g_cost[next] = new_g_cost
                    f_cost[next] = new_g_cost + self.h(self.map, next, goal)
                    came_from[next] = cur  # Set "next neighbor" parent to cur
                    fringe.put([f_cost[next], next])
        # Open set is empty but goal was never reached
        raise Exception("No solution found")


# %%
with open("input.txt", "r") as f:
    start, goal, [limit] = [
        Point(int(i) for i in re.findall(r"\d+", f.readline())) for j in range(3)
    ]
custom_constraint = [(limit_constraint, {"limit": limit})]

img = Image.open("./img/map.bmp")
grayscale = ImageOps.grayscale(Image.open("./img/map.bmp"))
map = np.array(grayscale).astype(int)
# %%
def real_cost(map: np.ndarray, _from: Point, _to: Point) -> float:
    x1, y1 = _from
    x2, y2 = _to
    delta = map[_from] - map[_to]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) + (
        0.5 * np.sign(delta) + 1
    ) * abs(delta)


# %%
def euclid(map: np.ndarray, _from: Point, _to: Point) -> float:
    x1, y1 = _from
    x2, y2 = _to
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def euclid_squared(map: np.ndarray, _from: Point, _to: Point) -> float:
    return (_from[0] - _to[0]) ** 2 + (_from[1] - _to[1]) ** 2


def lame(map: np.ndarray, _from: Point, _to: Point) -> float:
    return 0


def pythagoras(map: np.ndarray, _from: Point, _to: Point) -> float:
    x1, y1 = _from
    x2, y2 = _to
    a = (x1 - x2) ** 2 + (y1 - y2) ** 2
    b = abs(map[_from] - map[_to])
    return math.sqrt(a + b ** 2)


# args = (map, start, goal)
# for func in [real_cost, euclid, pythagoras]:
#     print(func(*args))

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


def display_to_img(img: Image.Image, path: list[Point], show=True, save_to_file=None):
    for point in path:
        img.putpixel(point, ImageColor.getrgb("red"))
    if show:
        img.show()
    if save_to_file:
        img.save(save_to_file)


def test(heuristic: CostFunc, timer=True, show=True, save=None):
    a_star = AStar(
        map,
        grid_neighbors,
        EIGHT_DIR,
        heuristic,
        real_cost,
        custom_constraint,
    )
    try:
        path, g_cost, f_cost = (
            timing(a_star.search)(start, goal) if timer else a_star.search(start, goal)
        )
        print(
            f"""[{heuristic.__name__}] From {start} to {goal} with limit = {limit}
    Total cost: {f_cost[goal]}
    Examined nodes: {len(f_cost)}
    Path nodes: {len(path)}
    """
        )
        if show or save:
            for point in f_cost.keys():
                img.putpixel(point, ImageColor.getrgb("yellow"))
            for point in path:
                img.putpixel(point, ImageColor.getrgb("red"))
        if show:
            img.show()
        if save:
            img.save(save)
    except Exception as e:
        print(e)


# %%
test(real_cost)

# %%
# import numpy as np

# v1 = np.array([j - i for (i, j) in zip([0, 0], [1, 1])])
# v2 = np.array([j - i for (i, j) in zip([1, 1], [3, 2])])
# print(v1 / v2)

# %%
