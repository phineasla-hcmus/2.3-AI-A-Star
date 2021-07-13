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

with open("input.txt", "r") as f:
    start, goal, [limit] = [
        Point(int(i) for i in re.findall(r"\d+", f.readline())) for j in range(3)
    ]

img = Image.open("./img/map.bmp")
# img = Image.open("./img/test.png")
grayscale = ImageOps.grayscale(img)
map = np.array(grayscale).astype(int)

# %%
def limit_constraint(obj, cur: Point, neighbor: Point, **kargs) -> bool:
    x1, y1 = cur
    x2, y2 = neighbor
    limit = kargs["limit"]
    map = obj.map
    return abs(map[y1, x1] - map[y2, x2]) <= limit


def in_bounds_constraint(obj, neighbor: Point) -> bool:
    x, y = neighbor
    y_limit, x_limit = obj.map.shape
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
def real_cost(map: np.ndarray, _from: Point, _to: Point) -> float:
    x1, y1 = _from
    x2, y2 = _to
    delta = map[y1, x1] - map[y2, x2]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) + (
        0.5 * np.sign(delta) + 1
    ) * abs(delta)


# %%
def manhattan(map: np.ndarray, _from: Point, _to: Point) -> float:
    return abs(_from[0] - _to[0]) + abs(_from[1] - _to[1])


def euclid(map: np.ndarray, _from: Point, _to: Point) -> float:
    x1, y1 = _from
    x2, y2 = _to
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def euclid_squared(map: np.ndarray, _from: Point, _to: Point) -> float:
    return (_from[0] - _to[0]) ** 2 + (_from[1] - _to[1]) ** 2


def ucs_fallback(map: np.ndarray, _from: Point, _to: Point) -> float:
    return 0


def chebyshev(map: np.ndarray, _from: Point, _to: Point) -> float:
    return max(abs(_from[0] - _to[0]), abs(_from[1] - _to[1]))


# Don't know how this works >-<
def chebyshev_3d(map: np.ndarray, _from: Point, _to: Point) -> float:
    return max(
        abs(_from[0] - _to[0]),
        abs(_from[1] - _to[1]),
        abs(map[_from[1], _from[0]] - map[_to[1], _to[0]]),
    )


def octile(map: np.ndarray, _from: Point, _to: Point) -> float:
    dx = abs(_from[0] - _to[0])
    dy = abs(_from[1] - _to[1])
    return dx + dy + (math.sqrt(2) - 2) * min(dx, dy)


def euclid_3d(map: np.ndarray, _from: Point, _to: Point) -> float:
    x1, y1 = _from
    x2, y2 = _to
    a = (x1 - x2) ** 2 + (y1 - y2) ** 2
    b = (map[y1, x1] - map[y2, x2]) ** 2
    return math.sqrt(a + b)


def diagonal_3d(map: np.ndarray, _from: Point, _to: Point) -> float:
    D1, D2, D3 = (1, math.sqrt(2), math.sqrt(3))
    x1, y1 = _from
    z1 = map[y1, x1]
    x2, y2 = _to
    z2 = map[y2, x2]
    dx, dy, dz = [abs(a - b) for a, b in ((x1, x2), (y1, y2), (z1, z2))]
    dmin = min(dx, dy, dz)
    dmax = max(dx, dy, dz)
    dmid = dx + dy + dz - dmin - dmax
    return (D3 - D2) * dmin + (D2 - D1) * dmid + D1 * dmax


def weighted_manhattan(map: np.ndarray, _from: Point, _to: Point) -> float:
    x1, y1 = _from
    z1 = map[y1, x1]
    x2, y2 = _to
    z2 = map[y2, x2]
    return sum(abs(a - b) / (a + b) for a, b in ((x1, x2), (y1, y2), (z1, z2)))


# %%
from functools import wraps
from time import time

custom_constraint = [(limit_constraint, {"limit": limit})]


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


def test(heuristic: CostFunc, start, goal, timer=True, show=True, save=None):
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
            # for point in f_cost.keys():
            #     color = ImageColor.getrgb("yellow")
            #     img.putpixel(point, color)
            for point in path:
                img.putpixel(point, ImageColor.getrgb("red"))
        if show:
            img.show()
        if save:
            img.save(save)
    except Exception as e:
        print(e)


# %%
s = (0, 255)
g = (255, 255)
test(real_cost, s, g, show=True)
print(real_cost(map, s, g))
print(euclid(map, s, g))

# %%
# def limit_constraint(map, cur: Point, neighbor: Point, **kargs) -> bool:
#     x1, y1 = cur
#     x2, y2 = neighbor
#     limit = kargs["limit"]
#     return abs(map[y1, x1] - map[y2, x2]) <= limit


# def in_bounds_constraint(map, neighbor: Point) -> bool:
#     x, y = neighbor
#     y_limit, x_limit = map.shape
#     return 0 <= x < x_limit and 0 <= y < y_limit


# print(f"limit = {limit}")
# for y in range(map.shape[0]):
#     for x in range(map.shape[1]):
#         cur = (x, y)
#         for move in EIGHT_DIR:
#             neighbor = tuple(i + j for i, j in zip(cur, move))
#             if in_bounds_constraint(map, neighbor):
#                 if not limit_constraint(map, cur, neighbor, limit=limit):
#                     img.putpixel(neighbor, ImageColor.getrgb("yellow"))
# img.show()
