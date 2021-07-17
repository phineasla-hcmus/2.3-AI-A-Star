# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import re
import math
import numpy as np
from PIL import Image, ImageOps, ImageColor
from queue import PriorityQueue
from typing import Callable, Dict, DefaultDict, Generator, Optional
from collections import defaultdict

Node = tuple[int, int]
CostFunc = Callable[[np.ndarray, Node, Node], float]

# NW, N, NE, W, E, SW, S, SE
EIGHT_DIR = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))
# N, W, E, S
FOUR_DIR = ((0, -1), (-1, 0), (1, 0), (0, 1))

with open("input.txt", "r") as f:
    start, goal, [limit] = [
        Node(int(i) for i in re.findall(r"\d+", f.readline())) for j in range(3)
    ]

img = Image.open("./img/map.bmp")
# img = Image.open("./img/test.png")
grayscale = ImageOps.grayscale(img)
map = np.array(grayscale).astype(int)

# %%
def under_limit(map: np.ndarray, cur: Node, neighbor: Node, **kargs) -> bool:
    x1, y1 = cur
    x2, y2 = neighbor
    limit = kargs["limit"]
    return abs(map[y1, x1] - map[y2, x2]) <= limit


def in_bounds(map: np.ndarray, neighbor: Node) -> bool:
    x, y = neighbor
    y_limit, x_limit = map.shape
    return 0 <= x < x_limit and 0 <= y < y_limit


def grid_neighbors(map: np.ndarray, cur: Node, moveset, custom_constraint=None):
    neighbors = []
    for move in moveset:
        neighbor = tuple(i + j for i, j in zip(cur, move))
        if in_bounds(map, neighbor) and all(
            iter()
            if custom_constraint is None
            else (f(map, cur, neighbor, **kwargs) for f, kwargs in custom_constraint)
        ):
            neighbors.append(neighbor)
    return neighbors


# %%
class AStar:
    def __init__(
        self,
        map: np.ndarray,
        neighbor_finder,
        moveset: tuple[Node],
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

    def search(self, start: Node, goal: Node, result_order_from_start: bool = False):
        fringe = PriorityQueue()
        fringe.put([0, start])
        came_from: Dict[Node, Optional[Node]] = {start: None}
        g_cost: DefaultDict[Node, float] = defaultdict(lambda: np.inf)
        g_cost[start] = 0
        f_cost: Dict[Node, float] = {start: self.h(map, start, goal)}

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
            for next in self.neighbors(self.map, cur, self.moveset, self.constraint):
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
def real_cost(map: np.ndarray, _from: Node, _to: Node) -> float:
    x1, y1 = _from
    x2, y2 = _to
    delta = map[y1, x1] - map[y2, x2]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) + (
        0.5 * np.sign(delta) + 1
    ) * abs(delta)


# %%
def manhattan(map: np.ndarray, _from: Node, _to: Node) -> float:
    return abs(_from[0] - _to[0]) + abs(_from[1] - _to[1])


def euclid(map: np.ndarray, _from: Node, _to: Node) -> float:
    x1, y1 = _from
    x2, y2 = _to
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def euclid_squared(map: np.ndarray, _from: Node, _to: Node) -> float:
    return (_from[0] - _to[0]) ** 2 + (_from[1] - _to[1]) ** 2


def ucs_fallback(map: np.ndarray, _from: Node, _to: Node) -> float:
    return 0


def chebyshev(map: np.ndarray, _from: Node, _to: Node) -> float:
    return max(abs(_from[0] - _to[0]), abs(_from[1] - _to[1]))


def chebyshev_3d(map: np.ndarray, _from: Node, _to: Node) -> float:
    return max(
        abs(_from[0] - _to[0]),
        abs(_from[1] - _to[1]),
        abs(map[_from[1], _from[0]] - map[_to[1], _to[0]]),
    )


def octile(map: np.ndarray, _from: Node, _to: Node) -> float:
    dx = abs(_from[0] - _to[0])
    dy = abs(_from[1] - _to[1])
    return dx + dy + (math.sqrt(2) - 2) * min(dx, dy)


def euclid_3d(map: np.ndarray, _from: Node, _to: Node) -> float:
    x1, y1 = _from
    x2, y2 = _to
    a = (x1 - x2) ** 2 + (y1 - y2) ** 2
    b = (map[y1, x1] - map[y2, x2]) ** 2
    return math.sqrt(a + b)


def diagonal_3d(map: np.ndarray, _from: Node, _to: Node) -> float:
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


def weighted_manhattan(map: np.ndarray, _from: Node, _to: Node) -> float:
    x1, y1 = _from
    z1 = map[y1, x1]
    x2, y2 = _to
    z2 = map[y2, x2]
    return sum(abs(a - b) / (a + b) for a, b in ((x1, x2), (y1, y2), (z1, z2)))


# %%
from functools import wraps
from time import time

custom_constraint = [(under_limit, {"limit": limit})]


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        return result, (te - ts)

    return wrap


def display_to_img(img: Image.Image, path: list[Node], show=True, save_to_file=None):
    for point in path:
        img.putpixel(point, ImageColor.getrgb("red"))
    if show:
        img.show()
    if save_to_file:
        img.save(save_to_file)


def test(
    heuristic: CostFunc,
    start=start,
    goal=goal,
    show=False,
    save_img=None,
    save_txt=None,
):
    a_star = AStar(
        map,
        grid_neighbors,
        EIGHT_DIR,
        heuristic,
        real_cost,
        custom_constraint,
    )
    try:
        (path, g_cost, f_cost), time_result = timing(a_star.search)(start, goal)
        print(
            f"""[{heuristic.__name__}] From {start} to {goal} with limit = {limit}
    Elapsed: {time_result}s
    Total cost: {f_cost[goal]}
    Examined nodes: {len(f_cost)}
    Path nodes: {len(path)}
    """
        )
        if show or save_img:
            # for point in f_cost.keys():
            #     color = ImageColor.getrgb("yellow")
            #     result.putpixel(point, color)
            result = img.copy()
            for point in path:
                result.putpixel(point, ImageColor.getrgb("red"))
        if show:
            result.show()
            result.close()
        if save_img:
            result.save(save_img)
            result.close()
        if save_txt:
            with open(save_txt, "w") as f:
                f.write(f"{f_cost[goal]:.2f}\n")
                f.write(f"{len(f_cost)}\n")
    except Exception as e:
        print(e)


# %%
from threading import Thread

heuristics = [
    real_cost,
    ucs_fallback,
    euclid,
    euclid_3d,
    euclid_squared,
    chebyshev,
    chebyshev_3d,
    octile,
    manhattan,
    weighted_manhattan,
    diagonal_3d,
]

for i, h in enumerate(heuristics):
    img_dir = f"./out/map{i}.bmp"
    txt_dir = f"./out/output{i}.txt"
    kwargs = {"heuristic": h, "save_img": img_dir, "save_txt": txt_dir}
    Thread(target=test, kwargs=kwargs).start()
