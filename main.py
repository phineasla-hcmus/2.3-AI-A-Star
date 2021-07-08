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
Constraint = Callable[[np.ndarray, Point, Point], bool]

moveset = (
    (-1, -1),
    (0, -1),
    (1, -1),  # NW, N, NE
    (-1, 0),
    (1, 0),  # W, E
    (-1, 1),
    (0, 1),
    (1, 1))  # SW, S, SE

with open("input.txt", "r") as f:

    def parse(f):
        return Point(int(i) for i in re.findall(r'\d+', f.readline()))

    start = parse(f)
    goal = parse(f)
    limit = int(f.read())

start, goal, limit

# %%
img = ImageOps.grayscale(Image.open("./img/map.bmp"))
map = np.array(img).astype(int)


# %%
def real_cost(map: np.ndarray, from_pos: Point, to_pos: Point) -> float:
    x1, y1 = from_pos
    x2, y2 = to_pos
    delta = map[to_pos] - map[from_pos]
    return math.sqrt((x2 - x1)**2 +
                     (y2 - y1)**2) + (0.5 * np.sign(delta) + 1) * abs(delta)


real_cost(map, start, goal)

# %%
def in_bounds_constraint(map: np.ndarray, cur: Point, origin: Point) -> bool:
        x, y = cur
        x_limit, y_limit = map.shape
        return 0 <= x < x_limit and 0 <= y < y_limit

def limit_constraint(map: np.ndarray, cur: Point, origin: Point) -> bool:
    


# %%
class AStar:
    def __init__(self, map: np.ndarray, heuristic: CostFunc,
                 real_cost: CostFunc, moveset: tuple[Point]) -> None:
        self.h = heuristic(map)
        self.g = real_cost(map)
        self.map = map
        self.moveset = moveset

    def in_bounds(self, pos: Point) -> bool:
        x, y = pos
        x_limit, y_limit = self.map.shape
        return 0 <= x < x_limit and 0 <= y < y_limit

    def neighbors(self, cur: Point):
        neighbors = []
        for move in moveset:

            neighbors.append(tuple(i + j for i, j in zip(cur, move)))
        return filter(self.in_bounds, neighbors)

    def search(self, start: Point, goal: Point):
        open_set = PriorityQueue()
        came_from: Dict[Point, Optional[Point]] = {start: None}
        g_score: DefaultDict[Point, float] = defaultdict(lambda: np.inf)
        f_score: DefaultDict[Point, float] = defaultdict(lambda: np.inf)
        open_set.put(start)
        g_score[start] = 0
        f_score[start] = 0

        while not open_set.empty():
            cur = open_set.get()
            if cur == goal:
                return "ADD RECONSTRUCT PATH FUNCTION HERE"
            for next in self.neighbors(cur):
                pass


a_star = AStar(map, real_cost, real_cost, moveset)
a_star.search(start, goal)
