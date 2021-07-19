import math
import numpy as np
from time import time
from functools import wraps
from typing import Callable


Node = tuple[int, int]
CostFunc = Callable[[np.ndarray, Node, Node], float]

# NW, N, NE, W, E, SW, S, SE
EIGHT_DIR = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))


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


def real_cost(map: np.ndarray, _from: Node, _to: Node) -> float:
    x1, y1 = _from
    x2, y2 = _to
    delta = map[y1, x1] - map[y2, x2]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) + (
        0.5 * np.sign(delta) + 1
    ) * abs(delta)


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        if result is None:
            return te - ts
        else:
            return result, (te - ts)

    return wrap
