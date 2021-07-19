import math
import numpy as np
from time import time
from functools import wraps
from typing import Callable


Node = tuple[int, int]
CostFunc = Callable[[np.ndarray, Node, Node], float]

# NW, N, NE, W, E, SW, S, SE
EIGHT_DIR = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))
# N, W, E, S
FOUR_DIR = ((0, -1), (-1, 0), (1, 0), (0, 1))


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


def manhattan(map: np.ndarray, _from: Node, _to: Node) -> float:
    return abs(_from[0] - _to[0]) + abs(_from[1] - _to[1])


def euclid(map: np.ndarray, _from: Node, _to: Node) -> float:
    x1, y1 = _from
    x2, y2 = _to
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# Extremely inadmissible heuristic
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
    return sum(
        abs(a - b) / (a + b) if a + b else 0 for a, b in ((x1, x2), (y1, y2), (z1, z2))
    )


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
