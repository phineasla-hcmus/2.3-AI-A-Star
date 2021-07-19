import json
from util import *
from queue import PriorityQueue
from collections import defaultdict
from typing import Dict, DefaultDict, Optional
from PIL import Image, ImageOps, ImageColor


def dijkstra(
    start: Node, map: np.ndarray, moveset, cost_func, neighbors_func, constraint=None
):
    fringe = PriorityQueue()
    fringe.put([0, start])
    g_cost = np.full(map.shape, np.inf, float)
    g_cost[start[1], start[0]] = 0

    while not fringe.empty():
        cur = fringe.get()[1]
        for next in neighbors_func(map, cur, moveset, constraint):
            new_cost = g_cost[cur[1], cur[0]] + cost_func(map, cur, next)
            if new_cost < g_cost[next[1], next[0]]:
                g_cost[next[1], next[0]] = new_cost
                fringe.put([new_cost, next])
    return g_cost


def default_landmarks(map: np.ndarray):
    h, w = map.shape
    return [
        (0, 0),
        (0, h // 2),
        (0, h - 1),
        (w // 2, 0),
        (w - 1, 0),
        (w - 1, h - 1),
        (w - 1, h // 2),
        (w // 2, h - 1),
    ]


def compute_landmarks(
    landmarks, map: np.ndarray, moveset, cost_func, neighbors_func, constraint=None
):
    return (
        dijkstra(landmark, map, moveset, cost_func, neighbors_func, constraint)
        for landmark in landmarks
    )


class ALT:
    def __init__(
        self,
        map: np.ndarray,
        moveset: tuple[Node],
        neighbor_finder,
        heuristic,
        real_cost,
        custom_constraint=None,
    ) -> None:
        self.map = map
        self.moveset = moveset
        self.neighbors = neighbor_finder
        self.landmarks = {}
        self.landmark_heursitic = heuristic
        self.g = real_cost
        self.constraint = custom_constraint

    def init_landmarks(self, landmark_coords: list[Node]):
        self.landmarks = {
            landmark: dijkstra(
                landmark,
                self.map,
                self.moveset,
                self.landmark_heursitic,
                self.neighbors,
                self.constraint,
            )
            for landmark in landmark_coords
        }

    def load_landmarks(self, file, clear=True):
        if clear:
            self.landmarks.clear()
        with np.load(file) as data:
            for landmark, distance in data.items():
                self.landmarks[tuple(json.loads(landmark))] = distance

    def save_landmarks(self, file):
        np.savez(
            file,
            **{
                json.dumps(landmark): distance
                for landmark, distance in self.landmarks.items()
            },
        )

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

    def h(self, _from, _to):
        # for each landmark, abs(landmark_dis[from] - landmark_dis[to]), then get max
        return max(
            abs(landmark[_from[1], _from[0]] - landmark[_to[1], _to[0]])
            for landmark in self.landmarks.values()
        )

    def search(self, start: Node, goal: Node, result_order_from_start: bool = False):
        fringe = PriorityQueue()
        fringe.put([0, start])
        came_from: Dict[Node, Optional[Node]] = {start: None}
        g_cost: DefaultDict[Node, float] = defaultdict(lambda: np.inf)
        g_cost[start] = 0
        f_cost: Dict[Node, float] = {start: self.h(start, goal)}

        while not fringe.empty():
            cur = fringe.get()[1]
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
                    g_cost[next] = new_g_cost
                    f_cost[next] = new_g_cost + self.h(next, goal)
                    came_from[next] = cur  # Set "next neighbor" parent to cur
                    fringe.put([f_cost[next], next])
        # Open set is empty but goal was never reached
        raise Exception("No solution found")


img = Image.open("./img/map.bmp")
grayscale = ImageOps.grayscale(img)
map = np.array(grayscale).astype(int)

heuristic = real_cost
start = (0, 0)
goal = (511, 511)
limit = 10

astar = ALT(
    map,
    EIGHT_DIR,
    grid_neighbors,
    heuristic,
    real_cost,
    [(under_limit, {"limit": limit})],
)

# time_took = timing(astar.init_landmarks)(default_landmarks(map))
# print(f"Elapsed: {time_took}s")
# astar.save_landmarks("landmarks")
astar.load_landmarks("landmarks.npz")
(path, g_cost, f_cost), time_result = timing(astar.search)(start, goal)
print(
    f"""[{heuristic.__name__}] From {start} to {goal} with limit = {limit}
    Elapsed: {time_result}s
    Total cost: {f_cost[goal]}
    Examined nodes: {len(f_cost)}
    Path nodes: {len(path)}
    """
)
