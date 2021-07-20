import json
from util import *
from queue import PriorityQueue
from collections import defaultdict
from typing import Dict, DefaultDict, Optional


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
        # (0, h // 4),
        (0, h // 2),
        # (0, h * 3 // 4),
        (0, h - 1),
        # (w // 4, 0),
        (w // 2, 0),
        # (w * 3 // 4, 0),
        (w - 1, 0),
        # (w - 1, h // 4),
        (w - 1, h // 2),
        # (w - 1, h * 3 // 4),
        (w - 1, h - 1),
        # (w // 4, h - 1),
        (w // 2, h - 1),
        # (w * 3 // 4, h - 1),
        # (w // 2, h // 2),
        # (w // 2, h * 3 // 4),
        # (w * 3 // 4, h // 2),
        # (w * 3 // 4, h * 3 // 4),
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
        neighbor_finder,
        moveset: tuple[Node],
        real_cost: CostFunc,
        custom_constraint=None,
    ) -> None:
        self.map = map
        self.g = real_cost
        self.neighbors = neighbor_finder
        self.moveset = moveset
        self.constraint = custom_constraint
        self.landmarks = {}

    # Precompute distance from each landmark in landmark_coords to all nodes
    def init_landmarks(self, landmark_coords: list[Node], cost: CostFunc = None):
        self.landmarks = {
            landmark: dijkstra(
                landmark,
                self.map,
                self.moveset,
                self.g if cost is None else cost,
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

    def save_landmarks(self, file, compressed=True):
        save = np.savez_compressed if compressed else np.savez
        save(
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
