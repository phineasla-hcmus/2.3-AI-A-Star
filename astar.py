# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import re
from util import *
from queue import PriorityQueue
from collections import defaultdict
from typing import Dict, DefaultDict, Optional
from PIL import Image, ImageOps, ImageColor, ImageDraw


with open("input.txt", "r") as f:
    start, goal, [limit] = [
        tuple(int(i) for i in re.findall(r"\d+", f.readline())) for j in range(3)
    ]

img = Image.open("./img/map.bmp")
grayscale = ImageOps.grayscale(img)
map = np.array(grayscale).astype(int)


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
        self.map = map
        self.h = heuristic
        self.g = real_cost
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
                    f_cost[next] = new_g_cost + self.h(self.map, next, goal)
                    came_from[next] = cur  # Set "next neighbor" parent to cur
                    fringe.put([f_cost[next], next])
        # Open set is empty but goal was never reached
        raise Exception("No solution found")


def test(
    heuristic: CostFunc,
    start=start,
    goal=goal,
    limit=limit,
    show=False,
    save_img=None,
    save_txt=None,
):
    custom_constraint = [(under_limit, {"limit": limit})]
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
            result = Image.new("RGBA", img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(result)
            draw.point([node for node in f_cost.keys()], (0, 0, 255, 100))
            draw.point(path, "red")

        if show:
            Image.alpha_composite(img, result).show()
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
# from threading import Thread

# for i, h in enumerate(heuristics):
#     img_dir = f"./out/map{i+1}.bmp"
#     txt_dir = f"./out/output{i+1}.txt"
#     kwargs = {"heuristic": h, "save_img": img_dir, "save_txt": txt_dir}
#     Thread(target=test, kwargs=kwargs).start()

# test(euclid_3d, show=True)
