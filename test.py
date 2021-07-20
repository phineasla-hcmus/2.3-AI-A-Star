import os
import re
from util import *
from alt import ALT, default_landmarks
from astar import AStar
from threading import Thread
from PIL import Image, ImageOps, ImageColor, ImageDraw

with open("input.txt", "r") as f:
    start, goal, [limit] = [
        tuple(int(i) for i in re.findall(r"\d+", f.readline())) for j in range(3)
    ]

img = Image.open("./img/map.bmp")
grayscale = ImageOps.grayscale(img)
map = np.array(grayscale).astype(int)
custom_constraint = [(under_limit, {"limit": limit})]

# DO NOT USE THIS FOR TIME BENCHMARK
PARALLEL_TEST = False

TRANSPARENCY = 0.25
OPACITY = int(255 * TRANSPARENCY)
PATH_COLOR = ImageColor.getrgb("red")
SPACE_COLOR = (*ImageColor.getrgb("yellow"), OPACITY)
# Draw expanded nodes onto the image
DRAW_SPACE = False

OUT_IMG_DIR = "./out/"
OUT_TXT_DIR = "./out/"

HEURISTICS = [
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


def test(
    pathfinding,
    tag,
    start=start,
    goal=goal,
    limit=limit,
    show=False,
    save_img=None,
    save_txt=None,
):
    try:
        (path, g_cost, f_cost), time_result = timing(pathfinding.search)(start, goal)
        log = f"""[{tag}] From {start} to {goal} with limit = {limit}
    Elapsed: {time_result}s
    Total cost: {f_cost[goal]}
    Examined nodes: {len(f_cost)}
    Path nodes: {len(path)}
    """
        print(log)
        if show or save_img:
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            if DRAW_SPACE:
                draw.point([node for node in f_cost.keys()], SPACE_COLOR)
            draw.point(path, PATH_COLOR)
            result = Image.alpha_composite(img, overlay)
        if show:
            result.show()
        if save_img:
            result.save(save_img)
        if save_txt:
            with open(save_txt, "w") as f:
                f.write(f"{f_cost[goal]:.2f}\n")
                f.write(f"{len(f_cost)}\n")
    except Exception as e:
        print(e)


def astar_test():
    for i, h in enumerate(HEURISTICS):
        # img_dir = f"{OUT_IMG_DIR}{h.__name__}.bmp" if OUT_IMG_DIR else None
        # txt_dir = f"{OUT_TXT_DIR}{h.__name__}.txt" if OUT_TXT_DIR else None
        img_dir = f"{OUT_IMG_DIR}map{i+1}.bmp" if OUT_IMG_DIR else None
        txt_dir = f"{OUT_TXT_DIR}output{i+1}.txt" if OUT_TXT_DIR else None
        astar = AStar(map, grid_neighbors, EIGHT_DIR, h, real_cost, custom_constraint)
        kwargs = {
            "pathfinding": astar,
            "tag": h.__name__,
            "save_img": img_dir,
            "save_txt": txt_dir,
        }
        if PARALLEL_TEST:
            Thread(target=test, kwargs=kwargs).start()
        else:
            test(**kwargs)


def alt_test():
    alt = ALT(map, grid_neighbors, EIGHT_DIR, real_cost, custom_constraint)
    landmarks_file = f"./landmarks8_limit{limit}.npz"
    # Precomputed data was generated with limit = 10
    if not os.path.exists(landmarks_file):
        # For reference, 8 landmarks with 512x512 map was generated in 4 minutes
        print("PRECOMPUTED DATA NOT FOUND, GENERATING NEW ONE, CAN TAKE A WHILE...")
        alt.init_landmarks(default_landmarks(map), real_cost)
        alt.save_landmarks(landmarks_file)
    else:
        alt.load_landmarks(landmarks_file)
    test(
        alt,
        "ALT",
        save_img=f"{OUT_IMG_DIR}map11.bmp" if OUT_IMG_DIR else None,
        save_txt=f"{OUT_IMG_DIR}output11.txt" if OUT_TXT_DIR else None,
    )


if __name__ == "__main__":
    astar_test()
    alt_test()
