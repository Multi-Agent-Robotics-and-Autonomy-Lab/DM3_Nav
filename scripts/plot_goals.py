import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

scene_name = "7MXmsvcQjpJ"
file_name = f"/home-robot/data/datasets/objectnav/hm3d/v2/val/content/{scene_name}.json"

# Either provide ep_id or categories
ep_id = 20
start_position = None

# categories = ["bed", "toilet"]



output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

with open(file_name, "r") as f:
    data = json.load(f)

if not ep_id is None:
    scene_data = data["episodes"][ep_id]
    categories = [scene_data["object_category"]]
    start_position = scene_data["start_position"]
    
goals_by_category = data["goals_by_category"]

by_category = defaultdict(list)

for _, goals in goals_by_category.items():
    for goal in goals:
        cat = goal.get("object_category")
        if cat not in categories:
            continue

        gx, gy, gz = goal["position"]

        if not start_position is None:
            if gy > start_position[1] + 1.5 or gy < start_position[1] - 1.5:
                continue
        # if gy > 1:
        #     continue
        view_pts = []
        for vp in goal.get("view_points", []):
            px, _, pz = vp["agent_state"]["position"]
            view_pts.append((px, pz))

        by_category[cat].append({
            "goal": (gx, gz),
            "views": view_pts,
        })

category_cmaps = {
    "plant": plt.cm.Greens,
    "chair": plt.cm.Blues,
    "toilet": plt.cm.Reds,
    "sofa": plt.cm.Purples,
    "bed": plt.cm.Greys,
}

default_cmap = plt.cm.Greys

plt.figure(figsize=(10, 12))

for cat, instances in by_category.items():
    cmap = category_cmaps.get(cat, default_cmap)

    n = len(instances)
    if n == 0:
        continue
    print("len for cat:", cat, n)

    shades = np.linspace(0.35, 0.9, n)

    for idx, inst in enumerate(instances):
        color = cmap(shades[idx])

        # Viewpoints
        if inst["views"]:
            xs, zs = zip(*inst["views"])
            plt.scatter(
                xs,
                zs,
                s=12,
                alpha=0.5,
                color=color,
            )

        # Goal position
        gx, gz = inst["goal"]
        plt.scatter(
            gx,
            gz,
            s=140,
            marker="x",
            linewidths=2,
            color=color,
        )

if start_position is not None:
    plt.scatter(
        [start_position[0]],
        [start_position[2]],
        s=12,
        linewidths=2,
        color="red",
    )
plt.xlabel("X")
plt.ylabel("Z")
plt.title(f"Goal Viewpoints (colored by instance, shaded by category)")
plt.axis("equal")
plt.grid(True)
plt.gca().invert_yaxis()

out_path = os.path.join(
    output_dir,
    f"categories_{'_'.join(categories)}_instances.png"
)
plt.savefig(out_path, dpi=150)
plt.close()

print(f"Saved: {out_path}")
