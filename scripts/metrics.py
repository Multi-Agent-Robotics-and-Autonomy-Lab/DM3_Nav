import json
import os
import numpy as np

def read_metrics(folder):
    file_path = os.path.join("datadump", "results", folder, "per_episode_metrics.json")
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

MULTI = True

success_key = "multiagent_goat_success" if MULTI else "goat_sub-task_success"
spl_key = "multiagent_goat_spl" if MULTI else "goat_sub-task_spl"

# metrics = read_metrics("goat_det_yolo")
# metrics = read_metrics("multi_goat_det")
metrics = read_metrics("multi_goat_det_fixed_visited")

results = {episode.split("_")[0]: {success_key:[], spl_key:[]} for episode in metrics.keys()}

for episode in metrics.keys():
    scene = episode.split("_")[0]
    ep_metrics = metrics[episode]["metrics"]


    for goal_metrics in ep_metrics:
        results[scene][success_key].append(goal_metrics[success_key])
        results[scene][spl_key].append(goal_metrics[spl_key])


for scene in results.keys():
    spl = np.mean(results[scene][spl_key])
    success = np.mean(results[scene][success_key])
    print(f"\n------- Scene {scene}:")
    print(f"  Success Rate: {success}")
    print(f"  SPL: {spl}")
