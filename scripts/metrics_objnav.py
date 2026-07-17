import json
import os
import numpy as np

def read_metrics(folder):
    # file_path = os.path.join("outputs", folder, "results", "stretch_openvocab__0", "per_episode_metrics.json")
    file_path = os.path.join("datadump", "results", folder, "per_episode_metrics.json")
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

# metrics = read_metrics("hm3d")
# metrics = read_metrics("MA_Burgard_objnav2023_sharemap_hybrid_first2")
# metrics = read_metrics("test_with_yolo_4")
metrics = read_metrics("one_ep_hybrid_stop_early")


results = {episode.split("_")[0]: {"spl":[], "success":[]} for episode in metrics.keys()}

for episode in metrics.keys():
    scene = episode.split("_")[0]
    ep_metrics = metrics[episode]["metrics"][0]

    if np.isnan(ep_metrics["spl"]):
        continue
    results[scene]["spl"].append(ep_metrics["spl"])
    results[scene]["success"].append(ep_metrics["success"])

scene_stats = []

for scene in results.keys():
    spl = np.mean(results[scene]["spl"])
    success = np.mean(results[scene]["success"])
    scene_stats.append((scene, success, spl))

# Sort by success rate (highest first)
scene_stats.sort(key=lambda x: x[1], reverse=True)

for scene, success, spl in scene_stats:
    print(f"\n------- Scene {scene}:")
    print(f"  Success Rate: {success}")
    print(f"  SPL: {spl}")
    
# for scene in results.keys():
#     spl = np.mean(results[scene]["spl"])
#     success = np.mean(results[scene]["success"])
#     print(f"\n------- Scene {scene}:")
#     print(f"  Success Rate: {success}")
#     print(f"  SPL: {spl}")
