import json
import os
import numpy as np

def read_metrics(folder):
    # file_path = os.path.join("outputs", folder, "results", "stretch_openvocab__0", "per_episode_metrics.json")
    file_path = os.path.join("datadump", "results", folder, "per_episode_metrics.json")
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

# metrics1 = read_metrics("multi_hm3d_0.7")
# metrics2 = read_metrics("multi_hm3d_0.7_bc_40_stop60")
metrics1 = read_metrics("multi_hm3d_0.7_new_without_eatly_stop_fake_0_5")
metrics2 = read_metrics("multi_hm3d_0.7_new_without_eatly_stop")

count1 = 0
count2 = 0
for episode in metrics1.keys():
    scene = episode.split("_")[0]
    episode_id = episode.split("_")[1]

    ep_metrics1 = metrics1[episode]["metrics"][0]
    ep_metrics2 = metrics2.get(episode)

    if ep_metrics2 is None:
        # print("Episode not seen yet: ", scene, episode_id)
        pass
    else:
        ep_metrics2 = ep_metrics2["metrics"][0]
        success1 = ep_metrics1["success"]
        success2 = ep_metrics2["success"]
        if success1 != success2:
            if success1 == 1:
                count1 += 1
            else:
                count2 += 1
            print("Epsiode: ", scene, episode_id, 
                    " Success1: ", success1, 
                    " Success2: ", success2)

print(f"Success count for metrics1: {count1}")
print(f"Success count for metrics2: {count2}")



#     results[scene]["success"].append(ep_metrics1["success"])

# for scene in results.keys():
#     spl = np.mean(results[scene]["spl"])
#     success = np.mean(results[scene]["success"])
#     print(f"\n------- Scene {scene}:")
#     print(f"  Success Rate: {success}")
#     print(f"  SPL: {spl}")
