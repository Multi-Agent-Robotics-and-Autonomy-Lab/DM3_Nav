import os
import json
import numpy as np
from os.path import join

def print_cumulative_results(name):
    base_dir = "datadump/results"
    
    file_path = join(base_dir, name, "per_episode_metrics.json")
    with open(file_path, "r") as f:
        metrics = json.load(f)
    
    
    first_ep = next(iter(metrics.values()))
    metric_keys = first_ep["metrics"][0].keys()
    
    stats = {}
    for metric in metric_keys:
        values = [
            y[metric]
            for scene_ep_id in metrics.keys()
            for y in metrics[scene_ep_id]["metrics"]
        ]
        stats[f"{metric}_mean"] = np.round(np.nanmean(values), 4)
        stats[f"{metric}_median"] = np.round(np.nanmedian(values), 4)

    values = [
        metrics[scene_ep_id]["total_num_steps"] - 1
        for scene_ep_id in metrics.keys()
    ]
    stats["timesteps_mean"] = np.round(np.nanmean(values), 4)
    stats["timesteps_median"] = np.round(np.nanmedian(values), 4)
    
    print(f"Cumulative stats: {stats['timesteps_mean']}")
    

if __name__ == "__main__":
    name = "multi_goat_det_steps200_1agents"
    print_cumulative_results(name)
    name = "multi_goat_det_steps200_2agents"
    print_cumulative_results(name)
    name = "multi_goat_det_steps200_3agents"
    print_cumulative_results(name)
    name = "multi_goat_det_steps200_4agents"
    print_cumulative_results(name)