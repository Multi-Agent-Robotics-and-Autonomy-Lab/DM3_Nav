import os
import json
import numpy as np
from os.path import join

def merge_parallel_results(name, num_episodes=36, chunk_size=5):
    """
    Merge results from parallel runs into a single result folder.
    
    Args:
        name: Base name of the experiment (without the _X_Y suffix)
        num_episodes: Total number of episodes
        chunk_size: Number of episodes per chunk
    """
    base_dir = "datadump/results"
    output_dir = join(base_dir, f"{name}_full")
    os.makedirs(output_dir, exist_ok=True)
    
    merged_metrics = {}
    
    for start in range(0, num_episodes, chunk_size):
        end = min(start + chunk_size, num_episodes)
        chunk_folder = join(base_dir, f"{name}_{start}_{end}")
        per_ep_file = join(chunk_folder, "per_episode_metrics.json")
        
        assert os.path.exists(per_ep_file), f"File not found: {per_ep_file}"
            
        with open(per_ep_file, "r") as f:
            chunk_data = json.load(f)
        
        # Check for duplicates
        for key in chunk_data:
            if key in merged_metrics:
                print(f"Warning: Duplicate episode {key}, overwriting...")
        
        merged_metrics.update(chunk_data)
        print(f"Loaded {len(chunk_data)} episodes from {chunk_folder}")
    
    

    if not merged_metrics:
        print("No metrics to process!")
        
    print(f"\nTotal merged episodes: {len(merged_metrics)}")

    with open(join(output_dir, "per_episode_metrics.json"), "w") as fp:
        json.dump(merged_metrics, fp, indent=4)
    
    
    first_ep = next(iter(merged_metrics.values()))
    metric_keys = first_ep["metrics"][0].keys()
    
    stats = {}
    for metric in metric_keys:
        values = [
            y[metric]
            for scene_ep_id in merged_metrics.keys()
            for y in merged_metrics[scene_ep_id]["metrics"]
        ]
        stats[f"{metric}_mean"] = np.round(np.nanmean(values), 4)
        stats[f"{metric}_median"] = np.round(np.nanmedian(values), 4)
    
    with open(join(output_dir, "cumulative_metrics.json"), "w") as fp:
        json.dump(stats, fp, indent=4)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Cumulative stats: {stats}")
    
    return merged_metrics, stats


if __name__ == "__main__":
    name = "goat_det"
    merge_parallel_results(name, num_episodes=10, chunk_size=5)