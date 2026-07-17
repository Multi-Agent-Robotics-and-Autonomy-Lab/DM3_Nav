import os
import sys
import json
import rospy
import argparse
import threading
import numpy as np
import time 
from tqdm import tqdm
from pathlib import Path

# TODO Install home_robot, home_robot_sim and remove this
base_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(
    0,
    str(base_path / "src/home_robot"),
)
sys.path.insert(
    0,
    str(base_path / "src/home_robot_hw"),
)

from omegaconf import DictConfig, OmegaConf
from home_robot.utils.logger import get_logger
from home_robot_hw.env.scout_goat_env import ScoutGoatEnv
# from home_robot.agent.goat_agent.goat_agent import GoatAgent
from home_robot.agent.goat_agent.realworld_goat_agent import RealWorldGoatAgent
from home_robot.core.interfaces import DiscreteNavigationAction

def read_args():
    """
    These options override default configs.
    """
    parser = argparse.ArgumentParser()
    project_config_default = "projects/habitat_goat/configs/agent/scout_eval.yaml"
    parser.add_argument(
        "--project_config_path",
        type=str,
        default=project_config_default,
        help="Path to config yaml",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name of the experiment (overrides EXP_NAME in config)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["habitat_objnav_2022", "habitat_objnav_2023", "goat"],
    )
    parser.add_argument(
        "--scene",
        type=int,
        nargs="*",
        default=None,
        metavar=("START", "END"),
        help="Scenes range: --scene [start] [end]",
    )

    parser.add_argument(
        "--yolo",
        action="store_const",
        const=1,
        default=None,
        help="Enable YOLO (1 if passed, None otherwise)",
    )

    parser.add_argument(
        "--cat_match_threshold",
        type=float,
        default=None,
        help="score threshold for category matching in GOAT",
    )
    parser.add_argument(
        "--task_config",
        type=str,
        default="projects/habitat_goat/configs/example_tasks.json",
        help="Path to task config file",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="Number of episodes to run",
    )
    args = parser.parse_args()
    return args


def read_configs(args):
    config = OmegaConf.load(args.project_config_path)
    config.NUM_AGENTS = 1
    config.GROUND_TRUTH_SEMANTICS = False

    if args.name is not None:
        config.EXP_NAME = args.name
    
    if args.yolo is not None:
        config.USE_YOLO = 1
    
    
    if args.cat_match_threshold is not None:
        config.AGENT.cat_match_threshold = args.cat_match_threshold

    config.REAL_WORLD = True
    return config


def save_results(results, env, results_dir, ep_step, all_subtask_metrics, agent, obs):
    obs_tasks = []
    for task in obs.task_observations["tasks"]:
        obs_task = {}
        for key, value in task.items():
            if key == "image":
                continue
            obs_task[key] = value
        obs_tasks.append(obs_task)

    ep_results = {}
    all_subtask_metrics = [all_subtask_metrics[key] for key in sorted(all_subtask_metrics.keys())]
    for m in all_subtask_metrics:
        if "spl" in m and np.isnan(m["spl"]):
            m["success"] = np.nan
    ep_results["metrics"] = all_subtask_metrics
    ep_results["total_num_steps"] = ep_step
    if agent.seq_goals:
        ep_results["sub_task_timesteps"] = agent.subtask_timesteps[:len(all_subtask_metrics)]
    ep_results["tasks"] = obs_tasks

    for metric in all_subtask_metrics[0].keys():
        values = [y[metric] for y in all_subtask_metrics]
        ep_results[f"{metric}_mean"] = np.round(
            np.nanmean(values),
            4,
        )
        ep_results[f"{metric}_median"] = np.round(
            np.nanmedian(values),
            4,
        )

    results[f"{env.scene_id}_{env.episode_id}"] = ep_results
    for scene_ep_id in results:
        for m in results[scene_ep_id]["metrics"]:
            if "spl" in m and np.isnan(m["spl"]):
                m["success"] = np.nan
                m["distance_to_goal"] = np.nan
    with open(os.path.join(results_dir, "per_episode_metrics.json"), "w") as fp:
        json.dump(results, fp, indent=4)


    stats = {}
    for metric in all_subtask_metrics[0].keys():
        values = [
            y[metric]
            for scene_ep_id in results.keys()
            for y in results[scene_ep_id]["metrics"]
        ]
        stats[f"{metric}_mean"] = np.round(
            np.nanmean(values),
            4,
        )
        stats[f"{metric}_median"] = np.round(
            np.nanmedian(values),
            4,
        )

    with open(os.path.join(results_dir, "cumulative_metrics.json"), "w") as fp:
        json.dump(stats, fp, indent=4)

def main():
    args = read_args()

    print("Arguments:")
    print(json.dumps(vars(args), indent=4))
    print("-" * 100)

    logger = get_logger()

    config =  read_configs(args)
    rospy.init_node("eval_episode_scout_goat")
    rate = rospy.Rate(config.ENVIRONMENT.control_frequency)

    logger.info("Starting code")

    env = ScoutGoatEnv(
        config=config,
        task_config_file=args.task_config,
    )
    camera_params = env.sync_camera_params()

    # agent: GoatAgent = GoatAgent(
    #     config, env.semantic_category_mapping.vocabulary
    # )
    agent: RealWorldGoatAgent= RealWorldGoatAgent(
        config, env.semantic_category_mapping.vocabulary, camera_params, agent_id=1, adhoc_ip="10.0.0.2"
    )

    results_dir = os.path.join(config.DUMP_LOCATION, "results", config.EXP_NAME)
    os.makedirs(results_dir, exist_ok=True)

    results = {}
    results_file = os.path.join(results_dir, "per_episode_metrics.json")
    if os.path.exists(results_file):
        with open(results_file, "r") as fp:
            results = json.load(fp)

    max_episodes = args.num_episodes if args.num_episodes else len(env.episodes)
    for i in range(max_episodes):
        env.reset()
        logger.info(f"Evaluating scene {env.scene_id} episode {env.episode_id}")
        if f"{env.scene_id}_{env.episode_id}" in list(results.keys()):
            continue
        
        env.reset_vis_dir()
        agent.reset(env.scene_id, env.episode_id)

        episode_start = time.time()
        ep_step = 0
        all_subtask_metrics = {}
        pbar = tqdm(
            total=config.AGENT.max_steps, file=sys.__stdout__, dynamic_ncols=True
        )

        old_task_idx = -1
        while not env.episode_over and not rospy.is_shutdown():
            if (time.time() - episode_start) / 60 > config.ENVIRONMENT.episode_timeout_minutes:
                logger.warning("Episode timeout")
                break
            if config.SEQ:
                if env.current_task_idx != old_task_idx:
                    logger.info(
                        f"Starting task {env.current_task_idx} in scene {env.scene_id} episode {env.episode_id}"
                    )
                    old_task_idx = env.current_task_idx
            ep_step += 1
            logger.info(
                f"-------------------- Episode step {ep_step} --------------------"
            )
            env.timestep = agent.get_subtask_timestep() + 1

            obs = env.get_observation()
            agent.update_state(obs)
            action, info, stuck = agent.act()
            if stuck and action["action"] != DiscreteNavigationAction.STOP: 
                action = agent._process_action(DiscreteNavigationAction.STOP)
                agent.handle_stop(action)

            logger.info(f"Action taken: {action['action']}")
            update_thread = threading.Thread(target=agent.update_maps)
            update_thread.start()
            env.apply_action(action, info)
            update_thread.join()
            pbar.update(1)

            if action["action"] == DiscreteNavigationAction.STOP:
                env.add_subepisode_metrics(all_subtask_metrics, action)
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break

        pbar.reset()
        logger.info(
            f"------------------------ Episode {env.scene_id} {env.episode.episode_id} over ------------------------"
        )
        pbar.close()
        while not rospy.is_shutdown():
            print("sleeping")
            try:
                rospy.sleep(5)
            except:
                print("Exiting...")
        # save_results(results, env, results_dir, ep_step, all_subtask_metrics, agent, obs)

if __name__ == "__main__":
    main()
