import os
import sys
import json
import yaml
from typing import List, Union
from tqdm import tqdm
from pathlib import Path
import cProfile
import pstats


# TODO Install home_robot, home_robot_sim and remove this
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent / "src/home_robot"),
)
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent / "src/home_robot_sim"),
)

from habitat.core.env import Env
from omegaconf import DictConfig, OmegaConf
from habitat.config.default import get_config
from home_robot.utils.logger import get_logger
from home_robot.core.interfaces import DiscreteNavigationAction
from home_robot.agent.goat_agent.multiagent_goat_agent import SimulationMultiAgentGoatAgent
from home_robot.agent.goat_agent.realworld_goat_agent import RealWorldGoatAgent
from home_robot_sim.env.habitat_goat_env.habitat_goat_env import (
    MultiAgentHabitatGoatEnv,
)

from eval_episode import read_args, save_results

DATASET_CONFIGS = {
    "habitat_objnav_2023": "benchmark/nav/objectnav/objectnav_hm3d_rgbd_with_semantic.yaml",  # V2
    "goat": "benchmark/nav/goat/multiagent_goat_hm3d_rgbd_with_semantic.yaml",
}

def read_configs(args):
    project_config = OmegaConf.load(args.project_config_path)
    if args.dataset is not None:
        project_config.DATASET = args.dataset
    habitat_config_path = DATASET_CONFIGS[project_config.DATASET]
    habitat_config = get_config(habitat_config_path)
    config = DictConfig({**habitat_config, **project_config})

    if project_config.DATASET == "goat":
        config.habitat.dataset.split = "val_seen"
    else:
        config.habitat.dataset.split = "val"

    config.habitat.task.type = "MultiAgent" + config.habitat.task.type
    config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.min_depth = 0.0

    base_agent_conf = config.habitat.simulator.agents.pop("main_agent")
    import copy

    agents = []
    for i in range(config.NUM_AGENTS):
        agents.append(f"agent{i}")
        agent_conf = copy.deepcopy(base_agent_conf)

        config.habitat.simulator.agents[f"agent{i}"] = agent_conf
    config.habitat.simulator.agents_order = agents

    # with open("./ma_merged_config.yaml", "w") as f:
    #     f.write(yaml.dump(OmegaConf.to_container(config), sort_keys=False))

    all_scenes = os.listdir(
        os.path.dirname(
            config.habitat.dataset.data_path.format(split=config.habitat.dataset.split)
        )
        + "/content/"
    )
    all_scenes = sorted([x.split(".")[0] for x in all_scenes if x.endswith(".json.gz")])
    logger.debug(f"All scenes: {all_scenes}")

    scenes = slice(None, None)
    if args.scene is not None:
        if len(args.scene) == 1:
            scenes = slice(args.scene[0], None)

        if len(args.scene) == 2:
            scenes = slice(args.scene[0], args.scene[1])

    config.habitat.dataset.content_scenes = all_scenes[scenes]

    if args.name is not None:
        config.EXP_NAME = args.name
    
    if args.yolo is not None:
        config.USE_YOLO = 1
    
    if args.gt is not None:
        config.GROUND_TRUTH_SEMANTICS = 1
    
    if args.cat_match_threshold is not None:
        config.AGENT.cat_match_threshold = args.cat_match_threshold

    return config


if __name__ == "__main__":
    args = read_args()

    print("Arguments:")
    print(json.dumps(vars(args), indent=4))
    print("-" * 100)

    logger = get_logger()

    config = read_configs(args)

    logger.info("Starting code")
    logger.info(f"Using scenes: {config.habitat.dataset.content_scenes}")

    habitat_env = Env(config)
    env = MultiAgentHabitatGoatEnv(habitat_env, config=config)
    agents: List[SimulationMultiAgentGoatAgent] = []
    # agents: List[RealWorldGoatAgent] = []
    for i in range(config.NUM_AGENTS):
        agents.append(SimulationMultiAgentGoatAgent(config, env.semantic_category_mapping.vocabulary, agent_id=i))
        # agents.append(RealWorldGoatAgent(config, env.semantic_category_mapping.vocabulary, i))

    results_dir = os.path.join(config.DUMP_LOCATION, "results", config.EXP_NAME)
    os.makedirs(results_dir, exist_ok=True)

    results = {}
    results_file = os.path.join(results_dir, "per_episode_metrics.json")
    if os.path.exists(results_file):
        with open(results_file, "r") as fp:
            results = json.load(fp)

    for i in range(len(env.habitat_env.episodes)):
        env.reset()
        logger.info(f"Evaluating scene {env.scene_id} episode {env.episode_id}")
        if f"{env.scene_id}_{env.episode_id}" in list(results.keys()):
            continue

        env.reset_vis_dir()
        for agent in agents:
            agent.reset(env.scene_id, env.episode_id)

        ep_step = 0
        all_subtask_metrics = env.init_subepisode_metrics()
        pbar = tqdm(
            total=config.AGENT.max_steps, file=sys.__stdout__, dynamic_ncols=True
        )
        pbar.set_description(f"{env.scene_id}_{env.episode_id}")

        # profiler = cProfile.Profile()
        # profiler.enable()
        while not env.episode_over:

            ep_step += 1
            logger.info(
                f"-------------------- Episode step {ep_step} --------------------"
            )
            for i in range(len(agents)):
                logger.debug(
                    f"Agent{i} state: {env.habitat_env.sim.agents[i].get_state().position}"
                )
            env.timestep = agent.get_subtask_timestep() + 1
            observations = env.get_observation()
            if len(agents) == 1:
                observations = [observations]

            actions = []
            infos = []
            stucks = []
            for agent, obs in zip(agents, observations):
                agent.update_state(obs)
                agent.update_maps()

            for agent in agents:
                other_agents = list(
                    filter(lambda x: x.agent_id != agent.agent_id, agents)
                )
                agent.simulate_receive_map(other_agents)
                action, info, stuck = agent.act()

                actions.append(action)
                infos.append(info)
                stucks.append(stuck)

            if all(stucks):
                actions = []
                for agent in agents:
                    action = agent._process_action((None, DiscreteNavigationAction.STOP))
                    agent.handle_stop(action)
                    actions.append(action)

            any_moving = not all(
                action["action"] == DiscreteNavigationAction.STOP for action in actions
            )
            # Don't stop the episode if an agent is still looking. Only needed in simulation and objectnav (because of using Stop instead of MultiAgentStop)
            for action, agent in zip(actions, agents):
                if action["action_args"]["task_idx"] is None and action["action"] == DiscreteNavigationAction.STOP:
                    if any_moving:
                        action["action"] = DiscreteNavigationAction.TURN_RIGHT

            logger.info(f"Actions taken: {actions}")
            env.apply_action(actions, info=infos)
            pbar.update(1)

            stopped_agent = next(
                (i for i, a in enumerate(actions) if a["action"] == 0), None
            )

            if stopped_agent is None:
                continue

            env.add_subepisode_metrics(all_subtask_metrics, actions)

        # profiler.disable()
        # stats = pstats.Stats(profiler)
        # # stats.sort_stats('tottime')
        # stats.sort_stats('cumulative')
        # stats.print_stats(100)

        logger.info(
            f"------------------------ Episode {env.scene_id} {env.episode.episode_id} over ------------------------"
        )
        pbar.close()
        save_results(
            results, env, results_dir, ep_step, all_subtask_metrics, agent, obs
        )
