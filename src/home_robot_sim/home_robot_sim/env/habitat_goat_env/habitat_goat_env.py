from typing import Any, Dict, List, Optional, Tuple, Union, cast

import os
import habitat
import numpy as np
from habitat.sims.habitat_simulator.actions import HabitatSimActions

import home_robot
from home_robot.perception.constants import (
    SemanticCategoryMapping,
    HabitatObjNav2022Categories,
    GoatCategories,
)
from home_robot.utils.constants import (
    MAX_DEPTH_REPLACEMENT_VALUE,
    MIN_DEPTH_REPLACEMENT_VALUE,
)
from home_robot_sim.env.habitat_abstract_env import HabitatEnv
from home_robot_sim.env.habitat_goat_env.visualizer import Visualizer
from home_robot.perception.detection.maskrcnn.maskrcnn_perception import (
    MaskRCNNPerception,
)


from home_robot.utils.logger import get_logger
from home_robot.utils.visualization import visualize_semantic_with_labels

logger = get_logger()


class HabitatGoatEnv(HabitatEnv):
    semantic_category_mapping: SemanticCategoryMapping

    def __init__(self, habitat_env: habitat.core.env.Env, config):
        super().__init__(habitat_env)

        self.config = config
        self.min_depth = (
            config.habitat.simulator.agents.agent0.sim_sensors.depth_sensor.min_depth
        )
        self.max_depth = (
            config.habitat.simulator.agents.agent0.sim_sensors.depth_sensor.max_depth
        )
        self.ground_truth_semantics = config.GROUND_TRUTH_SEMANTICS
        self.task_type = config.habitat.task.type
        self.timestep = 0
        self.current_episode = None


        if config.AGENT.SEMANTIC_MAP.semantic_categories == "dataset":
            dataset_type = config.habitat.dataset.type
            if dataset_type == "Goat-v1":
                self.semantic_category_mapping = GoatCategories(self.fetch_vocabulary())
                logger.debug(f"{self.semantic_category_mapping.cat_id_to_goal_name}")
            elif dataset_type == "ObjectNav-v1":
                self.semantic_category_mapping = HabitatObjNav2022Categories()
            else:
                raise NotImplementedError
        else:
            #! This is when we want to use custom vocabularies, and not only possible goals
            raise NotImplementedError

        self.semantic_category_mapping.reset_instance_id_to_category_id(
            self.habitat_env
        )

        self.visualization_level = config.VISUALIZATION_LEVEL
        if self.visualization_level > 0:
            self.visualizer = Visualizer(config, self.semantic_category_mapping)


    def fetch_vocabulary(self):
        if self.config.habitat.dataset.type == "Goat-v1":
            vocabulary = sorted(
                {
                    "_".join(cat.split(" "))
                    for cat in self.habitat_env._dataset.all_categories
                }
            )
            logger.info(f"Vocabulary: {vocabulary}")
        else:
            raise NotImplementedError

        return vocabulary

    def reset(self):
        habitat_obs = self.habitat_env.reset()
        self.current_episode = self.habitat_env.current_episode

        self._last_obs = self._preprocess_obs(habitat_obs)
        if self.visualization_level > 0:
            self.visualizer.reset()

        self.scene_id = self.habitat_env.current_episode.scene_id.split("/")[-1].split(
            "."
        )[0]
        self.episode = self.habitat_env.current_episode
        self.episode_id = self.episode.episode_id

        self.semantic_category_mapping.reset_instance_id_to_category_id(
            self.habitat_env
        )
    
    def reset_vis_dir(self):
        if self.visualization_level > 0:
            dir_name = f"{self.scene_id}_{self.episode_id}"
            if self.config.SEQ:
                dir_name = f"{dir_name}_{self.current_task_idx}"
            self.visualizer.set_vis_dir(
                dir_name
            )

    def _preprocess_obs(
        self, habitat_obs: habitat.core.simulator.Observations
    ) -> home_robot.core.interfaces.Observations:
        if habitat_obs.get("multigoal") is None:
            # I can also get habitat_obs["objectgoal"] here, but it is not compatible, because multigoal returns text category
            goals = self._preprocess_goals(
                [{"category": self.current_episode.object_category}]
            )
        else:
            goals = self._preprocess_goals(habitat_obs["multigoal"])

        observations = []
        for agent_id in range(self.config.NUM_AGENTS):
            agent_obs = habitat_obs[agent_id]
            depth = self._preprocess_depth(agent_obs[f"depth"])
            obs = home_robot.core.interfaces.Observations(
                rgb=agent_obs["rgb"],
                depth=depth,
                compass=habitat_obs["compass"][agent_id],
                gps=self._preprocess_xy(habitat_obs["gps"][agent_id]),
                task_observations={
                    "tasks": goals,
                    "top_down_map": self.get_episode_metrics().get("goat_top_down_map"),
                },
                camera_pose=None,
                third_person_image=None,
            )
            obs = self._preprocess_semantic(obs, agent_obs[f"semantic"])
            observations.append(obs)
        
        if len(observations) == 1:
            return observations[0]
        return observations

    def visualize_semantic_with_labels(
        self,
        semantic_array: np.ndarray,
        palette: list,
        postfix: str = "",
    ):

        save_path = os.path.join(
            self.visualizer.vis_dir, f"{self.timestep+1}_0.sem_input{postfix}.png"
        )
        visualize_semantic_with_labels(semantic_array, palette, save_path)

    def _preprocess_semantic(
        self, obs: home_robot.core.interfaces.Observations, habitat_semantic: np.ndarray
    ) -> home_robot.core.interfaces.Observations:
        habitat_semantic = habitat_semantic[:,:,-1]
        if self.ground_truth_semantics:
            # * shape of habitat_semantic: (H, W, 1), shape of obs.semantic: (H, W) (only numbers change)

            # self.visualize_semantic_with_labels(
            #     semantic_array=habitat_semantic,
            #     palette=self.semantic_category_mapping.map_color_palette,
            #     postfix="_gt",
            # )
            # NEW
            instance_id_to_category_id = (
                self.semantic_category_mapping.instance_id_to_category_id
            )
            max_id = habitat_semantic.max()
            if max_id >= len(instance_id_to_category_id):
                #! This can happen, because of the problems in labels. For some reason, the label is not in the list of all objects. This is not a problem of the code.
                logger.warning(f"Warning: semantic ID {max_id} exceeds mapping size {len(instance_id_to_category_id)}")
            habitat_semantic[habitat_semantic >= len(instance_id_to_category_id)] = 0 # 0 is unknown
            obs.semantic = instance_id_to_category_id[habitat_semantic]
            obs.task_observations["instance_frame"] = habitat_semantic
            obs.task_observations["instance_scores"] = np.ones(np.unique(habitat_semantic).shape[0])
            # self.visualize_semantic_with_labels(
            #     semantic_array=obs.semantic + 10,
            #     palette=self.semantic_category_mapping.map_color_palette,
            # )

            # OLD
            # obs.semantic = np.vectorize(lambda x: self.hm3d_mapping.get(x, 0))(habitat_semantic)[..., 0]
            # obs.task_observations["instance_map"] = habitat_semantic[:, :, -1] + 1
            # self.visualize_semantic_with_labels(
            #     semantic_array=obs.semantic+10,
            #     palette=self.semantic_category_mapping.map_color_palette,
            # )
        return obs

    def _preprocess_depth(self, depth: np.array) -> np.array:
        rescaled_depth = self.min_depth + depth * (self.max_depth - self.min_depth)
        rescaled_depth[depth == 0.0] = MIN_DEPTH_REPLACEMENT_VALUE
        rescaled_depth[depth == 1.0] = MAX_DEPTH_REPLACEMENT_VALUE
        return rescaled_depth[:, :, -1]

    def _preprocess_goals(self, goals):
        for goal_v in goals:
            goal_v["semantic_id"] = self.semantic_category_mapping.goal_name_to_cat_id[
                "_".join(goal_v["category"].split(" "))
            ]
            if goal_v.get("image") is not None:
                goal_v["type"] = "imagenav"
            elif goal_v.get("description") is not None:
                goal_v["type"] = "languagenav"
            else:
                goal_v["type"] = "objectnav"

        return goals

    def _preprocess_action(self, action) -> int:
        action["action"] =  HabitatSimActions[action["action"].name.lower()]
        return action

    def _process_info(self, info: Dict[str, Any]) -> Any:
        if self.visualization_level > 0:
            self.visualizer.visualize(**info)


    def apply_action(
        self,
        action: List[home_robot.core.interfaces.Action],
        info: Optional[Dict[str, Any]] = None,
        prev_obs: Optional[home_robot.core.interfaces.Observations] = None,
    ):
        super().apply_action(action, info, prev_obs)
    
    def add_subepisode_metrics(self, all_metrics, action):
        ep_metrics = self.get_episode_metrics()
        ep_metrics.pop("goat_top_down_map", None)

        task_idx = action["action_args"]["task_idx"]

        if self.task_type == "Goat-v1":
            ep_metrics["goat_sub-task_success"] = ep_metrics["goat_sub-task_success"][task_idx]
            ep_metrics["goat_distance_to_sub-goal"] = ep_metrics["goat_distance_to_sub-goal"][0]
            all_metrics[task_idx] = ep_metrics
        else:
            ep_metrics["distance_to_goal"] = ep_metrics["distance_to_goal"][0]
            all_metrics[0] = ep_metrics

        logger.info("-------------------------")
        logger.info(
            f"{self.scene_id}_{self.episode_id}_{task_idx} {ep_metrics}"
        )
        logger.info("-------------------------")
    
    @property
    def current_task_idx(self) -> int:
        return self.habitat_env.task.current_task_idx if "Goat-v1" in self.task_type else 0

class MultiAgentHabitatGoatEnv(HabitatGoatEnv):
    semantic_category_mapping: SemanticCategoryMapping

    def _preprocess_action(self, actions: List[home_robot.core.interfaces.Action]) -> int:
        for action in actions:
            action["action"] =  HabitatSimActions[action["action"].name.lower()]
        return actions

    def _process_info(self, infos: List[Dict[str, Any]]) -> Any:
        for info in infos:
            super()._process_info(info)

    def add_subepisode_metrics(self, all_metrics, actions):
        ep_metrics = self.get_episode_metrics()
        ep_metrics.pop("goat_top_down_map", None)

        stopped_agents = []
        stopped_tasks = []
        for a in actions:
            if a["action"] == 0:
                stopped_agents.append(a["action_args"]["agent_id"])
                stopped_tasks.append(a["action_args"]["task_idx"])

        # print("task type: ", self.task_type, ep_metrics)
        if self.task_type == "MultiAgentObjectNav-v1":
            ep_metrics["distance_to_goal"] = ep_metrics["distance_to_goal"][stopped_agents[0]]
            all_metrics[0] = ep_metrics
        elif self.task_type == "MultiAgentGoat-v1":
            for agent_id, task_idx in zip(stopped_agents, stopped_tasks):
                metrics = ep_metrics.copy()
                if not task_idx is None and (all_metrics.get(task_idx) is None or all_metrics[task_idx]["multiagent_goat_success"] == 0):
                    metrics["multiagent_goat_success"] = metrics["multiagent_goat_success"][task_idx]
                    metrics["multiagent_goat_distance_to_sub-goal"] = metrics["multiagent_goat_distance_to_sub-goal"][task_idx][agent_id]
                    all_metrics[task_idx] = metrics
                    for task_idx in all_metrics.keys():
                        all_metrics[task_idx]["multiagent_goat_spl"] = metrics["multiagent_goat_spl"]
                    logger.info("-------------------------")
                    logger.info(
                        f"{self.scene_id}_{self.episode_id}_{task_idx} {metrics}"
                    )
                    logger.info("-------------------------")
        else:
            raise Exception(f"{self.task_type} Not implemented")

    def init_subepisode_metrics(self):
        all_subtask_metrics = {}
        if self.task_type == "MultiAgentGoat-v1":
            for i in range(len(self.episode.tasks)):
                all_subtask_metrics[i] = {"multiagent_goat_success": 0, "multiagent_goat_distance_to_sub-goal": 0.0, "multiagent_goat_spl": 0.0} #!myTODO:  dist to sub goal is wrong
        return all_subtask_metrics

    def reset_vis_dir(self):
        if self.visualization_level > 0:
            self.visualizer.set_vis_dir(
                f"{self.scene_id}_{self.episode_id}"
            )
