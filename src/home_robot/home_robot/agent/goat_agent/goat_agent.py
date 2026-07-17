# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import time
import torch
import psutil
import logging
import numpy as np
from dataclasses import dataclass
import home_robot.utils.pose as pu
from .goat_matching import GoatMatching
from scipy.ndimage import binary_erosion
from typing import Any, Dict, List, Tuple
from home_robot.utils.logger import get_logger
from home_robot.utils.camera_params import CameraParams
from home_robot.core.abstract_agent import Agent
from home_robot.utils.visualization import visualize_semantic_with_labels, visualize_depth_filter
from home_robot.mapping.semantic.categorical_2d_semantic_map_state import (
    Categorical2DSemanticMapState,
)
from home_robot.mapping.semantic.categorical_2d_semantic_map_module import (
    Categorical2DSemanticMapModule,
)
from home_robot.core.interfaces import DiscreteNavigationAction, Observations
from home_robot.mapping.semantic.instance_tracking_modules import InstanceMemory
from home_robot.navigation_planner.fixed_discrete_planner import DiscretePlanner
from home_robot.utils.constants import MAX_DEPTH_REPLACEMENT_VALUE

# ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# ARUCO_PARAMS = cv2.aruco.DetectorParameters()
# ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
# MARKER_EXPAND = {
#     0: (2.5, 2.5, 2.8, 2),
#     1: (2.5, 2.5, 2, 2.5),
#     2: (2.5, 2.5, 2, 2.8),
#     3: (2.5, 2.5, 2, 2.5),
# }

MARKER_EXPAND = (2.5, 2.5, 1.5, 1.5)  # up, down, left, right

@dataclass
class Task:
    type: str
    goal_semantic_id: int
    goal_image: np.ndarray = None
    goal_image_processed: np.ndarray = None
    goal_image_keypoints: np.ndarray = None
    goal_description: str = None

class AgentLogger(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return f"[AGENT] {msg}", kwargs


class GoatAgent(Agent):
    """Simple object nav agent based on a 2D semantic map
    Works for tasks: Objectnav, Goat-v1, MultiagentGoat-V1

    """

    def __init__(
        self, config, vocabulary, camera_params=None, agent_id=None, device_id: int = 0
    ):
        self.real_world = config.REAL_WORLD
        
        self.is_multiagent = not agent_id is not None
        self.agent_id = agent_id
        self._log = get_logger(agent_id=agent_id)
        self.log = AgentLogger(self._log, None)
        self.max_steps = config.AGENT.max_steps
        self.task_type = self._get_task_type(config)
        self.seq_goals = bool(config.SEQ)
        self.use_yolo = bool(config.USE_YOLO)

        self.record_instance_ids = (
            True  # Code doesn't work with False, I can fix this later
        )
        self.visualization_level = config.VISUALIZATION_LEVEL

        self.instance_memory = None
        if self.record_instance_ids:
            self.instance_memory = InstanceMemory(
                config=config,
                mask_cropped_instances=False,
                padding_cropped_instances=200,
                log=self._log,
            )

        self.matching = GoatMatching(
            device=0,  # config.simulator_gpu_id
            config=config.AGENT.SUPERGLUE,
            default_vis_dir=f"{config.DUMP_LOCATION}/images/{config.EXP_NAME}",
            print_images=self.visualization_level > 1,
            instance_memory=self.instance_memory,
            log=self._log,
            cat_match_threshold=config.AGENT.cat_match_threshold,
        )
        if config.NO_GPU:
            self.device = torch.device("cpu")
        else:
            self.device_id = device_id
            self.device = torch.device(f"cuda:{self.device_id}")

        self.num_sem_categories = len(vocabulary) + 1 # For other robots.
        if camera_params is None:
            camera_params = self._get_camera_params(config)
        self.camera_params = camera_params

        agent_cell_radius = int(
            np.ceil(config.AGENT.radius * 100.0 / config.AGENT.SEMANTIC_MAP.map_resolution)
        )

        self.semantic_map_module = Categorical2DSemanticMapModule(
            device=self.device,
            camera_params=camera_params,
            num_sem_categories=self.num_sem_categories,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            vision_range=config.AGENT.SEMANTIC_MAP.vision_range,
            been_close_to_radius=config.AGENT.SEMANTIC_MAP.been_close_to_radius,
            target_blacklisting_radius=config.AGENT.SEMANTIC_MAP.target_blacklisting_radius,
            global_downscaling=config.AGENT.SEMANTIC_MAP.global_downscaling,
            du_scale=config.AGENT.SEMANTIC_MAP.du_scale,
            cat_pred_threshold=config.AGENT.SEMANTIC_MAP.cat_pred_threshold,
            exp_pred_threshold=config.AGENT.SEMANTIC_MAP.exp_pred_threshold,
            map_pred_threshold=config.AGENT.SEMANTIC_MAP.map_pred_threshold,
            min_obs_height_cm=config.AGENT.SEMANTIC_MAP.min_obs_height_cm,
            record_instance_ids=self.record_instance_ids,
            instance_memory=self.instance_memory,
            max_instances=getattr(config.AGENT.SEMANTIC_MAP, "max_instances", 0),
            exploration_type=config.AGENT.exploration_type,
            agent_cell_radius=agent_cell_radius,
            print_images=self.visualization_level > 2,
            log=self._log,
            real_world=self.real_world,
            # mask_stairs=self.real_world,
            start_obs_dilation=config.AGENT.PLANNER.min_obs_dilation_selem_radius,
        )
        self.inst_goal_id = None

        self.semantic_map = Categorical2DSemanticMapState(
            device=self.device,
            num_sem_categories=self.num_sem_categories,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            global_downscaling=config.AGENT.SEMANTIC_MAP.global_downscaling,
            record_instance_ids=self.record_instance_ids,
            instance_memory=self.instance_memory,
            visualization_level=self.visualization_level,
            # close_frontier_radius=10.0,  #! myTODO: Hardcoded 5
            agent_id=agent_id,
        )
        self.max_subtasks_per_episode = config.ENVIRONMENT.max_subtasks_per_episode


        turn_angle = config.ENVIRONMENT.turn_angle if self.real_world else config.habitat.simulator.turn_angle
        if config.AGENT.panorama_start:
            panorama_start_steps = int(360 / turn_angle)
        else:
            panorama_start_steps = 0

        self.planner = DiscretePlanner(
            turn_angle=turn_angle,
            collision_threshold=config.AGENT.PLANNER.collision_threshold,
            step_size=config.AGENT.PLANNER.step_size,
            obs_dilation_selem_radius=config.AGENT.PLANNER.obs_dilation_selem_radius,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            visualization_level=self.visualization_level,
            dump_location=config.DUMP_LOCATION,
            exp_name=config.EXP_NAME,
            min_obs_dilation_selem_radius=config.AGENT.PLANNER.min_obs_dilation_selem_radius,
            map_downsample_factor=config.AGENT.PLANNER.map_downsample_factor,
            map_update_frequency=config.AGENT.PLANNER.map_update_frequency,
            discrete_actions=config.AGENT.PLANNER.discrete_actions,
            panorama_start_steps=panorama_start_steps,
            instance_memory=self.instance_memory,
            goal_filtering=config.AGENT.SEMANTIC_MAP.goal_filtering,
            semantic_map=self.semantic_map,
            frontier_metric=config.AGENT.frontier_metric,
            agent_id=self.agent_id,
            ground_truth_semantics=config.GROUND_TRUTH_SEMANTICS,
            task_type=self.task_type,
            stop_distance=config.AGENT.PLANNER.stop_distance,
            real_world=self.real_world,
        )

        self.subtask_timesteps = None
        self.total_timesteps = None 
        self.last_pose = None
        self.reject_visited_targets = False
        self.blacklist_target = False

        self.current_task_idx = 0

        self.communication_radius = config.AGENT.COMMUNICATION.radius
        self.ground_truth_semantics = config.GROUND_TRUTH_SEMANTICS
        if not self.ground_truth_semantics:
            self._setup_perception(config, vocabulary)
        self.match_memory = True
        self.search_found_goal_freq = config.AGENT.search_found_goal_freq

        self.neighbor_color = (np.array([60, 70, 70]), np.array([85, 230, 225]))
        self.neighbor_width_range = (0.05, 0.50)   # meters
        self.neighbor_height_range = (0.10, 0.20)  # meters
        self.robot_max_depth_spread = 0.30      # meters


    def _get_task_type(self, config) -> str:
        if self.real_world:
            return 'Goat-v1'
        else:
            return config.habitat.task.type

    def _get_camera_params(self, config) -> CameraParams:
        assert not self.real_world
        sensor = config.habitat.simulator.agents.agent0.sim_sensors.depth_sensor
        return CameraParams.from_habitat_config(
            sensor_cfg=sensor,
            min_depth=sensor.min_depth,
            max_depth=sensor.max_depth,
        )



    def _setup_perception(self, config, vocabulary):
        if "Goat-v1" in self.task_type:
            from home_robot.perception.detection.detic.detic_perception import (
                DeticPerception,
            )
            self.segmentation = DeticPerception(
                vocabulary="custom",
                custom_vocabulary=","
                + ",".join(vocabulary),
                sem_gpu_id=(-1 if config.NO_GPU else 0),
            )
            if self.use_yolo:
                from ultralytics import YOLOWorld
                self.yolo = YOLOWorld('yolov8s-worldv2.pt')  # or 'yolov8m-worldv2.pt' for better accuracy
                self.yolo.set_classes(vocabulary)
        else:
            from home_robot.perception.detection.maskrcnn.maskrcnn_perception import (
                MaskRCNNPerception
            )
            # MaskRCNN IDs are the same as our semantic category mappoing vocab.
            self.segmentation = MaskRCNNPerception(
                sem_pred_prob_thr=0.8,
                sem_gpu_id=(-1 if config.NO_GPU else 0),
            )
            if self.use_yolo:
                from ultralytics import YOLOv10
                self.yolo = YOLOv10.from_pretrained('jameslahm/yolov10n', verbose=False)

    def get_subtask_timestep(self) -> int:
        """
        Only has meaning when we have sequential goals and a single agent. Otherwise, it is the total timestep.
        """
        if self.seq_goals:
            return self.subtask_timesteps[self.current_task_idx]
        else:
            return self.total_timesteps

    def reset(self, scene_id, episode_id):
        """Initialize agent state. Reset is at the beginning of a new episode (not each task)."""
        self.total_timesteps = 0
        if self.seq_goals:
            self.subtask_timesteps = [0] * self.max_subtasks_per_episode
        self.last_pose = np.zeros(3)
        self.semantic_map.init_map_and_pose()
        if self.instance_memory is not None:
            self.instance_memory.reset()
        self.reject_visited_targets = False
        self.blacklist_target = False
        self.navigate_to_best = False

        self.current_task_idx = -1
        self.reset_for_next_task()
        self.planner.reset()
        self.matching.step = 0
        self.inst_goal_id = None

        self.stuck_counter = 0
        self.reset_vis_dir(scene_id, episode_id, 0)
        # self.history_scores = []

    def handle_stop(self, action):
        self.reset_for_next_task()

    def reset_for_next_task(self) -> None:
        """Reset for a new task, and not a new episode."""
        self.inst_goal_id = None
        if self.seq_goals:
            self.current_task_idx += 1
        self.navigate_to_best = False
        self.match_memory = True
        self.planner.reset_for_next_task()

    def _update_steps(self):
        self.total_timesteps += 1
        if self.seq_goals:
            self.subtask_timesteps[self.current_task_idx] += 1
        self.matching.step = self.total_timesteps
        self.semantic_map_module.timestep = self.get_subtask_timestep()
        self.planner.total_timesteps = self.total_timesteps
        self.planner.timestep = self.get_subtask_timestep()
        self.log.info(
            f"---------------- Updating state - step:{self.get_subtask_timestep()} ----------------"
        )
        self.log.debug(
            f"Available RAM: {psutil.virtual_memory().available / 1e9:.2f} GB"
        )

    def _safety_stop(self):
        self.too_close = False
        depth = self._curr_obs.depth
        h, w = depth.shape[:2]
        width = 50
        height = 400
        cy, cx = h // 2, w // 2
        center_pixels = depth[cy - width//2 : cy + width//2,
                        cx - height//2 : cx + height//2]
        close_pixels = np.sum(center_pixels < 0.6)
        self.too_close = close_pixels >= 50 and self.real_world
        if self.too_close:
            print("Too close, center pixels: ", close_pixels, np.min(center_pixels), np.max(center_pixels))

    def update_state(self, obs):
        self._curr_obs = obs
        self._update_steps()
        self._update_pose()
        self.tasks = self._preprocess_tasks(obs.task_observations["tasks"])
        self._safety_stop()


    def _preprocess_tasks(self, tasks_obs) -> List[Task]:
        tasks = []
        for task_obs in tasks_obs:
            task = Task(
                type=task_obs["type"],
                goal_semantic_id=task_obs["semantic_id"]
            )
            if task.type == "imagenav":
                task.goal_image = task_obs["image"]
                task.goal_image_processed, task.goal_image_keypoints = (
                    self.matching.get_goal_image_keypoints(task.goal_image)
                )
            elif task.type == "languagenav":
                task.goal_description = task_obs["description"]
            tasks.append(task)
        return tasks

    def act(self, **kwargs) -> Tuple[DiscreteNavigationAction, Dict[str, Any], bool]:
        """Act end-to-end."""
        action, vis_inputs = self._get_best_action(**kwargs)
        action = self._process_action(action)
        info = self._get_vis_info(vis_inputs, action)

        stuck = False
        if self.get_subtask_timestep() >= self.max_steps or self.stuck_counter > 30:
            self.log.warning(
                "Reached max number of steps for subgoal, or stuck somewhere, calling STOP"
            )
            stuck = True

        if action["action"] == DiscreteNavigationAction.STOP:
            self.handle_stop(action)
        
        self.log.debug(f"Action: {str(action['action'])}")

        return action, info, stuck

    def update_maps(self):
        obs_preprocessed, instance_scores, category_scores = self._preprocess_obs(self._curr_obs)
        # * before module call obs.shape is [380+3+1+num_instances]
        # Update map with observations and generate map features
        self.semantic_map_module(
            obs_preprocessed,
            self.semantic_map,
            instance_scores,
            category_scores
        )
        # if self.total_timesteps % 10 == 0:
        #     torch.save(self.semantic_map.global_map, os.path.join(self.planner.vis_dir, f'map_{self.total_timesteps}.pt'))
        self._search_for_goal()

    def _get_vis_info(self, vis_inputs, action):
        if self.visualization_level < 1:
            return None

        is_local = vis_inputs.get("is_local", True)
        obs = self._curr_obs
        info = {
            "agent_id": self.agent_id,
            "rgb_frame": self.frame_yolo.copy() if self.use_yolo else obs.rgb[:, :, ::-1],
            "depth_frame": obs.depth.copy(),
            "semantic_frame": obs.semantic if obs.task_observations.get("semantic_frame") is None else obs.task_observations["semantic_frame"],
            "top_down_map": obs.task_observations.get("top_down_map"),
            "is_collision": False,  #!myTODO
            "inst_goal_id": self.inst_goal_id,
            "timestep": self.get_subtask_timestep(),
            "explored_map": self.semantic_map.get_explored_map(is_local).copy(),
            "semantic_map_1D": self.semantic_map.get_semantic_map_1D(is_local),
            # "frontier_map": self.semantic_map.get_frontier_map(is_local),
            "been_close_map": self.semantic_map.get_been_close_map(is_local).copy(),
            "visited_map": self.semantic_map.get_visited_map(is_local).copy(),
            "robot_loc": self.semantic_map.get_loc(is_local),
            "robot_orientation": self.semantic_map.global_pose.cpu()[2],
            # "instance_memory": self.instance_memory,
            **vis_inputs,
        }
        if "obstacle_map" not in info:
            info["obstacle_map"] = self.semantic_map.get_obstacle_map(is_local).copy()
        self._get_task_info(obs, action, info)
        return info

    def _get_task_info(self, obs, action, info):
        current_task = obs.task_observations["tasks"][self.current_task_idx]
        info["task_type"] = current_task["type"]
        goal_text_desc = {x: y for x, y in current_task.items() if x != "image"}
        info["caption"] = str(goal_text_desc)
        if current_task["type"] == "imagenav":
            info["goal_image"] = current_task["image"]
        else:
            info["third_person_image"] = obs.third_person_image

        if isinstance(action["action"], DiscreteNavigationAction):
            info["caption"] += f" | Action: {str(action['action']).split('.')[-1]}"
        else:
            info["caption"] += f" | Action: {str(action['action'])}"

    def _update_pose(self):
        obs = self._curr_obs
        curr_pose = np.array([obs.gps[0], obs.gps[1], obs.compass[0]])
        pose_delta = torch.tensor(
            pu.get_rel_pose_change(curr_pose, self.last_pose), device=self.device
        )
        self.last_pose = curr_pose

        new_global = pu.get_new_pose(
            self.semantic_map.global_pose.clone(),
            pose_delta
        )
        self.semantic_map.global_pose = new_global
        self.semantic_map.local_pose = new_global - self.semantic_map.origins

        if torch.norm(pose_delta[:2]).item() < 0.05:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

    def _preprocess_obs(self, obs: Observations):
        """Take a home-robot observation, preprocess it to put it into the correct format for the
        semantic map."""
        def filter_instances_by_depth(
            obs,
            erosion_iters=1,
            mad_thresh=3.0,
        ):
            """
            Returns a cleaned instance_map where depth outliers are removed.
            """
            instance_frame = obs.task_observations["instance_frame"]
            depth_frame = obs.depth

            mask_union = np.zeros_like(instance_frame, dtype=bool)

            instance_ids = np.unique(instance_frame)
            instance_ids = instance_ids[instance_ids > 0]
            for inst_id in instance_ids:
                inst_mask = instance_frame == inst_id
                inst_mask = binary_erosion(inst_mask, iterations=erosion_iters)
                if inst_mask.sum() == 0:
                    inst_mask = instance_frame == inst_id

                instance_depth = depth_frame[inst_mask]

                median = np.median(instance_depth)
                mad = np.median(np.abs(instance_depth - median))

                if mad == 0:
                    depth_mask = np.abs(depth_frame - median) < 1e-3
                else:
                    depth_mask = np.abs(depth_frame - median) <= mad_thresh * mad


                final_inst_mask = inst_mask & depth_mask
                mask_union = mask_union | final_inst_mask

            obs.semantic = obs.semantic * mask_union
            obs.task_observations["instance_frame"] = obs.task_observations["instance_frame"] * mask_union


        def mask_other_robots_color(
            rgb,
            depth,
            depth_margin_behind: float = 0.50,
            depth_margin_front: float = 0.25
        ):
            robot_mask = np.zeros(depth.shape[:2], dtype=bool)
            h, w = depth.shape[:2]
            fx = fy = self.semantic_map_module.camera_matrix.f
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            np.save(
                os.path.join(self.planner.vis_dir, f"{self.total_timesteps}_hsv.npy"),
                hsv
            )

            hsv_lower, hsv_upper = self.neighbor_color
            color_mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 150:
                    continue
                x, y, bw, bh = cv2.boundingRect(contour)
                contour_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(contour_mask, [contour], -1, 255, -1)
                region_depths = depth[contour_mask > 0]
                valid = region_depths[region_depths < self.camera_params.max_depth]
                if len(valid) < 150:
                    self.log.debug(f"Contour rejected: valid_depth={len(valid)} < 150")
                    continue
                median_depth = np.median(valid)
                spread = np.percentile(valid, 90) - np.percentile(valid, 10)
                if spread > self.robot_max_depth_spread:
                    self.log.debug(f"Contour rejected: spread={spread:.2f} > {self.robot_max_depth_spread}")
                    continue
                real_w = (bw / fx) * median_depth
                real_h = (bh / fy) * median_depth
                if not (self.neighbor_width_range[0] < real_w < self.neighbor_width_range[1]):
                    self.log.debug(f"Contour rejected: width={real_w:.2f}m not in {self.neighbor_width_range}")
                    continue
                if not (self.neighbor_height_range[0] < real_h < self.neighbor_height_range[1]):
                    self.log.debug(f"Contour rejected: height={real_h:.2f}m not in {self.neighbor_height_range}")
                    continue
                if y + bh < h * 0.35:
                    self.log.debug(f"Contour rejected: bottom_y={y+bh} too high in image")
                    continue
                self.log.debug(
                    f"Robot detected: bbox=({x},{y},{bw},{bh}), "
                    f"size={real_w:.2f}x{real_h:.2f}m, depth={median_depth:.2f}m"
                )

                cx, cy = x + bw / 2.0, y + bh / 2.0
                up, down, left, right = MARKER_EXPAND
                x_min = max(0, int(cx - left * bw))
                x_max = min(w, int(cx + right * bw))
                y_min = max(0, int(cy - up * bh))
                y_max = min(h, int(cy + down * bh))

                roi_depth = depth[y_min:y_max, x_min:x_max]
                depth_match = (
                    (roi_depth > median_depth - depth_margin_behind) &
                    (roi_depth < median_depth + depth_margin_front) &
                    (roi_depth > 0)
                )
                robot_mask[y_min:y_max, x_min:x_max] |= depth_match

            self.log.debug(f"Robot mask: {robot_mask.sum()} pixels")
            if self.visualization_level > 0:
                vis = rgb.copy()
                #  color mask blue overlay
                vis[color_mask > 0] = (vis[color_mask > 0] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)
                # final robot mask red overlay
                vis[robot_mask] = (vis[robot_mask] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(np.uint8)
                # contour bounding boxes: passed all checks (green) or failed (gray)
                for contour in contours:
                    x, y, bw, bh = cv2.boundingRect(contour)
                    cv2.rectangle(vis, (x, y), (x + bw, y + bh), (150, 150, 150), 1)
                if robot_mask.any():
                    # Draw the expanded region
                    cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.imwrite(
                    os.path.join(self.planner.vis_dir, f"{self.total_timesteps}_robot_detect.png"),
                    cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                )

            return robot_mask

        def mask_other_robots(
            rgb: np.ndarray,
            depth: np.ndarray,
            depth_margin_behind: float = 0.80,
            depth_margin_front: float = 0.40,
        ) -> np.ndarray:
            """
            Detect ArUco markers on other robots and return a binary mask
            of pixels belonging to other robots.

            Returns:
                robot_mask: (H, W) bool mask, True where other robot is detected
            """
            robot_mask = np.zeros(depth.shape[:2], dtype=bool)
            h, w = depth.shape[:2]

            corners, ids, _ = ARUCO_DETECTOR.detectMarkers(rgb)

            if ids is None:
                return robot_mask

            for i, marker_id in enumerate(ids.flatten()):
                if marker_id not in MARKER_EXPAND:
                    continue

                c = corners[i][0]
                side_lengths = [
                    np.linalg.norm(c[j] - c[(j + 1) % 4]) for j in range(4)
                ]
                marker_size_px = max(side_lengths)

                cx, cy = np.mean(c[:, 0]), np.mean(c[:, 1])
                marker_region = depth[
                    max(0, int(cy - marker_size_px // 4)):min(h, int(cy + marker_size_px // 4)),
                    max(0, int(cx - marker_size_px // 4)):min(w, int(cx + marker_size_px // 4)),
                ]
                valid_marker_depths = marker_region[
                    (marker_region > 0) & (marker_region < MAX_DEPTH_REPLACEMENT_VALUE - 10)
                ]

                if len(valid_marker_depths) == 0:
                    continue
                marker_depth = np.median(valid_marker_depths)

                up, down, left, right = MARKER_EXPAND[marker_id]
                x_min = max(0, int(cx - left * marker_size_px))
                x_max = min(w, int(cx + right * marker_size_px))
                y_min = max(0, int(cy - up * marker_size_px))
                y_max = min(h, int(cy + down * marker_size_px))

                roi_depth = depth[y_min:y_max, x_min:x_max]
                depth_match = (
                    (roi_depth > marker_depth - depth_margin_behind) &
                    (roi_depth < marker_depth + depth_margin_front) &
                    (roi_depth > 0)
                )
                robot_mask[y_min:y_max, x_min:x_max] |= depth_match

            return robot_mask

        category_scores = None
        if not self.ground_truth_semantics:
            if "Goat-v1" in self.task_type :
                obs = self.segmentation.predict(obs, draw_instance_predictions=True)
            else: # HM3D Objnav Challenge
                obs = self.segmentation.predict(obs)
                # print(f"obs.cls id: ", obs.task_observations["instance_classes"])
                # print(f"obs.scores: ", obs.task_observations["instance_scores"])

            from home_robot.perception.constants import coco_categories_mapping,  coco_map_color_palette
            # if self.visualization_level > 2:
            #     self.visualize_semantic_with_labels(
            #         semantic_array=obs.semantic + 10,
            #         palette=coco_map_color_palette,
            #         postfix="_with_rednet_sem"
            #     )
            #     self.visualize_semantic_with_labels(
            #         # semantic_array=obs.task_observations["instance_frame"] + 10,
            #         semantic_array=obs.task_observations["rednet_semantic_frame"] + 10,
            #         palette=coco_map_color_palette,
            #         postfix="_rednet_sem_frame"
            #     )
            filter_instances_by_depth(obs)
            # obs = self.segmentation_red.predict(obs, draw_instance_predictions=True)
            # if self.visualization_level > 2:
            #     self.visualize_semantic_with_labels(
            #         semantic_array=obs.semantic + 10,
            #         palette=self.semantic_category_mapping.map_color_palette,
            #         postfix="_with_rednet_sem"
            #     )
            #     self.visualize_semantic_with_labels(
            #         semantic_array=obs.task_observations["instance_frame"] + 10,
            #         palette=self.semantic_category_mapping.map_color_palette,
            #         postfix="_with_rednet_instance"
            #     )


            if self.use_yolo:
                yolo_output = self.yolo(source=obs.rgb, conf=0.2, verbose=False)
                category_scores = {i: [] for i in range(self.num_sem_categories + 1)}  # +1 for 1-based indexing 
                for box in yolo_output[0].boxes:
                    cls = int(box.cls[0])  # 0 to num_categories-1
                    if "Goat-v1" in self.task_type :
                        category_scores[cls + 1].append(box.conf[0].item())  # Store at 1 to num_categories
                    else:
                        if cls in coco_categories_mapping:
                            category_scores[coco_categories_mapping[cls]+1].append(box.conf[0].item())
                self.frame_yolo = cv2.cvtColor(yolo_output[0].plot(), cv2.COLOR_BGR2RGB)

                for i in range(self.num_sem_categories + 1):
                    if len(category_scores[i]) > 0:
                        category_scores[i] = np.max(category_scores[i])
                    else:
                        category_scores[i] = 0

        rgb = torch.from_numpy(obs.rgb).to(self.device)
        # other_robots_mask = mask_other_robots(obs.rgb, obs.depth)
        other_robots_mask = mask_other_robots_color(obs.rgb, obs.depth)
        if self.visualization_level > 2:
            visualize_depth_filter(
                rgb=obs.rgb,
                robot_mask=other_robots_mask,
                depth=obs.depth,
                save_dir=self.planner.vis_dir,
                timestep=self.total_timesteps,
            )
        depth = (
            torch.from_numpy(obs.depth).unsqueeze(-1).to(self.device) * 100.0
        )  # m to cm

        obs.semantic[other_robots_mask] = self.num_sem_categories # Add other robots to semantics
        semantic = torch.eye(self.num_sem_categories + 1, device=self.device)[
            torch.from_numpy(obs.semantic).to(self.device)
        ][
            :, :, 1:
        ]  # one-hot encode and remove background class

        path = os.path.join(self.planner.vis_dir, f"{self.total_timesteps}_sem.png")
        if self.visualization_level > 2:
            success = cv2.imwrite(path, obs.task_observations["semantic_frame"])
        ts_path = path.rsplit(".", 1)[0] + ".ts"
        with open(ts_path, "w") as f:
            f.write(str(time.time()))
        obs_preprocessed = torch.cat([rgb, depth, semantic], dim=-1)

        if self.record_instance_ids:
            # * Why using instance_frame which are the raw semantics? To differentiate between objects with diff raw semantics but same category in our classes.
            instance_frame = obs.task_observations["instance_frame"]
            unique_ids, new_instance_frame = np.unique(instance_frame, return_inverse=True)
            new_instance_frame = new_instance_frame.reshape(instance_frame.shape)
            new_instance_frame = torch.from_numpy(new_instance_frame).to(self.device)

            # One-hot encode
            instance_frame_onehot = torch.eye(len(unique_ids), device=self.device)[
                new_instance_frame
            ]
            if unique_ids[0] == 0:
                instance_frame_onehot = instance_frame_onehot[:,:, 1:] # First layer is background
            else:
                assert 0 not in unique_ids, "Expected no background (0) in unique_ids"
            

            # For ground truth, scores are all 1.0
            inst_scores = np.concatenate(([0],obs.task_observations["instance_scores"]))[unique_ids][1:]

            obs_preprocessed = torch.cat(
                [obs_preprocessed, instance_frame_onehot], dim=-1
            )
        obs_preprocessed = obs_preprocessed.permute(2, 0, 1)


        assert obs.camera_pose is None


        # * preprocessed obs shape is (1, 3+1+num_sem_classes+num_instances, H, W)
        return obs_preprocessed, inst_scores, category_scores

    def _get_best_action(self, **kwargs):
        task = self.tasks[self.current_task_idx]
        action, vis_input = self.planner.plan(
            self.inst_goal_id,
            task.goal_semantic_id,
        )

        if action is not None:
            return action, vis_input

        if self.inst_goal_id is not None:
            self.log.info("Couldn't navigate to goal, stopping")
            return DiscreteNavigationAction.STOP, {}

        # Note that here, inst_goal_id is None, otherwise we would have stopped
        self.log.info("No reachable frontiers, forcing a match against memory")
        self.match_memory = True
        self._search_for_goal(select_best=True)

        if self.inst_goal_id is None:
            self.log.info("No match found in memory, stopping.")
            return DiscreteNavigationAction.STOP, {}

        return self._get_best_action(**kwargs)

    @torch.no_grad()
    def _search_for_goal(self, select_best=False):
        """
        Searches for goal in current observation, and also in memory if it is the first timestep of the task.
        Set value for self.inst_goal_id.
        select_best forces matching to the best object, even if it doesn't pass matching threshold
        """
        #! myTODO: Put %10 here, so that we again check with obs every 10 steps, so we might get better matches. Can be more intelligent.
        if not self.inst_goal_id is None and self.get_subtask_timestep() % self.search_found_goal_freq != 0:
            self.log.info(f"Already found instance goal, not searching anymore.")
        else:
            # Match a goal against every instance in memory the moment the subtask starts.
            # We also search in memory when env is fully explored, but that is handled somewhere else.
            inst_goal_id, _ = self.matching.search_for_goal(
                self.tasks[self.current_task_idx],
                self.match_memory,
                self.semantic_map.global_pose,
                score_thresh=0 if select_best else None
            )
            if inst_goal_id is not None:
                # Else, we should not replace, maybe we have previously seen a goal and moving toward it.
                self.inst_goal_id = inst_goal_id

        self.match_memory = False

    def reset_vis_dir(self, scene_id, episode_id, current_task_idx=None):
        if self.seq_goals:
            dir_name = f"{scene_id}_{episode_id}_{current_task_idx}"
        else:
            dir_name = f"{scene_id}_{episode_id}"

        if self.agent_id is not None:
            dir_name = os.path.join(dir_name, f"agent_{self.agent_id}")

        self.planner.set_vis_dir(dir_name)
        self.matching.set_vis_dir(dir_name)
        self.semantic_map.vis_dir = self.planner.vis_dir
        self.semantic_map_module.vis_dir = self.planner.vis_dir

    def _process_action(self, action):
        if self.too_close == True:
            if action == DiscreteNavigationAction.MOVE_FORWARD:
                print("Too close, not executing forward")
                action = None
        return {
            "action": action,
            "action_args": {
                "agent_id": 0 if self.agent_id is None else self.agent_id,
                "task_idx": self.current_task_idx,
            },
        }

    def visualize_semantic_with_labels(
        self,
        semantic_array: np.ndarray,
        palette: list,
        postfix: str = "",
    ):

        if self.visualization_level > 1:
            save_path = os.path.join(
                self.planner.vis_dir, f"{self.get_subtask_timestep()}_0.sem_input{postfix}.png"
            )
            visualize_semantic_with_labels(semantic_array, palette, save_path)