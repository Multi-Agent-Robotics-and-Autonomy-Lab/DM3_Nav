# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import cv2
import math
import skfmm
import logging
import numpy as np
import skimage.morphology
from typing import List, Tuple
from scipy.ndimage import label
from bresenham import bresenham
from skimage.draw import polygon
import home_robot.utils.pose as pu
from .fmm_planner import FMMPlanner
from scipy.spatial import ConvexHull
from home_robot.core.interfaces import (
    ContinuousNavigationAction,
    DiscreteNavigationAction,
)
from home_robot.utils.visualization import (
    visualize_map,
    visualize_distance_frontiers,
    visualize_semantic_frontiers,
)
from home_robot.utils.logger import get_logger
from scipy.ndimage import distance_transform_edt
from home_robot.mapping.semantic.constants import MapConstants as MC
from home_robot.mapping.semantic.categorical_2d_semantic_map_state import (
    Categorical2DSemanticMapState,
)
from home_robot.mapping.semantic.instance_tracking_modules import InstanceMemory


script_dir = os.path.dirname(
    os.path.abspath(__file__)
)  # Directory of the current script
CO_LOCATION_WEIGHTS = np.load(os.path.join(script_dir, f"co_location_matrix.npy"))


def add_boundary(mat: np.ndarray, value=1) -> np.ndarray:
    h, w = mat.shape
    new_mat = np.zeros((h + 2, w + 2)) + value
    new_mat[1 : h + 1, 1 : w + 1] = mat
    return new_mat


def remove_boundary(mat: np.ndarray, value=1) -> np.ndarray:
    return mat[value:-value, value:-value]


class PlannerLogger(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        # modify the message however you want
        return f"[PLANNER] {msg}", kwargs

class DiscretePlanner:
    """
    This class translates planner inputs into a discrete low-level action
    using an FMM planner.

    This is a wrapper used to navigate to a particular object/goal location.
    """

    def __init__(
        self,
        turn_angle: float,
        collision_threshold: float,
        step_size: int,
        obs_dilation_selem_radius: int,
        map_size_cm: int,
        map_resolution: int,
        dump_location: str,
        exp_name: str,
        visualization_level: int,
        stop_distance: float,
        min_obs_dilation_selem_radius: int = 1,
        map_downsample_factor: float = 1.0,
        map_update_frequency: int = 1,
        discrete_actions: bool = True,
        continuous_angle_tolerance: float = 30.0,
        panorama_start_steps: int = 0,
        semantic_map: Categorical2DSemanticMapState = None,
        instance_memory: InstanceMemory = None,
        goal_filtering=False,
        frontier_metric: str = "distance",
        agent_id=None,
        ground_truth_semantics=None,
        task_type=None,
        real_world=False,
    ):
        """
        Similar to old DiscretePlanner, but with changes to:
        - How to go to past pose.
        - How to replan if planning failed.
        Arguments:
            turn_angle (float): agent turn angle (in degrees)
            collision_threshold (float): forward move distance under which we
             consider there's a collision (in meters)
            obs_dilation_selem_radius: radius (in cells) of obstacle dilation
             structuring element
            obs_dilation_selem_radius: radius (in cells) of goal dilation
             structuring element
            map_size_cm: global map size (in centimeters)
            map_resolution: size of map bins (in centimeters)
        """
        self.discrete_actions = discrete_actions
        self.default_vis_dir = f"{dump_location}/images/{exp_name}"
        os.makedirs(self.default_vis_dir, exist_ok=True)

        self.map_size_cm = map_size_cm
        self.map_resolution = map_resolution
        self.map_shape = (
            self.map_size_cm // self.map_resolution,
            self.map_size_cm // self.map_resolution,
        )
        self.turn_angle = turn_angle
        self.collision_threshold = collision_threshold
        self.step_size = step_size
        self.start_obs_dilation_selem_radius = obs_dilation_selem_radius
        self.min_obs_dilation_selem_radius = min_obs_dilation_selem_radius
        #! myTODO: I think the unit of goal tolerance is in pixels, so I should do some conversions here. Right now this is hardcoded.
        self.continuous_angle_tolerance = continuous_angle_tolerance

        self.map_downsample_factor = map_downsample_factor
        self.map_update_frequency = map_update_frequency
        self.panorama_start_steps = panorama_start_steps

        self.semantic_map = semantic_map
        self.instance_memory: InstanceMemory = instance_memory
        self.goal_filtering = goal_filtering
        self.frontier_metric = frontier_metric

        self.agent_id = agent_id
        self.log = PlannerLogger(get_logger(agent_id=agent_id), None)
        self.prefix = ""
        self.visualization_level = visualization_level
        self.ground_truth_semantics = ground_truth_semantics
        self.stop_distance = int(stop_distance / self.map_resolution)
        self.prev_frontier = np.zeros(self.map_shape, dtype=np.uint8)
        self._last_neighbor_set = set()
        self.planner = FMMPlanner(
            self.log,
            step_size=self.step_size,
            print_images=self.visualization_level > 1,
            stop_distance=self.stop_distance,
            real_world=real_world,
        )
        self.real_world = real_world

    def reset(self):
        self.dd = None
        self.vis_dir = self.default_vis_dir
        self.collision_map = np.zeros(self.map_shape, dtype=bool)
        self.col_width = 1
        self.last_global_pose = None
        self.curr_global_pose = [
            self.map_size_cm / 100.0 / 2.0,
            self.map_size_cm / 100.0 / 2.0,
            0.0,
        ]
        self.last_action = None
        self.timestep = 0
        self.total_timesteps = 0
        self.curr_frontier_dilation = self.start_obs_dilation_selem_radius
        self.curr_instance_dilation = self.start_obs_dilation_selem_radius
        self.episode_panorama_start_steps = self.panorama_start_steps

        self.reset_for_next_task()
    
    def reset_for_next_task(self):
        self.moved_forward = False

    def set_vis_dir(self, dir_name):
        self.vis_dir = os.path.join(self.default_vis_dir, dir_name)
        self.planner.set_vis_dir(self.vis_dir)
        os.makedirs(self.vis_dir, exist_ok=True)

    def plan(
        self,
        inst_goal_id: int = None,
        goal_semantic_id: int = None,
        fallback_to_frontier=True,
        postfix="",
        neighbor_locs=None,
    ) -> Tuple[DiscreteNavigationAction, np.ndarray]:
        """Plan a low-level action.

        Args:
            obstacle_map: (M, M) binary local obstacle map prediction
            goal_map: (M, M) binary array denoting goal location
            sensor_pose: (7,) array denoting global pose (x, y, o)
             and local map boundaries planning window (gx1, gx2, gy1, gy2)

        Returns:
            action: low-level action
        """
        reachable = False
        stop = False
        try_best = fallback_to_frontier == False
        vis_input = {}
        inst_goal_found = not inst_goal_id is None

        # if inst_goal_found:
        #     self.episode_panorama_start_steps = 0

        if self.total_timesteps < self.episode_panorama_start_steps:
            return (
                DiscreteNavigationAction.TURN_RIGHT,
                vis_input,
            )  # If failed to plan, visualize locally.

        self.last_global_pose = self.curr_global_pose
        self.curr_global_pose = self.semantic_map.global_pose

        # Check collisions if we have just moved and are uncertain
        if self.last_action == DiscreteNavigationAction.MOVE_FORWARD:
            self._check_collision(postfix)

        self.log.info(f"---- Starting planning ---- ")
        self.log.info(f"> Global Location: {self.semantic_map.global_loc}")
        self.log.info(f"> Local Location: {self.semantic_map.local_loc}")
        self.log.info(f"> Instance goal found: {inst_goal_found}")


        if inst_goal_found:
            (
                reachable,
                stop,
                short_term_goal,
                is_local,
                vis_input,
            ) = self.plan_to_instance_goal(inst_goal_id, try_best, postfix)
        else:
            self.log.debug("No instance goal provided.")

        if not stop and not reachable:
            if fallback_to_frontier:
                (
                    reachable,
                    stop,
                    short_term_goal,
                    is_local,
                    vis_input,
                ) = self.plan_to_frontier_goal(
                    goal_semantic_id, postfix, neighbor_locs=neighbor_locs
                )

        if not (stop or reachable):
            action = None
        else:
            action = self.get_action(
                stop,
                short_term_goal,
                self.semantic_map.get_loc(is_local),
                None, # Not passing viewpoint orientation, because we are not going exactly to the viewpoint
            )

        self.last_action = action
        return action, vis_input

    def get_action(
        self, stop, short_term_goal, location, best_viewpoint=None
    ):
        """
        Gets discrete/continuous action given short-term goal. Agent orients to short term goal if stop=True
        """
        if self.moved_forward:
            self.log.debug("Already toward the goal, stopping.")
            return DiscreteNavigationAction.STOP

        viewpoint_orientation = None
        if best_viewpoint is not None:
            viewpoint_orientation = best_viewpoint.pose[2]
        angle_agent = pu.normalize_angle(self.curr_global_pose[2])

        if stop == False:
            # self.log.debug(f"angle: {self.curr_global_pose[2]}, {angle_agent}")
            stg_x, stg_y = short_term_goal
            relative_stg_x, relative_stg_y = stg_x - location[0], stg_y - location[1]
            # self.log.debug(f"rel x and y: {relative_stg_x} {relative_stg_y}")
            angle_st_goal = math.degrees(math.atan2(relative_stg_x, relative_stg_y))
            # self.log.debug(f"angle st gl: {angle_st_goal}")
            relative_angle_to_stg = pu.normalize_angle(angle_agent - angle_st_goal)
            # self.log.debug(f"relative angle to stg: {relative_angle_to_stg.item()}")

            if self.discrete_actions:
                if relative_angle_to_stg > (self.turn_angle / 2.0) + 2:
                    action = DiscreteNavigationAction.TURN_RIGHT
                elif relative_angle_to_stg < -(self.turn_angle / 2.0) - 2:
                    action = DiscreteNavigationAction.TURN_LEFT
                else:
                    action = DiscreteNavigationAction.MOVE_FORWARD
            else:
                if abs(relative_angle_to_stg) > 10.0:
                    action = ContinuousNavigationAction(np.array([0,0,-relative_angle_to_stg.cpu().item()]))
                else:
                    action = DiscreteNavigationAction.MOVE_FORWARD
        else:
            # Try to orient towards the goal object
            if viewpoint_orientation is None:
                # Compute angle to the final goal
                goal_x, goal_y = short_term_goal
                angle_goal = math.degrees(
                    math.atan2(goal_x - location[0], goal_y - location[1])
                )
                # Compute angle to the final goal
                relative_angle_to_closest_goal = pu.normalize_angle(
                    angle_agent - angle_goal
                )
            else:
                relative_angle_to_closest_goal = pu.normalize_angle(
                    angle_agent - viewpoint_orientation
                )
            self.log.debug(
                f">>> orienting towards the goal: {relative_angle_to_closest_goal}"
            )
            if self.discrete_actions:
                if relative_angle_to_closest_goal > 2 * self.turn_angle / 3.0:
                    action = DiscreteNavigationAction.TURN_RIGHT
                elif relative_angle_to_closest_goal < -2 * self.turn_angle / 3.0:
                    action = DiscreteNavigationAction.TURN_LEFT
                else:
                    # self.log.debug("Already toward the goal, taking a last step toward the goal.")
                    # action = DiscreteNavigationAction.MOVE_FORWARD
                    action = DiscreteNavigationAction.STOP
                    self.moved_forward = True
            else:
                if abs(relative_angle_to_closest_goal) > 10.0:
                    action = ContinuousNavigationAction(np.array([0,0,-relative_angle_to_closest_goal.cpu().item()]))
                else:
                    action = DiscreteNavigationAction.STOP
                    self.moved_forward = True
        return action

    def get_traversible(self, is_local, dilation_raduis):
        obstacles = self.semantic_map.get_obstacle_map(is_local)

        # Add other robot as temporary obstacle before dilation
        robot_sem_idx = MC.NON_SEM_CHANNELS + self.semantic_map.num_sem_categories - 1
        m = self.semantic_map.local_map if is_local else self.semantic_map.global_map
        other_robot = m[robot_sem_idx].cpu().numpy() > 0.5
        # Don't mask frontiers if neighbors nearby, maybe you can go to them later
        self.neighbor_nearby = np.any(other_robot) and self.real_world
        obstacles = obstacles | other_robot

        dilated_obstacles = cv2.dilate(obstacles.astype(np.uint8), skimage.morphology.disk(dilation_raduis), iterations=1).astype(bool)
        dilated_obstacles[self.semantic_map.get_visited_map(is_local, full=True) == 1] = 0
        if is_local:
            gx1, gx2, gy1, gy2 = self.semantic_map.lmb
            collision_map = self.collision_map[gx1:gx2, gy1:gy2] == 1
        else:
            collision_map = self.collision_map == 1
        dilated_obstacles = dilated_obstacles | collision_map
        robot_loc = self.semantic_map.get_loc(is_local)
        dilated_obstacles[robot_loc] = 0
        traversible = ~dilated_obstacles
        return traversible
    
    def raycast_from_viewpoint_to_goal_mask(
        self,
        traversible,
        goal_instance_map,
        viewpoint_location,
        method):

        self.log.debug(f"Viewpoint is : {viewpoint_location}")
        if method == "line_to_com":
            goal_indices = np.argwhere(goal_instance_map == 1)
            center_of_mass = goal_indices.mean(axis=0).astype(int)
            self.log.debug(f"Center of mass of the goal_mapis ({center_of_mass})")
            goal_point = center_of_mass
        else:
            # Find closest goal cell in goal_map to the viewpoint location
            goal_indices = np.argwhere(goal_instance_map == 1)
            dists = np.linalg.norm(
                goal_indices - np.array(viewpoint_location)[None, :], axis=1
            )
            closest_instance_idx = goal_indices[np.argmin(dists)]
            self.log.debug(
                f"Closest goal index in goal_map to goal_pose is ({closest_instance_idx})"
            )
            goal_point = closest_instance_idx

        line_coords = list(
            bresenham(
                goal_point[0],
                goal_point[1],
                viewpoint_location[0],
                viewpoint_location[1],
            )
        )

        first_pixels = []
        in_segment = False

        for x, y in line_coords:
            if not (0 <= x < traversible.shape[0] and 0 <= y < traversible.shape[1]):
                continue

            if traversible[x, y] == 1 and goal_instance_map[x, y] != 1:
                if not in_segment:
                    # Start of a new segment
                    self.log.info(f"Adding traversible pose {x, y}")
                    first_pixels.append((x, y))
                    in_segment = True
            else:
                in_segment = False

        first_pixels.append(viewpoint_location)
        return first_pixels


    def get_goal_map_pose(
        self,
        traversible,
        goal_instance_map,
        viewpoint_location,
        is_local,
        pose_idx,
        method,
    ):
        self.log.info(
            f"Creating goal map using viewpoint. Choosing {pose_idx}th traversible viewpoint."
        )
        candidate_locations = self.raycast_from_viewpoint_to_goal_mask(traversible, goal_instance_map, viewpoint_location,method)
        if pose_idx >= len(candidate_locations):
            self.log.info(f"No traversible view found for the instance goal.")
            return None

        goal_location = candidate_locations[pose_idx]

        goal_map = np.zeros_like(goal_instance_map, dtype=np.uint8)
        goal_map[goal_location[0], goal_location[1]] = 1
        if self.visualization_level > 1:
            self.visualize_get_goal_map(
                traversible,
                goal_instance_map,
                candidate_locations,
                viewpoint_location,
                goal_location,
                pose_idx,
                is_local,
            )
        return goal_map

    def _get_closest_free_cell(self, goal_instance_map, traversible):
        kernel_size = 2
        while kernel_size < 20:
            dilated_goal_instance_map = cv2.dilate(
                goal_instance_map.astype(np.uint8),
                skimage.morphology.disk(kernel_size),
                iterations=1,
            )
            free_goal_cells = np.logical_and(
                dilated_goal_instance_map == 1, traversible == 1
            )
            if np.any(free_goal_cells):
                break
            kernel_size += 1
        return free_goal_cells

    def get_goal_map_closest_to_viewpoint(
        self, traversible, goal_instance_map, viewpoint_location, is_local, try_index
    ):
        """
        Sometimes the free cells are on another side of the wall, and in that cases, the agent will need to continue exploring the map, and sometimes can never go there.
        """
        self.log.info(f"Creating goal map using closest goal to viewpoint.")
        if try_index != 0:
            self.log.info(f"Closest to viewpoint only works for try_index=0")
            return None
        # Here, I have to first dilate the goal instance map a bit, then from all free goal cells, choose closest to the viewpoint
        free_goal_cells = self._get_closest_free_cell(goal_instance_map, traversible)
        viewpoint_map = np.zeros_like(goal_instance_map)
        viewpoint_map[viewpoint_location[0], viewpoint_location[1]] = 1.0
        distances_from_viewpoint = self.planner.set_multi_goal(viewpoint_map, traversible, self.timestep)
        distances_from_viewpoint[free_goal_cells == 0] = 100000

        closest_goal_map = distances_from_viewpoint == distances_from_viewpoint.min()
        goal_location = np.unravel_index(
            closest_goal_map.argmax(), closest_goal_map.shape
        )
        goal_map = np.zeros_like(goal_instance_map, dtype=np.uint8)
        goal_map[goal_location[0], goal_location[1]] = 1

        features = [(goal_instance_map, [0, 165, 255])]  # orange - All instance cells
        points = []
        points.append((viewpoint_location, [255, 0, 0]))
        points.append((goal_location, [0, 0, 255]))

        if self.visualization_level > 1:
            visualize_map(
                traversible.shape,
                self.vis_dir,
                f"{self.prefix}{self.timestep}_6.get_closest_to_viewpoint{'' if is_local else '_global'}.png",
                traversible=traversible,
                features=features,
                points=points,
            )
        return goal_map

    def get_hybrid_goal_map2(
        self, traversible, goal_instance_map, viewpoint_location, is_local, try_index
    ):
        """
        This is a hybrid to approach. We get closest cluster of free cells to the viewpoint. We try to dilate a goal a lot, so it can contain cells on different sides of the goal.
        Finally, we choose the closest cell in the cluster to a goal point.
        """
        self.log.info(f"Creating goal map using hybrid method.")
        candidate_locations = self.raycast_from_viewpoint_to_goal_mask(traversible, goal_instance_map, viewpoint_location, "line_to_closest")
        if try_index >= len(candidate_locations):
            self.log.info(f"No traversible view found for the instance goal.")
            return None

        candidate_location = candidate_locations[try_index]
        candidate_map = np.zeros_like(goal_instance_map, dtype=np.uint8)
        candidate_map[candidate_location[0], candidate_location[1]] = 1

        kernel_size = 10
        candidate_map = cv2.dilate(
            candidate_map.astype(np.uint8),
            skimage.morphology.disk(kernel_size),
            iterations=1,
        )
        free_goal_cells = np.logical_and(
            candidate_map == 1, traversible == 1
        )
        

        # Compute each pixel's distance to the nearest goal cell
        goal_distance_map = distance_transform_edt(goal_instance_map == 0)
        min_dist = goal_distance_map[free_goal_cells].min()
        goal_location = tuple(np.argwhere(
            free_goal_cells & (goal_distance_map == min_dist)
        )[0])


        goal_map = np.zeros_like(goal_instance_map, dtype=bool)
        goal_map[goal_location[0], goal_location[1]] = 1


        features = [(goal_instance_map, [0, 165, 255]), (free_goal_cells, [100,100,0])]  # orange - goal_instance_map
        points = []
        points.append((viewpoint_location, [255, 0, 0]))
        points.append((goal_location, [0, 0, 255]))

        if self.visualization_level > 1:
            visualize_map(
                traversible.shape,
                self.vis_dir,
                f"{self.prefix}{self.timestep}_6.get_hybrid_goal2{'' if is_local else '_global'}.png",
                traversible=traversible,
                features=features,
                points=points,
            )
        return goal_map

    def get_hybrid_goal_map(
        self, traversible, goal_instance_map, viewpoint_location, is_local, try_index
    ):
        """
        This is a hybrid to approach. We get closest cluster of free cells to the viewpoint. We try to dilate a goal a lot, so it can contain cells on different sides of the goal.
        Finally, we choose the closest cell in the cluster to a goal point.
        """
        self.log.info(f"Creating goal map using hybrid method.")
        if try_index != 0:
            self.log.info(f"Hybrid only works for try_index=0")
            return None

        kernel_size = 10
        dilated_goal_instance_map = cv2.dilate(
            goal_instance_map.astype(np.uint8),
            skimage.morphology.disk(kernel_size),
            iterations=1,
        )
        free_goal_cells = np.logical_and(
            dilated_goal_instance_map == 1, traversible == 1
        )
        labeled_map, num_clusters = label(free_goal_cells, structure=np.ones((3, 3)))
        if num_clusters == 0:
            free_goal_cells = self._get_closest_free_cell(
                goal_instance_map, traversible
            )
            labeled_map, num_clusters = label(free_goal_cells, structure=np.ones((3, 3)))
        

        viewpoint_map = np.zeros_like(goal_instance_map)
        viewpoint_map[viewpoint_location[0], viewpoint_location[1]] = 1.0

        traversible_ma = np.ma.masked_values(traversible * 1, 0)
        traversible_ma[viewpoint_location[0], viewpoint_location[1]] = 0
        distances_from_viewpoint = skfmm.distance(traversible_ma)
        distances_from_viewpoint = np.ma.filled(distances_from_viewpoint, np.max(distances_from_viewpoint) + 1)
        distances_from_viewpoint[free_goal_cells != 1] = 100000

        # Get closest cluster to viewpoint
        features = []  # orange - All instance cells
        min_distance_per_cluster = []
        for cluster_id in range(1, num_clusters + 1):
            cluster_mask = labeled_map == cluster_id
            features.append((cluster_mask, [0, int(255/(num_clusters+1) * cluster_id), 0]))
            min_dist_in_cluster = np.min(distances_from_viewpoint[cluster_mask])
            min_distance_per_cluster.append(min_dist_in_cluster)

        closest_cluster_id = np.argmin(min_distance_per_cluster) + 1
        closest_cluster_mask = labeled_map == closest_cluster_id


        # Choose a point in cluster

        # Compute each pixel's distance to the nearest goal cell
        goal_distance_map = distance_transform_edt(goal_instance_map == 0)
        cluster_distances_to_goal = goal_distance_map[closest_cluster_mask]
        min_dist = cluster_distances_to_goal.min()
        close_to_goal_mask = np.logical_and(
            closest_cluster_mask,
            goal_distance_map <= min_dist * 1.2
        )

        closest_idx = np.argmin(distances_from_viewpoint[close_to_goal_mask])
        candidate_coords = np.argwhere(close_to_goal_mask)
        goal_location = tuple(candidate_coords[closest_idx])

        goal_map = np.zeros_like(goal_instance_map, dtype=bool)
        goal_map[goal_location[0], goal_location[1]] = 1

        features.append((goal_instance_map, [0, 165, 255]))  # orange - All instance cells
        points = []
        points.append((viewpoint_location, [255, 0, 0]))
        points.append((goal_location, [0, 0, 255]))

        if self.visualization_level > 1:
            visualize_map(
                traversible.shape,
                self.vis_dir,
                f"{self.prefix}{self.timestep}_6.get_hybrid_goal{'' if is_local else '_global'}.png",
                traversible=traversible,
                features=features,
                points=points,
            )
        return goal_map

    def get_hybrid_goal_map3(
        self, traversible, goal_instance_map, viewpoint_location, is_local, try_index
    ):
        """
        This is a hybrid to approach. We get closest cluster of free cells to the viewpoint. We try to dilate a goal a lot, so it can contain cells on different sides of the goal.
        Finally, we choose the closest cell in the cluster to a goal point.
        """
        self.log.info(f"Creating goal map using hybrid method.")
        if try_index != 0:
            self.log.info(f"Hybrid only works for try_index=0")
            return None

        kernel_size = 10
        max_retries = 3  # original, 2x, 3x

        # Compute Euclidean distance from viewpoint to goal center
        goal_coords = np.argwhere(goal_instance_map == 1)
        goal_center = goal_coords.mean(axis=0)
        euclidean_dist = np.linalg.norm(np.array(viewpoint_location) - goal_center)
        threshold = 2.0 * euclidean_dist

        valid_cluster_ids = []
        valid_min_distances = []
        features = []

        viewpoint_map = np.zeros_like(goal_instance_map)
        viewpoint_map[viewpoint_location[0], viewpoint_location[1]] = 1.0
        traversible_ma = np.ma.masked_values(traversible * 1, 0)
        traversible_ma[viewpoint_location[0], viewpoint_location[1]] = 0
        distances_from_viewpoint = skfmm.distance(traversible_ma)
        distances_from_viewpoint = np.ma.filled(distances_from_viewpoint, np.max(distances_from_viewpoint) + 1)
        for attempt in range(max_retries):
            current_kernel = kernel_size * (attempt + 1)
            dilated_goal_instance_map = cv2.dilate(
                goal_instance_map.astype(np.uint8),
                skimage.morphology.disk(current_kernel),
                iterations=1,
            )
            free_goal_cells = np.logical_and(
                dilated_goal_instance_map == 1, traversible == 1
            )
            labeled_map, num_clusters = label(free_goal_cells, structure=np.ones((3, 3)))
            if num_clusters == 0:
                free_goal_cells = self._get_closest_free_cell(
                    goal_instance_map, traversible
                )
                labeled_map, num_clusters = label(free_goal_cells, structure=np.ones((3, 3)))

            distances_from_viewpoint_masked = distances_from_viewpoint.copy()
            distances_from_viewpoint_masked[free_goal_cells != 1] = 100000

            features = []
            min_distance_per_cluster = []
            for cluster_id in range(1, num_clusters + 1):
                cluster_mask = labeled_map == cluster_id
                features.append((cluster_mask, [0, int(255 / (num_clusters + 1) * cluster_id), 0]))
                min_dist_in_cluster = np.min(distances_from_viewpoint_masked[cluster_mask])
                min_distance_per_cluster.append(min_dist_in_cluster)

            valid_cluster_ids = []
            valid_min_distances = []
            valid_features = []
            for i, cluster_id in enumerate(range(1, num_clusters + 1)):
                if min_distance_per_cluster[i] <= threshold:
                    valid_cluster_ids.append(cluster_id)
                    valid_min_distances.append(min_distance_per_cluster[i])
                    valid_features.append(features[i])

            if len(valid_cluster_ids) > 0:
                self.log.info(f"Found {len(valid_cluster_ids)} valid clusters with kernel_size={current_kernel} (attempt {attempt + 1}).")
                features = valid_features
                break
            else:
                self.log.info(f"All clusters filtered out with kernel_size={current_kernel} (attempt {attempt + 1}), retrying...")

        if len(valid_cluster_ids) == 0:
            self.log.info("All retries exhausted, falling back to closest cluster without filtering.")
            valid_cluster_ids = list(range(1, num_clusters + 1))
            valid_min_distances = min_distance_per_cluster

        closest_idx_in_valid = np.argmin(valid_min_distances)
        closest_cluster_id = valid_cluster_ids[closest_idx_in_valid]
        closest_cluster_mask = labeled_map == closest_cluster_id
        max_reachable_dist = (np.min(distances_from_viewpoint_masked[closest_cluster_mask])+5) * 4
        reachable_cluster_mask = np.logical_and(
            closest_cluster_mask,
            distances_from_viewpoint_masked <= max_reachable_dist
        )

        # Choose a point in cluster

        # Compute each pixel's distance to the nearest goal cell
        goal_distance_map = distance_transform_edt(goal_instance_map == 0)
        cluster_distances_to_goal = goal_distance_map[reachable_cluster_mask]
        min_dist = cluster_distances_to_goal.min()
        close_to_goal_mask = np.logical_and(
            reachable_cluster_mask,
            goal_distance_map <= min_dist * 1.2
        )

        closest_idx = np.argmin(distances_from_viewpoint_masked[close_to_goal_mask])
        candidate_coords = np.argwhere(close_to_goal_mask)
        goal_location = tuple(candidate_coords[closest_idx])

        goal_map = np.zeros_like(goal_instance_map, dtype=bool)
        goal_map[goal_location[0], goal_location[1]] = 1

        if self.visualization_level > 1:
            features.append((goal_instance_map, [0, 165, 255]))  # orange - All instance cells
            points = []
            points.append((viewpoint_location, [255, 0, 0]))
            points.append((goal_location, [0, 0, 255]))
            features.append((reachable_cluster_mask, [255, 0, 255]))  # magenta - reachable portion
            features.append((close_to_goal_mask, [0, 255, 255]))      # yellow - final candidates
            visualize_map(
                traversible.shape,
                self.vis_dir,
                f"{self.prefix}{self.timestep}_6.get_hybrid_goal{'' if is_local else '_global'}.png",
                traversible=traversible,
                features=features,
                points=points,
            )
        return goal_map

    def get_goal_map(
        self,
        traversible,
        goal_instance_map,
        best_viewpoint,
        is_local,
        try_index,
        method,
        try_best=False,
    ):
        """
        Args:
            goal_map
            view_pose: Global loc that we can see the goal instance.
            method: "line_to_com" or "line_to_closest" or "closest_to_viewpoint"
        """
        if self.ground_truth_semantics and np.sum(goal_instance_map) < 47 and not try_best:
            self.log.info(f"Goal instance map too small ({np.sum(goal_instance_map)} cells). Not planning to it.")
            return None
        viewpoint_location = (
        self.semantic_map.global_pose_to_local_location(best_viewpoint.pose)
            if is_local
            else self.semantic_map.global_pose_to_global_location(best_viewpoint.pose)
        )
        if method in ["line_to_com", "line_to_closest"]:
            goal_map = self.get_goal_map_pose(
                traversible,
                goal_instance_map,
                viewpoint_location,
                is_local,
                try_index,
                method,
            )
        elif method == "closest_to_viewpoint":  # closest_to_viewpoint
            goal_map = self.get_goal_map_closest_to_viewpoint(
                traversible, goal_instance_map, viewpoint_location, is_local, try_index
            )
        elif method == "hybrid":
            goal_map = self.get_hybrid_goal_map3(
                traversible, goal_instance_map, viewpoint_location, is_local, try_index
            )
        if goal_map is not None:
            assert goal_map.dtype == bool
            assert np.sum(goal_map) > 0

        return goal_map

    def _get_short_term_goal(
        self,
        traversible: np.ndarray,
        goal_map: np.ndarray,
        location: List[int],
        goal_instance_map=None,
        postfix: str = "",
    ) -> Tuple[Tuple[int, int], np.ndarray, bool, bool]:
        """Get short-term goal.

        Args:
            obstacle_map: (M, M) binary local obstacle map prediction
            goal_map: (M, M) binary array denoting goal location
            start: start location (x, y)
            planning_window: local map boundaries (gx1, gx2, gy1, gy2)
            plan_to_dilated_goal: for objectnav; plans to dialted goal points instead of explicitly checking reach.

        Returns:
            short_term_goal: short-term goal position (x, y) in map
            replan: binary flag to indicate we couldn't find a plan to reach
             the goal
            stop: binary flag to indicate we've reached the goal
        """
        if goal_instance_map is not None:
            goal_cells = np.argwhere(goal_instance_map == 1)
            distances = np.linalg.norm(goal_cells - np.asarray(location), axis=1)
            dist_to_closest = float(distances.min())
            # print("dist to closest goal cell: ", dist_to_closest)
            if dist_to_closest < self.stop_distance:
                return True, True, None

        self.log.debug(f"Getting short-term goal")
        if not np.any(goal_map):
            self.log.info(
                f"Couldn't find any navigable goal points in the map. Should only happen for frontier."
            )
            return (
                False,
                False,
                None,
            )

        self.dd = self.planner.set_multi_goal(
            goal_map,
            traversible,
            self.timestep,
            None,
            self.map_downsample_factor,
            self.map_update_frequency,
            vis_postfix=postfix,
        )

        # This is where we create the planner to get the trajectory to this state
        stg_x, stg_y, reachable, stop = self.planner.get_short_term_goal(
            location, timestep=self.timestep, prefix=self.prefix, postfix=postfix
        )

        short_term_goal = int(stg_x), int(stg_y)

        if self.visualization_level > 0:
            points = [
                ([location[0], location[1]], [255, 0, 0]),  # start blue
                (
                    [short_term_goal[0], short_term_goal[1]],
                    [0, 255, 0],
                ),  # stg green
            ]
            visualize_map(
                goal_map.shape,
                self.vis_dir,
                f"{self.prefix}{self.timestep}_12.stg{postfix}.png",
                points=points,
                traversible=traversible,
                goal_map=goal_map,
            )

        return reachable, stop, short_term_goal

    def _check_collision(self, postfix):
        """Check whether we had a collision and update the collision map."""
        collision_map_change = False
        x1, y1, t1 = self.last_global_pose.cpu().numpy()
        x2, y2, _ = self.curr_global_pose.cpu().numpy()
        # buf = 4
        buf = 2
        length = 2

        # You must move at least 5 cm when doing forward actions
        # Otherwise we assume there has been a collision
        if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
            self.col_width += 2
            # if self.col_width == 7:
            #     length = 4
            # buf = 3
            self.col_width = min(self.col_width, 5)
        else:
            self.col_width = 1

        dist = pu.get_l2_distance(x1, x2, y1, y2)

        if dist < self.collision_threshold:
            # We have a collision
            width = self.col_width

            # Add obstacles to the collision map
            for i in range(length):
                for j in range(width):
                    wx = x1 + self.map_resolution / 100 * (
                        (i + buf) * np.cos(np.deg2rad(t1))
                        + (j - width // 2) * np.sin(np.deg2rad(t1))
                    )
                    wy = y1 + self.map_resolution / 100 * (
                        (i + buf) * np.sin(np.deg2rad(t1))
                        - (j - width // 2) * np.cos(np.deg2rad(t1))
                    )
                    r, c = wy, wx
                    r, c = int(r * 100 / self.map_resolution), int(
                        c * 100 / self.map_resolution
                    )
                    [r, c] = pu.threshold_poses([r, c], self.collision_map.shape)
                    self.collision_map[r, c] = 1
                    collision_map_change = True
        
        robot_loc = self.semantic_map.global_pose_to_global_location(self.curr_global_pose)
        self.collision_map[robot_loc[0], robot_loc[1]] = 0

        if collision_map_change and self.visualization_level > 1:
            obstacle_map = self.semantic_map.get_obstacle_map(False)
            init_location = self.semantic_map.global_pose_to_global_location(self.last_global_pose)
            visualize_map(
                obstacle_map.shape,
                self.vis_dir,
                f"{self.prefix}{self.timestep}_4.collision_map_update{postfix}.png",
                points=[(init_location, [120, 0, 0]), (robot_loc, [255, 0, 0])],
                traversible=1 - obstacle_map,
                features=[
                    (self.collision_map, [0, 120, 120]),
                    (np.logical_and(self.collision_map, obstacle_map), [0, 255, 255]),
                ],  # light yellow / yellow
            )

    def get_largest_cluster(self, goal_instance_map, is_local) -> None:
        def convex_hull(binary_map):
            coords = np.column_stack(np.nonzero(binary_map))
            if len(coords) < 3:
                return binary_map

            hull = ConvexHull(coords)
            hull_coords = coords[hull.vertices]

            rr, cc = polygon(hull_coords[:, 0], hull_coords[:, 1], binary_map.shape)
            hull_mask = np.zeros_like(binary_map, dtype=np.uint8)
            hull_mask[rr, cc] = 1
            hull_mask = cv2.dilate(hull_mask, np.ones((3, 3)), iterations=1)

            return hull_mask

        if not self.goal_filtering:
            return

        self.log.debug("Clustering Instance map and selecting the largest cluster.")
        # init_goal_map_count = goal_instance_map.sum()

        labeled_map, _ = label(goal_instance_map) # Use deault structure of only vert or hor connection, to not include noises
        component_sizes = np.bincount(labeled_map.ravel())
        component_sizes[0] = 0
        if component_sizes.max() == 0:
            self.log.warning(f"Instance map is empty — no clusters found.")
            return goal_instance_map  # return empty map, let caller handle it

        largest_label = component_sizes.argmax()
        clustered_map = labeled_map == largest_label

        clustered_map_convex_hull = None
        if clustered_map.sum() > 0:
            # convex hull
            # Im not sure this was a good idea! so commented for now
            pass
            # try:
            #     clustered_map_convex_hull = convex_hull(clustered_map)
            #     self.log.debug("Choosing largest cluster for instance map.")
            #     self.log.debug(
            #         f"Goal map cells count changed from {init_goal_map_count} to {clustered_map.sum()}"
            #     )
            # except:
            #     self.log.debug(
            #         "Convex hull failed. Using the largest cluster without convex hull."
            #     )
        else:
            raise Exception("No largest cluster found in the instance map.")
            

        if self.visualization_level > 1:
            visualize_map(
                goal_instance_map.shape,
                self.vis_dir,
                f"{self.prefix}{self.timestep}_3.cluster_goal.png",
                goal_map=clustered_map_convex_hull,  # Clustered goal map convex hull is red
                dilated_goal_map=goal_instance_map,  # All init goal points are magenta
                traversible=~self.semantic_map.get_obstacle_map(is_local),
                features=[(clustered_map, [0, 255, 0])],  # Largest cluster is green
            )
        if clustered_map_convex_hull is None:
            assert clustered_map.dtype == bool
            return clustered_map
        return clustered_map_convex_hull

    def get_frontier_planning_maps(self):
        min_size = 12
        if self.curr_frontier_dilation == self.min_obs_dilation_selem_radius:
            min_size = 8
        traversible = self.get_traversible(True, self.curr_frontier_dilation)
        frontier_map = self.semantic_map.get_frontier_map(
            local=True, timestep=self.timestep, min_size=min_size, traversible=traversible
        )
        assert not np.any((frontier_map == 1) & (traversible == 0)), "Non-traversible frontier cells"

        if frontier_map.any():
            return frontier_map, traversible, True

        traversible = self.get_traversible(False, self.curr_frontier_dilation)
        frontier_map = self.semantic_map.get_frontier_map(
            local=False, timestep=self.timestep, min_size=min_size, traversible=traversible
        )

        if frontier_map.any():
            return frontier_map, traversible, False

        return None, None, False

    def _neighbor_set_changed(self, neighbor_locs):
        current_set = set(neighbor_locs.keys()) if neighbor_locs else set()
        changed = current_set != self._last_neighbor_set
        self._last_neighbor_set = current_set
        return changed

    def plan_to_frontier_goal(self, goal_category, postfix, neighbor_locs=None):
        saved_unreachable_global = self.semantic_map.global_map[MC.UNREACHABLE_FRONTIERS_MAP].clone()
        i = 0
        while True:
            self.log.info(f"Getting frontier planning maps")
            frontier_map, traversible, is_local = self.get_frontier_planning_maps()
            self.log.info(f"[DEBUG HANG] get_frontier_planning_maps returned: frontier={'None' if frontier_map is None else frontier_map.sum()} cells")
            if frontier_map is None:
                self.log.info(
                    "No frontiers remaining. Setting obstacle dilation to min to find possible frontiers."
                )
                self.curr_frontier_dilation, success = self.decrease_obstacle_dilation_radius(self.curr_frontier_dilation)
                if success:
                    self.semantic_map.reset_unreachable_frontier()
                    continue

                self.log.info("No frontiers remaining even with min dilation.")
                return False, False, None, None, {}

            robot_loc = self.semantic_map.get_loc(is_local)
            current_neighbor_set = set(neighbor_locs.keys()) if neighbor_locs else set()
            if is_local:
                lmb = self.semantic_map.lmb
                prev_frontier = self.prev_frontier[lmb[0]:lmb[1], lmb[2]:lmb[3]]
            else:
                prev_frontier = self.prev_frontier
            #! TODO: Save time started a frontier instead
            if self._last_neighbor_set != current_neighbor_set or np.all((prev_frontier & frontier_map) == 0)  or self.timestep % 8 == 0:
                best_frontier_map = self.get_best_frontier(
                    frontier_map,
                    traversible,
                    robot_loc,
                    goal_category,
                    is_local,
                    neighbor_locs=neighbor_locs,
                )
            else:
                self.log.debug("Using previous frontier map for planning.")
                best_frontier_map = prev_frontier & frontier_map

            self._last_neighbor_set = current_neighbor_set

            if self.visualization_level > 2:
                visualize_map(
                    traversible.shape,
                    self.vis_dir,
                    f"{self.timestep}_5.visited_map{postfix}.png",
                    points=[(robot_loc, [255, 0, 0])],
                    traversible=traversible,
                    frontier_map=self.semantic_map.get_visited_map(is_local)
                )
                visualize_map(
                    traversible.shape,
                    self.vis_dir,
                    f"{self.timestep}_6.unreachable_frontiers{f'_{i}'}{'' if is_local else '_global'}{postfix}.png",
                    points=[(robot_loc, [255, 0, 0])],
                    traversible=traversible,
                    frontier_map=self.semantic_map.get_unreachable_frontiers_map(is_local)
                )
            if self.visualization_level > 1:
                visualize_map(
                    traversible.shape,
                    self.vis_dir,
                    f"{self.timestep}_7.planning_input_frontier{f'_{i}'}{'' if is_local else '_global'}{postfix}.png",
                    points=[(robot_loc, [255, 0, 0])],
                    traversible=traversible,
                    dilated_goal_map=frontier_map,
                    frontier_map=best_frontier_map,
                )

            (
                reachable,
                stop,
                short_term_goal
            ) = self._get_short_term_goal(
                traversible,
                best_frontier_map,
                robot_loc,
                postfix=f"_frontier_{i}",
            )
            stop = False
            if reachable:
                self.log.info("Planning to frontier successfull.")
                if is_local:
                    lmb = self.semantic_map.lmb
                    global_frontier = np.zeros(self.map_shape, dtype=np.uint8)
                    global_frontier[lmb[0]:lmb[1], lmb[2]:lmb[3]] = best_frontier_map
                    self.prev_frontier = global_frontier
                else:
                    self.prev_frontier = best_frontier_map
                break

            self.log.info("Frontier not reachable.")
            self.semantic_map.set_unreachable_frontier(
                best_frontier_map, is_local
            )
            i += 1

        if self.neighbor_nearby:
            self.semantic_map.global_map[MC.UNREACHABLE_FRONTIERS_MAP] = saved_unreachable_global

        vis_input = {}
        if reachable:
            vis_input = {
                "short_term_goal": short_term_goal,
                "is_local": is_local,
            }
        vis_input["obstacle_map"] = ~traversible
        return reachable, stop, short_term_goal, is_local, vis_input

    def get_best_frontier(
        self,
        frontier_map,
        traversible,
        robot_loc,
        goal_category,
        is_local,
        neighbor_locs=None,
    ):
        def distance_to_frontier(frontier, distances, loc):
            # Choose the closest point in the frontier to the robot
            dists = np.linalg.norm(frontier - loc, axis=1)
            closest = frontier[np.argmin(dists)]
            assert (
                traversible[closest[0], closest[1]] == 1
            ), "Closest point is not traversible"
            # self.log.debug(f"Frontier: {k}, closest: {closest}")
            return distances[closest[0], closest[1]]

            # Choose the center. Problem: Sometimes occupide.
            # return distances[center[0], center[1]]
        assert (
            traversible[robot_loc[0], robot_loc[1]] == 1
        ), "Robot location is not traversible"

        structure = np.ones((3, 3))  # 8-connectivity
        labeled_map, num_features = label(frontier_map, structure=structure)
        frontiers = [np.argwhere(labeled_map == i) for i in range(1, num_features + 1)]

        if self.frontier_metric== "semantics":
            sem_weights = CO_LOCATION_WEIGHTS[goal_category]
            sem_layers = self.semantic_map.get_semantic_map(is_local)
            semantic_close_radius = 40

        frontier_scores, frontier_centers, top_k_semantic_classes = [], [], []

        self.log.debug(f"Getting best frontier")
        traversible_ma = np.ma.masked_values(traversible * 1, 0)
        traversible_ma[robot_loc[0], robot_loc[1]] = 0
        distances = skfmm.distance(traversible_ma)
        distances = np.ma.filled(distances, np.max(distances) + 1)

        if neighbor_locs is not None:
            for key, loc in neighbor_locs.items():
                neighbor_locs[key] = (
                    self.semantic_map.global_location_to_local_location(
                        loc
                    )
                    if is_local
                    else loc
                )

        neighbor_distance_cache = {}
        agent_distances, other_distances = [], []
        for k, frontier in enumerate(frontiers):
            distance = distance_to_frontier(frontier, distances, robot_loc)
            # self.log.debug(f"Frontier: {k}")
            # self.log.debug(f" - distance: {distance}, robot loc: {robot_loc}")
            # if distance == np.max(distances):
                # Should not mask them here, because we are not sure if it is the min dilation
                # Also it interferes with the backup logic of decreasing dilation and trying again, and then masking.
                # self.semantic_map.set_unreachable_frontier(
                #     labeled_map == k+1, is_local
                # )
                # continue


            center = frontier.mean(axis=0).astype(int)
            frontier_centers.append(center)

            if self.frontier_metric == "distance":
                agent_distances.append(distance)
                if neighbor_locs is None or len(neighbor_locs) == 0:
                    frontier_scores.append(1 / (distance + 1))
                    top_k_semantic_classes.append([])
                else:
                    neighbor_distances = []
                    for loc in neighbor_locs.values():
                        traversible_ma = np.ma.masked_values(traversible * 1, 0)
                        if self.semantic_map.is_location_in_local_map(loc):
                            key = tuple(loc)
                            if key not in neighbor_distance_cache:
                                traversible_ma[loc[0], loc[1]] = 0
                                ndistances = skfmm.distance(traversible_ma)
                                ndistances = np.ma.filled(
                                    ndistances, np.max(ndistances) + 1
                                )
                                neighbor_distance_cache[key] = ndistances
                                neighbor_distance = distance_to_frontier(
                                    frontier, ndistances, loc
                                )
                            else:
                                ndistances = neighbor_distance_cache[key]
                                neighbor_distance = distance_to_frontier(
                                    frontier, ndistances, loc
                                )

                            # self.log.debug(
                            #     f"     - neighbor dis: {neighbor_distance}, neighbor loc: {neighbor_loc}"
                            # )
                        else:
                            neighbor_distance = 100000

                        neighbor_distances.append(neighbor_distance)

                    other_distances.append(neighbor_distances)
                    neighbor_distance = min(neighbor_distances)
                    # The number in denominator removes division by 0. The value in the numinator ensures that if two agents have the same distance, we choose the frontier with smallest distance.
                    frontier_scores.append(
                        (neighbor_distance + 10e-5) / (distance + 10e-6)
                    )
                    # self.log.debug(
                    #     f"min neighbor dis: {neighbor_distance}, frontier score: {frontier_scores[-1]}"
                    # )
                    top_k_semantic_classes.append([])

            elif self.frontier_metric== "semantics":
                local_map = sem_layers[
                    1 : 52 + 1,
                    center[0] - semantic_close_radius : center[0] + semantic_close_radius,
                    center[1] - semantic_close_radius : center[1] + semantic_close_radius,
                ]
                neighbor_classes = (
                    np.where(local_map.any(axis=(1, 2)))[0] + 1
                )  # +1 to match ids

                # self.log.debug(f"frontier {k}")
                if len(neighbor_classes) > 0:
                    scores = sem_weights[neighbor_classes]
                    frontier_sem_score = np.mean(scores)
                    top_classes = neighbor_classes[np.argsort(scores)[-3:]].tolist()
                    # self.log.debug(f"Top classes: {top_classes}, Scores: {scores}, Frontier score: {frontier_sem_score}")
                else:
                    frontier_sem_score = np.mean(sem_weights)
                    # self.log.debug(f"No classes found in the local map. Using mean score: {frontier_sem_score}")
                    top_classes = []

                frontier_sem_score = np.exp(8 * frontier_sem_score)
                frontier_sem_score /= distance
                # self.log.debug(f"distance: {distance}, final score: {frontier_sem_score}")
                frontier_scores.append(frontier_sem_score)
                top_k_semantic_classes.append(top_classes)
            else:
                raise Exception(f"Unknown metric: {self.frontier_metric}")
        if self.visualization_level > 1:
            if self.frontier_metric== "distance":
                visualize_distance_frontiers(
                    self.vis_dir,
                    traversible=traversible,
                    frontier_map=frontier_map,
                    frontier_centers=frontier_centers,
                    frontier_scores=frontier_scores,
                    robot_loc=robot_loc,
                    agent_dists=agent_distances,
                    other_agents_dists=other_distances,
                    top_k=5,
                    save_path=f"{self.prefix}{self.timestep}_14.frontier_scores{'' if is_local else '_global'}.png",
                    neighbor_locs=neighbor_locs,
                )
            else:
                visualize_semantic_frontiers(
                    self.vis_dir,
                    traversible=traversible,
                    frontier_map=frontier_map,
                    frontier_centers=frontier_centers,
                    frontier_scores=frontier_scores,
                    top_k_semantic_classes=top_k_semantic_classes,
                    robot_loc=robot_loc,
                    top_k=5,
                    save_path=f"{self.prefix}_14.frontier_scores{'' if is_local else '_global'}.png",
                )
        assert (
            len(frontier_scores) != 0
        ), "No frontiers found, but frontier_map is not empty."

        my_priority = 1
        if neighbor_locs is not None:
            for agent_id, loc in neighbor_locs.items():
                if loc == robot_loc and agent_id < self.agent_id:
                    my_priority += 1

        if my_priority <= len(frontier_scores):
            self.log.debug(
                f"Getting {my_priority}th best frontier (agent: {self.agent_id})"
            )
            best_frontier = np.argsort(frontier_scores)[-my_priority]
        else:
            # If more agents on the same location than the number of frontiers, go to the best
            best_frontier = np.argsort(frontier_scores)[-1]

        best_frontier_map = np.zeros_like(traversible)
        best_frontier = frontiers[best_frontier]
        best_frontier_map[best_frontier[:, 0], best_frontier[:, 1]] = 1
        return best_frontier_map

    def plan_to_instance_goal(self, instance_goal_id, try_best, postfix, method="hybrid"):
        (
            goal_instance_map,
            best_viewpoint,
            is_local,
            traversible,
        ) = self.get_instance_planning_maps(instance_goal_id, method)
        robot_loc = self.semantic_map.get_loc(is_local)

        if self.visualization_level > 2:
            visualize_map(
                traversible.shape,
                self.vis_dir,
                f"{self.timestep}_5.visited_map{postfix}.png",
                points=[(robot_loc, [255, 0, 0])],
                traversible=traversible,
                frontier_map=self.semantic_map.get_visited_map(is_local)
            )
            instance_on_obstacles = goal_instance_map & ~traversible
            viewpoint_location = (
            self.semantic_map.global_pose_to_local_location(best_viewpoint.pose)
                if is_local
                else self.semantic_map.global_pose_to_global_location(best_viewpoint.pose)
            )
            visualize_map(
                traversible.shape,
                self.vis_dir,
                f"{self.timestep}_7.planning_input_instance{postfix}.png",
                points=[(robot_loc, [255, 0, 0]), (viewpoint_location, [120, 0, 0])],
                traversible=traversible,
                goal_map=goal_instance_map,
                features=[(instance_on_obstacles, [0, 255, 255])],  # yellow
            )

        i = 0
        try_idx = 0
        stop = False
        reachable = False
        force_global = False
        short_term_goal = None
        while True:
            goal_map = self.get_goal_map(
                traversible,
                goal_instance_map,
                best_viewpoint,
                is_local,
                try_idx,
                method,
                try_best=try_best,
            )
            if goal_map is None:
                # This mean we couldn't find any traversible pose. Even with the minimum dilation radius.
                break

            self.log.info(
                f"Trying to plan to instance goal with\n\t - pose_idx: {try_idx}\n\t - {'local' if is_local else 'global'}\n\t - obs dilation: {self.curr_frontier_dilation}"
            )

            (reachable, stop, short_term_goal) = (
                self._get_short_term_goal(
                    traversible,
                    goal_map,
                    robot_loc,
                    goal_instance_map,
                    postfix=f"_attempt_{i}_pose_{try_idx}{'' if is_local else '_global'}{postfix}",
                )
            )

            if stop or reachable:
                break

            i += 1

            self.log.info("Could not find a path to the high-level goal.")
            if is_local:
                force_global = True
            else:
                force_global = False
                self.curr_instance_dilation, success = self.decrease_obstacle_dilation_radius(self.curr_instance_dilation)
                if not success:
                    try_idx += 1
                    #! myTODO: Important: This might not be lots of heurisitc. Maybe its better to use something like BLACKLISTED_TARGET_MAP, however, that has its own issues.
                    self.semantic_map.merge_map(
                        goal_map, MC.OBSTACLE_MAP, is_local
                    )
                    self.curr_frontier_dilation = self.start_obs_dilation_selem_radius
            (
                goal_instance_map,
                best_viewpoint,
                is_local,
                traversible,
            ) = self.get_instance_planning_maps(
                instance_goal_id, method, force_global=force_global
            )
            robot_loc = self.semantic_map.get_loc(is_local)

        if reachable:
            self.log.debug(f"Planning to instance goal successfull.")
            if stop:
                self.log.debug(f"We need to stop.")
            else:
                self.log.debug(f"Short term goal: {short_term_goal}")
        if stop:
            x_goal, y_goal = np.where(goal_instance_map)
            short_term_goal = int(np.mean(x_goal)), int(np.mean(y_goal))

        vis_input = {}
        if reachable:
            vis_input = {
                "short_term_goal": short_term_goal,
                "is_local": is_local,
                "inst_goal_found": True,
            }
        vis_input["obstacle_map"] = 1 - traversible
        vis_input["goal_instance_map"] = goal_instance_map

        return (
            reachable,
            stop,
            short_term_goal,
            is_local,
            vis_input,
        )

    def get_goal_instance_map_and_viewpoint(
        self, instance_goal_id, method, force_global=False
    ):
        instance_views = self.instance_memory.instances[instance_goal_id].instance_views

        local_goal_instance_map = self.semantic_map.get_instance_map(
            instance_goal_id, local=True
        )
        global_goal_instance_map = self.semantic_map.get_instance_map(
            instance_goal_id, local=False
        )

        all_views_local_locations = [
            self.semantic_map.global_pose_to_local_location(view.pose)
            for view in instance_views
        ]
        are_views_local = all(
            [
                self.semantic_map.is_location_in_local_map(view_local_location)
                for view_local_location in all_views_local_locations
            ]
        )

        is_local = (
            are_views_local
            and np.any(local_goal_instance_map)
            and np.count_nonzero(local_goal_instance_map)
            == np.count_nonzero(global_goal_instance_map)
            and not force_global
        )

        if is_local:
            goal_instance_map = local_goal_instance_map
            all_views_locations = all_views_local_locations
        else:
            goal_instance_map = global_goal_instance_map
            all_views_locations = [
                self.semantic_map.global_pose_to_global_location(view.pose)
                for view in instance_views
            ]
        
        assert goal_instance_map.sum() > 0, f"Goal instance map is empty for instance {instance_goal_id}."

        goal_instance_map = self.get_largest_cluster(goal_instance_map, is_local)

        if method == "hybrid":
            goal_indices = np.argwhere(goal_instance_map == 1)
            center = goal_indices.mean(axis=0)  # can be float
            all_views_locations = np.stack(all_views_locations)
            best_view = np.argmin(np.linalg.norm(all_views_locations - center, axis=1))
        else:
            best_view = np.argmax([view.object_coverage for view in instance_views])

        best_view_pose = instance_views[best_view].pose
        best_view_loc = (
            self.semantic_map.global_pose_to_local_location(best_view_pose)
            if is_local
            else self.semantic_map.global_pose_to_global_location(best_view_pose)
        )
        self.log.debug(
            f"viewpoint location is: {best_view_loc}, with coverage {instance_views[best_view].object_coverage}."
        )

        self.log.debug(
            f">>> Goal instance {instance_goal_id} {'not' if not is_local else ''} present in local map."
        )
        return (
            goal_instance_map,
            instance_views[best_view],
            is_local,
        )

    def get_instance_planning_maps(self, instance_goal_id, method, force_global=False):
        goal_instance_map, best_viewpoint, is_local = (
            self.get_goal_instance_map_and_viewpoint(
                instance_goal_id, method, force_global=force_global
            )
        )
        traversible = self.get_traversible(is_local, self.curr_instance_dilation)
        return (
            goal_instance_map,
            best_viewpoint,
            is_local,
            traversible,
        )


    def decrease_obstacle_dilation_radius(self, curr_radius):
        if curr_radius > self.min_obs_dilation_selem_radius:
            self.log.info(
                f"Decreasing obstacle dilation radius to {curr_radius - 1}."
            )
            return curr_radius - 1, True
        else:
            return curr_radius, False

    def visualize_get_goal_map(
        self,
        traversible,
        goal_instance_map,
        first_pixels,
        viewpoint_location,
        goal_location,
        index,
        is_local,
    ):
        features = [(goal_instance_map, [0, 165, 255])]  # orange - All instance cells
        points = []
        for pixel in first_pixels:
            points.append((pixel, [0, 255, 0]))  # green - All poses

        # blue - viewpoint
        points.append((viewpoint_location, [255, 0, 0]))

        # red - final goal
        points.append((goal_location, [0, 0, 255]))

        visualize_map(
            traversible.shape,
            self.vis_dir,
            f"{self.timestep}_6.interpolate_goal_idx{index}{'' if is_local else '_global'}.png",
            traversible=traversible,
            features=features,
            points=points,
        )
