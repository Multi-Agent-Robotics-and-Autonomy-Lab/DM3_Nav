# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import scipy
import skimage.morphology
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN

from home_robot.mapping.semantic.constants import MapConstants as MC
from home_robot.utils.morphology import binary_dilation
from scipy.ndimage import label

from home_robot.utils.logger import get_logger
logger = get_logger()

import cv2
import os


class LanguageNavSemanticFrontierExplorationPolicy(nn.Module):
    """
    Policy to select high-level goals for Object Goal Navigation:
    go to object goal if it is mapped and explore frontier (closest
    unexplored region) otherwise.
    """

    def __init__(
        self,
        exploration_strategy: str,
        num_of_semantic_categories: int,
        warmup_steps: int,
    ):
        super().__init__()
        assert exploration_strategy in [
            "seen_frontier",
            "been_close_to_frontier",
            "semantic",
        ]
        self.exploration_strategy = exploration_strategy

        self.dilate_explored_kernel = nn.Parameter(
            torch.from_numpy(skimage.morphology.disk(10))
            .unsqueeze(0)
            .unsqueeze(0)
            .float(),
            requires_grad=False,
        )
        self.select_border_kernel = nn.Parameter(
            torch.from_numpy(skimage.morphology.disk(1))
            .unsqueeze(0)
            .unsqueeze(0)
            .float(),
            requires_grad=False,
        )
        self.num_of_semantic_categories = num_of_semantic_categories
        self.warmup_steps = warmup_steps

    @property
    def goal_update_steps(self):
        return 1

    def forward(
        self,
        map_features,
        step,
        robot_position,
        object_category=None,
        reject_visited_targets=False,
    ):
        """
        Arguments:
            map_features: semantic map features of shape
             (batch_size, 9 + num_sem_categories, M, M)
            object_category: object goal category
        Returns:
            goal_map: binary map encoding goal(s) of shape (batch_size, M, M)
            found_goal: binary variables to denote whether we found the object
            goal category of shape (batch_size,)
        """
        assert object_category is not None

        goal_map, found_goal = self.reach_goal_if_in_map(
            map_features, object_category, reject_visited_targets=reject_visited_targets
        )

        if self.exploration_strategy == "semantic" and step > self.warmup_steps:
            goal_map = self.semantic_exploration(
                map_features, goal_map, found_goal, object_category, robot_position
            )
        else:
            goal_map = self.explore_otherwise(map_features, goal_map, found_goal)
        return goal_map, found_goal

    def semantic_exploration(
        self, map_features, goal_map, found_goal, category, robot_position
    ):
        if found_goal[0]:
            return goal_map

        sem_weights = np.random.rand(
            self.num_of_semantic_categories, self.num_of_semantic_categories
        )
        sem_weights = sem_weights[category]

        frontier_map = self.get_frontier_map(map_features)
        structure = np.ones((3, 3))  # 8-connectivity
        labeled_map, num_features = label(frontier_map, structure=structure)

        frontiers = [
            frontiers.append(np.argwhere(labeled_map == i))
            for i in range(1, num_features + 1)
        ]

        sem_layers = map_features[
            0,
            2 * MC.NON_SEM_CHANNELS : 2 * MC.NON_SEM_CHANNELS
            + self.num_of_semantic_categories,
            :,
            :,
        ]
        r = 40
        frontier_scores = []
        for frontier in frontiers:
            center = int(frontier.mean(axis=0))
            local_map = sem_layers[
                :, center[0] - r : center[0] + r, center[1] - r : center[1] + r
            ]
            neighbor_classes = np.where(local_map.any(axis=(1, 2)))[0]

            if len(neighbor_classes) > 0:
                frontier_sem_score = np.mean(
                    sem_weights[neighbor_classes]
                ) / np.linalg.norm(
                    robot_position, center
                )  #! MyTODO: Geodesic distance
            else:
                frontier_sem_score = np.mean(sem_weights) / np.linalg.norm(
                    robot_position, center
                )  #! MyTODO: Geodesic distance
            frontier_scores.append(frontier_sem_score)

        # Select the frontier with the highest score
        best_frontier = np.argmax(frontier_scores)
        goal_map = np.zeros(map_features.shape[-2:], dtype=np.uint8)
        goal_map[tuple(best_frontier.T)] = 1
        return goal_map

    def reach_goal_if_in_map(
        self,
        map_features,
        goal_category,
        reject_visited_targets=False,
    ):
        """If the desired goal is in the semantic map, reach it."""
        batch_size, _, height, width = map_features.shape
        device = map_features.device

        goal_map = torch.zeros((batch_size, height, width), device=device)
        found_goal_current = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for e in range(batch_size):
            # if the category goal was not found previously
            if not found_goal_current[e]:
                # the category to navigate to
                category_map = map_features[
                    e, goal_category[e] + 2 * MC.NON_SEM_CHANNELS, :, :
                ]

                if reject_visited_targets:
                    # remove the target objects that the agent has already been close to
                    category_map = category_map * (
                        1 - map_features[e, MC.BLACKLISTED_TARGETS_MAP, :, :]
                    )
                # if the desired category is found with required constraints, set goal for navigation
                if (category_map == 1).sum() > 0:
                    goal_map[e] = category_map == 1
                    found_goal_current[e] = True
        return goal_map, found_goal_current

    def get_frontier_map(self, map_features):
        # Select unexplored area
        if self.exploration_strategy == "seen_frontier":
            frontier_map = (map_features[:, [MC.EXPLORED_MAP], :, :] == 0).float()
        elif self.exploration_strategy == "been_close_to_frontier":
            frontier_map = (map_features[:, [MC.BEEN_CLOSE_MAP], :, :] == 0).float()

        # Dilate explored area
        frontier_map = 1 - binary_dilation(
            1 - frontier_map, self.dilate_explored_kernel
        )

        # Select the frontier
        frontier_map = (
            binary_dilation(frontier_map, self.select_border_kernel) - frontier_map
        )
        return frontier_map

    def explore_otherwise(self, map_features, goal_map, found_goal):
        """Explore closest unexplored region otherwise."""
        frontier_map = self.get_frontier_map(map_features)
        batch_size = map_features.shape[0]
        for e in range(batch_size):
            if not found_goal[e]:
                goal_map[e] = frontier_map[e]

        return goal_map




class LanguageNavFrontierExplorationPolicy(nn.Module):
    """
    Policy to select high-level goals for Object Goal Navigation:
    go to object goal if it is mapped and explore frontier (closest
    unexplored region) otherwise.
    """

    def __init__(self, exploration_strategy: str, close_frontier_radius: float = 10.0, goto_past_pose=False):
        super().__init__()
        assert exploration_strategy in [
            "seen_frontier",
            "been_close_to_frontier",
            "fixed",
        ]
        self.exploration_strategy = exploration_strategy

        self.dilate_explored_kernel = nn.Parameter(
            torch.from_numpy(skimage.morphology.disk(10))
            .unsqueeze(0)
            .unsqueeze(0)
            .float(),
            requires_grad=False,
        )
        self.select_border_kernel = nn.Parameter(
            torch.from_numpy(skimage.morphology.disk(1))
            .unsqueeze(0)
            .unsqueeze(0)
            .float(),
            requires_grad=False,
        )

        self.close_frontier_radius = close_frontier_radius
        self.goto_past_pose = goto_past_pose 


    def reach_single_category(
        self, map_features, category, reject_visited_targets, location=None, instance_memory=None, num_sem_categories=None, timestep=None
    ):
        # if the goal is found, reach it
        goal_map, found_goal, goal_pose = self.reach_goal_if_in_map(
            map_features, category, reject_visited_targets=reject_visited_targets, instance_memory=instance_memory, num_sem_categories=num_sem_categories, location=location
        )
        # otherwise, do frontier exploration
        goal_map = self.explore_otherwise(map_features, goal_map, found_goal, location, timestep=timestep)
        if found_goal[0]==True:
            logger.info(f"Found goal category {category[0]} in the map, returning it as goal.")
        else:
            logger.info(
                f"Did not find goal category {category[0]} in the map, exploring frontier instead. {found_goal[0]}"
            )

        return goal_map, found_goal, goal_pose

    def forward(
        self,
        map_features,
        object_category=None,
        reject_visited_targets=False,
        location=None,
        instance_memory=None,
        num_sem_categories=None,
        timestep=None,
    ):
        """
        Arguments:
            map_features: semantic map features of shape
             (batch_size, 9 + num_sem_categories, M, M)
            object_category: object goal category
        Returns:
            goal_map: binary map encoding goal(s) of shape (batch_size, M, M)
            found_goal: binary variables to denote whether we found the object
            goal category of shape (batch_size,)
        """
        assert object_category is not None

        # Here, the goal is specified by a single object
        return self.reach_single_category(
            map_features, object_category, reject_visited_targets, location=location, instance_memory=instance_memory, num_sem_categories=num_sem_categories, timestep=timestep
        )

    def reach_goal_if_in_map(
        self,
        map_features,
        goal_category,
        reject_visited_targets=False,
        instance_memory=None,
        num_sem_categories=None,
        location=None,
    ):
        """If the desired goal is in the semantic map, reach it."""
        logger.debug(f"Searching for goal category {goal_category[0]} in the map.")
        goal_pose = None

        batch_size, _, height, width = map_features.shape
        device = map_features.device

        goal_map = torch.zeros((batch_size, height, width), device=device)
        found_goal_current = False

        # if the category goal was not found previously
        if not found_goal_current:
            # the category to navigate to
            category_map = map_features[
                goal_category + 2 * MC.NON_SEM_CHANNELS, :, :
            ]

            if reject_visited_targets:
                # remove the target objects that the agent has already been close to
                category_map = category_map * (
                    1 - map_features[MC.BLACKLISTED_TARGETS_MAP, :, :]
                )
            # if the desired category is found with required constraints, set goal for navigation
            if (category_map == 1).sum() > 0:
                logger.debug("Found goal category in the map.")

                found_goal_current = True

                if self.goto_past_pose:
                    logger.debug("Returning past pose for the goal.")
                    goal_map[0], goal_pose = self.get_goal_map_for_category(goal_category[0], instance_memory, map_features, num_sem_categories, location)
                else:
                    logger.debug("Returning goal cells as the goal map.")
                    goal_map[0] = category_map == 1
            else:
                logger.debug("Did not find goal category in the map.")
        return goal_map, found_goal_current, goal_pose
    
    def get_goal_map_for_category(self, category, instance_memory, local_map, num_sem_categories, location, mode="closest_pose"):
        """
        Get the goal map for a specific category from the instance memory, by checking adding all the poses.
        location is (y,x) in the grid. local_map[y,x] is the value for the cell robot is currently at.
        """
        #! myTODO: Instead of this, we can return all poses and all possible goals, and later convert them back to best location for that pose, and then choose the closest.
        instance_map = local_map[0][
            MC.NON_SEM_CHANNELS
            + num_sem_categories : MC.NON_SEM_CHANNELS
            + 2 * num_sem_categories,
            :,
            :,
        ]

        best_view = None
        best_inst_key = None
        best_metric = 0
        for (inst_key, inst) in instance_memory.instance_views[0].items():
            if inst.category_id not in category:
                continue
            views = inst.instance_views
            max_coverage_view = np.argmax([view.object_coverage for view in views])
            if mode == "best_pose":
                if views[max_coverage_view].object_coverage > best_metric:
                    best_metric = views[max_coverage_view].object_coverage
                    best_view = views[max_coverage_view]
                    best_inst_key = inst_key
            elif mode == "closest_pose":
                #! myTODO: This doesn't work with blacklisting logic.
                if np.linalg.norm(views[max_coverage_view].pose[:2]- location.cpu()) < best_metric or best_metric == 0:
                    best_metric = np.linalg.norm(views[max_coverage_view].pose[:2]- location.cpu())
                    best_view = views[max_coverage_view]
                    best_inst_key = inst_key

        pose = best_view.pose
        curr_x, curr_y, curr_o, gy1, _, gx1, _ = pose.tolist()

        inst_map_idx = instance_map == best_inst_key
        inst_map_idx = torch.argmax(torch.sum(inst_map_idx, axis=(1, 2)))
        goal_map = (instance_map[inst_map_idx] == best_inst_key).to(torch.float)
        
        #! The output goal pose is the actual index in global map
        goal_pose = [[curr_o, curr_y * 100.0 / 5 , curr_x * 100.0 / 5]]

        logger.debug(f">>> Goal instance {best_inst_key} best past pose is: {goal_pose}, with coverage {best_metric}. Returning goal_pose in addition to goal_map.")
        return goal_map, goal_pose
