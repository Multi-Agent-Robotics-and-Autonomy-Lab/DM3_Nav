# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import cv2
import skfmm
import numpy as np
from numpy import ma
from typing import List
import matplotlib.cm as cm
from bresenham import bresenham
from matplotlib.colors import Normalize
from home_robot.utils.visualization import visualize_map



def convert_to_cmap(subset):
    final_img = np.flipud(subset.copy())
    unique_vals = np.unique(final_img)
    if len(unique_vals) < 2:
        return np.zeros(final_img.shape + (3,), dtype=np.uint8)
    second_max = unique_vals[-2]
    final_img[final_img == np.max(final_img)] = second_max + 1

    vmin, vmax = np.nanmin(final_img), np.nanmax(final_img)
    norm = Normalize(vmin=vmin, vmax=vmax)

    cmap = cm.get_cmap("plasma")

    rgba_img = cmap(norm(final_img))
    rgb_img = (rgba_img[:, :, :3] * 255).astype(np.uint8)
    return rgb_img


class FMMPlanner:
    """
    Fast Marching Method Planner.
    This is just the core FMM logic.
    """

    def __init__(
        self,
        log,
        stop_distance,
        scale: int = 1,
        step_size: int = 5,
        print_images=True,
        debug=False,
        real_world=False
    ):
        """
        Arguments:
            traversible: (M + 1, M + 1) binary map encoding traversible regions
            scale: map scale
            step_size: maximum distance of the short-term goal selected by the
             planner
            vis_dir: folder where to dump visualization
        """
        self.print_images = print_images
        self.scale = scale
        self.step_size = step_size
        self.stop_distance = stop_distance

        self.du = int(self.step_size / (self.scale * 1.0))
        self.fmm_dist = None
        # self.debug = True
        self.debug = debug
        # self.goal_map = None
        self.log = log
        self.real_world = real_world
    
    def set_vis_dir(self, vis_dir):
        self.vis_dir = vis_dir

    def set_goal(self, goal):
        """Set planner goal. Goal should be of size 2, containing x and y positions."""
        traversible_ma = ma.masked_values(self.traversible * 1, 0)
        goal_x, goal_y = int(goal[0] / (self.scale * 1.0)), int(
            goal[1] / (self.scale * 1.0)
        )

        traversible_ma[goal_x, goal_y] = 0
        dd = skfmm.distance(traversible_ma, dx=1)
        dd = ma.filled(dd, np.max(dd) + 1)
        self.fmm_dist = dd
        return

    def set_multi_goal(
        self,
        goal_map: np.ndarray,
        traversible,
        timestep: int = 0,
        dd: np.ndarray = None,
        map_downsample_factor: float = 1.0,
        map_update_frequency: int = 1,
        number="",
        vis_postfix=""
    ):
        """Set long-term goal(s) used to compute distance from a binary
        goal map.
        dd: distance map for when we want to reuse previously computed ones (instead of updating at each step)
        map_update_frequency: skfmm.distance call made every n steps
        map_downsample_factor: 1 for no downsampling, 2 for halving both image dimensions.
        """
        assert map_downsample_factor >= 1.0
        if self.scale != 1.0:
            traversible = cv2.resize(
                traversible,
                (traversible.shape[1] // self.scale, traversible.shape[0] // self.scale),
                interpolation=cv2.INTER_NEAREST,
            )
            traversible = np.rint(traversible).astype(np.uint8)

        if map_downsample_factor > 1.0:
            l, w = self.traversible.shape
            traversible = cv2.resize(
                traversible,
                dsize=(int(l / map_downsample_factor), int(w / map_downsample_factor)),
            )
            print(f"Downsampling goal and traversible maps {map_downsample_factor}x.")
            goal_map_copy = goal_map.copy()
            goal_map = cv2.resize(
                goal_map,
                dsize=(int(l / map_downsample_factor), int(w / map_downsample_factor)),
                interpolation=cv2.INTER_NEAREST,
            )

            if goal_map.sum() == 0:
                # dilating goal map so as to not lose pixels when resizing
                kernel = np.ones((2, 2), np.uint8)
                goal_map = cv2.dilate(goal_map_copy, kernel, iterations=1)
                goal_map = cv2.resize(
                    goal_map,
                    dsize=(
                        int(l / map_downsample_factor),
                        int(w / map_downsample_factor),
                    ),
                    interpolation=cv2.INTER_NEAREST,
                )

        traversible_ma = ma.masked_values(traversible * 1, 0)
        traversible_ma[goal_map == 1] = 0

        # This is where we actually call the FMM algorithm!!
        # It will compute the distance from each traversible point to the goal.
        if (timestep - 1) % map_update_frequency == 0 or dd is None:
            dd = skfmm.distance(traversible_ma, dx=1 * map_downsample_factor)
            dd = ma.filled(dd, np.max(dd) + 1)
            if self.debug:
                print(f"Computing skfmm.distance (timestep: {timestep})")
        else:
            if self.debug:
                print(f"Reusing previous skfmm.distance value (timestep: {timestep})")

        if map_downsample_factor > 1.0:
            dd = cv2.resize(dd, (l, w))  # upsampling

        self.fmm_dist = dd
        # self.goal_map = goal_map

        if self.print_images and timestep != 0 and len(number) > 0:
            r, c = traversible.shape  # for visualizing (downsampled) traversible map
            dist_vis = np.zeros((r, c * 3))
            dist_vis[:, :c] = np.flipud(traversible)
            dist_vis[:, c : 2 * c] = np.flipud(goal_map)
            dist_vis[:, 2 * c :] = np.flipud(self.fmm_dist / self.fmm_dist.max())

            output_name = f"{timestep}_{number}.distance_to_goal{vis_postfix}.png"
            cv2.imwrite(
                os.path.join(self.vis_dir, output_name),
                (dist_vis * 255).astype(int),
            )
        return dd

    def visualize_get_short_term_goal(self, vis_list, timestep, prefix="", postfix=""):
        sub_h, sub_w = vis_list[0].shape
        dist_vis = np.zeros((sub_h * 2, sub_w * 2, 3))
        dist_vis[:sub_h, :sub_w] = convert_to_cmap(vis_list[0])
        dist_vis[:sub_h, sub_w : 2 * sub_w, :] = vis_list[1]
        dist_vis[sub_h:, :sub_w] = convert_to_cmap(vis_list[2])
        dist_vis[sub_h:, sub_w:] = convert_to_cmap(vis_list[3])

        cv2.imwrite(
            os.path.join(
                self.vis_dir, f"{prefix}{timestep}_11.get_stg_details{postfix}.png"
            ),
            (dist_vis).astype(int),
        )

    def filter_unreachable_goals(
        self, subset, mask, robot_pos, obstacle_mask, ray_thickness=1
    ):
        safe_mask = np.zeros_like(mask, dtype=bool)
        candidates = np.argwhere(mask)

        for x, y in candidates:
            line_coords = list(bresenham(robot_pos[0], robot_pos[1], x, y))[1:-1]
            reachable = True
            for r, c in line_coords:
                # Check neighborhood around this point
                for dx in range(-ray_thickness, ray_thickness + 1):
                    for dy in range(-ray_thickness, ray_thickness + 1):
                        nr, nc = r + dx, c + dy
                        if 0 <= nr < subset.shape[0] and 0 <= nc < subset.shape[1]:
                            if obstacle_mask[nr, nc]:
                                reachable = False
                                break
                    if not reachable:
                        break
                if not reachable:
                    break

            if reachable:
                safe_mask[x, y] = True

            # line_coords = [(r, c) for r, c in line_coords if 0 <= r < subset.shape[0] and 0 <= c < subset.shape[1]]

            # if not any(obstacle_mask[r, c] for r, c in line_coords):
            #     safe_mask[x, y] = True

        masked_subset = np.copy(subset)
        masked_subset[np.logical_and(mask, ~safe_mask)] = np.max(subset)
        return masked_subset

    def get_short_term_goal(self, state: List[float], timestep=0, prefix="", postfix=""):
        for radius in range(2, self.step_size+1)[::-1]:
            stg_x, stg_y, reachable, stop = self.get_short_term_goal_util(state, radius, timestep, prefix=prefix, postfix=postfix + f"_step{radius}")
            if reachable:
                break
        return stg_x, stg_y, reachable, stop

    def get_short_term_goal_util(self, state: List[float], radius, timestep=0, prefix="", postfix=""):
        """Compute the short-term goal closest to the current state.

        Arguments:
            state: current location
        """
        scale = self.scale * 1.0
        state = [x / scale for x in state]
        dx, dy = state[0] - int(state[0]), state[1] - int(state[1])
        mask = FMMPlanner.get_mask(
            dx, dy, scale, radius, self.step_size
        )
        # dist_mask = FMMPlanner.get_dist(dx, dy, scale, step_size)

        state = [int(x) for x in state]
        # max_value = self.fmm_dist.shape[0] ** 2
        max_value = np.max(self.fmm_dist)

        #! max in self.fmm_dist actually means obstacle. (it is set to: actual_max+1)
        #! Hence, we also set padded cells to this maximum.
        dist = np.pad(
            self.fmm_dist,
            self.du,
            "constant",
            # constant_values=self.fmm_dist.shape[0] ** 2,
            constant_values=max_value,
        )
        subset = dist[
            state[0] : state[0] + 2 * self.du + 1, state[1] : state[1] + 2 * self.du + 1
        ]
        obstacle_mask = subset == subset.max()

        assert (
            subset.shape[0] == 2 * self.du + 1 and subset.shape[1] == 2 * self.du + 1
        ), "Planning error: unexpected subset shape {}".format(subset.shape)

        vis_list = []
        vis_list.append(subset.copy())

        subset *= mask
        # subset += (1 - mask) * self.fmm_dist.shape[0] ** 2
        subset += (1 - mask) * max_value

        vis_list.append(np.flipud(mask[..., None].copy() * 255))
        vis_list.append(subset.copy())

        stop = subset[self.du, self.du] < self.stop_distance and obstacle_mask[self.du, self.du] != True
        self.log.debug(
            f"[FMM] Distance to fmm navigable goal pt (subset[self.du, self.du]) = {subset[self.du, self.du]}"
        )
        self.log.debug(f"self.stop_distance {self.stop_distance}")
        self.log.debug(f"stop {stop}")

        subset -= subset[self.du, self.du]
        reachable_subset = self.filter_unreachable_goals(
            subset, mask, (self.du, self.du), obstacle_mask, ray_thickness=1 if self.real_world else 0
        )
        # ratio1 = subset / dist_mask
        # subset[ratio1 < -1.5] = 1

        # Method 1
        movement_threshold = -radius / 2
        progress_mask = np.logical_and(mask.astype(bool), reachable_subset < movement_threshold)

        if np.any(progress_mask):
            obstacle_dist = cv2.distanceTransform(
                (1 - obstacle_mask).astype(np.uint8), cv2.DIST_L2, 5
            ) + 1

            # First, find the best goal by FMM distance
            candidate_subset = np.where(progress_mask, reachable_subset, max_value)
            best_x, best_y = np.unravel_index(np.argmin(candidate_subset), candidate_subset.shape)


            min_clearance_threshold = 9 if self.real_world else 3# cells
            if obstacle_dist[best_x, best_y] < min_clearance_threshold:
                # Too close to obstacle — pick safest among progress candidates
                clearance = np.where(progress_mask, obstacle_dist, 0)
                stg_x, stg_y = np.unravel_index(np.argmax(clearance), clearance.shape)
            else:
                # Enough clearance — pick closest to goal
                stg_x, stg_y = best_x, best_y

            vis_list.append(obstacle_dist * progress_mask)
            reachable = True
        else:
            vis_list.append(reachable_subset)
            stg_x, stg_y = self.du, self.du
            reachable = stop

        # Method 2
        # vis_list.append(reachable_subset.copy())
        # stg_x, stg_y = np.unravel_index(np.argmin(reachable_subset), subset.shape)
        # # Rechable if stg distance is less than current location (negative).
        # reachable = (subset[stg_x, stg_y] < -0.0001) or stop

        self.log.debug(
            f"subset[stgx, stgy] = {subset[stg_x, stg_y]}"
        )


        if self.print_images:
            self.visualize_get_short_term_goal(vis_list, timestep, prefix, postfix)

        return (
            (stg_x + state[0] - self.du) * scale,
            (stg_y + state[1] - self.du) * scale,
            reachable,
            stop,
        )

    @staticmethod
    def get_mask(sx, sy, scale, radius, step_size):
        """Set everything in a circle around the agent to 1; else set to zero"""
        min_radius = (radius - 1) ** 2
        size = int(step_size // scale) * 2 + 1
        mask = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                cond1 = (
                    ((i + 0.5) - (size // 2 + sx)) ** 2
                    + ((j + 0.5) - (size // 2 + sy)) ** 2
                ) <= radius**2
                cond2 = (
                    ((i + 0.5) - (size // 2 + sx)) ** 2
                    + ((j + 0.5) - (size // 2 + sy)) ** 2
                ) > min_radius
                if cond1 and cond2:
                    mask[i, j] = 1
        mask[size // 2, size // 2] = 1
        return mask

    @staticmethod
    def get_dist(sx, sy, scale, step_size):
        size = int(step_size // scale) * 2 + 1
        mask = np.zeros((size, size)) + 1e-10
        for i in range(size):
            for j in range(size):
                if (
                    ((i + 0.5) - (size // 2 + sx)) ** 2
                    + ((j + 0.5) - (size // 2 + sy)) ** 2
                ) <= step_size**2:
                    mask[i, j] = max(
                        5,
                        (
                            ((i + 0.5) - (size // 2 + sx)) ** 2
                            + ((j + 0.5) - (size // 2 + sy)) ** 2
                        )
                        ** 0.5,
                    )
        return mask
