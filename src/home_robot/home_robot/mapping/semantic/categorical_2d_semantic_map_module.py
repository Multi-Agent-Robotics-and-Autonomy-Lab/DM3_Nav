# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import torch
import logging
import matplotlib
import numpy as np
import torch.nn as nn
import skimage.morphology
from skimage.draw import disk
import matplotlib.pyplot as plt
from bresenham import bresenham
from collections import defaultdict
import home_robot.utils.pose as pu
from torch import IntTensor, Tensor
import home_robot.utils.depth as du
from torch.nn import functional as F
import home_robot.utils.rotation as ru
from typing import Optional, Tuple, List
import home_robot.mapping.map_utils as mu
from home_robot.mapping.semantic.constants import MapConstants as MC
from home_robot.utils.spot import draw_circle_segment, fill_convex_hull
from home_robot.mapping.semantic.instance_tracking_modules import InstanceMemory

# For debugging input and output maps - shows matplotlib visuals
debug_maps = False
matplotlib.use("Agg")

class UpdateStateLogger(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        # modify the message however you want
        return f"[UPDATE_STATE] {msg}", kwargs

def compute_known_cells_map(
    obstacle_map_tensor, robot_pos, max_range, gaze_width, num_beams=360
):
    device = obstacle_map_tensor.device
    H, W = obstacle_map_tensor.shape
    
    # Dilate obstacles
    obstacle_map = (obstacle_map_tensor >= 1).float()
    kernel = torch.ones(1, 1, 7, 7, device=device)
    obstacle_map = torch.nn.functional.conv2d(
        obstacle_map.unsqueeze(0).unsqueeze(0),
        kernel,
        padding=3
    ).squeeze() > 0
    
    cx, cy = robot_pos
    center_angle = 0.0
    half_fov_rad = np.radians(gaze_width / 2.0)
    
    angles = torch.linspace(
        center_angle - half_fov_rad,
        center_angle + half_fov_rad,
        num_beams,
        device=device
    )
    
    dx = torch.cos(angles)
    dy = torch.sin(angles)
    
    max_steps = int(max_range * 1.5)
    t = torch.linspace(0, max_range, max_steps, device=device)
    
    x_coords = cx + dx.unsqueeze(1) * t.unsqueeze(0)
    y_coords = cy + dy.unsqueeze(1) * t.unsqueeze(0)
    
    x_int = torch.round(x_coords).long()
    y_int = torch.round(y_coords).long()
    
    valid_mask = (x_int >= 0) & (x_int < W) & (y_int >= 0) & (y_int < H)
    
    # just to avoid index issues
    x_clamped = torch.clamp(x_int, 0, W - 1)
    y_clamped = torch.clamp(y_int, 0, H - 1)
    
    obstacle_at_point = obstacle_map[x_clamped, y_clamped] | ~valid_mask
    
    cumsum_obstacles = torch.cumsum(obstacle_at_point.float(), dim=1)
    visible_cells = valid_mask & (cumsum_obstacles <= 1)
    
    visible_x = x_int[visible_cells]
    visible_y = y_int[visible_cells]
    
    known_map = torch.zeros((H, W), dtype=torch.float32, device=device)
    known_map[visible_x, visible_y] = 1
    
    return known_map

def get_fp_exp_pred(self, fp_map_pred):
    if self.exploration_type == "raycast":
        fp_exp_pred = compute_known_cells_map(
            fp_map_pred,
            (0, fp_map_pred.shape[1] // 2),
            self.gaze_distance * 100 / self.resolution,
            self.gaze_width,
            num_beams=360,
        )
        return fp_exp_pred
    elif self.exploration_type == "default":
        fp_exp_pred = fp_map_pred.copy()
        fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
    elif self.exploration_type == "hull":
        fp_exp_pred = fp_map_pred.copy()
        fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
        fp_exp_pred = fp_exp_pred.clip(0, 1)
        # set the current agent position as 1
        fp_exp_pred[0, fp_exp_pred.shape[-1] // 2] = 1

        # fill convex hull
        filled = fill_convex_hull(fp_exp_pred.cpu())
        fp_exp_pred = torch.tensor(filled)

    # uses a fixed cone infront of the camerea
    elif self.exploration_type == "gaze":
        fp_exp_pred = torch.zeros_like(fp_map_pred)
        view_image = torch.zeros(fp_map_pred.shape[-2:])
        # get the desired radius in cells
        dist = self.gaze_distance * 100 / self.resolution
        view_image = draw_circle_segment(
            view_image, (0, fp_exp_pred.shape[-1] // 2), dist, 0, self.gaze_width
        )
        fp_exp_pred = view_image
    # uses depth point projections but limits the fov and distance using the code
    elif self.exploration_type == "gaze_projected":
        fp_exp_pred = fp_map_pred.copy()
        fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
        view_image = torch.zeros(fp_map_pred.shape[-2:])
        # get the desired radius in cells
        dist = self.gaze_distance * 100 / self.resolution
        view_image = (
            draw_circle_segment(
                view_image,
                (0, fp_exp_pred.shape[-1] // 2),
                dist,
                0,
                self.gaze_width,
            )
            / 255
        )
        fp_exp_pred *= view_image.to(fp_exp_pred.device)
    else:
        raise Exception("not implemented")
    return fp_exp_pred


class Categorical2DSemanticMapModule(nn.Module):
    """
    This class is responsible for updating a dense 2D semantic map with one channel
    per object category, the local and global maps and poses, and generating
    map features — it is a stateless PyTorch module with no trainable parameters.

    Map proposed in:
    Object Goal Navigation using Goal-Oriented Semantic Exploration
    https://arxiv.org/pdf/2007.00643.pdf
    https://github.com/devendrachaplot/Object-Goal-Navigation
    """

    # If true, display point cloud visualizations using Open3d
    debug_mode = False

    def __init__(
        self,
        device,
        camera_params,
        num_sem_categories: int,
        map_size_cm: int,
        map_resolution: int,
        vision_range: int,
        been_close_to_radius: int,
        global_downscaling: int,
        du_scale: int,
        cat_pred_threshold: float,
        exp_pred_threshold: float,
        map_pred_threshold: float,
        min_depth: float = 0.5,
        max_depth: float = 3.5,
        min_obs_height_cm: int = 25,
        target_blacklisting_radius: int = None,
        record_instance_ids: bool = False,
        instance_memory: Optional[InstanceMemory] = None,
        max_instances: int = 0,
        dilation_for_instances: int = 5,
        padding_for_instance_overlap: int = 5,
        exploration_type="default",
        agent_cell_radius: int = 1,
        print_images: bool = False,
        log=None,
        real_world=False,
        start_obs_dilation=0,
        log_odds_occ=0.5,
        log_odds_free=0.8,
        max_log_odds=5.0,
    ):
        """
        Arguments:
            frame_height: first-person frame height
            frame_width: first-person frame width
            camera_height: camera sensor height (in metres)
            hfov: horizontal field of view (in degrees)
            num_sem_categories: number of semantic segmentation categories
            map_size_cm: global map size (in centimetres)
            map_resolution: size of map bins (in centimeters)
            vision_range: diameter of the circular region of the local map
             that is visible by the agent located in its center (unit is
             the number of local map cells)
            been_close_to_radius: radius (in centimeters) of been close to region
            target_blacklisting_radius: radius (in centimeters) of region
             around target that will be blacklisted (if invalid target)
            global_downscaling: ratio of global over local map
            du_scale: frame downscaling before projecting to point cloud
            cat_pred_threshold: number of depth points to be in bin to
             classify it as a certain semantic category
            exp_pred_threshold: number of depth points to be in bin to
             consider it as explored
            map_pred_threshold: number of depth points to be in bin to
             consider it as obstacle
            min_obs_height_cm: minimum height of obstacles (in centimetres)
            record_instance_ids: whether to record instance ids in the 2d semantic map
            exploration_type: how to define explored area
            gaze_width: hfov in degrees for use with the gaze based exploration
            gaze_distance: depth to be considered explored with gaze based exploration
        """
        super().__init__()

        self.device = device

        self.screen_h = camera_params.height
        self.screen_w = camera_params.width
        self.hfov = camera_params.hfov
        self.vfov = camera_params.vfov
        self.camera_matrix = camera_params.camera_matrix  # Namespace(xc, zc, f)
        self.agent_height = camera_params.camera_height * 100.0

        self.gaze_width = self.hfov
        self.gaze_distance = camera_params.max_depth

        self.num_sem_categories = num_sem_categories

        self.resolution = map_resolution
        self.global_map_size_cm = map_size_cm
        self.global_downscaling = global_downscaling
        self.local_map_size_cm = self.global_map_size_cm // self.global_downscaling
        self.global_map_size = self.global_map_size_cm // self.resolution
        self.local_map_size = self.local_map_size_cm // self.resolution
        self.xy_resolution = self.z_resolution = map_resolution
        self.vision_range = vision_range
        self.been_close_to_radius = been_close_to_radius
        if target_blacklisting_radius is not None:
            self.target_blacklisting_radius = target_blacklisting_radius
        self.du_scale = du_scale
        self.cat_pred_threshold = cat_pred_threshold
        self.exp_pred_threshold = exp_pred_threshold
        self.map_pred_threshold = map_pred_threshold

        self.max_depth = max_depth * 100.0
        self.min_depth = min_depth * 100.0
        self.max_voxel_height = int(360 / self.z_resolution)
        self.min_voxel_height = int(-40 / self.z_resolution)
        self.min_obs_height_cm = min_obs_height_cm
        self.min_obstacle_height = int(
            self.min_obs_height_cm / self.z_resolution - self.min_voxel_height
        )

        self.max_obstacle_height = int(
            (self.agent_height + 1) / self.z_resolution - self.min_voxel_height
        )
        self.shift_loc = [self.vision_range * self.xy_resolution // 2, 0, np.pi / 2.0]

        # For cleaning up maps
        self.record_instance_ids = record_instance_ids
        self.padding_for_instance_overlap = padding_for_instance_overlap
        self.dilation_for_instances = dilation_for_instances
        self.instance_memory = instance_memory
        self.max_instances = max_instances
        self.exploration_type = exploration_type
        self.agent_cell_radius = agent_cell_radius
        self.vis_dir = None
        self.timestep = 0

        self.avg_pooling_layer = nn.AvgPool2d(self.du_scale)
        self._disk_masks = {}
        self.print_images = print_images
        self.log = UpdateStateLogger(log, None)
        self.mask_stairs = real_world
        self.real_world = real_world
        self.start_obs_dilation = start_obs_dilation


        self.log_odds_occ = log_odds_occ          # 0.8
        self.log_odds_free = log_odds_free         # 0.4
        self.max_log_odds = max_log_odds           # 10.0
        self._prev_local_pose = None


    @torch.no_grad()
    def forward(
        self,
        obs: Tensor,
        state,
        instance_scores: Tensor,
        category_scores,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, IntTensor, Tensor]:
        """Update maps and poses with a sequence of observations and generate map
        features at each time step.

        Arguments:
            seq_obs: sequence of frames containing (RGB, depth, segmentation)
             of shape (3 + 1 + num_sem_categories,
             frame_height, frame_width)
            seq_dones: binary flag that indicate episode restarts
            seq_camera_poses: sequence of (4, 4) extrinsic camera matrices
            init_local_map: initial local map before any updates of shape
             (MC.NON_SEM_CHANNELS + num_sem_categories, M, M)
            init_global_map: initial global map before any updates of shape
             (MC.NON_SEM_CHANNELS + num_sem_categories, M * ds, M * ds)
            init_local_pose: initial local pose before any updates of shape
             (3)
            init_global_pose: initial global pose before any updates of shape
             (3)
            init_lmb: initial local map boundaries of shape (4)
            init_origins: initial local map origins of shape (3)

        Returns:
            seq_map_features: sequence of semantic map features of shape
             (2 * MC.NON_SEM_CHANNELS + num_sem_categories, M, M)
            final_local_map: final local map after all updates of shape
             (MC.NON_SEM_CHANNELS + num_sem_categories, M, M)
            final_global_map: final global map after all updates of shape
             (MC.NON_SEM_CHANNELS + num_sem_categories, M * ds, M * ds)
            seq_local_pose: sequence of local poses of shape
             (3)
            seq_global_pose: sequence of global poses of shape
             (3)
            seq_lmb: sequence of local map boundaries of shape
             (4)
            seq_origins: sequence of local map origins of shape
             (3)
        """
        self.log.debug(f"Updating maps and current position")


        state.local_map = self._update_local_map_and_pose(
            obs,
            state.local_pose,
            state.local_map,
            state.origins,
            state.lmb,
            instance_scores,
            category_scores,
        )
        # updates in place
        self._update_global_map_and_pose(state)

        state.local_map, state.local_pose, state.lmb, state.origins = mu.get_local_parameters_from_global_pose(
            state.global_map,
            state.global_pose,
            state.map_size_parameters,
        )
        self._prev_local_pose = state.local_pose.clone()

        self.log.debug(f"Updated pose: global={state.global_pose.tolist()}, local={state.local_pose.tolist()}, lmb: {state.lmb.tolist()}")
        self.log.debug(f"Updated loc: global={state.global_loc}")
        return state
        

    def _aggregate_instance_map_channels_per_category(
        self, curr_map, num_instance_channels
    ):
        """Aggregate map channels for instances (input: one binary channel per instance in [0, 1])
        by category (output: one channel per category containing instance IDs).
        curr_map is the map created using current observation before merging with global map.
        """

        # Called init, because it is not aggregated per semantic channel. No "id"s yet.
        temp_instance_map = curr_map[
            MC.NON_SEM_CHANNELS
            + self.num_sem_categories : MC.NON_SEM_CHANNELS
            + self.num_sem_categories
            + num_instance_channels,
        ]
        # now we add all instances with the same category to a single channel. Again called temp, because ids are temp_ids and not global ids. Note that 0 means nothing here.
        aggregated_temp_instance_map = torch.zeros(
            self.num_sem_categories,
            curr_map.shape[1],
            curr_map.shape[2],
            device=curr_map.device,
            dtype=curr_map.dtype,
        )

        if num_instance_channels > 0:
            # create category id to instance id list mapping
            category_id_to_temp_id_list = defaultdict(list)
            # loop over unprocessed instances
            unprocessed_views = self.instance_memory.unprocessed_views
            for temp_id, instance in unprocessed_views.items():
                category_id_to_temp_id_list[instance.category_id].append(temp_id)

            for category_id in category_id_to_temp_id_list.keys():
                temp_ids = category_id_to_temp_id_list[category_id]
                instance_map_onehot = temp_instance_map[[i - 1 for i in temp_ids]]
                
                if len(temp_ids) > 1:
                    binary_maps = (instance_map_onehot > 1e-5)
                    merged = set()
                    for j in range(len(temp_ids)):
                        if j in merged:
                            continue
                        for k in range(j + 1, len(temp_ids)):
                            if k in merged:
                                continue
                            
                            intersection = (binary_maps[j] & binary_maps[k]).sum().item()
                            smaller = min(binary_maps[j].sum().item(), binary_maps[k].sum().item())
                            
                            if smaller > 0 and intersection / smaller > 0.3:
                                score_j = unprocessed_views[temp_ids[j]].score
                                score_k = unprocessed_views[temp_ids[k]].score
                                
                                winner, loser = (j, k) if score_j >= score_k else (k, j)
                                
                                instance_map_onehot[winner] = torch.maximum(instance_map_onehot[winner], instance_map_onehot[loser])
                                instance_map_onehot[loser] = 0
                                merged.add(loser)
                                
                                if loser == j:
                                    break

                instance_map_onehot = torch.cat(
                    (1e-5 * torch.ones_like(instance_map_onehot[:1]), instance_map_onehot),
                    dim=0,
                )
                # Each entry is either a temp id, or 0 for no instance
                category_instance_map = instance_map_onehot.argmax(dim=0)
                idx_to_temp_id = [0] + temp_ids
                category_instance_map = torch.tensor(
                    idx_to_temp_id, device=category_instance_map.device
                )[category_instance_map]
                #! 1
                aggregated_temp_instance_map[category_id - 1] = category_instance_map
                # self.log.debug(f"Aggregated category {category_id} with temp instance ids {temp_ids}")

        assert not curr_map[
            MC.NON_SEM_CHANNELS + self.num_sem_categories + num_instance_channels :,
        ].any()
        assert (
            curr_map[
                MC.NON_SEM_CHANNELS + self.num_sem_categories + num_instance_channels :,
            ].shape[0]
            == 0
        )

        curr_map = torch.cat(
            (
                curr_map[: MC.NON_SEM_CHANNELS + self.num_sem_categories],
                aggregated_temp_instance_map,
            ),
            dim=0,
        )

        return curr_map

    def draw_line(self, matrix, x1, y1, x2, y2, padding=1):
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if x1 < x2:
            sx = 1
        else:
            sx = -1
        if y1 < y2:
            sy = 1
        else:
            sy = -1

        err = dx - dy

        while True:
            for i in range(-padding, padding + 1):
                for j in range(-padding, padding + 1):
                    x = x1 + i
                    y = y1 + j
                    if 0 <= x < matrix.shape[2] and 0 <= y < matrix.shape[1]:
                        matrix[0, y, x] = 1  # Set x-value to 1
                        matrix[1, y, x] = 1  # Set y-value to 1

            if x1 == x2 and y1 == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

    def get_stairs(self, voxels, visible_ground):
        """
        Heuristic to mark stair-like regions as obstacles
        based on absence of ground in visible region.
        """
        if self.mask_stairs:
            H, W = voxels.shape[:2]
            return (
                torch.zeros(H, W, dtype=torch.uint8, device=voxels.device),
                torch.zeros(H, W, dtype=torch.uint8, device=voxels.device)
            )

        visible_ground = visible_ground.cpu().numpy()
        #! myTODO: Hardcoded. Fix this later
        visible_ground[80:] = 0

        #! myTODO: x is hardcoded. This means if you don't see anything with z between -x to x (which right now is min_obs_height cm) in a location, this means it is a downward stair.
        x = int(10 / self.z_resolution)
        ground_plane = voxels[
            :, :, -4 * x - self.min_voxel_height : x - self.min_voxel_height
        ]
        ground_plane = ground_plane.sum(axis=2).cpu().numpy()
        ground_plane = np.where(ground_plane >= 1, 1, 0).astype(np.uint8)

        # ground_points = np.column_stack(np.nonzero(ground_plane))
        # points = ground_points[:, [1, 0]].astype(np.float32)  # (x, y)
        # polygon = alpha_shape(points, alpha=0.005)
        # ground_plane = rasterize_polygon(polygon, ground_plane.shape)
        selem = np.ones((7, 7), dtype=np.uint8)
        ground_plane = cv2.dilate(ground_plane, selem, iterations=1)

        X, Y = ground_plane.shape
        robot_x = 0
        robot_y = Y // 2

        xx, yy = np.meshgrid(np.arange(X), np.arange(Y), indexing="ij")
        dx = (xx - robot_x) * self.xy_resolution
        dy = (yy - robot_y) * self.xy_resolution

        dx[dx == 0] = 1e-6  # avoid division by zero

        # Convert FOVs to radians
        hfov_rad = np.deg2rad(self.hfov)
        vfov_rad = np.deg2rad(self.vfov)

        horizontal_angle = np.arctan2(dy, dx)
        within_hfov = np.abs(horizontal_angle) <= (hfov_rad / 2)

        min_visible_dist = (
            int(self.agent_height / np.tan(vfov_rad / 2) / self.z_resolution) + 10
        )
        within_vfov = np.zeros_like(within_hfov, dtype=bool)
        within_vfov[min_visible_dist:, :] = 1
        # ground_dist = np.sqrt(dx**2 + dy**2)
        # within_vfov = ground_dist >= min_visible_dist

        within_fov = within_hfov & within_vfov

        stair_mask = (ground_plane == 0) & within_fov & visible_ground
        stair_mask_vis = stair_mask.copy()

        stair_mask = cv2.dilate(stair_mask.astype(np.uint8), selem, iterations=2)
        # #! myTODO: 10 is hardcoded
        # # Extend to agents location
        # x_indices = np.where(stair_mask[min_visible_dist] == 1)[0]
        # rows = np.arange(10, min_visible_dist + 1).reshape(
        #     -1, 1
        # )  # shape: (min_visible_dist-10+1, 1)
        # rr, cc = np.meshgrid(rows, x_indices, indexing="ij")
        # stair_mask[rr, cc] = 1
        if self.print_images:
            plt.clf()
            plt.subplot(321)
            plt.title("ground plane")
            plt.imshow(np.flipud(ground_plane))
            plt.subplot(322)
            plt.title("withinfov")
            plt.imshow(np.flipud(within_fov))
            plt.subplot(323)
            plt.title("visible_ground")
            plt.imshow(np.flipud(visible_ground))
            plt.subplot(324)
            plt.title("stairs_mask")
            plt.imshow(np.flipud(stair_mask_vis))
            plt.subplot(325)
            plt.title("stairs_mask extended")
            plt.imshow(np.flipud(stair_mask))
            plt.savefig(self.vis_dir + f"/{self.timestep}_1.stairs.png")
        return torch.tensor(stair_mask, dtype=torch.uint8).to(
            voxels.device
        ), torch.tensor(ground_plane, dtype=torch.uint8).to(voxels.device)

    def _update_local_map_and_pose(
        self,
        obs: Tensor,
        curr_pose: Tensor,
        prev_map: Tensor,
        origins: Tensor,
        lmb: Tensor,
        instance_scores: Tensor,
        category_scores,
    ) -> Tuple[Tensor, Tensor]:

        # Index of the extra visibility channel appended to agent_view
        _VIS_CHANNEL = -1  # always the last channel

        obs_channels, h, w = obs.size()
        device, dtype = obs.device, obs.dtype

        tilt = torch.zeros(0)
        agent_height = self.agent_height
        yaw = torch.tensor(0)
        depth = obs[3, :, :].float()

        point_cloud_t = du.get_point_cloud_from_z_t(
            depth, self.camera_matrix, device, scale=self.du_scale
        )

        if self.debug_mode:
            from home_robot.utils.point_cloud import show_point_cloud
            rgb = obs[:3, :: self.du_scale, :: self.du_scale].permute(1, 2, 0)
            xyz = point_cloud_t.reshape(-1, 3)
            rgb_flat = rgb.reshape(-1, 3)
            show_point_cloud(
                (xyz / 100.0).cpu().numpy(),
                (rgb_flat / 255.0).cpu().numpy(),
                orig=np.zeros(3),
            )

        tilt_deg = torch.rad2deg(tilt).item() if tilt.numel() > 0 else 0.0
        point_cloud_base_coords = du.transform_camera_view_t(
            point_cloud_t, agent_height, tilt_deg, device
        )
        if self.real_world:
            camera_forward_offset_cm = 17.5
            point_cloud_base_coords[..., 1] += camera_forward_offset_cm

        point_cloud_map_coords = du.transform_pose_t(
            point_cloud_base_coords, self.shift_loc, device
        )

        voxel_channels = 1 + self.num_sem_categories
        num_instance_channels = 0
        if self.record_instance_ids:
            num_instance_channels = obs_channels - 4 - self.num_sem_categories
            voxel_channels += num_instance_channels

        assert obs.shape[0] == 4 + self.num_sem_categories + num_instance_channels

        init_grid = torch.zeros(
            voxel_channels,
            self.vision_range,
            self.vision_range,
            self.max_voxel_height - self.min_voxel_height,
            device=device,
            dtype=torch.float32,
        )
        feat = torch.ones(
            voxel_channels,
            self.screen_h // self.du_scale * self.screen_w // self.du_scale,
            device=device,
            dtype=torch.float32,
        )

        semantic_channels = obs[4 : 4 + self.num_sem_categories]

        if self.record_instance_ids:
            instance_channels = obs[4 + self.num_sem_categories :]
            self.instance_memory.unprocessed_views = {}
            self.instance_memory.temp_id_to_global_id = {0: 0}
 
            if num_instance_channels > 0:
                self.instance_memory.process_instances(
                    semantic_channels,
                    instance_channels,
                    instance_scores,
                    category_scores,
                    point_cloud_t.squeeze(0),
                    torch.concat([curr_pose + origins, lmb], axis=0),
                    image=obs[:3],
                )

        feat[1:, :] = obs[4:, :, :].view(
            obs_channels - 4, h // self.du_scale * w // self.du_scale
        )

        XYZ_cm_std = point_cloud_map_coords.float()
        XYZ_cm_std[..., :2] = XYZ_cm_std[..., :2] / self.xy_resolution
        XYZ_cm_std[..., :2] = (
            (XYZ_cm_std[..., :2] - self.vision_range // 2.0) / self.vision_range * 2.0
        )
        XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / self.z_resolution
        XYZ_cm_std[..., 2] = (
            (
                XYZ_cm_std[..., 2]
                - (self.max_voxel_height + self.min_voxel_height) // 2.0
            )
            / (self.max_voxel_height - self.min_voxel_height)
            * 2.0
        )
        XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2)
        XYZ_cm_std = XYZ_cm_std.view(
            XYZ_cm_std.shape[0],
            XYZ_cm_std.shape[1],
            XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3],
        )

        voxels = du.splat_feat_nd(init_grid, feat, XYZ_cm_std).transpose(1, 2)

        agent_height_proj = voxels[
            ..., self.min_obstacle_height : self.max_obstacle_height
        ].sum(3)
        all_height_proj = voxels.sum(3)

        # ================================================================
        # LAYER 1: Observation — raw counts → evidence
        # ================================================================

        obs_obstacle_evidence = agent_height_proj[0, :, :] / self.map_pred_threshold
        robot_channel_idx = self.num_sem_categories
        robot_projected = all_height_proj[robot_channel_idx] > 1
        obs_obstacle_evidence[robot_projected] = 0

        obs_semantic_evidence = all_height_proj[1:] / self.cat_pred_threshold

        obs_visible = (all_height_proj[0] > 0).float()

        fp_exp_pred = get_fp_exp_pred(self, obs_obstacle_evidence > 0.5)
        stairs_map, ground_plane = self.get_stairs(voxels[0], fp_exp_pred >= 1)

        # Build agent_view with visibility as extra last channel
        local_size = self.local_map_size_cm // self.xy_resolution
        num_channels = MC.NON_SEM_CHANNELS + self.num_sem_categories
        if self.record_instance_ids:
            num_channels += num_instance_channels
        num_channels_with_vis = num_channels + 1  # +1 for visibility

        agent_view = torch.zeros(num_channels_with_vis, local_size, local_size, device=device, dtype=dtype)

        x1 = local_size // 2 - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = local_size // 2
        y2 = y1 + self.vision_range

        agent_view[MC.GROUND_PLANE, y1:y2, x1:x2] = ground_plane * 1.0
        agent_view[MC.STAIRS, y1:y2, x1:x2] = stairs_map * 1.0
        agent_view[MC.OBSTACLE_MAP, y1:y2, x1:x2] = obs_obstacle_evidence
        agent_view[MC.GAZE_EXPLORED_MAP, y1:y2, x1:x2] = fp_exp_pred
        agent_view[MC.NON_SEM_CHANNELS:num_channels, y1:y2, x1:x2] = obs_semantic_evidence
        agent_view[_VIS_CHANNEL, y1:y2, x1:x2] = obs_visible

        # Transform to map coordinates (single grid_sample for all channels)
        st_pose = curr_pose.clone().detach()
        st_pose[:2] = -(
            (
                st_pose[:2] * 100.0 / self.xy_resolution
                - self.local_map_size_cm // (self.xy_resolution * 2)
            )
            / (self.local_map_size_cm // (self.xy_resolution * 2))
        )
        st_pose[2] = 90.0 - (st_pose[2])

        st_pose_adjusted = st_pose.clone()
        st_pose_adjusted[2] -= yaw.to(st_pose_adjusted.device) * 180 / np.pi

        rot_mat, trans_mat = ru.get_grid(st_pose_adjusted, agent_view.size(), dtype)
        rotated = F.grid_sample(agent_view.unsqueeze(0), rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True).squeeze(0)

        # Extract visibility and evidence before clamping
        visible_mask = translated[_VIS_CHANNEL] > 0.5
        obs_evidence_in_map = torch.clamp(translated[MC.OBSTACLE_MAP], min=0.0)

        sem_start = MC.NON_SEM_CHANNELS
        sem_end = MC.NON_SEM_CHANNELS + self.num_sem_categories
        sem_evidence_in_map = torch.clamp(translated[sem_start:sem_end], min=0.0)

        # Strip visibility channel, clamp rest for binary channels
        translated = translated[:num_channels]
        translated = torch.clamp(translated, min=0.0, max=1.0)

        if self.record_instance_ids:
            translated = self._aggregate_instance_map_channels_per_category(
                translated, num_instance_channels
            )

        # ================================================================
        # LAYER 2: Accumulation
        # ================================================================

        current_map = prev_map.clone()

        # --- Obstacle: log-odds ---
        # obs_log_update = obs_evidence_in_map * self.log_odds_occ - self.log_odds_free
        # current_map[MC.OBSTACLE_MAP] = torch.where(
        #     visible_mask,
        #     (prev_map[MC.OBSTACLE_MAP] + obs_log_update).clamp(
        #         -self.max_log_odds, self.max_log_odds
        #     ),
        #     prev_map[MC.OBSTACLE_MAP],
        # )

        gaze_mask = translated[MC.GAZE_EXPLORED_MAP] > 0.5

        # Positive update: only where depth points actually landed
        occ_update = torch.where(
            visible_mask & (obs_evidence_in_map > 0),
            # obs_evidence_in_map * self.log_odds_occ, #! Removed this to remove  dynamic obstacles much faster
            self.log_odds_occ,
            torch.zeros_like(obs_evidence_in_map),
        )
        # Negative update: anywhere in gaze FOV (we can see it's empty)
        free_update = torch.where(
            gaze_mask,
            torch.full_like(obs_evidence_in_map, self.log_odds_free),
            torch.zeros_like(obs_evidence_in_map),
        )

        current_map[MC.OBSTACLE_MAP] = torch.where(
            visible_mask | gaze_mask,
            (prev_map[MC.OBSTACLE_MAP] + occ_update - free_update).clamp(
                -self.max_log_odds, self.max_log_odds
            ),
            prev_map[MC.OBSTACLE_MAP],
        )


        # --- Semantics: log-odds per category (excluding robot channel) ---
        robot_sem_idx = MC.NON_SEM_CHANNELS + self.num_sem_categories - 1
        sem_log_update = sem_evidence_in_map * self.log_odds_occ - self.log_odds_free
        prev_sem = prev_map[sem_start:sem_end]
        updated_sem = torch.where(
            visible_mask.unsqueeze(0).expand_as(prev_sem),
            (prev_sem + sem_log_update).clamp(
                -self.max_log_odds, self.max_log_odds
            ),
            prev_sem,
        )
        current_map[sem_start:sem_end] = updated_sem

        # Robot semantic: overwrite (transient, not accumulated)
        current_map[robot_sem_idx] = translated[robot_sem_idx]
        robot_present = current_map[robot_sem_idx] > 0.5
        current_map[MC.OBSTACLE_MAP][robot_present] = 0

        # Gaze explored, ground plane, stairs: max (monotonic)
        for ch in [MC.GAZE_EXPLORED_MAP, MC.GROUND_PLANE, MC.STAIRS, MC.BEEN_CLOSE_MAP]:
            current_map[ch] = torch.maximum(prev_map[ch], translated[ch])

        # Instance channels: overwrite
        if self.record_instance_ids:
            inst_start = MC.NON_SEM_CHANNELS + self.num_sem_categories
            current_map[inst_start:] = translated[inst_start:]

        # Stairs → obstacle
        stairs_obstacle = (current_map[MC.STAIRS] > 0.5) & (current_map[MC.GROUND_PLANE] < 0.5)
        current_map[MC.OBSTACLE_MAP][stairs_obstacle] = self.max_log_odds

        # ================================================================
        # Derived channels
        # ================================================================

        curr_loc = curr_pose[:2].flip(0)
        curr_loc = (curr_loc * 100.0 / self.xy_resolution).round().int().tolist()

        prev_local_pose = self._prev_local_pose if self._prev_local_pose is not None else curr_pose.clone()
        prev_loc = prev_local_pose[:2].flip(0)
        prev_loc = (prev_loc * 100.0 / self.xy_resolution).round().int().tolist()

        current_map[MC.AGENT_VISITED_MAP] = self._get_update_visited_map(
            curr_loc, prev_loc, current_map[MC.AGENT_VISITED_MAP]
        )
        current_map[MC.VISITED_MAP] = (
            (current_map[MC.AGENT_VISITED_MAP] == 1) | (current_map[MC.VISITED_MAP] == 1)
        )


        obstacle_binary = self._threshold_obstacles(current_map[MC.OBSTACLE_MAP])
        dilated_obstacle = cv2.dilate(
            obstacle_binary.astype(np.uint8),
            skimage.morphology.disk(self.start_obs_dilation),
            iterations=1,
        ).astype(bool)
        traversible_np = (~dilated_obstacle).astype(float)



        if not self.real_world or self.timestep % 5 == 0:
            visited_np = current_map[MC.VISITED_MAP].detach().cpu().numpy() == 1
            traversible_ma = np.ma.masked_values(traversible_np * 1, 0)
            traversible_ma[visited_np == 1] = 0

            import skfmm
            distances = skfmm.distance(traversible_ma)
            distances = np.ma.filled(distances, np.max(distances) + 1)
            distances = torch.from_numpy(distances).to(current_map.device)

            current_map[MC.BEEN_CLOSE_MAP] = 0
            current_map[MC.BEEN_CLOSE_MAP][
                distances <= self.been_close_to_radius // self.resolution
            ] = 1

        current_map[MC.EXPLORED_MAP] = (
            (current_map[MC.GAZE_EXPLORED_MAP] > 0.5)
            | (current_map[MC.BEEN_CLOSE_MAP] == 1.0)
        )

        radius = self.target_blacklisting_radius // self.resolution
        self._set_disk_to_one(radius, current_map, MC.BLACKLISTED_TARGETS_MAP, curr_loc)

        if debug_maps:
            import matplotlib
            matplotlib.use("Agg")
            cm = current_map.cpu()
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes[0, 0].set_title("obstacle log-odds")
            im = axes[0, 0].imshow(cm[MC.OBSTACLE_MAP].numpy(), cmap="RdBu_r",
                                    vmin=-self.max_log_odds, vmax=self.max_log_odds)
            fig.colorbar(im, ax=axes[0, 0], fraction=0.046)
            axes[0, 1].set_title("obstacle (>0)")
            axes[0, 1].imshow(cm[MC.OBSTACLE_MAP].numpy() > 0)
            axes[0, 2].set_title("visible this frame")
            axes[0, 2].imshow(visible_mask.cpu().numpy())
            sem_1d = cm[sem_start:sem_end].numpy()
            no_cat = sem_1d.max(0) <= 0
            sem_vis = sem_1d.argmax(0) + 1
            sem_vis[no_cat] = 0
            axes[1, 0].set_title("semantic (argmax >0)")
            axes[1, 0].imshow(sem_vis, cmap="tab20")
            axes[1, 1].set_title("explored")
            axes[1, 1].imshow(cm[MC.EXPLORED_MAP].numpy() > 0.5)
            axes[1, 2].set_title("frame obstacle evidence")
            im2 = axes[1, 2].imshow(obs_evidence_in_map.cpu().numpy(), cmap="hot", vmin=0, vmax=3)
            fig.colorbar(im2, ax=axes[1, 2], fraction=0.046)
            for ax in axes.flat:
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(self.vis_dir + f"/{self.timestep}_0.local_map.png")
            plt.close()

        return current_map

    def _update_global_map_instances_for_one_channel(
        self,
        global_instances: Tensor,
        local_map: Tensor,
        x_range: tuple,
        y_range: tuple,
        max_instance_id: int,
    ) -> Tensor:
        """
        Update one instance channels in the global map from one instance channels in the local map:
        aggregate local instances with existing global instances or create new global instances.

        Args:
            global_instances (Tensor): The global map tensor.
            local_map (Tensor): The local map tensor.
            x_range (tuple): The range of indices in the x-axis for the local map in the global map.
            y_range (tuple): The range of indices in the y-axis for the local map in the global map.

        Returns:
            Tensor: The updated global instances tensor.

        """
        p = self.padding_for_instance_overlap  # default: 1
        d = self.dilation_for_instances  # default: 0

        H = global_instances.shape[0]
        W = global_instances.shape[1]

        x1, x2 = x_range
        y1, y2 = y_range

        # padding added on each side
        t_p = min(x1, p)
        b_p = min(H - x2, p)
        l_p = min(y1, p)
        r_p = min(W - y2, p)

        # the indices of the padded local_map in the global map
        x_start = x1 - t_p
        x_end = x2 + b_p
        y_start = y1 - l_p
        y_end = y2 + r_p

        local_map = torch.round(local_map)

        # pad the local map
        extended_local_map = F.pad(local_map.float(), (l_p, r_p), mode="replicate")
        extended_local_map = F.pad(
            extended_local_map.transpose(1, 0), (t_p, b_p), mode="replicate"
        ).transpose(1, 0)

        self.instance_dilation_selem = skimage.morphology.disk(d)
        # dilate the extended local map
        if d > 0:
            extended_dilated_local_map = torch.round(
                torch.tensor(
                    cv2.dilate(
                        extended_local_map.cpu().numpy(),
                        self.instance_dilation_selem,
                        iterations=1,
                    ),
                    device=local_map.device,
                    dtype=local_map.dtype,
                )
            )
        else:
            extended_dilated_local_map = torch.clone(extended_local_map)
        # Get the instances from the global map within the local map's region

        self._create_or_update_global_instances(
            extended_dilated_local_map,
            global_instances[x_start:x_end, y_start:y_end],
            max_instance_id,
            torch.unique(extended_local_map).tolist(),
        )

        # Update the global map with the associated instances from the local map

        # only to speed up
        max_temp_id = int(max(self.instance_memory.temp_id_to_global_id.keys()))
        temp_id_lookup = np.full(max_temp_id + 1, -1, dtype=np.int16)  # -1 for unmapped
        for temp_id, global_id in self.instance_memory.temp_id_to_global_id.items():
            temp_id_lookup[temp_id] = global_id
        global_instances_in_local = temp_id_lookup[local_map.cpu().numpy().astype(int)]

        global_instances[x1:x2, y1:y2] = torch.maximum(
            global_instances[x1:x2, y1:y2],
            torch.tensor(
                global_instances_in_local,
                dtype=torch.int64,
                device=global_instances.device,
            ),
        )
        return global_instances

    def _create_or_update_global_instances(
        self,
        extended_local_labels: Tensor,
        global_instances_within_local: Tensor,
        max_instance_id: int,
        temp_instance_ids: List[int],
    ) -> dict:
        """
        Creates a global instance for each local instance if it does not already exist, otherwise only update the global instance.
        It also creates a mapping of local instance IDs to global instance IDs internally by calling update_temp_id.

        Args:
            extended_local_labels: Labels of instances in the extended local map.
            global_instances_within_local: Instances from the global map within the local map's region.
        """
        # Associate instances in the local map with corresponding instances in the global map
        for temp_id in temp_instance_ids:
            if temp_id == 0:
                # ignore 0 as it does not correspond to an instance
                continue
            # pixels corresponding to
            local_instance_pixels = extended_local_labels == temp_id

            # Check for overlapping instances in the global map
            overlapping_instances = global_instances_within_local[local_instance_pixels]
            unique_overlapping_instances = torch.unique(overlapping_instances)

            unique_overlapping_instances = unique_overlapping_instances[
                unique_overlapping_instances != 0
            ]
            if len(unique_overlapping_instances) >= 1:
                # If there is a corresponding instance in the global map, pick the first one and associate it
                global_instance_id = int(unique_overlapping_instances[0].item())
            else:
                # If there are no corresponding instances, create a new instance
                max_instance_id += 1
                global_instance_id = max_instance_id
            # update the id in instance memory
            # self.log.debug(f"Mapping temp id {temp_id} to global id {global_instance_id}")
            self.instance_memory.update_temp_id(temp_id, global_instance_id)

    def _update_global_map_instances(
        self, global_map: Tensor, local_map: Tensor, lmb: Tensor
    ) -> Tensor:
        """
        Update instance channels in the global map from instance channels in the local map:
        aggregate local instances with existing global instances or create new global instances.

        Args:
            e (int): The index of the environment.
            global_map (Tensor): The global map tensor.
            local_map (Tensor): The local map tensor.
            lmb (Tensor): The tensor containing the ranges of indices for the local map in the global map.

        Returns:
            Used to return global map, but if we are updating it in place, we don't need to return anything.
        """
        # TODO Can we vectorize this across categories? (Only needed if speed bottleneck)
        # self.log.debug("Updating global map instances.")
        for i in range(self.num_sem_categories):
            if (
                torch.sum(local_map[MC.NON_SEM_CHANNELS + i + self.num_sem_categories])
                > 0
            ):
                max_instance_id = (
                    torch.max(
                        global_map[
                            MC.NON_SEM_CHANNELS
                            + self.num_sem_categories : MC.NON_SEM_CHANNELS
                            + 2 * self.num_sem_categories,
                        ]
                    )
                    .int()
                    .item()
                )
                # if the local map has any object instances, update the global map with instance ids
                # self.log.debug(f"Updating global map instances for category {i}, current max id {max_instance_id}.")
                instance_channel = self._update_global_map_instances_for_one_channel(
                    global_map[MC.NON_SEM_CHANNELS + self.num_sem_categories + i],
                    local_map[MC.NON_SEM_CHANNELS + self.num_sem_categories + i],
                    (lmb[0], lmb[1]),
                    (lmb[2], lmb[3]),
                    max_instance_id,
                )
                global_map[MC.NON_SEM_CHANNELS + self.num_sem_categories + i] = (
                    instance_channel
                )

    def _update_global_map_and_pose(
        self,
        state,
    ):
        """Update global map and pose and re-center local map and pose for a
        particular environment.
        """

        global_map = state.global_map
        robot_sem_idx = MC.NON_SEM_CHANNELS + self.num_sem_categories - 1
        global_map[robot_sem_idx] = 0
        lmb = state.lmb

        if self.record_instance_ids:
            assert global_map.shape[0] == MC.NON_SEM_CHANNELS + self.num_sem_categories * 2
            self._update_global_map_instances(global_map, state.local_map, lmb)
            global_map[
                : MC.NON_SEM_CHANNELS + self.num_sem_categories,
                lmb[0] : lmb[1],
                lmb[2] : lmb[3],
            ] = state.local_map[: MC.NON_SEM_CHANNELS + self.num_sem_categories]
        else:
            global_map[:, lmb[0] : lmb[1], lmb[2] : lmb[3]] = state.local_map

        state.local_map = global_map[:, lmb[0] : lmb[1], lmb[2] : lmb[3]]

    def _get_disk_mask(self, radius):
        """Cache disk masks for reuse"""
        if radius not in self._disk_masks:
            y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
            mask = x**2 + y**2 <= radius**2
            self._disk_masks[radius] = torch.from_numpy(mask).to(self.device)
        return self._disk_masks[radius]

    def _set_disk_to_one(self, radius, current_map, channel, curr_loc):
        y, x = curr_loc
        disk_mask = self._get_disk_mask(radius)

        H, W = current_map.shape[1:]
        y_min = max(y - radius, 0)
        y_max = min(y + radius + 1, H)
        x_min = max(x - radius, 0)
        x_max = min(x + radius + 1, W)

        disk_y_min = y_min - (y - radius)
        disk_y_max = disk_y_min + (y_max - y_min)
        disk_x_min = x_min - (x - radius)
        disk_x_max = disk_x_min + (x_max - x_min)

        current_map[channel, y_min:y_max, x_min:x_max][
            disk_mask[disk_y_min:disk_y_max, disk_x_min:disk_x_max] == 1
        ] = 1

    def _get_update_visited_map(self, current_loc, prev_loc, visited_map):
        """
        Marks the visited_map with a thick line between start and end.
        """
        thickness = self.agent_cell_radius + 1
        line_points = list(
            bresenham(prev_loc[0], prev_loc[1], current_loc[0], current_loc[1])
        )

        for x, y in line_points:
            if 0 <= x < visited_map.shape[0] and 0 <= y < visited_map.shape[1]:
                rr, cc = disk((x, y), radius=thickness, shape=visited_map.shape)
                visited_map[rr, cc] = 1
        visited_map[
            current_loc[0]
            - self.agent_cell_radius : current_loc[0]
            + self.agent_cell_radius
            + 1,
            current_loc[1]
            - self.agent_cell_radius : current_loc[1]
            + self.agent_cell_radius
            + 1,
        ] = 1
        return visited_map

    @staticmethod
    def _threshold_obstacles(log_odds_map):
        """log-odds → clean binary obstacle map."""
        binary = (log_odds_map.cpu().numpy() > 0).astype(np.uint8)
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, skimage.morphology.disk(1)).astype(bool)