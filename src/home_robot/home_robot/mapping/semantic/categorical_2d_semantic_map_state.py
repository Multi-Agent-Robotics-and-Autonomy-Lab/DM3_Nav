import cv2
import os
import torch
import numpy as np
import skimage.morphology
from typing import Optional
import torch.nn.functional as F
from scipy.ndimage import label
from home_robot.mapping.map_utils import MapSizeParameters, init_map_and_pose
from home_robot.mapping.semantic.constants import MapConstants as MC
from home_robot.mapping.semantic.instance_tracking_modules import InstanceMemory


class Categorical2DSemanticMapState:

    def __init__(
        self,
        device: torch.device,
        num_sem_categories: int,
        map_resolution: int,
        map_size_cm: int,
        global_downscaling: int,
        visualization_level,
        record_instance_ids: bool = False,
        instance_memory: Optional[InstanceMemory] = None,
        agent_id: int = 0,
    ):
        self.device = device
        self.num_sem_categories = num_sem_categories

        self.map_size_parameters = MapSizeParameters(
            map_resolution, map_size_cm, global_downscaling
        )
        self.resolution = map_resolution
        self.global_map_size_cm = map_size_cm
        self.global_downscaling = global_downscaling
        self.local_map_size_cm = self.global_map_size_cm // self.global_downscaling
        self.global_map_size = self.global_map_size_cm // self.resolution
        self.local_map_size = self.local_map_size_cm // self.resolution

        num_channels = self.num_sem_categories + MC.NON_SEM_CHANNELS
        if record_instance_ids:
            num_channels += self.num_sem_categories
            self.instance_memory = instance_memory

        self.num_channels = num_channels
        self.vis_dir = None
        self.agent_id = agent_id
        self.visualization_level = visualization_level

    def init_map_and_pose(self):
        self.global_map, self.global_pose, self.local_map, self.local_pose, self.lmb, self.origins = init_map_and_pose(
            self.map_size_parameters,
            self.device,
            self.num_channels,
        )
        self.frontier_map = np.zeros(
            (self.local_map_size, self.local_map_size)
        )

    # ------------------------------------------------------------------
    # Getters (Layer 3: log-odds → binary decisions)
    # ------------------------------------------------------------------

    def _get_map(self, local):
        return self.local_map if local else self.global_map

    def get_obstacle_map(self, local=True) -> np.ndarray:
        """Log-odds > 0 means occupied. Morphological opening removes noise."""
        raw = self._get_map(local)[MC.OBSTACLE_MAP].cpu().numpy()
        binary = (raw > 0).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, skimage.morphology.disk(1))
        return binary.astype(bool)

    def get_explored_map(self, local=True) -> np.ndarray:
        return (self._get_map(local)[MC.EXPLORED_MAP].cpu().numpy() > 0.5).astype(bool)

    def get_visited_map(self, local=True, full=False) -> np.ndarray:
        layer = MC.VISITED_MAP if full else MC.AGENT_VISITED_MAP
        return (self._get_map(local)[layer].cpu().numpy() > 0.5).astype(bool)

    def get_been_close_map(self, local=True) -> np.ndarray:
        return (self._get_map(local)[MC.BEEN_CLOSE_MAP].cpu().numpy() > 0.5).astype(bool)

    def get_blacklisted_targets_map(self, local=True) -> np.ndarray:
        return (self._get_map(local)[MC.BLACKLISTED_TARGETS_MAP].cpu().numpy() > 0.5).astype(bool)

    def get_semantic_map(self, local=True) -> np.ndarray:
        """Returns raw log-odds semantic map. Use > 0 for presence."""
        m = self._get_map(local)
        return np.copy(m[MC.NON_SEM_CHANNELS : MC.NON_SEM_CHANNELS + self.num_sem_categories].cpu().float().numpy())

    def get_semantic_map_1D(self, local=True) -> np.ndarray:
        """Returns (category_map, no_category_mask).
        category_map: argmax category ID (1-indexed) per cell.
        no_category_mask: True where no category has positive log-odds."""
        semantic_map = self.get_semantic_map(local)
        no_cat_mask = semantic_map.max(axis=0) <= 0
        semantic_map = semantic_map.argmax(axis=0) + 1
        return semantic_map, no_cat_mask

    def get_instances_map(self, local=True) -> np.ndarray:
        m = self._get_map(local)
        return m[MC.NON_SEM_CHANNELS + self.num_sem_categories:].cpu().float().numpy()

    def get_instance_map(self, instance_id, local=True):
        instances_map = self.get_instances_map(local)
        inst_map_idx = instances_map == instance_id
        inst_map_idx = np.argmax(np.sum(inst_map_idx, axis=(1, 2)))
        return (instances_map[inst_map_idx] == instance_id).astype(bool)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def local_to_global(self, row_local, col_local):
        lmb = self.lmb.cpu()
        row_global = row_local + lmb[0] - self.global_map_size // 2
        col_global = col_local + lmb[2] - self.global_map_size // 2
        return row_global, col_global

    def global_to_local(self, row_global, col_global):
        lmb = self.lmb.cpu()
        row_local = row_global - lmb[0] + self.global_map_size // 2
        col_local = col_global - lmb[2] + self.global_map_size // 2
        return row_local, col_local

    def global_pose_to_global_location(self, global_pose):
        return (int(global_pose[1] * 100.0 / self.resolution), int(global_pose[0] * 100.0 / self.resolution))

    def global_location_to_local_location(self, global_location) -> tuple:
        return (global_location[0] - self.lmb[0].item(), global_location[1] - self.lmb[2].item())

    def global_pose_to_local_location(self, global_pose) -> tuple:
        return self.global_location_to_local_location(self.global_pose_to_global_location(global_pose))

    def is_location_in_local_map(self, local_location):
        return 0 <= local_location[0] < self.local_map_size and 0 <= local_location[1] < self.local_map_size

    @property
    def local_loc(self):
        location = self.local_pose[:2]
        location = (location * 100.0 / self.resolution).round().int().tolist()
        return location[1], location[0]

    @property
    def global_loc(self):
        location = self.global_pose[:2]
        location = (location * 100.0 / self.resolution).round().int().tolist()
        return location[1], location[0]

    def get_loc(self, local=True):
        return self.local_loc if local else self.global_loc

    # ------------------------------------------------------------------
    # Frontiers
    # ------------------------------------------------------------------

    def get_frontier_map(self, traversible, local=True, timestep=None, min_size=10):
        def remove_small_frontiers(frontier_map, min_size):
            labeled_map, num_features = label(frontier_map)
            cleaned_map = np.zeros_like(frontier_map)
            for region_id in range(1, num_features + 1):
                region = labeled_map == region_id
                if np.sum(region) >= min_size:
                    cleaned_map[region] = 1
            return cleaned_map

        assert traversible.dtype == bool
        known_map = self.get_explored_map(local)
        known_map = known_map | ~traversible

        free_space = torch.tensor(known_map & traversible, dtype=torch.float32)

        kernel = torch.tensor(
            [[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)

        free_neighbors = (F.conv2d(
            free_space.unsqueeze(0).unsqueeze(0),
            kernel,
            padding=1
        ).squeeze() > 0).numpy().astype(bool)

        frontier_map = ~known_map & free_neighbors
        frontier_map2 = remove_small_frontiers(frontier_map, min_size)
        frontier_map4 = frontier_map2 & ~self.get_unreachable_frontiers_map(local)

        self.print_maps(
            frontier_map=frontier_map,
            frontier_map2=frontier_map2,
            frontier_map4=frontier_map4,
            obstacle_map=~traversible,
            known_map=known_map,
            timestep=timestep,
            local=local,
        )
        return frontier_map4

    def print_maps(
        self, frontier_map, obstacle_map, known_map, local, timestep=None,
        frontier_map2=None, frontier_map4=None
    ):
        if timestep is None or self.visualization_level < 3:
            return

        H, W = known_map.shape
        vis_map = np.ones((H, W, 3), dtype=np.uint8) * 255
        vis_map[known_map == 1] = [117, 117, 117]
        vis_map[obstacle_map == 1] = [0, 0, 0]

        panels = [frontier_map, frontier_map2, frontier_map4]
        vis_map = np.concatenate([vis_map] * (1 + len(panels)), axis=1)
        for i, fm in enumerate(panels):
            if fm is not None:
                vis_map[:, (i + 1) * W:(i + 2) * W, :][fm == 1] = [255, 0, 0]

        cv2.imwrite(
            os.path.join(self.vis_dir, f"{timestep}_2.frontiers{'' if local else '_global'}.png"),
            np.flipud(vis_map)
        )

    def get_unreachable_frontiers_map(self, local=True) -> np.ndarray:
        return (self._get_map(local)[MC.UNREACHABLE_FRONTIERS_MAP].cpu().numpy() > 0).astype(bool)

    def reset_unreachable_frontier(self):
        self.global_map[MC.UNREACHABLE_FRONTIERS_MAP] = 0

    def set_unreachable_frontier(self, frontier_map, local=True) -> np.ndarray:
        self.merge_map(frontier_map, MC.UNREACHABLE_FRONTIERS_MAP, local)

    def merge_map(self, new_map, layer, local):
        if local:
            merged = torch.logical_or(
                torch.tensor(new_map, device=self.device),
                self.local_map[layer]
            )
            self.local_map[layer] = merged
            self.global_map[layer, self.lmb[0]:self.lmb[1], self.lmb[2]:self.lmb[3]] = merged
        else:
            merged = torch.logical_or(
                torch.tensor(new_map, device=self.device),
                self.global_map[layer]
            )
            self.global_map[layer] = merged
            self.local_map[layer] = self.global_map[layer, self.lmb[0]:self.lmb[1], self.lmb[2]:self.lmb[3]]