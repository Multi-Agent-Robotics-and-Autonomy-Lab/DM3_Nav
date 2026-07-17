import os
import cv2
import torch
import logging
import numpy as np
from torch import Tensor
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict
from home_robot.mapping.semantic.constants import MapConstants as MC

class MapMergerLogger(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return f"[MAP MERGE] {msg}", kwargs

class MapMerger:
    def __init__(
        self,
        num_sem_categories: int,
        resolution,
        log=None,
        ransac_thresh: float = 10.0,
        iou_threshold: float = 0.3,
        semantic_weight: float = 1.0,
        vis_dir: Optional[str] = None,
    ):
        self.num_sem_categories = num_sem_categories # Note that this includes last layer which is other robots loc
        self.resolution = resolution
        self.ransac_thresh = ransac_thresh
        self.iou_threshold = iou_threshold
        self.semantic_weight = semantic_weight
        self.vis_dir = vis_dir
        self.timestep = 0
        self._cached_transforms: Dict[int, Dict] = {}
        self._merged_masks: Dict[int, torch.Tensor] = {}
        self.log = MapMergerLogger(log, None)
        if log is None:
            self.log = logging.getLogger()


    def clear_merged_masks(self):
        self._merged_masks.clear()

    def clear_cached_transforms(self):
        self._cached_transforms.clear()

    def has_cached_transform(self, neighbor_id: int) -> bool:
        return neighbor_id in self._cached_transforms

    def transform_location(self, neighbor_id: int, location) -> Optional[torch.Tensor]:
        if not self.has_cached_transform(neighbor_id):
            return None

        transform = self._cached_transforms[neighbor_id]["transform"]
        T = torch.from_numpy(transform).to(dtype=torch.float64)
        p = (T @ torch.tensor([location[1], location[0], 1.0], dtype=torch.float64)).round()
        return (int(p[1]), int(p[0]))

    def get_transformed_map(self, global_map, neighbor_global_map, neighbor_id):
        device = global_map.device
        if neighbor_global_map is not None:
            neighbor_global_map = neighbor_global_map.to(device)

        if neighbor_global_map is not None:
            if neighbor_id in self._cached_transforms and self._cached_transforms[neighbor_id]["iou"] > 0.90:
                transform = self._cached_transforms[neighbor_id]["transform"]
                self.log.debug(f"Using cached transform for neighbor {neighbor_id}, iou>90")
            else:
                transform = self._estimate_transform(global_map, neighbor_global_map)
        else:
            transform = None

        if transform is None:
            self.log.debug("WARNING: Could not estimate transform.")
            return None

        warped_neighbor = self._warp_map(neighbor_global_map, transform, global_map.shape)

        cached_iou = None
        if neighbor_id in self._cached_transforms:
            cached_iou = self._cached_transforms[neighbor_id]["iou"]

        if cached_iou is not None and cached_iou >= 0.9:
            iou = cached_iou
        else:
            iou = self._alignment_confidence(global_map, warped_neighbor)
            self.log.debug(f"Alignment IoU: {iou:.4f} (threshold: {self.iou_threshold})")
            min_iou = min(self.iou_threshold, cached_iou) if cached_iou is not None else self.iou_threshold
            if iou < min_iou:
                self.log.debug("WARNING: Low alignment confidence.")
                return None

        self._cached_transforms[neighbor_id] = {
            "iou": iou,
            "transform": transform
        }
        self.log.debug(f"Transform cached for neighbor {neighbor_id}")

        return warped_neighbor

    def _alignment_confidence(self, map_A: Tensor, warped_B: Tensor) -> float:
        exp_A = map_A[MC.EXPLORED_MAP].cpu().numpy() > 0
        exp_B = warped_B[MC.EXPLORED_MAP].cpu().numpy() > 0
        # exp_A = self._get_data_mask(map_A).cpu().numpy()
        # exp_B = self._get_data_mask(warped_B).cpu().numpy()
        overlap = exp_A & exp_B

        if overlap.sum() < 50:
            return 0.0

        obs_A = map_A[MC.OBSTACLE_MAP].cpu().numpy() > 0
        obs_B = warped_B[MC.OBSTACLE_MAP].cpu().numpy() > 0


        a = obs_A[overlap]
        b = obs_B[overlap]

        intersection = (a & b).sum()
        union = (a | b).sum()

        if union == 0:
            return 0.0

        return float(intersection / union)

    def _estimate_transform(self, map_A: Tensor, map_B: Tensor) -> Optional[np.ndarray]:
        """
        Estimate rigid transform (rotation + translation) from B's frame to A's frame
        using ORB features on obstacle map + semantic landmark correspondences.
        Returns 2x3 affine matrix or None.
        """
        obs_A = ((map_A[MC.OBSTACLE_MAP].cpu().numpy() > 0) * 255).astype(np.uint8)
        obs_B = ((map_B[MC.OBSTACLE_MAP].cpu().numpy() > 0) * 255).astype(np.uint8)


        exp_A= (map_A[MC.EXPLORED_MAP].cpu().numpy() > 0).astype(np.uint8) * 255
        exp_B= (map_B[MC.EXPLORED_MAP].cpu().numpy() > 0).astype(np.uint8) * 255

        pts_A, pts_B = [], []

        # --- ORB features on obstacle map ---
        orb_pA, orb_pB = self._orb_matches(obs_A, obs_B, exp_A, exp_B)
        if orb_pA is not None:
            pts_A.append(orb_pA)
            pts_B.append(orb_pB)

        # --- Semantic landmark matching ---
        sem_pA, sem_pB = self._semantic_landmark_matches(map_A, map_B)
        if sem_pA is not None:
            pts_A.append(sem_pA)
            pts_B.append(sem_pB)

        if len(pts_A) == 0:
            return None

        all_pts_A = np.vstack(pts_A).astype(np.float32)
        all_pts_B = np.vstack(pts_B).astype(np.float32)

        if len(all_pts_A) < 3:
            return None

        # RANSAC for rigid transform (rotation + translation, no scale)
        M, inliers = cv2.estimateAffinePartial2D(
            all_pts_B.reshape(-1, 1, 2),
            all_pts_A.reshape(-1, 1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thresh,
        )

        if M is None or inliers is None:
            return None

        return M

    def _get_data_mask(self, map_tensor: Tensor, threshold: float = 0.01) -> torch.Tensor:
        protected_channels = {MC.AGENT_VISITED_MAP, MC.BLACKLISTED_TARGETS_MAP, 
                            MC.NON_SEM_CHANNELS + self.num_sem_categories - 1}
        
        mask = torch.zeros(map_tensor.shape[1:], dtype=torch.bool, device=map_tensor.device)
        
        for c in range(min(map_tensor.shape[0], MC.NON_SEM_CHANNELS + self.num_sem_categories)):
            if c in protected_channels:
                continue
            mask = mask | (map_tensor[c].abs() > threshold)
        
        return mask

    def _orb_matches(
        self,
        obs_A: np.ndarray,
        obs_B: np.ndarray,
        mask_A: np.ndarray,
        mask_B: np.ndarray,
        max_features: int = 1000,
        top_k: int = 100,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        orb = cv2.ORB_create(max_features)

        kp1, des1 = orb.detectAndCompute(obs_A, mask_A)
        kp2, des2 = orb.detectAndCompute(obs_B, mask_B)

        if des1 is None or des2 is None or len(des1) < 3 or len(des2) < 3:
            return None, None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda m: m.distance)[:top_k]

        if len(matches) < 3:
            return None, None

        pts_A = np.array([kp1[m.queryIdx].pt for m in matches])
        pts_B = np.array([kp2[m.trainIdx].pt for m in matches])
        return pts_A, pts_B

    def _semantic_landmark_matches(
        self,
        map_A: Tensor,
        map_B: Tensor,
        min_component_size: int = 10,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Match semantic object centroids between maps.
        Same semantic class + similar local obstacle context = putative match.
        """
        sem_start = MC.NON_SEM_CHANNELS
        sem_end = MC.NON_SEM_CHANNELS + self.num_sem_categories-1 #! Don't consider last layer which is other robots loc

        sem_A = map_A[sem_start:sem_end].cpu().numpy()
        sem_B = map_B[sem_start:sem_end].cpu().numpy()
        obs_A = (map_A[MC.OBSTACLE_MAP].cpu().numpy() > 0).astype(float)
        obs_B = (map_B[MC.OBSTACLE_MAP].cpu().numpy() > 0).astype(float)

        landmarks_A = self._extract_semantic_landmarks(sem_A, obs_A, min_component_size)
        landmarks_B = self._extract_semantic_landmarks(sem_B, obs_B, min_component_size)

        if not landmarks_A or not landmarks_B:
            return None, None

        pts_A, pts_B = [], []

        for la in landmarks_A:
            best_dist = float("inf")
            best_lb = None
            for lb in landmarks_B:
                if la["label"] != lb["label"]:
                    continue
                geo_dist = np.linalg.norm(la["descriptor"] - lb["descriptor"])
                area_ratio = min(la["area"], lb["area"]) / max(la["area"], lb["area"])
                score = geo_dist - self.semantic_weight * area_ratio
                if score < best_dist:
                    best_dist = score
                    best_lb = lb
            if best_lb is not None:
                pts_A.append(la["centroid"])
                pts_B.append(best_lb["centroid"])

        if len(pts_A) < 3:
            return None, None

        return np.array(pts_A), np.array(pts_B)

    def _extract_semantic_landmarks(
        self,
        sem_channels: np.ndarray,
        obstacle_map: np.ndarray,
        min_size: int,
    ) -> list:
        """Extract centroids + descriptors for each semantic object instance."""
        landmarks = []

        for cat_idx in range(self.num_sem_categories - 1):
            binary = (sem_channels[cat_idx] > 0.5).astype(np.uint8)
            if binary.sum() < min_size:
                continue

            n_comp, comp_map = cv2.connectedComponents(binary)
            for comp_id in range(1, n_comp):
                mask = comp_map == comp_id
                if mask.sum() < min_size:
                    continue

                ys, xs = np.where(mask)
                cx, cy = float(xs.mean()), float(ys.mean())
                area = len(xs)

                desc = self._radial_obstacle_descriptor(obstacle_map, cx, cy)

                landmarks.append(
                    {
                        "label": cat_idx,
                        "centroid": np.array([cx, cy]),
                        "area": area,
                        "descriptor": desc,
                    }
                )
        return landmarks

    def _radial_obstacle_descriptor(
        self,
        obs_map: np.ndarray,
        cx: float,
        cy: float,
        radius: int = 25,
        n_rings: int = 8,
    ) -> np.ndarray:
        """Rotation-invariant descriptor: obstacle density in concentric rings."""
        h, w = obs_map.shape
        y0, y1 = max(0, int(cy) - radius), min(h, int(cy) + radius)
        x0, x1 = max(0, int(cx) - radius), min(w, int(cx) + radius)
        patch = obs_map[y0:y1, x0:x1]

        ph, pw = patch.shape
        if ph < 3 or pw < 3:
            return np.zeros(n_rings)

        local_cy, local_cx = cy - y0, cx - x0
        r_max = min(local_cx, local_cy, pw - local_cx, ph - local_cy, radius)
        if r_max < 2:
            return np.zeros(n_rings)

        ys, xs = np.ogrid[:ph, :pw]
        dist = np.sqrt((xs - local_cx) ** 2 + (ys - local_cy) ** 2)

        desc = np.zeros(n_rings)
        for i in range(n_rings):
            r_in = i * r_max / n_rings
            r_out = (i + 1) * r_max / n_rings
            ring = (dist >= r_in) & (dist < r_out)
            if ring.sum() > 0:
                desc[i] = patch[ring].mean()
        return desc


    def _warp_map(
        self, neighbor_map: Tensor, transform: np.ndarray, target_shape: tuple
    ) -> Tensor:
        """Warp all channels of neighbor_map into our frame."""
        C, H, W = target_shape
        device = neighbor_map.device
        warped = torch.zeros(C, H, W, dtype=neighbor_map.dtype, device=device)

        for c in range(C):
            layer = neighbor_map[c].cpu().numpy().astype(np.float32)
            w = cv2.warpAffine(
                layer,
                transform,
                (W, H),
                flags=cv2.INTER_NEAREST,
                borderValue=0,
            )
            warped[c] = torch.from_numpy(w).to(device)

        return warped

    def _merge(self, global_map: Tensor, data) -> Tensor:
        """
        Merge warped neighbor into our map.
        Only merges cells that haven't been merged before from this neighbor.
        """
        transformed_map = data["transformed_map"]
        neighbor_id = data["agent_id"]

        merged = global_map.clone()
        device = global_map.device

        protected_channels = torch.tensor(
            [MC.AGENT_VISITED_MAP, MC.BLACKLISTED_TARGETS_MAP,
            MC.NON_SEM_CHANNELS + self.num_sem_categories - 1],
            device=device,
        )
        all_channels = torch.arange(global_map.shape[0], device=device)
        merge_channel_mask = ~torch.isin(all_channels, protected_channels) & (
            all_channels < (MC.NON_SEM_CHANNELS + self.num_sem_categories)
        )

        neighbor_has_data = self._get_data_mask(transformed_map)
        # if neighbor_id not in self._merged_masks:
        #     self._merged_masks[neighbor_id] = torch.zeros(
        #         global_map.shape[1:], dtype=torch.bool, device=device
        #     )
        # already_merged = self._merged_masks[neighbor_id]


        # new_cells = neighbor_has_data & ~already_merged
        new_cells = neighbor_has_data
        if new_cells.sum() == 0:
            return merged
        for c in all_channels[merge_channel_mask]:
            ours = global_map[c][new_cells]
            theirs = transformed_map[c][new_cells]
            take_theirs = theirs.abs() > ours.abs()
            merged[c][new_cells] = torch.where(take_theirs, theirs, ours)
        # if neighbor_id >= 0:
        #     self._merged_masks[neighbor_id] = self._merged_masks[neighbor_id] | neighbor_has_data

        return merged

    @staticmethod
    def _transform_location(loc: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """
        Transform (x, y) cell coords from neighbor's frame to our frame.
        Convention: loc = (row, col), i.e. map[x, y] = map[row, col].
        The cv2 affine transform operates in (col, row) space, so we swap.
        """
        row, col = loc[0], loc[1]
        p = transform @ np.array([col, row, 1.0])
        out_col, out_row = p[0], p[1]
        return np.round(np.array([out_row, out_col])).astype(np.int64)


    def _get_crop_bounds(self, *maps_and_locs):
        all_rows, all_cols = [], []
        for item in maps_and_locs:
            if isinstance(item, Tensor):
                mask = self._get_data_mask(item).cpu().numpy()
                if mask.any():
                    rs, cs = np.where(mask)
                    all_rows.extend([rs.min(), rs.max()])
                    all_cols.extend([cs.min(), cs.max()])
            elif isinstance(item, (list, tuple, np.ndarray)) and len(item) == 2:
                all_rows.append(int(item[0]))
                all_cols.append(int(item[1]))
        if not all_rows:
            return None
        pad = 10
        H = maps_and_locs[0].shape[-2] if isinstance(maps_and_locs[0], Tensor) else 960
        W = maps_and_locs[0].shape[-1] if isinstance(maps_and_locs[0], Tensor) else 960
        r0 = max(0, min(all_rows) - pad)
        r1 = min(H, max(all_rows) + pad)
        c0 = max(0, min(all_cols) - pad)
        c1 = min(W, max(all_cols) + pad)
        return r0, r1, c0, c1

    def _equalize_bounds(self, bounds, bounds_B):
        """Expand both bounds to the same size so zoom level matches."""
        if bounds is None or bounds_B is None:
            return bounds, bounds_B
        r0, r1, c0, c1 = bounds
        r0b, r1b, c0b, c1b = bounds_B
        max_h = max(r1 - r0, r1b - r0b)
        max_w = max(c1 - c0, c1b - c0b)
        def _expand(r0, r1, c0, c1, th, tw):
            cr, cc = (r0 + r1) / 2, (c0 + c1) / 2
            return (int(cr - th / 2), int(cr + th / 2),
                    int(cc - tw / 2), int(cc + tw / 2))
        return _expand(r0, r1, c0, c1, max_h, max_w), _expand(r0b, r1b, c0b, c1b, max_h, max_w)

    def _get_semantic_dots(self, map_tensor, min_size=10):
        sem_start = MC.NON_SEM_CHANNELS
        sem_end = MC.NON_SEM_CHANNELS + self.num_sem_categories - 1
        sem = map_tensor[sem_start:sem_end].cpu().numpy()
        obs = (map_tensor[MC.OBSTACLE_MAP].cpu().numpy() > 0).astype(float)
        return self._extract_semantic_landmarks(sem, obs, min_size)

    def _plot_semantic_dots(self, ax, landmarks, color='green', alpha=0.95, size=25):
        if not landmarks:
            return
        xs = [lm["centroid"][0] for lm in landmarks]
        ys = [lm["centroid"][1] for lm in landmarks]
        ax.scatter(xs, ys, c=color, s=size, alpha=alpha, edgecolors='none', zorder=5)

    def _setup_panel(self, ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)

    def _apply_crop(self, ax, bounds):
        if bounds:
            r0, r1, c0, c1 = bounds
            ax.set_xlim(c0, c1)
            # ax.set_ylim(r1, r0)
            ax.set_ylim(r0, r1)

    def _visualize_failed(self, map_A, map_B, loc_A, loc_B, reason=None):
        bounds_A = self._get_crop_bounds(map_A, loc_A)
        bounds_B = self._get_crop_bounds(map_B, loc_B)
        bounds_A, bounds_B = self._equalize_bounds(bounds_A, bounds_B)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        obs_A = (map_A[MC.OBSTACLE_MAP].cpu().numpy() > 0).astype(float)
        obs_B = (map_B[MC.OBSTACLE_MAP].cpu().numpy() > 0).astype(float)

        axes[0].imshow(1 - obs_A, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[0].plot(loc_A[1], loc_A[0], "bo", markersize=10)
        self._plot_semantic_dots(axes[0], self._get_semantic_dots(map_A))
        axes[0].set_title("Robot 1 - Obstacle Map")
        self._setup_panel(axes[0])
        self._apply_crop(axes[0], bounds_A)

        axes[1].imshow(1 - obs_B, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[1].plot(loc_B[1], loc_B[0], "ro", markersize=10)
        self._plot_semantic_dots(axes[1], self._get_semantic_dots(map_B))
        axes[1].set_title("Robot 2 - Obstacle Map (own frame)")
        self._setup_panel(axes[1])
        self._apply_crop(axes[1], bounds_B)

        title = "Map Merge FAILED - Distance too far" if reason == "dist" else "Map Merge FAILED - Alignment could not be estimated"
        plt.suptitle(title, color="red")
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, f"{self.timestep}_15.map_merge_FAILED.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _visualize(self, map_A, loc_A, merged, data):
        transformed_map = data["transformed_map"]
        original_map = data["map"]
        loc_B_original = data["location"]
        loc_B_transformed = data["transformed_loc"]

        obs_A = (map_A[MC.OBSTACLE_MAP].cpu().numpy() > 0).astype(float)
        obs_merged = (merged[MC.OBSTACLE_MAP].cpu().numpy() > 0).astype(float)
        obs_B_original = (original_map[MC.OBSTACLE_MAP].cpu().numpy() > 0).astype(float)
        obs_B_warped = (transformed_map[MC.OBSTACLE_MAP].cpu().numpy() > 0).astype(float)

        bounds = self._get_crop_bounds(map_A, transformed_map, merged, loc_A, loc_B_transformed)
        bounds_B = self._get_crop_bounds(original_map, loc_B_original)
        bounds, bounds_B = self._equalize_bounds(bounds, bounds_B)

        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        plt.rc('axes', titlesize=20)

        # Panel 1: Robot 1
        axes[0].imshow(1 - obs_A, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[0].plot(loc_A[1], loc_A[0], "bo", markersize=10)
        self._plot_semantic_dots(axes[0], self._get_semantic_dots(map_A))
        axes[0].set_title("Robot 1 - Obstacle Map")
        self._setup_panel(axes[0])
        self._apply_crop(axes[0], bounds)

        # Panel 2: Robot 2 (own frame)
        axes[1].imshow(1 - obs_B_original, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[1].plot(loc_B_original[1], loc_B_original[0], "ro", markersize=10)
        self._plot_semantic_dots(axes[1], self._get_semantic_dots(original_map))
        axes[1].set_title("Robot 2 - Obstacle Map")
        self._setup_panel(axes[1])
        self._apply_crop(axes[1], bounds_B)

        # Panel 3: Overlay
        overlay = np.ones((*obs_A.shape, 3))
        overlay[:, :, 0] -= obs_A
        overlay[:, :, 1] -= obs_A
        overlay[:, :, 1] -= obs_B_warped
        overlay[:, :, 2] -= obs_B_warped
        overlay = np.clip(overlay, 0, 1)
        axes[2].imshow(overlay, origin="lower")
        axes[2].plot(loc_A[1], loc_A[0], "bo", markersize=10)
        axes[2].plot(loc_B_transformed[1], loc_B_transformed[0], "ro", markersize=10)
        axes[2].set_title("Alignment Overlay")
        self._setup_panel(axes[2])
        self._apply_crop(axes[2], bounds)

        # Panel 4: Merged
        axes[3].imshow(1 - obs_merged, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[3].plot(loc_A[1], loc_A[0], "bo", markersize=10)
        axes[3].plot(loc_B_transformed[1], loc_B_transformed[0], "ro", markersize=10)
        self._plot_semantic_dots(axes[3], self._get_semantic_dots(merged))
        dist_cells = np.sqrt(
            (loc_A[0] - loc_B_transformed[0]) ** 2
            + (loc_A[1] - loc_B_transformed[1]) ** 2
        )
        axes[3].set_title(f"Merged Map")
        # axes[3].set_title(f"Merged Map (dist={dist_cells * self.resolution:.2f}m)")
        self._setup_panel(axes[3])
        self._apply_crop(axes[3], bounds)

        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle
        legend_handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8),
        ]
        legend_labels = ["Robot 1", "Robot 2", "Semantic landmarks"]
        axes[0].legend(legend_handles, legend_labels, loc='upper left',
                ncol=1, fontsize=16,
                markerscale=1.2, framealpha=0.8)
        plt.tight_layout()

        plt.savefig(os.path.join(self.vis_dir, f"{self.timestep}_15.map_merge_SUCCESS.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    map_A = torch.load("/home/agilex2/projects/search/SemanticSearch/datadump/images/final_results/comm3/mymap_85.pt")
    map_B = torch.load("/home/agilex2/projects/search/SemanticSearch/datadump/images/final_results/comm3/othermap_85.pt")
    num_sem = int((map_A.shape[0] - MC.NON_SEM_CHANNELS) / 2)

    # Robot locations in their own frames (x, y) in pixel coords
    loc_A = [548, 408]
    loc_B = [505, 403]
    data = {
        "agent_id": 1,
        "map": map_B,
        "location": loc_B
    }

    map_merger = MapMerger(
        num_sem_categories=num_sem,
        resolution=0.05,
        ransac_thresh=10.0,
        iou_threshold=0.1,
    )
    map_merger.vis_dir = "."

    transfomed_map = map_merger.get_transformed_map(
        global_map=map_A,
        neighbor_global_map=map_B,
        neighbor_id=1
    )
    if transfomed_map is None:
        map_merger._visualize_failed(
            map_A,
            map_B,
            loc_A,
            loc_B
        )
    else:
        data["transformed_map"] = transfomed_map
        assert transfomed_map is not None
        merged = map_merger._merge(
            map_A,
            data
        )

        transformed_loc = map_merger.transform_location(
            data["agent_id"], data["location"]
        )
        data["transformed_loc"] = transformed_loc

        map_merger._visualize(
            map_A,
            loc_A,
            merged,
            data,
        )

    map_A = torch.load("/home/agilex2/projects/search/SemanticSearch/datadump/images/final_results/comm3/mymap_90.pt")
    map_B = torch.load("/home/agilex2/projects/search/SemanticSearch/datadump/images/final_results/comm3/othermap_90.pt")
    data = {
        "agent_id": 1,
        "map": map_B,
        "location": loc_B
    }

    transfomed_map = map_merger.get_transformed_map(
        global_map=map_A,
        neighbor_global_map=map_B,
        neighbor_id=1
    )
    data["transformed_map"] = transfomed_map
    if transfomed_map is None:
        map_merger._visualize_failed(
            map_A,
            map_B,
            loc_A,
            loc_B
        )
    else:
        merged = map_merger._merge(
            map_A,
            data
        )

        transformed_loc = map_merger.transform_location(
            data["agent_id"], data["location"]
        )
        data["transformed_loc"] = transformed_loc

        map_merger.timestep += 1
        map_merger._visualize(
            map_A,
            loc_A,
            merged,
            data,
        )