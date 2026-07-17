# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import numpy as np
from PIL import Image
from typing import Tuple
import matplotlib.pyplot as plt


def show_image(rgb):
    """Simple helper function to show images"""
    plt.figure()
    plt.imshow(rgb)
    plt.show()


def show_image_with_mask(rgb, mask):
    """tool for showing a mask and some other stuff"""
    plt.figure()
    plt.subplot(131)
    plt.imshow(rgb)
    plt.subplot(132)
    plt.imshow(mask)
    plt.subplot(133)
    _mask = mask[:, :, None]
    _mask = np.repeat(_mask, 3, axis=-1)
    plt.imshow(_mask * rgb / 255.0)
    plt.show()


def get_contour_points(
    pos: Tuple[float, float, float],
    origin: Tuple[float, float],
    size: int = 20,
) -> np.ndarray:
    x, y, o = pos
    pt1 = (int(x) + origin[0], int(y) + origin[1])
    pt2 = (
        int(x + size / 1.5 * np.cos(o + np.pi * 4 / 3)) + origin[0],
        int(y + size / 1.5 * np.sin(o + np.pi * 4 / 3)) + origin[1],
    )
    pt3 = (int(x + size * np.cos(o)) + origin[0], int(y + size * np.sin(o)) + origin[1])
    pt4 = (
        int(x + size / 1.5 * np.cos(o - np.pi * 4 / 3)) + origin[0],
        int(y + size / 1.5 * np.sin(o - np.pi * 4 / 3)) + origin[1],
    )

    return np.array([pt1, pt2, pt3, pt4])


def draw_line(
    start: Tuple[int, int],
    end: Tuple[int, int],
    mat: np.ndarray,
    steps: int = 25,
    w: int = 1,
) -> np.ndarray:
    for i in range(steps + 1):
        x = int(np.rint(start[0] + (end[0] - start[0]) * i / steps))
        y = int(np.rint(start[1] + (end[1] - start[1]) * i / steps))
        mat[x - w : x + w, y - w : y + w] = 1
    return mat


def visualize_map(input_shape, dir, name, features=None, points=None, traversible=None, goal_map=None, dilated_goal_map=None, frontier_map=None):
    shape = input_shape + (3,)
    white = np.ones(shape, dtype=np.uint8) * 255

    if traversible is not None:
        white[traversible == 0] = [0, 0, 0] # black

    if dilated_goal_map is not None:
        white[dilated_goal_map == 1] = [255, 0, 255] # magenta
    
    if goal_map is not None:
        white[goal_map == 1] = [0, 0, 255] # red

    if frontier_map is not None:
        white[frontier_map == 1] = [255, 255, 0] # cyan

    if features is not None:
        for feature_map, color in features:
            if feature_map is not None:
                white[feature_map == 1] = color
    
    if points is not None:
        for point, color in points:
            white[point[0], point[1]] = color

    white = np.flipud(white)
    # logger.debug(f"SAVING 4.dilate")
    cv2.imwrite(
        os.path.join(dir, name),
        white,
    )


def visualize_frontier_scores_matplotlib(
    dir,
    traversible,
    frontier_map,
    frontier_centers,
    frontier_scores,
    top_k_semantic_classes,
    robot_loc=None,
    top_k=3,
    save_path="frontiers_with_scores.png"
):
    fig, ax = plt.subplots(figsize=(8, 8))
    img = np.ones(traversible.shape + (3,), dtype=np.uint8) * 255
    img[traversible == 0] = [0, 0, 0]  # obstacles as black
    img[frontier_map == 1] = [0, 255, 255]  # frontiers as cyan

    img = np.flipud(img)

    ax.imshow(img)

    # Sort and select top-k frontiers by score
    # top_indices = np.argsort(frontier_scores)[-top_k:]
    top_indices = np.argsort(frontier_scores)

    for i in top_indices:
        center = frontier_centers[i]
        flipped_center = (center[1], traversible.shape[0] - center[0])

        ax.plot(flipped_center[0], flipped_center[1], "o", color="red", markersize=5)

        # annotation text
        score_text = f"{frontier_scores[i]:.2f}"
        sem_classes = top_k_semantic_classes[i]
        sem_text = ", ".join(f"#{cls}" for cls in sem_classes)

        full_text = f"{score_text}\n{sem_text}"

        # annotation with offset
        ax.annotate(
            full_text,
            xy=flipped_center,
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=5,
            color="black",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="gray", lw=1),
        )

    if robot_loc is not None:
        flipped_robot = (robot_loc[1], traversible.shape[0] - robot_loc[0])
        ax.plot(flipped_robot[0], flipped_robot[1], "x", color="blue", markersize=8, label="Robot")

    ax.set_title("Top Frontier Scores")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(dir,save_path), dpi=300)
    plt.close()


def visualize_frontiers(
    dir,
    traversible,
    frontier_map,
    frontier_centers,
    frontier_scores,
    frontier_texts,
    robot_loc=None,
    neighbor_locs=None,
    top_k=3,
    save_path="frontiers_with_scores.png"
):
    """
    General visualization function for frontiers.

    Args:
        dir: directory to save the plot
        traversible: 2D numpy array of traversible map (0 = obstacle, 1 = free)
        frontier_map: 2D numpy array marking frontiers
        frontier_centers: list of (row, col) frontier center coordinates
        frontier_scores: list or array of scores for each frontier
        frontier_texts: list of strings (same length as frontier_centers), label for each frontier
        robot_loc: optional (row, col) of robot location
        top_k: number of frontiers to display (highest scores)
        save_path: filename for saving
    """
    assert len(frontier_centers) == len(frontier_scores)
    fig, ax = plt.subplots(figsize=(8, 8))
    img = np.ones(traversible.shape + (3,), dtype=np.uint8) * 255
    img[traversible == 0] = [0, 0, 0]        # obstacles black
    img[frontier_map == 1] = [0, 255, 255]   # frontiers cyan
    img = np.flipud(img)

    ax.imshow(img)

    top_indices = np.argsort(frontier_scores)[::-1]

    for order, i in enumerate(top_indices):
        center = frontier_centers[i]
        flipped_center = (center[1], traversible.shape[0] - center[0])

        ax.plot(flipped_center[0], flipped_center[1], "o", color="red", markersize=5)

        if order >= top_k:
            continue

        # annotation with offset
        ax.annotate(
            frontier_texts[i],
            xy=flipped_center,
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=5,
            color="black",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="gray", lw=1),
        )

    if robot_loc is not None:
        flipped_robot = (robot_loc[1], traversible.shape[0] - robot_loc[0])
        ax.plot(flipped_robot[0], flipped_robot[1], "x", color="blue", markersize=8, label="Robot")

    if neighbor_locs is not None:
        for loc in neighbor_locs.values():
            flipped_robot = (loc[1], traversible.shape[0] - loc[0])
            ax.plot(flipped_robot[0], flipped_robot[1], "x", color="green", markersize=8, label="Neighbor")

    ax.set_title("Top Frontier Scores")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(dir, save_path), dpi=300)
    plt.close()


def visualize_semantic_frontiers(
    dir,
    traversible,
    frontier_map,
    frontier_centers,
    frontier_scores,
    top_k_semantic_classes,
    robot_loc=None,
    top_k=3,
    save_path="frontiers_with_scores.png"
):
    """
    Wrapper function that builds semantic labels and calls visualize_frontiers.
    """
    frontier_texts = []
    for score, sem_classes in zip(frontier_scores, top_k_semantic_classes):
        score_text = f"{score:.2f}"
        sem_text = ", ".join(f"#{cls}" for cls in sem_classes)
        full_text = f"{score_text}\n{sem_text}"
        frontier_texts.append(full_text)

    visualize_frontiers(
        dir=dir,
        traversible=traversible,
        frontier_map=frontier_map,
        frontier_centers=frontier_centers,
        frontier_scores=frontier_scores,
        frontier_texts=frontier_texts,
        robot_loc=robot_loc,
        top_k=top_k,
        save_path=save_path,
    )

def visualize_distance_frontiers(
    dir,
    traversible,
    frontier_map,
    frontier_centers,
    frontier_scores,
    agent_dists,
    other_agents_dists,
    robot_loc=None,
    neighbor_locs=None,
    top_k=3,
    save_path="frontiers_with_agent_dists.png"
):
    """
    Wrapper function that builds labels including agent distance,
    other agents' distances, and final score.
    """
    frontier_texts = []
    num_frontiers = len(frontier_scores)

    for i in range(num_frontiers):
        score_text = f"Score: {frontier_scores[i]:.2f}"
        agent_text = f"MyDist: {agent_dists[i]:.2f}"

        others_text = ""
        if len(other_agents_dists) > 0:
            others = [f"{d:.2f}" for d in other_agents_dists[i]]
            others_text = "Others: " + ", ".join(others) if others else "Others: -"

        full_text = f"{agent_text}\n{others_text}\n{score_text}"
        frontier_texts.append(full_text)

    visualize_frontiers(
        dir=dir,
        traversible=traversible,
        frontier_map=frontier_map,
        frontier_centers=frontier_centers,
        frontier_scores=frontier_scores,
        frontier_texts=frontier_texts,
        robot_loc=robot_loc,
        neighbor_locs=neighbor_locs,
        top_k=top_k,
        save_path=save_path,
    )

#! THis is inside habitat goat env:
def visualize_semantic_with_labels(
    semantic_array: np.ndarray,
    palette: list,
    save_path: str,
    label_min_pixels: int = 50,
    font_scale: float = 0.4,
    thickness: int = 1,
):
    """
    Visualizes a semantic map with color palette and overlays ID numbers on each region.

    Args:
        semantic_array (np.ndarray): 2D array of shape (H, W) with semantic IDs.
        palette (list): A flat list of RGB values, e.g., [R0, G0, B0, R1, G1, B1, ...].
        save_path (str): File path to save the resulting image (e.g., 'out.png').
        label_min_pixels (int): Minimum number of pixels required to label a region.
        font_scale (float): Font scale for the overlaid text.
        thickness (int): Text thickness.
    """
    # So slow, just for debugging purposes
    # scale_factor = 4
    # semantic_array= cv2.resize(
    #     semantic_array, 
    #     (semantic_array.shape[1]*scale_factor, semantic_array.shape[0]*scale_factor),
    #     interpolation=cv2.INTER_NEAREST
    # )

    palette_img = Image.new("P", (semantic_array.shape[1], semantic_array.shape[0]))
    palette_img.putpalette(palette)
    palette_img.putdata(semantic_array.flatten().astype(np.uint8))
    palette_img = palette_img.convert("RGB")

    semantic_map_cv = np.array(palette_img)[
        :, :, ::-1
    ].copy()  # RGB -> BGR, and make it OpenCV-safe

    unique_ids = np.unique(semantic_array)
    for sid in unique_ids:
        coords = np.argwhere(semantic_array == sid)
        # center_coord = coords[len(coords) // 2]
        center_coord = np.mean(coords, axis=0).astype(int)
        if len(coords) < label_min_pixels:
            continue
        y, x = center_coord[:2]
        cv2.putText(
            semantic_map_cv,
            str(int(sid)),
            (x, y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=(255, 0, 0),
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    cv2.imwrite(save_path, semantic_map_cv)

def visualize_depth_filter(
    rgb: np.ndarray,
    robot_mask: np.ndarray,
    depth: np.ndarray,
    save_dir: str,
    timestep: int,
    max_depth: float = 5.0,
):
    os.makedirs(save_dir, exist_ok=True)

    # Depth visualization
    depth_vis = depth.copy()
    depth_vis[depth_vis > max_depth] = 0.0
    dmax = depth_vis.max()
    if dmax > 0:
        depth_vis = depth_vis / dmax * 255.0
    depth_vis = depth_vis.astype(np.uint8)

    # RGB with mask overlay in red
    rgb_vis = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    rgb_vis[robot_mask] = [0, 0, 255]

    combined = np.hstack([
        rgb_vis,
        cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR),
    ])

    cv2.imwrite(
        os.path.join(save_dir, f"{timestep}_16.depth_robot_mask.png"),
        combined,
    )