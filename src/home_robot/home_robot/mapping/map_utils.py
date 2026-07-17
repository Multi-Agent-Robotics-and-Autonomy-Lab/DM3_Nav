# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torch import Tensor
import numpy as np


class MapSizeParameters:
    def __init__(self, resolution, map_size_cm, global_downscaling):
        self.resolution = resolution
        self.global_map_size_cm = map_size_cm
        self.global_downscaling = global_downscaling
        self.local_map_size_cm = self.global_map_size_cm // self.global_downscaling
        self.global_map_size = self.global_map_size_cm // self.resolution
        self.local_map_size = self.local_map_size_cm // self.resolution


def init_map_and_pose(
    map_size_parameters: MapSizeParameters,
    device: torch.device,
    num_channels
):
    """Initialize global and local map and sensor pose variables
    for a given environment.
    """
    p = map_size_parameters
    global_pose = torch.zeros(3, device=device)
    global_pose.fill_(0.0)
    global_pose[:2] = p.global_map_size_cm / 100.0 / 2.0

    # Initialize starting agent locations
    x, y = (global_pose[:2] * 100 / p.resolution).round().int()
    global_map = torch.zeros(
        num_channels,
        p.global_map_size,
        p.global_map_size,
        device=device,
    )
    global_map.fill_(0.0)
    global_map[2:4, y - 1 : y + 2, x - 1 : x + 2] = 1.0

    return [global_map, global_pose] + get_local_parameters_from_global_pose(
        global_map,
        global_pose,
        map_size_parameters,
    )


def get_local_parameters_from_global_pose(
    global_map: Tensor,
    global_pose: Tensor,
    map_size_parameters: MapSizeParameters,
):
    """
    Using a global pose, finds lmb, origins, local_map, and local_pose.
    """
    p = map_size_parameters
    global_loc = (global_pose[:2] * 100 / p.resolution).round().int()
    lmb = get_local_map_boundaries(global_loc, map_size_parameters)
    origins = torch.tensor(
        [
            lmb[2] * p.resolution / 100.0,
            lmb[0] * p.resolution / 100.0,
            0.0,
        ], device=global_map.device
    )
    local_map = global_map[:, lmb[0] : lmb[1], lmb[2] : lmb[3]]
    local_pose = global_pose - origins
    return [local_map, local_pose, lmb, origins]


def get_local_map_boundaries(
    global_loc: torch.IntTensor, map_size_parameters: MapSizeParameters
) -> torch.IntTensor:
    """Get local map boundaries from global sensor location."""
    p = map_size_parameters
    x, y = global_loc
    device, dtype = global_loc.device, global_loc.dtype

    if p.global_downscaling > 1:
        y1, x1 = y - p.local_map_size // 2, x - p.local_map_size // 2
        y2, x2 = y1 + p.local_map_size, x1 + p.local_map_size

        if y1 < 0:
            y1 = torch.tensor(0, device=device, dtype=dtype)
            y2 = torch.tensor(p.local_map_size, device=device, dtype=dtype)
        if y2 > p.global_map_size:
            y1 = torch.tensor(
                p.global_map_size - p.local_map_size, device=device, dtype=dtype
            )
            y2 = torch.tensor(p.global_map_size, device=device, dtype=dtype)

        if x1 < 0:
            x1 = torch.tensor(0, device=device, dtype=dtype)
            x2 = torch.tensor(p.local_map_size, device=device, dtype=dtype)
        if x2 > p.global_map_size:
            x1 = torch.tensor(
                p.global_map_size - p.local_map_size, device=device, dtype=dtype
            )
            x2 = torch.tensor(p.global_map_size, device=device, dtype=dtype)

    else:
        y1 = torch.tensor(0, device=device, dtype=dtype)
        y2 = torch.tensor(p.global_map_size, device=device, dtype=dtype)
        x1 = torch.tensor(0, device=device, dtype=dtype)
        x2 = torch.tensor(p.global_map_size, device=device, dtype=dtype)

    return torch.stack([y1, y2, x1, x2])
