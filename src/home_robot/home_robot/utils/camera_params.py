from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from argparse import Namespace


@dataclass
class CameraParams:
    height: int
    width: int
    fx: float
    fy: float
    cx: float
    cy: float
    min_depth: float
    max_depth: float
    camera_height: float

    hfov: float = field(init=False)
    vfov: float = field(init=False)
    camera_matrix: Namespace = field(init=False)

    def __post_init__(self):
        self.hfov = np.degrees(2 * np.arctan(self.width / 2.0 / self.fx))
        self.vfov = np.degrees(2 * np.arctan(self.height / 2.0 / self.fy))
        self.camera_matrix = Namespace(xc=self.cx, zc=self.cy, f=self.fx)

    @classmethod
    def from_ros_camera(
        cls, camera_info: dict, camera_height: float, min_depth: float, max_depth: float
    ) -> CameraParams:
        """Real world: from camera.get_info()"""
        return cls(
            height=camera_info["height"],
            width=camera_info["width"],
            fx=camera_info["fx"],
            fy=camera_info["fy"],
            cx=camera_info["px"],
            cy=camera_info["py"],
            min_depth=min_depth,
            max_depth=max_depth,
            camera_height=camera_height,
        )

    @classmethod
    def from_habitat_config(
        cls, sensor_cfg, min_depth: float, max_depth: float
    ) -> CameraParams:
        """Sim: from habitat depth_sensor config. fx/fy derived from hfov."""
        hfov_rad = np.deg2rad(float(sensor_cfg.hfov))
        fx = (sensor_cfg.width / 2.0) / np.tan(hfov_rad / 2.0)
        aspect = sensor_cfg.height / sensor_cfg.width
        fy = (sensor_cfg.height / 2.0) / np.tan(
            np.arctan(np.tan(hfov_rad / 2.0) * aspect)
        )
        return cls(
            height=sensor_cfg.height,
            width=sensor_cfg.width,
            fx=fx,
            fy=fy,
            cx=(sensor_cfg.width - 1) / 2.0,
            cy=(sensor_cfg.height - 1) / 2.0,
            min_depth=min_depth,
            max_depth=max_depth,
            camera_height=sensor_cfg.position[1],
        )
