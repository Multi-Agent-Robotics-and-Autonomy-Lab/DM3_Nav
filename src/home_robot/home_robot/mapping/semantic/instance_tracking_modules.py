import os
import shutil
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# MIN_PIXELS = 1000
# MIN_EDGE = 15
print("DONT FORGET THIS CHANGE")
MIN_PIXELS = 50
MIN_EDGE = 10

class InstanceView:
    """
    Stores information about a single view of an instance

    bbox: bounding box of instance in the current image
    timestep: timestep at which the current view was recorded
    cropped_image: cropped image of instance in the current image
    embedding: embedding of instance in the current image
    mask: mask of instance in the current image
    point_cloud: point cloud of instance in the current image
    category_id: category id of instance in the current image
    """

    bbox: Tuple[int, int, int, int]
    timestep: int
    cropped_image: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None
    # mask of instance in the current image
    mask: np.ndarray = None
    # point cloud of instance in the current image
    point_cloud: np.ndarray = None
    category_id: Optional[int] = None
    pose: np.ndarray = None
    instance_id: Optional[int] = None
    object_coverage: Optional[int] = None
    score: float = None

    def __init__(
        self,
        bbox,
        timestep,
        cropped_image,
        embedding,
        mask,
        point_cloud,
        pose,
        object_coverage,
        score,
        category_id,
    ):
        """
        Initialize InstanceView
        """
        self.bbox = bbox
        self.timestep = timestep
        self.cropped_image = cropped_image
        self.embedding = embedding
        self.mask = mask
        self.point_cloud = point_cloud
        self.pose = pose
        self.category_id = category_id
        self.object_coverage = object_coverage
        self.score = score


class Instance:
    """
    A single instance found in the environment. Each instance is composed of a list of InstanceView objects, each of which is a view of the instance at a particular timestep.
    """

    def __init__(self):
        """
        Initialize Instance

        name: name of instance
        category_id: category id of instance
        instance_views: list of InstanceView objects
        """
        self.id = None
        self.name = None
        self.category_id = None
        self.instance_views = []

    def _get_valid_views(self, all_images, last_view):
        views = []
        all_views = [self.instance_views[-1]] if last_view else self.instance_views
        for inst_view in all_views:
            # Note: Using bbox shape instead of cropped image shape, because cropped image doesn't always add a fixed padding.
            bbox_shape =  inst_view.bbox[1] - inst_view.bbox[0]
            # logger.debug(f"Evaluating instance {self.id}")
            # logger.debug(f"Total pixels in cropped image: {bbox_shape.prod()} ? {MIN_PIXELS}")
            # logger.debug(f"Minimum edge size in cropped image: {bbox_shape} ? {MIN_EDGE} : {(bbox_shape < MIN_EDGE).any()}")
            if bbox_shape.prod() < MIN_PIXELS or (bbox_shape < MIN_EDGE).any():
                continue
            if last_view:
                img = all_images[inst_view.timestep].cpu().numpy()
                img = np.transpose(img, (1, 2, 0))
            else:
                img = inst_view.cropped_image

            views.append(img)
        return views
    
    def _get_score(self, last_k=10, agg="mean"):
        scores = []
        for inst_view in self.instance_views:
            scores.append(inst_view.score)
        

        # scores = scores[-last_k:]
        # return np.mean(scores)

        return np.max(scores)



class InstanceMemory:
    """
    InstanceMemory stores information about instances found in the environment. It stores a list of Instance objects, each of which is a single instance found in the environment.

    images: list of egocentric images at each timestep
    instance_views: list of InstanceView objects at each timestep
    point_cloud: list of point clouds at each timestep
    unprocessed_views: list of unprocessed InstanceView objects at each timestep, before they are added to an Instance object
    timesteps: list of timesteps
    """

    images: List[torch.Tensor] = list()
    instances: Dict[int, Instance] = dict()
    point_cloud: List[torch.Tensor] = None
    unprocessed_views: Dict[int, InstanceView] = dict()
    temp_id_to_global_id: Dict[int, int] = dict()
    timesteps: int = 0

    def __init__(
        self,
        config=None,
        save_dir="instances",
        mask_cropped_instances=False,
        padding_cropped_instances=0,
        category_id_to_category_name=None,
        log=None
    ):
        self.du_scale = config.AGENT.SEMANTIC_MAP.du_scale
        self.print_images = config.VISUALIZATION_LEVEL > 2
        self.mask_cropped_instances = mask_cropped_instances
        self.padding_cropped_instances = padding_cropped_instances
        self.category_id_to_category_name = category_id_to_category_name

        if config is not None:
            self.save_dir = os.path.join(
                config.DUMP_LOCATION, "instances", config.EXP_NAME
            )
        else:
            self.save_dir = save_dir

        if self.print_images:
            shutil.rmtree(self.save_dir, ignore_errors=True)

        self.reset()
        self.log = log

    def reset(self):
        self.images = []
        self.point_cloud = []
        self.instances = dict()
        self.unprocessed_views = dict()
        self.temp_id_to_global_id = dict()
        self.timesteps = 0

    def update_temp_id(self, temp_id: int, global_instance_id: int):
        # fetch instance view from the list of unprocessed views
        # if global_instance_id already exists, add a new instance view to it
        # otherwise, create a new global instance with the given global_instance_id

        # get instance view
        instance_view = self.unprocessed_views.get(temp_id, None)
        assert not instance_view is None

        # get global instance
        global_instance = self.instances.get(global_instance_id, None)
        if global_instance is None:
            # create a new global instance
            global_instance = Instance()
            global_instance.id = global_instance_id
            global_instance.category_id = instance_view.category_id
            global_instance.instance_views.append(instance_view)
            self.instances[global_instance_id] = global_instance
        else:
            # add instance view to global instance
            global_instance.instance_views.append(instance_view)
        self.temp_id_to_global_id[int(temp_id)] = global_instance_id
        if self.print_images:
            category_name = (
                f"cat_{instance_view.category_id}"
                if self.category_id_to_category_name is None
                else self.category_id_to_category_name[instance_view.category_id]
            )
            instance_write_path = os.path.join(
                self.save_dir, f"{global_instance_id}_{category_name}"
            )
            os.makedirs(instance_write_path, exist_ok=True)

            step = instance_view.timestep
            full_image = self.images[step]
            full_image = full_image.numpy().astype(np.uint8).transpose(1, 2, 0)
            full_image = full_image[..., ::-1]
            # overlay mask on image
            mask = np.zeros(full_image.shape, full_image.dtype)
            mask[:, :] = (0, 0, 255)
            mask = cv2.bitwise_and(mask, mask, mask=instance_view.mask.astype(np.uint8))
            masked_image = cv2.addWeighted(mask, 1, full_image, 1, 0)
            cv2.imwrite(
                os.path.join(
                    instance_write_path,
                    f"step_{self.timesteps}_local_id_{temp_id}.png",
                ),
                masked_image,
            )
    def process_instances(
        self,
        semantic_frame_onehot: torch.Tensor,
        instance_frame_onehot: torch.Tensor,
        instance_scores,
        category_scores,
        point_cloud: torch.Tensor,
        pose: torch.Tensor,
        image: torch.Tensor,
    ):
        instance_frame = instance_frame_onehot.argmax(dim=0).int() + 1
        no_instance_mask = instance_frame_onehot.sum(0) == 0
        instance_frame[no_instance_mask] = 0
        max_vals, semantic_frame = semantic_frame_onehot.max(dim=0)
        semantic_frame[max_vals == 0] = -1 
        semantic_frame = semantic_frame.int() + 1
        
        self.images.append(image.detach().cpu())
        self.point_cloud.append(point_cloud.unsqueeze(0).detach().cpu())
        
        pose_cpu = pose.cpu()
        
        # temp_instance_ids = torch.unique(instance_frame)
        temp_instance_ids = torch.nonzero(
            instance_frame_onehot.flatten(1).sum(1), as_tuple=False
        ).squeeze(1) + 1

        instance_frame_downsampled = torch.nn.functional.interpolate(
            instance_frame.unsqueeze(0).unsqueeze(0).float(),
            scale_factor=1 / self.du_scale,
            mode="nearest",
        ).squeeze(0).squeeze(0).int()
        
        # self.log.debug(f"In process instances")
        for temp_instance_id in temp_instance_ids:
            assert temp_instance_id != 0
            
            instance_mask = instance_frame == temp_instance_id
            
            category_id = semantic_frame[instance_mask].unique()
            category_id = category_id[0].item()
            # self.log.debug(f"Temp instance id: {temp_instance_id}, category id: {category_id}")
            
            if category_id == 0:
                continue
            
            # bbox = (
            #     torch.stack(
            #         [
            #             instance_mask.nonzero().min(dim=0)[0],
            #             instance_mask.nonzero().max(dim=0)[0] + 1,
            #         ]
            #     )
            #     .cpu()
            #     .numpy()
            # )
            rows = instance_mask.any(dim=1).nonzero().flatten()
            cols = instance_mask.any(dim=0).nonzero().flatten()

            bbox = np.array([
                [rows.min().item(), cols.min().item()],
                [rows.max().item() + 1, cols.max().item() + 1]
            ])

            
            instance_mask_downsampled = instance_frame_downsampled == temp_instance_id
            # instance_mask_downsampled = (
            #     torch.nn.functional.interpolate(
            #         instance_mask.unsqueeze(0).unsqueeze(0).float(),
            #         scale_factor=1 / self.du_scale,
            #         mode="nearest",
            #     )
            #     .squeeze(0)
            #     .squeeze(0)
            #     .bool()
            # )
            
            if self.mask_cropped_instances:
                masked_image = image * instance_mask
            else:
                masked_image = image
            
            p = self.padding_cropped_instances
            h, w = masked_image.shape[1:]
            cropped_image = (
                masked_image[
                    :,
                    max(bbox[0, 0] - p, 0) : min(bbox[1, 0] + p, h),
                    max(bbox[0, 1] - p, 0) : min(bbox[1, 1] + p, w),
                ]
                .permute(1, 2, 0)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            
            instance_mask_cpu = instance_mask.cpu().numpy().astype(bool)
            
            embedding = None
            
            point_cloud_instance = point_cloud[instance_mask_downsampled.cpu().numpy()]
            
            object_coverage = np.sum(instance_mask_cpu) / instance_mask_cpu.size

            if category_scores is None:
                score = instance_scores[temp_instance_id-1]
            else:
                score = (instance_scores[temp_instance_id-1] + category_scores.get(category_id, 0))/2

            
            instance_view = InstanceView(
                bbox=bbox,
                timestep=self.timesteps,
                cropped_image=cropped_image,
                embedding=embedding,
                mask=instance_mask_cpu,
                point_cloud=point_cloud_instance.cpu().numpy(),
                category_id=category_id,
                pose=pose_cpu,
                object_coverage=object_coverage,
                score=score,
            )
            
            self.unprocessed_views[temp_instance_id.item()] = instance_view
        
            #! myTODO: Add a variable to control if we should save these or not.
            # save cropped image with timestep in filename
            # os.makedirs(f"{self.save_dir}/all", exist_ok=True)
            # cv2.imwrite(
            #     f"{self.save_dir}/all/{self.timesteps + 1}_{instance_id.item()}.png",
            #     cropped_image[:,:,::-1],
            # )

        self.timesteps += 1