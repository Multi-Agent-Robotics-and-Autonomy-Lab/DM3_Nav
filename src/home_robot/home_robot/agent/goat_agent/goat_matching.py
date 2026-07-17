# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import clip
import torch
import logging
import matplotlib
import numpy as np
from tqdm import tqdm
from torchvision.transforms import ToPILImage
from typing import Any, Dict, List, Optional, Tuple, Union
from home_robot.agent.imagenav_agent.superglue import Matching
from home_robot.mapping.semantic.constants import MapConstants as MC
from home_robot.mapping.semantic.instance_tracking_modules import InstanceMemory

matplotlib.use("Agg")

class MatchingLogger(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return f"[MATCHING] {msg}", kwargs

class GoatMatching(Matching):
    def __init__(
        self,
        device: int,
        config: Dict[str, Any],
        default_vis_dir: str,
        print_images: bool,
        instance_memory: InstanceMemory,
        log,
        cat_match_threshold: float,
    ) -> None:
        super().__init__(device, config, default_vis_dir, print_images)

        self.score_func = config.score_function
        assert self.score_func in ["confidence_sum", "match_count"]

        # generate clip embeddings by loading clip model
        self.device = device
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device)
        self.instance_memory = instance_memory
        self.step = 0
        self.score_thresh = {
            "languagenav": config.score_thresh_lang,
            "imagenav": config.score_thresh_image,
            "objectnav": cat_match_threshold,
        }
        self.log = MatchingLogger(log, None) 


    def match_to_category(
        self, instances, global_pose
    ):
        #! myTODO: This is last steps global_pose, but I think it doesn't matter much. Ideally, I think we should do all these steps after SemMapModule.
        all_confidences = []
        for inst in instances:

            score = inst._get_score()


            #! myTODO: All following are based on how good the goal is, not "if the goal is correct". I should probably use both in the pipeline

            # # pick a view with maximum object coverage
            # best_view = np.argmax([view.object_coverage for view in inst.instance_views])

            # #1 Score based on coverage:
            # # score = instance_views[best_view].object_coverage

            # #2 Score based on distance:
            # instance_pose = inst.instance_views[best_view].pose
            # global_xy = global_pose[:2].cpu()
            # instance_xy = instance_pose[:2]
            # score = 1 / (torch.norm(global_xy - instance_xy).item()+1)

            # #3 Score based on distance and coverage:
            # #! myTODO: Very important because we should not go to poses were only a couple of pixels are from the object.
            # #! However, many times when we get close, we get better views which also have lower distances. 

            all_confidences.append([score])
        return all_confidences

    def match_instances_to_goal(
        self,
        instances,
        last_view,
        task,
        global_pose,
    ):
        candidate_instances = []
        for inst in instances:
            if inst.category_id != task.goal_semantic_id:
                continue
            candidate_instances.append(inst)

        valid_data = [
            (inst, views)
            for inst in candidate_instances
            if (views := inst._get_valid_views(self.instance_memory.images, last_view))
        ]

        candidate_instances, all_views = zip(*valid_data) if valid_data else ([], [])
        instance_view_counts = [len(v) for v in all_views]
        all_views = [x for views in all_views for x in views]

        confidences = []
        if len(all_views) > 0:
            if task.type == "imagenav":
                confidences = self.match_image_to_image(
                    all_views,
                    task.goal_image_processed,
                    task.goal_image_keypoints,
                )
            elif task.type == "languagenav":
                confidences = self.match_language_to_image(
                    all_views,
                    task.goal_description,
                )
            elif task.type == "objectnav":
                confidences = self.match_to_category(
                    candidate_instances, global_pose
                )
                instance_view_counts = [1]*len(candidate_instances)

            confidences = np.array(np.split(
                confidences, np.cumsum(instance_view_counts)[:-1]
            ))
            candidate_instances = [inst.id for inst in candidate_instances]
        else:
            candidate_instances = []
        return confidences, candidate_instances
    

    @torch.no_grad()
    def match_image_to_image(
        self,
        rgb_images: List[np.ndarray],
        goal_image: torch.Tensor,
        goal_image_keypoints: Dict[str, Any],
    ):
        """Computes and describes keypoints using SuperPoint and matches
        keypoints between an RGB image and a goal image using SuperGlue.
        Either goal_image or goal_image_keypoints must be provided.
        Returns:
            tensor of goal image keypoints
            tensor of rgb image keypoints
            tensor of keypoint matches
            tensor of match confidences
        """
        assert isinstance(goal_image, torch.Tensor) # Already prreprocessed
        assert isinstance(rgb_images, list)
        assert isinstance(rgb_images[0], np.ndarray)

        all_confidences = []

        # TODO Can we batch this for loop to speed it up? It is a bottleneck
        self.log.debug("Computing matching score with each view...")
        for i in range(len(rgb_images)):
            rgb_image_processed = self._preprocess_image(rgb_images[i])

            matcher_inputs = {
                "image0": goal_image,
                "image1": rgb_image_processed,
                **goal_image_keypoints,
            }
            pred = self.matcher(matcher_inputs)

            matches = pred["matches0"].cpu().numpy()
            confidence = pred["matching_scores0"].cpu().numpy()
            self._visualize(matcher_inputs, pred, f"{self.step}_{i}")

            confidence = confidence[matches != -1].sum().item()
            all_confidences.append(confidence)
        return all_confidences

    @torch.no_grad()
    def match_language_to_image(self, views_orig, language_goal, **kwargs):
        """Compute matching scores from a language goal to images."""
        batch_size = 64
        language_goal = language_goal.replace("Instruction: ", "")
        language_goal = clip.tokenize(language_goal).to(self.device)
        language_goal = self.clip_model.encode_text(language_goal)
        # get clip embedding for views with a batch size of batch_size

        views = views_orig
        if views[0].shape[0] == 3:
            views = torch.stack(
                [
                    self.clip_preprocess(
                        ToPILImage()(v.transpose(2, 1, 0).astype(np.uint8))
                    )
                    for v in views
                ],
                dim=0,
            )
        else:
            views = torch.stack(
                [self.clip_preprocess(ToPILImage()(v.astype(np.uint8))) for v in views],
                dim=0,
            )
        view_embeddings = torch.cat(
            [
                self.clip_model.encode_image(v.to(self.device))
                for v in views.split(batch_size)
            ],
            dim=0,
        )
        # normalize the embeddings
        view_embeddings = view_embeddings / view_embeddings.norm(dim=-1, keepdim=True)
        language_goal = language_goal / language_goal.norm(dim=-1, keepdim=True)
        # compute cosines similarity
        similarity = (language_goal @ view_embeddings.T).squeeze(0)
        return similarity.detach().cpu().numpy().flatten()

    def get_best_match(self, confidences, instance_ids, score_thresh, agg_fn):
        """instance_ids are global"""
        scores = self.aggregate_scores_per_instance(
            confidences, agg_fn
        )
        sorted_inst_ids = np.argsort(scores)[::-1]
        idx = 0
        self.log.debug(
            f"Getting best match. Scores: {scores}, instance_ids: {instance_ids}, score threshold: {score_thresh}."
        )
        self.log.debug(f"TEMP sorted_inst_ids {sorted_inst_ids}, {sorted_inst_ids[idx]}")
        while (
            idx < len(sorted_inst_ids) and scores[sorted_inst_ids[idx]] >= score_thresh
        ):
            self.log.debug(f"TEMP idx {idx}, sorted inst ids: {sorted_inst_ids}")
            inst_idx = sorted_inst_ids[idx]
            idx += 1
            self.log.debug(
                f"Trying to localize instance {instance_ids[inst_idx]} with score {scores[inst_idx]}"
            )
            best_instance_id = instance_ids[inst_idx]
            if best_instance_id == -1:
                # * The reason that this might happen is sometimes someobjects are overriding that instance when projecting into 2D (object with highest point is chosen in a specifc cell).
                self.log.debug("No id provided for this instance. Skipping.")
                continue
            # * We does not check it goal map is non-empty here. Somewehre else, I should make sure we can navigate to this goal.
            return best_instance_id, scores[inst_idx]

        self.log.debug("Goal does not match any instance.")
        return None, None

    def aggregate_scores_per_instance(self, confidences, agg_fn):
        agg_scores = []
        for inst_idx, inst_confidences in enumerate(confidences):
            if agg_fn == "max":
                agg_scores.append(np.max(inst_confidences))
            elif agg_fn == "mean":
                agg_scores.append(np.mean(inst_confidences))
            elif agg_fn == "median":
                agg_scores.append(np.median(inst_confidences))
            else:
                raise NotImplementedError
            # logger.debug(f"Instance {inst_idx+1} score: {max(inst_view_scores)}")
            self.log.debug(f"Instance {inst_idx+1} score: {agg_scores[-1]}")
        return agg_scores

    def search_for_goal(
        self,
        task,
        match_memory,
        global_pose,
        agg_fn: str = "max",
        score_thresh=None,
    ) -> int:
        """
        Searches for goal in current observation, and also in memory if specified. 
        """
        if score_thresh is None:
            score_thresh = self.score_thresh[task.type]

        mem_match_confidences, mem_match_instance_ids = [], []
        if match_memory:
            self.log.info("--------Matching against memory!--------")
            (
                mem_match_confidences,
                mem_match_instance_ids,
            ) = self.match_instances_to_goal(
                self.instance_memory.instances.values(),
                False,
                task,
                global_pose
            )

        #! myTODO: Should I overwrite Mem with obs, or otherwise?
        if len(mem_match_confidences) > 0:
            self.log.debug(
                f"Matching with memory: {len(mem_match_confidences)} instances"
            )
            inst_goal_id, best_score = self.get_best_match(
                mem_match_confidences, mem_match_instance_ids, score_thresh, agg_fn
            )
            if inst_goal_id is not None:
                self.log.info(f"Goal instance {inst_goal_id} found by matching with memory.")
                return inst_goal_id, best_score
            else:
                self.log.debug(f"No matches found in the memory")



        obs_match_confidences, obs_match_instance_ids = (
            self.match_instances_to_goal(
                [self.instance_memory.instances[k] for k in self.instance_memory.temp_id_to_global_id.values() if k!=0],
                True,
                task,
                global_pose
            )
        )

        self.log.debug(
            f"Matching with observation: {len(obs_match_confidences)} instances"
        )
        if len(obs_match_confidences) > 0:
            self.log.debug(
                f"Global instance ids: {obs_match_instance_ids}"
            )
            inst_goal_id, best_score = self.get_best_match(
                obs_match_confidences, obs_match_instance_ids, score_thresh, agg_fn
            )
            if inst_goal_id is not None:
                self.log.debug(
                    f"Goal instance {inst_goal_id} found in this step by matching with observation."
                )
                return inst_goal_id, best_score
            else:
                self.log.debug(f"No matches found with observation")

        return None, None
