"""GOAT dataset reader (DM3Goat-v1)."""
import json
import os
from typing import List, Optional

from habitat.core.registry import registry
from habitat.datasets.goat.goat_dataset import GoatDatasetV1
from habitat.datasets.pointnav.pointnav_dataset import (
    CONTENT_SCENES_PATH_FIELD,
    DEFAULT_SCENE_PATH_PREFIX,
)
from habitat.tasks.nav.goat_task import GoatEpisode


@registry.register_dataset(name="DM3Goat-v1")
class DM3GoatDatasetV1(GoatDatasetV1):

    all_categories: List[str] = []

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        if len(deserialized["episodes"]) == 0:
            return

        if "goals" not in deserialized:
            deserialized = self.dedup_goals(deserialized)


        self.goals = deserialized["goals"]

        for i, composite_episode in enumerate(deserialized["episodes"]):
            composite_episode["goals"] = []
            composite_episode = GoatEpisode(**composite_episode)

            composite_episode.episode_id = str(i)

            if scenes_dir is not None:
                if composite_episode.scene_id.startswith(
                    DEFAULT_SCENE_PATH_PREFIX
                ):
                    composite_episode.scene_id = composite_episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                composite_episode.scene_id = os.path.join(
                    scenes_dir, "", composite_episode.scene_id
                )

            composite_episode.goals = []

            for goal in composite_episode.tasks:
                goal_type = goal[1]
                goal_category = goal[0]
                if goal_category not in self.all_categories:
                    self.all_categories.append(goal_category)
                goal_inst_id = goal[2]

                dset_same_cat_goals = [x for x in self.goals.values() if x[0]['object_category'] == goal_category]

                assert len(dset_same_cat_goals) == 1, f"more than 1 goal categories for {goal_category}"

                if goal_type == "object":
                    composite_episode.goals.append(dset_same_cat_goals[0])
                else:
                    goal_inst = [x for x in dset_same_cat_goals[0] if x['object_id'] == goal_inst_id]
                    assert len(goal_inst) == 1
                    composite_episode.goals.append(goal_inst)

            self.episodes.append(composite_episode)  # type: ignore [attr-defined]
