"""Multi-agent embodied task."""
from typing import Any, List

from habitat.core.dataset import Episode
from habitat.core.embodied_task import EmbodiedTask


class MultiAgentEmbodiedTask(EmbodiedTask):
    def step(self, action: List[Any], episode: Episode):
        observations = {}
        for a in action:
            self._sim.acting_agent_id = a["action_args"]["agent_id"]
            self._step_single_action(
                observations, a["action"], a, episode
            )

        observations.update(
            self.sensor_suite.get_observations(
                    observations=observations,
                    episode=episode,
                    action=action,
                    task=self,
                )
        )
        self._is_episode_active = self._check_episode_is_active(
            observations=observations, action=action, episode=episode
        )
        return observations
