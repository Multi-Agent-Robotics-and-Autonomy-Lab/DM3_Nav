"""Configs of the multi-agent GOAT measures and stop action."""
import os

import attr
from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.config_store import ConfigStore
from hydra.plugins.search_path_plugin import SearchPathPlugin

from habitat.config.default_structured_configs import (
    ActionConfig,
    MeasurementConfig,
    register_hydra_plugin,
)

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

cs = ConfigStore.instance()


@attr.s(auto_attribs=True, slots=True)
class MultiAgentGOATStopActionConfig(ActionConfig):
    type: str = "MultiAgentGOATStopAction"


@attr.s(auto_attribs=True, slots=True)
class MultiAgentGOATSuccessMeasurementConfig(MeasurementConfig):
    type: str = "MultiAgentGOATSuccess"
    success_distance: float = 0.2


@attr.s(auto_attribs=True, slots=True)
class MultiAgentGOATSPLMeasurementConfig(MeasurementConfig):
    type: str = "MultiAgentGOATSPL"


@attr.s(auto_attribs=True, slots=True)
class MultiAgentGOATDistanceToSubGoalMeasurementConfig(MeasurementConfig):
    type: str = "MultiAgentGOATDistanceToSubGoal"
    distance_to: str = "POINT"
    goals_attr: str = "goals"
    distance_from: str = "BASE"


cs.store(
    package="habitat.task.actions.multiagent_goat_stop",
    group="habitat/task/actions",
    name="multiagent_goat_stop",
    node=MultiAgentGOATStopActionConfig,
)
cs.store(
    package="habitat.task.measurements.multiagent_goat_distance_to_sub-goal",
    group="habitat/task/measurements",
    name="multiagent_goat_distance_to_sub-goal",
    node=MultiAgentGOATDistanceToSubGoalMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.multiagent_goat_success",
    group="habitat/task/measurements",
    name="multiagent_goat_success",
    node=MultiAgentGOATSuccessMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.multiagent_goat_spl",
    group="habitat/task/measurements",
    name="multiagent_goat_spl",
    node=MultiAgentGOATSPLMeasurementConfig,
)



class DM3ConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="habitat_dm3", path=f"file://{CONFIG_DIR}")


register_hydra_plugin(DM3ConfigPlugin)
