"""Multi-agent sensor suite."""
from typing import Any, Dict, Iterable, List

from gym import spaces

from habitat.core.simulator import Observations, Sensor, SensorSuite


class MultiAgentSensorSuite:
    r"""Represents a set of sensors, with each sensor being identified
    through a unique id.
    """

    sensor_suites: Dict[str, SensorSuite]
    observation_spaces: Dict[str, spaces.Dict]

    def __init__(self, sensors: List[Iterable[Sensor]]) -> None:
        """Constructor

        :param sensors: list containing sensors for the environment, uuid of
            each sensor must be unique.
        """
        self.num_agents = len(sensors)
        self.observation_spaces = spaces.Dict()
        self.sensor_suites = {}
        for agent_id, agent_sensors in enumerate(sensors):
            agent_sensor_suit = SensorSuite(agent_sensors)
            self.sensor_suites[agent_id] = agent_sensor_suit
            self.observation_spaces[agent_id] = (
                agent_sensor_suit.observation_spaces
            )

    def get(self, uuid: str, agent_id: int) -> Sensor:
        return self.sensor_suits[agent_id].get(uuid)

    def get_observations(
        self, observations, *args: Any, **kwargs: Any
    ) -> Observations:
        r"""Collects data from all sensors and returns it packaged inside
        a dict of   agent_id -> :ref:`Observations`.
        """

        return {
            agent_id: sensor_suite.get_observations(
                observations[agent_id], *args, **kwargs
            ) 
            for agent_id, sensor_suite in self.sensor_suites.items() if agent_id in observations
        }
