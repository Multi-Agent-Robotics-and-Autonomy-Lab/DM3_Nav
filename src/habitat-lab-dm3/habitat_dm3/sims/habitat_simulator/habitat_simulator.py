"""Multi-agent simulator (MultiAgentSim-v0)."""
from typing import List, Optional

import habitat_sim
from gym import spaces
from omegaconf import DictConfig

from habitat.config.default import get_agent_config
from habitat.core.registry import registry
from habitat.core.simulator import Observations
from habitat.sims.habitat_simulator.habitat_simulator import (
    HabitatSim,
    HabitatSimSensor,
    overwrite_config,
)
from habitat_dm3.core.simulator import MultiAgentSensorSuite


@registry.register_simulator(name="MultiAgentSim-v0")
class MultiAgentHabitatSim(HabitatSim):
    r"""A Multi Agent Simulator wrapper over habitat-sim
    """
    def __init__(self, config: DictConfig) -> None:
        self.habitat_config = config

        self.num_agents = len(self.habitat_config.agents)
        sensors = []
        for agent_config in self.habitat_config.agents.values():
            agent_sensors = []
            for sensor_cfg in agent_config.sim_sensors.values():
                sensor_type = registry.get_sensor(sensor_cfg.type)

                assert (
                    sensor_type is not None
                ), "invalid sensor type {}".format(sensor_cfg.type)
                sensor = sensor_type(sensor_cfg)
                agent_sensors.append(sensor)
            sensors.append(agent_sensors)

        self._sensor_suite = MultiAgentSensorSuite(sensors)

        self.sim_config = self.create_sim_config(self._sensor_suite)
        self._current_scene = self.sim_config.sim_cfg.scene_id
        habitat_sim.Simulator.__init__(self, self.sim_config)

        obj_attr_mgr = self.get_object_template_manager()
        for path in self.habitat_config.additional_object_paths:
            obj_attr_mgr.load_configs(path)
        self._action_space = spaces.Discrete(
            len(
                self.sim_config.agents[
                    self.habitat_config.default_agent_id
                ].action_space
            )
        )
        self._prev_sim_obs: Optional[Observations] = None

    def create_sim_config(
        self, _sensor_suite: MultiAgentSensorSuite
    ) -> habitat_sim.Configuration:
        sim_config = habitat_sim.SimulatorConfiguration()
        # Check if Habitat-Sim is post Scene Config Update
        if not hasattr(sim_config, "scene_id"):
            raise RuntimeError(
                "Incompatible version of Habitat-Sim detected, please upgrade habitat_sim"
            )
        overwrite_config(
            config_from=self.habitat_config.habitat_sim_v0,
            config_to=sim_config,
            # Ignore key as it gets propogated to sensor below
            ignore_keys={"gpu_gpu"},
        )
        sim_config.scene_dataset_config_file = (
            self.habitat_config.scene_dataset
        )
        sim_config.scene_id = self.habitat_config.scene

        agents_config = []
        for agent_id, agent_sensor_suite in _sensor_suite.sensor_suites.items():
            agent_config = habitat_sim.AgentConfiguration()
            sensor_specifications = []
            overwrite_config(
                config_from=get_agent_config(self.habitat_config, agent_id),
                config_to=agent_config,
                # These keys are only used by Hab-Lab
                ignore_keys={
                    "is_set_start_state",
                    # This is the Sensor Config. Unpacked below
                    "sensors",
                    "sim_sensors",
                    "start_position",
                    "start_rotation",
                    "robot_urdf",
                    "robot_type",
                    "joint_start_noise",
                    "ik_arm_urdf",
                },
            )
            for sensor in agent_sensor_suite.sensors.values():
                assert isinstance(sensor, HabitatSimSensor)
                sim_sensor_cfg = sensor._get_default_spec()  # type: ignore[operator]
                overwrite_config(
                    config_from=sensor.config,
                    config_to=sim_sensor_cfg,
                    # These keys are only used by Hab-Lab
                    # or translated into the sensor config manually
                    ignore_keys=sensor._config_ignore_keys,
                    # TODO consider making trans_dict a sensor class var too.
                    trans_dict={
                        "sensor_model_type": lambda v: getattr(
                            habitat_sim.FisheyeSensorModelType, v
                        ),
                        "sensor_subtype": lambda v: getattr(
                            habitat_sim.SensorSubType, v
                        ),
                    },
                )
                sim_sensor_cfg.uuid = sensor.uuid
                sim_sensor_cfg.resolution = list(
                    sensor.observation_space.shape[:2]
                )

                # TODO(maksymets): Add configure method to Sensor API to avoid
                # accessing child attributes through parent interface
                # We know that the Sensor has to be one of these Sensors
                sim_sensor_cfg.sensor_type = sensor.sim_sensor_type
                sim_sensor_cfg.gpu2gpu_transfer = (
                    self.habitat_config.habitat_sim_v0.gpu_gpu
                )
                sensor_specifications.append(sim_sensor_cfg)

            agent_config.sensor_specifications = sensor_specifications
            agent_config.action_space = registry.get_action_space_configuration(
                self.habitat_config.action_space_config
            )(self.habitat_config).get()
            agents_config.append(agent_config)

        return habitat_sim.Configuration(sim_config, agents_config)

    def reset(self) -> Observations:
        sim_obs = habitat_sim.Simulator.reset(self, list(range(len(self.agents))))
        if self._update_agents_state():
            sim_obs = self.get_sensor_observations(list(range(len(self.agents))))

        self._prev_sim_obs = sim_obs
        return self._sensor_suite.get_observations(sim_obs)
    
    def get_all_observations(self):
        return self._sensor_suite.get_observations(self._prev_sim_obs)


    @property
    def previous_step_collided(self):
        r"""Whether or not the previous step resulted in a collision

        Returns:
            bool: True if the previous step resulted in a collision, false otherwise

        Warning:
            This feild is only updated when :meth:`step`, :meth:`reset`, or :meth:`get_observations_at` are
            called.  It does not update when the agent is moved to a new loction.  Furthermore, it
            will _always_ be false after :meth:`reset` or :meth:`get_observations_at` as neither of those
            result in an action (step) being taken.
        """
        
        prev_step_collided = [self._prev_sim_obs[agent_id].get("collided", False) for agent_id in self._prev_sim_obs.keys()]
        if len(prev_step_collided) == 1:
            prev_step_collided = prev_step_collided[0]
        return prev_step_collided

    acting_agent_id: int = 0  # set by MultiAgentEmbodiedTask.step

    def step(self, action):
        if not isinstance(action, dict):
            action = {self.acting_agent_id: action}
        return super().step(action)

    def get_observations_at(
        self,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        keep_agent_at_new_pose: bool = False,
        agent_id: Optional[int] = -1,
    ) -> Optional[Observations]:
        if agent_id == -1:
            agent_id = self.acting_agent_id
        single_agent = False
        if agent_id is None:
            single_agent = True
            agent_id = 0
        current_state = self.get_agent_state(agent_id)
        if position is None or rotation is None:
            success = True
        else:
            success = self.set_agent_state(
                position, rotation, agent_id, reset_sensors=False
            )

        if success:
            sim_obs = self.get_sensor_observations(agent_ids=[agent_id])

            if single_agent:
                sim_obs = sim_obs[agent_id]
                self._prev_sim_obs = sim_obs
            else:
                self._prev_sim_obs[agent_id] = sim_obs[agent_id]

            observations = self._sensor_suite.get_observations(sim_obs)
            if not keep_agent_at_new_pose:
                self.set_agent_state(
                    current_state.position,
                    current_state.rotation,
                    agent_id,
                    reset_sensors=False,
                )
            return observations
        else:
            return None
