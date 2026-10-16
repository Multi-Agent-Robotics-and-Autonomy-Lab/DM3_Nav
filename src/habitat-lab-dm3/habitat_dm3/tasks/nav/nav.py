"""Navigation sensors, measures and tasks (DM3X extends habitat's X)."""
from typing import TYPE_CHECKING, Any, Optional

import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
import quaternion
from gurobipy import GRB

from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import EmbodiedTask
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.tasks.nav.nav import (
    SPL,
    DistanceToGoal,
    DistanceToGoalReward,
    EpisodicCompassSensor,
    EpisodicGPSSensor,
    GOATDistanceToSubGoal,
    GOATSubTaskSPL,
    GOATSubTaskStopAction,
    GOATSubTaskSuccess,
    NavigationEpisode,
    SoftSPL,
    StopAction,
    Success,
)
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat_dm3.core.embodied_task import MultiAgentEmbodiedTask

if TYPE_CHECKING:
    from omegaconf import DictConfig


@registry.register_sensor(name="DM3CompassSensor")
class DM3EpisodicCompassSensor(EpisodicCompassSensor):

    def get_agent_start_pos(self, episode, task, agent_id):
        agent_config = get_agent_config(self._sim.habitat_config, agent_id=agent_id)
        return agent_config.start_position, quaternion_from_coeff(
            agent_config.start_rotation
        )

    def get_agent_current_pose(self, sim, agent_id):
        agent_state = self._sim.get_agent_state(agent_id)
        return agent_state.position, agent_state.rotation

    def get_observation(self, observations, episode, task, *args, **kwargs):
        observations = []
        num_agents = len(self._sim.agents)
        for agent_id in range(num_agents):

            _, rotation_world_start = self.get_agent_start_pos(episode, task, agent_id)
            _, rotation_world_agent = self.get_agent_current_pose(self._sim, agent_id)
            # print("Agent", agent_id, "Rotation", rotation_world_agent, "Rotation world start", rotation_world_start)

            if isinstance(rotation_world_agent, quaternion.quaternion):
                observations.append(self._quat_to_xy_heading(
                    rotation_world_agent.inverse() * rotation_world_start
                ))
            else:
                raise ValueError("Agent's rotation was not a quaternion")
        return observations


@registry.register_sensor(name="DM3GPSSensor")
class DM3EpisodicGPSSensor(EpisodicGPSSensor):

    def get_agent_start_pos(self, episode, task, agent_id):
        agent_config = get_agent_config(self._sim.habitat_config, agent_id=agent_id)
        return agent_config.start_position, quaternion_from_coeff(
            agent_config.start_rotation
        )

    def get_agent_current_pose(self, sim, agent_id):
        agent_state = self._sim.get_agent_state(agent_id)
        return agent_state.position, agent_state.rotation

    def get_observation(
        self, observations, episode, task, *args: Any, **kwargs: Any
    ):

        observations = []
        num_agents = len(self._sim.agents)
        for agent_id in range(num_agents):
            start_position, rotation_world_start = self.get_agent_start_pos(
                episode, task, agent_id
            )
            origin = np.array(start_position, dtype=np.float32)

            agent_position, _ = self.get_agent_current_pose(self._sim, agent_id)

            agent_position = quaternion_rotate_vector(
                rotation_world_start.inverse(), agent_position - origin
            )
            if self._dimensionality == 2:
                observations.append(np.array(
                    [-agent_position[2], agent_position[0]], dtype=np.float32
                ))
            else:
                observations.append(agent_position.astype(np.float32))
        return observations


@registry.register_measure
class DM3Success(Success):

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()

        if isinstance(distance_to_target, dict):
            distance_to_target = min(distance_to_target.values())

        if (
            hasattr(task, "is_stop_called")
            and task.is_stop_called  # type: ignore
            and distance_to_target < self._success_distance
        ):
            self._metric = 1.0
        else:
            self._metric = 0.0


@registry.register_measure
class DM3GOATSubTaskSuccess(GOATSubTaskSuccess):

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self._metric = {}
        task.measurements.check_measure_dependencies(
            self.uuid, [GOATDistanceToSubGoal.cls_uuid]
        )
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        agents_dist = task.measurements.measures[
            GOATDistanceToSubGoal.cls_uuid
        ].get_metric()

        distance_to_target = []
        for goal_idx, agent_id in task.stops_called.items():
            if not self._metric.get(goal_idx) is None:
                continue
            distance_to_target = agents_dist[agent_id] 
            if distance_to_target < self._success_distance:
                self._metric[goal_idx] = 1.0
            else:
                self._metric[goal_idx] = 0.0


@registry.register_measure
class DM3SPL(SPL):

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid, Success.cls_uuid]
        )

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        if isinstance(self._start_end_episode_distance, dict):
            self._start_end_episode_distance = self._start_end_episode_distance[0]
        self.update_metric(  # type:ignore
            episode=episode, task=task, *args, **kwargs
        )


@registry.register_measure
class DM3GOATSubTaskSPL(GOATSubTaskSPL):

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self.reset_subtask = False
        task.measurements.check_measure_dependencies(
            self.uuid,
            [GOATDistanceToSubGoal.cls_uuid, GOATSubTaskSuccess.cls_uuid],
        )

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = min(task.measurements.measures[
            GOATDistanceToSubGoal.cls_uuid
        ].get_metric().values())
        self.update_metric(  # type:ignore
            episode=episode, task=task, *args, **kwargs
        )

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        if self.reset_subtask:
            self.reset_subtask = False
            self._agent_episode_distance = 0.0
            #! myTODO: Strogner metric.
            temp = task.measurements.measures[
                GOATDistanceToSubGoal.cls_uuid
            ].get_metric().values()
            assert len(temp) == 1
            self._start_end_episode_distance = min(temp)

        ep_success = task.measurements.measures[
            GOATSubTaskSuccess.cls_uuid
        ].get_metric()
        ep_success = ep_success.get(task.current_task_idx-1, 0.0)

        current_position = self._sim.get_agent_state().position
        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = ep_success * (
            self._start_end_episode_distance
            / max(
                self._start_end_episode_distance, self._agent_episode_distance
            )
        )

        if task.update_goal:
            self.reset_subtask = True


@registry.register_measure
class MultiAgentGOATSPL(SPL):
    r"""Multiagent GOAT SPL
    """

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        super().__init__(sim, config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "multiagent_goat_spl"
    
    def _get_disance_mat(self, positions):
        n_nodes = len(positions)
        dist = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    dist[i, j] = self._sim.geodesic_distance(
                        positions[i],
                        [positions[j]]
                    )
                else:
                    dist[i, j] = 0
        return dist           
    
    def _remove_unreachable_positions(self, start_position, goal_positions):
        reachable_goal_positions = []
        for goal_instance_positions in goal_positions:
            valid_positions = []
            for position in goal_instance_positions:
                dist = self._sim.geodesic_distance(
                    start_position,
                    [position]
                )
                if not np.isinf(dist):
                    valid_positions.append(position)
            if valid_positions:
                reachable_goal_positions.append(valid_positions)

        return [[start_position]] + reachable_goal_positions 
        
    def _get_optimal_distance(self, episode):
        start_position = self._sim.get_agent_state().position
        goal_positions = [[g["position"] for g in goals] for goals in episode.goals]

        cluster_positions = self._remove_unreachable_positions(start_position, goal_positions)

        node_positions = []
        node_to_cluster = []
        cluster_to_nodes = []
        for c_idx, cluster in enumerate(cluster_positions):
            node_indices = list(range(len(node_positions), len(node_positions) + len(cluster)))
            cluster_to_nodes.append(node_indices)
            node_positions.extend(cluster)
            node_to_cluster.extend([c_idx] * len(cluster))

        node_positions = np.array(node_positions)
        
        dist = self._get_disance_mat(node_positions)

        routes, route_distances, max_dist = solve_GTSP(cluster_to_nodes, dist, self.num_agents)

        # # TEMP CODE
        # print(f"\nStatus: Optimal")
        # print(f"Maximum Agent Distance: {max_dist:.3f}\n")
        
        # total_dist = 0
        # for k in range(self.num_agents):
        #     route = routes[k]
        #     visited_clusters = [node_to_cluster[node] for node in route]
        #     print(f"Agent {k}: {' -> '.join(map(str, route))}")
        #     print(f"  Clusters: {' -> '.join(map(str, visited_clusters))}")
        #     print(f"  Distance: {route_distances[k]:.3f}")
        #     total_dist += route_distances[k]
        
        # print(f"\nTotal Distance (all agents): {total_dist:.3f}")
        # print(f"Max Agent Distance: {max_dist:.3f}")
        
        # self.visualize_gtsp_solution(all_positions, node_to_cluster, routes, route_distances, max_dist)
        return max_dist

    def visualize_gtsp_solution(self, all_positions, node_to_cluster, routes, route_distances, max_dist, filename="mgtsp_solution.png"):
        all_positions = np.vstack((all_positions[:, 0], all_positions[:, 2])).T
        depot_idx = 0
        n_agents = len(routes)
        n_clusters = max(node_to_cluster) + 1
        
        plt.figure(figsize=(12, 9))
        colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow"]
        
        cluster_colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
        for c in range(n_clusters):
            cluster_positions = all_positions[[i for i, cl in enumerate(node_to_cluster) if cl == c]]
            plt.scatter(
                cluster_positions[:, 0],
                cluster_positions[:, 1],
                c=[cluster_colors[c]],
                s=200,
                alpha=0.3,
                edgecolors='black',
                linewidths=0.5,
                label=f"Cluster {c}" if c < 10 else None
            )
        
        for k in range(n_agents):
            route = routes[k]
            route_positions = all_positions[route]
            plt.plot(
                route_positions[:, 0],
                route_positions[:, 1],
                "-",
                color=colors[k % len(colors)],
                linewidth=0.5,
                markersize=10,
                label=f"Agent {k} (d={route_distances[k]:.2f})",
                alpha=0.8,
                zorder=10
            )
            
            for i in range(len(route) - 1):
                plt.annotate(
                    "",
                    xy=all_positions[route[i + 1]],
                    xytext=all_positions[route[i]],
                    arrowprops=dict(arrowstyle="->", color=colors[k % len(colors)], lw=2),
                    zorder=9
                )
        
        # plt.scatter(
        #     all_positions[depot_idx, 0],
        #     all_positions[depot_idx, 1],
        #     c="gold",
        #     s=400,
        #     marker="*",
        #     edgecolors="black",
        #     linewidths=2,
        #     zorder=11,
        #     label="Depot",
        # )
        
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Multi-Agent GTSP Solution (Min-Max)\nMax Agent Distance: {max_dist:.3f}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()


    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [MultiAgentGOATSuccess.cls_uuid],
        )
        self.num_agents = len(self._sim.agents)
        self._previous_positions = {}
        self._agents_episode_distance = {}
        for agent_id in range(self.num_agents):
            self._previous_positions[agent_id] = self._sim.get_agent_state(agent_id).position
            self._agents_episode_distance[agent_id] = 0.0

        self._optimal_distance = -1

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        if self._optimal_distance == -1:
            self._optimal_distance = self._get_optimal_distance(episode)
        ep_success = task.measurements.measures[
            MultiAgentGOATSuccess.cls_uuid
        ].get_metric().values()
        if len(ep_success) > 0:
            ep_success = np.mean(list(ep_success)).item()
        else:
            ep_success = 0.0
        

        for agent_id in range(self.num_agents):
            current_position = self._sim.get_agent_state(agent_id).position
            self._agents_episode_distance[agent_id] += self._euclidean_distance(
                current_position, self._previous_positions[agent_id]
            )
            self._previous_positions[agent_id] = current_position

        if self._optimal_distance is None:
            self._metric = 0.0
        else:
            self._metric = ep_success * (
                self._optimal_distance
                / max(
                    self._optimal_distance, max(self._agents_episode_distance.values())
                )
            )


@registry.register_measure
class DM3SoftSPL(SoftSPL):

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        if isinstance(self._start_end_episode_distance, dict):
            self._start_end_episode_distance = self._start_end_episode_distance[0]
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(self, episode, task, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        if isinstance(distance_to_target, dict):
            distance_to_target = min(distance_to_target.values())

        ep_soft_success = max(
            0, (1 - distance_to_target / self._start_end_episode_distance)
        )

        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = ep_soft_success * (
            self._start_end_episode_distance
            / max(
                self._start_end_episode_distance, self._agent_episode_distance
            )
        )


@registry.register_measure
class DM3DistanceToGoal(DistanceToGoal):

    def get_base_position(self, agent_id):
        return self._sim.get_agent_state(agent_id).position

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._previous_positions = {}
        self._metrics = {}
        if self._distance_to == "VIEW_POINTS":
            self._episode_view_points = [
                view_point.agent_state.position
                for goal in getattr(episode, self._goals_attr)
                for view_point in goal.view_points
            ]
        self.update_metric(episode=episode, *args, **kwargs)  # type: ignore

    def update_metric(
        self, episode: NavigationEpisode, *args: Any, **kwargs: Any
    ):
        for agent_id in range(len(self._sim.agents)):
            if self._distance_from == "END_EFFECTOR":
                current_position = self.get_end_effector_position()
            else:
                current_position = self.get_base_position(agent_id)

            if not agent_id in self._previous_positions or not np.allclose(
                self._previous_positions[agent_id], current_position, atol=1e-4
            ) or kwargs.get("recalculate"): #! myTODO: A better mechanism here
                if self._distance_to == "EUCLIDEAN_POINT":
                    distance_to_target = min(
                        [
                            np.linalg.norm(
                                np.array(goal.position) - current_position,
                                ord=2,
                                axis=-1,
                            )
                            for goal in getattr(episode, self._goals_attr)
                        ]
                    )
                elif self._distance_to == "POINT":
                    distance_to_target = self._sim.geodesic_distance(
                        current_position,
                        [
                            goal.position
                            for goal in getattr(episode, self._goals_attr)
                        ],
                        episode,
                    )
                elif self._distance_to == "VIEW_POINTS":
                    #! myTODO: This slows things. Better design.
                    episode._shortest_path_cache = None
                    distance_to_target = self._sim.geodesic_distance(
                        current_position, self._episode_view_points, episode
                    )
                else:
                    logger.error(
                        f"Non valid distance_to parameter was provided: {self._distance_to }"
                    )
                self._previous_positions[agent_id] = (
                    current_position[0],
                    current_position[1],
                    current_position[2],
                )

                self._metrics[agent_id] = distance_to_target

        self._metric = self._metrics


@registry.register_measure
class DM3GOATDistanceToSubGoal(DM3DistanceToGoal):
    """The measure calculates a distance towards a sub-task goal."""

    cls_uuid: str = "goat_distance_to_sub-goal"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        super().__init__(sim, config, **kwargs)

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self.subtask_idx = -1
        self.reset_subtask(episode)
        self.update_metric(episode=episode, *args, **kwargs)  # type: ignore
    
    def reset_subtask(self, episode):
        self.subtask_idx += 1
        self._previous_positions = {}
        self._metrics = {}
        self.reset_next_step = False
        episode._shortest_path_cache = None
        self.update_goal_viewpoints(episode, self.subtask_idx)
    
    def update_goal_viewpoints(self, episode, task_idx):
        if self._distance_to == "VIEW_POINTS":
            self._episode_view_points = []
            for goal in getattr(episode, self._goals_attr)[task_idx]:
                self._episode_view_points.extend(
                    [vp['agent_state']['position'] for vp in goal["view_points"]]
                )

    def update_metric(
        self, episode: NavigationEpisode, *args: Any, **kwargs: Any
    ):
        if self.reset_next_step:
            self.reset_subtask(episode)
        super().update_metric(episode, *args, **kwargs)
        if isinstance(self._metric, float):
            self._metric = {0: self._metric}
        
        if kwargs["task"].update_goal:
            self.reset_next_step = True


@registry.register_measure
class DM3DistanceToGoalReward(DistanceToGoalReward):

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [self.distance_to_goal_cls.cls_uuid]
        )
        self._previous_distance = task.measurements.measures[
            self.distance_to_goal_cls.cls_uuid
        ].get_metric()
        if isinstance(self._previous_distance, dict):
            self._previous_distance = min(self._previous_distance.values())
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        distance_to_target = task.measurements.measures[
            self.distance_to_goal_cls.cls_uuid
        ].get_metric()
        if isinstance(distance_to_target, dict):
            distance_to_target = min(distance_to_target.values())
        self._metric = -(distance_to_target - self._previous_distance)
        self._previous_distance = distance_to_target


@registry.register_task_action
class DM3GOATSubTaskStopAction(GOATSubTaskStopAction):

    def step(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        task.stops_called[kwargs.get("task_idx")] = kwargs.get("agent_id")
        if not task.update_goal and not task.is_stop_called:
            task.current_task_idx += 1
            if task.current_task_idx != task.num_tasks:
                task.is_stop_called = False
                task.update_goal = True
            else:
                task.update_goal = False
                task.is_stop_called = True  # type: ignore
        
        return self._sim.get_observations_at(agent_id=kwargs.get("agent_id"))  # type: ignore


@registry.register_measure
class MultiAgentGOATSuccess(Success):
    r"""Whether or not the GOAT agent succeeded at a sub-task

    This measure depends on DistanceToGoal measure.
    """

    cls_uuid: str = "multiagent_goat_success"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        super().__init__(sim, config)

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self._metric = {}
        task.measurements.check_measure_dependencies(
            self.uuid, [MultiAgentGOATDistanceToSubGoal.cls_uuid]
        )
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        distance_to_subgoal = task.measurements.measures[
            MultiAgentGOATDistanceToSubGoal.cls_uuid
        ].get_metric()

        for goal_idx, agent_id in task.stops_called.items():
            if not self._metric.get(goal_idx) is None:
                continue
            # print("Stops called: ", task.stops_called)
            # TODO: dist to subgoal is wrong
            # print("Dist to subgoal: ", distance_to_subgoal)
            try:
                distance_to_target = distance_to_subgoal[goal_idx][agent_id]
            except KeyError:
                print("tasks stops called: ", task.stops_called, goal_idx, agent_id)
            self._metric[goal_idx] = 1.0 if distance_to_target < self._success_distance else 0.0


@registry.register_measure
class MultiAgentGOATDistanceToSubGoal(DM3DistanceToGoal):
    """The measure calculates a distance towards a sub-task goal."""

    cls_uuid: str = "multiagent_goat_distance_to_sub-goal"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        super().__init__(sim, config, **kwargs)

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._previous_positions = {}
        self._metrics = {}
        episode._shortest_path_cache = None
        self.set_goal_viewpoints(episode)
        self.update_metric(episode=episode, *args, **kwargs)  # type: ignore
    
    def set_goal_viewpoints(self, episode):
        if not self._distance_to == "VIEW_POINTS":
            return

        episode_goals = getattr(episode, self._goals_attr)
        self.goals_viewpoints = []
        for task_idx in range(len(episode_goals)):
            _goal_view_points = []
            for goal in episode_goals[task_idx]:
                _goal_view_points.extend(
                    [vp['agent_state']['position'] for vp in goal["view_points"]]
                )
            self.goals_viewpoints.append(_goal_view_points)

    def update_metric(
        self, episode: NavigationEpisode, *args: Any, **kwargs: Any
    ):
        final_metric = {}
        episode_goals = getattr(episode, self._goals_attr)
        for task_idx in range(len(episode_goals)):
            self._episode_view_points = self.goals_viewpoints[task_idx]
            super().update_metric(episode, *args, **kwargs, recalculate=True)
            final_metric[task_idx] = self._metric.copy()

        self._metric = final_metric


@registry.register_task_action
class MultiAgentGOATStopAction(StopAction):
    """
    Compared to GOATSubTaskStopAction, this action is for non-sequential GOAT task, and considers which agent is calling the action!
    """
    name: str = "multiagent_goat_stop"

    def step(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        if kwargs.get("task_idx") is None:
            task.idle_agents += 1
        else:
            task.stops_called[kwargs.get("task_idx")] = kwargs.get("agent_id")
        
        if len(task.stops_called) == task.num_tasks:
            task.is_stop_called = True
        
        return self._sim.get_observations_at(agent_id=kwargs.get("agent_id"))  # type: ignore


@registry.register_task(name="MultiAgentNav-v0")
class MultiAgentNavigationTask(MultiAgentEmbodiedTask):
    def __init__(
        self,
        config: "DictConfig",
        sim: Simulator,
        dataset: Optional[Dataset] = None,
    ) -> None:
        super().__init__(config=config, sim=sim, dataset=dataset)

    def overwrite_sim_config(self, config: Any, episode: Episode) -> Any:
        with read_write(config):
            config.simulator.scene = episode.scene_id
            if (
                episode.start_position is not None
                and episode.start_rotation is not None
            ):        
                num_agents = len(config.simulator.agents)
                # base = np.quaternion(*episode.start_rotation)
                for i in range(num_agents):
                    agent_config = get_agent_config(config.simulator, agent_id=i)
                    agent_config.start_position = episode.start_position

                    agent_config.start_rotation = [
                        float(k) for k in episode.start_rotation
                    ]
                    # angle = 2 * np.pi * i / num_agents
                    # yaw_rot = quaternion.from_euler_angles([0, angle, 0])
                    # new_quat = yaw_rot * base
                    # agent_config.start_rotation = [new_quat.x, new_quat.y, new_quat.z, new_quat.w]
                    agent_config.is_set_start_state = True
        return config

    def _check_episode_is_active(self, *args: Any, **kwargs: Any) -> bool:
        return not getattr(self, "is_stop_called", False)


def solve_GTSP(cluster_nodes, dist, num_agents):
    n_nodes = dist.shape[0]
    n_clusters = len(cluster_nodes)
    depot_idx = cluster_nodes[0][0]

    model = gp.Model("MultiAgent_GTSP")
    model.setParam("OutputFlag", 0)
    
    x = {}
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                for k in range(num_agents):
                    x[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")
    
    u = {}
    for i in range(n_nodes):
        if i != depot_idx:
            for k in range(num_agents):
                u[i, k] = model.addVar(
                    lb=1, ub=n_clusters - 1, vtype=GRB.CONTINUOUS, name=f"u_{i}_{k}"
                )
    
    max_dist = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="max_dist")
    for k in range(num_agents):
        model.addConstr(
            max_dist
            >= gp.quicksum(
                dist[i][j] * x[i, j, k]
                for i in range(n_nodes)
                for j in range(n_nodes)
                if i != j
            ),
            name=f"max_dist_{k}",
        )
    
    model.setObjective(max_dist, GRB.MINIMIZE)
    
    for c in range(n_clusters):
        if c != 0:
            model.addConstr(
                gp.quicksum(
                    x[i, j, k]
                    for j in cluster_nodes[c]
                    for i in range(n_nodes)
                    for k in range(num_agents)
                    if i != j
                )
                == 1,
                name=f"visit_cluster_{c}",
            )
    
    for k in range(num_agents):
        model.addConstr(
            gp.quicksum(x[depot_idx, j, k] for j in range(n_nodes) if j != depot_idx) == 1,
            name=f"leave_depot_{k}",
        )
    
    for k in range(num_agents):
        for j in range(n_nodes):
            if j != depot_idx:
                in_flow = gp.quicksum(x[i, j, k] for i in range(n_nodes) if i != j)
                out_flow = gp.quicksum(x[j, i, k] for i in range(n_nodes) if i != j)
                model.addConstr(in_flow - out_flow >= 0, name=f"flow_lower_{j}_{k}")
                model.addConstr(in_flow - out_flow <= 1, name=f"flow_upper_{j}_{k}")
    
    for k in range(num_agents):
        model.addConstr(
            gp.quicksum(
                gp.quicksum(x[i, j, k] for i in range(n_nodes) if i != j)
                - gp.quicksum(x[j, i, k] for i in range(n_nodes) if i != j)
                for j in range(n_nodes) if j != depot_idx
            ) == 1,
            name=f"one_end_{k}"
        )
    
    for k in range(num_agents):
        for i in range(n_nodes):
            if i != depot_idx:
                for j in range(n_nodes):
                    if j != depot_idx and i != j:
                        model.addConstr(
                            u[i, k] - u[j, k] + n_clusters * x[i, j, k] <= n_clusters - 1,
                            name=f"mtz_{i}_{j}_{k}",
                        )
    
    model.optimize()
    
    if model.status != GRB.OPTIMAL:
        return None, None, None
    
    routes = {k: [] for k in range(num_agents)}
    for k in range(num_agents):
        current = depot_idx
        route = [depot_idx]
        visited = {depot_idx}
        
        while True:
            next_node = None
            for j in range(n_nodes):
                if j != current and (current, j, k) in x:
                    if x[current, j, k].X > 0.5:
                        next_node = j
                        break
            
            if next_node is None:
                break
            
            route.append(next_node)
            visited.add(next_node)
            current = next_node
            
            if len(route) > n_nodes:
                break
        
        routes[k] = route
    
    route_distances = {}
    for k in range(num_agents):
        route = routes[k]
        route_distances[k] = sum(dist[route[i]][route[i + 1]] for i in range(len(route) - 1))
    
    return routes, route_distances, model.objVal
