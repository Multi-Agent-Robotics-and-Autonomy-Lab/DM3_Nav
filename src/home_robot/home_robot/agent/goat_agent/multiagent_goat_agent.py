# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import time
import queue
import logging
import numpy as np
from .goat_agent import GoatAgent, Task
from .map_merger import MapMerger
from typing import Any, Dict, List, Tuple
from home_robot.core.interfaces import DiscreteNavigationAction


class CommunicationLogger(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return f"[COMM] {msg}", kwargs


class BaseMultiAgentGoatAgent(GoatAgent):
    def __init__(self, config, vocabulary, camera_params=None, agent_id=None, device_id: int = 0):
        super().__init__(config, vocabulary, camera_params, agent_id, device_id)
        self.inst_goals = None
        self.tasks_done = None
        self.tasks_failed = None
        self.active_task = None
        self.others_active_task_expiration = None
        self.communication_cooldown = config.AGENT.COMMUNICATION.cooldown
        self.location_valid = config.AGENT.COMMUNICATION.location_valid  # for how many steps received locations are valid
        self.active_task_cooldown = 20
        self.map_merger = MapMerger(
            num_sem_categories=self.num_sem_categories,
            resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            ransac_thresh=10.0,
            iou_threshold=0.2,
            log=self._log,
        )
        self.neighbors = {}
        self._recv_queue = queue.Queue()
        self.comm_log = CommunicationLogger(self._log, {})
        self._full_map_sent_to = set()

    def act(self, **kwargs):
        #  no lock needed because only the main thread calls act() and modifies state here
        self._drain_recv_queue()
        neighbor_locs = self._get_neighbor_locs()
        for key in list(self.others_active_task_expiration.keys()):
            if (
                self._get_communication_time_unit()
                > self.others_active_task_expiration[key]
            ):
                del self.others_active_task_expiration[key]
        return super().act(neighbor_locs=neighbor_locs)

    def _get_neighbor_locs(self):
        neighbor_locs = {}
        for agent_id, data in self.neighbors.items():
            if self._get_communication_time_unit() - data["time"] < self.location_valid:
                neighbor_locs[agent_id] = data["loc"]
        return neighbor_locs

    def _drain_recv_queue(self):
        all_messages = []
        while not self._recv_queue.empty():
            try:
                all_messages.append(self._recv_queue.get_nowait())
            except queue.Empty:
                break

        if not all_messages:
            return

        per_agent = {}
        for msg in all_messages:
            aid = msg["agent_id"]
            if aid not in per_agent:
                per_agent[aid] = {"latest": msg, "latest_with_map": None}
            per_agent[aid]["latest"] = msg
            if msg.get("map") is not None:
                per_agent[aid]["latest_with_map"] = msg

        for aid, entries in per_agent.items():
            latest = entries["latest"]
            latest_map = entries["latest_with_map"]

            if latest_map is not None:
                self.comm_log.info(
                    f"Agent {self.agent_id} merging map from Agent {aid} "
                    f"at step {self.total_timesteps}"
                )
                self._merge_communication_data(latest_map)

                if latest is not latest_map:
                    self.comm_log.info(
                        f"Agent {self.agent_id} merging latest location/tasks from Agent {aid} "
                        f"at step {self.total_timesteps}"
                    )
                    self._merge_communication_data(latest)
            else:
                self.comm_log.info(
                    f"Agent {self.agent_id} merging data from Agent {aid} "
                    f"at step {self.total_timesteps} (no map)"
                )
                self._merge_communication_data(latest)

    def _preprocess_tasks(self, tasks_obs) -> List[Task]:
        tasks = super()._preprocess_tasks(tasks_obs)
        if self.inst_goals is None:
            self.inst_goals = [None] * len(tasks)
            self.tasks_done = [False] * len(tasks)
            self.tasks_failed = [False] * len(tasks)
            self.others_active_task_expiration = {}
        return tasks

    @torch.no_grad()
    def _search_for_goal(self):
        for i, task in enumerate(self.tasks):
            if self.tasks_done[i] or self.tasks_failed[i]:
                continue

            if self.inst_goals[i] is None or self.total_timesteps % self.search_found_goal_freq == 0:
                inst_goal_id, best_score = self.matching.search_for_goal(
                    task,
                    self.match_memory,
                    self.semantic_map.global_pose,
                    score_thresh=0 if self.navigate_to_best else None,
                )
                if inst_goal_id is not None:
                    if self.inst_goals[i] is None or self.inst_goals[i]["score"] < best_score or True:
                        self.log.info(
                            f"Found (better) instance goal for task {i} with score {best_score}"
                        )
                        self.inst_goals[i] = {
                            "id": inst_goal_id,
                            "score": best_score
                        }
            else:
                self.log.debug(
                    f"Instance {self.inst_goals[i]} already found for goal {i}"
                )

        self.match_memory = False

    def _get_best_action(self, **kwargs):
        if all(
            [
                self.tasks_done[i] or self.tasks_failed[i]
                for i in range(len(self.tasks_done))
            ]
        ):
            self.log.info("All tasks done or failed, stopping")
            return [None, DiscreteNavigationAction.STOP], {}

        if self.active_task is None:
            for i in range(len(self.tasks)):
                if (
                    self.tasks_done[i]
                    or self.tasks_failed[i]
                    or self.inst_goals[i] is None
                    or i in self.others_active_task_expiration
                ):
                    continue
                self.active_task = i
                break

        neighbor_locs = kwargs.get("neighbor_locs", [])
        if self.active_task is not None:
            action, vis_input = self.planner.plan(
                self.inst_goals[self.active_task]["id"],
                self.tasks[self.active_task].goal_semantic_id,
                neighbor_locs=neighbor_locs,
                fallback_to_frontier=False,
            )
            if action is None:
                self.tasks_failed[self.active_task] = True
                return [self.active_task, DiscreteNavigationAction.STOP], vis_input
            else:
                return [self.active_task, action], vis_input

        # If the code reaches here, active task is None.
        action, vis_input = self.planner.plan(
            neighbor_locs=neighbor_locs,
        )
        if action is not None:
            return [None, action], vis_input

        if self.navigate_to_best:
            self.log.warning(
                "No reachable goal/frontier, and tried all possible goals. Stopping"
            )
            return (
                None,
                DiscreteNavigationAction.STOP,
            ), {}  # Stop with None, means idle
        self.log.warning("Reducing threshold to 0 for all goals from now on.")
        self.navigate_to_best = True
        self.match_memory = True
        self._search_for_goal()

        return self._get_best_action(neighbor_locs=neighbor_locs)

    def _process_action(self, action):
        if self.too_close == True:
            if action[1] == DiscreteNavigationAction.MOVE_FORWARD:
                print("Too close, not executing forward")
                action[1] = None
                self.planner.last_action = None
        return {
            "action": action[1],
            "action_args": {
                "agent_id": 0 if self.agent_id is None else self.agent_id,
                "task_idx": action[0],
            },
        }

    def reset(self, scene_id, episode_id):
        super().reset(scene_id, episode_id)
        self.inst_goals = None
        self.tasks_done = None
        self.tasks_failed = None
        self.active_task = None
        self.others_active_task_expiration = None
        self.map_shared_time = {}
        while not self._recv_queue.empty():
            try:
                self._recv_queue.get_nowait()
            except queue.Empty:
                break
        self._full_map_sent_to = set()

    def reset_vis_dir(self, scene_id, episode_id, current_task_idx=None):
        super().reset_vis_dir(scene_id, episode_id, None)
        self.map_merger.vis_dir = self.semantic_map.vis_dir

    def _get_task_info(self, obs, action, info):
        if action["action_args"]["task_idx"] is not None:
            current_task = obs.task_observations["tasks"][
                action["action_args"]["task_idx"]
            ]
            info["task_type"] = current_task["type"]
            goal_text_desc = {x: y for x, y in current_task.items() if x != "image"}
            info["caption"] = str(goal_text_desc)
            if current_task["type"] == "imagenav":
                info["goal_image"] = current_task["image"]
            else:
                info["third_person_image"] = obs.third_person_image
        else:
            remaining_goals = []
            for i, task in enumerate(obs.task_observations["tasks"]):
                if not self.tasks_done[i]:
                    remaining_goals.append(task["category"])
            info["caption"] = f"Remaining Goals: [{', '.join(remaining_goals)}]"

        if isinstance(action["action"], DiscreteNavigationAction):
            info["caption"] += f" | Action: {str(action['action']).split('.')[-1]}"
        else:
            info["caption"] += f" | Action: {str(action['action'])}"

    def handle_stop(self, action):
        self.reset_for_next_task()
        self.active_task = None
        task_idx = action["action_args"]["task_idx"]
        if task_idx is not None:
            self.tasks_done[task_idx] = True
        else:
            self.log.debug("IDLE. Waiting for other agents to complete their tasks.")

    def _get_communication_data(self):
        return {
            "map": self.semantic_map.global_map,
            # "instances": self.instance_memory.instances,
            "agent_id": self.agent_id,
            "tasks_done": self.tasks_done,
            "active_task": self.active_task,
            "active_task_score": float(self.inst_goals[self.active_task]["score"]) if self.active_task is not None else None,
            "location": [
                int(self.semantic_map.global_loc[0]),
                int(self.semantic_map.global_loc[1]),
            ],
            "time": self._get_communication_time_unit()
        }

    def _merge_communication_data(self, data):
        neighbor_id = data["agent_id"]

        if data.get("map") is not None:
            self._merge_map(data)
        else:
            data["transformed_loc"] = self.map_merger.transform_location(
                neighbor_id, data["location"]
            )

        if data.get("transformed_loc") is not None:
            self.neighbors[neighbor_id] = {
                "time": data["time"],
                "loc": data["transformed_loc"],
            }
        self._merge_task_info(data)

    def _merge_map(self, data):
        """Align and merge neighbor's map into ours."""
        neighbor_id = data["agent_id"]

        transfomed_map = self.map_merger.get_transformed_map(
            global_map=self.semantic_map.global_map,
            neighbor_global_map=data["map"],
            neighbor_id=data["agent_id"],
        )
        # import os
        # torch.save(self.semantic_map.global_map, os.path.join(self.planner.vis_dir, f'mymap_{self.total_timesteps}.pt'))
        # torch.save(data["map"].to(self.semantic_map.device), os.path.join(self.planner.vis_dir, f'othermap_{self.total_timesteps}.pt'))


        # Get the transformed map (may come pre-computed from simulation)
        if transfomed_map is None:
            self._vis_merge_failed(data)
            return

        data["transformed_map"] = transfomed_map
        # Apply merge
        merged = self.map_merger._merge(
            # self.semantic_map.global_map, data["transformed_map"]
            self.semantic_map.global_map,
            data,
        )

        transformed_loc = self.map_merger.transform_location(
            data["agent_id"], data["location"]
        )
        data["transformed_loc"] = transformed_loc
        data["transformed_map"] = transfomed_map

        self.comm_log.info(
            f"Map aligned with Agent {neighbor_id}, neighor_loc: {transformed_loc}"
        )


        if self.visualization_level > 0:
            self.map_merger._visualize(
                self.semantic_map.global_map,
                self.semantic_map.global_loc,
                merged,
                data,
            )
        self.semantic_map.global_map[:] = merged
        lmb = self.semantic_map.lmb
        self.semantic_map.local_map[:] = self.semantic_map.global_map[
            :, lmb[0] : lmb[1], lmb[2] : lmb[3]
        ]

    def _vis_merge_failed(self, data, reason=None):
        if self.visualization_level > 0 and data.get("map") is not None:
            self.map_merger._visualize_failed(
                self.semantic_map.global_map,
                data["map"],
                self.semantic_map.global_loc,
                data["location"],
                reason=reason,
            )

    def _merge_task_info(self, data):
        """Merge task completion and active task deconfliction."""
        self.log.debug(f"Merging task done. Mine: {self.tasks_done}, neighbors: {data['tasks_done']}")
        if self.tasks_done is None:
            self.tasks_done = data["tasks_done"]
        else:
            self.tasks_done = [
                a or b for a, b in zip(self.tasks_done, data["tasks_done"])
            ]
        self.log.debug(f"After: {self.tasks_done}")

        self._handle_neighbor_active_task(data["agent_id"], data["active_task"], data["active_task_score"])

    def _handle_neighbor_active_task(self, agent_id, active_task, active_task_score):
        if active_task is None:
            self.log.debug(f"Neighbor active task is None")
            return

        same_task = active_task == self.active_task
        if not same_task:
            self.others_active_task_expiration[active_task] = (
                self._get_communication_time_unit() + self.active_task_cooldown
            )
            self.log.debug(f"Blocking neighbor's task {active_task} until {self.others_active_task_expiration[active_task]}")
            return

        # Same task
        has_priority = agent_id < self.agent_id
        my_score = self.inst_goals[active_task]["score"]
        self.log.debug(f"Neighbor {agent_id} active task: {active_task}, same_task: {same_task}, has_priority: {has_priority} (my score: {my_score}, neighbor score: {active_task_score})")
        if my_score < active_task_score or (my_score == active_task_score and has_priority):
            self.others_active_task_expiration[active_task] = (
                self._get_communication_time_unit() + self.active_task_cooldown
            )
            self.active_task = None
            self.log.debug(f"Yielding task {active_task} to neighbor {agent_id} (higher priority)")
        

    def _update_steps(self):
        super()._update_steps()
        self.map_merger.timestep = self.get_subtask_timestep()


class SimulationMultiAgentGoatAgent(BaseMultiAgentGoatAgent):
    def __init__(self, config, vocabulary, camera_params=None, agent_id=None, device_id=0):
        super().__init__(config, vocabulary, camera_params, agent_id, device_id)
        for i in range(config.NUM_AGENTS):
            self.map_merger._cached_transforms[i] = {"iou": 1.0, "transform": np.eye(2, 3)}

    def simulate_receive_map(self, other_agents):
        neighbors = []
        for agent in other_agents:
            dist = (
                agent.semantic_map.global_pose[:2] - self.semantic_map.global_pose[:2]
            ).norm()
            if dist < self.communication_radius:
                neighbors.append(agent)
                self.log.debug(
                    f"Communicating with agent: {agent.agent_id} with distance {dist}"
                )
        for neighbor in neighbors:
            data = neighbor._get_communication_data()

            self.comm_log.info(
                f"Agent {self.agent_id} <- Agent {data['agent_id']}: "
                f"received map at step {self.total_timesteps}"
            )
            last = self.map_shared_time.get(neighbor.agent_id, 0)
            if self.total_timesteps - last > self.communication_cooldown or (
                neighbor.agent_id not in self._full_map_sent_to
            ):
                self.map_shared_time[data["agent_id"]] = (
                    self.total_timesteps
                )  # In realworld agent, this happens in sender thread, and real time is used.
                self._full_map_sent_to.add(neighbor.agent_id)
            else:
                data.pop("map")

            self._recv_queue.put(data)

    def _get_communication_time_unit(self):
        return self.total_timesteps
