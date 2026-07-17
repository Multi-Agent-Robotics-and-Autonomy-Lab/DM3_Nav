import time
import socket
import struct
import threading
import queue
import io
import json
from .goat_agent import Task
from typing import List, Optional

import numpy as np
import torch
import zmq
import subprocess
import re


from .multiagent_goat_agent import BaseMultiAgentGoatAgent
class RealWorldGoatAgent(BaseMultiAgentGoatAgent):
    """
    Real-world multi-agent GOAT agent.
    Extends BaseMultiAgentGoat with UDP discovery + ZMQ communication.

    - Sender thread = discover neighbors + pairwise communicate
    - Receiver thread = accept incoming data, ACK immediately, push to queue
    - Main thread = drain queue and merge during act()
    """

    BEACON_MAGIC = b"MAGT"

    def __init__(
        self,
        config,
        vocabulary,
        camera_params=None,
        agent_id=None,
        device_id: int = 0,
        broadcast_port: int = 9900,
        data_port: int = 9901,
        beacon_interval: float = 1.0,
        adhoc_ip=None
    ):
        super().__init__(config, vocabulary, camera_params, agent_id, device_id)

        self.broadcast_port = broadcast_port
        self.data_port = data_port + agent_id
        self.beacon_interval = beacon_interval

        # Queue is thread safe by design
        self._comm_running = False

        assert not adhoc_ip is None
        self.adhoc_ip = adhoc_ip
        self.broadcast_addr = self.adhoc_ip.rsplit('.', 1)[0] + '.255'
        self._neighbor_macs = {}
        # self.rssi_threshold = -80


    def reset(self, scene_id, episode_id):
        self.stop_communication()
        super().reset(scene_id, episode_id)

    def _preprocess_tasks(self, tasks_obs) -> List[Task]:
        tasks = super()._preprocess_tasks(tasks_obs)
        # Communication starts after we have processed initial tasks, so episodes parameters are set.
        self.start_communication()
        return tasks

    def start_communication(self):
        if self._comm_running:
            return
        self._comm_running = True
        self._sender_thread = threading.Thread(
            target=self._sender_loop, name=f"Agent{self.agent_id}-Sender", daemon=True
        )
        self._receiver_thread = threading.Thread(
            target=self._receiver_loop, name=f"Agent{self.agent_id}-Receiver", daemon=True
        )
        self._sender_thread.start()
        self._receiver_thread.start()
        self.comm_log.info(f"Agent {self.agent_id} comm started on port {self.data_port}")

    def stop_communication(self):
        if not self._comm_running:
            return
        self._comm_running = False
        if hasattr(self, '_sender_thread'):
            self._sender_thread.join()
        if hasattr(self, '_receiver_thread'):
            self._receiver_thread.join()



    def _update_maps(self):
        obs_preprocessed, instance_scores, category_scores = self._preprocess_obs(self._curr_obs)
        self.semantic_map_module(
            obs_preprocessed,
            self.pose_delta,
            self.semantic_map,
            instance_scores,
            category_scores
        )

    def _sender_loop(self):
        ctx = zmq.Context()

        tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        tx_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        rx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rx_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        rx_sock.bind(("", self.broadcast_port))
        rx_sock.settimeout(0.2)

        last_beacon = 0
        while self._comm_running:
            try:
                now = time.time()

                if now - last_beacon >= self.beacon_interval:
                    self.comm_log.debug(f"Broadcasting beacon on port {self.broadcast_port}")
                    msg = self.BEACON_MAGIC + struct.pack("IH", self.agent_id, self.data_port)
                    try:
                        tx_sock.sendto(msg, (self.broadcast_addr, self.broadcast_port))
                    except OSError:
                        pass
                    last_beacon = now

                neighbors = {}
                while True:
                    try:
                        data, addr = rx_sock.recvfrom(64)
                        if len(data) < 10 or data[:4] != self.BEACON_MAGIC:
                            time.sleep(0.1)
                            continue
                        agent_id, port = struct.unpack("IH", data[4:10])
                        if agent_id != self.agent_id:
                            self.comm_log.debug(f"Beacond received from {agent_id}")
                            neighbors[agent_id] = {"ip": addr[0], "port": port}
                            if agent_id not in self._neighbor_macs:
                                out = subprocess.check_output(["ip", "neigh", "show", addr[0], "dev", "wlan1"], text=True)
                                m = re.search(r"lladdr (\S+)", out)
                                if m:
                                    self._neighbor_macs[agent_id] = m.group(1)

                    except socket.timeout:
                        break

                for agent_id, info in neighbors.items():
                    send_map = False
                    last = self.map_shared_time.get(agent_id, -1)
                    if (time.time() - last > self.communication_cooldown or
                        (agent_id not in self._full_map_sent_to)) and self.total_timesteps > 12:
                        send_map = True

                    try:
                        packed = self._pack_comm_data(send_map)
                    except Exception as e:
                        self.comm_log.error(f"Pack failed: {e}")
                        continue

                    self.comm_log.debug(f"Sending data to {agent_id}, map: {send_map}")
                    if self._send_to_neighbor(ctx, agent_id, info, packed):
                        self.comm_log.debug(f"Send to {agent_id} successfull")
                        if send_map:
                            self.map_shared_time[agent_id] = time.time()
                            self._full_map_sent_to.add(agent_id)

                time.sleep(0.1)

            except Exception as e:
                self.comm_log.error(f"Sender loop error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)  # don't spin on repeated errors


        tx_sock.close()
        rx_sock.close()
        ctx.term()

    def _send_to_neighbor(
        self, ctx: zmq.Context, aid: int, info: dict, packed: bytes
    ) -> bool:
        try:
            sock = ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.SNDTIMEO, 30000)
            sock.setsockopt(zmq.RCVTIMEO, 30000)
            # sock.setsockopt(zmq.SNDTIMEO, 5000)
            # sock.setsockopt(zmq.RCVTIMEO, 10000)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(f"tcp://{info['ip']}:{info['port']}")
            sock.send(packed)
            self.comm_log.debug("Waiting for ack")
            ack = sock.recv()
            self.comm_log.debug("Received ack")
            sock.close()
            self.comm_log.debug(
                f"Agent {self.agent_id} -> Agent {aid}[{info['ip']}:{info['port']}]: "
                f"sent {len(packed)} bytes at step {self.total_timesteps}"
            )
            return True
        except Exception as e:
            self.comm_log.warning(
                f"Agent {self.agent_id} -> Agent {aid}: send failed: {e}"
            )
            return False

    def _receiver_loop(self):
        self.comm_log.debug(f"Started receiver loop")
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.setsockopt(zmq.RCVTIMEO, 300000)
        # sock.bind(f"tcp://*:{self.data_port}")
        sock.bind(f"tcp://{self.adhoc_ip}:{self.data_port}")

        while self._comm_running:
            try:
                self.comm_log.debug(f"Receiver: Waiting for data t: {time.time()}")
                raw = sock.recv()
                try:
                    data = self._unpack_comm_data(raw)
                    if data is None:
                        self.comm_log.warning("Received invalid data")
                        sock.send(b"ERR")
                        continue
                    # ACK immediately — don't hold the sender waiting for merge
                    sock.send(b"OK")
                    # rssi = self.get_neighbor_rssi(data["agent_id"])
                    # data["rssi"] = rssi
                    # if rssi is not None:
                        # self.comm_log.info(f"Neighbor {data['agent_id']} RSSI: {rssi} dBm")
                        # if rssi < self.rssi_threshold:
                        #     self.comm_log.info(f"Dropping data from {data['agent_id']}, RSSI {rssi} too weak")
                        #     continue

                    self.comm_log.info(
                        f"Agent {self.agent_id} <- Agent {data['agent_id']}: "
                        f"received data at step {self.total_timesteps}, t: {time.time()}, map: {not data.get('map') is None}"
                    )
                    if not data.get("map") is None:
                        self.comm_log.debug(f"data contains map")
                    data["time"] = time.time()
                    self._recv_queue.put(data)
                except Exception as e:
                    self.comm_log.error(f"Receiver process error: {e}")
                    traceback.print_exc()
                    sock.send(b"ERR")
            except zmq.Again:
                continue
            except Exception as e:
                self.comm_log.error(f"Receiver loop error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)

        sock.close()
        ctx.term()

    def _pack_comm_data(self, send_map) -> bytes:
        data = self._get_communication_data()
        global_map = data.pop("map")
        data["has_map"] = send_map

        buf = io.BytesIO()
        buf.write(struct.pack("I", self.agent_id))

        meta_bytes = json.dumps(data).encode()
        buf.write(struct.pack("I", len(meta_bytes)))
        buf.write(meta_bytes)

        if send_map:
            map_np = global_map.cpu().numpy()
            nonzero_mask = np.any(map_np != 0, axis=0)
            rows, cols = np.where(nonzero_mask)
            values = map_np[:, rows, cols].astype(np.float16)  # cut the size in half

            n_nonzero = len(rows)
            buf.write(struct.pack("I", n_nonzero))
            buf.write(rows.astype(np.int32).tobytes())
            buf.write(cols.astype(np.int32).tobytes())
            buf.write(values.tobytes())

            self.comm_log.debug(
                f"Map packed: {n_nonzero}/{nonzero_mask.size} cells "
                f"({n_nonzero / nonzero_mask.size * 100:.1f}%), "
                f"size: {buf.tell() / 1024 / 1024:.1f}MB"
            )

        return buf.getvalue()

    def _unpack_comm_data(self, raw: bytes) -> Optional[dict]:
        try:
            buf = io.BytesIO(raw)
            agent_id = struct.unpack("I", buf.read(4))[0]
            meta_len = struct.unpack("I", buf.read(4))[0]
            meta = json.loads(buf.read(meta_len).decode())

            has_map = meta.pop("has_map", True)
            data = {
                "agent_id": agent_id,
                **meta,
            }

            if has_map:
                C, H, W = self.semantic_map.global_map.shape
                n_nonzero = struct.unpack("I", buf.read(4))[0]

                rows = np.frombuffer(buf.read(n_nonzero * 4), dtype=np.int32)
                cols = np.frombuffer(buf.read(n_nonzero * 4), dtype=np.int32)
                values = np.frombuffer(buf.read(n_nonzero * C * 2), dtype=np.float16).reshape(C, n_nonzero)

                map_np = np.zeros((C, H, W), dtype=np.float32)
                map_np[:, rows, cols] = values.astype(np.float32)
                data["map"] = torch.from_numpy(map_np)

            return data
        except Exception as e:
            self.comm_log.error(f"Unpack failed: {e}")
            return None

    def _get_communication_data(self):
        data = super()._get_communication_data()
        data["time"] = time.time()
        return data

    def _get_communication_time_unit(self):
        return time.time()

    def get_neighbor_rssi(self, agent_id):
        mac = self._neighbor_macs.get(agent_id)
        if not mac:
            return None
        try:
            out = subprocess.check_output(
                ["iw", "wlan1", "station", "get", mac],
                text=True, stderr=subprocess.DEVNULL
            )
            m = re.search(r"signal:\s+(-\d+)", out)
            return int(m.group(1)) if m else None
        except subprocess.CalledProcessError:
            return None
