import habitat
import habitat_sim
import numpy as np
import matplotlib.pyplot as plt
from habitat.config.default import get_config, read_write

config_path = "/home-robot/src/third_party/habitat-lab/habitat-baselines/habitat_baselines/config/goat/modular_goat_hm3d_fixed.yaml"
scene_id = "Dd4bFSTQ8gi"
episode_id = 3
# global_pose = [20.52, 24.53]
new_location = [ 6.9469576 , -0.07609773, -2.1983232 ]
new_rotation = [0.0318580120801926, 0, -0.999492466449738, 0]
if len(new_rotation) == 4:
    # d, a, b ,c
    new_rotation =  new_rotation[1:] +[new_rotation[0]] 


from habitat_baselines.config.default import _BASELINES_CFG_DIR

cfg = get_config(config_path, configs_dir=_BASELINES_CFG_DIR)

with read_write(cfg):
    cfg.habitat.dataset.content_scenes = [scene_id]

env = habitat.Env(config=cfg)
obs = env.reset()

while not int(env.current_episode.episode_id) == episode_id:
    env.reset()

print(f"Evaluating scene {scene_id} episode {episode_id}")


init_state = env.sim.agents[0].get_state()
print("Initial Agent State:", init_state)
print("Initial gps:", obs["gps"])


# new_location = [
#     init_state.position[0] + global_pose[1] - 24,
#     init_state.position[2] + 24 - global_pose[0],
# ]


agent_state = habitat_sim.AgentState()
# agent_state.position = np.array([new_location[0], init_state[1], new_location[1]])
# agent_state.rotation = init_state.rotation
agent_state.position = new_location
agent_state.rotation = new_rotation
# agent_state.sensor_states.clear()


env.sim.agents[0].set_state(agent_state)
print("Updated Agent State:", env.sim.agents[0].get_state())

# obs = env.sim.get_observations_at(agent_state.position, agent_state.rotation)
obs = env.sim.get_sensor_observations()
rgb = obs["rgb"]
depth = obs["depth"]

# --- Visualization ---
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(rgb)
plt.title("RGB Observation")

plt.subplot(1, 2, 2)
plt.imshow(depth.squeeze(), cmap="plasma")
plt.title("Depth Observation")

plt.show()

env.close()
