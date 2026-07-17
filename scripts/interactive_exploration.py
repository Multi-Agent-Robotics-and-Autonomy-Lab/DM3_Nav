import habitat
from habitat.config.default import get_config, read_write
import matplotlib.pyplot as plt

# scene_id = "bCPU9suPUw9"
scene_id = "q5QZSEeHe5g"

episode_id = 25

# habitat_config_path = "benchmark/nav/objectnav/multiagent_objectnav_hm3d_rgbd_with_semantic.yaml" # V2
# habitat_config_path = "benchmark/nav/objectnav/objectnav_hm3d_2022_rgbd_with_semantic.yaml"  # V1
habitat_config_path = "benchmark/nav/objectnav/objectnav_hm3d_rgbd_with_semantic.yaml"  # V2
cfg = get_config(habitat_config_path)
with read_write(cfg):
    cfg.habitat.dataset.content_scenes = [scene_id]
    cfg.habitat.dataset.split = "val"

env = habitat.Env(config=cfg)
obs = env.reset()

while not int(env.current_episode.episode_id) == episode_id:
    obs = env.reset()

print(f"Evaluating scene {scene_id} episode {episode_id}")
# Uncomment to go to specific start position/rotation
# START_POS = [3.21677, -2.66978, 1.04331]
# START_ROT = [0, 0, 0, 1]  # identity quaternion
# episode = env.current_episode

# episode.start_position = START_POS
# episode.start_rotation = START_ROT

# if not env.sim.is_navigable(episode.start_position):
#     episode.start_position = env.sim.pathfinder.snap_point(
#         episode.start_position
#     )

# env.sim.set_agent_state(
#     position=episode.start_position,
#     rotation=episode.start_rotation,
#     reset_sensors=True,
# )

env._elapsed_steps = 0


action_mapping = {
    "w": "move_forward",
    "a": "turn_left",
    "d": "turn_right",
    "q": "stop"
}
action_to_int = {
    "stop" : 0,
    "move_forward" : 1,
    "turn_left" : 2,
    "turn_right" : 3,
    "look_up" : 4,
    "look_down" : 5}
plt.ion()

fig = plt.figure(figsize=(10, 12))

# try:
#     mng = plt.get_current_fig_manager()
#     mng.window.wm_geometry("+0+0")          # Move to top-left corner
#     mng.window.wm_attributes("-topmost", 1) 
#     mng.window.geometry("1850x2160")        # Resize window (left half of 1920x1080 screen)
# except Exception as e:
#     print("Could not reposition window:", e)

# def show_obs(obs):
#     # if multiagent
#     obs = obs[0] 
#     rgb = obs["rgb"]
#     depth = obs["depth"]

#     plt.clf()

#     # RGB
#     plt.subplot(2, 1, 1)
#     plt.imshow(rgb)
#     plt.title("RGB Observation")
#     plt.axis("off")

#     # Depth
#     plt.subplot(2, 1, 2)
#     plt.imshow(depth.squeeze(), cmap="plasma")
#     plt.title("Depth Observation")
#     plt.axis("off")

#     plt.tight_layout()
#     plt.pause(0.001)

OUTPUT_IMAGE = "obs.png"

def show_obs(obs):
    # # Multi-agent handling
    print("Obs keys: ", obs.keys())
    if 0 in obs.keys():
        obs = obs[0]

    rgb = obs["rgb"]
    depth = obs["depth"]

    fig = plt.figure(figsize=(10, 12))

    # RGB
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.imshow(rgb)
    ax1.set_title("RGB Observation")
    ax1.axis("off")

    # Depth
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.imshow(depth.squeeze(), cmap="plasma")
    ax2.set_title("Depth Observation")
    ax2.axis("off")

    fig.tight_layout()
    fig.savefig(OUTPUT_IMAGE, dpi=150)
    plt.close(fig)

show_obs(obs)

while True:
    key = input("Enter action [w/a/d/q]: ").lower().strip()
    if len(key) > 1:
        key = key[0]
    if key not in action_mapping:
        print("Invalid key. Use w/a/d/q.")
        continue
    action = action_mapping[key]
    if action == "stop":
        print("Stopping.")
        break

    action = action_to_int[action]
    action = {
        "action": action,
        "action_args": {
            "agent_id": 0,
            "task_idx": 0,
        },
    }
    obs = env.step(action)
    state = env.sim.agents[0].get_state()
    print(f"Agent state is: {state.position}")
    show_obs(obs)

env.close()
plt.ioff()  
plt.close()

