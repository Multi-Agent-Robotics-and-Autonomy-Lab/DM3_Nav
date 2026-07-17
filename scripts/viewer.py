from habitat_sim.utils.common import quat_from_angle_axis
import habitat_sim

def main():
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = "data/scene_datasets/habitat-test-scenes/apartment_0.glb"

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])

    sim = habitat_sim.Simulator(cfg)

    # Move forward
    sim.step("move_forward")

    # Save a screenshot or display RGB/semantic/depth
    obs = sim.get_sensor_observations()
    # show or save obs['color_sensor'], etc.

    sim.close()

if __name__ == "__main__":
    main()
