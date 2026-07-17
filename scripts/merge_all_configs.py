import yaml
from omegaconf import OmegaConf
from habitat.config.default import get_config

# habitat_config_path = "benchmark/nav/goat/goat_hm3d_rgbd_with_semantic.yaml"
habitat_config_path = "benchmark/nav/objectnav/objectnav_hm3d_rgbd_with_semantic.yaml" # V2
cfg = get_config(habitat_config_path)  
with open("./merged_config.yaml", "w") as f:
    f.write(yaml.dump(OmegaConf.to_container(cfg), sort_keys=False))