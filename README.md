# DM³-Nav: Decentralized Multi-Agent Multimodal Multi-Object Semantic Navigation

**Amin Kashiri, Atharva Jamsandekar, Yasin Yazıcıoğlu** — Northeastern University

Accepted at IROS 2026.

This is the `IROS2026` branch: the code and settings that reproduce the IROS 2026 simulation results.

DM³-Nav is a fully decentralized multi-robot semantic navigation system. A team of robots locates multiple objects specified by category, language description, or reference image, coordinating only through ad-hoc pairwise communication, with no central planner and no shared global map.

## Demo

[▶️ Watch the real-world demo](https://drive.google.com/file/d/1QiUSCn5rIvtuTUqtuXLPgmt6S8x9-MCZ/view?usp=drive_link): two robots finding 8 targets in an office, using onboard sensing and compute only.

## Setup

Requirements: an NVIDIA GPU, Docker and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### 1. Download the HM3D scenes

You can get access to HM3D dataset by following this [link](https://matterport.com/habitat-matterport-3d-research-dataset). Then create an API token on your Matterport account. Only the v0.2 val scenes are needed.

Set these in your shell first:

```bash
HM3D_DATA=$HOME/hm3d_data                    # where the HM3D scenes go
HM3D_TOKEN_ID=<your_api_token_id>            # Matterport API token
HM3D_TOKEN_SECRET=<your_api_token_secret>
BASE_IMAGE=fairembodied/habitat-challenge:homerobot-ovmm-challenge-2023
```

With Docker:

```bash
docker run --rm -it -v $HM3D_DATA:/dl $BASE_IMAGE bash -c \
    "source /opt/conda/etc/profile.d/conda.sh && conda activate home-robot && \
     python -m habitat_sim.utils.datasets_download --username $HM3D_TOKEN_ID --password $HM3D_TOKEN_SECRET \
       --uids hm3d_val_v0.2 --data-path /dl"
```

Or without Docker, in a Python environment with `habitat-sim` ([link](https://github.com/facebookresearch/habitat-sim)) installed:

```bash
python -m habitat_sim.utils.datasets_download --username $HM3D_TOKEN_ID --password $HM3D_TOKEN_SECRET \
    --uids hm3d_val_v0.2 --data-path $HM3D_DATA
```

### 2. Create the container

With the Dockerfile:

```bash
git clone --branch IROS2026 https://github.com/Multi-Agent-Robotics-and-Autonomy-Lab/DM3_nav.git
cd DM3_nav
docker build -f DM3_Nav.DOCKERFILE -t dm3-nav .
docker run -dit --gpus all --name dm3-nav \
    -v $HM3D_DATA/versioned_data/hm3d-0.2:/home-robot/data/versioned_data/hm3d-0.2:ro \
    dm3-nav bash
```

Or with `init_container.sh`, starting from the base image:

```bash
docker run -dit --gpus all --name dm3-nav \
    -v $HM3D_DATA/versioned_data/hm3d-0.2:/home-robot/data/versioned_data/hm3d-0.2:ro \
    $BASE_IMAGE bash
docker exec -it dm3-nav bash -c \
    "wget -q https://raw.githubusercontent.com/Multi-Agent-Robotics-and-Autonomy-Lab/DM3_nav/IROS2026/init_container.sh -O /tmp/init_container.sh && \
     bash /tmp/init_container.sh"
```

Open a shell in the container with:

```bash
docker exec -it dm3-nav bash
```

## Usage

Inside the container:

```bash
source /opt/conda/etc/profile.d/conda.sh && conda activate home-robot
cd /home-robot

# Multi-agent HM3D ObjectNav with the IROS 2026 settings
python projects/habitat_goat/multiagent_eval_episode.py \
    --project_config_path projects/habitat_goat/configs/agent/hm3d_iros_final.yaml --name iros_final

# Multi-agent GOAT
python projects/habitat_goat/multiagent_eval_episode.py \
    --project_config_path projects/habitat_goat/configs/agent/hm3d_eval_new.yaml

# Single-agent
python projects/habitat_goat/eval_episode.py
```

- Results are written to `datadump/results/<name>/per_episode_metrics.json`.
- `--scene START END` limits a run to a range of scenes.
- If you rerun with the same `--name`, episodes that are already recorded are skipped.
- Multi-agent GOAT runs compute MSPL with Gurobi, which needs a license: mount it into the container and set `GRB_LICENSE_FILE` to its path.

## Troubleshooting

### `Unable to create windowless context`

The container can't see the GPU's graphics (EGL) libraries. Generate a CDI spec on the host and recreate the container with `--device nvidia.com/gpu=all` instead of `--gpus all`:

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
docker rm -f dm3-nav
docker run -dit --device nvidia.com/gpu=all -e NVIDIA_DRIVER_CAPABILITIES=all --name dm3-nav \
    -v $HM3D_DATA/versioned_data/hm3d-0.2:/home-robot/data/versioned_data/hm3d-0.2:ro \
    dm3-nav bash
```

## Citation

```bibtex
@article{kashiri2026dm,
  title   = {{DM$^3$-Nav}: Decentralized Multi-Agent Multimodal Multi-Object Semantic Navigation},
  author  = {Kashiri, Amin and Jamsandekar, Atharva and Yaz{\i}c{\i}o{\u{g}}lu, Yasin},
  journal = {arXiv preprint arXiv:2604.22014},
  year    = {2026}
}
```