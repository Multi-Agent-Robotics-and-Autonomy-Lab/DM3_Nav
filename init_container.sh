#!/bin/bash
# DM3-Nav — manual container setup (alternative to DM3_Nav.Dockerfile).
# Run inside a container started from the base image:
#   docker run -it --gpus all -v /path/to/hm3d-0.2:/home-robot/data/versioned_data/hm3d-0.2:ro \
#       fairembodied/habitat-challenge:homerobot-ovmm-challenge-2023 bash

REPO_URL=https://github.com/Multi-Agent-Robotics-and-Autonomy-Lab/DM3_nav.git
BRANCH=IROS2026

# HM3D credentials — needed only if you download HM3D instead of mounting it.
# Get your own: https://matterport.com/habitat-matterport-3d-research-dataset
HM3D_USERNAME="<your_hm3d_username>"
HM3D_PASSWORD="<your_hm3d_password>"

curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install -y git-lfs

# Replace the base image's /home-robot with the DM3-Nav repo.
# Clear it first (keeping data/, which holds the HM3D mount)
git clone --recurse-submodules --branch ${BRANCH} ${REPO_URL} /tmp/dm3nav || exit 1
find /home-robot -mindepth 1 -maxdepth 1 ! -name data -exec rm -rf {} +
cp -a /tmp/dm3nav/. /home-robot/
rm -rf /tmp/dm3nav
cd /home-robot

source /opt/conda/etc/profile.d/conda.sh
conda activate home-robot
git submodule update --init --recursive \
    src/home_robot/home_robot/agent/imagenav_agent/SuperGluePretrainedNetwork \
    yolov10
export TORCH_CUDA_ARCH_LIST="8.6"
pip install -e src/third_party/detectron2
pip install -r src/home_robot/home_robot/perception/detection/detic/Detic/requirements.txt
pip install -e src/third_party/habitat-lab/habitat-lab
pip install -e src/third_party/habitat-lab/habitat-baselines
pip install "gym>=0.25" bresenham gdown
pip install sophuspy --upgrade
pip install -e yolov10          # ultralytics fork that provides the YOLOv10 and YOLO-World classes
pip install huggingface-hub==0.36.0
pip install gurobipy==12.0.3    # habitat's nav.py imports it; MSPL also needs a license 
pip install pyzmq               # realworld_goat_agent imports zmq; multiagent_eval_episode imports that agent in sim too

mkdir -p src/home_robot/home_robot/perception/detection/detic/Detic/models
wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
    -O src/home_robot/home_robot/perception/detection/detic/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
    --no-check-certificate

gdown https://drive.google.com/uc?id=1N0UbpXK3v7oTphC4LoDqlNeMHbrwkbPe
unzip goat-bench.zip
mv data/datasets/goat_bench data/datasets/goat_openvocab
mv data/datasets/goat_openvocab/hm3d/v1 data/datasets/goat_openvocab/hm3d/v0.1.2_fixed

# ObjectNav HM3D episodes: v1 (habitat_objnav_2022 config) and v2 (habitat_objnav_2023 config)
mkdir -p data/datasets/objectnav/hm3d
wget https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip -O data/datasets/objectnav_hm3d_v1.zip
wget https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip -O data/datasets/objectnav_hm3d_v2.zip
unzip -q data/datasets/objectnav_hm3d_v1.zip -d data/datasets/objectnav/hm3d && mv data/datasets/objectnav/hm3d/objectnav_hm3d_v1 data/datasets/objectnav/hm3d/v1
unzip -q data/datasets/objectnav_hm3d_v2.zip -d data/datasets/objectnav/hm3d && mv data/datasets/objectnav/hm3d/objectnav_hm3d_v2 data/datasets/objectnav/hm3d/v2

# HM3D scenes: method 1: mount the HM3D dataset when creating the container at data/versioned_data/hm3d-0.2/hm3d (docker run -v ... at the top of this script). The HM3D dataset is available at https://matterport.com/habitat-matterport-3d-research-dataset
# HM3D scenes: method 2: download with your credentials. Uncomment the following lines and fill in your HM3D_USERNAME and HM3D_PASSWORD above. The download will take a long time (tens of GBs). 
# yes | python -m habitat_sim.utils.datasets_download \
#     --username ${HM3D_USERNAME} --password ${HM3D_PASSWORD} --uids hm3d_full

# Link the HM3D dataset to the expected location in data/scene_datasets. The symlink is needed for the task datasets to find the scenes.
mkdir -p data/scene_datasets
ln -sfn /home-robot/data/versioned_data/hm3d-0.2/hm3d data/scene_datasets/hm3d_v0.2
ln -sfn /home-robot/data/versioned_data/hm3d-0.2/hm3d data/scene_datasets/hm3d


# Gurobi license, needed only for MSPL in multi-agent GOAT runs. Put gurobi.lic at
# /root/.gurobi/gurobi.lic and add this line to ~/.bashrc:
#   export GRB_LICENSE_FILE=/root/.gurobi/gurobi.lic