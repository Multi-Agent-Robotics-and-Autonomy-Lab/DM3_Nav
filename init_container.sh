#!/bin/bash
# DM3-Nav — manual container setup (alternative to DM3_Nav.Dockerfile).
# Run inside a container started from the base image:
#   docker run -it --gpus all -v /path/to/hm3d-0.2:/home-robot/data/versioned_data \
#       fairembodied/habitat-challenge:homerobot-ovmm-challenge-2023 bash

REPO_URL=https://github.com/Multi-Agent-Robotics-and-Autonomy-Lab/DM3_nav.git
BRANCH=main

# HM3D credentials — needed only if you download HM3D instead of mounting it.
# Get your own (access agreement): https://matterport.com/habitat-matterport-3d-research-dataset
HM3D_USERNAME=<your_hm3d_username>
HM3D_PASSWORD=<your_hm3d_password>

curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt install git-lfs

# Overlay the DM3-Nav repo onto /home-robot (keeps the mounted data volume)
git clone --recurse-submodules --branch ${BRANCH} ${REPO_URL} /tmp/dm3nav
cp -a /tmp/dm3nav/. /home-robot/
rm -rf /tmp/dm3nav
cd /home-robot

source /opt/conda/etc/profile.d/conda.sh
conda activate home-robot
git submodule update --init --recursive src/third_party/detectron2 \
    src/home_robot/home_robot/perception/detection/detic/Detic \
    src/third_party/contact_graspnet \
    src/home_robot/home_robot/agent/imagenav_agent/SuperGluePretrainedNetwork/ \
    src/third_party/habitat-lab/
export TORCH_CUDA_ARCH_LIST="8.6"
pip install -e src/third_party/detectron2
pip install -r src/home_robot/home_robot/perception/detection/detic/Detic/requirements.txt
pip install -e src/third_party/habitat-lab/habitat-lab
pip install -e src/third_party/habitat-lab/habitat-baselines
pip install "gym>=0.25" bresenham gdown
pip install sophuspy --upgrade

mkdir -p home-robot/src/home_robot/home_robot/perception/detection/detic/Detic/models
wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
    -O home-robot/src/home_robot/home_robot/perception/detection/detic/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
    --no-check-certificate

gdown https://drive.google.com/uc?id=1N0UbpXK3v7oTphC4LoDqlNeMHbrwkbPe
unzip goat-bench.zip
mv data/datasets/goat_bench data/datasets/goat_openvocab
mv data/datasets/goat_openvocab/hm3d/v1 data/datasets/goat_openvocab/hm3d/v0.1.2_fixed

# HM3D scenes: mount them (see docker run above), or download with your credentials:
mkdir data/scene_datasets
ln -s /home-robot/data/versioned_data/hm3d-0.2/hm3d data/scene_datasets/hm3d
# yes | python -m habitat_sim.utils.datasets_download \
#     --username ${HM3D_USERNAME} --password ${HM3D_PASSWORD} --uids hm3d_full