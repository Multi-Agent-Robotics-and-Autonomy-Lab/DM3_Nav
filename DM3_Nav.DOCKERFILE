# DM3-Nav — Docker build.
# HM3D requires an access agreement (https://matterport.com/habitat-matterport-3d-research-dataset).
# Pass your own credentials at build time:
#   docker build -f DM3_Nav.Dockerfile --build-arg HM3D_USERNAME=... --build-arg HM3D_PASSWORD=... -t dm3-nav .

FROM fairembodied/habitat-challenge:homerobot-ovmm-challenge-2023

ARG REPO_URL=https://github.com/Multi-Agent-Robotics-and-Autonomy-Lab/DM3_nav.git
ARG BRANCH=main
ARG HM3D_USERNAME
ARG HM3D_PASSWORD

# Install git-lfs
RUN apt-get update && apt-get install -y curl gnupg && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs

# Replace the base home-robot tree with the DM3-Nav repo
RUN rm -rf /home-robot && \
    git clone --recurse-submodules --branch ${BRANCH} ${REPO_URL} /home-robot

WORKDIR /home-robot

SHELL ["/bin/bash", "-c"]
ENV CONDA_DEFAULT_ENV=home-robot

# Activate conda and install everything
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate home-robot && \
    git submodule update --init --recursive \
        src/third_party/detectron2 \
        src/home_robot/home_robot/perception/detection/detic/Detic \
        src/third_party/contact_graspnet \
        src/home_robot/home_robot/agent/imagenav_agent/SuperGluePretrainedNetwork/ \
        src/third_party/habitat-lab/ && \
    export TORCH_CUDA_ARCH_LIST="8.6" && \
    pip install -e src/third_party/detectron2 && \
    pip install -r src/home_robot/home_robot/perception/detection/detic/Detic/requirements.txt && \
    pip install -e src/third_party/habitat-lab/habitat-lab && \
    pip install -e src/third_party/habitat-lab/habitat-baselines && \
    pip install "gym>=0.25" bresenham gdown && \
    gdown https://drive.google.com/uc?id=1N0UbpXK3v7oTphC4LoDqlNeMHbrwkbPe

# Download pretrained Detic model
RUN mkdir -p home-robot/src/home_robot/home_robot/perception/detection/detic/Detic/models && \
    wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
        -O home-robot/src/home_robot/home_robot/perception/detection/detic/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
        --no-check-certificate


# Download goat benchmark dataset and unzip
RUN unzip goat-bench.zip && \
    mv data/datasets/goat_bench data/datasets/goat_openvocab && \
    mv data/datasets/goat_openvocab/hm3d/v1 data/datasets/goat_openvocab/hm3d/v0.1.2_fixed


# Download HM3D dataset
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate home-robot && \
    yes | python -m habitat_sim.utils.datasets_download \
    --username ${HM3D_USERNAME} \
    --password ${HM3D_PASSWORD} \
    --uids hm3d_full


