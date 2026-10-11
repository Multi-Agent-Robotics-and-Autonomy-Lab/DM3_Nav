# DM3-Nav — Docker build.
# Pass your own credentials at build time:
#   docker build -f DM3_Nav.DOCKERFILE --build-arg HM3D_USERNAME=... --build-arg HM3D_PASSWORD=... -t dm3-nav .
# Or build without them and mount an HM3D copy when you start the container:
#   docker run -it --gpus all -v /path/to/hm3d-0.2:/home-robot/data/versioned_data/hm3d-0.2:ro dm3-nav

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
        src/home_robot/home_robot/agent/imagenav_agent/SuperGluePretrainedNetwork \
        yolov10 && \
    export TORCH_CUDA_ARCH_LIST="8.6" && \
    pip install -e src/third_party/detectron2 && \
    pip install -r src/home_robot/home_robot/perception/detection/detic/Detic/requirements.txt && \
    pip install -e src/third_party/habitat-lab/habitat-lab && \
    pip install -e src/third_party/habitat-lab/habitat-baselines && \
    pip install "gym>=0.25" bresenham gdown && \
    pip install sophuspy --upgrade && \
    pip install -e yolov10 && \
    pip install huggingface-hub==0.36.0 && \
    pip install gurobipy==12.0.3 && \
    pip install pyzmq && \
    gdown https://drive.google.com/uc?id=1N0UbpXK3v7oTphC4LoDqlNeMHbrwkbPe

# Download pretrained Detic model
RUN mkdir -p src/home_robot/home_robot/perception/detection/detic/Detic/models && \
    wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
        -O src/home_robot/home_robot/perception/detection/detic/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
        --no-check-certificate


# Download goat benchmark dataset and unzip
RUN unzip goat-bench.zip && \
    mv data/datasets/goat_bench data/datasets/goat_openvocab && \
    mv data/datasets/goat_openvocab/hm3d/v1 data/datasets/goat_openvocab/hm3d/v0.1.2_fixed

# ObjectNav HM3D episodes: v1 (habitat_objnav_2022 config) and v2 (habitat_objnav_2023 config)
RUN mkdir -p data/datasets/objectnav/hm3d && \
    wget https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip -O data/datasets/objectnav_hm3d_v1.zip && \
    wget https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip -O data/datasets/objectnav_hm3d_v2.zip && \
    unzip -q data/datasets/objectnav_hm3d_v1.zip -d data/datasets/objectnav/hm3d && mv data/datasets/objectnav/hm3d/objectnav_hm3d_v1 data/datasets/objectnav/hm3d/v1 && \
    unzip -q data/datasets/objectnav_hm3d_v2.zip -d data/datasets/objectnav/hm3d && mv data/datasets/objectnav/hm3d/objectnav_hm3d_v2 data/datasets/objectnav/hm3d/v2

# HM3D scenes: method 1: mount the HM3D dataset when creating the container at data/versioned_data/hm3d-0.2/hm3d (docker run -v ... at the top of this file). The HM3D dataset is available at https://matterport.com/habitat-matterport-3d-research-dataset
# HM3D scenes: method 2: download with your credentials. Pass --build-arg HM3D_USERNAME=... --build-arg HM3D_PASSWORD=... to docker build (top of this file); without them this step is skipped. The download will take a long time (tens of GBs).
RUN if [ -n "${HM3D_USERNAME}" ]; then \
        source /opt/conda/etc/profile.d/conda.sh && \
        conda activate home-robot && \
        yes | python -m habitat_sim.utils.datasets_download \
        --username ${HM3D_USERNAME} \
        --password ${HM3D_PASSWORD} \
        --uids hm3d_full; \
    fi

# Link the HM3D dataset to the expected location in data/scene_datasets. The symlink is needed for the task datasets to find the scenes.
RUN mkdir -p data/scene_datasets && \
    ln -sfn /home-robot/data/versioned_data/hm3d-0.2/hm3d data/scene_datasets/hm3d_v0.2 && \
    ln -sfn /home-robot/data/versioned_data/hm3d-0.2/hm3d data/scene_datasets/hm3d


