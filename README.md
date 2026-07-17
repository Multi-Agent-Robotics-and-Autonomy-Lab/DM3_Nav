# DM³-Nav: Decentralized Multi-Agent Multimodal Multi-Object Semantic Navigation

**Amin Kashiri, Atharva Jamsandekar, Yasin Yazıcıoğlu** — Northeastern University

Accepted at IROS 2026.

DM³-Nav is a fully decentralized multi-robot semantic navigation system. A team of robots locates multiple objects specified by category, language description, or reference image, coordinating only through ad-hoc pairwise communication, with no central planner and no shared global map.

## Demo

[▶️ Watch the real-world demo](https://drive.google.com/file/d/1QiUSCn5rIvtuTUqtuXLPgmt6S8x9-MCZ/view?usp=drive_link): two robots finding 8 targets in an office, using onboard sensing and compute only.

## Setup

HM3D requires an [access agreement](https://matterport.com/habitat-matterport-3d-research-dataset). Build the image with your own credentials:

```bash
docker build -f DM3_Nav.Dockerfile \
    --build-arg HM3D_USERNAME=<your_username> \
    --build-arg HM3D_PASSWORD=<your_password> \
    -t dm3-nav .
```

Alternatively, run `init_container.sh` inside a container started from the base image.

## Usage

Inside the container (conda env `home-robot`):

```bash
# Single-agent
python projects/habitat_goat/eval_episode.py

# Multi-agent
python projects/habitat_goat/multiagent_eval_episode.py
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