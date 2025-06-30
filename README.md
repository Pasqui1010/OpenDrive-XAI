# OpenDrive-XAI

Robust, explainable end-to-end autonomous driving stack built for real-time research & education.

**License  Apache-2.0**

---

## 1. Features

* Multi-sensor (camera / LiDAR / radar) → BEV Vision Transformer backbone
* MPC / Diffusion-based planner with adversarial robustness (MAD)
* Built-in CARLA scenario regression & Isaac Sim synthetic-data tools
* First-class explainability: faithful attention maps, causal graphs
* ROS 2 native, but gRPC ports for non-ROS consumers

## 2. Quick Start (≤10 min)

```bash
# clone
git clone https://github.com/Pasqui1010/opendrive-xai.git
cd opendrive-xai && ./scripts/download_sample_data.sh

# run container with sample scenario
docker compose up --build quickstart
# open http://localhost:8888 for real-time BEV & XAI overlays
```

> Don't have Docker? See docs/quickstart_no_docker.md for a small-model CPU demo in Google Colab.

## 3. Repository Layout (TL;DR)

```
src/            core code (perception, planning, sim)
docs/           mkdocs sources
scripts/        helper scripts & developer tooling
tests/          unit & closed-loop smoke tests
environment/    dockerfiles & Conda envs
```

## 4. Contributing

We welcome research prototypes, bug reports, and new evaluation scenarios!
Please read [CONTRIBUTING.md](CONTRIBUTING.md) for code style, DCO sign-off, and the "good first issue" label.

## 5. Roadmap

See [ROADMAP.md](ROADMAP.md) for milestone details — immediate goal: **v0.1 lane-keeping demo (Q2 2025)**.

## 6. Citation

If you use OpenDrive-XAI in academic work, please cite:

```bibtex
@misc{opendrive2025,
  title  = {OpenDrive-XAI},
  author = {Pasqui1010 et al.},
  year   = {2025},
  url    = {https://github.com/Pasqui1010/opendrive-xai}
}
``` 