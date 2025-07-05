# OpenDrive-XAI

> **⚠️ Python 3.12 Required:**
> This project currently requires **Python 3.12** due to dependency compatibility (notably pydantic and CARLA). Python 3.13+ is not yet supported. Please ensure you are using Python 3.12 when setting up your environment.

![CI](https://github.com/Pasqui1010/opendrive-xai/actions/workflows/ci.yml/badge.svg)

Robust, explainable end-to-end autonomous driving stack built for real-time research & education.

**License  Apache-2.0**

---

## 1. Features

* Multi-sensor (camera / LiDAR / radar) → BEV Vision Transformer backbone
* MPC / Diffusion-based planner with adversarial robustness (MAD)
* Built-in CARLA scenario regression & Isaac Sim synthetic-data tools
* First-class explainability: faithful attention maps, causal graphs
* ROS 2 native, but gRPC ports for non-ROS consumers
* **✅ Recent Fixes**: VehicleInterface integration, Pydantic V2 compatibility, PyTorch warnings resolved

## 2. Quick Start (≤10 min)

```bash
# clone
git clone https://github.com/Pasqui1010/opendrive-xai.git
cd opendrive-xai && ./scripts/download_sample_data.sh

# run container with sample scenario
docker compose up --build quickstart
# open http://localhost:8888 for real-time BEV & XAI overlays
```

See also [docs/quickstart_no_docker.md](docs/quickstart_no_docker.md) for a CPU-only setup.

## 3. Repository Layout (TL;DR)

```
src/            core code (perception, planning, sim)
docs/           mkdocs sources
scripts/        helper scripts & developer tooling
tests/          unit & closed-loop smoke tests
environment/    dockerfiles & Conda envs
```

## 4. Recent Updates (Latest)

### ✅ **System Improvements**
- **VehicleInterface Integration**: Fixed import errors and integration issues
- **Pydantic V2 Migration**: Updated validators to use `@field_validator` syntax
- **PyTorch Warnings**: Resolved tensor variance calculation warnings
- **Test Suite**: 100/105 tests passing (95.2% success rate)
- **Deployment**: All major deployment stages now pass successfully

### 🔧 **Technical Fixes**
- Fixed `OpenDriveXAIConfig` import in test suite
- Resolved missing `Dict, Any` type imports
- Updated attention gate variance calculation with proper error handling
- Improved error messages and validation logic

### 📊 **Current Status**
- **Core Fixes**: 11/11 tests passing ✅
- **Integration Tests**: 13/13 tests passing ✅
- **System Integration**: All major components operational ✅
- **Deployment Pipeline**: Environment validation, dependencies, configuration, directories, tests, monitoring all pass ✅

## 5. Contributing

We welcome research prototypes, bug reports, and new evaluation scenarios!
Please read [CONTRIBUTING.md](CONTRIBUTING.md) for code style, DCO sign-off, and the "good first issue" label.

## 6. Roadmap

See [ROADMAP.md](ROADMAP.md) for milestone details — immediate goal: **v0.1 lane-keeping demo (Q2 2025)**.

## 7. Citation

If you use OpenDrive-XAI in academic work, please cite:

```bibtex
@misc{opendrive2025,
  title  = {OpenDrive-XAI},
  author = {Pasqui1010 et al.},
  year   = {2025},
  url    = {https://github.com/Pasqui1010/opendrive-xai}
}
``` 