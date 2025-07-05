# CPU-only Quick-Start (no Docker)

> **⚠️ Python 3.12 Required:**
> This project currently requires **Python 3.12** due to dependency compatibility (notably pydantic and CARLA). Python 3.13+ is not yet supported. Please ensure you are using Python 3.12 when setting up your environment.

> Goal: run the lane-keeping demo on any machine without GPU & without installing ROS.

1. Install Python 3.11+ and Git.
2. Clone & enter the repo:

```bash
git clone https://github.com/Pasqui1010/opendrive-xai.git
cd opendrive-xai
```
3. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```
4. Install dependencies:

```bash
pip install -r requirements.txt
```
5. Download the small dataset (≈50 MB):

```bash
./scripts/download_sample_data.sh
```
6. Run the lane-keeping notebook:

```bash
pip install jupyterlab
jupyter lab docs/notebooks/lane_keep_demo.ipynb
```

The notebook replays a short CARLA log and shows the BEV inference plus planned waypoints—all on CPU. Estimated runtime < 2 minutes.

If you later install Docker + NVIDIA drivers, switch to the Docker workflow in the main README for GPU-accelerated training. 