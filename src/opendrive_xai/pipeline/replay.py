"""Run encoder over sample dataset and export a GIF visualisation.

Usage (after downloading sample data):

```bash
python -m opendrive_xai.pipeline.replay --out demo.gif
```pyt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import imageio.v3 as iio
import matplotlib

# Use a non-interactive backend so that unit tests and headless environments
# (e.g. CI) do not require a display server. The Agg backend also provides the
# `tostring_rgb` method that we rely on when converting the rendered canvas
# to a NumPy array.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF

from ..data import CarlaSampleDataset
from ..perception import TinyBEVEncoder
from ..logger import get_logger

logger = get_logger(__name__)


def _vis_frame(img, bev_tensor: torch.Tensor) -> np.ndarray:  # noqa: D401
    """Overlay BEV feature map heatmap on the RGB frame and return as numpy."""
    bev = bev_tensor.mean(0).cpu().numpy()  # (H, W)
    bev_norm = (bev - bev.min()) / (np.ptp(bev) + 1e-6)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.imshow(img)
    ax.imshow(bev_norm, cmap="viridis", alpha=0.6)
    ax.axis("off")
    fig.tight_layout(pad=0)

    fig.canvas.draw()
    # ``buffer_rgba`` is available for all Agg backends and returns an ARGB
    # byte string. We reshape and drop the alpha channel to obtain an RGB
    # image array suitable for ImageIO.
    arr = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)  # type: ignore[attr-defined]
    arr = arr.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)
    return arr


def run(dataset_root: Path | str | None = None, out_path: Path | str = "replay.gif") -> None:  # noqa: D401
    ds = CarlaSampleDataset(root=dataset_root)
    model = TinyBEVEncoder(bev_channels=64).eval()
    torch.set_grad_enabled(False)

    frames: List[np.ndarray] = []
    for img, _ in ds:
        inp = TF.to_tensor(img).unsqueeze(0)  # (1,3,H,W)
        bev = model(inp)[0]  # (C,h,w)
        frames.append(_vis_frame(img, bev))

    iio.imwrite(out_path, np.stack(frames), duration=100)
    logger.info("Exported %d frames to %s", len(frames), out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run encoder on sample log and export GIF")
    parser.add_argument("--root", type=str, default=None, help="Dataset root (default: data/sample)")
    parser.add_argument("--out", type=str, default="replay.gif", help="Output GIF path")
    args = parser.parse_args()
    run(args.root, args.out) 