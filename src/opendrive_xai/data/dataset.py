"""Minimal dataset loader for the sample CARLA log.

The sample dataset is a ZIP extracted by `scripts/download_sample_data.sh` into
`<project>/data/sample/` and contains a folder structure:

```
frames/
  000001.png
  000002.png
poses.txt               # Nx7: timestamp x y z qx qy qz qw
```

This loader is intentionally simple—no torch deps—to keep the project light at
this stage. Future versions may switch to PyTorch `Dataset`.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator, List, Tuple

from PIL import Image

__all__ = ["CarlaSampleDataset"]


class CarlaSampleDataset:
    """Lightweight iterator over sample CARLA frames."""

    def __init__(self, root: Path | str | None = None) -> None:
        from .. import DATA_DIR  # avoid circular at import time

        self.root = Path(root) if root else Path(DATA_DIR) / "sample"
        self.images_dir = self.root / "frames"
        self.poses_file = self.root / "poses.txt"

        if not self.images_dir.exists() or not self.poses_file.exists():
            raise FileNotFoundError(
                "Sample data not found. Run scripts/download_sample_data.sh first."
            )

        # Parse poses file once
        self._poses: List[Tuple[str, List[float]]] = []
        with self.poses_file.open() as f:
            reader = csv.reader(f, delimiter=" ")
            for row in reader:
                timestamp = row[0]
                pose = [float(x) for x in row[1:]]
                self._poses.append((timestamp, pose))

    def __len__(self) -> int:  # noqa: D401
        return len(self._poses)

    def __iter__(self) -> Iterator[Tuple[Image.Image, List[float]]]:  # type: ignore[misc]
        for timestamp, pose in self._poses:
            img_path = self.images_dir / f"{timestamp}.png"
            yield Image.open(img_path).convert("RGB"), pose
