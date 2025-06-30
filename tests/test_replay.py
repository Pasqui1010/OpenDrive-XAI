from pathlib import Path
import numpy as np
import imageio.v3 as iio
from types import SimpleNamespace

import pytest

from opendrive_xai.pipeline import replay


def _make_stub_dataset(tmp: Path):
    (tmp / "frames").mkdir()
    poses = []
    for i in range(2):
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        iio.imwrite(tmp / "frames" / f"00000{i}.png", arr)
        poses.append(f"00000{i} 0 0 0 0 0 0 1\n")
    (tmp / "poses.txt").write_text("".join(poses))


@pytest.mark.parametrize("out_name", ["demo.gif"])
def test_replay(tmp_path: Path, out_name: str):
    _make_stub_dataset(tmp_path)
    replay.run(tmp_path, tmp_path / out_name)
    assert (tmp_path / out_name).exists() 