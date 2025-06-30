import pytest
from pathlib import Path

from opendrive_xai.data import CarlaSampleDataset


def test_dataset_missing(tmp_path: Path):
    # Point to an empty dir, expect FileNotFoundError
    with pytest.raises(FileNotFoundError):
        _ = CarlaSampleDataset(root=tmp_path) 