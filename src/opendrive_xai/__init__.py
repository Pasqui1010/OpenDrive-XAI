"""OpenDrive-XAI core package."""

__all__ = [
    "Config",
    "get_logger",
]

from pathlib import Path

from .config import Config  # noqa: E402  pylint: disable=wrong-import-position
from .logger import get_logger  # noqa: E402  pylint: disable=wrong-import-position

__version__ = "0.0.1"

# Convenience: load default config at import time
DEFAULT_CFG = Config()
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
