"""Central configuration object.
Currently minimal; will grow alongside the project.
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, validator


class Config(BaseModel):
    """Project-wide configuration (loaded at runtime)."""

    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data")
    log_level: str = "INFO"

    # Training hyper-params (placeholders)
    batch_size: int = 4
    num_workers: int = 2

    # Optional experiment name (for wandb, tensorboard, …)
    experiment: Optional[str] = None

    @validator("log_level")
    def _validate_level(cls, val: str) -> str:  # noqa: N805
        levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if val.upper() not in levels:
            raise ValueError(f"Invalid log_level '{val}'. Choose from {levels}.")
        return val.upper()

    class Config:
        arbitrary_types_allowed = True
        frozen = True 