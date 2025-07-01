"""Central configuration object.
Currently minimal; will grow alongside the project.
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict


class Config(BaseModel):
    """Project-wide configuration (loaded at runtime)."""

    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])

    # Ensure that a default data directory always exists so that downstream
    # components and unit-tests do not need to guard against its absence.
    def _default_data_dir() -> Path:  # noqa: D401
        path = Path(__file__).resolve().parents[2] / "data"
        # Create the directory (and parents) if it is missing. This is a
        # no-op when the directory already exists but guarantees the path is
        # present on fresh checkouts or CI runners.
        path.mkdir(parents=True, exist_ok=True)
        return path

    data_dir: Path = Field(default_factory=_default_data_dir)
    log_level: str = "INFO"

    # Training hyper-params (placeholders)
    batch_size: int = 4
    num_workers: int = 2

    # Optional experiment name (for wandb, tensorboard, …)
    experiment: Optional[str] = None

    @field_validator("log_level")
    @classmethod
    def _validate_level(cls, val: str) -> str:
        levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if val.upper() not in levels:
            raise ValueError(f"Invalid log_level '{val}'. Choose from {levels}.")
        return val.upper()

    # Pydantic v2 configuration
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True) 