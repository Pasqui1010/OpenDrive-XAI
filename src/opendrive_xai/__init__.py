"""OpenDrive-XAI core package."""

__all__ = [
    "get_config",
    "get_config_manager",
    "validate_config",
    "get_device",
    "is_debug_mode",
    "get_logger",
    "start_monitoring",
    "stop_monitoring",
    "get_system_status",
    "CausalWorldModel",
    "WorldState",
    "CausalWorldModelTrainer",
    "InterpretableDrivingSystem",
    "ModularSystemTrainer",
    "PerformanceMonitor",
    "SafetyMonitor",
    "SystemHealthMonitor",
]

from pathlib import Path

# Configuration
from .config import (
    get_config,
    get_config_manager,
    validate_config,
    get_device,
    is_debug_mode,
)

# Monitoring
from .monitoring import (
    start_monitoring,
    stop_monitoring,
    get_system_status,
    PerformanceMonitor,
    SafetyMonitor,
    SystemHealthMonitor,
)

# Core models
from .causal_world_model import (
    CausalWorldModel,
    WorldState,
    CausalWorldModelTrainer,
)

# Modular components
from .modular_components import (
    InterpretableDrivingSystem,
    ModularSystemTrainer,
)

# Legacy imports for backward compatibility
from .logger import get_logger

__version__ = "0.1.0"

# Convenience: load default config at import time
DEFAULT_CFG = get_config()
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
