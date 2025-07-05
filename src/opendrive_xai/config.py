"""
Configuration management for OpenDrive-XAI.

This module provides a centralized configuration system that supports:
- Environment-based configuration
- Validation of configuration parameters
- Default values for all components
- Type safety and documentation
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, field_validator
import torch


class SensorConfig(BaseModel):
    """Configuration for vehicle sensors."""
    
    camera: Dict[str, Any] = Field(
        default={
            "resolution": [640, 480],
            "fps": 30,
            "fov": 90,
            "enabled": True
        },
        description="Camera sensor configuration"
    )
    
    lidar: Dict[str, Any] = Field(
        default={
            "points_per_scan": 1000,
            "fps": 10,
            "max_range": 100,
            "enabled": True
        },
        description="LiDAR sensor configuration"
    )
    
    radar: Dict[str, Any] = Field(
        default={
            "max_range": 200,
            "fps": 20,
            "enabled": True
        },
        description="Radar sensor configuration"
    )
    
    imu: Dict[str, Any] = Field(
        default={
            "fps": 100,
            "enabled": True
        },
        description="IMU sensor configuration"
    )


class ActuatorConfig(BaseModel):
    """Configuration for vehicle actuators."""
    
    steering: Dict[str, Any] = Field(
        default={
            "max_angle": 30,
            "response_time": 0.1,
            "enabled": True
        },
        description="Steering actuator configuration"
    )
    
    throttle: Dict[str, Any] = Field(
        default={
            "max_value": 1.0,
            "response_time": 0.05,
            "enabled": True
        },
        description="Throttle actuator configuration"
    )
    
    brake: Dict[str, Any] = Field(
        default={
            "max_value": 1.0,
            "response_time": 0.05,
            "enabled": True
        },
        description="Brake actuator configuration"
    )


class SafetyConfig(BaseModel):
    """Configuration for safety monitoring."""
    
    thresholds: Dict[str, float] = Field(
        default={
            "collision_risk": 0.7,
            "speed_violation": 0.8,
            "lane_deviation": 0.6,
            "traffic_violation": 0.9,
            "sensor_failure": 0.5
        },
        description="Safety thresholds for different risk factors"
    )
    
    emergency_actions: Dict[str, str] = Field(
        default={
            "collision_imminent": "emergency_brake",
            "sensor_failure": "safe_stop",
            "system_failure": "emergency_stop"
        },
        description="Emergency actions for different safety violations"
    )
    
    monitoring_frequency: float = Field(
        default=10.0,
        description="Safety monitoring frequency in Hz"
    )


class ModelConfig(BaseModel):
    """Configuration for AI models."""
    
    causal_world_model: Dict[str, Any] = Field(
        default={
            "state_dim": 64,
            "hidden_dim": 128,
            "num_layers": 3,
            "num_heads": 8,
            "dropout": 0.1
        },
        description="Causal world model configuration"
    )
    
    interpretable_system: Dict[str, Any] = Field(
        default={
            "perception_dim": 512,
            "planning_dim": 256,
            "control_dim": 64,
            "max_history_size": 100
        },
        description="Interpretable driving system configuration"
    )
    
    training: Dict[str, Any] = Field(
        default={
            "learning_rate": 1e-4,
            "batch_size": 32,
            "num_epochs": 100,
            "validation_split": 0.2
        },
        description="Training configuration"
    )


class SystemConfig(BaseModel):
    """Configuration for system integration."""
    
    control_frequency: float = Field(
        default=20.0,
        description="Control loop frequency in Hz"
    )
    
    max_control_latency: float = Field(
        default=0.1,
        description="Maximum allowed control latency in seconds"
    )
    
    thread_safety: Dict[str, Any] = Field(
        default={
            "use_threading": True,
            "max_worker_threads": 4,
            "queue_size": 100
        },
        description="Thread safety configuration"
    )
    
    logging: Dict[str, Any] = Field(
        default={
            "level": "INFO",
            "file": "opendrive_xai.log",
            "max_file_size": "10MB",
            "backup_count": 5
        },
        description="Logging configuration"
    )


class SimulationConfig(BaseModel):
    """Configuration for simulation environment."""
    
    simulator: str = Field(
        default="carla",
        description="Simulator to use (carla, gazebo, isaac)"
    )
    
    environment: Dict[str, Any] = Field(
        default={
            "map": "Town01",
            "weather": "ClearNoon",
            "traffic_density": 0.3,
            "pedestrian_density": 0.1
        },
        description="Simulation environment configuration"
    )
    
    scenarios: List[str] = Field(
        default=[
            "straight_road",
            "intersection",
            "parking",
            "emergency_brake"
        ],
        description="List of scenarios to test"
    )


class OpenDriveXAIConfig(BaseModel):
    """Main configuration class for OpenDrive-XAI."""
    
    # Component configurations
    sensors: SensorConfig = Field(default_factory=SensorConfig)
    actuators: ActuatorConfig = Field(default_factory=ActuatorConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    
    # Global settings
    device: str = Field(
        default="auto",
        description="Device to use (auto, cpu, cuda, mps)"
    )
    
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode for additional logging and validation"
    )
    
    save_checkpoints: bool = Field(
        default=True,
        description="Save model checkpoints during training"
    )
    
    checkpoint_dir: str = Field(
        default="checkpoints",
        description="Directory to save checkpoints"
    )
    
    @field_validator('device', mode='before')
    def validate_device(cls, v):
        """Validate and set device automatically."""
        if v == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return v
    
    @field_validator('checkpoint_dir')
    def validate_checkpoint_dir(cls, v):
        """Ensure checkpoint directory exists."""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v


class ConfigManager:
    """Manager for configuration loading, validation, and access."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> OpenDriveXAIConfig:
        """Load configuration from file or environment variables."""
        config_dict = {}
        
        # Load from file if provided
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                config_dict = json.load(f)
        
        # Override with environment variables
        config_dict.update(self._load_from_env())
        
        # Create configuration object
        return OpenDriveXAIConfig(**config_dict)
    
    def _load_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}
        
        # Device configuration
        if 'OPENDRIVE_DEVICE' in os.environ:
            env_config['device'] = os.environ['OPENDRIVE_DEVICE']
        
        # Debug mode
        if 'OPENDRIVE_DEBUG' in os.environ:
            env_config['debug_mode'] = os.environ['OPENDRIVE_DEBUG'].lower() == 'true'
        
        # Control frequency
        if 'OPENDRIVE_CONTROL_FREQ' in os.environ:
            env_config['system'] = env_config.get('system', {})
            env_config['system']['control_frequency'] = float(os.environ['OPENDRIVE_CONTROL_FREQ'])
        
        # Safety thresholds
        if 'OPENDRIVE_COLLISION_THRESHOLD' in os.environ:
            env_config['safety'] = env_config.get('safety', {})
            env_config['safety']['thresholds'] = env_config['safety'].get('thresholds', {})
            env_config['safety']['thresholds']['collision_risk'] = float(os.environ['OPENDRIVE_COLLISION_THRESHOLD'])
        
        return env_config
    
    def get_config(self) -> OpenDriveXAIConfig:
        """Get the current configuration."""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        current_dict = self.config.dict()
        current_dict.update(updates)
        self.config = OpenDriveXAIConfig(**current_dict)
    
    def save_config(self, path: Optional[str] = None) -> None:
        """Save current configuration to file."""
        save_path = path or self.config_path or "config.json"
        with open(save_path, 'w') as f:
            json.dump(self.config.dict(), f, indent=2)
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return any issues."""
        issues = []
        
        # Check device availability
        if self.config.device == "cuda" and not torch.cuda.is_available():
            issues.append("CUDA device requested but not available")
        
        # Check control frequency
        if self.config.system.control_frequency <= 0:
            issues.append("Control frequency must be positive")
        
        # Check safety thresholds
        for name, threshold in self.config.safety.thresholds.items():
            if not 0 <= threshold <= 1:
                issues.append(f"Safety threshold {name} must be between 0 and 1")
        
        # Check model dimensions
        if self.config.model.causal_world_model['state_dim'] <= 0:
            issues.append("State dimension must be positive")
        
        return issues
    
    def get_device(self) -> str:
        """Get the configured device."""
        return self.config.device
    
    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled."""
        return self.config.debug_mode
    
    def get_sensor_config(self) -> Dict[str, Any]:
        """Get sensor configuration."""
        return self.config.sensors.dict()
    
    def get_actuator_config(self) -> Dict[str, Any]:
        """Get actuator configuration."""
        return self.config.actuators.dict()
    
    def get_safety_config(self) -> Dict[str, Any]:
        """Get safety configuration."""
        return self.config.safety.dict()
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config.model.dict()
    
    def get_system_config(self) -> Dict[str, Any]:
        """Get system configuration."""
        return self.config.system.dict()
    
    def get_simulation_config(self) -> Dict[str, Any]:
        """Get simulation configuration."""
        return self.config.simulation.dict()


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """Get or create global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


def get_config() -> OpenDriveXAIConfig:
    """Get the current configuration."""
    return get_config_manager().get_config()


def update_config(updates: Dict[str, Any]) -> None:
    """Update the global configuration."""
    get_config_manager().update_config(updates)


def save_config(path: Optional[str] = None) -> None:
    """Save the current configuration."""
    get_config_manager().save_config(path)


def validate_config() -> List[str]:
    """Validate the current configuration."""
    return get_config_manager().validate_config()


# Convenience functions for common configuration access
def get_device() -> str:
    """Get the configured device."""
    return get_config_manager().get_device()


def is_debug_mode() -> bool:
    """Check if debug mode is enabled."""
    return get_config_manager().is_debug_mode()


def get_sensor_config() -> Dict[str, Any]:
    """Get sensor configuration."""
    return get_config_manager().get_sensor_config()


def get_actuator_config() -> Dict[str, Any]:
    """Get actuator configuration."""
    return get_config_manager().get_actuator_config()


def get_safety_config() -> Dict[str, Any]:
    """Get safety configuration."""
    return get_config_manager().get_safety_config()


def get_model_config() -> Dict[str, Any]:
    """Get model configuration."""
    return get_config_manager().get_model_config()


def get_system_config() -> Dict[str, Any]:
    """Get system configuration."""
    return get_config_manager().get_system_config()


def get_simulation_config() -> Dict[str, Any]:
    """Get simulation configuration."""
    return get_config_manager().get_simulation_config()
