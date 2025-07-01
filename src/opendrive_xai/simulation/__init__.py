"""Simulation environments for autonomous driving training and testing."""

__all__ = ["CarlaEnvironment", "CameraConfig", "VehicleState", "SensorData"]

try:
    from .carla_env import CarlaEnvironment, CameraConfig, VehicleState, SensorData
except ImportError:
    # CARLA not available
    pass
