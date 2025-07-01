"""CARLA simulation environment interface for autonomous driving training.

This module provides a clean interface to CARLA simulator for multi-camera
data collection, vehicle control, and scenario generation as outlined in Phase 1.
"""

from __future__ import annotations

import time
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import weakref

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False
    # Define dummy classes for type hints
    class carla:
        class World: pass
        class Client: pass
        class Vehicle: pass
        class Sensor: pass
        class Transform: pass
        class Location: pass
        class Rotation: pass

__all__ = ["CarlaEnvironment", "CameraConfig", "VehicleState", "SensorData"]


@dataclass
class CameraConfig:
    """Configuration for camera sensors."""
    width: int = 1024
    height: int = 768
    fov: int = 110
    location: Tuple[float, float, float] = (2.0, 0.0, 1.4)  # x, y, z relative to vehicle
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # pitch, yaw, roll
    sensor_id: str = "camera_front"


@dataclass
class VehicleState:
    """Current vehicle state information."""
    location: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    angular_velocity: Tuple[float, float, float]
    acceleration: Tuple[float, float, float]
    timestamp: float


@dataclass
class SensorData:
    """Sensor data container."""
    camera_images: Dict[str, np.ndarray]  # sensor_id -> RGB image
    vehicle_state: VehicleState
    timestamp: float
    frame_id: int


class CarlaEnvironment:
    """High-level interface to CARLA simulator for autonomous driving."""
    
    def __init__(self, 
                 host: str = 'localhost',
                 port: int = 2000,
                 timeout: float = 10.0,
                 map_name: str = 'Town01',
                 weather_preset: str = 'ClearNoon'):
        """Initialize CARLA environment.
        
        Args:
            host: CARLA server host
            port: CARLA server port
            timeout: Connection timeout
            map_name: CARLA map to load
            weather_preset: Weather conditions
        """
        if not CARLA_AVAILABLE:
            raise ImportError("CARLA Python API not available. Install CARLA first.")
            
        self.host = host
        self.port = port
        self.timeout = timeout
        self.map_name = map_name
        self.weather_preset = weather_preset
        
        # CARLA objects
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.vehicle: Optional[carla.Vehicle] = None
        self.cameras: Dict[str, carla.Sensor] = {}
        
        # Data collection
        self.sensor_data: Dict[str, Any] = {}
        self.data_queue: List[SensorData] = []
        self.frame_count = 0
        
        # State tracking
        self.is_connected = False
        self.is_recording = False
        
        self.logger = logging.getLogger(__name__)
        
    def connect(self) -> bool:
        """Connect to CARLA server and setup world."""
        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout)
            
            # Get available maps
            available_maps = self.client.get_available_maps()
            self.logger.info(f"Available maps: {available_maps}")
            
            # Load the specified map
            if self.map_name not in available_maps:
                self.logger.warning(f"Map {self.map_name} not found, using first available")
                self.map_name = available_maps[0]
                
            self.world = self.client.load_world(self.map_name)
            self.logger.info(f"Loaded map: {self.map_name}")
            
            # Set weather
            self._set_weather(self.weather_preset)
            
            # Set synchronous mode for deterministic simulation
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05  # 20 FPS
            self.world.apply_settings(settings)
            
            self.is_connected = True
            self.logger.info("Successfully connected to CARLA")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to CARLA: {e}")
            self.is_connected = False
            return False
    
    def spawn_vehicle(self, 
                     vehicle_model: str = 'vehicle.tesla.model3',
                     spawn_point: Optional[carla.Transform] = None) -> bool:
        """Spawn ego vehicle in the world.
        
        Args:
            vehicle_model: Blueprint name for the vehicle
            spawn_point: Specific spawn location, if None uses random
            
        Returns:
            True if vehicle spawned successfully
        """
        if not self.is_connected:
            self.logger.error("Not connected to CARLA")
            return False
            
        try:
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.find(vehicle_model)
            
            # Get spawn point
            if spawn_point is None:
                spawn_points = self.world.get_map().get_spawn_points()
                spawn_point = np.random.choice(spawn_points)
            
            # Spawn vehicle
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            self.logger.info(f"Spawned vehicle: {vehicle_model}")
            
            # Enable autopilot for baseline behavior
            self.vehicle.set_autopilot(True)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to spawn vehicle: {e}")
            return False
    
    def setup_cameras(self, camera_configs: List[CameraConfig]) -> bool:
        """Setup multiple cameras on the vehicle.
        
        Args:
            camera_configs: List of camera configurations
            
        Returns:
            True if all cameras setup successfully
        """
        if self.vehicle is None:
            self.logger.error("No vehicle spawned")
            return False
            
        try:
            blueprint_library = self.world.get_blueprint_library()
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            
            for config in camera_configs:
                # Configure camera
                camera_bp.set_attribute('image_size_x', str(config.width))
                camera_bp.set_attribute('image_size_y', str(config.height))
                camera_bp.set_attribute('fov', str(config.fov))
                
                # Set camera transform relative to vehicle
                camera_transform = carla.Transform(
                    carla.Location(x=config.location[0], y=config.location[1], z=config.location[2]),
                    carla.Rotation(pitch=config.rotation[0], yaw=config.rotation[1], roll=config.rotation[2])
                )
                
                # Spawn camera
                camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
                
                # Setup data callback
                weak_self = weakref.ref(self)
                camera.listen(lambda image, sensor_id=config.sensor_id: 
                             self._on_camera_data(weak_self, image, sensor_id))
                
                self.cameras[config.sensor_id] = camera
                self.logger.info(f"Setup camera: {config.sensor_id}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup cameras: {e}")
            return False
    
    @staticmethod
    def _on_camera_data(weak_self, image, sensor_id: str):
        """Callback for camera data."""
        self_ref = weak_self()
        if self_ref is not None:
            # Convert CARLA image to numpy array
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))  # BGRA
            rgb_array = array[:, :, :3][:, :, ::-1]  # Convert BGRA to RGB
            
            self_ref.sensor_data[sensor_id] = {
                'image': rgb_array,
                'timestamp': image.timestamp,
                'frame': image.frame
            }
    
    def step(self) -> Optional[SensorData]:
        """Step the simulation and collect sensor data.
        
        Returns:
            SensorData if successful, None otherwise
        """
        if not self.is_connected or self.vehicle is None:
            return None
            
        try:
            # Advance simulation
            self.world.tick()
            
            # Get vehicle state
            transform = self.vehicle.get_transform()
            velocity = self.vehicle.get_velocity()
            angular_velocity = self.vehicle.get_angular_velocity()
            acceleration = self.vehicle.get_acceleration()
            
            vehicle_state = VehicleState(
                location=(transform.location.x, transform.location.y, transform.location.z),
                rotation=(transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll),
                velocity=(velocity.x, velocity.y, velocity.z),
                angular_velocity=(angular_velocity.x, angular_velocity.y, angular_velocity.z),
                acceleration=(acceleration.x, acceleration.y, acceleration.z),
                timestamp=time.time()
            )
            
            # Collect camera images
            camera_images = {}
            for sensor_id in self.cameras.keys():
                if sensor_id in self.sensor_data:
                    camera_images[sensor_id] = self.sensor_data[sensor_id]['image'].copy()
            
            # Create sensor data container
            if camera_images:  # Only return data if we have camera images
                sensor_data = SensorData(
                    camera_images=camera_images,
                    vehicle_state=vehicle_state,
                    timestamp=time.time(),
                    frame_id=self.frame_count
                )
                
                self.frame_count += 1
                
                if self.is_recording:
                    self.data_queue.append(sensor_data)
                    
                return sensor_data
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error in simulation step: {e}")
            return None
    
    def apply_control(self, throttle: float, steer: float, brake: float):
        """Apply control commands to the vehicle.
        
        Args:
            throttle: Throttle input [0.0, 1.0]
            steer: Steering input [-1.0, 1.0]
            brake: Brake input [0.0, 1.0]
        """
        if self.vehicle is None:
            return
            
        control = carla.VehicleControl()
        control.throttle = np.clip(throttle, 0.0, 1.0)
        control.steer = np.clip(steer, -1.0, 1.0)
        control.brake = np.clip(brake, 0.0, 1.0)
        
        self.vehicle.apply_control(control)
    
    def start_recording(self):
        """Start data collection."""
        self.is_recording = True
        self.data_queue.clear()
        self.logger.info("Started recording")
    
    def stop_recording(self) -> List[SensorData]:
        """Stop data collection and return collected data."""
        self.is_recording = False
        data = self.data_queue.copy()
        self.logger.info(f"Stopped recording. Collected {len(data)} frames")
        return data
    
    def save_episode(self, data: List[SensorData], output_dir: Path):
        """Save collected episode data to disk.
        
        Args:
            data: List of sensor data frames
            output_dir: Directory to save data
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save images
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        # Save poses and metadata
        poses_file = output_dir / "poses.txt"
        metadata_file = output_dir / "metadata.txt"
        
        with open(poses_file, 'w') as f_poses, open(metadata_file, 'w') as f_meta:
            for i, frame in enumerate(data):
                timestamp = f"{i:06d}"
                
                # Save camera images
                for sensor_id, image in frame.camera_images.items():
                    import cv2
                    image_path = frames_dir / f"{timestamp}_{sensor_id}.png"
                    cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                
                # Save pose (timestamp x y z qx qy qz qw)
                loc = frame.vehicle_state.location
                rot = frame.vehicle_state.rotation
                # Convert Euler to quaternion (simplified)
                qx, qy, qz, qw = 0, 0, 0, 1  # Placeholder - implement proper conversion
                
                f_poses.write(f"{timestamp} {loc[0]:.6f} {loc[1]:.6f} {loc[2]:.6f} "
                            f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
                
                # Save metadata
                f_meta.write(f"{timestamp}: frame_id={frame.frame_id}, "
                           f"timestamp={frame.timestamp:.6f}\n")
        
        self.logger.info(f"Saved episode with {len(data)} frames to {output_dir}")
    
    def _set_weather(self, preset: str):
        """Set weather conditions."""
        weather_presets = {
            'ClearNoon': carla.WeatherParameters.ClearNoon,
            'CloudyNoon': carla.WeatherParameters.CloudyNoon,
            'WetNoon': carla.WeatherParameters.WetNoon,
            'WetCloudyNoon': carla.WeatherParameters.WetCloudyNoon,
            'SoftRainNoon': carla.WeatherParameters.SoftRainNoon,
            'MidRainyNoon': carla.WeatherParameters.MidRainyNoon,
            'HardRainNoon': carla.WeatherParameters.HardRainNoon,
            'ClearSunset': carla.WeatherParameters.ClearSunset,
        }
        
        weather = weather_presets.get(preset, carla.WeatherParameters.ClearNoon)
        self.world.set_weather(weather)
        self.logger.info(f"Set weather to: {preset}")
    
    def cleanup(self):
        """Clean up CARLA resources."""
        try:
            # Destroy cameras
            for camera in self.cameras.values():
                if camera is not None:
                    camera.destroy()
            self.cameras.clear()
            
            # Destroy vehicle
            if self.vehicle is not None:
                self.vehicle.destroy()
                self.vehicle = None
            
            # Reset synchronous mode
            if self.world is not None:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
            
            self.is_connected = False
            self.logger.info("Cleaned up CARLA resources")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


def create_default_camera_setup() -> List[CameraConfig]:
    """Create default 6-camera setup for autonomous driving.
    
    Returns:
        List of camera configurations covering 360° view
    """
    return [
        CameraConfig(sensor_id="front", location=(2.0, 0.0, 1.4), rotation=(0, 0, 0)),
        CameraConfig(sensor_id="front_left", location=(1.8, -0.5, 1.4), rotation=(0, -45, 0)),
        CameraConfig(sensor_id="front_right", location=(1.8, 0.5, 1.4), rotation=(0, 45, 0)),
        CameraConfig(sensor_id="rear", location=(-2.0, 0.0, 1.4), rotation=(0, 180, 0)),
        CameraConfig(sensor_id="rear_left", location=(-1.8, -0.5, 1.4), rotation=(0, -135, 0)),
        CameraConfig(sensor_id="rear_right", location=(-1.8, 0.5, 1.4), rotation=(0, 135, 0)),
    ] 