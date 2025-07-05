"""
Vehicle Interface for Mechatronics Integration

This module provides the interface between the AI system and physical vehicle hardware,
including sensor fusion, actuator control, and safety monitoring for autonomous driving.
"""

import time
import threading
import queue
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
from dataclasses import dataclass
from enum import Enum
import logging
import can
from can.interfaces.vector import VectorBus
import serial

logger = logging.getLogger(__name__)


class SensorType(Enum):
    """Enumeration of sensor types."""
    CAMERA = "camera"
    LIDAR = "lidar"
    RADAR = "radar"
    IMU = "imu"
    GPS = "gps"
    WHEEL_ENCODER = "wheel_encoder"


class ActuatorType(Enum):
    """Enumeration of actuator types."""
    STEERING = "steering"
    THROTTLE = "throttle"
    BRAKE = "brake"
    GEAR = "gear"


@dataclass
class SensorData:
    """Container for sensor data with timestamp and metadata."""
    sensor_type: SensorType
    timestamp: float
    data: np.ndarray
    metadata: Dict[str, Any]
    quality_score: float  # 0.0 to 1.0


@dataclass
class ActuatorCommand:
    """Container for actuator commands with safety limits."""
    actuator_type: ActuatorType
    timestamp: float
    command: float
    min_limit: float
    max_limit: float
    safety_override: bool = False


@dataclass
class VehicleState:
    """Complete vehicle state including all sensors and actuators."""
    timestamp: float
    position: np.ndarray  # [x, y, z] in meters
    orientation: np.ndarray  # [roll, pitch, yaw] in radians
    velocity: np.ndarray  # [vx, vy, vz] in m/s
    acceleration: np.ndarray  # [ax, ay, az] in m/s²
    angular_velocity: np.ndarray  # [wx, wy, wz] in rad/s
    wheel_speeds: np.ndarray  # [fl, fr, rl, rr] in m/s
    steering_angle: float  # in radians
    throttle_position: float  # 0.0 to 1.0
    brake_pressure: float  # in bar
    gear_position: int  # -1 (reverse), 0 (neutral), 1-6 (forward)
    battery_voltage: float  # in volts
    system_temperature: float  # in Celsius


class SensorInterface:
    """
    Interface for managing sensor data acquisition and fusion.
    Handles multiple sensor types with different update rates and formats.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sensors = {}
        self.data_queues = {}
        self.fusion_thread = None
        self.running = False
        
        # Sensor fusion parameters
        self.fusion_rate = config.get('fusion_rate', 100)  # Hz
        self.sensor_timeouts = config.get('sensor_timeouts', {
            SensorType.CAMERA: 0.1,  # 100ms
            SensorType.LIDAR: 0.05,  # 50ms
            SensorType.RADAR: 0.02,  # 20ms
            SensorType.IMU: 0.01,    # 10ms
            SensorType.GPS: 0.1,     # 100ms
            SensorType.WHEEL_ENCODER: 0.01  # 10ms
        })
        
        # Initialize sensor interfaces
        self._initialize_sensors()
    
    def _initialize_sensors(self):
        """Initialize sensor interfaces based on configuration."""
        for sensor_type, sensor_config in self.config.get('sensors', {}).items():
            sensor_enum = SensorType(sensor_type)
            
            if sensor_enum == SensorType.CAMERA:
                self.sensors[sensor_enum] = CameraInterface(sensor_config)
            elif sensor_enum == SensorType.LIDAR:
                self.sensors[sensor_enum] = LidarInterface(sensor_config)
            elif sensor_enum == SensorType.RADAR:
                self.sensors[sensor_enum] = RadarInterface(sensor_config)
            elif sensor_enum == SensorType.IMU:
                self.sensors[sensor_enum] = IMUInterface(sensor_config)
            elif sensor_enum == SensorType.GPS:
                self.sensors[sensor_enum] = GPSInterface(sensor_config)
            elif sensor_enum == SensorType.WHEEL_ENCODER:
                self.sensors[sensor_enum] = WheelEncoderInterface(sensor_config)
            
            # Create data queue for each sensor
            self.data_queues[sensor_enum] = queue.Queue(maxsize=100)
    
    def start(self):
        """Start sensor data acquisition and fusion."""
        self.running = True
        
        # Start sensor threads
        for sensor_type, sensor in self.sensors.items():
            thread = threading.Thread(
                target=self._sensor_acquisition_loop,
                args=(sensor_type, sensor),
                daemon=True
            )
            thread.start()
        
        # Start fusion thread
        self.fusion_thread = threading.Thread(
            target=self._fusion_loop,
            daemon=True
        )
        self.fusion_thread.start()
        
        logger.info("Sensor interface started")
    
    def stop(self):
        """Stop sensor data acquisition."""
        self.running = False
        logger.info("Sensor interface stopped")
    
    def _sensor_acquisition_loop(self, sensor_type: SensorType, sensor):
        """Thread function for sensor data acquisition."""
        while self.running:
            try:
                # Get sensor data
                data = sensor.get_data()
                if data is not None:
                    # Create sensor data container
                    sensor_data = SensorData(
                        sensor_type=sensor_type,
                        timestamp=time.time(),
                        data=data,
                        metadata=sensor.get_metadata(),
                        quality_score=sensor.get_quality_score()
                    )
                    
                    # Add to queue (non-blocking)
                    try:
                        self.data_queues[sensor_type].put_nowait(sensor_data)
                    except queue.Full:
                        # Remove oldest data if queue is full
                        try:
                            self.data_queues[sensor_type].get_nowait()
                            self.data_queues[sensor_type].put_nowait(sensor_data)
                        except queue.Empty:
                            pass
                
                # Sleep based on sensor update rate
                time.sleep(1.0 / sensor.get_update_rate())
                
            except Exception as e:
                logger.error(f"Error in sensor acquisition loop for {sensor_type}: {e}")
                time.sleep(0.1)
    
    def _fusion_loop(self):
        """Thread function for sensor fusion."""
        last_fusion_time = time.time()
        
        while self.running:
            current_time = time.time()
            
            # Check if it's time for fusion
            if current_time - last_fusion_time >= 1.0 / self.fusion_rate:
                try:
                    # Collect latest data from all sensors
                    fused_data = self._collect_sensor_data()
                    
                    # Perform sensor fusion
                    vehicle_state = self._perform_fusion(fused_data)
                    
                    # Store fused state
                    self._store_fused_state(vehicle_state)
                    
                    last_fusion_time = current_time
                    
                except Exception as e:
                    logger.error(f"Error in fusion loop: {e}")
            
            time.sleep(0.001)  # 1ms sleep
    
    def _collect_sensor_data(self) -> Dict[SensorType, SensorData]:
        """Collect latest data from all sensors."""
        fused_data = {}
        
        for sensor_type, data_queue in self.data_queues.items():
            try:
                # Get latest data (non-blocking)
                latest_data = None
                while not data_queue.empty():
                    latest_data = data_queue.get_nowait()
                
                if latest_data is not None:
                    # Check if data is fresh enough
                    if time.time() - latest_data.timestamp <= self.sensor_timeouts[sensor_type]:
                        fused_data[sensor_type] = latest_data
                    else:
                        logger.warning(f"Stale data from {sensor_type}")
                        
            except queue.Empty:
                pass
        
        return fused_data
    
    def _perform_fusion(self, sensor_data: Dict[SensorType, SensorData]) -> VehicleState:
        """
        Perform sensor fusion to create unified vehicle state.
        
        Args:
            sensor_data: Dictionary of sensor data by type
            
        Returns:
            Fused vehicle state
        """
        # Initialize with default values
        vehicle_state = VehicleState(
            timestamp=time.time(),
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([0.0, 0.0, 0.0]),
            acceleration=np.array([0.0, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, 0.0]),
            wheel_speeds=np.array([0.0, 0.0, 0.0, 0.0]),
            steering_angle=0.0,
            throttle_position=0.0,
            brake_pressure=0.0,
            gear_position=0,
            battery_voltage=12.0,
            system_temperature=25.0
        )
        
        # Fuse GPS data for position
        if SensorType.GPS in sensor_data:
            gps_data = sensor_data[SensorType.GPS]
            vehicle_state.position[:2] = gps_data.data[:2]  # lat, lon
        
        # Fuse IMU data for orientation and motion
        if SensorType.IMU in sensor_data:
            imu_data = sensor_data[SensorType.IMU]
            vehicle_state.orientation = imu_data.data[3:6]  # roll, pitch, yaw
            vehicle_state.angular_velocity = imu_data.data[6:9]  # wx, wy, wz
            vehicle_state.acceleration = imu_data.data[0:3]  # ax, ay, az
        
        # Fuse wheel encoder data for velocity
        if SensorType.WHEEL_ENCODER in sensor_data:
            wheel_data = sensor_data[SensorType.WHEEL_ENCODER]
            vehicle_state.wheel_speeds = wheel_data.data
            # Calculate vehicle velocity from wheel speeds
            vehicle_state.velocity[0] = np.mean(wheel_data.data)  # Forward velocity
        
        return vehicle_state
    
    def _store_fused_state(self, vehicle_state: VehicleState):
        """Store fused vehicle state for access by other components."""
        # In a real implementation, this would store to shared memory
        # or a thread-safe data structure
        self._latest_vehicle_state = vehicle_state
    
    def get_latest_state(self) -> Optional[VehicleState]:
        """Get the latest fused vehicle state."""
        return getattr(self, '_latest_vehicle_state', None)


class ActuatorInterface:
    """
    Interface for managing actuator control and safety.
    Handles steering, throttle, brake, and gear control with safety limits.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.actuators = {}
        self.safety_monitor = SafetyMonitor()
        self.command_queue = queue.Queue(maxsize=50)
        self.control_thread = None
        self.running = False
        
        # Control parameters
        self.control_rate = config.get('control_rate', 100)  # Hz
        self.max_steering_rate = config.get('max_steering_rate', 1.0)  # rad/s
        self.max_acceleration = config.get('max_acceleration', 3.0)  # m/s²
        self.max_deceleration = config.get('max_deceleration', -5.0)  # m/s²
        
        # Initialize actuator interfaces
        self._initialize_actuators()
    
    def _initialize_actuators(self):
        """Initialize actuator interfaces based on configuration."""
        for actuator_type, actuator_config in self.config.get('actuators', {}).items():
            actuator_enum = ActuatorType(actuator_type)
            
            if actuator_enum == ActuatorType.STEERING:
                self.actuators[actuator_enum] = SteeringInterface(actuator_config)
            elif actuator_enum == ActuatorType.THROTTLE:
                self.actuators[actuator_enum] = ThrottleInterface(actuator_config)
            elif actuator_enum == ActuatorType.BRAKE:
                self.actuators[actuator_enum] = BrakeInterface(actuator_config)
            elif actuator_enum == ActuatorType.GEAR:
                self.actuators[actuator_enum] = GearInterface(actuator_config)
    
    def start(self):
        """Start actuator control loop."""
        self.running = True
        
        # Start control thread
        self.control_thread = threading.Thread(
            target=self._control_loop,
            daemon=True
        )
        self.control_thread.start()
        
        logger.info("Actuator interface started")
    
    def stop(self):
        """Stop actuator control."""
        self.running = False
        logger.info("Actuator interface stopped")
    
    def send_command(self, command: ActuatorCommand):
        """
        Send actuator command with safety validation.
        
        Args:
            command: Actuator command to execute
        """
        # Validate command
        if not self._validate_command(command):
            logger.warning(f"Invalid command rejected: {command}")
            return
        
        # Add to command queue
        try:
            self.command_queue.put_nowait(command)
        except queue.Full:
            logger.warning("Command queue full, dropping command")
    
    def _validate_command(self, command: ActuatorCommand) -> bool:
        """Validate actuator command against safety limits."""
        # Check command bounds
        if command.command < command.min_limit or command.command > command.max_limit:
            return False
        
        # Check rate limits
        if not self._check_rate_limits(command):
            return False
        
        # Check safety constraints
        if not self.safety_monitor.check_command(command):
            return False
        
        return True
    
    def _check_rate_limits(self, command: ActuatorCommand) -> bool:
        """Check rate limits for actuator commands."""
        # Implementation would track previous commands and check rates
        # For now, return True
        return True
    
    def _control_loop(self):
        """Main control loop for actuator execution."""
        while self.running:
            try:
                # Get command from queue
                command = self.command_queue.get(timeout=0.01)
                
                # Execute command
                self._execute_command(command)
                
            except queue.Empty:
                # No commands, continue
                pass
            except Exception as e:
                logger.error(f"Error in control loop: {e}")
            
            time.sleep(1.0 / self.control_rate)
    
    def _execute_command(self, command: ActuatorCommand):
        """Execute actuator command."""
        if command.actuator_type in self.actuators:
            actuator = self.actuators[command.actuator_type]
            actuator.set_command(command.command)
        else:
            logger.error(f"Unknown actuator type: {command.actuator_type}")


class SafetyMonitor:
    """
    Safety monitoring system for vehicle control.
    Implements safety checks and emergency procedures.
    """
    
    def __init__(self):
        self.safety_limits = {
            'max_speed': 30.0,  # m/s
            'max_steering_angle': 0.5,  # rad
            'max_lateral_acceleration': 5.0,  # m/s²
            'min_distance_to_obstacle': 2.0,  # m
            'max_brake_pressure': 100.0,  # bar
        }
        
        self.emergency_states = {
            'emergency_brake': False,
            'steering_override': False,
            'system_shutdown': False
        }
    
    def check_command(self, command: ActuatorCommand) -> bool:
        """
        Check if command is safe to execute.
        
        Args:
            command: Actuator command to check
            
        Returns:
            True if command is safe, False otherwise
        """
        # Check steering limits
        if command.actuator_type == ActuatorType.STEERING:
            if abs(command.command) > self.safety_limits['max_steering_angle']:
                return False
        
        # Check brake limits
        elif command.actuator_type == ActuatorType.BRAKE:
            if command.command > self.safety_limits['max_brake_pressure']:
                return False
        
        # Additional safety checks would be implemented here
        return True
    
    def check_vehicle_state(self, vehicle_state: VehicleState) -> Dict[str, bool]:
        """
        Check vehicle state for safety violations.
        
        Args:
            vehicle_state: Current vehicle state
            
        Returns:
            Dictionary of safety violations
        """
        violations = {
            'speed_violation': False,
            'steering_violation': False,
            'acceleration_violation': False,
            'obstacle_violation': False
        }
        
        # Check speed
        speed = np.linalg.norm(vehicle_state.velocity)
        if speed > self.safety_limits['max_speed']:
            violations['speed_violation'] = True
        
        # Check steering angle
        if abs(vehicle_state.steering_angle) > self.safety_limits['max_steering_angle']:
            violations['steering_violation'] = True
        
        # Check lateral acceleration
        lateral_acc = vehicle_state.acceleration[1]  # ay
        if abs(lateral_acc) > self.safety_limits['max_lateral_acceleration']:
            violations['acceleration_violation'] = True
        
        return violations
    
    def trigger_emergency_brake(self):
        """Trigger emergency brake procedure."""
        self.emergency_states['emergency_brake'] = True
        logger.warning("Emergency brake triggered")
    
    def reset_emergency_states(self):
        """Reset emergency states."""
        for key in self.emergency_states:
            self.emergency_states[key] = False


# Sensor interface implementations (simplified)
class CameraInterface:
    def __init__(self, config):
        self.config = config
        self.update_rate = config.get('update_rate', 30)
    
    def get_data(self):
        # Simulated camera data
        return np.random.rand(480, 640, 3)
    
    def get_metadata(self):
        return {'resolution': (640, 480), 'fps': 30}
    
    def get_quality_score(self):
        return 0.95


class LidarInterface:
    def __init__(self, config):
        self.config = config
        self.update_rate = config.get('update_rate', 10)
    
    def get_data(self):
        # Simulated LiDAR point cloud
        return np.random.rand(1000, 3)
    
    def get_metadata(self):
        return {'points': 1000, 'range': 100}
    
    def get_quality_score(self):
        return 0.98


class RadarInterface:
    def __init__(self, config):
        self.config = config
        self.update_rate = config.get('update_rate', 20)
    
    def get_data(self):
        # Simulated radar detections
        return np.random.rand(10, 4)  # [range, azimuth, velocity, amplitude]
    
    def get_metadata(self):
        return {'detections': 10, 'range': 200}
    
    def get_quality_score(self):
        return 0.92


class IMUInterface:
    def __init__(self, config):
        self.config = config
        self.update_rate = config.get('update_rate', 100)
    
    def get_data(self):
        # Simulated IMU data [ax, ay, az, roll, pitch, yaw, wx, wy, wz]
        return np.random.randn(9)
    
    def get_metadata(self):
        return {'accelerometer_range': '±2g', 'gyroscope_range': '±250°/s'}
    
    def get_quality_score(self):
        return 0.99


class GPSInterface:
    def __init__(self, config):
        self.config = config
        self.update_rate = config.get('update_rate', 10)
    
    def get_data(self):
        # Simulated GPS data [lat, lon, alt, accuracy]
        return np.array([37.7749, -122.4194, 10.0, 2.0])
    
    def get_metadata(self):
        return {'satellites': 8, 'hdop': 1.2}
    
    def get_quality_score(self):
        return 0.85


class WheelEncoderInterface:
    def __init__(self, config):
        self.config = config
        self.update_rate = config.get('update_rate', 100)
    
    def get_data(self):
        # Simulated wheel speeds [fl, fr, rl, rr]
        return np.random.rand(4) * 10  # m/s
    
    def get_metadata(self):
        return {'resolution': 100, 'wheels': 4}
    
    def get_quality_score(self):
        return 0.97


# Actuator interface implementations (simplified)
class SteeringInterface:
    def __init__(self, config):
        self.config = config
        self.current_angle = 0.0
    
    def set_command(self, angle):
        self.current_angle = np.clip(angle, -0.5, 0.5)


class ThrottleInterface:
    def __init__(self, config):
        self.config = config
        self.current_position = 0.0
    
    def set_command(self, position):
        self.current_position = np.clip(position, 0.0, 1.0)


class BrakeInterface:
    def __init__(self, config):
        self.config = config
        self.current_pressure = 0.0
    
    def set_command(self, pressure):
        self.current_pressure = np.clip(pressure, 0.0, 100.0)


class GearInterface:
    def __init__(self, config):
        self.config = config
        self.current_gear = 0
    
    def set_command(self, gear):
        self.current_gear = int(np.clip(gear, -1, 6))


class VehicleInterface:
    """
    Main vehicle interface that integrates sensors, actuators, and safety monitoring.
    Provides a unified API for AI system integration.
    """
    
    def __init__(self, sensor_config: Dict[str, Any], actuator_config: Dict[str, Any], device: str = "cpu"):
        """
        Initialize vehicle interface with sensor and actuator configurations.
        
        Args:
            sensor_config: Configuration for sensors
            actuator_config: Configuration for actuators
            device: Device for tensor operations ("cpu" or "cuda")
        """
        self.device = device
        self.sensor_config = sensor_config
        self.actuator_config = actuator_config
        
        # Initialize components
        self.sensor_interface = SensorInterface(sensor_config)
        self.actuator_interface = ActuatorInterface(actuator_config)
        self.safety_monitor = SafetyMonitor()
        
        # State tracking
        self.latest_vehicle_state = None
        self.last_control_commands = None
        
        logger.info(f"VehicleInterface initialized with device: {device}")
    
    def start(self):
        """Start all interface components."""
        self.sensor_interface.start()
        self.actuator_interface.start()
        logger.info("VehicleInterface started")
    
    def stop(self):
        """Stop all interface components."""
        self.sensor_interface.stop()
        self.actuator_interface.stop()
        logger.info("VehicleInterface stopped")
    
    def fuse_sensor_data(self, camera_data: torch.Tensor, lidar_data: torch.Tensor, radar_data: torch.Tensor) -> torch.Tensor:
        """
        Fuse multi-modal sensor data into a unified representation.
        
        Args:
            camera_data: Camera image tensor [B, C, H, W]
            lidar_data: LiDAR point cloud tensor [B, N, 3]
            radar_data: Radar detection tensor [B, M, 4]
            
        Returns:
            Fused sensor data tensor
        """
        # Ensure tensors are on the correct device
        camera_data = camera_data.to(self.device)
        lidar_data = lidar_data.to(self.device)
        radar_data = radar_data.to(self.device)
        
        batch_size = camera_data.shape[0]
        
        # Simple fusion approach: flatten and concatenate features
        # In practice, this would involve more sophisticated fusion algorithms
        
        # Process camera data (simple feature extraction)
        camera_features = torch.mean(camera_data, dim=[2, 3])  # [B, C]
        
        # Process LiDAR data (simple point cloud features)
        lidar_features = torch.mean(lidar_data, dim=1)  # [B, 3]
        
        # Process radar data (simple detection features)
        radar_features = torch.mean(radar_data, dim=1)  # [B, 4]
        
        # Concatenate all features
        fused_features = torch.cat([camera_features, lidar_features, radar_features], dim=1)
        
        return fused_features
    
    def apply_control(self, control_commands: torch.Tensor) -> bool:
        """
        Apply control commands to vehicle actuators.
        
        Args:
            control_commands: Control tensor [B, 3] containing [steering, throttle, brake]
            
        Returns:
            True if control was applied successfully, False otherwise
        """
        try:
            # Ensure tensor is on CPU for actuator commands
            control_commands = control_commands.cpu()
            
            # Extract commands for the first batch item
            if control_commands.dim() == 2:
                commands = control_commands[0]
            else:
                commands = control_commands
            
            steering_angle = float(commands[0])
            throttle_position = float(commands[1])
            brake_pressure = float(commands[2])
            
            # Create actuator commands
            current_time = time.time()
            
            steering_cmd = ActuatorCommand(
                actuator_type=ActuatorType.STEERING,
                timestamp=current_time,
                command=steering_angle,
                min_limit=-0.5,
                max_limit=0.5
            )
            
            throttle_cmd = ActuatorCommand(
                actuator_type=ActuatorType.THROTTLE,
                timestamp=current_time,
                command=throttle_position,
                min_limit=0.0,
                max_limit=1.0
            )
            
            brake_cmd = ActuatorCommand(
                actuator_type=ActuatorType.BRAKE,
                timestamp=current_time,
                command=brake_pressure,
                min_limit=0.0,
                max_limit=100.0
            )
            
            # Check safety before applying commands
            if not self.safety_monitor.check_command(steering_cmd):
                logger.warning("Steering command failed safety check")
                return False
            
            if not self.safety_monitor.check_command(throttle_cmd):
                logger.warning("Throttle command failed safety check")
                return False
            
            if not self.safety_monitor.check_command(brake_cmd):
                logger.warning("Brake command failed safety check")
                return False
            
            # Apply commands
            self.actuator_interface.send_command(steering_cmd)
            self.actuator_interface.send_command(throttle_cmd)
            self.actuator_interface.send_command(brake_cmd)
            
            # Store last commands for monitoring
            self.last_control_commands = control_commands
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying control commands: {e}")
            return False
    
    def check_safety_status(self) -> Dict[str, Any]:
        """
        Check current safety status of the vehicle.
        
        Returns:
            Dictionary containing safety status information
        """
        safety_status = {
            'is_safe': True,
            'violations': {},
            'emergency_states': self.safety_monitor.emergency_states.copy(),
            'last_update': time.time()
        }
        
        # Check vehicle state if available
        if self.latest_vehicle_state is not None:
            violations = self.safety_monitor.check_vehicle_state(self.latest_vehicle_state)
            safety_status['violations'] = violations
            
            # Overall safety is false if any violation is detected
            safety_status['is_safe'] = not any(violations.values())
        
        return safety_status
    
    def get_vehicle_state(self) -> Optional[VehicleState]:
        """
        Get the latest vehicle state from sensor fusion.
        
        Returns:
            Latest vehicle state or None if not available
        """
        return self.sensor_interface.get_latest_state()
    
    def update_vehicle_state(self):
        """Update the latest vehicle state from sensor interface."""
        self.latest_vehicle_state = self.sensor_interface.get_latest_state()
    
    def get_sensor_data(self) -> Dict[str, Any]:
        """
        Get raw sensor data for debugging/monitoring.
        
        Returns:
            Dictionary of sensor data
        """
        sensor_data = {}
        
        # Collect data from all sensors
        for sensor_type, sensor in self.sensor_interface.sensors.items():
            try:
                data = sensor.get_data()
                metadata = sensor.get_metadata()
                quality = sensor.get_quality_score()
                
                sensor_data[sensor_type.value] = {
                    'data': data,
                    'metadata': metadata,
                    'quality_score': quality,
                    'timestamp': time.time()
                }
            except Exception as e:
                logger.warning(f"Error getting data from {sensor_type}: {e}")
                sensor_data[sensor_type.value] = None
        
        return sensor_data
    
    def get_actuator_status(self) -> Dict[str, Any]:
        """
        Get current actuator status.
        
        Returns:
            Dictionary of actuator status
        """
        actuator_status = {}
        
        # Get status from all actuators
        for actuator_type, actuator in self.actuator_interface.actuators.items():
            try:
                if actuator_type == ActuatorType.STEERING:
                    status = {'angle': actuator.current_angle}
                elif actuator_type == ActuatorType.THROTTLE:
                    status = {'position': actuator.current_position}
                elif actuator_type == ActuatorType.BRAKE:
                    status = {'pressure': actuator.current_pressure}
                elif actuator_type == ActuatorType.GEAR:
                    status = {'gear': actuator.current_gear}
                else:
                    status = {'status': 'unknown'}
                
                actuator_status[actuator_type.value] = {
                    'status': status,
                    'timestamp': time.time()
                }
            except Exception as e:
                logger.warning(f"Error getting status from {actuator_type}: {e}")
                actuator_status[actuator_type.value] = None
        
        return actuator_status 