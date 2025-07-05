"""
AI-Vehicle Integration Module

This module integrates the causal world model and modular neural components
with the physical vehicle interface, managing real-time control, safety,
and performance monitoring.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
import logging
import threading
from dataclasses import dataclass
from queue import Queue, Empty
import copy

from .vehicle_interface import VehicleInterface
from ..causal_world_model import CausalWorldModel, WorldState
from ..modular_components import InterpretableDrivingSystem

logger = logging.getLogger(__name__)


def validate_tensor(tensor: torch.Tensor, name: str, expected_shape: Optional[Tuple] = None, 
                   expected_dtype: Optional[torch.dtype] = None, allow_nan: bool = False) -> torch.Tensor:
    """
    Validate tensor properties and return validated tensor.
    
    Args:
        tensor: Tensor to validate
        name: Name for error messages
        expected_shape: Expected shape (None for any shape)
        expected_dtype: Expected dtype (None for any dtype)
        allow_nan: Whether to allow NaN values
        
    Returns:
        Validated tensor
        
    Raises:
        ValueError: If validation fails
    """
    if tensor is None:
        raise ValueError(f"{name} cannot be None")
    
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    
    if expected_shape is not None and tensor.shape != expected_shape:
        raise ValueError(f"{name} expected shape {expected_shape}, got {tensor.shape}")
    
    if expected_dtype is not None and tensor.dtype != expected_dtype:
        raise ValueError(f"{name} expected dtype {expected_dtype}, got {tensor.dtype}")
    
    if not allow_nan and torch.isnan(tensor).any():
        raise ValueError(f"{name} contains NaN values")
    
    if torch.isinf(tensor).any():
        raise ValueError(f"{name} contains infinite values")
    
    return tensor


@dataclass
class ControlCommand:
    """Thread-safe control command structure."""
    steering: float
    throttle: float
    brake: float
    timestamp: float
    priority: int = 0  # Higher priority commands override lower ones
    
    def __post_init__(self):
        """Validate control command values."""
        if not -1 <= self.steering <= 1:
            raise ValueError("steering must be between -1 and 1")
        if not 0 <= self.throttle <= 1:
            raise ValueError("throttle must be between 0 and 1")
        if not 0 <= self.brake <= 1:
            raise ValueError("brake must be between 0 and 1")
        if self.timestamp < 0:
            raise ValueError("timestamp must be non-negative")
        if self.priority < 0:
            raise ValueError("priority must be non-negative")


@dataclass
class SafetyStatus:
    """Thread-safe safety status structure."""
    is_safe: bool
    violations: List[str]
    warnings: List[str]
    timestamp: float
    emergency_level: int = 0  # 0: normal, 1: warning, 2: critical, 3: emergency
    
    def __post_init__(self):
        """Validate safety status values."""
        if not isinstance(self.is_safe, bool):
            raise ValueError("is_safe must be a boolean")
        if not isinstance(self.violations, list):
            raise ValueError("violations must be a list")
        if not isinstance(self.warnings, list):
            raise ValueError("warnings must be a list")
        if self.timestamp < 0:
            raise ValueError("timestamp must be non-negative")
        if not 0 <= self.emergency_level <= 3:
            raise ValueError("emergency_level must be between 0 and 3")


class ThreadSafeBuffer:
    """
    Thread-safe buffer for storing sensor data and control commands.
    """
    
    def __init__(self, max_size: int = 100):
        # Validate parameters
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        
        self.max_size = max_size
        self._lock = threading.RLock()
        self._data = {}
        self._timestamps = {}
    
    def put(self, key: str, value: Any, timestamp: float = None):
        """
        Put data into the buffer with thread safety.
        
        Args:
            key: Data key
            value: Data value
            timestamp: Optional timestamp
        """
        if not isinstance(key, str):
            raise ValueError("key must be a string")
        
        if timestamp is None:
            timestamp = time.time()
        
        with self._lock:
            self._data[key] = value
            self._timestamps[key] = timestamp
            
            # Remove oldest data if buffer is full
            if len(self._data) > self.max_size:
                oldest_key = min(self._timestamps.keys(), key=lambda k: self._timestamps[k])
                del self._data[oldest_key]
                del self._timestamps[oldest_key]
    
    def get(self, key: str, default: Any = None) -> Tuple[Any, float]:
        """
        Get data from the buffer with thread safety.
        
        Args:
            key: Data key
            default: Default value if key not found
            
        Returns:
            Tuple of (value, timestamp)
        """
        if not isinstance(key, str):
            raise ValueError("key must be a string")
        
        with self._lock:
            if key in self._data:
                return self._data[key], self._timestamps[key]
            else:
                return default, 0.0
    
    def get_all(self) -> Dict[str, Tuple[Any, float]]:
        """
        Get all data from the buffer with thread safety.
        
        Returns:
            Dictionary of {key: (value, timestamp)}
        """
        with self._lock:
            return {key: (self._data[key], self._timestamps[key]) 
                   for key in self._data.keys()}
    
    def clear(self):
        """Clear all data from the buffer with thread safety."""
        with self._lock:
            self._data.clear()
            self._timestamps.clear()
    
    def size(self) -> int:
        """Get current buffer size with thread safety."""
        with self._lock:
            return len(self._data)


class AIVehicleIntegration:
    """
    Thread-safe AI-vehicle integration system.
    
    This class manages the integration between the causal world model,
    modular neural components, and the physical vehicle interface.
    """
    
    def __init__(
        self,
        causal_world_model: CausalWorldModel,
        interpretable_system: InterpretableDrivingSystem,
        vehicle_interface: VehicleInterface,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        control_frequency: float = 20.0,  # Hz
        safety_check_frequency: float = 10.0,  # Hz
        max_control_latency: float = 0.1  # seconds
    ):
        # Validate inputs
        if not isinstance(causal_world_model, CausalWorldModel):
            raise ValueError("causal_world_model must be a CausalWorldModel")
        if not isinstance(interpretable_system, InterpretableDrivingSystem):
            raise ValueError("interpretable_system must be an InterpretableDrivingSystem")
        if not isinstance(vehicle_interface, VehicleInterface):
            raise ValueError("vehicle_interface must be a VehicleInterface")
        if control_frequency <= 0:
            raise ValueError("control_frequency must be positive")
        if safety_check_frequency <= 0:
            raise ValueError("safety_check_frequency must be positive")
        if max_control_latency <= 0:
            raise ValueError("max_control_latency must be positive")
        
        self.causal_world_model = causal_world_model
        self.interpretable_system = interpretable_system
        self.vehicle_interface = vehicle_interface
        self.device = device
        self.control_frequency = control_frequency
        self.safety_check_frequency = safety_check_frequency
        self.max_control_latency = max_control_latency
        
        # Thread-safe buffers
        self.sensor_buffer = ThreadSafeBuffer(max_size=50)
        self.control_buffer = ThreadSafeBuffer(max_size=20)
        self.safety_buffer = ThreadSafeBuffer(max_size=10)
        
        # Thread synchronization
        self._lock = threading.RLock()
        self._control_lock = threading.Lock()
        self._safety_lock = threading.Lock()
        
        # Control command queue (thread-safe)
        self.control_queue = Queue(maxsize=100)
        
        # Threading state
        self._running = False
        self._control_thread = None
        self._safety_thread = None
        
        # Performance monitoring
        self.performance_metrics = {
            'total_processing_time': 0.0,
            'num_processed_frames': 0,
            'avg_processing_time': 0.0,
            'max_processing_time': 0.0,
            'min_processing_time': float('inf'),
            'safety_violations': 0,
            'emergency_stops': 0
        }
        
        # Thread-safe performance metrics
        self._metrics_lock = threading.Lock()
        
        logger.info("AI-Vehicle Integration initialized successfully")
    
    def start(self):
        """Start the AI-vehicle integration system."""
        with self._lock:
            if self._running:
                logger.warning("AI-Vehicle Integration is already running")
                return
            
            self._running = True
            
            # Start control thread
            self._control_thread = threading.Thread(
                target=self._control_loop,
                name="ControlThread",
                daemon=True
            )
            self._control_thread.start()
            
            # Start safety monitoring thread
            self._safety_thread = threading.Thread(
                target=self._safety_monitoring_loop,
                name="SafetyThread",
                daemon=True
            )
            self._safety_thread.start()
            
            logger.info("AI-Vehicle Integration started")
    
    def stop(self):
        """Stop the AI-vehicle integration system."""
        with self._lock:
            if not self._running:
                logger.warning("AI-Vehicle Integration is not running")
                return
            
            self._running = False
            
            # Wait for threads to finish
            if self._control_thread and self._control_thread.is_alive():
                self._control_thread.join(timeout=5.0)
            
            if self._safety_thread and self._safety_thread.is_alive():
                self._safety_thread.join(timeout=5.0)
            
            # Clear buffers
            self.sensor_buffer.clear()
            self.control_buffer.clear()
            self.safety_buffer.clear()
            
            # Clear control queue
            while not self.control_queue.empty():
                try:
                    self.control_queue.get_nowait()
                except Empty:
                    break
            
            logger.info("AI-Vehicle Integration stopped")
    
    def process_sensor_data(self, sensor_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Process sensor data and generate control commands.
        
        Args:
            sensor_data: Dictionary of sensor data
            
        Returns:
            Dictionary containing control commands, safety status, and interpretability info
        """
        # Validate input
        if not isinstance(sensor_data, dict):
            raise ValueError("sensor_data must be a dictionary")
        
        start_time = time.time()
        
        try:
            # Store sensor data in thread-safe buffer
            self.sensor_buffer.put('latest', sensor_data, time.time())
            
            # Fuse sensor data
            fused_data = self.vehicle_interface.fuse_sensor_data(**sensor_data)
            
            if fused_data is None:
                logger.warning("Sensor fusion failed, using fallback control")
                return self._generate_fallback_control()
            
            # Validate fused data
            fused_data = validate_tensor(fused_data, "fused_data", expected_dtype=torch.float32)
            
            # Process through interpretable system
            with self._control_lock:
                system_outputs = self.interpretable_system(fused_data)
            
            # Extract control commands
            control_signal = system_outputs['control_signal']
            interpretability_info = system_outputs['interpretability_info']
            
            # Validate control signal
            control_signal = validate_tensor(control_signal, "control_signal", expected_dtype=torch.float32)
            
            # Convert to control command
            control_command = ControlCommand(
                steering=float(control_signal[0, 0].item()),
                throttle=float(control_signal[0, 1].item()),
                brake=float(control_signal[0, 2].item()),
                timestamp=time.time(),
                priority=0
            )
            
            # Store control command in thread-safe buffer
            self.control_buffer.put('latest', control_command, control_command.timestamp)
            
            # Add to control queue
            try:
                self.control_queue.put_nowait(control_command)
            except:
                logger.warning("Control queue full, dropping command")
            
            # Check safety status
            safety_status = self._check_safety_status(sensor_data, control_command)
            
            # Store safety status in thread-safe buffer
            self.safety_buffer.put('latest', safety_status, safety_status.timestamp)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self._update_performance_metrics(processing_time)
            
            # Return results
            return {
                'control_commands': torch.tensor([
                    [control_command.steering, control_command.throttle, control_command.brake]
                ], device=self.device),
                'safety_status': {
                    'is_safe': safety_status.is_safe,
                    'violations': safety_status.violations,
                    'warnings': safety_status.warnings,
                    'emergency_level': safety_status.emergency_level
                },
                'interpretability_info': interpretability_info,
                'performance_metrics': self._get_performance_metrics()
            }
            
        except Exception as e:
            logger.error(f"Error processing sensor data: {e}")
            return self._generate_fallback_control()
    
    def _control_loop(self):
        """Main control loop running in separate thread."""
        control_interval = 1.0 / self.control_frequency
        
        while self._running:
            try:
                start_time = time.time()
                
                # Get latest control command from queue
                try:
                    control_command = self.control_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                # Check if command is too old
                if time.time() - control_command.timestamp > self.max_control_latency:
                    logger.warning("Control command too old, skipping")
                    continue
                
                # Apply control with thread safety
                with self._control_lock:
                    success = self.vehicle_interface.apply_control(
                        torch.tensor([[control_command.steering, control_command.throttle, control_command.brake]])
                    )
                
                if not success:
                    logger.error("Failed to apply control command")
                
                # Sleep to maintain control frequency
                elapsed_time = time.time() - start_time
                sleep_time = max(0, control_interval - elapsed_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                logger.error(f"Error in control loop: {e}")
                time.sleep(0.1)
    
    def _safety_monitoring_loop(self):
        """Safety monitoring loop running in separate thread."""
        safety_interval = 1.0 / self.safety_check_frequency
        
        while self._running:
            try:
                start_time = time.time()
                
                # Get latest sensor data
                sensor_data, _ = self.sensor_buffer.get('latest')
                if sensor_data is None:
                    time.sleep(safety_interval)
                    continue
                
                # Get latest control command
                control_command, _ = self.control_buffer.get('latest')
                
                # Check safety status
                safety_status = self._check_safety_status(sensor_data, control_command)
                
                # Store safety status
                self.safety_buffer.put('latest', safety_status, safety_status.timestamp)
                
                # Handle emergency situations
                if safety_status.emergency_level >= 3:
                    self._handle_emergency()
                
                # Sleep to maintain safety check frequency
                elapsed_time = time.time() - start_time
                sleep_time = max(0, safety_interval - elapsed_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                logger.error(f"Error in safety monitoring loop: {e}")
                time.sleep(0.1)
    
    def _check_safety_status(self, sensor_data: Dict[str, torch.Tensor], 
                           control_command: Optional[ControlCommand]) -> SafetyStatus:
        """
        Check safety status with thread safety.
        
        Args:
            sensor_data: Current sensor data
            control_command: Current control command
            
        Returns:
            Safety status
        """
        with self._safety_lock:
            violations = []
            warnings = []
            emergency_level = 0
            
            # Check vehicle interface safety
            vehicle_safety = self.vehicle_interface.check_safety_status()
            
            if not vehicle_safety.get('is_safe', True):
                violations.append("Vehicle interface safety violation")
                emergency_level = max(emergency_level, 2)
            
            # Check sensor data validity
            for sensor_name, data in sensor_data.items():
                if data is None:
                    warnings.append(f"Missing {sensor_name} data")
                    emergency_level = max(emergency_level, 1)
                elif torch.isnan(data).any():
                    violations.append(f"Invalid {sensor_name} data (NaN values)")
                    emergency_level = max(emergency_level, 2)
                elif torch.isinf(data).any():
                    violations.append(f"Invalid {sensor_name} data (infinite values)")
                    emergency_level = max(emergency_level, 2)
            
            # Check control command validity
            if control_command is not None:
                # Check for extreme control values
                if abs(control_command.steering) > 0.9:
                    warnings.append("Extreme steering command")
                    emergency_level = max(emergency_level, 1)
                
                if control_command.throttle > 0.9 and control_command.brake > 0.1:
                    violations.append("Conflicting throttle and brake commands")
                    emergency_level = max(emergency_level, 2)
            
            # Determine overall safety
            is_safe = len(violations) == 0 and emergency_level < 3
            
            return SafetyStatus(
                is_safe=is_safe,
                violations=violations,
                warnings=warnings,
                timestamp=time.time(),
                emergency_level=emergency_level
            )
    
    def _handle_emergency(self):
        """Handle emergency situations with thread safety."""
        with self._control_lock:
            # Apply emergency stop
            emergency_command = ControlCommand(
                steering=0.0,
                throttle=0.0,
                brake=1.0,
                timestamp=time.time(),
                priority=3
            )
            
            # Clear control queue and add emergency command
            while not self.control_queue.empty():
                try:
                    self.control_queue.get_nowait()
                except Empty:
                    break
            
            try:
                self.control_queue.put_nowait(emergency_command)
            except:
                pass
            
            # Apply emergency stop immediately
            self.vehicle_interface.apply_control(
                torch.tensor([[0.0, 0.0, 1.0]])
            )
            
            # Update performance metrics
            with self._metrics_lock:
                self.performance_metrics['emergency_stops'] += 1
            
            logger.warning("Emergency stop applied")
    
    def _generate_fallback_control(self) -> Dict[str, Any]:
        """Generate fallback control when processing fails."""
        fallback_command = ControlCommand(
            steering=0.0,
            throttle=0.0,
            brake=0.5,  # Gentle braking
            timestamp=time.time(),
            priority=1
        )
        
        return {
            'control_commands': torch.tensor([[0.0, 0.0, 0.5]], device=self.device),
            'safety_status': {
                'is_safe': False,
                'violations': ['Processing failure - using fallback control'],
                'warnings': ['System degraded'],
                'emergency_level': 1
            },
            'interpretability_info': {},
            'performance_metrics': self._get_performance_metrics()
        }
    
    def _update_performance_metrics(self, processing_time: float):
        """Update performance metrics with thread safety."""
        with self._metrics_lock:
            self.performance_metrics['total_processing_time'] += processing_time
            self.performance_metrics['num_processed_frames'] += 1
            self.performance_metrics['avg_processing_time'] = (
                self.performance_metrics['total_processing_time'] / 
                self.performance_metrics['num_processed_frames']
            )
            self.performance_metrics['max_processing_time'] = max(
                self.performance_metrics['max_processing_time'], 
                processing_time
            )
            self.performance_metrics['min_processing_time'] = min(
                self.performance_metrics['min_processing_time'], 
                processing_time
            )
    
    def _get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics with thread safety."""
        with self._metrics_lock:
            return copy.deepcopy(self.performance_metrics)
    
    def monitor_safety(self) -> Dict[str, Any]:
        """Monitor current safety status."""
        safety_status, timestamp = self.safety_buffer.get('latest')
        
        if safety_status is None:
            return {
                'is_safe': False,
                'violations': ['No safety data available'],
                'warnings': ['Safety monitoring not initialized'],
                'emergency_level': 1
            }
        
        return {
            'is_safe': safety_status.is_safe,
            'violations': safety_status.violations,
            'warnings': safety_status.warnings,
            'emergency_level': safety_status.emergency_level,
            'timestamp': timestamp
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        with self._lock:
            return {
                'running': self._running,
                'control_thread_alive': self._control_thread.is_alive() if self._control_thread else False,
                'safety_thread_alive': self._safety_thread.is_alive() if self._safety_thread else False,
                'sensor_buffer_size': self.sensor_buffer.size(),
                'control_buffer_size': self.control_buffer.size(),
                'safety_buffer_size': self.safety_buffer.size(),
                'control_queue_size': self.control_queue.qsize(),
                'performance_metrics': self._get_performance_metrics(),
                'safety_status': self.monitor_safety()
            }
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop() 