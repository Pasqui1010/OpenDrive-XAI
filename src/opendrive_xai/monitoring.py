"""
Monitoring and logging system for OpenDrive-XAI.

This module provides comprehensive monitoring capabilities including:
- Performance metrics collection and analysis
- Safety monitoring and alerting
- Real-time system health checks
- Logging with structured data
- Performance benchmarking
"""

import time
import logging
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
import json
from pathlib import Path
import torch
import numpy as np
from datetime import datetime, timedelta

from .config import get_config, get_safety_config, get_system_config


@dataclass
class PerformanceMetrics:
    """Performance metrics for system components."""
    
    # Timing metrics
    inference_time: float = 0.0
    preprocessing_time: float = 0.0
    postprocessing_time: float = 0.0
    total_latency: float = 0.0
    
    # Memory metrics
    memory_usage: float = 0.0
    gpu_memory_usage: float = 0.0
    
    # Throughput metrics
    frames_per_second: float = 0.0
    samples_per_second: float = 0.0
    
    # Quality metrics
    prediction_confidence: float = 0.0
    safety_score: float = 0.0
    
    # Error metrics
    error_count: int = 0
    warning_count: int = 0
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SafetyMetrics:
    """Safety monitoring metrics."""
    
    # Risk assessments
    collision_risk: float = 0.0
    speed_violation_risk: float = 0.0
    lane_deviation_risk: float = 0.0
    traffic_violation_risk: float = 0.0
    
    # System health
    sensor_health: Dict[str, float] = field(default_factory=dict)
    actuator_health: Dict[str, float] = field(default_factory=dict)
    model_health: float = 1.0
    
    # Safety violations
    safety_violations: List[str] = field(default_factory=list)
    emergency_actions: List[str] = field(default_factory=list)
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize performance monitor.
        
        Args:
            max_history: Maximum number of metrics to keep in history
        """
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.component_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.lock = threading.Lock()
        
        # Performance thresholds
        self.latency_threshold = 0.1  # 100ms
        self.memory_threshold = 0.8   # 80% of available memory
        self.fps_threshold = 10.0     # Minimum 10 FPS
        
        # Callbacks for alerts
        self.alert_callbacks: List[Callable] = []
    
    def start_timing(self, component: str = "main") -> float:
        """Start timing for a component."""
        return time.time()
    
    def end_timing(self, start_time: float, component: str = "main") -> float:
        """End timing and record metrics."""
        elapsed = time.time() - start_time
        
        with self.lock:
            metrics = PerformanceMetrics()
            metrics.inference_time = elapsed
            metrics.total_latency = elapsed
            metrics.timestamp = datetime.now()
            
            # Record component-specific metrics
            self.component_metrics[component].append(metrics)
            
            # Check for performance issues
            self._check_performance_issues(metrics, component)
        
        return elapsed
    
    def record_metrics(self, metrics: PerformanceMetrics, component: str = "main") -> None:
        """Record performance metrics."""
        with self.lock:
            self.metrics_history.append(metrics)
            self.component_metrics[component].append(metrics)
            
            # Check for performance issues
            self._check_performance_issues(metrics, component)
    
    def _check_performance_issues(self, metrics: PerformanceMetrics, component: str) -> None:
        """Check for performance issues and trigger alerts."""
        issues = []
        
        if metrics.total_latency > self.latency_threshold:
            issues.append(f"High latency: {metrics.total_latency:.3f}s")
        
        if metrics.memory_usage > self.memory_threshold:
            issues.append(f"High memory usage: {metrics.memory_usage:.1%}")
        
        if metrics.frames_per_second < self.fps_threshold:
            issues.append(f"Low FPS: {metrics.frames_per_second:.1f}")
        
        if issues:
            self._trigger_alerts(component, issues)
    
    def _trigger_alerts(self, component: str, issues: List[str]) -> None:
        """Trigger performance alerts."""
        alert = {
            "type": "performance",
            "component": component,
            "issues": issues,
            "timestamp": datetime.now().isoformat()
        }
        
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logging.error(f"Error in alert callback: {e}")
    
    def get_recent_metrics(self, component: str = "main", count: int = 100) -> List[PerformanceMetrics]:
        """Get recent metrics for a component."""
        with self.lock:
            return list(self.component_metrics[component])[-count:]
    
    def get_average_metrics(self, component: str = "main", window: int = 100) -> PerformanceMetrics:
        """Get average metrics over a window."""
        recent = self.get_recent_metrics(component, window)
        if not recent:
            return PerformanceMetrics()
        
        avg_metrics = PerformanceMetrics()
        avg_metrics.inference_time = np.mean([m.inference_time for m in recent])
        avg_metrics.total_latency = np.mean([m.total_latency for m in recent])
        avg_metrics.frames_per_second = np.mean([m.frames_per_second for m in recent])
        avg_metrics.memory_usage = np.mean([m.memory_usage for m in recent])
        avg_metrics.timestamp = datetime.now()
        
        return avg_metrics
    
    def add_alert_callback(self, callback: Callable) -> None:
        """Add a callback for performance alerts."""
        self.alert_callbacks.append(callback)
    
    def clear_history(self) -> None:
        """Clear all metrics history."""
        with self.lock:
            self.metrics_history.clear()
            for component_metrics in self.component_metrics.values():
                component_metrics.clear()


class SafetyMonitor:
    """Monitor safety metrics and trigger alerts."""
    
    def __init__(self):
        """Initialize safety monitor."""
        self.safety_history: deque = deque(maxlen=1000)
        self.lock = threading.Lock()
        
        # Load safety thresholds from config
        safety_config = get_safety_config()
        self.thresholds = safety_config['thresholds']
        self.emergency_actions = safety_config['emergency_actions']
        
        # Alert callbacks
        self.alert_callbacks: List[Callable] = []
        self.emergency_callbacks: List[Callable] = []
    
    def record_safety_metrics(self, metrics: SafetyMetrics) -> None:
        """Record safety metrics."""
        with self.lock:
            self.safety_history.append(metrics)
            
            # Check for safety violations
            violations = self._check_safety_violations(metrics)
            if violations:
                self._trigger_safety_alerts(violations, metrics)
    
    def _check_safety_violations(self, metrics: SafetyMetrics) -> List[str]:
        """Check for safety violations."""
        violations = []
        
        if metrics.collision_risk > self.thresholds['collision_risk']:
            violations.append(f"High collision risk: {metrics.collision_risk:.3f}")
        
        if metrics.speed_violation_risk > self.thresholds['speed_violation']:
            violations.append(f"Speed violation risk: {metrics.speed_violation_risk:.3f}")
        
        if metrics.lane_deviation_risk > self.thresholds['lane_deviation']:
            violations.append(f"Lane deviation risk: {metrics.lane_deviation_risk:.3f}")
        
        if metrics.traffic_violation_risk > self.thresholds['traffic_violation']:
            violations.append(f"Traffic violation risk: {metrics.traffic_violation_risk:.3f}")
        
        if metrics.model_health < 0.5:
            violations.append(f"Low model health: {metrics.model_health:.3f}")
        
        return violations
    
    def _trigger_safety_alerts(self, violations: List[str], metrics: SafetyMetrics) -> None:
        """Trigger safety alerts."""
        alert = {
            "type": "safety",
            "violations": violations,
            "metrics": {
                "collision_risk": metrics.collision_risk,
                "speed_violation_risk": metrics.speed_violation_risk,
                "lane_deviation_risk": metrics.lane_deviation_risk,
                "traffic_violation_risk": metrics.traffic_violation_risk,
                "model_health": metrics.model_health
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Check for emergency conditions
        emergency_conditions = [
            "High collision risk" in violations,
            "Low model health" in violations
        ]
        
        if any(emergency_conditions):
            self._trigger_emergency_alerts(alert)
        else:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logging.error(f"Error in safety alert callback: {e}")
    
    def _trigger_emergency_alerts(self, alert: Dict[str, Any]) -> None:
        """Trigger emergency alerts."""
        for callback in self.emergency_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logging.error(f"Error in emergency callback: {e}")
    
    def add_alert_callback(self, callback: Callable) -> None:
        """Add a callback for safety alerts."""
        self.alert_callbacks.append(callback)
    
    def add_emergency_callback(self, callback: Callable) -> None:
        """Add a callback for emergency alerts."""
        self.emergency_callbacks.append(callback)
    
    def get_recent_safety_metrics(self, count: int = 100) -> List[SafetyMetrics]:
        """Get recent safety metrics."""
        with self.lock:
            return list(self.safety_history)[-count:]
    
    def get_safety_summary(self, window: int = 100) -> Dict[str, Any]:
        """Get safety summary over a window."""
        recent = self.get_recent_safety_metrics(window)
        if not recent:
            return {}
        
        return {
            "avg_collision_risk": np.mean([m.collision_risk for m in recent]),
            "avg_speed_violation_risk": np.mean([m.speed_violation_risk for m in recent]),
            "avg_lane_deviation_risk": np.mean([m.lane_deviation_risk for m in recent]),
            "avg_model_health": np.mean([m.model_health for m in recent]),
            "total_violations": sum(len(m.safety_violations) for m in recent),
            "total_emergency_actions": sum(len(m.emergency_actions) for m in recent)
        }


class SystemHealthMonitor:
    """Monitor overall system health."""
    
    def __init__(self):
        """Initialize system health monitor."""
        self.performance_monitor = PerformanceMonitor()
        self.safety_monitor = SafetyMonitor()
        self.health_history: deque = deque(maxlen=1000)
        self.lock = threading.Lock()
        
        # Health check intervals
        self.performance_check_interval = 1.0  # 1 second
        self.safety_check_interval = 0.1      # 100ms
        
        # Monitoring threads
        self.monitoring_thread = None
        self.running = False
    
    def start_monitoring(self) -> None:
        """Start continuous monitoring."""
        if self.monitoring_thread is not None:
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
            self.monitoring_thread = None
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        last_performance_check = time.time()
        last_safety_check = time.time()
        
        while self.running:
            current_time = time.time()
            
            # Performance monitoring
            if current_time - last_performance_check >= self.performance_check_interval:
                self._check_performance_health()
                last_performance_check = current_time
            
            # Safety monitoring
            if current_time - last_safety_check >= self.safety_check_interval:
                self._check_safety_health()
                last_safety_check = current_time
            
            time.sleep(0.01)  # 10ms sleep
    
    def _check_performance_health(self) -> None:
        """Check performance health."""
        # Get system metrics
        metrics = PerformanceMetrics()
        
        # Memory usage
        if torch.cuda.is_available():
            metrics.gpu_memory_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        
        # Record metrics
        self.performance_monitor.record_metrics(metrics, "system")
    
    def _check_safety_health(self) -> None:
        """Check safety health."""
        # This would typically get safety metrics from the driving system
        # For now, we'll create placeholder metrics
        metrics = SafetyMetrics()
        metrics.model_health = 1.0  # Placeholder
        
        self.safety_monitor.record_safety_metrics(metrics)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        with self.lock:
            performance_avg = self.performance_monitor.get_average_metrics("system", 100)
            safety_summary = self.safety_monitor.get_safety_summary(100)
            
            return {
                "performance": {
                    "avg_latency": performance_avg.total_latency,
                    "avg_fps": performance_avg.frames_per_second,
                    "memory_usage": performance_avg.memory_usage,
                    "gpu_memory_usage": performance_avg.gpu_memory_usage
                },
                "safety": safety_summary,
                "status": "healthy" if self._is_healthy() else "degraded",
                "timestamp": datetime.now().isoformat()
            }
    
    def _is_healthy(self) -> bool:
        """Check if system is healthy."""
        performance_avg = self.performance_monitor.get_average_metrics("system", 10)
        safety_summary = self.safety_monitor.get_safety_summary(10)
        
        # Check performance
        if performance_avg.total_latency > 0.1:  # 100ms threshold
            return False
        
        if performance_avg.frames_per_second < 10:  # 10 FPS threshold
            return False
        
        # Check safety
        if safety_summary.get("avg_collision_risk", 0) > 0.7:
            return False
        
        if safety_summary.get("avg_model_health", 1) < 0.5:
            return False
        
        return True
    
    def add_performance_alert_callback(self, callback: Callable) -> None:
        """Add performance alert callback."""
        self.performance_monitor.add_alert_callback(callback)
    
    def add_safety_alert_callback(self, callback: Callable) -> None:
        """Add safety alert callback."""
        self.safety_monitor.add_alert_callback(callback)
    
    def add_emergency_callback(self, callback: Callable) -> None:
        """Add emergency callback."""
        self.safety_monitor.add_emergency_callback(callback)


class StructuredLogger:
    """Structured logging with performance and safety context."""
    
    def __init__(self, log_file: str = "opendrive_xai.log"):
        """
        Initialize structured logger.
        
        Args:
            log_file: Path to log file
        """
        self.log_file = log_file
        self.logger = logging.getLogger("OpenDrive-XAI")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_performance(self, component: str, metrics: PerformanceMetrics) -> None:
        """Log performance metrics."""
        log_data = {
            "type": "performance",
            "component": component,
            "inference_time": metrics.inference_time,
            "total_latency": metrics.total_latency,
            "fps": metrics.frames_per_second,
            "memory_usage": metrics.memory_usage,
            "timestamp": metrics.timestamp.isoformat()
        }
        
        self.logger.info(f"Performance: {json.dumps(log_data)}")
    
    def log_safety(self, metrics: SafetyMetrics) -> None:
        """Log safety metrics."""
        log_data = {
            "type": "safety",
            "collision_risk": metrics.collision_risk,
            "speed_violation_risk": metrics.speed_violation_risk,
            "lane_deviation_risk": metrics.lane_deviation_risk,
            "model_health": metrics.model_health,
            "violations": metrics.safety_violations,
            "timestamp": metrics.timestamp.isoformat()
        }
        
        self.logger.info(f"Safety: {json.dumps(log_data)}")
    
    def log_alert(self, alert: Dict[str, Any]) -> None:
        """Log alerts."""
        self.logger.warning(f"Alert: {json.dumps(alert)}")
    
    def log_emergency(self, emergency: Dict[str, Any]) -> None:
        """Log emergency events."""
        self.logger.error(f"Emergency: {json.dumps(emergency)}")
    
    def log_system_event(self, event: str, data: Dict[str, Any]) -> None:
        """Log system events."""
        log_data = {
            "type": "system_event",
            "event": event,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"System Event: {json.dumps(log_data)}")


# Global monitoring instances
_performance_monitor: Optional[PerformanceMonitor] = None
_safety_monitor: Optional[SafetyMonitor] = None
_system_health_monitor: Optional[SystemHealthMonitor] = None
_structured_logger: Optional[StructuredLogger] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def get_safety_monitor() -> SafetyMonitor:
    """Get global safety monitor."""
    global _safety_monitor
    if _safety_monitor is None:
        _safety_monitor = SafetyMonitor()
    return _safety_monitor


def get_system_health_monitor() -> SystemHealthMonitor:
    """Get global system health monitor."""
    global _system_health_monitor
    if _system_health_monitor is None:
        _system_health_monitor = SystemHealthMonitor()
    return _system_health_monitor


def get_structured_logger() -> StructuredLogger:
    """Get global structured logger."""
    global _structured_logger
    if _structured_logger is None:
        system_config = get_system_config()
        log_file = system_config['logging']['file']
        _structured_logger = StructuredLogger(log_file)
    return _structured_logger


def start_monitoring() -> None:
    """Start all monitoring systems."""
    get_system_health_monitor().start_monitoring()
    
    # Set up alert callbacks
    logger = get_structured_logger()
    performance_monitor = get_performance_monitor()
    safety_monitor = get_safety_monitor()
    
    performance_monitor.add_alert_callback(logger.log_alert)
    safety_monitor.add_alert_callback(logger.log_alert)
    safety_monitor.add_emergency_callback(logger.log_emergency)


def stop_monitoring() -> None:
    """Stop all monitoring systems."""
    get_system_health_monitor().stop_monitoring()


def get_system_status() -> Dict[str, Any]:
    """Get comprehensive system status."""
    return get_system_health_monitor().get_system_status() 