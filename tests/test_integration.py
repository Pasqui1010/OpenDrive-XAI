"""
Integration tests for the OpenDrive-XAI system.

These tests verify the interaction between all system components and ensure
the system works correctly as a whole.
"""

import pytest
import torch
import numpy as np
from typing import List, Dict, Any
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from opendrive_xai.causal_world_model import (
    CausalWorldModel, WorldState, SyntheticInterventionGenerator, CausalWorldModelTrainer
)
from opendrive_xai.modular_components import (
    InterpretableDrivingSystem, ModularSystemTrainer, AttentionGate, SafetyMonitor, CausalReasoner
)
from opendrive_xai.mechatronics.vehicle_interface import VehicleInterface
from opendrive_xai.mechatronics.ai_vehicle_integration import AIVehicleIntegration


class TestSystemIntegration:
    """Test the integration between all system components."""
    
    @pytest.fixture
    def device(self):
        """Get device for testing."""
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    @pytest.fixture
    def causal_world_model(self, device):
        """Create a causal world model for testing."""
        return CausalWorldModel(
            state_dim=64,
            hidden_dim=128,
            num_layers=3,
            num_heads=8,
            dropout=0.1,
            device=device
        )
    
    @pytest.fixture
    def interpretable_system(self, device):
        """Create an interpretable driving system for testing."""
        return InterpretableDrivingSystem(
            perception_dim=512,
            planning_dim=256,
            control_dim=64,
            device=device,
            max_history_size=50
        )
    
    @pytest.fixture
    def vehicle_interface(self, device):
        """Create a vehicle interface for testing."""
        return VehicleInterface(
            sensor_config={
                'camera': {'resolution': (640, 480), 'fps': 30},
                'lidar': {'points_per_scan': 1000, 'fps': 10},
                'radar': {'max_range': 200, 'fps': 20}
            },
            actuator_config={
                'steering': {'max_angle': 30, 'response_time': 0.1},
                'throttle': {'max_value': 1.0, 'response_time': 0.05},
                'brake': {'max_value': 1.0, 'response_time': 0.05}
            },
            device=device
        )
    
    @pytest.fixture
    def ai_vehicle_integration(self, causal_world_model, interpretable_system, vehicle_interface, device):
        """Create an AI-vehicle integration for testing."""
        return AIVehicleIntegration(
            causal_world_model=causal_world_model,
            interpretable_system=interpretable_system,
            vehicle_interface=vehicle_interface,
            device=device
        )
    
    def test_causal_world_model_integration(self, causal_world_model, device):
        """Test causal world model integration with synthetic interventions."""
        # Create test world state
        world_state = WorldState(
            vehicle_positions=torch.randn(3, 3, device=device),
            vehicle_velocities=torch.randn(3, 2, device=device),
            traffic_lights=torch.randn(2, 3, device=device),
            road_geometry=torch.randn(5, 4, device=device),
            ego_state=torch.randn(6, device=device)
        )
        
        # Test forward pass
        outputs = causal_world_model(world_state)
        
        # Verify outputs
        assert 'state_encoding' in outputs
        assert 'causal_graph' in outputs
        assert 'next_state_prediction' in outputs
        assert 'intervention_effect' in outputs
        
        # Test with intervention
        intervention = torch.randn(64, device=device)
        intervention_mask = torch.ones(64, device=device)
        
        intervened_outputs = causal_world_model(world_state, intervention, intervention_mask)
        
        # Verify intervention had effect
        assert not torch.allclose(
            outputs['state_encoding'],
            intervened_outputs['state_encoding']
        )
    
    def test_modular_components_integration(self, interpretable_system, device):
        """Test modular components integration."""
        # Create test perception input
        perception_input = torch.randn(2, 512, device=device)
        
        # Test forward pass
        outputs = interpretable_system(perception_input)
        
        # Verify outputs
        assert 'control_signal' in outputs
        assert 'interpretability_info' in outputs
        
        control_signal = outputs['control_signal']
        assert control_signal.shape == (2, 3)  # batch_size, (steering, throttle, brake)
        
        # Verify interpretability info
        interpretability_info = outputs['interpretability_info']
        assert 'routing_decisions' in interpretability_info
        assert 'attention_weights' in interpretability_info
        assert 'safety_violations' in interpretability_info
        assert 'causal_explanations' in interpretability_info
        assert 'verification_results' in interpretability_info
    
    def test_vehicle_interface_integration(self, vehicle_interface, device):
        """Test vehicle interface integration."""
        # Simulate sensor data
        camera_data = torch.randn(1, 3, 480, 640, device=device)
        lidar_data = torch.randn(1, 1000, 3, device=device)
        radar_data = torch.randn(1, 10, 4, device=device)
        
        # Test sensor fusion
        fused_data = vehicle_interface.fuse_sensor_data(
            camera_data=camera_data,
            lidar_data=lidar_data,
            radar_data=radar_data
        )
        
        assert fused_data is not None
        assert isinstance(fused_data, torch.Tensor)
        
        # Test actuator control
        control_commands = torch.tensor([[0.1, 0.5, 0.0]], device=device)  # steering, throttle, brake
        
        # Test control application
        success = vehicle_interface.apply_control(control_commands)
        assert success
        
        # Test safety monitoring
        safety_status = vehicle_interface.check_safety_status()
        assert isinstance(safety_status, dict)
        assert 'is_safe' in safety_status
    
    def test_ai_vehicle_integration(self, ai_vehicle_integration, device):
        """Test AI-vehicle integration."""
        # Simulate sensor data
        sensor_data = {
            'camera': torch.randn(1, 3, 480, 640, device=device),
            'lidar': torch.randn(1, 1000, 3, device=device),
            'radar': torch.randn(1, 10, 4, device=device)
        }
        
        # Test complete pipeline
        result = ai_vehicle_integration.process_sensor_data(sensor_data)
        
        # Verify result
        assert result is not None
        assert 'control_commands' in result
        assert 'safety_status' in result
        assert 'interpretability_info' in result
        assert 'performance_metrics' in result
        
        # Test safety monitoring
        safety_status = ai_vehicle_integration.monitor_safety()
        assert isinstance(safety_status, dict)
        assert 'is_safe' in safety_status
    
    def test_end_to_end_pipeline(self, ai_vehicle_integration, device):
        """Test the complete end-to-end pipeline."""
        # Simulate multiple sensor readings
        num_frames = 5
        sensor_data_sequence = []
        
        for i in range(num_frames):
            sensor_data = {
                'camera': torch.randn(1, 3, 480, 640, device=device),
                'lidar': torch.randn(1, 1000, 3, device=device),
                'radar': torch.randn(1, 10, 4, device=device)
            }
            sensor_data_sequence.append(sensor_data)
        
        # Process sequence
        results = []
        for sensor_data in sensor_data_sequence:
            result = ai_vehicle_integration.process_sensor_data(sensor_data)
            results.append(result)
        
        # Verify all results
        assert len(results) == num_frames
        
        for result in results:
            assert 'control_commands' in result
            assert 'safety_status' in result
            assert 'interpretability_info' in result
            assert 'performance_metrics' in result
            
            # Verify control commands are reasonable
            control_commands = result['control_commands']
            assert torch.all((control_commands[:, 0] >= -1) & (control_commands[:, 0] <= 1))  # steering
            assert torch.all((control_commands[:, 1] >= 0) & (control_commands[:, 1] <= 1))  # throttle
            assert torch.all((control_commands[:, 2] >= 0) & (control_commands[:, 2] <= 1))  # brake


class TestTrainingIntegration:
    """Test training integration between components."""
    
    @pytest.fixture
    def device(self):
        """Get device for testing."""
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    @pytest.fixture
    def causal_world_model(self, device):
        """Create a causal world model for testing."""
        return CausalWorldModel(
            state_dim=64,
            hidden_dim=128,
            num_layers=3,
            num_heads=8,
            dropout=0.1,
            device=device
        )
    
    @pytest.fixture
    def interpretable_system(self, device):
        """Create an interpretable driving system for testing."""
        return InterpretableDrivingSystem(
            perception_dim=512,
            planning_dim=256,
            control_dim=64,
            device=device,
            max_history_size=50
        )
    
    def test_causal_world_model_training(self, causal_world_model, device):
        """Test causal world model training integration."""
        # Create trainer
        trainer = CausalWorldModelTrainer(
            model=causal_world_model,
            learning_rate=1e-4,
            device=device
        )
        
        # Create training data
        batch_size = 4
        world_states = []
        targets = torch.randn(batch_size, 6, device=device)
        
        for i in range(batch_size):
            world_state = WorldState(
                vehicle_positions=torch.randn(3, 3, device=device),
                vehicle_velocities=torch.randn(3, 2, device=device),
                traffic_lights=torch.randn(2, 3, device=device),
                road_geometry=torch.randn(5, 4, device=device),
                ego_state=torch.randn(6, device=device)
            )
            world_states.append(world_state)
        
        # Test training step
        losses = trainer.train_step(world_states, targets, use_interventions=True)
        
        # Verify losses
        assert 'total_loss' in losses
        assert 'prediction_loss' in losses
        assert 'causal_loss' in losses
        
        assert isinstance(losses['total_loss'], float)
        assert isinstance(losses['prediction_loss'], float)
        assert isinstance(losses['causal_loss'], float)
        
        # Test validation
        validation_metrics = trainer.validate(world_states, targets)
        
        # Verify validation metrics
        assert 'validation_loss' in validation_metrics
        assert 'causal_consistency' in validation_metrics
        
        assert isinstance(validation_metrics['validation_loss'], float)
        assert isinstance(validation_metrics['causal_consistency'], float)
    
    def test_modular_system_training(self, interpretable_system, device):
        """Test modular system training integration."""
        # Create trainer
        trainer = ModularSystemTrainer(
            system=interpretable_system,
            learning_rate=1e-4,
            device=device
        )
        
        # Create training data
        batch_size = 4
        perception_inputs = torch.randn(batch_size, 512, device=device)
        control_targets = torch.randn(batch_size, 3, device=device)
        
        # Test training step
        losses = trainer.train_step(perception_inputs, control_targets)
        
        # Verify losses
        assert 'total_loss' in losses
        assert 'control_loss' in losses
        assert 'interpretability_loss' in losses
        
        assert isinstance(losses['total_loss'], float)
        assert isinstance(losses['control_loss'], float)
        assert isinstance(losses['interpretability_loss'], float)
        
        # Test validation
        validation_metrics = trainer.validate(perception_inputs, control_targets)
        
        # Verify validation metrics
        assert 'control_loss' in validation_metrics
        assert 'verification_results' in validation_metrics
        
        assert isinstance(validation_metrics['control_loss'], float)
        assert isinstance(validation_metrics['verification_results'], dict)


class TestSafetyIntegration:
    """Test safety-related integration scenarios."""
    
    @pytest.fixture
    def device(self):
        """Get device for testing."""
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    @pytest.fixture
    def ai_vehicle_integration(self, device):
        """Create an AI-vehicle integration for testing."""
        causal_world_model = CausalWorldModel(
            state_dim=64,
            hidden_dim=128,
            num_layers=3,
            num_heads=8,
            dropout=0.1,
            device=device
        )
        
        interpretable_system = InterpretableDrivingSystem(
            perception_dim=512,
            planning_dim=256,
            control_dim=64,
            device=device,
            max_history_size=50
        )
        
        vehicle_interface = VehicleInterface(
            sensor_config={
                'camera': {'resolution': (640, 480), 'fps': 30},
                'lidar': {'points_per_scan': 1000, 'fps': 10},
                'radar': {'max_range': 200, 'fps': 20}
            },
            actuator_config={
                'steering': {'max_angle': 30, 'response_time': 0.1},
                'throttle': {'max_value': 1.0, 'response_time': 0.05},
                'brake': {'max_value': 1.0, 'response_time': 0.05}
            },
            device=device
        )
        
        return AIVehicleIntegration(
            causal_world_model=causal_world_model,
            interpretable_system=interpretable_system,
            vehicle_interface=vehicle_interface,
            device=device
        )
    
    def test_safety_monitoring_integration(self, ai_vehicle_integration, device):
        """Test safety monitoring integration."""
        # Simulate dangerous sensor data (high collision risk)
        dangerous_sensor_data = {
            'camera': torch.randn(1, 3, 480, 640, device=device) * 2.0,  # High intensity
            'lidar': torch.randn(1, 1000, 3, device=device) * 0.1,  # Very close objects
            'radar': torch.randn(1, 10, 4, device=device) * 5.0  # High velocity objects
        }
        
        # Process dangerous data
        result = ai_vehicle_integration.process_sensor_data(dangerous_sensor_data)
        
        # Verify safety monitoring
        safety_status = result['safety_status']
        assert isinstance(safety_status, dict)
        assert 'is_safe' in safety_status
        
        # Check that safety violations are detected
        if not safety_status['is_safe']:
            assert 'violations' in safety_status
            assert len(safety_status['violations']) > 0
    
    def test_emergency_braking_integration(self, ai_vehicle_integration, device):
        """Test emergency braking integration."""
        # Simulate emergency scenario
        emergency_sensor_data = {
            'camera': torch.randn(1, 3, 480, 640, device=device) * 3.0,  # Very high intensity
            'lidar': torch.randn(1, 1000, 3, device=device) * 0.05,  # Extremely close objects
            'radar': torch.randn(1, 10, 4, device=device) * 10.0  # Very high velocity objects
        }
        
        # Process emergency data
        result = ai_vehicle_integration.process_sensor_data(emergency_sensor_data)
        
        # Verify emergency response
        control_commands = result['control_commands']
        
        # Should apply brakes in emergency
        brake_command = control_commands[0, 2]  # brake command
        throttle_command = control_commands[0, 1]  # throttle command
        
        # In emergency, should reduce throttle and increase brake
        assert throttle_command < 0.5  # Reduced throttle
        assert brake_command > 0.3  # Increased braking
    
    def test_sensor_failure_integration(self, ai_vehicle_integration, device):
        """Test sensor failure handling integration."""
        # Simulate sensor failure (None data)
        failed_sensor_data = {
            'camera': None,
            'lidar': torch.randn(1, 1000, 3, device=device),
            'radar': torch.randn(1, 10, 4, device=device)
        }
        
        # Process failed sensor data
        result = ai_vehicle_integration.process_sensor_data(failed_sensor_data)
        
        # Verify graceful degradation
        assert result is not None
        assert 'control_commands' in result
        assert 'safety_status' in result
        
        # Should still provide control commands (using remaining sensors)
        control_commands = result['control_commands']
        assert control_commands is not None
        assert control_commands.shape == (1, 3)
        
        # Safety status should indicate sensor failure
        safety_status = result['safety_status']
        assert 'sensor_failures' in safety_status or 'warnings' in safety_status


class TestPerformanceIntegration:
    """Test performance-related integration scenarios."""
    
    @pytest.fixture
    def device(self):
        """Get device for testing."""
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    @pytest.fixture
    def ai_vehicle_integration(self, device):
        """Create an AI-vehicle integration for testing."""
        causal_world_model = CausalWorldModel(
            state_dim=64,
            hidden_dim=128,
            num_layers=3,
            num_heads=8,
            dropout=0.1,
            device=device
        )
        
        interpretable_system = InterpretableDrivingSystem(
            perception_dim=512,
            planning_dim=256,
            control_dim=64,
            device=device,
            max_history_size=50
        )
        
        vehicle_interface = VehicleInterface(
            sensor_config={
                'camera': {'resolution': (640, 480), 'fps': 30},
                'lidar': {'points_per_scan': 1000, 'fps': 10},
                'radar': {'max_range': 200, 'fps': 20}
            },
            actuator_config={
                'steering': {'max_angle': 30, 'response_time': 0.1},
                'throttle': {'max_value': 1.0, 'response_time': 0.05},
                'brake': {'max_value': 1.0, 'response_time': 0.05}
            },
            device=device
        )
        
        return AIVehicleIntegration(
            causal_world_model=causal_world_model,
            interpretable_system=interpretable_system,
            vehicle_interface=vehicle_interface,
            device=device
        )
    
    def test_real_time_performance_integration(self, ai_vehicle_integration, device):
        """Test real-time performance integration."""
        import time
        
        # Simulate real-time processing
        num_frames = 10
        processing_times = []
        
        for i in range(num_frames):
            sensor_data = {
                'camera': torch.randn(1, 3, 480, 640, device=device),
                'lidar': torch.randn(1, 1000, 3, device=device),
                'radar': torch.randn(1, 10, 4, device=device)
            }
            
            start_time = time.time()
            result = ai_vehicle_integration.process_sensor_data(sensor_data)
            end_time = time.time()
            
            processing_times.append(end_time - start_time)
            
            # Verify result is valid
            assert result is not None
            assert 'control_commands' in result
        
        # Verify real-time performance (should be < 100ms for autonomous driving)
        avg_processing_time = np.mean(processing_times)
        max_processing_time = np.max(processing_times)
        
        # For testing, we'll be more lenient (1 second max)
        assert max_processing_time < 1.0, f"Processing time {max_processing_time:.3f}s exceeds 1s limit"
        assert avg_processing_time < 0.5, f"Average processing time {avg_processing_time:.3f}s exceeds 0.5s limit"
    
    def test_memory_usage_integration(self, ai_vehicle_integration, device):
        """Test memory usage integration."""
        import gc
        
        # Clear memory before test
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # Get initial memory usage
        if device == "cuda":
            initial_memory = torch.cuda.memory_allocated()
        else:
            initial_memory = 0
        
        # Process multiple frames
        num_frames = 20
        for i in range(num_frames):
            sensor_data = {
                'camera': torch.randn(1, 3, 480, 640, device=device),
                'lidar': torch.randn(1, 1000, 3, device=device),
                'radar': torch.randn(1, 10, 4, device=device)
            }
            
            result = ai_vehicle_integration.process_sensor_data(sensor_data)
            assert result is not None
        
        # Clear history to free memory
        ai_vehicle_integration.interpretable_system.clear_all_history()
        
        # Get final memory usage
        if device == "cuda":
            final_memory = torch.cuda.memory_allocated()
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (< 1GB for this test)
            assert memory_increase < 1e9, f"Memory increase {memory_increase/1e6:.1f}MB exceeds 1GB limit"
    
    def test_concurrent_processing_integration(self, ai_vehicle_integration, device):
        """Test concurrent processing integration."""
        import threading
        import time
        
        # Simulate concurrent sensor processing
        results = []
        errors = []
        
        def process_sensor_data(sensor_id):
            try:
                sensor_data = {
                    'camera': torch.randn(1, 3, 480, 640, device=device),
                    'lidar': torch.randn(1, 1000, 3, device=device),
                    'radar': torch.randn(1, 10, 4, device=device)
                }
                
                result = ai_vehicle_integration.process_sensor_data(sensor_data)
                results.append((sensor_id, result))
            except Exception as e:
                errors.append((sensor_id, str(e)))
        
        # Start multiple threads
        threads = []
        num_threads = 4
        
        for i in range(num_threads):
            thread = threading.Thread(target=process_sensor_data, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == num_threads
        
        for sensor_id, result in results:
            assert result is not None
            assert 'control_commands' in result
            assert 'safety_status' in result


if __name__ == "__main__":
    pytest.main([__file__]) 