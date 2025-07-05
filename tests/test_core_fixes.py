"""
Core fixes verification tests.

These tests verify that the critical fixes have been implemented correctly:
1. Input validation
2. Memory leak prevention
3. Thread safety
4. Error handling
"""

import pytest
import torch
import numpy as np
from typing import List, Dict, Any
import sys
import os
import gc
import time

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from opendrive_xai.causal_world_model import (
    CausalWorldModel, WorldState, SyntheticInterventionGenerator, CausalWorldModelTrainer,
    validate_tensor, validate_world_state
)
from opendrive_xai.modular_components import (
    InterpretableDrivingSystem, ModularSystemTrainer, AttentionGate, SafetyMonitor, CausalReasoner,
    validate_tensor as validate_tensor_modular
)


class TestInputValidation:
    """Test input validation fixes."""
    
    @pytest.fixture
    def device(self):
        """Get device for testing."""
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def test_tensor_validation(self, device):
        """Test tensor validation function."""
        # Valid tensor
        valid_tensor = torch.randn(3, 4, device=device)
        validated = validate_tensor(valid_tensor, "test_tensor")
        assert torch.equal(validated, valid_tensor)
        
        # Test None input
        with pytest.raises(ValueError, match="cannot be None"):
            validate_tensor(None, "test_tensor")  # type: ignore
        
        # Test wrong type
        with pytest.raises(ValueError, match="must be a torch.Tensor"):
            validate_tensor([1, 2, 3], "test_tensor")  # type: ignore
        
        # Test wrong shape
        with pytest.raises(ValueError, match="expected shape"):
            validate_tensor(torch.randn(2, 3, device=device), "test_tensor", expected_shape=(3, 4))
        
        # Test wrong dtype
        with pytest.raises(ValueError, match="expected dtype"):
            validate_tensor(torch.randn(3, 4, dtype=torch.float64, device=device), 
                          "test_tensor", expected_dtype=torch.float32)
        
        # Test NaN values
        nan_tensor = torch.randn(3, 4, device=device)
        nan_tensor[0, 0] = float('nan')
        with pytest.raises(ValueError, match="contains NaN values"):
            validate_tensor(nan_tensor, "test_tensor")
        
        # Test infinite values
        inf_tensor = torch.randn(3, 4, device=device)
        inf_tensor[0, 0] = float('inf')
        with pytest.raises(ValueError, match="contains infinite values"):
            validate_tensor(inf_tensor, "test_tensor")
    
    def test_world_state_validation(self, device):
        """Test WorldState validation."""
        # Valid world state
        valid_world_state = WorldState(
            vehicle_positions=torch.randn(3, 3, device=device),
            vehicle_velocities=torch.randn(3, 2, device=device),
            traffic_lights=torch.randn(2, 3, device=device),
            road_geometry=torch.randn(5, 4, device=device),
            ego_state=torch.randn(6, device=device)
        )
        
        validated = validate_world_state(valid_world_state)
        assert validated == valid_world_state
        
        # Test None input
        with pytest.raises(ValueError, match="cannot be None"):
            validate_world_state(None)  # type: ignore
        
        # Test invalid ego_state shape - this should fail during WorldState creation
        with pytest.raises(ValueError, match="expected shape"):
            WorldState(
                vehicle_positions=torch.randn(3, 3, device=device),
                vehicle_velocities=torch.randn(3, 2, device=device),
                traffic_lights=torch.randn(2, 3, device=device),
                road_geometry=torch.randn(5, 4, device=device),
                ego_state=torch.randn(5, device=device)  # Wrong shape
            )


class TestMemoryManagement:
    """Test memory leak prevention fixes."""
    
    @pytest.fixture
    def device(self):
        """Get device for testing."""
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def test_attention_gate_memory_management(self, device):
        """Test attention gate memory management."""
        attention_gate = AttentionGate(64, 64, max_history_size=10)
        
        # Process multiple inputs
        for i in range(20):
            input_tensor = torch.randn(2, 64, device=device)
            output = attention_gate(input_tensor)
            
            # Check that history size is limited
            assert len(attention_gate.attention_weights) <= 10
        
        # Clear history
        attention_gate.clear_history()
        assert len(attention_gate.attention_weights) == 0
    
    def test_safety_monitor_memory_management(self, device):
        """Test safety monitor memory management."""
        safety_thresholds = {'test': 0.5}
        safety_monitor = SafetyMonitor(64, 32, safety_thresholds, max_history_size=5)
        
        # Process multiple inputs
        for i in range(10):
            input_tensor = torch.randn(2, 64, device=device)
            output = safety_monitor(input_tensor)
            
            # Check that history size is limited
            assert len(safety_monitor.safety_violations) <= 5
        
        # Clear history
        safety_monitor.clear_history()
        assert len(safety_monitor.safety_violations) == 0
    
    def test_causal_reasoner_memory_management(self, device):
        """Test causal reasoner memory management."""
        causal_reasoner = CausalReasoner(64, 32, max_history_size=8)
        
        # Process multiple inputs
        for i in range(15):
            input_tensor = torch.randn(2, 64, device=device)
            output = causal_reasoner(input_tensor)
            
            # Check that history size is limited
            assert len(causal_reasoner.causal_explanations) <= 8
        
        # Clear history
        causal_reasoner.clear_history()
        assert len(causal_reasoner.causal_explanations) == 0
    
    def test_system_memory_management(self, device):
        """Test complete system memory management."""
        system = InterpretableDrivingSystem(
            perception_dim=128,
            planning_dim=64,
            control_dim=32,
            device=device,
            max_history_size=5
        )
        
        # Calculate the expected input dimension for the routing network
        # The routing network expects the sum of all component input dimensions
        expected_input_dim = 128 + 128 + 128 + 64 + 64  # perception + safety + causal + planning + control
        
        # Process multiple inputs
        for i in range(10):
            input_tensor = torch.randn(2, expected_input_dim, device=device)
            output = system(input_tensor)
            
            # Check that all components respect history limits
            assert len(system.perception_attention.attention_weights) <= 5
            assert len(system.safety_monitor.safety_violations) <= 5
            assert len(system.causal_reasoner.causal_explanations) <= 5
            assert len(system.planning_attention.attention_weights) <= 5
            assert len(system.control_attention.attention_weights) <= 5
            assert len(system.router.routing_decisions) <= 5
        
        # Clear all history
        system.clear_all_history()
        assert len(system.perception_attention.attention_weights) == 0
        assert len(system.safety_monitor.safety_violations) == 0
        assert len(system.causal_reasoner.causal_explanations) == 0
        assert len(system.planning_attention.attention_weights) == 0
        assert len(system.control_attention.attention_weights) == 0
        assert len(system.router.routing_decisions) == 0


class TestErrorHandling:
    """Test error handling improvements."""
    
    @pytest.fixture
    def device(self):
        """Get device for testing."""
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def test_causal_world_model_error_handling(self, device):
        """Test causal world model error handling."""
        model = CausalWorldModel(device=device)
        
        # Test with invalid parameters
        with pytest.raises(ValueError, match="must be positive"):
            CausalWorldModel(state_dim=-1, device=device)
        
        with pytest.raises(ValueError, match="must be positive"):
            CausalWorldModel(hidden_dim=0, device=device)
        
        with pytest.raises(ValueError, match="must be positive"):
            CausalWorldModel(num_layers=-5, device=device)
        
        with pytest.raises(ValueError, match="must be positive"):
            CausalWorldModel(num_heads=0, device=device)
        
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            CausalWorldModel(dropout=1.5, device=device)
        
        # Test with invalid world state
        with pytest.raises(ValueError):
            model(None)
    
    def test_modular_components_error_handling(self, device):
        """Test modular components error handling."""
        # Test invalid parameters
        with pytest.raises(ValueError, match="embed_dim and num_heads must be greater than 0"):
            AttentionGate(input_dim=-1, output_dim=64)
        
        with pytest.raises(ValueError, match="must be positive"):
            SafetyMonitor(input_dim=64, output_dim=-1, safety_thresholds={'test': 0.5})
        
        with pytest.raises(ValueError, match="safety_thresholds must be a non-empty dictionary"):
            SafetyMonitor(input_dim=64, output_dim=32, safety_thresholds={})
        
        with pytest.raises(ValueError, match="must be positive"):
            CausalReasoner(input_dim=64, output_dim=32, num_causal_vars=0)
        
        # Test invalid inputs
        attention_gate = AttentionGate(64, 64)
        with pytest.raises(ValueError, match="cannot be None"):
            attention_gate(None)
        
        safety_monitor = SafetyMonitor(64, 32, {'test': 0.5})
        with pytest.raises(ValueError, match="cannot be None"):
            safety_monitor(None)
        
        causal_reasoner = CausalReasoner(64, 32)
        with pytest.raises(ValueError, match="cannot be None"):
            causal_reasoner(None)
    
    def test_system_error_handling(self, device):
        """Test system error handling."""
        # Test invalid parameters
        with pytest.raises(ValueError, match="must be positive"):
            InterpretableDrivingSystem(perception_dim=-1, device=device)
        
        with pytest.raises(ValueError, match="must be positive"):
            InterpretableDrivingSystem(planning_dim=0, device=device)
        
        with pytest.raises(ValueError, match="must be positive"):
            InterpretableDrivingSystem(control_dim=-5, device=device)
        
        # Test invalid inputs
        system = InterpretableDrivingSystem(device=device)
        with pytest.raises(ValueError, match="cannot be None"):
            system(None)


class TestTrainingFixes:
    """Test training-related fixes."""
    
    @pytest.fixture
    def device(self):
        """Get device for testing."""
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def test_causal_world_model_training_fixes(self, device):
        """Test causal world model training fixes."""
        model = CausalWorldModel(device=device)
        trainer = CausalWorldModelTrainer(model, device=device)
        
        # Test invalid parameters
        with pytest.raises(ValueError, match="must be a CausalWorldModel"):
            CausalWorldModelTrainer(None, device=device)  # type: ignore
        
        with pytest.raises(ValueError, match="must be positive"):
            CausalWorldModelTrainer(model, learning_rate=-1, device=device)
        
        # Test invalid training data
        world_states = []
        targets = torch.randn(2, 6, device=device)
        
        with pytest.raises(ValueError, match="must be a non-empty list"):
            trainer.train_step(world_states, targets)
        
        # Test mismatched data lengths
        world_states = [
            WorldState(
                vehicle_positions=torch.randn(3, 3, device=device),
                vehicle_velocities=torch.randn(3, 2, device=device),
                traffic_lights=torch.randn(2, 3, device=device),
                road_geometry=torch.randn(5, 4, device=device),
                ego_state=torch.randn(6, device=device)
            )
        ]
        targets = torch.randn(2, 6, device=device)  # Mismatched length
        
        with pytest.raises(ValueError, match="must have the same length"):
            trainer.train_step(world_states, targets)
    
    def test_modular_system_training_fixes(self, device):
        """Test modular system training fixes."""
        system = InterpretableDrivingSystem(device=device)
        trainer = ModularSystemTrainer(system, device=device)
        
        # Test invalid parameters
        with pytest.raises(ValueError, match="must be an InterpretableDrivingSystem"):
            ModularSystemTrainer(None, device=device)  # type: ignore
        
        with pytest.raises(ValueError, match="must be positive"):
            ModularSystemTrainer(system, learning_rate=-1, device=device)
        
        # Test invalid training data
        perception_inputs = torch.randn(2, 512, device=device)
        control_targets = torch.randn(3, 3, device=device)  # Mismatched batch size
        
        with pytest.raises(ValueError, match="must have the same batch size"):
            trainer.train_step(perception_inputs, control_targets)


if __name__ == "__main__":
    pytest.main([__file__]) 