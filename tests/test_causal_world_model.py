"""
Unit tests for the causal world model module.

Tests cover:
- Basic functionality
- Edge cases
- Error handling
- Performance characteristics
- Causal reasoning capabilities
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch

from src.opendrive_xai.causal_world_model import (
    WorldState,
    CausalWorldModel,
    SyntheticInterventionGenerator,
    CausalWorldModelTrainer
)


class TestWorldState:
    """Test the WorldState dataclass."""
    
    def test_world_state_creation(self):
        """Test creating a WorldState with valid data."""
        # Create test data
        vehicle_positions = torch.randn(3, 3)  # 3 vehicles, 3D positions
        vehicle_velocities = torch.randn(3, 2)  # 3 vehicles, 2D velocities
        traffic_lights = torch.randn(2, 3)  # 2 traffic lights, 3D positions + state
        road_geometry = torch.randn(5, 4)  # 5 road segments, 2D endpoints
        ego_state = torch.randn(6)  # ego vehicle state
        
        # Create WorldState
        world_state = WorldState(
            vehicle_positions=vehicle_positions,
            vehicle_velocities=vehicle_velocities,
            traffic_lights=traffic_lights,
            road_geometry=road_geometry,
            ego_state=ego_state
        )
        
        # Verify all fields are set correctly
        assert torch.equal(world_state.vehicle_positions, vehicle_positions)
        assert torch.equal(world_state.vehicle_velocities, vehicle_velocities)
        assert torch.equal(world_state.traffic_lights, traffic_lights)
        assert torch.equal(world_state.road_geometry, road_geometry)
        assert torch.equal(world_state.ego_state, ego_state)
    
    def test_world_state_empty_vehicles(self):
        """Test WorldState with no other vehicles."""
        # Create test data with no vehicles
        vehicle_positions = torch.empty(0, 3)
        vehicle_velocities = torch.empty(0, 2)
        traffic_lights = torch.randn(2, 3)
        road_geometry = torch.randn(5, 4)
        ego_state = torch.randn(6)
        
        # Create WorldState
        world_state = WorldState(
            vehicle_positions=vehicle_positions,
            vehicle_velocities=vehicle_velocities,
            traffic_lights=traffic_lights,
            road_geometry=road_geometry,
            ego_state=ego_state
        )
        
        # Verify empty tensors are handled correctly
        assert world_state.vehicle_positions.shape == (0, 3)
        assert world_state.vehicle_velocities.shape == (0, 2)


class TestCausalWorldModel:
    """Test the CausalWorldModel class."""
    
    @pytest.fixture
    def model(self):
        """Create a test model instance."""
        return CausalWorldModel(
            state_dim=32,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
            device="cpu"
        )
    
    @pytest.fixture
    def sample_world_state(self):
        """Create a sample world state for testing."""
        return WorldState(
            vehicle_positions=torch.randn(2, 3),
            vehicle_velocities=torch.randn(2, 2),
            traffic_lights=torch.randn(1, 3),
            road_geometry=torch.randn(3, 4),
            ego_state=torch.randn(6)
        )
    
    def test_model_initialization(self, model):
        """Test model initialization and architecture."""
        # Check that all components are created
        assert hasattr(model, 'state_encoder')
        assert hasattr(model, 'causal_attention')
        assert hasattr(model, 'dynamics_predictor')
        assert hasattr(model, 'causal_graph_learner')
        assert hasattr(model, 'intervention_predictor')
        
        # Check dimensions
        assert model.state_dim == 32
        assert model.hidden_dim == 64
        assert model.device == "cpu"
    
    def test_encode_state(self, model, sample_world_state):
        """Test state encoding functionality."""
        # Encode state
        encoded_state = model.encode_state(sample_world_state)
        
        # Check output shape
        assert encoded_state.shape == (model.state_dim,)
        assert encoded_state.dtype == torch.float32
        
        # Check that encoding is deterministic
        encoded_state2 = model.encode_state(sample_world_state)
        assert torch.allclose(encoded_state, encoded_state2)
    
    def test_encode_state_empty_vehicles(self, model):
        """Test state encoding with no other vehicles."""
        world_state = WorldState(
            vehicle_positions=torch.empty(0, 3),
            vehicle_velocities=torch.empty(0, 2),
            traffic_lights=torch.randn(1, 3),
            road_geometry=torch.randn(3, 4),
            ego_state=torch.randn(6)
        )
        
        encoded_state = model.encode_state(world_state)
        assert encoded_state.shape == (model.state_dim,)
    
    def test_learn_causal_graph(self, model):
        """Test causal graph learning."""
        # Create test states
        states = torch.randn(4, model.state_dim)  # batch of 4 states
        
        # Learn causal graph
        causal_graph = model.learn_causal_graph(states)
        
        # Check output shape
        assert causal_graph.shape == (4, model.state_dim, model.state_dim)
        
        # Check that weights sum to 1 (softmax normalization)
        assert torch.allclose(causal_graph.sum(dim=-1), torch.ones(4, model.state_dim))
    
    def test_predict_with_intervention(self, model):
        """Test intervention prediction."""
        # Create test data
        current_state = torch.randn(model.state_dim)
        intervention = torch.randn(model.state_dim)
        intervention_mask = torch.bernoulli(torch.ones(model.state_dim) * 0.3)
        
        # Predict intervention effect
        effect = model.predict_with_intervention(current_state, intervention, intervention_mask)
        
        # Check output shape
        assert effect.shape == (model.state_dim,)
    
    def test_generate_counterfactual(self, model):
        """Test counterfactual generation."""
        # Create test data
        factual_state = torch.randn(model.state_dim)
        factual_outcome = torch.randn(6)  # ego state dimension
        intervention = torch.randn(model.state_dim)
        intervention_mask = torch.bernoulli(torch.ones(model.state_dim) * 0.3)
        
        # Generate counterfactual
        counterfactual = model.generate_counterfactual(
            factual_state, factual_outcome, intervention, intervention_mask
        )
        
        # Check output shape
        assert counterfactual.shape == (6,)  # ego state dimension
    
    def test_forward_pass_no_intervention(self, model, sample_world_state):
        """Test forward pass without intervention."""
        # Forward pass
        outputs = model(sample_world_state)
        
        # Check output structure
        assert 'state_encoding' in outputs
        assert 'causal_graph' in outputs
        assert 'next_state_prediction' in outputs
        assert 'intervention_effect' in outputs
        
        # Check shapes
        assert outputs['state_encoding'].shape == (model.state_dim,)
        assert outputs['causal_graph'].shape == (1, model.state_dim, model.state_dim)
        assert outputs['next_state_prediction'].shape == (6,)
        assert outputs['intervention_effect'].shape == (model.state_dim,)
    
    def test_forward_pass_with_intervention(self, model, sample_world_state):
        """Test forward pass with intervention."""
        # Create intervention
        intervention = torch.randn(model.state_dim)
        intervention_mask = torch.bernoulli(torch.ones(model.state_dim) * 0.3)
        
        # Forward pass with intervention
        outputs = model(sample_world_state, intervention, intervention_mask)
        
        # Check that intervention effect is not zero
        assert not torch.allclose(outputs['intervention_effect'], torch.zeros_like(intervention))
    
    def test_model_device_handling(self):
        """Test model device handling."""
        if torch.cuda.is_available():
            # Test CUDA device
            model_cuda = CausalWorldModel(device="cuda")
            assert next(model_cuda.parameters()).device.type == "cuda"
        
        # Test CPU device
        model_cpu = CausalWorldModel(device="cpu")
        assert next(model_cpu.parameters()).device.type == "cpu"


class TestSyntheticInterventionGenerator:
    """Test the SyntheticInterventionGenerator class."""
    
    @pytest.fixture
    def generator(self):
        """Create a test generator instance."""
        return SyntheticInterventionGenerator(device="cpu")
    
    @pytest.fixture
    def sample_world_state(self):
        """Create a sample world state for testing."""
        return WorldState(
            vehicle_positions=torch.randn(2, 3),
            vehicle_velocities=torch.randn(2, 2),
            traffic_lights=torch.randn(1, 3),
            road_geometry=torch.randn(3, 4),
            ego_state=torch.randn(6)
        )
    
    def test_generator_initialization(self, generator):
        """Test generator initialization."""
        assert generator.device == "cpu"
    
    def test_generate_traffic_intervention(self, generator, sample_world_state):
        """Test traffic intervention generation."""
        # Test different intervention types
        intervention_types = ["vehicle_speed_change", "traffic_light_change"]
        
        for intervention_type in intervention_types:
            intervention, intervention_mask = generator.generate_traffic_intervention(
                sample_world_state, intervention_type
            )
            
            # Check shapes
            assert intervention.shape == (64,)  # state_dim
            assert intervention_mask.shape == (64,)
            
            # Check that mask is binary
            assert torch.all((intervention_mask == 0) | (intervention_mask == 1))
    
    def test_generate_traffic_intervention_random(self, generator, sample_world_state):
        """Test random traffic intervention generation."""
        intervention, intervention_mask = generator.generate_traffic_intervention(
            sample_world_state, "random_type"
        )
        
        # Check shapes
        assert intervention.shape == (64,)
        assert intervention_mask.shape == (64,)
    
    def test_generate_safety_intervention(self, generator, sample_world_state):
        """Test safety intervention generation."""
        # Test different safety scenarios
        safety_scenarios = ["emergency_brake", "sudden_obstacle"]
        
        for safety_scenario in safety_scenarios:
            intervention, intervention_mask = generator.generate_safety_intervention(
                sample_world_state, safety_scenario
            )
            
            # Check shapes
            assert intervention.shape == (64,)
            assert intervention_mask.shape == (64,)
    
    def test_generate_safety_intervention_generic(self, generator, sample_world_state):
        """Test generic safety intervention generation."""
        intervention, intervention_mask = generator.generate_safety_intervention(
            sample_world_state, "generic_scenario"
        )
        
        # Check shapes
        assert intervention.shape == (64,)
        assert intervention_mask.shape == (64,)


class TestCausalWorldModelTrainer:
    """Test the CausalWorldModelTrainer class."""
    
    @pytest.fixture
    def model(self):
        """Create a test model instance."""
        return CausalWorldModel(state_dim=32, device="cpu")
    
    @pytest.fixture
    def trainer(self, model):
        """Create a test trainer instance."""
        return CausalWorldModelTrainer(model, learning_rate=1e-4, device="cpu")
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        world_states = [
            WorldState(
                vehicle_positions=torch.randn(2, 3),
                vehicle_velocities=torch.randn(2, 2),
                traffic_lights=torch.randn(1, 3),
                road_geometry=torch.randn(3, 4),
                ego_state=torch.randn(6)
            ) for _ in range(4)
        ]
        targets = torch.randn(4, 6)  # batch of 4, ego state dimension
        
        return world_states, targets
    
    def test_trainer_initialization(self, trainer, model):
        """Test trainer initialization."""
        assert trainer.model == model
        assert trainer.device == "cpu"
        assert hasattr(trainer, 'optimizer')
        assert hasattr(trainer, 'intervention_generator')
    
    def test_compute_causal_loss(self, trainer):
        """Test causal loss computation."""
        # Create test data
        factual_outcome = torch.randn(6)
        counterfactual_outcome = torch.randn(6)
        intervention = torch.randn(32)
        
        # Compute loss
        loss = trainer.compute_causal_loss(factual_outcome, counterfactual_outcome, intervention)
        
        # Check that loss is a scalar tensor
        assert loss.shape == ()
        assert loss.dtype == torch.float32
        assert loss.item() >= 0  # Loss should be non-negative
    
    def test_train_step_no_interventions(self, trainer, sample_data):
        """Test training step without interventions."""
        world_states, targets = sample_data
        
        # Training step
        losses = trainer.train_step(world_states, targets, use_interventions=False)
        
        # Check loss structure
        assert 'total_loss' in losses
        assert 'prediction_loss' in losses
        assert 'causal_loss' in losses
        
        # Check that losses are non-negative
        for loss_name, loss_value in losses.items():
            assert loss_value >= 0
    
    def test_train_step_with_interventions(self, trainer, sample_data):
        """Test training step with interventions."""
        world_states, targets = sample_data
        
        # Training step with interventions
        losses = trainer.train_step(world_states, targets, use_interventions=True)
        
        # Check loss structure
        assert 'total_loss' in losses
        assert 'prediction_loss' in losses
        assert 'causal_loss' in losses
        
        # Check that causal loss is computed
        assert losses['causal_loss'] > 0
    
    def test_validate(self, trainer, sample_data):
        """Test validation."""
        world_states, targets = sample_data
        
        # Validation
        metrics = trainer.validate(world_states, targets)
        
        # Check metrics structure
        assert 'validation_loss' in metrics
        assert 'causal_consistency' in metrics
        
        # Check that metrics are reasonable
        assert metrics['validation_loss'] >= 0
        assert metrics['causal_consistency'] >= 0


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_model_with_zero_dimensions(self):
        """Test model with zero dimensions."""
        with pytest.raises(ValueError):
            CausalWorldModel(state_dim=0)
    
    def test_model_with_negative_dimensions(self):
        """Test model with negative dimensions."""
        with pytest.raises(ValueError):
            CausalWorldModel(state_dim=-1)
    
    def test_intervention_with_mismatched_shapes(self):
        """Test intervention with mismatched shapes."""
        model = CausalWorldModel(state_dim=32, device="cpu")
        current_state = torch.randn(32)
        intervention = torch.randn(16)  # Wrong size
        intervention_mask = torch.ones(32)
        
        with pytest.raises(RuntimeError):
            model.predict_with_intervention(current_state, intervention, intervention_mask)
    
    def test_empty_batch(self):
        """Test with empty batch."""
        model = CausalWorldModel(state_dim=32, device="cpu")
        states = torch.empty(0, 32)
        
        with pytest.raises(RuntimeError):
            model.learn_causal_graph(states)


class TestPerformance:
    """Test performance characteristics."""
    
    def test_model_memory_usage(self):
        """Test model memory usage."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss
        
        # Create model
        model = CausalWorldModel(state_dim=64, hidden_dim=128)
        
        memory_after = process.memory_info().rss
        memory_increase = memory_after - memory_before
        
        # Check that memory increase is reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024
    
    def test_forward_pass_speed(self):
        """Test forward pass speed."""
        import time
        
        model = CausalWorldModel(state_dim=32, device="cpu")
        world_state = WorldState(
            vehicle_positions=torch.randn(5, 3),
            vehicle_velocities=torch.randn(5, 2),
            traffic_lights=torch.randn(2, 3),
            road_geometry=torch.randn(10, 4),
            ego_state=torch.randn(6)
        )
        
        # Warm up
        for _ in range(10):
            model(world_state)
        
        # Time forward pass
        start_time = time.time()
        for _ in range(100):
            model(world_state)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        
        # Check that forward pass is reasonably fast (less than 10ms)
        assert avg_time < 0.01


if __name__ == "__main__":
    pytest.main([__file__]) 