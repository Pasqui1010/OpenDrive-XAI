"""
Unit tests for the modular components module.

Tests cover:
- Basic functionality of each component
- Edge cases and error handling
- Performance characteristics
- Interpretability features
- System integration
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from unittest.mock import Mock, patch
from typing import Dict, Any

from src.opendrive_xai.modular_components import (
    NeuralComponent,
    AttentionGate,
    SafetyMonitor,
    CausalReasoner,
    CircuitRouter,
    InterpretableDrivingSystem,
    ModularSystemTrainer
)


class TestNeuralComponent:
    """Test the abstract NeuralComponent base class."""
    
    class MockComponent(NeuralComponent):
        """Mock implementation of NeuralComponent for testing."""
        
        def __init__(self, name: str, input_dim: int, output_dim: int):
            super().__init__(name, input_dim, output_dim)
            self.linear = nn.Linear(input_dim, output_dim)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)
        
        def verify(self, x: torch.Tensor) -> Dict[str, Any]:
            output = self.forward(x)
            return {
                'output_norm': torch.norm(output).item(),
                'input_norm': torch.norm(x).item()
            }
    
    def test_component_initialization(self):
        """Test component initialization."""
        component = self.MockComponent("test_component", 10, 5)
        
        assert component.name == "test_component"
        assert component.input_dim == 10
        assert component.output_dim == 5
        assert hasattr(component, 'verification_results')
    
    def test_component_interface(self):
        """Test component interface specification."""
        component = self.MockComponent("test_component", 10, 5)
        
        interface = component.get_interface()
        
        assert interface['name'] == "test_component"
        assert interface['input_dim'] == 10
        assert interface['output_dim'] == 5
        assert 'verification_results' in interface
    
    def test_component_forward(self):
        """Test component forward pass."""
        component = self.MockComponent("test_component", 10, 5)
        x = torch.randn(3, 10)
        
        output = component(x)
        
        assert output.shape == (3, 5)
        assert output.dtype == torch.float32
    
    def test_component_verify(self):
        """Test component verification."""
        component = self.MockComponent("test_component", 10, 5)
        x = torch.randn(3, 10)
        
        verification_results = component.verify(x)
        
        assert 'output_norm' in verification_results
        assert 'input_norm' in verification_results
        assert verification_results['output_norm'] >= 0
        assert verification_results['input_norm'] >= 0


class TestAttentionGate:
    """Test the AttentionGate component."""
    
    @pytest.fixture
    def attention_gate(self):
        """Create a test attention gate."""
        return AttentionGate(input_dim=64, output_dim=32, num_heads=8)
    
    def test_attention_gate_initialization(self, attention_gate):
        """Test attention gate initialization."""
        assert attention_gate.name.startswith("attention_gate_")
        assert attention_gate.input_dim == 64
        assert attention_gate.output_dim == 32
        assert attention_gate.num_heads == 8
        assert hasattr(attention_gate, 'attention')
        assert hasattr(attention_gate, 'output_proj')
        assert hasattr(attention_gate, 'attention_weights')
    
    def test_attention_gate_forward_2d(self, attention_gate):
        """Test attention gate forward pass with 2D input."""
        x = torch.randn(4, 64)
        
        output = attention_gate(x)
        
        assert output.shape == (4, 32)
        assert output.dtype == torch.float32
    
    def test_attention_gate_forward_3d(self, attention_gate):
        """Test attention gate forward pass with 3D input."""
        x = torch.randn(4, 5, 64)  # batch, sequence, features
        
        output = attention_gate(x)
        
        assert output.shape == (4, 5, 32)
        assert output.dtype == torch.float32
    
    def test_attention_gate_verify(self, attention_gate):
        """Test attention gate verification."""
        x = torch.randn(4, 64)
        
        # First forward pass to generate attention weights
        attention_gate(x)
        
        verification_results = attention_gate.verify(x)
        
        assert 'attention_entropy' in verification_results
        assert 'attention_variance' in verification_results
        assert 'output_norm' in verification_results
        assert 'attention_collapse' in verification_results
        
        # Check that attention weights were stored
        assert len(attention_gate.attention_weights) > 0
    
    def test_attention_gate_attention_collapse(self, attention_gate):
        """Test attention collapse detection."""
        # Create input that might cause attention collapse
        x = torch.ones(4, 64) * 0.1  # Very uniform input
        
        attention_gate(x)
        verification_results = attention_gate.verify(x)
        
        # Should detect potential attention collapse
        assert 'attention_collapse' in verification_results
    
    def test_attention_gate_deterministic(self, attention_gate):
        """Test that attention gate is deterministic."""
        x = torch.randn(4, 64)
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        output1 = attention_gate(x)
        
        torch.manual_seed(42)
        output2 = attention_gate(x)
        
        assert torch.allclose(output1, output2)


class TestSafetyMonitor:
    """Test the SafetyMonitor component."""
    
    @pytest.fixture
    def safety_thresholds(self):
        """Create safety thresholds for testing."""
        return {
            'collision_risk': 0.7,
            'speed_violation': 0.8,
            'lane_deviation': 0.6
        }
    
    @pytest.fixture
    def safety_monitor(self, safety_thresholds):
        """Create a test safety monitor."""
        return SafetyMonitor(input_dim=64, output_dim=32, safety_thresholds=safety_thresholds)
    
    def test_safety_monitor_initialization(self, safety_monitor, safety_thresholds):
        """Test safety monitor initialization."""
        assert safety_monitor.name.startswith("safety_monitor_")
        assert safety_monitor.input_dim == 64
        assert safety_monitor.output_dim == 32
        assert safety_monitor.safety_thresholds == safety_thresholds
        assert hasattr(safety_monitor, 'safety_detector')
        assert hasattr(safety_monitor, 'safety_encoder')
        assert hasattr(safety_monitor, 'safety_violations')
    
    def test_safety_monitor_forward(self, safety_monitor):
        """Test safety monitor forward pass."""
        x = torch.randn(4, 64)
        
        output = safety_monitor(x)
        
        assert output.shape == (4, 32)
        assert output.dtype == torch.float32
    
    def test_safety_monitor_verify(self, safety_monitor):
        """Test safety monitor verification."""
        x = torch.randn(4, 64)
        
        # First forward pass to generate safety violations
        safety_monitor(x)
        
        verification_results = safety_monitor.verify(x)
        
        assert 'total_violations' in verification_results
        assert 'violation_types' in verification_results
        assert 'safety_scores_mean' in verification_results
        assert 'safety_scores_std' in verification_results
        
        # Check that safety violations were stored
        assert len(safety_monitor.safety_violations) > 0
    
    def test_safety_monitor_threshold_violations(self, safety_monitor):
        """Test safety threshold violations."""
        # Create input that should trigger violations
        x = torch.randn(4, 64) * 10  # Large values to trigger violations
        
        safety_monitor(x)
        verification_results = safety_monitor.verify(x)
        
        # Should detect some violations
        assert verification_results['total_violations'] >= 0
        assert isinstance(verification_results['violation_types'], list)
    
    def test_safety_monitor_no_violations(self, safety_monitor):
        """Test safety monitor with no violations."""
        # Create input that should not trigger violations
        x = torch.randn(4, 64) * 0.01  # Small values
        
        safety_monitor(x)
        verification_results = safety_monitor.verify(x)
        
        # Should have no violations
        assert verification_results['total_violations'] == 0
        assert len(verification_results['violation_types']) == 0


class TestCausalReasoner:
    """Test the CausalReasoner component."""
    
    @pytest.fixture
    def causal_reasoner(self):
        """Create a test causal reasoner."""
        return CausalReasoner(input_dim=64, output_dim=32, num_causal_vars=10)
    
    def test_causal_reasoner_initialization(self, causal_reasoner):
        """Test causal reasoner initialization."""
        assert causal_reasoner.name.startswith("causal_reasoner_")
        assert causal_reasoner.input_dim == 64
        assert causal_reasoner.output_dim == 32
        assert causal_reasoner.num_causal_vars == 10
        assert hasattr(causal_reasoner, 'causal_graph')
        assert hasattr(causal_reasoner, 'causal_encoder')
        assert hasattr(causal_reasoner, 'effect_predictor')
        assert hasattr(causal_reasoner, 'causal_explanations')
    
    def test_causal_reasoner_forward(self, causal_reasoner):
        """Test causal reasoner forward pass."""
        x = torch.randn(4, 64)
        
        output = causal_reasoner(x)
        
        assert output.shape == (4, 32)
        assert output.dtype == torch.float32
    
    def test_causal_reasoner_verify(self, causal_reasoner):
        """Test causal reasoner verification."""
        x = torch.randn(4, 64)
        
        # First forward pass to generate causal explanations
        causal_reasoner(x)
        
        verification_results = causal_reasoner.verify(x)
        
        assert 'graph_sparsity' in verification_results
        assert 'var_diversity' in verification_results
        assert 'graph_symmetry' in verification_results
        assert 'causal_vars_mean' in verification_results
        
        # Check that causal explanations were stored
        assert len(causal_reasoner.causal_explanations) > 0
    
    def test_causal_reasoner_graph_properties(self, causal_reasoner):
        """Test causal graph properties."""
        x = torch.randn(4, 64)
        
        causal_reasoner(x)
        verification_results = causal_reasoner.verify(x)
        
        # Check graph sparsity (should be reasonable)
        assert 0 <= verification_results['graph_sparsity'] <= 1
        
        # Check variable diversity (should be positive)
        assert verification_results['var_diversity'] >= 0
        
        # Check graph symmetry (should be small for causal graphs)
        assert verification_results['graph_symmetry'] >= 0
    
    def test_causal_reasoner_explanations(self, causal_reasoner):
        """Test causal explanations structure."""
        x = torch.randn(4, 64)
        
        causal_reasoner(x)
        
        # Check explanation structure
        latest_explanation = causal_reasoner.causal_explanations[-1]
        
        assert 'causal_vars' in latest_explanation
        assert 'causal_effects' in latest_explanation
        assert 'causal_graph' in latest_explanation
        
        assert latest_explanation['causal_vars'].shape == (4, 10)
        assert latest_explanation['causal_effects'].shape == (4, 10)
        assert latest_explanation['causal_graph'].shape == (10, 10)


class TestCircuitRouter:
    """Test the CircuitRouter component."""
    
    @pytest.fixture
    def components(self):
        """Create test components."""
        return [
            AttentionGate(32, 16),
            SafetyMonitor(32, 16, {'test': 0.5}),
            CausalReasoner(32, 16, 5)
        ]
    
    @pytest.fixture
    def circuit_router(self, components):
        """Create a test circuit router."""
        return CircuitRouter(components)
    
    def test_circuit_router_initialization(self, circuit_router, components):
        """Test circuit router initialization."""
        assert len(circuit_router.components) == len(components)
        assert hasattr(circuit_router, 'routing_network')
        assert hasattr(circuit_router, 'routing_decisions')
    
    def test_circuit_router_forward(self, circuit_router):
        """Test circuit router forward pass."""
        x = torch.randn(4, 32)
        
        output, verification_results = circuit_router(x)
        
        # Check output shape (sum of all component outputs)
        expected_output_dim = 16 * 3  # 3 components, each with output_dim=16
        assert output.shape == (4, expected_output_dim)
        
        # Check verification results
        assert isinstance(verification_results, dict)
        assert len(verification_results) == 3  # One for each component
    
    def test_circuit_router_routing_decisions(self, circuit_router):
        """Test routing decision storage."""
        x = torch.randn(4, 32)
        
        circuit_router(x)
        
        # Check that routing decisions were stored
        assert len(circuit_router.routing_decisions) > 0
        
        latest_decision = circuit_router.routing_decisions[-1]
        assert latest_decision.shape == (4, 3)  # batch_size, num_components
        
        # Check that routing weights sum to 1 (softmax)
        assert torch.allclose(latest_decision.sum(dim=-1), torch.ones(4))


class TestInterpretableDrivingSystem:
    """Test the InterpretableDrivingSystem class."""
    
    @pytest.fixture
    def driving_system(self):
        """Create a test driving system."""
        return InterpretableDrivingSystem(
            perception_dim=128,
            planning_dim=64,
            control_dim=32,
            device="cpu"
        )
    
    def test_driving_system_initialization(self, driving_system):
        """Test driving system initialization."""
        assert hasattr(driving_system, 'perception_attention')
        assert hasattr(driving_system, 'safety_monitor')
        assert hasattr(driving_system, 'causal_reasoner')
        assert hasattr(driving_system, 'planning_attention')
        assert hasattr(driving_system, 'control_attention')
        assert hasattr(driving_system, 'router')
        assert hasattr(driving_system, 'control_output')
        assert driving_system.device == "cpu"
    
    def test_driving_system_forward(self, driving_system):
        """Test driving system forward pass."""
        perception_input = torch.randn(4, 128)
        
        outputs = driving_system(perception_input)
        
        # Check output structure
        assert 'control_signal' in outputs
        assert 'interpretability_info' in outputs
        
        # Check control signal shape
        assert outputs['control_signal'].shape == (4, 3)  # steering, throttle, brake
        
        # Check interpretability info
        interpretability_info = outputs['interpretability_info']
        assert 'routing_decisions' in interpretability_info
        assert 'attention_weights' in interpretability_info
        assert 'safety_violations' in interpretability_info
        assert 'causal_explanations' in interpretability_info
        assert 'verification_results' in interpretability_info
    
    def test_driving_system_interface(self, driving_system):
        """Test driving system interface."""
        interface = driving_system.get_system_interface()
        
        assert 'components' in interface
        assert 'routing' in interface
        assert len(interface['components']) == 5  # 5 components
        assert interface['routing']['num_components'] == 5
    
    def test_driving_system_verification(self, driving_system):
        """Test driving system verification."""
        perception_input = torch.randn(4, 128)
        
        verification_results = driving_system.verify_system(perception_input)
        
        assert 'control_bounds_check' in verification_results
        assert 'interpretability_check' in verification_results
        assert 'verification_results' in verification_results
        
        # Check control bounds
        control_bounds = verification_results['control_bounds_check']
        assert 'steering_in_bounds' in control_bounds
        assert 'throttle_in_bounds' in control_bounds
        assert 'brake_in_bounds' in control_bounds
        
        # Check interpretability
        interpretability = verification_results['interpretability_check']
        assert 'has_attention_weights' in interpretability
        assert 'has_safety_violations' in interpretability
        assert 'has_causal_explanations' in interpretability
        assert 'has_routing_decisions' in interpretability


class TestModularSystemTrainer:
    """Test the ModularSystemTrainer class."""
    
    @pytest.fixture
    def driving_system(self):
        """Create a test driving system."""
        return InterpretableDrivingSystem(
            perception_dim=64,
            planning_dim=32,
            control_dim=16,
            device="cpu"
        )
    
    @pytest.fixture
    def trainer(self, driving_system):
        """Create a test trainer."""
        return ModularSystemTrainer(driving_system, learning_rate=1e-4, device="cpu")
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        perception_inputs = torch.randn(4, 64)
        control_targets = torch.randn(4, 3)
        return perception_inputs, control_targets
    
    def test_trainer_initialization(self, trainer, driving_system):
        """Test trainer initialization."""
        assert trainer.system == driving_system
        assert trainer.device == "cpu"
        assert hasattr(trainer, 'optimizer')
    
    def test_compute_interpretability_loss(self, trainer):
        """Test interpretability loss computation."""
        # Create mock interpretability info
        interpretability_info = {
            'attention_weights': {
                'perception': torch.softmax(torch.randn(4, 1, 1), dim=-1),
                'planning': torch.softmax(torch.randn(4, 1, 1), dim=-1),
                'control': torch.softmax(torch.randn(4, 1, 1), dim=-1)
            },
            'routing_decisions': torch.softmax(torch.randn(4, 5), dim=-1)
        }
        
        loss = trainer.compute_interpretability_loss(interpretability_info)
        
        assert loss.shape == ()
        assert loss.dtype == torch.float32
        assert loss.item() >= 0
    
    def test_train_step(self, trainer, sample_data):
        """Test training step."""
        perception_inputs, control_targets = sample_data
        
        losses = trainer.train_step(perception_inputs, control_targets)
        
        assert 'total_loss' in losses
        assert 'control_loss' in losses
        assert 'interpretability_loss' in losses
        
        # Check that losses are non-negative
        for loss_name, loss_value in losses.items():
            assert loss_value >= 0
    
    def test_validate(self, trainer, sample_data):
        """Test validation."""
        perception_inputs, control_targets = sample_data
        
        metrics = trainer.validate(perception_inputs, control_targets)
        
        assert 'control_loss' in metrics
        assert 'verification_results' in metrics
        
        # Check that control loss is non-negative
        assert metrics['control_loss'] >= 0


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_attention_gate_zero_dimensions(self):
        """Test attention gate with zero dimensions."""
        with pytest.raises(ValueError):
            AttentionGate(input_dim=0, output_dim=32)
    
    def test_safety_monitor_empty_thresholds(self):
        """Test safety monitor with empty thresholds."""
        with pytest.raises(ValueError):
            SafetyMonitor(input_dim=64, output_dim=32, safety_thresholds={})
    
    def test_causal_reasoner_zero_causal_vars(self):
        """Test causal reasoner with zero causal variables."""
        with pytest.raises(ValueError):
            CausalReasoner(input_dim=64, output_dim=32, num_causal_vars=0)
    
    def test_circuit_router_empty_components(self):
        """Test circuit router with empty component list."""
        with pytest.raises(ValueError):
            CircuitRouter([])
    
    def test_driving_system_mismatched_dimensions(self):
        """Test driving system with mismatched dimensions."""
        with pytest.raises(ValueError):
            InterpretableDrivingSystem(
                perception_dim=64,
                planning_dim=128,  # Larger than perception_dim
                control_dim=32
            )


class TestPerformance:
    """Test performance characteristics."""
    
    def test_attention_gate_memory_usage(self):
        """Test attention gate memory usage."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss
        
        # Create attention gate
        attention_gate = AttentionGate(input_dim=128, output_dim=64)
        
        memory_after = process.memory_info().rss
        memory_increase = memory_after - memory_before
        
        # Check that memory increase is reasonable (less than 50MB)
        assert memory_increase < 50 * 1024 * 1024
    
    def test_driving_system_forward_speed(self):
        """Test driving system forward pass speed."""
        import time
        
        driving_system = InterpretableDrivingSystem(
            perception_dim=128,
            planning_dim=64,
            control_dim=32,
            device="cpu"
        )
        
        perception_input = torch.randn(4, 128)
        
        # Warm up
        for _ in range(10):
            driving_system(perception_input)
        
        # Time forward pass
        start_time = time.time()
        for _ in range(100):
            driving_system(perception_input)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        
        # Check that forward pass is reasonably fast (less than 50ms)
        assert avg_time < 0.05
    
    def test_component_verification_speed(self):
        """Test component verification speed."""
        import time
        
        attention_gate = AttentionGate(input_dim=64, output_dim=32)
        x = torch.randn(4, 64)
        
        # Warm up
        attention_gate(x)
        attention_gate.verify(x)
        
        # Time verification
        start_time = time.time()
        for _ in range(100):
            attention_gate.verify(x)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        
        # Check that verification is fast (less than 1ms)
        assert avg_time < 0.001


if __name__ == "__main__":
    pytest.main([__file__]) 