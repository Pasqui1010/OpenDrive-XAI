"""
Modular Neural Components for Interpretable Autonomous Driving

This module implements a circuit-style neural architecture with explicit interfaces
and verifiable components for interpretable decision-making.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
from collections import deque

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


class NeuralComponent(ABC, nn.Module):
    """
    Abstract base class for modular neural components.
    Each component has explicit inputs, outputs, and verification methods.
    """
    
    def __init__(self, name: str, input_dim: int, output_dim: int, max_history_size: int = 100):
        super().__init__()
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_history_size = max_history_size
        self.verification_results = {}
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the component."""
        pass
    
    @abstractmethod
    def verify(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Verify the component's behavior and return verification metrics.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of verification results
        """
        pass
    
    def get_interface(self) -> Dict[str, Any]:
        """Get the component's interface specification."""
        return {
            'name': self.name,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'verification_results': self.verification_results
        }
    
    def _add_to_history(self, history_list: List, item: Any):
        """
        Add item to history with size limit to prevent memory leaks.
        
        Args:
            history_list: List to add item to
            item: Item to add
        """
        history_list.append(item)
        if len(history_list) > self.max_history_size:
            history_list.pop(0)  # Remove oldest item


class AttentionGate(NeuralComponent):
    """
    Attention gate that controls information flow between components.
    Provides interpretable attention weights.
    """
    
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 8, max_history_size: int = 50):
        super().__init__(f"attention_gate_{input_dim}_{output_dim}", input_dim, output_dim, max_history_size)
        self.num_heads = num_heads
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(input_dim, output_dim)
        
        # Attention weight history for interpretability (with size limit)
        self.attention_weights = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention mechanism."""
        # Validate input
        x = validate_tensor(x, "input", expected_dtype=torch.float32)
        
        # Add sequence dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Apply attention
        attended, attention_weights = self.attention(x, x, x)
        
        # Store attention weights for interpretability (with size limit)
        self._add_to_history(self.attention_weights, attention_weights.detach().cpu())
        
        # Project to output dimension
        output = self.output_proj(attended)
        
        return output.squeeze(1) if output.dim() == 3 else output
    
    def verify(self, x: torch.Tensor) -> Dict[str, Any]:
        """Verify attention behavior."""
        # Validate input
        x = validate_tensor(x, "input", expected_dtype=torch.float32)
        
        with torch.no_grad():
            # Check attention weight distribution
            if len(self.attention_weights) > 0:
                latest_weights = self.attention_weights[-1]
                attention_entropy = -torch.sum(
                    latest_weights * torch.log(latest_weights + 1e-8), dim=-1
                ).mean()
                
                # Check for attention collapse (all weights equal)
                # Add small epsilon to handle edge cases with small tensors
                if latest_weights.numel() > 1:
                    attention_variance = torch.var(latest_weights, dim=-1, correction=0).mean()
                else:
                    attention_variance = torch.tensor(0.0)
            else:
                attention_entropy = torch.tensor(0.0)
                attention_variance = torch.tensor(0.0)
            
            # Check output range
            output = self.forward(x)
            output_norm = torch.norm(output, dim=-1).mean()
            
            return {
                'attention_entropy': attention_entropy.item(),
                'attention_variance': attention_variance.item(),
                'output_norm': output_norm.item(),
                'attention_collapse': attention_variance.item() < 0.01,
                'history_size': len(self.attention_weights)
            }
    
    def clear_history(self):
        """Clear attention weight history to free memory."""
        self.attention_weights.clear()
        logger.info(f"Cleared attention weight history for {self.name}")


class SafetyMonitor(NeuralComponent):
    """
    Safety monitoring component that checks for dangerous conditions.
    Provides explicit safety signals.
    """
    
    def __init__(self, input_dim: int, output_dim: int, safety_thresholds: Dict[str, float], max_history_size: int = 100):
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        super().__init__(f"safety_monitor_{input_dim}_{output_dim}", input_dim, output_dim, max_history_size)
        
        # Validate safety thresholds
        if not isinstance(safety_thresholds, dict) or len(safety_thresholds) == 0:
            raise ValueError("safety_thresholds must be a non-empty dictionary")
        
        for key, value in safety_thresholds.items():
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"safety_thresholds[{key}] must be a non-negative number")
        
        self.safety_thresholds = safety_thresholds
        
        # Safety detection network
        self.safety_detector = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, len(safety_thresholds))
        )
        
        # Safety signal encoder
        self.safety_encoder = nn.Linear(len(safety_thresholds), output_dim)
        
        # Safety violation history (with size limit)
        self.safety_violations = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with safety monitoring."""
        # Validate input
        x = validate_tensor(x, "input", expected_dtype=torch.float32)
        
        # Detect safety violations
        safety_scores = self.safety_detector(x)
        
        # Check against thresholds
        safety_violations = torch.zeros_like(safety_scores)
        for i, (safety_type, threshold) in enumerate(self.safety_thresholds.items()):
            safety_violations[:, i] = (safety_scores[:, i] > threshold).float()
        
        # Store violations for monitoring (with size limit)
        self._add_to_history(self.safety_violations, safety_violations.detach().cpu())
        
        # Encode safety signals
        safety_signals = self.safety_encoder(safety_violations)
        
        return safety_signals
    
    def verify(self, x: torch.Tensor) -> Dict[str, Any]:
        """Verify safety monitoring behavior."""
        # Validate input
        x = validate_tensor(x, "input", expected_dtype=torch.float32)
        
        with torch.no_grad():
            safety_scores = self.safety_detector(x)
            
            # Check for safety violations
            total_violations = 0
            violation_types = []
            
            for i, (safety_type, threshold) in enumerate(self.safety_thresholds.items()):
                violations = (safety_scores[:, i] > threshold).sum().item()
                total_violations += violations
                if violations > 0:
                    violation_types.append(safety_type)
            
            return {
                'total_violations': total_violations,
                'violation_types': violation_types,
                'safety_scores_mean': safety_scores.mean().item(),
                'safety_scores_std': safety_scores.std().item(),
                'history_size': len(self.safety_violations)
            }
    
    def clear_history(self):
        """Clear safety violation history to free memory."""
        self.safety_violations.clear()
        logger.info(f"Cleared safety violation history for {self.name}")


class CausalReasoner(NeuralComponent):
    """
    Causal reasoning component that infers cause-effect relationships.
    Provides interpretable causal explanations.
    """
    
    def __init__(self, input_dim: int, output_dim: int, num_causal_vars: int = 10, max_history_size: int = 50):
        super().__init__(f"causal_reasoner_{input_dim}_{output_dim}", input_dim, output_dim, max_history_size)
        
        # Validate parameters
        if num_causal_vars <= 0:
            raise ValueError("num_causal_vars must be positive")
        
        self.num_causal_vars = num_causal_vars
        
        # Causal graph learner
        self.causal_graph = nn.Parameter(torch.randn(num_causal_vars, num_causal_vars))
        
        # Causal variable encoder
        self.causal_encoder = nn.Linear(input_dim, num_causal_vars)
        
        # Causal effect predictor
        self.effect_predictor = nn.Sequential(
            nn.Linear(num_causal_vars, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
        # Causal explanation history (with size limit)
        self.causal_explanations = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with causal reasoning."""
        # Validate input
        x = validate_tensor(x, "input", expected_dtype=torch.float32)
        
        # Encode to causal variables
        causal_vars = self.causal_encoder(x)
        
        # Apply causal graph
        causal_effects = torch.matmul(causal_vars, self.causal_graph)
        
        # Predict final effects
        output = self.effect_predictor(causal_effects)
        
        # Store causal explanation (with size limit)
        explanation = {
            'causal_vars': causal_vars.detach().cpu(),
            'causal_effects': causal_effects.detach().cpu(),
            'causal_graph': self.causal_graph.detach().cpu()
        }
        self._add_to_history(self.causal_explanations, explanation)
        
        return output
    
    def verify(self, x: torch.Tensor) -> Dict[str, Any]:
        """Verify causal reasoning behavior."""
        # Validate input
        x = validate_tensor(x, "input", expected_dtype=torch.float32)
        
        with torch.no_grad():
            causal_vars = self.causal_encoder(x)
            
            # Check causal graph sparsity
            graph_sparsity = (self.causal_graph.abs() < 0.1).float().mean()
            
            # Check causal variable diversity
            var_diversity = torch.std(causal_vars, dim=0).mean()
            
            # Check for causal cycles (should be minimal)
            graph_symmetry = torch.norm(self.causal_graph - self.causal_graph.t())
            
            return {
                'graph_sparsity': graph_sparsity.item(),
                'var_diversity': var_diversity.item(),
                'graph_symmetry': graph_symmetry.item(),
                'causal_vars_mean': causal_vars.mean().item(),
                'history_size': len(self.causal_explanations)
            }
    
    def clear_history(self):
        """Clear causal explanation history to free memory."""
        self.causal_explanations.clear()
        logger.info(f"Cleared causal explanation history for {self.name}")


class CircuitRouter(nn.Module):
    """
    Circuit-style router that manages information flow between components.
    Provides explicit routing decisions for interpretability.
    """
    
    def __init__(self, component_list: List[NeuralComponent], max_history_size: int = 100):
        super().__init__()
        
        # Validate component list
        if not isinstance(component_list, list) or len(component_list) == 0:
            raise ValueError("component_list must be a non-empty list")
        
        for i, component in enumerate(component_list):
            if not isinstance(component, NeuralComponent):
                raise ValueError(f"component_list[{i}] must be a NeuralComponent")
        
        self.components = nn.ModuleList(component_list)
        self.max_history_size = max_history_size
        self.routing_decisions = []
        
        # Use the first component's input dimension for routing decisions
        # (assuming all components receive the same perception input)
        routing_input_dim = component_list[0].input_dim
        
        self.routing_network = nn.Sequential(
            nn.Linear(routing_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, len(component_list)),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass with explicit routing."""
        # Validate input
        x = validate_tensor(x, "input", expected_dtype=torch.float32)
        
        # Use only the perception dimension for routing decisions
        perception_dim = self.components[0].input_dim
        routing_input = x[:, :perception_dim]
        
        # Compute routing weights
        routing_weights = self.routing_network(routing_input)
        
        # Store routing decisions (with size limit)
        self._add_to_history(self.routing_decisions, routing_weights.detach().cpu())
        
        # Route to components
        component_outputs = []
        
        for i, component in enumerate(self.components):
            # Use appropriate input for each component
            input_dim = int(component.input_dim)  # Explicit cast to int for linter
            if input_dim <= x.shape[1]:
                # Use the first part of the input if component can handle it
                component_input = x[:, :input_dim]
            else:
                # If component expects more dimensions than available, pad with zeros
                padding_size = input_dim - x.shape[1]
                padding = torch.zeros(x.shape[0], padding_size, device=x.device)
                component_input = torch.cat([x, padding], dim=1)
            
            # Apply routing weight to component input
            routed_input = component_input * routing_weights[:, i:i+1]
            component_output = component(routed_input)
            component_outputs.append(component_output)
        
        # Combine component outputs
        combined_output = torch.cat(component_outputs, dim=-1)
        
        # Collect verification results
        verification_results = {}
        for i, component in enumerate(self.components):
            # Use appropriate input for verification (same logic as forward pass)
            if component.input_dim <= x.shape[1]:
                component_input = x[:, :component.input_dim]
            else:
                padding = torch.zeros(x.shape[0], component.input_dim - x.shape[1], device=x.device)
                component_input = torch.cat([x, padding], dim=1)
            verification_results[component.name] = component.verify(component_input)
        
        return combined_output, verification_results
    
    def _add_to_history(self, history_list: List, item: Any):
        """Add item to routing decisions history with size limit."""
        history_list.append(item)
        if len(history_list) > self.max_history_size:
            history_list.pop(0)  # Remove oldest item
    
    def clear_history(self):
        """Clear routing decision history to free memory."""
        self.routing_decisions.clear()
        logger.info("Cleared routing decision history")


class InterpretableDrivingSystem(nn.Module):
    """
    Complete interpretable driving system using modular components.
    """
    
    def __init__(
        self,
        perception_dim: int = 512,
        planning_dim: int = 256,
        control_dim: int = 64,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_history_size: int = 100
    ):
        super().__init__()
        
        # Validate parameters
        if perception_dim <= 0:
            raise ValueError("perception_dim must be positive")
        if planning_dim <= 0:
            raise ValueError("planning_dim must be positive")
        if control_dim <= 0:
            raise ValueError("control_dim must be positive")
        
        self.device = device
        self.max_history_size = max_history_size
        
        # Define safety thresholds
        safety_thresholds = {
            'collision_risk': 0.7,
            'speed_violation': 0.8,
            'lane_deviation': 0.6,
            'traffic_violation': 0.9
        }
        
        # Create modular components
        self.perception_attention = AttentionGate(perception_dim, perception_dim, max_history_size=max_history_size)
        self.safety_monitor = SafetyMonitor(perception_dim, 64, safety_thresholds, max_history_size=max_history_size)
        self.causal_reasoner = CausalReasoner(perception_dim, planning_dim, max_history_size=max_history_size)
        self.planning_attention = AttentionGate(planning_dim, planning_dim, max_history_size=max_history_size)
        self.control_attention = AttentionGate(planning_dim, control_dim, max_history_size=max_history_size)
        
        # Circuit router
        self.router = CircuitRouter([
            self.perception_attention,
            self.safety_monitor,
            self.causal_reasoner,
            self.planning_attention,
            self.control_attention
        ], max_history_size=max_history_size)
        
        # Final control output
        self.control_output = nn.Linear(control_dim, 3)  # steering, throttle, brake
        
        self.to(device)
    
    def forward(self, perception_input: torch.Tensor) -> Dict[str, Any]:
        """
        Forward pass through the interpretable driving system.
        """
        # Validate input
        perception_input = validate_tensor(perception_input, "perception_input", expected_dtype=torch.float32)
        
        # Route through components
        routed_output, verification_results = self.router(perception_input)
        
        # Only use the output of the last component (control_attention) for control_output
        # The last component's output is at the end of the concatenated outputs
        control_output_dim = self.control_attention.output_dim
        control_attention_output = routed_output[:, -control_output_dim:]
        control_signal = self.control_output(control_attention_output)
        
        # Collect interpretability information
        interpretability_info = {
            'routing_decisions': self.router.routing_decisions[-1] if self.router.routing_decisions else None,
            'attention_weights': {
                'perception': self.perception_attention.attention_weights[-1] if self.perception_attention.attention_weights else None,
                'planning': self.planning_attention.attention_weights[-1] if self.planning_attention.attention_weights else None,
                'control': self.control_attention.attention_weights[-1] if self.control_attention.attention_weights else None
            },
            'safety_violations': self.safety_monitor.safety_violations[-1] if self.safety_monitor.safety_violations else None,
            'causal_explanations': self.causal_reasoner.causal_explanations[-1] if self.causal_reasoner.causal_explanations else None,
            'verification_results': verification_results
        }
        
        return {
            'control_signal': control_signal,
            'interpretability_info': interpretability_info
        }
    
    def get_system_interface(self) -> Dict[str, Any]:
        """Get the complete system interface specification."""
        interface = {
            'components': {},
            'routing': {
                'num_components': len(self.router.components),
                'routing_decisions': len(self.router.routing_decisions)
            },
            'memory_usage': {
                'max_history_size': self.max_history_size,
                'current_history_sizes': {
                    'perception_attention': len(self.perception_attention.attention_weights),
                    'planning_attention': len(self.planning_attention.attention_weights),
                    'control_attention': len(self.control_attention.attention_weights),
                    'safety_monitor': len(self.safety_monitor.safety_violations),
                    'causal_reasoner': len(self.causal_reasoner.causal_explanations),
                    'router': len(self.router.routing_decisions)
                }
            }
        }
        
        for component in self.router.components:
            interface['components'][component.name] = component.get_interface()
        
        return interface
    
    def verify_system(self, perception_input: torch.Tensor) -> Dict[str, Any]:
        """
        Comprehensive system verification.
        
        Args:
            perception_input: Test input data
            
        Returns:
            Dictionary of system verification results
        """
        # Validate input
        perception_input = validate_tensor(perception_input, "perception_input", expected_dtype=torch.float32)
        
        with torch.no_grad():
            # Forward pass
            outputs = self.forward(perception_input)
            
            # System-level checks
            control_signal = outputs['control_signal']
            interpretability_info = outputs['interpretability_info']
            
            # Check control signal bounds
            control_bounds_check = {
                'steering_in_bounds': torch.all((control_signal[:, 0] >= -1) & (control_signal[:, 0] <= 1)),
                'throttle_in_bounds': torch.all((control_signal[:, 1] >= 0) & (control_signal[:, 1] <= 1)),
                'brake_in_bounds': torch.all((control_signal[:, 2] >= 0) & (control_signal[:, 2] <= 1))
            }
            
            # Check interpretability metrics
            interpretability_check = {
                'has_attention_weights': all(w is not None for w in interpretability_info['attention_weights'].values()),
                'has_safety_violations': interpretability_info['safety_violations'] is not None,
                'has_causal_explanations': interpretability_info['causal_explanations'] is not None,
                'has_routing_decisions': interpretability_info['routing_decisions'] is not None
            }
            
            return {
                'control_bounds_check': control_bounds_check,
                'interpretability_check': interpretability_check,
                'verification_results': interpretability_info['verification_results']
            }
    
    def clear_all_history(self):
        """Clear all component history to free memory."""
        self.perception_attention.clear_history()
        self.planning_attention.clear_history()
        self.control_attention.clear_history()
        self.safety_monitor.clear_history()
        self.causal_reasoner.clear_history()
        self.router.clear_history()
        logger.info("Cleared all component history")


class ModularSystemTrainer:
    """
    Trainer for the modular interpretable driving system.
    """
    
    def __init__(
        self,
        system: InterpretableDrivingSystem,
        learning_rate: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        # Validate inputs
        if not isinstance(system, InterpretableDrivingSystem):
            raise ValueError("system must be an InterpretableDrivingSystem")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        self.system = system
        self.optimizer = torch.optim.AdamW(system.parameters(), lr=learning_rate)
        self.device = device
    
    def compute_interpretability_loss(
        self,
        interpretability_info: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Compute loss that encourages interpretability.
        
        Args:
            interpretability_info: Interpretability information from forward pass
            
        Returns:
            Interpretability loss
        """
        if not isinstance(interpretability_info, dict):
            raise ValueError("interpretability_info must be a dictionary")
        
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Encourage diverse attention weights
        for attention_weights in interpretability_info['attention_weights'].values():
            if attention_weights is not None:
                # Encourage attention diversity (avoid collapse)
                attention_entropy = -torch.sum(
                    attention_weights * torch.log(attention_weights + 1e-8), dim=-1
                ).mean()
                loss = loss - attention_entropy * 0.1  # Encourage higher entropy
        
        # Encourage sparse routing decisions
        routing_decisions = interpretability_info['routing_decisions']
        if routing_decisions is not None:
            # Encourage sparsity in routing
            routing_sparsity = torch.mean(routing_decisions)
            loss = loss + routing_sparsity * 0.1  # Encourage lower average routing weight
        
        return loss
    
    def train_step(
        self,
        perception_inputs: torch.Tensor,
        control_targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            perception_inputs: Perception data
            control_targets: Target control signals
            
        Returns:
            Dictionary of losses
        """
        # Validate inputs
        perception_inputs = validate_tensor(perception_inputs, "perception_inputs", expected_dtype=torch.float32)
        control_targets = validate_tensor(control_targets, "control_targets", expected_dtype=torch.float32)
        
        if perception_inputs.shape[0] != control_targets.shape[0]:
            raise ValueError("perception_inputs and control_targets must have the same batch size")
        
        self.system.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.system(perception_inputs)
        control_signal = outputs['control_signal']
        interpretability_info = outputs['interpretability_info']
        
        # Control loss
        control_loss = F.mse_loss(control_signal, control_targets)
        
        # Interpretability loss
        interpretability_loss = self.compute_interpretability_loss(interpretability_info)
        
        # Total loss
        total_loss = control_loss + 0.1 * interpretability_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'control_loss': control_loss.item(),
            'interpretability_loss': interpretability_loss.item()
        }
    
    def validate(
        self,
        perception_inputs: torch.Tensor,
        control_targets: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Validate the system.
        
        Args:
            perception_inputs: Perception data
            control_targets: Target control signals
            
        Returns:
            Dictionary of validation metrics
        """
        # Validate inputs
        perception_inputs = validate_tensor(perception_inputs, "perception_inputs", expected_dtype=torch.float32)
        control_targets = validate_tensor(control_targets, "control_targets", expected_dtype=torch.float32)
        
        if perception_inputs.shape[0] != control_targets.shape[0]:
            raise ValueError("perception_inputs and control_targets must have the same batch size")
        
        self.system.eval()
        
        with torch.no_grad():
            # Forward pass
            outputs = self.system(perception_inputs)
            control_signal = outputs['control_signal']
            
            # Control accuracy
            control_loss = F.mse_loss(control_signal, control_targets)
            
            # System verification
            verification_results = self.system.verify_system(perception_inputs)
            
            return {
                'control_loss': control_loss.item(),
                'verification_results': verification_results
            } 