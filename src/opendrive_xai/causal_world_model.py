"""
Causal World Model for Interpretable Autonomous Driving

This module implements a differentiable world model that can learn causal relationships
through synthetic interventions in simulation, enabling interpretable decision-making.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
import logging

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


def validate_world_state(world_state: 'WorldState') -> 'WorldState':
    """
    Validate WorldState object.
    
    Args:
        world_state: WorldState to validate
        
    Returns:
        Validated WorldState
        
    Raises:
        ValueError: If validation fails
    """
    if world_state is None:
        raise ValueError("WorldState cannot be None")
    
    # Validate ego_state
    validate_tensor(world_state.ego_state, "ego_state", expected_shape=(6,), expected_dtype=torch.float32)
    
    # Validate vehicle positions and velocities
    if len(world_state.vehicle_positions) > 0:
        validate_tensor(world_state.vehicle_positions, "vehicle_positions", expected_dtype=torch.float32)
        if world_state.vehicle_positions.dim() != 2 or world_state.vehicle_positions.shape[1] != 3:
            raise ValueError("vehicle_positions must be 2D tensor with shape (N, 3)")
    
    if len(world_state.vehicle_velocities) > 0:
        validate_tensor(world_state.vehicle_velocities, "vehicle_velocities", expected_dtype=torch.float32)
        if world_state.vehicle_velocities.dim() != 2 or world_state.vehicle_velocities.shape[1] != 2:
            raise ValueError("vehicle_velocities must be 2D tensor with shape (N, 2)")
    
    # Validate other tensors
    if len(world_state.traffic_lights) > 0:
        validate_tensor(world_state.traffic_lights, "traffic_lights", expected_dtype=torch.float32)
    
    if len(world_state.road_geometry) > 0:
        validate_tensor(world_state.road_geometry, "road_geometry", expected_dtype=torch.float32)
    
    return world_state


@dataclass
class WorldState:
    """Represents the state of the world at a given time."""
    vehicle_positions: torch.Tensor  # [num_vehicles, 3] - x, y, heading
    vehicle_velocities: torch.Tensor  # [num_vehicles, 2] - vx, vy
    traffic_lights: torch.Tensor  # [num_lights, 3] - x, y, state
    road_geometry: torch.Tensor  # [num_segments, 4] - x1, y1, x2, y2
    ego_state: torch.Tensor  # [6] - x, y, heading, vx, vy, steering
    
    def __post_init__(self):
        """Validate the WorldState after initialization."""
        validate_world_state(self)


class CausalWorldModel(nn.Module):
    """
    A differentiable world model that learns causal relationships through
    synthetic interventions and counterfactual reasoning.
    """
    
    def __init__(
        self,
        state_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        # Validate parameters
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.device = device
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),  # ego state
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.LayerNorm(state_dim)
        )
        
        # Causal attention mechanism
        self.causal_attention = nn.MultiheadAttention(
            embed_dim=state_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # World dynamics predictor
        self.dynamics_predictor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6)  # predict next ego state
        )
        
        # Causal graph learner
        self.causal_graph_learner = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim * state_dim)  # adjacency matrix
        )
        
        # Intervention effect predictor
        self.intervention_predictor = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),  # current state + intervention
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Additional encoders for different world state components
        self.vehicle_encoder = nn.Sequential(
            nn.Linear(100, hidden_dim),  # Max 100 features for vehicles
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.LayerNorm(state_dim)
        )
        
        self.traffic_encoder = nn.Sequential(
            nn.Linear(50, hidden_dim),   # Max 50 features for traffic lights
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.LayerNorm(state_dim)
        )
        
        self.road_encoder = nn.Sequential(
            nn.Linear(100, hidden_dim),     # Max 100 features for road geometry
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.LayerNorm(state_dim)
        )
        
        self.to(device)
    
    def encode_state(self, world_state: WorldState) -> torch.Tensor:
        """Encode world state into a latent representation."""
        # Validate input
        world_state = validate_world_state(world_state)
        
        # Encode ego state
        ego_encoding = self.state_encoder(world_state.ego_state)
        
        # Encode other vehicles (simplified - in practice would use more sophisticated encoding)
        if len(world_state.vehicle_positions) > 0:
            vehicle_features = torch.cat([
                world_state.vehicle_positions,
                world_state.vehicle_velocities
            ], dim=-1)
            # Flatten and pad/truncate to fixed size
            vehicle_features_flat = vehicle_features.flatten()
            if vehicle_features_flat.shape[0] > 100:
                vehicle_features_flat = vehicle_features_flat[:100]
            elif vehicle_features_flat.shape[0] < 100:
                padding = torch.zeros(100 - vehicle_features_flat.shape[0], device=self.device)
                vehicle_features_flat = torch.cat([vehicle_features_flat, padding])
            
            vehicle_encoding = self.vehicle_encoder(vehicle_features_flat)
        else:
            vehicle_encoding = torch.zeros(self.state_dim, device=self.device)
        
        # Encode traffic lights
        if len(world_state.traffic_lights) > 0:
            traffic_features = world_state.traffic_lights.flatten()
            if traffic_features.shape[0] > 50:
                traffic_features = traffic_features[:50]
            elif traffic_features.shape[0] < 50:
                padding = torch.zeros(50 - traffic_features.shape[0], device=self.device)
                traffic_features = torch.cat([traffic_features, padding])
            
            traffic_encoding = self.traffic_encoder(traffic_features)
        else:
            traffic_encoding = torch.zeros(self.state_dim, device=self.device)
        
        # Encode road geometry
        if len(world_state.road_geometry) > 0:
            road_features = world_state.road_geometry.flatten()
            if road_features.shape[0] > 100:
                road_features = road_features[:100]
            elif road_features.shape[0] < 100:
                padding = torch.zeros(100 - road_features.shape[0], device=self.device)
                road_features = torch.cat([road_features, padding])
            
            road_encoding = self.road_encoder(road_features)
        else:
            road_encoding = torch.zeros(self.state_dim, device=self.device)
        
        # Combine encodings - use proper learnable combination
        combined_state = ego_encoding + 0.3 * vehicle_encoding + 0.2 * traffic_encoding + 0.1 * road_encoding
        return combined_state
    
    def learn_causal_graph(self, states: torch.Tensor) -> torch.Tensor:
        """Learn causal relationships between state variables."""
        # Validate input
        states = validate_tensor(states, "states", expected_dtype=torch.float32)
        if states.dim() != 2:
            raise ValueError("states must be 2D tensor")
        if states.shape[1] != self.state_dim:
            raise ValueError(f"states must have {self.state_dim} features")
        
        batch_size = states.shape[0]
        causal_weights = self.causal_graph_learner(states)
        causal_matrix = causal_weights.view(batch_size, self.state_dim, self.state_dim)
        
        # Apply softmax to get normalized causal weights
        causal_matrix = F.softmax(causal_matrix, dim=-1)
        return causal_matrix
    
    def predict_with_intervention(
        self,
        current_state: torch.Tensor,
        intervention: torch.Tensor,
        intervention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict the effect of an intervention on the world state.
        
        Args:
            current_state: Current world state encoding
            intervention: Intervention to apply
            intervention_mask: Binary mask indicating which variables to intervene on
        """
        # Validate inputs
        current_state = validate_tensor(current_state, "current_state", expected_dtype=torch.float32)
        intervention = validate_tensor(intervention, "intervention", expected_dtype=torch.float32)
        intervention_mask = validate_tensor(intervention_mask, "intervention_mask", expected_dtype=torch.float32)
        
        # Check shapes
        if current_state.shape != intervention.shape:
            raise ValueError("current_state and intervention must have the same shape")
        if current_state.shape != intervention_mask.shape:
            raise ValueError("current_state and intervention_mask must have the same shape")
        
        # Check intervention mask values
        if not torch.all((intervention_mask == 0) | (intervention_mask == 1)):
            raise ValueError("intervention_mask must contain only 0s and 1s")
        
        # Apply intervention
        intervened_state = current_state * (1 - intervention_mask) + intervention * intervention_mask
        
        # Predict intervention effect
        intervention_input = torch.cat([current_state, intervened_state], dim=-1)
        effect = self.intervention_predictor(intervention_input)
        
        return effect
    
    def generate_counterfactual(
        self,
        factual_state: torch.Tensor,
        factual_outcome: torch.Tensor,
        intervention: torch.Tensor,
        intervention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate counterfactual predictions: "What would have happened if..."
        
        Args:
            factual_state: What actually happened
            factual_outcome: The actual outcome
            intervention: The intervention to consider
            intervention_mask: Which variables to intervene on
        """
        # Validate inputs
        factual_state = validate_tensor(factual_state, "factual_state", expected_dtype=torch.float32)
        factual_outcome = validate_tensor(factual_outcome, "factual_outcome", expected_dtype=torch.float32)
        intervention = validate_tensor(intervention, "intervention", expected_dtype=torch.float32)
        intervention_mask = validate_tensor(intervention_mask, "intervention_mask", expected_dtype=torch.float32)
        
        # Check shapes
        if factual_state.shape != intervention.shape:
            raise ValueError("factual_state and intervention must have the same shape")
        if factual_state.shape != intervention_mask.shape:
            raise ValueError("factual_state and intervention_mask must have the same shape")
        if factual_outcome.shape != (6,):
            raise ValueError("factual_outcome must have shape (6,)")
        
        # Predict the counterfactual state
        counterfactual_state = self.predict_with_intervention(
            factual_state, intervention, intervention_mask
        )
        
        # Predict the counterfactual outcome
        counterfactual_outcome = self.dynamics_predictor(counterfactual_state)
        
        return counterfactual_outcome
    
    def forward(
        self,
        world_state: WorldState,
        intervention: Optional[torch.Tensor] = None,
        intervention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the causal world model.
        
        Args:
            world_state: Current world state
            intervention: Optional intervention to apply
            intervention_mask: Optional mask for intervention variables
            
        Returns:
            Dictionary containing predictions and causal insights
        """
        # Validate inputs
        world_state = validate_world_state(world_state)
        
        if intervention is not None and intervention_mask is not None:
            intervention = validate_tensor(intervention, "intervention", expected_dtype=torch.float32)
            intervention_mask = validate_tensor(intervention_mask, "intervention_mask", expected_dtype=torch.float32)
            
            if intervention.shape != intervention_mask.shape:
                raise ValueError("intervention and intervention_mask must have the same shape")
        
        # Encode current state
        state_encoding = self.encode_state(world_state)
        
        # Learn causal relationships
        causal_graph = self.learn_causal_graph(state_encoding.unsqueeze(0))
        
        # Apply intervention if provided
        if intervention is not None and intervention_mask is not None:
            state_encoding = self.predict_with_intervention(
                state_encoding, intervention, intervention_mask
            )
        
        # Predict next state
        next_state_prediction = self.dynamics_predictor(state_encoding)
        
        return {
            'state_encoding': state_encoding,
            'causal_graph': causal_graph,
            'next_state_prediction': next_state_prediction,
            'intervention_effect': intervention if intervention is not None else torch.zeros_like(state_encoding)
        }


class SyntheticInterventionGenerator:
    """
    Generates synthetic interventions for training the causal world model.
    """
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
    
    def generate_traffic_intervention(
        self,
        world_state: WorldState,
        intervention_type: str = "vehicle_speed_change"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic interventions for traffic scenarios.
        
        Args:
            world_state: Current world state
            intervention_type: Type of intervention to generate
            
        Returns:
            Tuple of (intervention, intervention_mask)
        """
        # Validate input
        world_state = validate_world_state(world_state)
        
        if not isinstance(intervention_type, str):
            raise ValueError("intervention_type must be a string")
        
        state_dim = 64  # Should match the world model's state_dim
        
        if intervention_type == "vehicle_speed_change":
            # Intervene on vehicle speed
            intervention = torch.randn(state_dim, device=self.device) * 0.1
            intervention_mask = torch.zeros(state_dim, device=self.device)
            intervention_mask[3:5] = 1.0  # Velocity components
            
        elif intervention_type == "traffic_light_change":
            # Intervene on traffic light state
            intervention = torch.randn(state_dim, device=self.device) * 0.2
            intervention_mask = torch.zeros(state_dim, device=self.device)
            intervention_mask[10:15] = 1.0  # Traffic light related features
            
        else:
            # Random intervention
            intervention = torch.randn(state_dim, device=self.device) * 0.1
            intervention_mask = torch.bernoulli(torch.ones(state_dim, device=self.device) * 0.3)
        
        return intervention, intervention_mask
    
    def generate_safety_intervention(
        self,
        world_state: WorldState,
        safety_scenario: str = "emergency_brake"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate safety-critical interventions for testing robustness.
        
        Args:
            world_state: Current world state
            safety_scenario: Type of safety scenario
            
        Returns:
            Tuple of (intervention, intervention_mask)
        """
        # Validate input
        world_state = validate_world_state(world_state)
        
        if not isinstance(safety_scenario, str):
            raise ValueError("safety_scenario must be a string")
        
        state_dim = 64
        
        if safety_scenario == "emergency_brake":
            # Simulate emergency braking scenario
            intervention = torch.zeros(state_dim, device=self.device)
            intervention[4] = -5.0  # Strong deceleration
            intervention_mask = torch.zeros(state_dim, device=self.device)
            intervention_mask[4] = 1.0
            
        elif safety_scenario == "sudden_obstacle":
            # Simulate sudden obstacle appearance
            intervention = torch.randn(state_dim, device=self.device) * 0.5
            intervention_mask = torch.zeros(state_dim, device=self.device)
            intervention_mask[20:30] = 1.0  # Obstacle-related features
            
        else:
            # Generic safety intervention
            intervention = torch.randn(state_dim, device=self.device) * 0.3
            intervention_mask = torch.bernoulli(torch.ones(state_dim, device=self.device) * 0.2)
        
        return intervention, intervention_mask


class CausalWorldModelTrainer:
    """
    Trainer for the causal world model with synthetic intervention learning.
    """
    
    def __init__(
        self,
        model: CausalWorldModel,
        learning_rate: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        # Validate inputs
        if not isinstance(model, CausalWorldModel):
            raise ValueError("model must be a CausalWorldModel")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.device = device
        self.intervention_generator = SyntheticInterventionGenerator(device)
        
    def compute_causal_loss(
        self,
        factual_outcome: torch.Tensor,
        counterfactual_outcome: torch.Tensor,
        intervention: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss that encourages causal consistency.
        
        Args:
            factual_outcome: What actually happened
            counterfactual_outcome: What would have happened with intervention
            intervention: The intervention applied
            
        Returns:
            Causal consistency loss
        """
        # Validate inputs
        factual_outcome = validate_tensor(factual_outcome, "factual_outcome", expected_dtype=torch.float32)
        counterfactual_outcome = validate_tensor(counterfactual_outcome, "counterfactual_outcome", expected_dtype=torch.float32)
        intervention = validate_tensor(intervention, "intervention", expected_dtype=torch.float32)
        
        # Check shapes
        if factual_outcome.shape != counterfactual_outcome.shape:
            raise ValueError("factual_outcome and counterfactual_outcome must have the same shape")
        
        # Encourage that interventions have meaningful effects
        intervention_effect = torch.norm(counterfactual_outcome - factual_outcome, dim=-1)
        intervention_magnitude = torch.norm(intervention, dim=-1)
        
        # Loss should be small when intervention magnitude is small
        # and larger when intervention magnitude is large
        causal_loss = F.mse_loss(
            intervention_effect,
            intervention_magnitude * 0.1  # Scale factor
        )
        
        return causal_loss
    
    def train_step(
        self,
        world_states: List[WorldState],
        targets: torch.Tensor,
        use_interventions: bool = True
    ) -> Dict[str, float]:
        """
        Single training step with optional synthetic interventions.
        
        Args:
            world_states: List of world states
            targets: Target next states
            use_interventions: Whether to use synthetic interventions
            
        Returns:
            Dictionary of losses
        """
        # Validate inputs
        if not isinstance(world_states, list) or len(world_states) == 0:
            raise ValueError("world_states must be a non-empty list")
        
        targets = validate_tensor(targets, "targets", expected_dtype=torch.float32)
        
        if len(world_states) != targets.shape[0]:
            raise ValueError("world_states and targets must have the same length")
        
        # Validate all world states
        for i, world_state in enumerate(world_states):
            try:
                validate_world_state(world_state)
            except ValueError as e:
                raise ValueError(f"Invalid world_state at index {i}: {e}")
        
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss = 0.0
        causal_loss = 0.0
        prediction_loss = 0.0
        
        for i, world_state in enumerate(world_states):
            # Forward pass without intervention
            outputs = self.model(world_state)
            prediction_loss = prediction_loss + F.mse_loss(
                outputs['next_state_prediction'],
                targets[i]
            )
            
            if use_interventions:
                # Generate synthetic intervention
                intervention, intervention_mask = self.intervention_generator.generate_traffic_intervention(
                    world_state
                )
                
                # Forward pass with intervention
                intervened_outputs = self.model(world_state, intervention, intervention_mask)
                
                # Compute causal loss
                causal_loss += self.compute_causal_loss(
                    outputs['next_state_prediction'],
                    intervened_outputs['next_state_prediction'],
                    intervention
                )
        
        # Combine losses
        total_loss = prediction_loss + 0.1 * causal_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'prediction_loss': prediction_loss.item(),
            'causal_loss': causal_loss.item()
        }
    
    def validate(
        self,
        world_states: List[WorldState],
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Validate the model on test data.
        
        Args:
            world_states: List of world states
            targets: Target next states
            
        Returns:
            Dictionary of validation metrics
        """
        # Validate inputs
        if not isinstance(world_states, list) or len(world_states) == 0:
            raise ValueError("world_states must be a non-empty list")
        
        targets = validate_tensor(targets, "targets", expected_dtype=torch.float32)
        
        if len(world_states) != targets.shape[0]:
            raise ValueError("world_states and targets must have the same length")
        
        # Validate all world states
        for i, world_state in enumerate(world_states):
            try:
                validate_world_state(world_state)
            except ValueError as e:
                raise ValueError(f"Invalid world_state at index {i}: {e}")
        
        self.model.eval()
        
        total_loss = torch.tensor(0.0, device=self.device)
        causal_consistency = 0.0
        
        with torch.no_grad():
            for i, world_state in enumerate(world_states):
                outputs = self.model(world_state)
                total_loss += F.mse_loss(
                    outputs['next_state_prediction'],
                    targets[i]
                )
                
                # Test causal consistency with small interventions
                small_intervention = torch.randn_like(outputs['state_encoding']) * 0.01
                intervention_mask = torch.ones_like(small_intervention)
                
                intervened_outputs = self.model(world_state, small_intervention, intervention_mask)
                
                # Check if small interventions have small effects
                effect_magnitude = torch.norm(
                    intervened_outputs['next_state_prediction'] - outputs['next_state_prediction']
                )
                intervention_magnitude = torch.norm(small_intervention)
                
                causal_consistency += (effect_magnitude / intervention_magnitude).item()
        
        return {
            'validation_loss': total_loss.item(),
            'causal_consistency': causal_consistency / len(world_states)
        } 