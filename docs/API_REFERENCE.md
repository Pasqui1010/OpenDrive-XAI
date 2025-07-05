# OpenDrive-XAI API Reference

## Overview

OpenDrive-XAI is a comprehensive autonomous driving system that combines causal world modeling, modular neural components, and interpretable AI for safe and explainable autonomous navigation.

## Core Modules

### Causal World Model (`opendrive_xai.causal_world_model`)

The causal world model learns cause-effect relationships through synthetic interventions and provides interpretable predictions.

#### Classes

##### `CausalWorldModel`

A differentiable world model that learns causal relationships through synthetic interventions.

```python
class CausalWorldModel(nn.Module):
    def __init__(
        self,
        state_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the causal world model.
        
        Args:
            state_dim: Dimension of the state representation
            hidden_dim: Dimension of hidden layers
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            device: Device to run the model on
        """
```

**Methods:**

- `forward(world_state: WorldState, intervention: Optional[torch.Tensor] = None, intervention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]`
  - Forward pass through the causal world model
  - Returns predictions and causal insights

- `encode_state(world_state: WorldState) -> torch.Tensor`
  - Encode world state into latent representation

- `learn_causal_graph(states: torch.Tensor) -> torch.Tensor`
  - Learn causal relationships between state variables

- `predict_with_intervention(current_state: torch.Tensor, intervention: torch.Tensor, intervention_mask: torch.Tensor) -> torch.Tensor`
  - Predict the effect of an intervention

- `generate_counterfactual(factual_state: torch.Tensor, factual_outcome: torch.Tensor, intervention: torch.Tensor, intervention_mask: torch.Tensor) -> torch.Tensor`
  - Generate counterfactual predictions

##### `WorldState`

Represents the state of the world at a given time.

```python
@dataclass
class WorldState:
    vehicle_positions: torch.Tensor  # [num_vehicles, 3] - x, y, heading
    vehicle_velocities: torch.Tensor  # [num_vehicles, 2] - vx, vy
    traffic_lights: torch.Tensor  # [num_lights, 3] - x, y, state
    road_geometry: torch.Tensor  # [num_segments, 4] - x1, y1, x2, y2
    ego_state: torch.Tensor  # [6] - x, y, heading, vx, vy, steering
```

##### `CausalWorldModelTrainer`

Trainer for the causal world model with synthetic intervention learning.

```python
class CausalWorldModelTrainer:
    def __init__(
        self,
        model: CausalWorldModel,
        learning_rate: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
```

**Methods:**

- `train_step(world_states: List[WorldState], targets: torch.Tensor, use_interventions: bool = True) -> Dict[str, float]`
  - Single training step with optional synthetic interventions

- `validate(world_states: List[WorldState], targets: torch.Tensor) -> Dict[str, float]`
  - Validate the model on test data

##### `SyntheticInterventionGenerator`

Generates synthetic interventions for training the causal world model.

```python
class SyntheticInterventionGenerator:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
```

**Methods:**

- `generate_traffic_intervention(world_state: WorldState, intervention_type: str = "vehicle_speed_change") -> Tuple[torch.Tensor, torch.Tensor]`
  - Generate synthetic interventions for traffic scenarios

- `generate_safety_intervention(world_state: WorldState, safety_scenario: str = "emergency_brake") -> Tuple[torch.Tensor, torch.Tensor]`
  - Generate safety-critical interventions for testing robustness

### Modular Components (`opendrive_xai.modular_components`)

The modular components provide interpretable neural architectures with explicit interfaces and verifiable components.

#### Classes

##### `InterpretableDrivingSystem`

Complete interpretable driving system using modular components.

```python
class InterpretableDrivingSystem(nn.Module):
    def __init__(
        self,
        perception_dim: int = 512,
        planning_dim: int = 256,
        control_dim: int = 64,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_history_size: int = 100
    ):
```

**Methods:**

- `forward(perception_input: torch.Tensor) -> Dict[str, Any]`
  - Forward pass through the interpretable driving system
  - Returns control outputs and interpretability information

- `get_system_interface() -> Dict[str, Any]`
  - Get the complete system interface specification

- `verify_system(perception_input: torch.Tensor) -> Dict[str, Any]`
  - Comprehensive system verification

- `clear_all_history()`
  - Clear all component history to free memory

##### `AttentionGate`

Attention gate that controls information flow between components.

```python
class AttentionGate(NeuralComponent):
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 8, max_history_size: int = 50):
```

**Methods:**

- `forward(x: torch.Tensor) -> torch.Tensor`
  - Forward pass with attention mechanism

- `verify(x: torch.Tensor) -> Dict[str, Any]`
  - Verify attention behavior

- `clear_history()`
  - Clear attention weight history to free memory

##### `SafetyMonitor`

Safety monitoring component that checks for dangerous conditions.

```python
class SafetyMonitor(NeuralComponent):
    def __init__(self, input_dim: int, output_dim: int, safety_thresholds: Dict[str, float], max_history_size: int = 100):
```

**Methods:**

- `forward(x: torch.Tensor) -> torch.Tensor`
  - Forward pass with safety monitoring

- `verify(x: torch.Tensor) -> Dict[str, Any]`
  - Verify safety monitoring behavior

- `clear_history()`
  - Clear safety violation history to free memory

##### `CausalReasoner`

Causal reasoning component that infers cause-effect relationships.

```python
class CausalReasoner(NeuralComponent):
    def __init__(self, input_dim: int, output_dim: int, num_causal_vars: int = 10, max_history_size: int = 50):
```

**Methods:**

- `forward(x: torch.Tensor) -> torch.Tensor`
  - Forward pass with causal reasoning

- `verify(x: torch.Tensor) -> Dict[str, Any]`
  - Verify causal reasoning behavior

- `clear_history()`
  - Clear causal explanation history to free memory

##### `ModularSystemTrainer`

Trainer for the modular interpretable driving system.

```python
class ModularSystemTrainer:
    def __init__(
        self,
        system: InterpretableDrivingSystem,
        learning_rate: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
```

**Methods:**

- `train_step(perception_inputs: torch.Tensor, control_targets: torch.Tensor) -> Dict[str, float]`
  - Single training step

- `validate(perception_inputs: torch.Tensor, control_targets: torch.Tensor) -> Dict[str, Any]`
  - Validate the system

### Mechatronics Integration (`opendrive_xai.mechatronics`)

The mechatronics modules handle the integration between AI systems and physical vehicle hardware.

#### Classes

##### `VehicleInterface`

Interface for vehicle sensors and actuators.

```python
class VehicleInterface:
    def __init__(
        self,
        sensor_config: Dict[str, Dict[str, Any]],
        actuator_config: Dict[str, Dict[str, Any]],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
```

**Methods:**

- `fuse_sensor_data(camera_data: Optional[torch.Tensor] = None, lidar_data: Optional[torch.Tensor] = None, radar_data: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]`
  - Fuse data from multiple sensors

- `apply_control(control_commands: torch.Tensor) -> bool`
  - Apply control commands to vehicle actuators

- `check_safety_status() -> Dict[str, Any]`
  - Check current safety status

##### `AIVehicleIntegration`

Thread-safe AI-vehicle integration system.

```python
class AIVehicleIntegration:
    def __init__(
        self,
        causal_world_model: CausalWorldModel,
        interpretable_system: InterpretableDrivingSystem,
        vehicle_interface: VehicleInterface,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        control_frequency: float = 20.0,
        safety_check_frequency: float = 10.0,
        max_control_latency: float = 0.1
    ):
```

**Methods:**

- `start()`
  - Start the AI-vehicle integration system

- `stop()`
  - Stop the AI-vehicle integration system

- `process_sensor_data(sensor_data: Dict[str, torch.Tensor]) -> Dict[str, Any]`
  - Process sensor data and generate control commands

- `monitor_safety() -> Dict[str, Any]`
  - Monitor current safety status

- `get_system_status() -> Dict[str, Any]`
  - Get comprehensive system status

## Utility Functions

### Validation Functions

#### `validate_tensor`

Validate tensor properties and return validated tensor.

```python
def validate_tensor(
    tensor: torch.Tensor,
    name: str,
    expected_shape: Optional[Tuple] = None,
    expected_dtype: Optional[torch.dtype] = None,
    allow_nan: bool = False
) -> torch.Tensor:
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
```

#### `validate_world_state`

Validate WorldState object.

```python
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
```

## Configuration

### Sensor Configuration

```python
sensor_config = {
    'camera': {
        'resolution': (640, 480),
        'fps': 30
    },
    'lidar': {
        'points_per_scan': 1000,
        'fps': 10
    },
    'radar': {
        'max_range': 200,
        'fps': 20
    }
}
```

### Actuator Configuration

```python
actuator_config = {
    'steering': {
        'max_angle': 30,
        'response_time': 0.1
    },
    'throttle': {
        'max_value': 1.0,
        'response_time': 0.05
    },
    'brake': {
        'max_value': 1.0,
        'response_time': 0.05
    }
}
```

### Safety Thresholds

```python
safety_thresholds = {
    'collision_risk': 0.7,
    'speed_violation': 0.8,
    'lane_deviation': 0.6,
    'traffic_violation': 0.9
}
```

## Usage Examples

### Basic Usage

```python
import torch
from opendrive_xai.causal_world_model import CausalWorldModel, WorldState
from opendrive_xai.modular_components import InterpretableDrivingSystem
from opendrive_xai.mechatronics.vehicle_interface import VehicleInterface
from opendrive_xai.mechatronics.ai_vehicle_integration import AIVehicleIntegration

# Initialize components
device = "cuda" if torch.cuda.is_available() else "cpu"

causal_model = CausalWorldModel(device=device)
interpretable_system = InterpretableDrivingSystem(device=device)

sensor_config = {
    'camera': {'resolution': (640, 480), 'fps': 30},
    'lidar': {'points_per_scan': 1000, 'fps': 10},
    'radar': {'max_range': 200, 'fps': 20}
}

actuator_config = {
    'steering': {'max_angle': 30, 'response_time': 0.1},
    'throttle': {'max_value': 1.0, 'response_time': 0.05},
    'brake': {'max_value': 1.0, 'response_time': 0.05}
}

vehicle_interface = VehicleInterface(sensor_config, actuator_config, device=device)

# Create AI-vehicle integration
ai_integration = AIVehicleIntegration(
    causal_world_model=causal_model,
    interpretable_system=interpretable_system,
    vehicle_interface=vehicle_interface,
    device=device
)

# Start the system
ai_integration.start()

# Process sensor data
sensor_data = {
    'camera': torch.randn(1, 3, 480, 640, device=device),
    'lidar': torch.randn(1, 1000, 3, device=device),
    'radar': torch.randn(1, 10, 4, device=device)
}

result = ai_integration.process_sensor_data(sensor_data)

# Stop the system
ai_integration.stop()
```

### Training Example

```python
from opendrive_xai.causal_world_model import CausalWorldModelTrainer
from opendrive_xai.modular_components import ModularSystemTrainer

# Train causal world model
causal_model = CausalWorldModel(device=device)
causal_trainer = CausalWorldModelTrainer(causal_model, device=device)

# Create training data
world_states = [
    WorldState(
        vehicle_positions=torch.randn(3, 3, device=device),
        vehicle_velocities=torch.randn(3, 2, device=device),
        traffic_lights=torch.randn(2, 3, device=device),
        road_geometry=torch.randn(5, 4, device=device),
        ego_state=torch.randn(6, device=device)
    )
]
targets = torch.randn(1, 6, device=device)

# Training step
losses = causal_trainer.train_step(world_states, targets, use_interventions=True)

# Train modular system
interpretable_system = InterpretableDrivingSystem(device=device)
modular_trainer = ModularSystemTrainer(interpretable_system, device=device)

perception_inputs = torch.randn(1, 512, device=device)
control_targets = torch.randn(1, 3, device=device)

losses = modular_trainer.train_step(perception_inputs, control_targets)
```

## Error Handling

The system includes comprehensive error handling with meaningful error messages:

- **Input Validation**: All inputs are validated for type, shape, and value ranges
- **Memory Management**: Automatic memory cleanup with configurable history limits
- **Thread Safety**: Thread-safe operations for real-time systems
- **Graceful Degradation**: Fallback mechanisms when components fail

## Performance Considerations

- **Memory Usage**: Use `clear_history()` methods to free memory when needed
- **Real-time Performance**: Control frequency and safety check frequency are configurable
- **GPU Usage**: Models automatically use GPU when available
- **Thread Safety**: All operations are thread-safe for real-time applications

## Safety Features

- **Safety Monitoring**: Continuous safety monitoring with configurable thresholds
- **Emergency Handling**: Automatic emergency stop when safety violations are detected
- **Sensor Failure Handling**: Graceful degradation when sensors fail
- **Control Validation**: All control commands are validated before application 