import torch
import pytest
from typing import List

from opendrive_xai.perception import TinyBEVEncoder
from opendrive_xai.perception.vision_transformer import (
    MultiCameraBEVTransformer, 
    BEVProjection,
    MultiHeadAttention
)


def test_encoder_forward():
    model = TinyBEVEncoder(bev_channels=64)
    inp = torch.randn(2, 3, 224, 224)
    out = model(inp)
    assert out.shape == (2, 64, 7, 7)


def test_bev_projection_forward():
    """Test the new attention-driven BEV projection."""
    batch_size = 2
    num_cameras = 3
    seq_len = 10
    feature_dim = 512
    bev_size = (50, 50)  # Smaller for testing
    bev_channels = 128
    
    projection = BEVProjection(
        feature_dim=feature_dim,
        bev_size=bev_size,
        bev_channels=bev_channels,
        num_attention_heads=4
    )
    
    # Create dummy camera features
    camera_features = [
        torch.randn(batch_size, seq_len, feature_dim) 
        for _ in range(num_cameras)
    ]
    camera_poses = torch.randn(batch_size, num_cameras, 7)
    
    # Forward pass
    bev_output = projection(camera_features, camera_poses)
    
    # Check output shape
    expected_shape = (batch_size, bev_channels, bev_size[0], bev_size[1])
    assert bev_output.shape == expected_shape
    
    # Check that output is not all zeros
    assert not torch.allclose(bev_output, torch.zeros_like(bev_output))


def test_bev_projection_attention_weights():
    """Test that attention weights can be extracted."""
    projection = BEVProjection(
        feature_dim=256,
        bev_size=(20, 20),
        bev_channels=64,
        num_attention_heads=2
    )
    
    camera_features = [torch.randn(1, 5, 256) for _ in range(2)]
    camera_poses = torch.randn(1, 2, 7)
    
    # Forward pass
    _ = projection(camera_features, camera_poses)
    
    # Get attention weights
    attention_weights = projection.get_attention_weights()
    
    # Should have attention weights after forward pass
    assert attention_weights is not None
    assert attention_weights.dim() == 4  # [batch, heads, query_len, key_len]


def test_multicamera_bev_transformer_forward():
    """Test the complete MultiCameraBEVTransformer with new BEV projection."""
    batch_size = 1
    num_cameras = 2
    img_size = 224
    patch_size = 16
    d_model = 64  # Match bev_channels
    num_layers = 2  # Fewer layers for testing
    num_heads = 4
    bev_channels = 64  # Match d_model
    num_waypoints = 5
    
    model = MultiCameraBEVTransformer(
        num_cameras=num_cameras,
        patch_size=patch_size,
        img_size=img_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        bev_channels=bev_channels,
        num_waypoints=num_waypoints,
        enable_segmentation=False  # Disable for simpler testing
    )
    
    # Create dummy input
    multi_camera_input = torch.randn(batch_size, num_cameras, 3, img_size, img_size)
    camera_poses = torch.randn(batch_size, num_cameras, 7)
    
    # Forward pass
    trajectory, intermediates = model(multi_camera_input, camera_poses)
    
    # Check trajectory output
    expected_trajectory_shape = (batch_size, num_waypoints, 2)
    assert trajectory.shape == expected_trajectory_shape
    
    # Check that trajectory is not all zeros
    assert not torch.allclose(trajectory, torch.zeros_like(trajectory))
    
    # Check intermediates
    assert 'bev_features' in intermediates
    assert 'attention_weights' in intermediates
    
    # Check BEV features shape
    bev_features = intermediates['bev_features']
    expected_bev_shape = (batch_size, bev_channels, 200, 200)
    assert bev_features.shape == expected_bev_shape


def test_multicamera_bev_transformer_with_segmentation():
    """Test MultiCameraBEVTransformer with segmentation enabled."""
    d_model = 64
    bev_channels = 64
    model = MultiCameraBEVTransformer(
        num_cameras=2,
        patch_size=16,
        img_size=224,
        d_model=d_model,
        num_layers=2,
        num_heads=4,
        bev_channels=bev_channels,
        num_waypoints=5,
        enable_segmentation=True
    )
    
    batch_size = 1
    multi_camera_input = torch.randn(batch_size, 2, 3, 224, 224)
    camera_poses = torch.randn(batch_size, 2, 7)
    
    trajectory, intermediates = model(multi_camera_input, camera_poses)
    
    # Check trajectory
    assert trajectory.shape == (batch_size, 5, 2)
    
    # Check segmentation output
    assert 'bev_occupancy' in intermediates
    bev_occupancy = intermediates['bev_occupancy']
    assert bev_occupancy.shape == (batch_size, 2, 200, 200)  # 2 classes


def test_attention_mechanism():
    """Test the MultiHeadAttention mechanism used in BEV projection."""
    d_model = 64
    num_heads = 4
    seq_len = 10
    
    attention = MultiHeadAttention(d_model, num_heads, dropout=0.1)
    
    # Create dummy inputs
    query = torch.randn(2, seq_len, d_model)
    key = torch.randn(2, seq_len, d_model)
    value = torch.randn(2, seq_len, d_model)
    
    # Forward pass
    output, attention_weights = attention(query, key, value)
    
    # Check output shape
    assert output.shape == (2, seq_len, d_model)
    
    # Check attention weights shape
    assert attention_weights.shape == (2, num_heads, seq_len, seq_len)
    
    # Check that attention weights sum to 1
    attention_sums = attention_weights.sum(dim=-1)
    assert torch.allclose(attention_sums, torch.ones_like(attention_sums), atol=1e-6)


@pytest.mark.parametrize("bev_size", [(50, 50), (100, 100), (200, 200)])
def test_bev_projection_different_sizes(bev_size):
    """Test BEV projection with different grid sizes."""
    projection = BEVProjection(
        feature_dim=256,
        bev_size=bev_size,
        bev_channels=64,
        num_attention_heads=2
    )
    
    camera_features = [torch.randn(1, 5, 256) for _ in range(2)]
    camera_poses = torch.randn(1, 2, 7)
    
    bev_output = projection(camera_features, camera_poses)
    
    expected_shape = (1, 64, bev_size[0], bev_size[1])
    assert bev_output.shape == expected_shape


def test_bev_projection_gradient_flow():
    """Test that gradients flow properly through the BEV projection."""
    projection = BEVProjection(
        feature_dim=128,
        bev_size=(20, 20),
        bev_channels=32,
        num_attention_heads=2
    )
    
    camera_features = [torch.randn(1, 3, 128, requires_grad=True) for _ in range(2)]
    camera_poses = torch.randn(1, 2, 7)
    
    bev_output = projection(camera_features, camera_poses)
    
    # Create a dummy loss
    loss = bev_output.sum()
    
    # Backward pass
    loss.backward()
    
    # Check that at least one parameter has a nonzero gradient
    grads = [param.grad for param in projection.parameters() if param.grad is not None]
    assert any([not torch.allclose(g, torch.zeros_like(g)) for g in grads])
