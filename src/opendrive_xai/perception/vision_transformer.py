"""Vision Transformer backbone for multi-camera to BEV transformation.

This module implements the core ViT architecture with BEV projection capabilities
as outlined in the project plan, replacing the simple TinyBEVEncoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math

# Added for auxiliary BEV segmentation
from .bev_decoder import BEVLightDecoder

__all__ = ["MultiCameraBEVTransformer", "BEVProjection", "TemporalHead"]


class PositionalEncoding(nn.Module):
    """Positional encoding for Vision Transformer."""
    
    def __init__(self, d_model: int, max_seq_length: int = 5000):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe_buffer = getattr(self, 'pe', None)
        if pe_buffer is not None:
            return x + pe_buffer[:, :x.size(1)]
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head attention with explainability hooks for XAI."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Store attention weights for XAI
        self.attention_weights = None
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        
        # Transform queries, keys, values
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        self.attention_weights = attention_weights.detach()  # Store for XAI
        
        attention = self.dropout(attention_weights)
        context = torch.matmul(attention, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        output = self.w_o(context)
        return output, attention_weights


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with residual connections."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class BEVProjection(nn.Module):
    """Project 2D camera features to unified 3D Bird's-Eye-View space."""
    
    def __init__(self, feature_dim: int, bev_size: Tuple[int, int] = (200, 200), 
                 bev_channels: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        self.bev_size = bev_size
        self.bev_channels = bev_channels
        
        # Learnable camera-to-BEV projection matrices
        self.projection_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, bev_channels)
        )
        
        # Spatial embedding for BEV coordinates
        self.bev_embedding = nn.Parameter(
            torch.randn(bev_channels, bev_size[0], bev_size[1]) * 0.02
        )
        
    def forward(self, camera_features: List[torch.Tensor], 
                camera_poses: torch.Tensor) -> torch.Tensor:
        """
        Args:
            camera_features: List of [B, seq_len, feature_dim] from each camera
            camera_poses: [B, num_cameras, 7] (x,y,z,qx,qy,qz,qw)
        
        Returns:
            bev_features: [B, bev_channels, H, W] unified BEV representation
        """
        batch_size = camera_features[0].size(0)
        
        # Aggregate features from all cameras
        aggregated_features = []
        for i, cam_feat in enumerate(camera_features):
            # Project camera features to BEV space
            projected = self.projection_mlp(cam_feat)  # [B, seq_len, bev_channels]
            
            # Pool over sequence dimension (simplified - could use attention)
            pooled = projected.mean(dim=1)  # [B, bev_channels]
            aggregated_features.append(pooled)
        
        # Stack and process multi-camera features
        multi_cam_features = torch.stack(aggregated_features, dim=1)  # [B, num_cameras, bev_channels]
        
        # Combine features (simplified - actual implementation would use geometric projection)
        combined = multi_cam_features.mean(dim=1)  # [B, bev_channels]
        
        # Reshape to spatial BEV map
        bev_features = combined.unsqueeze(-1).unsqueeze(-1) * self.bev_embedding.unsqueeze(0)
        
        return bev_features


class TemporalHead(nn.Module):
    """Process temporal sequence of BEV maps for trajectory planning."""
    
    def __init__(self, bev_channels: int, hidden_dim: int = 512, num_layers: int = 2):
        super().__init__()
        
        self.temporal_transformer = nn.ModuleList([
            TransformerEncoderLayer(bev_channels, num_heads=8, d_ff=hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.positional_encoding = PositionalEncoding(bev_channels)
        
    def forward(self, bev_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bev_sequence: [B, seq_len, bev_channels, H, W]
        
        Returns:
            temporal_features: [B, seq_len, bev_channels] processed temporal features
        """
        B, T, C, H, W = bev_sequence.shape
        
        # Global average pooling over spatial dimensions
        pooled_bev = bev_sequence.mean(dim=(-2, -1))  # [B, T, C]
        
        # Add positional encoding
        temporal_input = self.positional_encoding(pooled_bev)
        
        # Process through temporal transformer
        for layer in self.temporal_transformer:
            temporal_input = layer(temporal_input)
            
        return temporal_input


class MultiCameraBEVTransformer(nn.Module):
    """Complete multi-camera to trajectory transformer architecture."""
    
    def __init__(self, 
                 num_cameras: int = 6,
                 patch_size: int = 16,
                 img_size: int = 224,
                 d_model: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 bev_channels: int = 256,
                 num_waypoints: int = 10,
                 enable_segmentation: bool = True):
        super().__init__()
        
        self.num_cameras = num_cameras
        self.patch_size = patch_size
        self.img_size = img_size
        self.d_model = d_model
        
        # Vision Transformer backbone for each camera
        num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embedding = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_model * 4)
            for _ in range(num_layers)
        ])
        
        # BEV projection module
        self.bev_projection = BEVProjection(d_model, bev_channels=bev_channels)
        
        # Temporal processing
        self.temporal_head = TemporalHead(bev_channels)
        
        # Planning head
        self.planning_head = nn.Sequential(
            nn.Linear(bev_channels, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_waypoints * 2)  # (x, y) coordinates for each waypoint
        )
        
        self._init_weights()
        
        # Auxiliary BEV segmentation head (optional)
        self.enable_segmentation = enable_segmentation
        if self.enable_segmentation:
            self.seg_head = BEVLightDecoder(bev_channels, num_classes=2)
        
    def _init_weights(self):
        """Initialize weights following ViT conventions."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)
    
    def forward_camera(self, x: torch.Tensor) -> torch.Tensor:
        """Process single camera input through ViT backbone."""
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embedding(x)  # [B, d_model, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, d_model]
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embedding
        
        # Process through transformer layers
        for layer in self.encoder_layers:
            x = layer(x)
            
        return x
    
    def forward(self, multi_camera_input: torch.Tensor, 
                camera_poses: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            multi_camera_input: [B, num_cameras, 3, H, W] or [B, T, num_cameras, 3, H, W]
            camera_poses: [B, num_cameras, 7] camera poses (x,y,z,qx,qy,qz,qw)
        
        Returns:
            waypoints: [B, num_waypoints, 2] predicted trajectory waypoints
            intermediates: dict with intermediate representations for XAI
        """
        is_temporal = len(multi_camera_input.shape) == 6
        
        if is_temporal:
            B, T, num_cameras, C, H, W = multi_camera_input.shape
            # Process each timestep
            camera_features = []
            for t in range(T):
                t_features = []
                for cam in range(num_cameras):
                    cam_feat = self.forward_camera(multi_camera_input[:, t, cam])
                    t_features.append(cam_feat[:, 0])  # Use CLS token
                camera_features.append(t_features)
                
            # Project to BEV for each timestep
            bev_sequence = []
            for t in range(T):
                bev_feat = self.bev_projection(camera_features[t], camera_poses)
                bev_sequence.append(bev_feat)
            
            bev_sequence = torch.stack(bev_sequence, dim=1)  # [B, T, C, H, W]
            
            # Process temporal sequence
            temporal_features = self.temporal_head(bev_sequence)  # [B, T, C]
            final_features = temporal_features[:, -1]  # Use last timestep
            
            # Use BEV of last frame for segmentation output if enabled
            if self.enable_segmentation:
                seg_logits = self.seg_head(bev_sequence[:, -1])  # [B, 2, H, W]
            
        else:
            B, num_cameras, C, H, W = multi_camera_input.shape
            
            # Process each camera
            camera_features = []
            for cam in range(num_cameras):
                cam_feat = self.forward_camera(multi_camera_input[:, cam])
                camera_features.append(cam_feat[:, 0])  # Use CLS token
            
            # Project to BEV
            bev_features = self.bev_projection(camera_features, camera_poses)
            final_features = bev_features.mean(dim=(-2, -1))  # Global average pooling
            
            if self.enable_segmentation:
                seg_logits = self.seg_head(bev_features)  # [B, 2, H, W]
        
        # Generate waypoints
        waypoints = self.planning_head(final_features)
        waypoints = waypoints.view(B, -1, 2)  # [B, num_waypoints, 2]
        
        # Collect intermediate representations for XAI
        attention_weights = []
        for layer in self.encoder_layers[-3:]:  # Last 3 layers
            if hasattr(layer, 'self_attention') and hasattr(layer.self_attention, 'attention_weights'):
                if layer.self_attention.attention_weights is not None:
                    attention_weights.append(layer.self_attention.attention_weights)
        
        intermediates = {
            'camera_features': camera_features,
            'bev_features': bev_features if not is_temporal else bev_sequence,
            'attention_weights': attention_weights,
        }
        
        if self.enable_segmentation:
            intermediates['seg_logits'] = seg_logits
        
        return waypoints, intermediates 