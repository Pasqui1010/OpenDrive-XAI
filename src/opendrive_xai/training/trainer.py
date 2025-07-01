"""Training pipeline for end-to-end autonomous driving with Vision Transformers.

This module implements training logic for the MultiCameraBEVTransformer with
explainability, robustness, and causal inference capabilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import wandb
from dataclasses import dataclass
import time
import json

from ..perception.vision_transformer import MultiCameraBEVTransformer
from ..config import Config

__all__ = ["E2ETrainer", "TrainingConfig", "LossComponents", "MetricsLogger"]


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    # Model architecture
    num_cameras: int = 6
    img_size: int = 224
    patch_size: int = 16
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    bev_channels: int = 256
    num_waypoints: int = 10
    
    # Training parameters
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    num_epochs: int = 100
    warmup_epochs: int = 10
    
    # Loss weights
    trajectory_weight: float = 1.0
    consistency_weight: float = 0.1
    attention_weight: float = 0.05
    causal_weight: float = 0.1
    
    # BEV segmentation auxiliary loss
    bev_seg_weight: float = 0.5
    
    # Robustness training
    adversarial_training: bool = True
    adversarial_epsilon: float = 8.0 / 255.0
    adversarial_alpha: float = 2.0 / 255.0
    adversarial_steps: int = 7
    
    # Data augmentation
    use_augmentation: bool = True
    augmentation_prob: float = 0.5
    
    # Regularization
    dropout_rate: float = 0.1
    label_smoothing: float = 0.1
    
    # Checkpointing
    save_every: int = 10
    eval_every: int = 5
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = "opendrive-xai"
    log_attention_maps: bool = True


@dataclass
class LossComponents:
    """Container for different loss components."""
    trajectory_loss: float
    consistency_loss: float
    attention_loss: float
    causal_loss: float
    total_loss: float


class MetricsLogger:
    """Metrics logging and tracking."""
    
    def __init__(self, use_wandb: bool = True):
        self.use_wandb = use_wandb
        self.metrics_history = []
        self.logger = logging.getLogger(__name__)
        
    def log_metrics(self, metrics: Dict[str, float], step: int, phase: str = "train"):
        """Log metrics to wandb and local storage."""
        # Add phase prefix
        prefixed_metrics = {f"{phase}/{k}": v for k, v in metrics.items()}
        
        if self.use_wandb:
            wandb.log(prefixed_metrics, step=step)
        
        # Local logging
        self.metrics_history.append({
            "step": step,
            "phase": phase,
            **metrics
        })
        
        # Console logging
        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step} [{phase}] - {metric_str}")
    
    def save_metrics(self, save_path: Path):
        """Save metrics history to file."""
        with open(save_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)


class AutonomousDrivingDataset(Dataset):
    """Dataset for autonomous driving training."""
    
    def __init__(self, data_dir: Path, transform=None, sequence_length: int = 5):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.sequence_length = sequence_length
        
        # Load data index
        self.data_samples = self._load_data_index()
        
    def _load_data_index(self) -> List[Dict]:
        """Load dataset index with camera images and trajectories."""
        # Placeholder implementation - should load actual dataset
        # This would typically load from preprocessed CARLA data
        samples = []
        
        # For now, create dummy data structure
        for i in range(1000):  # Dummy size
            sample = {
                'episode_id': f"episode_{i // 100}",
                'frame_id': i % 100,
                'camera_files': {
                    'front': self.data_dir / f"episode_{i // 100}" / "frames" / f"{i:06d}_front.png",
                    'rear': self.data_dir / f"episode_{i // 100}" / "frames" / f"{i:06d}_rear.png",
                    # Add other cameras...
                },
                'trajectory': np.random.randn(10, 2).astype(np.float32),  # Dummy trajectory
                'vehicle_state': {
                    'location': np.random.randn(3).astype(np.float32),
                    'rotation': np.random.randn(3).astype(np.float32),
                    'velocity': np.random.randn(3).astype(np.float32),
                }
            }
            samples.append(sample)
            
        return samples
    
    def __len__(self) -> int:
        return len(self.data_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get training sample."""
        sample = self.data_samples[idx]
        
        # Load camera images (dummy implementation)
        camera_images = []
        for camera_id in ['front', 'rear', 'left', 'right', 'front_left', 'front_right']:
            # Create dummy image
            image = torch.randn(3, 224, 224)
            camera_images.append(image)
        
        multi_camera_tensor = torch.stack(camera_images, dim=0)  # [num_cameras, 3, H, W]
        
        # Camera poses (dummy)
        camera_poses = torch.randn(6, 7)  # [num_cameras, 7] (x,y,z,qx,qy,qz,qw)
        
        # Ground truth trajectory
        trajectory = torch.from_numpy(sample['trajectory'])
        
        return {
            'multi_camera_input': multi_camera_tensor,
            'camera_poses': camera_poses,
            'trajectory': trajectory,
            'metadata': sample
        }


class E2ETrainer:
    """End-to-end trainer for autonomous driving with Vision Transformers."""
    
    def __init__(self, config: TrainingConfig, output_dir: Path):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = MultiCameraBEVTransformer(
            num_cameras=config.num_cameras,
            patch_size=config.patch_size,
            img_size=config.img_size,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            bev_channels=config.bev_channels,
            num_waypoints=config.num_waypoints
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )
        
        # Metrics logger
        self.metrics_logger = MetricsLogger(config.use_wandb)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Initialize wandb if enabled
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                config=config.__dict__,
                name=f"e2e_training_{int(time.time())}"
            )
    
    def compute_loss(self, 
                    predictions: torch.Tensor,
                    targets: torch.Tensor,
                    intermediates: Dict[str, Any],
                    bev_occupancy: torch.Tensor | None = None) -> LossComponents:
        """Compute multi-component loss function."""
        
        # 1. Trajectory loss (MSE between predicted and ground truth waypoints)
        trajectory_loss = F.mse_loss(predictions, targets)
        
        # 2. Consistency loss (encourage similar features from similar views)
        consistency_loss = torch.tensor(0.0, device=self.device)
        if 'camera_features' in intermediates and len(intermediates['camera_features']) > 1:
            features = torch.stack(intermediates['camera_features'], dim=1)  # [B, num_cameras, feature_dim]
            # Compute pairwise consistency
            feature_diff = features.unsqueeze(2) - features.unsqueeze(1)  # [B, num_cameras, num_cameras, feature_dim]
            consistency_loss = torch.mean(torch.norm(feature_diff, dim=-1))
        
        # 3. Attention regularization (encourage sparse, meaningful attention)
        attention_loss = torch.tensor(0.0, device=self.device)
        if 'attention_weights' in intermediates and intermediates['attention_weights']:
            for attn_weights in intermediates['attention_weights']:
                if attn_weights is not None:
                    # Encourage sparsity in attention
                    attention_entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=-1)
                    attention_loss += torch.mean(attention_entropy)
            attention_loss /= len(intermediates['attention_weights'])
        
        # 4. Causal loss (placeholder for causal inference regularization)
        causal_loss = torch.tensor(0.0, device=self.device)
        # TODO: Implement causal inference loss when causal modules are added
        
        # 5. BEV segmentation loss (optional)
        seg_loss = torch.tensor(0.0, device=self.device)
        if 'seg_logits' in intermediates and bev_occupancy is not None:
            seg_logits = intermediates['seg_logits']  # [B, 2, H, W]
            seg_loss = F.cross_entropy(seg_logits, bev_occupancy.long())
        
        # Combine losses
        total_loss = (
            self.config.trajectory_weight * trajectory_loss +
            self.config.consistency_weight * consistency_loss +
            self.config.attention_weight * attention_loss +
            self.config.causal_weight * causal_loss +
            self.config.bev_seg_weight * seg_loss
        )
        
        return LossComponents(
            trajectory_loss=trajectory_loss.item(),
            consistency_loss=consistency_loss.item(),
            attention_loss=attention_loss.item(),
            causal_loss=causal_loss.item(),
            total_loss=total_loss.item()
        )
    
    def adversarial_training_step(self, 
                                 batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform adversarial training step using PGD attack."""
        if not self.config.adversarial_training:
            return torch.tensor(0.0, device=self.device)
        
        multi_camera_input = batch['multi_camera_input'].clone().detach()
        camera_poses = batch['camera_poses']
        targets = batch['trajectory']
        
        # Initialize perturbation
        delta = torch.zeros_like(multi_camera_input, requires_grad=True)
        
        # PGD attack
        for _ in range(self.config.adversarial_steps):
            # Forward pass
            adv_input = multi_camera_input + delta
            adv_predictions, adv_intermediates = self.model(adv_input, camera_poses)
            
            # Compute loss
            loss_components = self.compute_loss(adv_predictions, targets, adv_intermediates)
            adv_loss = torch.tensor(loss_components.total_loss, device=self.device, requires_grad=True)
            
            # Backward pass
            adv_loss.backward()
            
            # Update perturbation
            grad = delta.grad.detach()
            delta.data = delta.data + self.config.adversarial_alpha * grad.sign()
            delta.data = torch.clamp(delta.data, -self.config.adversarial_epsilon, self.config.adversarial_epsilon)
            delta.grad.zero_()
        
        # Final adversarial forward pass
        adv_input = multi_camera_input + delta.detach()
        adv_predictions, adv_intermediates = self.model(adv_input, camera_poses)
        adv_loss_components = self.compute_loss(adv_predictions, targets, adv_intermediates)
        
        return torch.tensor(adv_loss_components.total_loss, device=self.device)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_metrics = {
            'loss': 0.0,
            'trajectory_loss': 0.0,
            'consistency_loss': 0.0,
            'attention_loss': 0.0,
            'causal_loss': 0.0,
            'adversarial_loss': 0.0
        }
        
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            multi_camera_input = batch['multi_camera_input'].to(self.device)
            camera_poses = batch['camera_poses'].to(self.device)
            targets = batch['trajectory'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions, intermediates = self.model(multi_camera_input, camera_poses)
            
            # Compute loss
            loss_components = self.compute_loss(predictions, targets, intermediates)
            loss = torch.tensor(loss_components.total_loss, device=self.device, requires_grad=True)
            
            # Adversarial training
            adv_loss = self.adversarial_training_step(batch)
            total_loss = loss + adv_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Update metrics
            epoch_metrics['loss'] += total_loss.item()
            epoch_metrics['trajectory_loss'] += loss_components.trajectory_loss
            epoch_metrics['consistency_loss'] += loss_components.consistency_loss
            epoch_metrics['attention_loss'] += loss_components.attention_loss
            epoch_metrics['causal_loss'] += loss_components.causal_loss
            epoch_metrics['adversarial_loss'] += adv_loss.item()
            
            self.global_step += 1
            
            # Log batch metrics
            if batch_idx % 50 == 0:
                self.logger.info(f"Epoch {self.current_epoch}, Batch {batch_idx}/{num_batches}, Loss: {total_loss.item():.4f}")
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
            
        return epoch_metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        val_metrics = {
            'loss': 0.0,
            'trajectory_loss': 0.0,
            'trajectory_error': 0.0,  # Additional metric: L2 error in meters
        }
        
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in val_loader:
                multi_camera_input = batch['multi_camera_input'].to(self.device)
                camera_poses = batch['camera_poses'].to(self.device)
                targets = batch['trajectory'].to(self.device)
                
                # Forward pass
                predictions, intermediates = self.model(multi_camera_input, camera_poses)
                
                # Compute loss
                loss_components = self.compute_loss(predictions, targets, intermediates)
                
                # Compute trajectory error in meters (assuming waypoints are in meters)
                trajectory_error = torch.mean(torch.norm(predictions - targets, dim=-1))
                
                val_metrics['loss'] += loss_components.total_loss
                val_metrics['trajectory_loss'] += loss_components.trajectory_loss
                val_metrics['trajectory_error'] += trajectory_error.item()
        
        # Average metrics
        for key in val_metrics:
            val_metrics[key] /= num_batches
            
        return val_metrics
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'global_step': self.global_step
        }
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'best_checkpoint.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint with val_loss: {val_loss:.4f}")
        
        # Save periodic checkpoints
        if epoch % self.config.save_every == 0:
            epoch_path = self.output_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, epoch_path)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop."""
        self.logger.info("Starting training...")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            self.metrics_logger.log_metrics(train_metrics, self.global_step, "train")
            
            # Validation
            if epoch % self.config.eval_every == 0:
                val_metrics = self.validate_epoch(val_loader)
                self.metrics_logger.log_metrics(val_metrics, self.global_step, "val")
                
                # Check if best model
                is_best = val_metrics['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['loss']
                
                # Save checkpoint
                self.save_checkpoint(epoch, val_metrics['loss'], is_best)
            
            # Update learning rate
            self.scheduler.step()
            
            # Log learning rate
            current_lr = self.scheduler.get_last_lr()[0]
            self.metrics_logger.log_metrics({'learning_rate': current_lr}, self.global_step, "train")
        
        # Save final metrics
        metrics_path = self.output_dir / 'training_metrics.json'
        self.metrics_logger.save_metrics(metrics_path)
        
        self.logger.info("Training completed!")
    
    def load_checkpoint(self, checkpoint_path: Path) -> bool:
        """Load model from checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.current_epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            self.best_val_loss = checkpoint['val_loss']
            
            self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return False 