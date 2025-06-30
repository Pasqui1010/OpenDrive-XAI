"""Simple CNN encoder that produces BEV-like feature map.

This is a *placeholder* CPU-friendly model built on torchvision's ResNet-18 with
its spatial resolution preserved (removing avg-pool & fc). A 1×1 conv reduces
channels to a smaller BEV feature dimension.
"""

from __future__ import annotations

import torch
from torchvision import models

__all__ = ["TinyBEVEncoder"]


class TinyBEVEncoder(torch.nn.Module):
    """ResNet18 backbone → channel-reduced feature map (~BEV stub)."""

    def __init__(self, bev_channels: int = 128):
        super().__init__()
        backbone = models.resnet18(weights=None)  # small & fast
        self.stem = torch.nn.Sequential(*list(backbone.children())[:-2])  # keep conv1..layer4
        self.reduce = torch.nn.Conv2d(512, bev_channels, kernel_size=1)
        self._init_weights()

    def _init_weights(self) -> None:  # noqa: D401
        torch.nn.init.kaiming_normal_(self.reduce.weight, mode="fan_out", nonlinearity="relu")
        if self.reduce.bias is not None:
            torch.nn.init.zeros_(self.reduce.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Return BEV-like feature map with shape (N, C, H/32, W/32)."""
        feat = self.stem(x)
        bev = self.reduce(feat)
        return bev 