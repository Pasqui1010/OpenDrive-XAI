from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["BEVLightDecoder"]


class BEVLightDecoder(nn.Module):
    """Lightweight decoder that upsamples BEV feature maps to a two-class occupancy grid.

    The design keeps computational overhead minimal (two small convolutions) so that it
    can run on-device without violating the <100 ms latency budget.
    """

    def __init__(self, in_channels: int = 256, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // 2, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, output_size: tuple[int, int] | None = (200, 200)) -> torch.Tensor:  # noqa: D401,E501
        """Return logits of shape ``[B, num_classes, H, W]``.

        Args:
            x: BEV feature map of shape ``[B, C, h, w]``.
            output_size: Final spatial resolution. Defaults to **(200, 200)** which matches
                the grid size used elsewhere in the project.  If *None*, keeps original
                resolution (useful for unit tests).
        """
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        if output_size is not None and (x.shape[-2] != output_size[0] or x.shape[-1] != output_size[1]):
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return x 