# model.py
# ─────────────────────────────────────────────────────────────────────────────
# ResNet-18 backbone fine-tuned for binary gravitational lens classification.
#
# Architecture:
#   • Backbone : ResNet-18 (optionally pretrained on ImageNet)
#   • Input    : (B, 3, H, W) — greyscale repeated across 3 channels
#   • Head     : Linear(512 → 1) — raw logit for BCEWithLogitsLoss
#   • Output   : scalar logit per image; apply sigmoid for probability
# ─────────────────────────────────────────────────────────────────────────────

import torch.nn as nn
from torchvision import models


class GravLensNet(nn.Module):
    """
    ResNet-18 adapted for binary gravitational lens classification.

    Args:
        pretrained (bool): Initialise backbone with ImageNet-1K weights.
                           Recommended True — significantly speeds convergence.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights       = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone      = models.resnet18(weights=weights)
        in_features   = backbone.fc.in_features               # 512
        backbone.fc   = nn.Linear(in_features, 1)             # binary head
        self.backbone = backbone

    def forward(self, x):
        """
        Args:
            x: Tensor (B, 3, H, W)
        Returns:
            logits: Tensor (B, 1)
        """
        return self.backbone(x)
