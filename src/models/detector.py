"""The assembled detector: backbone -> severity -> conditioning -> verdict."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..components.heads import Conditioner, SeverityHead


class Detector(nn.Module):
    """Backbone -> pooled features -> {severity, conditioning, detection}."""

    def __init__(self, backbone: nn.Module, dim: int = 1024, conditioner: str = "film"):
        super().__init__()
        self.backbone = backbone
        self.severity = SeverityHead(dim)
        self.conditioner = Conditioner(dim, conditioner)
        self.classifier = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(x)
        s_hat = self.severity(h.detach())      # see SeverityHead docstring
        h_corrected = self.conditioner(h, s_hat)
        return {
            "logit": self.classifier(h_corrected).squeeze(-1),
            "s_hat": s_hat,
            "h": h,
        }
