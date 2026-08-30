"""Detection head, severity head, FiLM conditioning arms.

~1M params on a 300M backbone -- 0.3% of the model, and the part that is ours.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..utils.transforms import OP_NAMES

N_OPS = len(OP_NAMES)


class SeverityHead(nn.Module):
    """Regresses the 8-dim severity vector from backbone features.

    Input is DETACHED, and that is load-bearing: the consistency loss wants
    features BLIND to degradation, this head wants features SENSITIVE to it.
    Trained jointly they fight and the trunk is mediocre at both. Detaching lets
    this head read surviving traces without being able to reshape them.
    """

    def __init__(self, dim: int, hidden: int = 256, bounded: bool = True):
        super().__init__()
        self.bounded = bounded
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, N_OPS)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out = self.net(h)
        # bounded=True (the original) squashes to [0,1] with a sigmoid. That was
        # a mistake: once the head settles near the target mean the sigmoid
        # flattens its gradients and it cannot climb out -- ours collapsed to a
        # constant 0.16 against a true range of 0..1. Unbounded output keeps the
        # gradient alive; the range is clamped at inference instead.
        return out.sigmoid() if self.bounded else out


class Conditioner(nn.Module):
    """Severity-conditioned feature correction. Three nesting arms:

      concat  glue s_hat on, let an MLP sort it out (REAGVIS, CVPRW 2026 --
              our comparison point)
      gate    one scalar per vector; can discount evidence, never repair it
      film    per-channel scale and shift (Perez et al. 2018). Says "this
              channel is suppressed by this degradation, scale it back".

    They nest, so one implementation gives all three and the ablation is free.
    """

    def __init__(self, dim: int, mode: str = "film", hidden: int = 256):
        super().__init__()
        self.mode = mode
        if mode == "film":
            self.net = nn.Sequential(
                nn.Linear(N_OPS, hidden), nn.ReLU(), nn.Linear(hidden, 2 * dim)
            )
            # Start as identity: gamma=1, beta=0, so training begins from the
            # unconditioned model rather than a random perturbation of it.
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        elif mode == "gate":
            self.net = nn.Sequential(
                nn.Linear(N_OPS, hidden), nn.ReLU(), nn.Linear(hidden, 1)
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        elif mode == "concat":
            self.net = nn.Sequential(
                nn.Linear(dim + N_OPS, dim), nn.ReLU(), nn.Linear(dim, dim)
            )
        elif mode == "none":
            self.net = None
        else:
            raise ValueError(f"unknown conditioner: {mode}")

    def forward(self, h: torch.Tensor, s_hat: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return h
        if self.mode == "film":
            gamma, beta = self.net(s_hat).chunk(2, dim=-1)
            return (1.0 + gamma) * h + beta
        if self.mode == "gate":
            return (1.0 + self.net(s_hat)) * h
        return self.net(torch.cat([h, s_hat], dim=-1))
