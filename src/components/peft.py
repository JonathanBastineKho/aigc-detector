"""Adaptation arms for the robustness-gap ablation. The comparison IS the point.

    frozen  no adaptation           full  everything trainable
    lora    low-rank, random init   svd   low-rank, MINOR-direction init
                                          (MiLoRA 2024 / Effort, ICML 2025)

Our claim, untested in this literature: DINOv3's pretraining is invariance to
blur/crop/scale/jitter -- our transform list -- and that invariance lives in the
PRINCIPAL singular directions. So a minor-direction adapter should preserve
robustness, not just semantic knowledge. Measured as clean AUC - laundered AUC.

The constraint is on DIRECTION, not rank: plain LoRA is low-rank but
directionally unconstrained, so gradient descent may overwrite the principal
directions. Minor-subspace init starts where the original map matters least.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn


class AdaptedLinear(nn.Module):
    """Frozen Linear plus trainable low-rank update: y = x(W + BA)^T + b.

    lora: A ~ N(0, 0.02), B = 0 -- adapter starts as a no-op.
    svd:  B, A carry the r smallest singular components, subtracted from the
          frozen base so the function is unchanged at init.
    """

    def __init__(self, base: nn.Linear, r: int = 32, mode: str = "svd", alpha: float = 1.0):
        super().__init__()
        self.r, self.mode, self.scale = r, mode, alpha
        out_f, in_f = base.weight.shape

        W = base.weight.data.float()
        if mode == "svd":
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            # Minor components: the r smallest singular values.
            Ur, Sr, Vhr = U[:, -r:], S[-r:], Vh[-r:, :]
            sqrt_s = torch.diag(Sr.sqrt())
            B = Ur @ sqrt_s                      # (out, r)
            A = sqrt_s @ Vhr                     # (r, in)
            # Remove the minor part from the frozen base: B@A restores it, so the
            # layer computes exactly W at initialisation.
            W = W - B @ A
        elif mode == "lora":
            A = torch.randn(r, in_f) * 0.02
            B = torch.zeros(out_f, r)
        else:
            raise ValueError(f"unknown mode: {mode}")

        # Frozen Parameters rather than buffers: buffers are invisible to
        # .parameters(), which would understate the model size -- and the <2B
        # budget claim depends on that count being right.
        self.weight = nn.Parameter(W.to(base.weight.dtype), requires_grad=False)
        self.bias = (
            nn.Parameter(base.bias.data.clone(), requires_grad=False)
            if base.bias is not None else None
        )
        self.lora_A = nn.Parameter(A.to(base.weight.dtype))
        self.lora_B = nn.Parameter(B.to(base.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = nn.functional.linear(x, self.weight, self.bias)
        return out + self.scale * nn.functional.linear(
            nn.functional.linear(x, self.lora_A), self.lora_B
        )


def _target_modules(model: nn.Module, patterns: tuple[str, ...]):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(p in name for p in patterns):
            yield name, module


def apply_peft(
    model: nn.Module,
    mode: str = "svd",
    r: int = 32,
    patterns: tuple[str, ...] = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"),
) -> nn.Module:
    """Freeze the backbone and install adapters; 'frozen'/'full' skip it.
    Targets attention and MLP projections, following TeleAI's NTIRE entry."""
    if mode == "frozen":
        for p in model.parameters():
            p.requires_grad_(False)
        return model
    if mode == "full":
        for p in model.parameters():
            p.requires_grad_(True)
        return model

    for p in model.parameters():
        p.requires_grad_(False)

    for name, module in list(_target_modules(model, patterns)):
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        setattr(parent, name.rsplit(".", 1)[-1], AdaptedLinear(module, r, mode))
    return model


@contextmanager
def adapters_disabled(model: nn.Module):
    """Temporarily restore the un-adapted backbone.

    AdaptedLinear computes W.x + scale*(BA).x, and the frozen W is the original
    pretrained weight, so scale=0 recovers the backbone as downloaded -- without
    a second 1.2 GB copy in memory. Lets one loaded model serve as both the
    trained detector and the frozen baseline.
    """
    mods = [m for m in model.modules() if isinstance(m, AdaptedLinear)]
    saved = [m.scale for m in mods]
    for m in mods:
        m.scale = 0.0
    try:
        yield model
    finally:
        for m, sc in zip(mods, saved):
            m.scale = sc


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
