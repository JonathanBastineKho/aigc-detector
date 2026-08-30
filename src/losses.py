"""Training objectives.

    L = BCE(clean) + BCE(laundered)
        + a * KL(p || p~)        a = 0.50   same verdict either way
        + b * MSE(h, h~)         b = 0.25   same features either way
        + c * SmoothL1(s^, s)    c = 0.50   severity, supervised for free

Coefficients are TeleAI's published NTIRE 2026 values. Both BCE terms carry the
SAME label -- laundering does not change whether an image was generated, and
teaching that invariance is the entire point.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

A_KL, B_FEAT, C_SEV = 0.50, 0.25, 0.50

# Severity targets are sparse -- ~25% of samples are clean (all zeros) and the
# rest activate 1-4 of 8 dimensions -- so predicting a small constant nearly
# minimises SmoothL1 on its own. At C_SEV=0.50 the head took that shortcut.
# Raising the weight makes the shortcut expensive relative to detection.
C_SEV_STRONG = 2.00


def symmetric_kl(logit_a: torch.Tensor, logit_b: torch.Tensor) -> torch.Tensor:
    """Agreement between the clean and laundered verdicts (DCPT / DOCL)."""
    p_a = torch.stack([logit_a, -logit_a], -1).log_softmax(-1)
    p_b = torch.stack([logit_b, -logit_b], -1).log_softmax(-1)
    return 0.5 * (F.kl_div(p_b, p_a, log_target=True, reduction="batchmean")
                  + F.kl_div(p_a, p_b, log_target=True, reduction="batchmean"))


def detection_loss(
    out_clean: dict,
    out_laundered: dict,
    y: torch.Tensor,
    severity: torch.Tensor,
    a: float = A_KL,
    b: float = B_FEAT,
    c: float = C_SEV,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Returns (total, per-term dict for logging)."""
    l_det = (F.binary_cross_entropy_with_logits(out_clean["logit"], y)
             + F.binary_cross_entropy_with_logits(out_laundered["logit"], y))

    l_kl = symmetric_kl(out_clean["logit"], out_laundered["logit"])

    # Clean features are the target, so they are detached: the laundered view
    # moves toward the clean one, not both toward some easy midpoint.
    l_feat = F.mse_loss(out_laundered["h"], out_clean["h"].detach())

    # Severity is only defined for the laundered view -- the clean view is
    # all-zero by construction, and supervising it teaches the head to output 0.
    l_sev = F.smooth_l1_loss(out_laundered["s_hat"], severity)

    total = l_det + a * l_kl + b * l_feat + c * l_sev
    return total, {"det": l_det.item(), "kl": l_kl.item(),
                   "feat": l_feat.item(), "sev": l_sev.item()}


def supcon_loss(features: torch.Tensor, labels: torch.Tensor,
                temperature: float = 0.07) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al. 2020).

    Optional fifth ablation arm. BCE finds a boundary that separates the
    generators seen in training; contrastive shapes the whole geometry, and
    geometry is what transfers to generators never seen. Negatives come from the
    batch, so this wants a large batch size to be worth much.
    """
    f = F.normalize(features, dim=-1)
    sim = f @ f.T / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()

    same = (labels[:, None] == labels[None, :]).float()
    eye = torch.eye(len(f), device=f.device)
    same, mask = same - same * eye, 1.0 - eye

    log_prob = sim - torch.log((mask * sim.exp()).sum(1, keepdim=True) + 1e-12)
    pos = (same * log_prob).sum(1) / same.sum(1).clamp(min=1)
    return -pos.mean()
