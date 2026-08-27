"""WaRPAD: training-free detection via high-frequency perturbation sensitivity.

Choi et al., NeurIPS 2025 (arXiv 2511.14030). Not our method; cite it.

    HFwav(x)  = cos( f(x), f(x - a*HF(x)) )          a = 0.1, on the RAW image
    WaRPAD(x) = mean over patches of HFwav(x_patch)

Higher score = embedding barely moved = more likely REAL. Paper reports mean
AUROC 0.834 on Synthbuster with DINOv2 ViT-L/14 (range 0.422-0.999).

Ours is the DINOv3 port and the measurement the paper does not make: where this
cue dies along a laundering chain.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image

D_RESCALE, D_PATCH, ALPHA = 896, 224, 0.1


def haar_highfreq(x: torch.Tensor) -> torch.Tensor:
    """High-frequency component via one Haar level (LL zeroed)."""
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    lh, hl, hh = (a + b - c - d) / 2, (a - b + c - d) / 2, (a - b - c + d) / 2
    out = torch.zeros_like(x)
    out[..., 0::2, 0::2] = (lh + hl + hh) / 2
    out[..., 0::2, 1::2] = (lh - hl - hh) / 2
    out[..., 1::2, 0::2] = (-lh + hl - hh) / 2
    out[..., 1::2, 1::2] = (-lh - hl + hh) / 2
    return out


def to_patches(img: Image.Image, d_rescale: int = D_RESCALE,
               d_patch: int = D_PATCH) -> torch.Tensor:
    """Rescale to d_rescale square, split into (d_rescale/d_patch)^2 patches.

    Returns raw [0,1] tensors -- the perturbation is applied before the model's
    mean/std normalisation, so alpha means what the paper says it means.
    """
    img = img.convert("RGB").resize((d_rescale, d_rescale), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0
    n = d_rescale // d_patch
    return (x.unfold(1, d_patch, d_patch).unfold(2, d_patch, d_patch)
             .permute(1, 2, 0, 3, 4).reshape(n * n, 3, d_patch, d_patch))


def _cls(backbone, x: torch.Tensor) -> torch.Tensor:
    """CLS token, not pooled features. DINOv3 carries register tokens, so index
    0 of the prefix block rather than assuming a single leading token."""
    tokens = backbone.forward_features(x)
    return tokens[:, 0] if tokens.ndim == 3 else tokens


@torch.no_grad()
def warpad_score(
    imgs: list[Image.Image],
    backbone,
    device: str,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
    alpha: float = ALPHA,
    d_rescale: int = D_RESCALE,
    batch: int = 16,
) -> np.ndarray:
    """Mean cosine similarity between clean and high-frequency-suppressed views.

    HIGH => stable under the perturbation => more likely real.
    """
    m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=device).view(1, 3, 1, 1)
    out = []
    for img in imgs:
        p = to_patches(img, d_rescale).to(device)
        q = p - alpha * haar_highfreq(p)          # subtract, per the paper

        sims = []
        for i in range(0, len(p), batch):
            f0 = Fn.normalize(_cls(backbone, (p[i:i + batch] - m) / s), dim=-1)
            f1 = Fn.normalize(_cls(backbone, (q[i:i + batch] - m) / s), dim=-1)
            sims.append((f0 * f1).sum(-1))
        out.append(float(torch.cat(sims).mean()))
    return np.array(out)
