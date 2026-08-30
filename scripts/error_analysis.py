#!/usr/bin/env python
"""Deliverable 5: representative false positives, false negatives, trade-offs.

Scores the held-out benchmark clean and under laundering, then pulls out the
cases the model is most confidently wrong about -- those are the informative
ones. A near-boundary mistake tells you little; a real photograph called
AI-generated at 0.99 tells you what the model has actually learned.

Writes a contact sheet of both error types plus a per-condition breakdown.

    python scripts/error_analysis.py --checkpoint checkpoints/lora_r32_film.pt
"""
import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.components.peft import apply_peft          # noqa: E402
from src.models.detector import Detector            # noqa: E402
from src.utils import dataset as data               # noqa: E402
from src.utils import transforms as T               # noqa: E402
from src.utils.dataset import CellDataset           # noqa: E402
from src.utils.features import align_bias, pick_device  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CELLS = ("clean", "jpeg_30", "resize_0.25")


@torch.no_grad()
def score(model, df, preprocess, cell, root, device, bs, workers):
    dl = DataLoader(CellDataset(df, preprocess, cell, root, False),
                    batch_size=bs, num_workers=workers)
    return np.concatenate([torch.sigmoid(model(x.to(device))["logit"])
                           .float().cpu().numpy() for x, _ in dl])


def contact_sheet(rows, root, title, path, n=8):
    """Show the images themselves. A table of scores does not tell you what the
    model is confused BY."""
    rows = rows.head(n)
    if not len(rows):
        return
    cols = min(4, len(rows))
    r = int(np.ceil(len(rows) / cols))
    fig, axes = plt.subplots(r, cols, figsize=(3.2 * cols, 3.6 * r))
    for ax, (_, row) in zip(np.atleast_1d(axes).ravel(), rows.iterrows()):
        ax.imshow(align_bias(data.load_image(row.path, root), 96, 224))
        ax.set_title(f"p(AI)={row.p:.3f}\n{row.image_id[:26]}", fontsize=8)
        ax.axis("off")
    for ax in np.atleast_1d(axes).ravel()[len(rows):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    logger.info("wrote %s", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--manifest", type=Path, default=Path("data/heldout/manifest.csv"))
    ap.add_argument("--n", type=int, default=8, help="examples per error type")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or pick_device()
    root = data.data_root()
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    targs = ck["args"]
    bb = timm.create_model(targs["backbone"], pretrained=True, num_classes=0)
    bb = apply_peft(bb, mode=targs["arm"], r=targs["rank"])
    model = Detector(bb, dim=bb.num_features, conditioner=targs["conditioner"],
                     bounded_severity=not targs.get("unbounded_severity", False))
    model.load_state_dict(ck["state_dict"], strict=not ck.get("slim"))
    model = model.eval().to(device)
    cfg = timm.data.resolve_model_data_config(bb)
    preprocess = timm.data.create_transform(**cfg, is_training=False)

    df = pd.read_csv(args.manifest)
    df = df[df.split == data.SPLIT_HELDOUT].reset_index(drop=True)
    logger.info("scoring %d benchmark images across %d conditions", len(df), len(CELLS))

    scores = {}
    for cell in CELLS:
        scores[cell] = score(model, df, preprocess, cell, root, device,
                             args.batch_size, args.workers)
        acc = ((scores[cell] > .5) == df.y.to_numpy()).mean()
        logger.info("  %-12s accuracy %.4f", cell, acc)

    out_rows = []
    for cell, p in scores.items():
        d = df.assign(p=p, cell=cell)
        # Confidently wrong is what is worth looking at; a 0.51 mistake is noise.
        fp = d[(d.y == 0) & (d.p > .5)].sort_values("p", ascending=False)
        fn = d[(d.y == 1) & (d.p <= .5)].sort_values("p")
        logger.info("  %-12s %d false positives, %d false negatives",
                    cell, len(fp), len(fn))
        out_rows.append({"cell": cell, "n": len(d),
                         "accuracy": float(((p > .5) == d.y).mean()),
                         "false_positives": len(fp), "false_negatives": len(fn),
                         "fp_rate": float(len(fp) / (d.y == 0).sum()),
                         "fn_rate": float(len(fn) / (d.y == 1).sum())})
        if cell == "clean":
            contact_sheet(fp, root, "False positives — real photos called AI-generated",
                          "results/figures/error_false_positives.png", args.n)
            contact_sheet(fn, root, "False negatives — AI images called authentic",
                          "results/figures/error_false_negatives.png", args.n)
            d.to_csv("results/tables/error_scores_clean.csv", index=False)

    summary = pd.DataFrame(out_rows)
    print()
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    summary.to_csv("results/tables/error_analysis.csv", index=False)
    logger.info("wrote results/tables/error_analysis.csv")


if __name__ == "__main__":
    main()
