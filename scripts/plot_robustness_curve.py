#!/usr/bin/env python
"""AUC against actual perceptual damage, rather than against transform names.

A table of 21 named conditions tells you what happened but not how they relate.
Measuring how much each transform actually degrades the image (1 - SSIM) puts
them all on one axis, so the shape of the degradation is visible and models can
be compared directly.

Needs no GPU -- AUCs come from the eval tables, damage from the images.

    python scripts/plot_robustness_curve.py
"""
import argparse
import glob
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.utils import dataset as data          # noqa: E402
from src.utils import transforms as T          # noqa: E402
from src.utils.features import align_bias      # noqa: E402


def damage_per_cell(root, manifest, n=60):
    """Mean 1 - SSIM against the untouched image, per battery cell and chain."""
    df = pd.read_csv(manifest)
    df = df[df.split == data.SPLIT_HELDOUT].sample(n, random_state=0)
    imgs = [align_bias(data.load_image(r.path, root)) for r in df.itertuples()]

    out = {}
    for name in list(T.BATTERY) + list(T.CHAIN_BATTERY):
        if name in T.CHAIN_BATTERY:
            chain = T.CHAIN_BATTERY[name]
        else:
            op, kw = T.BATTERY[name]
            chain = [] if op == "none" else [(op, kw)]
        vals = []
        for im in imgs:
            o = np.asarray(im.convert("L"), dtype=np.float32)
            laundered, _ = T.apply_chain(im, chain)
            l = np.asarray(laundered.convert("L"), dtype=np.float32)
            vals.append(1.0 - structural_similarity(o, l, data_range=255.0))
        out[name] = float(np.mean(vals))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/heldout/manifest.csv")
    ap.add_argument("--evals", nargs="+",
                    default=sorted(glob.glob("results/tables/eval_*heldout*.csv")))
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default="results/figures/robustness_curve.png")
    args = ap.parse_args()

    root = data.data_root()
    dmg = damage_per_cell(root, args.manifest, args.n)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.8, len(args.evals)))

    for f, c in zip(args.evals, colors):
        name = re.sub(r".*eval_|_r32_film|_heldout|\.csv", "", f) or "model"
        d = pd.read_csv(f)
        d = d[d.cell.isin(dmg)].copy()
        d["damage"] = d.cell.map(dmg)
        d = d.sort_values("damage")

        # SSIM compares pixels position by position, so a crop reads as
        # catastrophic (0.65) while leaving the content intact and the detector
        # almost unaffected. Those points sit far off the trend for a reason
        # that is about the metric, not the model -- plot them apart.
        is_crop = d.cell.str.contains("crop")
        singles = d[(d.kind != "chain") & ~is_crop] if "kind" in d.columns else d[~is_crop]
        chains = d[(d.kind == "chain") & ~is_crop] if "kind" in d.columns else d.iloc[0:0]
        crops = d[is_crop]

        ax.plot(singles.damage, singles.auc, "o-", color=c, lw=2, ms=5, label=name)
        if len(chains):
            ax.scatter(chains.damage, chains.auc, marker="X", s=90,
                       color=c, edgecolor="k", linewidth=.6, zorder=5)
        if len(crops):
            ax.scatter(crops.damage, crops.auc, marker="s", s=55, facecolor="none",
                       edgecolor=c, linewidth=1.4, zorder=5)

    for cell in ("clean", "jpeg_30", "resize_0.25", "noise_0.10"):
        if cell in dmg:
            ax.annotate(cell, (dmg[cell], ax.get_ylim()[0]), rotation=90,
                        fontsize=7, alpha=.55, va="bottom", ha="center")

    ax.set_xlabel("perceptual damage to the image   (1 − SSIM)")
    ax.set_ylabel("ROC-AUC on the held-out benchmark")
    ax.set_title("Detection degrades gracefully with visible damage\n"
                 "● single transforms    ✕ composed chains    □ crops "
                 "(SSIM over-reads spatial shift)", fontsize=10)
    ax.grid(alpha=.3)
    ax.legend(fontsize=9)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")

    tbl = pd.DataFrame({"cell": list(dmg), "damage": list(dmg.values())}).sort_values("damage")
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    tbl.to_csv("results/tables/cell_damage.csv", index=False)


if __name__ == "__main__":
    main()
