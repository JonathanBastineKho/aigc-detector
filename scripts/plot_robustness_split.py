#!/usr/bin/env python
"""The robustness result as a contrast, not a 21-row table.

Colour and mild operations barely register; structural degradation costs
several points. That split is the finding -- the detector reads spatial
structure rather than texture statistics -- and a two-column layout says it
faster than a table.

Axis is truncated to 0.90-1.00 and labelled as such: the whole result lives in
a five-point band, and a 0-1 axis would render every bar identical.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SURVIVES = [("jpeg_90", "JPEG 90"), ("blur_0.5", "Blur σ0.5"),
            ("jitter_0.2", "Colour jitter ±20%"), ("cast_0.2", "Colour cast"),
            ("crop_0.8", "Centre crop 80%")]
HURTS = [("jpeg_30", "JPEG 30"), ("blur_2.0", "Blur σ2.0"),
         ("resize_0.25", "Resize 0.25×"), ("noise_0.10", "Noise σ0.10"),
         ("heavy_launder", "5-op chain")]

GREEN, RED, TRACK = "#1B6B3A", "#C1123A", "#EDF1F5"
LO, HI = 0.90, 1.00


def panel(ax, rows, colour, title, clean):
    ax.set_title(title, fontsize=13, color=colour, fontweight="bold", loc="left", pad=14)
    for i, (label, auc) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, 1.0, color=TRACK, height=.52)
        frac = (auc - LO) / (HI - LO)
        ax.barh(y, frac, color=colour, height=.52)
        # Labels sit inside the axes: with two adjacent panels, text hanging
        # off the edges collides across the gutter.
        ax.text(-0.02, y, label, ha="right", va="center", fontsize=11)
        inside = frac > 0.25
        ax.text(frac - 0.02 if inside else frac + 0.02, y, f"{auc:.4f}",
                ha="right" if inside else "left", va="center", fontsize=11,
                color="white" if inside else colour, fontweight="bold")
    ax.axvline((clean - LO) / (HI - LO), color="#555", ls=":", lw=1.2)
    ax.set_xlim(0, 1); ax.set_ylim(-.6, len(rows) - .4)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval", default="results/tables/eval_lora_r32_film_heldout.csv")
    ap.add_argument("--out", default="results/figures/robustness_split.png")
    args = ap.parse_args()

    d = pd.read_csv(args.eval).set_index("cell").auc
    clean = float(d["clean"])

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.2),
                             gridspec_kw={"wspace": 0.55})
    panel(axes[0], [(lab, float(d[c])) for c, lab in SURVIVES if c in d],
          GREEN, "Barely registers", clean)
    panel(axes[1], [(lab, float(d[c])) for c, lab in HURTS if c in d],
          RED, "Costs real accuracy", clean)

    fig.suptitle(f"ROC-AUC on the held-out benchmark   ·   clean = {clean:.4f}   "
                 f"·   axis {LO:.2f}–{HI:.2f}",
                 fontsize=11, color="#444", y=1.02)
    fig.text(.5, -.06, "Dotted line marks clean performance. The detector reads "
                       "spatial structure: colour operations cost it almost "
                       "nothing, resolution loss costs several points.",
             ha="center", fontsize=10, color="#555", style="italic")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
