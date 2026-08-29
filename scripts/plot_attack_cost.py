#!/usr/bin/env python
"""Survival curves: how much laundering does evasion actually cost?

Standard robustness plots show AUC against a fixed list of transforms. This
shows the adversary's view -- what fraction of correct predictions survive as
the attacker's budget increases. The gap between two curves is the extra damage
an evader must accept to defeat the better model.

Reads the per-image costs written by attack_cost.py, so it needs no GPU.

    python scripts/plot_attack_cost.py
"""
import argparse
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Severity is normalised 0..1; these translate it back into something an
# attacker would recognise (see severity_vector in utils/transforms.py).
def as_jpeg(s):  return 95 - 70 * s
def as_scale(s): return 1 - 0.75 * s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="results/tables/attack_cost_per_image_*.csv")
    ap.add_argument("--out", default="results/figures/attack_cost.png")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"no per-image files matching {args.glob}. "
                         "Run attack_cost.py first (needs the updated version).")

    budgets = np.linspace(0, 1, 101)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    table = []
    for f in files:
        name = re.sub(r".*per_image_|_r32_film|_heldout|\.csv", "", f)
        d = pd.read_csv(f)
        cost = d.attack_cost.to_numpy()
        cost = np.nan_to_num(cost, nan=np.inf)          # never flipped = infinite cost

        survival = [(cost > b).mean() for b in budgets]
        ax.plot(budgets, survival, lw=2, label=name)

        # Fakes only: laundering is a one-way attack, and the fake curve is the
        # one an adversary cares about.
        cf = cost[d.y == 1]
        ax2.plot(budgets, [(cf > b).mean() for b in budgets], lw=2, label=name)

        table.append({"model": name,
                      **{f"survive@{b}": float((cost > b).mean())
                         for b in (0.2, 0.4, 0.6, 0.8)},
                      "never_flipped": float(np.isinf(cost).mean())})

    for a, title in ((ax, "All correct predictions"), (ax2, "AI images only")):
        a.set_xlabel("laundering budget (severity)")
        a.set_ylabel("fraction still classified correctly")
        a.set_title(title, fontsize=11)
        a.grid(alpha=.3)
        a.legend(fontsize=9)
        a.set_ylim(0, 1.02)
        top = a.secondary_xaxis("top", functions=(lambda s: s, lambda s: s))
        top.set_xticks([0.2, 0.4, 0.6, 0.8])
        top.set_xticklabels([f"JPEG {as_jpeg(b):.0f}\n{as_scale(b):.2f}x"
                             for b in (0.2, 0.4, 0.6, 0.8)], fontsize=8)

    fig.suptitle("Attack cost: how much visible damage evasion requires", fontsize=13)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")

    df = pd.DataFrame(table)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    df.to_csv("results/tables/attack_cost_survival.csv", index=False)
    print(f"\nwrote {out} and results/tables/attack_cost_survival.csv")


if __name__ == "__main__":
    main()
