#!/usr/bin/env python
"""Why laundering breaks a detector that still 'knows' the answer.

Laundering does not erase the evidence -- AUC barely moves. It shifts the whole
score distribution toward "real" while preserving the ordering, so a fixed
threshold silently stops working. This figure is the empirical justification for
severity-conditioned correction: the right threshold is a function of how
laundered the image is.
"""
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.utils import dataset as data, transforms as T          # noqa: E402
from src.utils.probe import load_cell                 # noqa: E402

BB = "vit_large_patch16_dinov3.lvd1689m"
SHOW = ["clean", "jpeg_70", "jpeg_30", "noise_0.10"]


def main() -> None:
    root = data.data_root()
    clf = joblib.load(root / "cache" / "probe.joblib")["clf"]

    scores = {}
    for cell in T.BATTERY:
        got = load_cell(root, BB, "val", cell)
        if got is None:
            continue
        X, y, _ = got
        scores[cell] = (clf.predict_proba(X)[:, 1], y)

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.42, wspace=0.28)

    # --- top row: score distributions degrading left to right ---------------
    for i, cell in enumerate(SHOW):
        ax = fig.add_subplot(gs[0, i])
        p, y = scores[cell]
        bins = np.linspace(0, 1, 41)
        ax.hist(p[y == 0], bins=bins, alpha=.75, label="real", color="#2E7D9A")
        ax.hist(p[y == 1], bins=bins, alpha=.75, label="AI", color="#C1483A")
        ax.axvline(.5, color="k", ls="--", lw=1.4)
        acc = ((p > .5) == y).mean()
        ax.set_title(f"{cell}\nacc @ 0.5 = {acc:.3f}", fontsize=10)
        ax.set_yscale("log")
        ax.set_xlabel("p(AI)")
        if i == 0:
            ax.set_ylabel("count (log)")
            ax.legend(fontsize=8, loc="upper center")

    # --- bottom left: AUC holds while accuracy falls -------------------------
    ax = fig.add_subplot(gs[1, :2])
    cells = list(scores)
    auc, a05, abest, thr = [], [], [], []
    for c in cells:
        p, y = scores[c]
        order = np.argsort(p)
        ps, ys = p[order], y[order]
        # AUC via rank statistic
        r = np.arange(1, len(ps) + 1)
        n1, n0 = ys.sum(), (1 - ys).sum()
        auc.append((r[ys == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
        a05.append(((p > .5) == y).mean())
        ts = np.quantile(p, np.linspace(.01, .99, 199))
        accs = [((p > t) == y).mean() for t in ts]
        k = int(np.argmax(accs))
        abest.append(accs[k]); thr.append(ts[k])

    order = np.argsort(a05)[::-1]
    x = np.arange(len(cells))
    ax.plot(x, np.array(auc)[order], "o-", label="AUC (ranking quality)", color="#2E7D9A")
    ax.plot(x, np.array(a05)[order], "s-", label="accuracy @ fixed 0.5", color="#C1483A")
    ax.plot(x, np.array(abest)[order], "^--", label="accuracy @ oracle threshold",
            color="#6A9955", alpha=.85)
    ax.set_xticks(x); ax.set_xticklabels([cells[i] for i in order], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("score"); ax.grid(alpha=.3)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_title("The evidence survives; the decision does not", fontsize=11)

    # --- bottom right: the threshold that would have worked ------------------
    ax = fig.add_subplot(gs[1, 2:])
    ax.plot(x, np.array(thr)[order], "D-", color="#8250C4")
    ax.axhline(.5, color="k", ls="--", lw=1.4, label="fixed threshold 0.5")
    ax.set_xticks(x); ax.set_xticklabels([cells[i] for i in order], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("accuracy-optimal threshold"); ax.grid(alpha=.3)
    ax.legend(fontsize=9)
    ax.set_title("The right threshold is a function of severity", fontsize=11)

    fig.suptitle("Laundering shifts scores toward 'real' without destroying the ranking",
                 fontsize=13, y=.975)
    out = Path("results/figures/threshold_vs_severity.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
