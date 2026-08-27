"""Linear probe on frozen DINOv3 features -- the day-1 detector and baseline.

Serves three jobs at once: a working submission before any training happens,
row 1 of the robustness-gap table (max preservation / zero adaptation), and a
cheap check that the whole data path is sound.

Deliberately linear. The question at this stage is whether the frozen backbone
already separates real from fake -- a linear boundary answers that. If it cannot,
no amount of head capacity helps and the features themselves must change, which
is what PEFT is for.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import data, transforms as T
from .features import DEFAULT_BACKBONE, cache_path


def load_cell(root: Path, backbone: str, split: str, cell: str,
              variant: str = "aligned"):
    p = cache_path(root, backbone, split, cell, variant)
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    return d["features"], d["y"].astype(int), d["image_ids"]


def fit(features: np.ndarray, y: np.ndarray, C: float = 1.0):
    """Standardise then fit. ~1k parameters -- too few to memorise generators."""
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=C),
    )
    clf.fit(features, y)
    return clf


def evaluate(clf, root: Path, backbone: str, split: str = "val",
             variant: str = "aligned") -> pd.DataFrame:
    """Score one fitted probe across every cached battery cell."""
    rows = []
    for cell in T.BATTERY:
        loaded = load_cell(root, backbone, split, cell, variant)
        if loaded is None:
            continue
        feats, y, _ = loaded
        p = clf.predict_proba(feats)[:, 1]
        rows.append({
            "cell": cell,
            "n": len(y),
            "auc": roc_auc_score(y, p),
            "acc": accuracy_score(y, p > 0.5),
        })
    df = pd.DataFrame(rows)
    if len(df) and "clean" in set(df.cell):
        clean_auc = float(df.loc[df.cell == "clean", "auc"].iloc[0])
        df["gap_vs_clean"] = clean_auc - df.auc
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit and evaluate the linear probe")
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--variant", default="aligned", choices=["aligned", "raw"])
    ap.add_argument("--out", default="results/tables/probe_robustness.csv")
    args = ap.parse_args()

    root = data.data_root()
    train = load_cell(root, args.backbone, "train", "clean", args.variant)
    if train is None:
        raise SystemExit("No cached train features. Run: python -m aigcd.features")

    Xtr, ytr, _ = train
    clf = fit(Xtr, ytr, args.C)
    print(f"fitted on {Xtr.shape[0]} images, {Xtr.shape[1]}-dim, C={args.C}")

    df = evaluate(clf, root, args.backbone, "val", args.variant)
    if df.empty:
        raise SystemExit("No cached val cells to evaluate.")

    print()
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    missing = [c for c in T.BATTERY if c not in set(df.cell)]
    if missing:
        print(f"\n{len(missing)} battery cells not yet cached: {', '.join(missing)}")
        print("  python -m aigcd.features --splits val --cells all")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    # Persist the fitted probe so predict.py does not refit on every run.
    import joblib
    model_out = root / "cache" / "probe.joblib"
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "backbone": args.backbone}, model_out)
    print(f"\nwrote {out}  and  {model_out.relative_to(root)}")


if __name__ == "__main__":
    main()
