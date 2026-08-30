#!/usr/bin/env python
"""Measure the model's own severity head, and fit a ridge as a fallback.

Two questions, one script:

  1. Does the severity head INSIDE the model estimate severity? (correlation
     and output range across the battery)
  2. If not, can a ridge on the same features do better? (the ceiling -- what is
     extractable from this representation at all)

If (1) works, the model conditions on its own estimate and the ridge is unused.
If (1) collapses but (2) works, the failure is optimisation rather than
representation, and the ridge drives abstention instead.

The jointly-trained severity head collapsed to predicting the mean -- 0.396
correlation across a 0.025 output range, against a true range of 0..1. The
information IS in the features: a ridge regression on the same representation
reaches 0.66 with full range. The failure is optimisation, not representation.

So we fit the estimator separately. This touches nothing in the detector: the
classifier path, and therefore every benchmark number, is unchanged. The output
is used only for the abstention decision.

    python scripts/fit_severity.py --checkpoint checkpoints/lora_r32_film_slim.pt
"""
import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import timm
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.components.peft import apply_peft          # noqa: E402
from src.models.detector import Detector            # noqa: E402
from src.utils import dataset as data               # noqa: E402
from src.utils import transforms as T               # noqa: E402
from src.utils.dataset import CellDataset           # noqa: E402
from src.utils.features import pick_device          # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build(checkpoint: Path, device: str):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    targs = ck["args"]
    bb = timm.create_model(targs["backbone"], pretrained=True, num_classes=0)
    bb = apply_peft(bb, mode=targs["arm"], r=targs["rank"])
    model = Detector(bb, dim=bb.num_features, conditioner=targs["conditioner"],
                     bounded_severity=not targs.get("unbounded_severity", False))
    model.load_state_dict(ck["state_dict"], strict=not ck.get("slim"))
    cfg = timm.data.resolve_model_data_config(bb)
    return (model.eval().to(device),
            timm.data.create_transform(**cfg, is_training=False),
            targs["backbone"])


@torch.no_grad()
def features_for(model, df, preprocess, cell, root, device, bs, workers, pre):
    """Returns (pre-FiLM features, the head's own severity estimate)."""
    dl = DataLoader(CellDataset(df, preprocess, cell, root, pre),
                    batch_size=bs, num_workers=workers)
    H, S = [], []
    for x, _ in dl:
        out = model(x.to(device))
        H.append(out["h"].float().cpu().numpy())
        S.append(out["s_hat"].float().cpu().numpy())
    return np.concatenate(H), np.concatenate(S)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--n", type=int, default=400, help="val images per battery cell")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or pick_device()
    root = data.data_root()
    model, preprocess, backbone_name = build(args.checkpoint, device)

    if args.manifest:
        mf = pd.read_csv(args.manifest); pre = mf.path.iloc[0].startswith("extracted/")
    else:
        mf, pre = data.build_manifest(root), False
    val = mf[mf.split == data.SPLIT_VAL].head(args.n)
    logger.info("fitting on %d images x %d cells", len(val), len(T.BATTERY))

    from PIL import Image
    dummy = Image.new("RGB", (224, 224))

    X, Y, S = [], [], []
    for cell in T.BATTERY:
        op, kw = T.BATTERY[cell]
        _, log = T.apply_chain(dummy, [] if op == "none" else [(op, kw)])
        target = T.severity_vector(log)
        feats, s_hat = features_for(model, val, preprocess, cell, root, device,
                                    args.batch_size, args.workers, pre)
        X.append(feats); Y.append(np.tile(target, (len(feats), 1))); S.append(s_hat)
        logger.info("  %-12s true %.3f   head estimates %.3f",
                    cell, target.max(), float(np.mean(s_hat.max(1))))

    X, Y, S = np.concatenate(X), np.concatenate(Y), np.concatenate(S)

    # --- question 1: does the model's OWN head work? -----------------------
    head_corr = float(np.corrcoef(Y.max(1), S.max(1))[0, 1])
    head_lo, head_hi = float(S.max(1).min()), float(S.max(1).max())
    logger.info("")
    logger.info("BUILT-IN HEAD   correlation %.3f   range %.3f-%.3f",
                head_corr, head_lo, head_hi)
    if head_hi - head_lo < 0.15:
        logger.warning("  -> collapsed: the head is predicting a near-constant")
    else:
        logger.info("  -> working: FiLM has a real signal to condition on")
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.3, random_state=0)
    ridge = Ridge(alpha=1.0).fit(Xtr, Ytr)

    P = ridge.predict(Xte)
    corr = float(np.corrcoef(Yte.max(1), P.max(1))[0, 1])
    logger.info("RIDGE FALLBACK  correlation %.3f   range %.3f-%.3f",
                corr, P.max(1).min(), P.max(1).max())
    logger.info("")
    logger.info("verdict: %s", "the head works -- ridge not needed"
                if head_corr > 0.55 and head_hi - head_lo > 0.15
                else "the head collapsed -- abstention should use the ridge")

    out = root / "cache" / f"severity_{args.checkpoint.stem}.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"ridge": ridge, "backbone": backbone_name,
                 "checkpoint": args.checkpoint.name, "correlation": corr,
                 "head_correlation": head_corr,
                 "head_range": [head_lo, head_hi]}, out)
    logger.info("wrote %s", out)


if __name__ == "__main__":
    import pandas as pd  # noqa: E402  (only needed with --manifest)
    main()
