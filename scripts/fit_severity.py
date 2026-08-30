#!/usr/bin/env python
"""Fit a severity estimator that actually works, separately from the model.

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
    """Pre-FiLM pooled features -- the same representation the severity head sees."""
    dl = DataLoader(CellDataset(df, preprocess, cell, root, pre),
                    batch_size=bs, num_workers=workers)
    return np.concatenate([model(x.to(device))["h"].float().cpu().numpy()
                           for x, _ in dl])


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

    X, Y = [], []
    for cell in T.BATTERY:
        op, kw = T.BATTERY[cell]
        _, log = T.apply_chain(dummy, [] if op == "none" else [(op, kw)])
        target = T.severity_vector(log)
        feats = features_for(model, val, preprocess, cell, root, device,
                             args.batch_size, args.workers, pre)
        X.append(feats); Y.append(np.tile(target, (len(feats), 1)))
        logger.info("  %-12s %d samples, target max %.3f", cell, len(feats), target.max())

    X, Y = np.concatenate(X), np.concatenate(Y)
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.3, random_state=0)
    ridge = Ridge(alpha=1.0).fit(Xtr, Ytr)

    P = ridge.predict(Xte)
    corr = float(np.corrcoef(Yte.max(1), P.max(1))[0, 1])
    logger.info("held-out correlation %.3f   predicted range %.3f-%.3f",
                corr, P.max(1).min(), P.max(1).max())

    out = root / "cache" / f"severity_{args.checkpoint.stem}.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"ridge": ridge, "backbone": backbone_name,
                 "checkpoint": args.checkpoint.name, "correlation": corr}, out)
    logger.info("wrote %s", out)


if __name__ == "__main__":
    import pandas as pd  # noqa: E402  (only needed with --manifest)
    main()
