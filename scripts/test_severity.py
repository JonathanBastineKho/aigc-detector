#!/usr/bin/env python
"""Does the model's own severity head track real laundering?

The head is trained to answer "how damaged is this image". Whether it learned
that, or just settled on the average, decides three things: FiLM has something
to condition on, the abstention rule can fire, and the demo can escalate.

Feeds images at known severities and compares the head's output to the truth.
Pass several checkpoints to compare them side by side.

    python scripts/test_severity.py --checkpoints checkpoints/lora_r32_film.pt \\
                                                  checkpoints/lora_r32_film_v2.pt
"""
import argparse
import logging
import sys
from pathlib import Path

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
from src.utils.features import pick_device          # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load(checkpoint: Path, device: str):
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
            bool(targs.get("unbounded_severity", False)))


@torch.no_grad()
def estimate(model, df, preprocess, cell, root, device, bs, workers, pre):
    dl = DataLoader(CellDataset(df, preprocess, cell, root, pre),
                    batch_size=bs, num_workers=workers)
    return np.concatenate([model(x.to(device))["s_hat"].float().cpu().numpy()
                           for x, _ in dl])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", required=True, nargs="+", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--n", type=int, default=200, help="val images per cell")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or pick_device()
    root = data.data_root()

    if args.manifest:
        mf = pd.read_csv(args.manifest); pre = mf.path.iloc[0].startswith("extracted/")
    else:
        mf, pre = data.build_manifest(root), False
    val = mf[mf.split == data.SPLIT_VAL].head(args.n)

    from PIL import Image
    dummy = Image.new("RGB", (224, 224))

    rows = []
    for ckpt in args.checkpoints:
        model, preprocess, unbounded = load(ckpt, device)
        logger.info("%s  (severity head: %s)", ckpt.name,
                    "unbounded" if unbounded else "sigmoid")

        true, pred = [], []
        for cell in T.BATTERY:
            op, kw = T.BATTERY[cell]
            _, log = T.apply_chain(dummy, [] if op == "none" else [(op, kw)])
            t = float(T.severity_vector(log).max())
            s_hat = estimate(model, val, preprocess, cell, root, device,
                             args.batch_size, args.workers, pre)
            p = float(np.clip(s_hat, 0, None).max(1).mean())
            true.append(t); pred.append(p)
            logger.info("    %-12s true %.3f   estimated %.3f", cell, t, p)

        true, pred = np.array(true), np.array(pred)
        rows.append({
            "checkpoint": ckpt.stem,
            "severity_head": "unbounded" if unbounded else "sigmoid",
            "correlation": float(np.corrcoef(true, pred)[0, 1]),
            "pred_min": float(pred.min()), "pred_max": float(pred.max()),
            # The range is the tell. A collapsed head can still show moderate
            # correlation while varying by 0.02 -- useless for a threshold.
            "pred_range": float(pred.max() - pred.min()),
        })

    out = pd.DataFrame(rows)
    print()
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    out.to_csv("results/tables/severity_head_test.csv", index=False)
    print("\nA usable head needs correlation > ~0.6 AND range > ~0.3.")
    print("Correlation alone is not enough: a flat head can still correlate.")


if __name__ == "__main__":
    main()
