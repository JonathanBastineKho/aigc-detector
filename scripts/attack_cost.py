"""How much laundering does it take to break a correct prediction?

Standard robustness evaluation asks "what is the AUC at JPEG-50?" -- performance
against a fixed menu someone else chose. This asks the adversary's question:
given an image the detector currently gets RIGHT, what is the cheapest
laundering that flips it?

Reported as attack cost in [0,1] (the severity encoding), so a higher median
means an evader must do more visible damage to their own image to get past.

Prior work measures evasion in L-infinity perturbation budgets with gradient
attacks. Real evaders do not compute gradients against your weights -- they
re-upload and screenshot. This measures that instead.

    python scripts/attack_cost.py --checkpoint checkpoints/svd_r32_film.pt
    python scripts/attack_cost.py --probe            # the frozen-probe baseline
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.components.peft import apply_peft            # noqa: E402
from src.models.detector import Detector              # noqa: E402
from src.utils import dataset as data                 # noqa: E402
from src.utils import transforms as T                 # noqa: E402
from src.utils.features import align_bias, load_backbone, pick_device  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Each ladder is one laundering strategy, ordered cheapest to most damaging.
# An evader picks whichever is cheapest for a given image, so the attack cost is
# the minimum across ladders, not the average.
LADDERS = {
    "jpeg":   [("jpeg", {"quality": q}) for q in (95, 90, 80, 70, 60, 50, 40, 30, 25)],
    "resize": [("resize", {"scale": s}) for s in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25)],
    "blur":   [("blur", {"sigma": s}) for s in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)],
    "noise":  [("noise", {"sigma": s}) for s in (0.01, 0.02, 0.05, 0.08, 0.10, 0.15)],
    # A realistic multi-op chain: repost, crop, thumbnail, repost again. Each
    # rung adds an operation rather than intensifying one.
    "chain": [
        ("jpeg", {"quality": 85}),
        ("crop", {"frac": 0.9}),
        ("resize", {"scale": 0.7}),
        ("jpeg", {"quality": 50}),
        ("resize", {"scale": 0.4}),
        ("jpeg", {"quality": 30}),
    ],
}


class Rung(Dataset):
    """Images with one ladder rung applied. Cumulative for chains, single otherwise."""

    def __init__(self, rows, preprocess, ops, root, pre_extracted):
        self.rows, self.preprocess, self.ops = rows, preprocess, ops
        self.root, self.pre = root, pre_extracted

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows.iloc[i]
        img = data.load_image(row.path, self.root)
        if not self.pre:
            img = align_bias(img)
        img, log = T.apply_chain(img, self.ops)
        return self.preprocess(img), float(T.severity_vector(log).max())


@torch.no_grad()
def scores_at(model_fn, rows, preprocess, ops, root, pre, device, bs, workers):
    loader = DataLoader(Rung(rows, preprocess, ops, root, pre),
                        batch_size=bs, num_workers=workers)
    out, cost = [], []
    for x, sev in loader:
        out.append(model_fn(x.to(device)))
        cost.append(sev.numpy())
    return np.concatenate(out), np.concatenate(cost)


def search(model_fn, rows, preprocess, root, pre, device, bs, workers):
    """Per-image minimum severity that flips a correct prediction.

    Walks each ladder rung by rung, dropping images as they flip, so each rung
    costs one forward pass over only the images still surviving.
    """
    y = rows.y.to_numpy()
    best = np.full(len(rows), np.nan)          # nan = never flipped

    for name, rungs in LADDERS.items():
        alive = np.ones(len(rows), dtype=bool)
        for k, op in enumerate(rungs):
            if not alive.any():
                break
            # Chains accumulate; single-op ladders replace.
            ops = rungs[:k + 1] if name == "chain" else [op]
            p, cost = scores_at(model_fn, rows[alive], preprocess, ops,
                                root, pre, device, bs, workers)
            flipped = (p > 0.5) != y[alive]

            idx = np.where(alive)[0][flipped]
            for j, c in zip(idx, cost[flipped]):
                best[j] = c if np.isnan(best[j]) else min(best[j], c)
            alive[idx] = False
        logger.info("  ladder %-7s -> %d/%d flipped", name,
                    int((~np.isnan(best)).sum()), len(rows))
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--probe", action="store_true", help="use the frozen linear probe")
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--n", type=int, default=500, help="images to attack")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or pick_device()
    root = data.data_root()

    if args.probe:
        import joblib
        bundle = joblib.load(root / "cache" / "probe.joblib")
        backbone, preprocess = load_backbone(bundle["backbone"], device)
        clf, name = bundle["clf"], "probe"

        def model_fn(x):
            return clf.predict_proba(backbone(x).float().cpu().numpy())[:, 1]
    else:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        targs = ckpt["args"]
        bb = timm.create_model(targs["backbone"], pretrained=True, num_classes=0)
        bb = apply_peft(bb, mode=targs["arm"], r=targs["rank"])
        model = Detector(bb, dim=bb.num_features, conditioner=targs["conditioner"])
        model.load_state_dict(ckpt["state_dict"])
        model = model.eval().to(device)
        cfg = timm.data.resolve_model_data_config(model.backbone)
        preprocess = timm.data.create_transform(**cfg, is_training=False)
        name = args.checkpoint.stem

        def model_fn(x):
            return torch.sigmoid(model(x)["logit"]).float().cpu().numpy()

    if args.manifest:
        mf, pre = pd.read_csv(args.manifest), True
    else:
        mf, pre = data.build_manifest(root), False
    val = mf[mf.split == data.SPLIT_VAL].head(args.n).reset_index(drop=True)

    # Only images the detector already gets right can be "attacked".
    p0, _ = scores_at(model_fn, val, preprocess, [], root, pre, device,
                      args.batch_size, args.workers)
    correct = (p0 > 0.5) == val.y.to_numpy()
    logger.info("%s: %d/%d correct on clean -- attacking those",
                name, correct.sum(), len(val))
    rows = val[correct].reset_index(drop=True)

    cost = search(model_fn, rows, preprocess, root, pre, device,
                  args.batch_size, args.workers)

    flipped = ~np.isnan(cost)
    summary = {
        "model": name,
        "n_attacked": len(rows),
        "flip_rate": float(flipped.mean()),
        "median_attack_cost": float(np.nanmedian(cost)) if flipped.any() else 1.0,
        "p10_attack_cost": float(np.nanpercentile(cost, 10)) if flipped.any() else 1.0,
        "never_flipped": int((~flipped).sum()),
        # Laundering is a one-way attack: it drags fakes toward "real" and
        # leaves reals alone. Splitting the flip rate shows that directly.
        "flip_rate_fake": float(flipped[rows.y == 1].mean()),
        "flip_rate_real": float(flipped[rows.y == 0].mean()),
    }
    for k, v in summary.items():
        logger.info("  %-22s %s", k, f"{v:.4f}" if isinstance(v, float) else v)

    out = Path("results/tables/attack_cost.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([summary])
    if out.exists():
        df = pd.concat([pd.read_csv(out).query("model != @name"), df], ignore_index=True)
    df.to_csv(out, index=False)
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
