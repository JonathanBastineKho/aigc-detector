#!/usr/bin/env python
"""Deliverable: image directory -> JSON with image_path and pred per image.

    python predict.py --input-dir path/to/images --output preds.json

`pred` is p(AI-generated) in [0, 1].

Preprocessing here must match training exactly. The probe learned on
bias-aligned inputs -- native-resolution 224 crop, recompressed to JPEG Q=96 --
so scoring raw images through it would silently shift the input distribution and
degrade predictions for no visible reason.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "src"))
from aigcd import data, features as F  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def find_images(input_dir: Path) -> list[Path]:
    """Every image under input_dir, recursively, in a stable order."""
    return sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


@torch.no_grad()
def predict(
    paths: list[Path],
    clf,
    model,
    preprocess,
    device: str,
    batch_size: int = 16,
    jpeg_q: int = 96,
    crop: int = 224,
) -> list[float]:
    """Score every path. Unreadable files get 0.5 rather than crashing the run --
    a submission script that dies on one corrupt file is worse than one that
    flags it as undecided."""
    preds: list[float] = []
    batch: list[torch.Tensor] = []

    def flush():
        if not batch:
            return
        feats = model(torch.stack(batch).to(device)).float().cpu().numpy()
        preds.extend(clf.predict_proba(feats)[:, 1].tolist())
        batch.clear()

    for p in tqdm(paths, desc="scoring", unit="img"):
        try:
            img = Image.open(p).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            print(f"  unreadable, scoring 0.5: {p} ({exc})", file=sys.stderr)
            flush()
            preds.append(0.5)
            continue
        batch.append(preprocess(F.align_bias(img, jpeg_q, crop)))
        if len(batch) == batch_size:
            flush()
    flush()
    return preds


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a directory of images for AIGC likelihood")
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--output", default="predictions.json", type=Path)
    ap.add_argument("--model", type=Path, default=None,
                    help="fitted probe (default: $AIGCD_DATA_ROOT/cache/probe.joblib)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--relative", action="store_true",
                    help="write paths relative to --input-dir instead of absolute")
    args = ap.parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"not a directory: {args.input_dir}")

    model_path = args.model or (data.data_root() / "cache" / "probe.joblib")
    if not model_path.exists():
        raise SystemExit(f"no fitted probe at {model_path}. Run: python -m aigcd.probe")

    paths = find_images(args.input_dir)
    if not paths:
        raise SystemExit(f"no images found under {args.input_dir}")

    bundle = joblib.load(model_path)
    device = args.device or F.pick_device()
    backbone, preprocess = F.load_backbone(bundle["backbone"], device)

    print(f"{len(paths)} images  |  device={device}  |  backbone={bundle['backbone']}")
    preds = predict(paths, bundle["clf"], backbone, preprocess, device, args.batch_size)

    records = [
        {
            "image_path": str(p.relative_to(args.input_dir) if args.relative else p.resolve()),
            "pred": round(float(s), 6),
        }
        for p, s in zip(paths, preds)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2))

    arr = np.array(preds)
    print(f"wrote {args.output}  ({len(records)} predictions)")
    print(f"  mean pred={arr.mean():.3f}   flagged AI (>0.5): {(arr > 0.5).sum()}/{len(arr)}")


if __name__ == "__main__":
    main()
