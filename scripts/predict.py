#!/usr/bin/env python
"""Required deliverable: image directory -> JSON with image_path and pred.

    python scripts/predict.py --input-dir path/to/images --output predictions.json

`pred` is p(AI-generated) in [0, 1].

Preprocessing matches training exactly -- a 224 crop at native resolution,
recompressed to JPEG quality 96. Scoring raw images would shift the input
distribution and degrade predictions for no visible reason.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.components.peft import apply_peft                      # noqa: E402
from src.models.detector import Detector                        # noqa: E402
from src.utils.features import align_bias, pick_device          # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_CKPT = Path("checkpoints/lora_r32_film_resize_slim.pt")


def find_images(input_dir: Path) -> list[Path]:
    """Every image under input_dir, recursively, in a stable order."""
    return sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def load_model(checkpoint: Path, device: str):
    """Slim checkpoints carry only adapters and heads; the frozen DINOv3 base is
    rebuilt from timm, which is exact because apply_peft is deterministic."""
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    targs = ck["args"]
    bb = timm.create_model(targs["backbone"], pretrained=True, num_classes=0)
    bb = apply_peft(bb, mode=targs["arm"], r=targs["rank"])
    model = Detector(bb, dim=bb.num_features, conditioner=targs["conditioner"],
                     bounded_severity=not targs.get("unbounded_severity", False))
    model.load_state_dict(ck["state_dict"], strict=not ck.get("slim"))
    cfg = timm.data.resolve_model_data_config(bb)
    align = "resize" if "resize" in str(targs.get("manifest", "")) else "crop"
    return (model.eval().to(device),
            timm.data.create_transform(**cfg, is_training=False), align)


@torch.no_grad()
def predict(paths, model, preprocess, align, device, batch_size=16):
    """Unreadable files score 0.5 rather than killing the run -- a submission
    script that dies on one corrupt file is worse than one that flags it."""
    preds: list[float] = []
    batch: list[torch.Tensor] = []

    def flush():
        if batch:
            logits = model(torch.stack(batch).to(device))["logit"]
            preds.extend(torch.sigmoid(logits).float().cpu().numpy().tolist())
            batch.clear()

    for p in tqdm(paths, desc="scoring", unit="img"):
        try:
            img = align_bias(Image.open(p).convert("RGB"), mode=align)
        except Exception as exc:  # noqa: BLE001
            print(f"  unreadable, scoring 0.5: {p} ({exc})", file=sys.stderr)
            flush()
            preds.append(0.5)
            continue
        batch.append(preprocess(img))
        if len(batch) == batch_size:
            flush()
    flush()
    return preds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--output", default="predictions.json", type=Path)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--relative", action="store_true",
                    help="write paths relative to --input-dir instead of absolute")
    args = ap.parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"not a directory: {args.input_dir}")
    if not args.checkpoint.exists():
        raise SystemExit(f"no checkpoint at {args.checkpoint}")

    paths = find_images(args.input_dir)
    if not paths:
        raise SystemExit(f"no images found under {args.input_dir}")

    device = args.device or pick_device()
    model, preprocess, align = load_model(args.checkpoint, device)
    print(f"{len(paths)} images  |  {args.checkpoint.name}  |  {device}  |  align={align}")

    preds = predict(paths, model, preprocess, align, device, args.batch_size)

    records = [
        {"image_path": str(p.relative_to(args.input_dir) if args.relative
                           else p.resolve()),
         "pred": round(float(s), 6)}
        for p, s in zip(paths, preds)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2))

    arr = np.array(preds)
    print(f"wrote {args.output}  ({len(records)} predictions)")
    print(f"  mean pred={arr.mean():.3f}   flagged AI (>0.5): {(arr > 0.5).sum()}/{len(arr)}")


if __name__ == "__main__":
    main()
