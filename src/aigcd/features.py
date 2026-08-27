"""DINOv3 feature extraction and caching.

Extraction is a one-off cost paid per (backbone, battery cell) pair, so results
go to disk and are never recomputed. Only deterministic inputs are cacheable:
clean images and the fixed battery cells qualify, randomly sampled training
chains do not.

Valid only while the backbone is frozen. Once SVD-PEFT starts updating weights
the cache is stale after the first step, and training reads images live.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import timm
import torch
from tqdm import tqdm

from io import BytesIO

from PIL import Image

from . import data, transforms as T

# timm mirror rather than facebook/* -- same weights, no license gate.
DEFAULT_BACKBONE = "vit_large_patch16_dinov3.lvd1689m"


def align_bias(img, jpeg_q: int = 96, crop: int = 224):
    """Remove dataset shortcuts unrelated to generation.

    SID_Set leaks the label twice: geometry (100% of fakes are 1024x1024, only
    4% of reals are square -- "is it square?" alone scores 0.98 AUC) and format
    (reals JPEG ~2.9 bpp, fakes PNG ~8.5 bpp).

    Follows GenImage's protocol ("Fake or JPEG?", arXiv 2403.17608): fixed crop
    at NATIVE resolution -- never resize, which interpolates away the generator
    fingerprints -- then one JPEG quality for both classes.

    Caveat for the writeup: reals end up double-compressed, fakes singly.
    GenImage accepts this; removing it means regenerating the dataset.
    """
    w, h = img.size
    if min(w, h) < crop:                       # upscale only when unavoidable
        r = crop / min(w, h)
        img = img.resize((int(w * r) + 1, int(h * r) + 1), Image.BICUBIC)
        w, h = img.size
    left, top = (w - crop) // 2, (h - crop) // 2
    img = img.crop((left, top, left + crop, top + crop))

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_backbone(name: str = DEFAULT_BACKBONE, device: str | None = None):
    """Return (model, preprocess). num_classes=0 gives pooled features."""
    device = device or pick_device()
    model = timm.create_model(name, pretrained=True, num_classes=0).eval().to(device)
    cfg = timm.data.resolve_model_data_config(model)
    return model, timm.data.create_transform(**cfg, is_training=False)


def cache_path(root: Path, backbone: str, split: str, cell: str,
               variant: str = "aligned") -> Path:
    """Cache identity is (backbone, variant, split, cell). Aligned and raw
    features are not interchangeable, so the variant is part of the path."""
    return (root / "cache" / "features" / backbone.replace("/", "_")
            / variant / split / f"{cell}.npz")


@torch.no_grad()
def extract(
    df,
    model,
    preprocess,
    cell: str = "clean",
    batch_size: int = 16,
    device: str | None = None,
    root: Path | None = None,
    bias_control: bool = True,
    jpeg_q: int = 96,
    crop: int = 224,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract pooled features for every row, optionally laundered first.

    Returns (image_ids, features). Row order matches the yielded order, not the
    dataframe's -- ids are returned so callers can align rather than assume.
    """
    device = device or pick_device()
    op_name, kwargs = T.BATTERY[cell]
    chain = [] if op_name == "none" else [(op_name, kwargs)]

    ids, feats, batch = [], [], []

    def flush():
        if not batch:
            return
        x = torch.stack(batch).to(device)
        feats.append(model(x).float().cpu().numpy())
        batch.clear()

    for image_id, img in tqdm(
        data.iter_images(df, root), total=len(df), desc=cell, unit="img"
    ):
        # Normalise the source first, then launder. Laundering models what
        # happens to an image in the world; alignment makes the sources
        # comparable before any of that.
        if bias_control:
            img = align_bias(img, jpeg_q, crop)
        if chain:
            img, _ = T.apply_chain(img, chain)
        batch.append(preprocess(img))
        ids.append(image_id)
        if len(batch) == batch_size:
            flush()
    flush()

    return np.array(ids), np.concatenate(feats, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Cache DINOv3 features")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--cells", nargs="+", default=["clean"],
                    help="battery cells, or 'all' for the full grid")
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, help="rows per split, for smoke tests")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-bias-control", action="store_true",
                    help="skip geometry/compression alignment (for the ablation)")
    ap.add_argument("--jpeg-q", type=int, default=96)
    ap.add_argument("--crop", type=int, default=224)
    args = ap.parse_args()
    bias_control = not args.no_bias_control
    variant = "aligned" if bias_control else "raw"

    device = args.device or pick_device()
    root = data.data_root()
    cells = list(T.BATTERY) if args.cells == ["all"] else args.cells

    print(f"device={device}  backbone={args.backbone}  variant={variant}"
          + (f"  jpeg_q={args.jpeg_q} crop={args.crop}" if bias_control else ""))
    manifest = data.build_manifest(root)

    model, preprocess = load_backbone(args.backbone, device)

    for split in args.splits:
        df = manifest[manifest.split == split]
        if args.limit:
            df = df.head(args.limit)
        # Guard every extraction, not just training entry points: a cached
        # held-out feature file is exactly how leakage happens by accident.
        if split != data.SPLIT_HELDOUT:
            data.assert_no_heldout(df)

        for cell in cells:
            out = cache_path(root, args.backbone, split, cell, variant)
            if out.exists():
                print(f"skip {variant}/{split}/{cell} (cached)")
                continue
            ids, feats = extract(df, model, preprocess, cell,
                                 args.batch_size, device, root,
                                 bias_control, args.jpeg_q, args.crop)
            out.parent.mkdir(parents=True, exist_ok=True)
            labels = df.set_index("image_id").y.reindex(ids).to_numpy()
            np.savez_compressed(out, image_ids=ids, features=feats, y=labels)
            print(f"  {split}/{cell}: {feats.shape} -> {out.relative_to(root)}")


if __name__ == "__main__":
    main()
