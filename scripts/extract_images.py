#!/usr/bin/env python
"""Pre-extract training images to individual files.

Reading images out of parquet per-sample costs a whole ~500 MB column decode to
fetch one image, leaving the GPU idle behind the data loader. num_workers
parallelises that waste rather than removing it. One pass here converts the
corpus to plain files; every epoch afterwards is a normal file read.

Bias control is applied here, so what lands on disk is exactly what the model
trains on. Extracting at the final size -- rather than larger with a random crop
at train time -- keeps training and evaluation on identical preprocessing; the
augmentation diversity comes from the laundering chains, which already include
crop_0.8.

    python scripts/extract_images.py --splits train val
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.utils import dataset as data                       # noqa: E402
from src.utils.features import align_bias    # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--jpeg-q", type=int, default=96)
    ap.add_argument("--mode", default="crop", choices=["crop", "resize"],
                    help="crop a native-resolution window, or resize the whole image")
    ap.add_argument("--out-dir", default="extracted")
    ap.add_argument("--limit", type=int, help="rows per split, for smoke tests")
    ap.add_argument("--wildfake", action="store_true",
                    help="include the WildFake generator pool (DDIM/DDPM/ADM + reals)")
    ap.add_argument("--generators", nargs="+",
                    help="which WildFake generators (default: all available)")
    ap.add_argument("--per-generator", type=int, default=4000)
    ap.add_argument("--rebuild", action="store_true",
                    help="discard the existing manifest instead of merging")
    args = ap.parse_args()

    root = data.data_root()
    out_root = root / args.out_dir
    existing_path = out_root / "manifest.csv"

    # Merge into an existing manifest rather than rebuilding it. Rebuilding
    # requires re-scanning the SID_Set parquets, which means keeping 16.8 GB on
    # disk purely to enumerate images that are already extracted. Adding a
    # generator should not cost that.
    existing = pd.read_csv(existing_path) if existing_path.exists() else None

    if existing is None:
        manifest = data.build_manifest(root, args.wildfake, args.generators,
                                       args.per_generator)
    else:
        logger.info("merging into existing manifest (%d rows)", len(existing))
        # Existing rows are KEPT unconditionally -- their images are already
        # extracted, so a source archive being deleted for disk space must not
        # silently drop them from training. Only genuinely new rows are scanned.
        frames = [existing]
        if (root / "sid_set" / "data").exists():
            frames.append(data._scan_sid_set(root))
        if args.wildfake:
            frames.append(data.scan_wildfake_pool(root, args.generators,
                                                  args.per_generator))
        manifest = pd.concat([f for f in frames if len(f)], ignore_index=True)
        manifest = manifest.drop_duplicates(subset="image_id", keep="first")
        logger.info("after merge: %d rows (+%d new)", len(manifest),
                    len(manifest) - len(existing))
    print(manifest.groupby(["source", "generator"], dropna=False).size().to_string(), "\n")

    rows = []
    for split in args.splits:
        df = manifest[manifest.split == split]
        if args.limit:
            df = df.head(args.limit)
        if split != data.SPLIT_HELDOUT:
            data.assert_no_heldout(df)

        split_dir = out_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        meta = df.set_index("image_id")

        written = skipped = 0
        already = set()
        if existing is not None:
            already = set(existing[existing.split == split].image_id)

        todo = df[~df.image_id.isin(already)]
        if len(todo) < len(df):
            logger.info("  %s: %d already indexed, %d new", split,
                        len(df) - len(todo), len(todo))
            rows.extend(existing[existing.split == split].to_dict("records"))
        df = todo

        for image_id, img in tqdm(data.iter_images(df, root), total=len(df), desc=split):
            dest = split_dir / f"{image_id}.jpg"
            if dest.exists():
                skipped += 1
            else:
                align_bias(img, args.jpeg_q, args.size, args.mode).save(
                    dest, "JPEG", quality=args.jpeg_q)
                written += 1

            row = meta.loc[image_id]
            rows.append({
                "image_id": image_id,
                # Relative to the data root, so the manifest survives being moved
                # between the Mac and the A100 box.
                "path": str(dest.relative_to(root)),
                "label": int(row.label), "y": int(row.y),
                "source": row.source, "generator": row.generator, "split": split,
            })
        print(f"  {split}: {written} written, {skipped} already present")

    out = out_root / "manifest.csv"
    df = pd.DataFrame(rows)

    # Merge with whatever is already indexed. Sources can then be deleted after
    # extraction -- the manifest is the durable artifact, not the archives.
    if out.exists() and not args.rebuild:
        prev = pd.read_csv(out)
        keep = prev[~prev.image_id.isin(df.image_id)] if len(df) else prev
        df = pd.concat([keep, df], ignore_index=True)
        print(f"  merged with existing manifest: {len(prev)} + {len(rows)} new")

    df = df.drop_duplicates("image_id", keep="last")
    # Drop rows whose extracted file has since been deleted.
    exists = df.path.map(lambda p: (root / p).exists())
    if (~exists).any():
        print(f"  dropping {int((~exists).sum())} rows with missing files")
        df = df[exists]
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(df)} rows)")
    print(df.groupby(["source", "generator"], dropna=False).size().to_string())
    print("Images are already bias-aligned -- training must NOT align them again.")


if __name__ == "__main__":
    main()
