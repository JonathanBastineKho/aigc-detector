"""Extract TikTok's held-out benchmark and verify it is uncontaminated.

The archives contain far more than the benchmark: coco.zip holds 163,846 images
of which only the 4,998 in val2017 count, and DALLE.zip holds 64,481 of which
only the 8,843 under Advanced/ count. Taking the wrong subset would silently
measure the headline number on the wrong data, so the counts are asserted.

Also perceptual-hashes the benchmark against the training pool. TikTok's rule is
"do not use the following data during training", and COCO is a common real-image
source -- the same photo can appear in training at a different JPEG quality,
where MD5 would not collide but phash will.

    python scripts/build_heldout.py
"""

import argparse
import logging
import sys
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.utils import dataset as data      # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IMG = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# (archive, path filter, label, expected count) -- counts from TikTok's §5.4 table.
SUBSETS = [
    ("Images/Real/coco.zip", "coco2017/val2017", data.SID_REAL, 4998, "COCO_val2017"),
    ("Images/Diffusion_based/DALLE.zip", "DALLE/Advanced", data.SID_SYNTHETIC, 8843,
     "DALLE_Advanced"),
]


def extract(root: Path, out_dir: Path) -> pd.DataFrame:
    rows = []
    for rel, needle, label, expected, name in SUBSETS:
        archive = root / "heldout" / "_archives" / rel
        if not archive.exists():
            raise SystemExit(f"missing {archive}. Run: ./scripts/download_data.sh stage1")

        with zipfile.ZipFile(archive) as zf:
            members = [n for n in zf.namelist()
                       if needle in n and Path(n).suffix.lower() in IMG]
            if len(members) != expected:
                raise SystemExit(
                    f"{name}: found {len(members)} images, expected {expected}. "
                    "The path filter is wrong -- do not proceed, the benchmark "
                    "number would be measured on the wrong data."
                )
            logger.info("%s: %d images (matches spec)", name, len(members))

            dest = out_dir / name
            dest.mkdir(parents=True, exist_ok=True)
            for m in tqdm(members, desc=name, unit="img"):
                # Flatten the full member path into the filename. DALL-E's six
                # subfolders reuse basenames -- writing by Path(m).name silently
                # overwrote 5,124 of 8,843 images and the manifest then pointed
                # multiple rows at the same file.
                flat = Path(m).relative_to(Path(m).parts[0]).as_posix().replace("/", "__")
                target = dest / flat
                if not target.exists():
                    with zf.open(m) as src, open(target, "wb") as dst:
                        dst.write(src.read())
                rows.append({
                    "image_id": f"ho_{name}_{Path(flat).stem}",
                    "path": str(target.relative_to(root)),
                    "label": label,
                    "y": int(label != data.SID_REAL),
                    "source": "WildFake",
                    "generator": name,
                    "split": data.SPLIT_HELDOUT,
                })
    df = pd.DataFrame(rows)
    for _, _, _, expected, name in SUBSETS:
        sub = df[df.generator == name]
        n_files = sub.path.nunique()
        if n_files != expected:
            raise SystemExit(
                f"{name}: {len(sub)} rows but only {n_files} distinct files on "
                f"disk (expected {expected}). Filenames are colliding."
            )
        logger.info("%s: %d rows, %d distinct files -- no collisions", name,
                    len(sub), n_files)
    return df


def leakage_check(df: pd.DataFrame, root: Path, train_manifest: Path | None):
    """phash the benchmark against the training pool. MD5 misses re-encodes."""
    if train_manifest is None or not train_manifest.exists():
        logger.warning("no training manifest given -- SKIPPING the leakage check. "
                       "Run with --train-manifest before reporting any number.")
        return
    import imagehash

    def hashes(paths):
        # load_image, not Image.open: training rows may address images inside
        # parquet shards or zips, which PIL cannot open directly.
        out = {}
        for p in tqdm(paths, desc="phash", unit="img"):
            try:
                out.setdefault(str(imagehash.phash(data.load_image(p, root))), []).append(p)
            except Exception:
                continue
        return out

    train = pd.read_csv(train_manifest)
    # Exclude held-out rows: the manifest may already index the benchmark, and
    # comparing it against itself would report every image as a collision.
    train = train[train.split != data.SPLIT_HELDOUT]
    train_h = hashes(train[train.y == 0].path.tolist())   # reals only: COCO is real
    logger.info("checking %d held-out reals against %d training reals",
                int((df.y == 0).sum()), len(train_h))
    held_h = hashes(df[df.y == 0].path.tolist())

    collisions = set(train_h) & set(held_h)
    if collisions:
        logger.error("LEAKAGE: %d perceptual-hash collisions between the held-out "
                     "benchmark and the training pool", len(collisions))
        for h in list(collisions)[:5]:
            logger.error("  %s  <->  %s", held_h[h][0], train_h[h][0])
        raise SystemExit("Remove the colliding training images and re-extract.")
    logger.info("leakage check clean: 0 collisions across %d held-out reals",
                len(held_h))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-manifest", type=Path,
                    default=Path("data/extracted/manifest.csv"))
    ap.add_argument("--out", default="heldout")
    args = ap.parse_args()

    root = data.data_root()
    df = extract(root, root / args.out)
    leakage_check(df, root, args.train_manifest)

    out = root / args.out / "manifest.csv"
    df.to_csv(out, index=False)
    logger.info("wrote %s (%d rows: %d real, %d AI)",
                out, len(df), int((df.y == 0).sum()), int((df.y == 1).sum()))


if __name__ == "__main__":
    main()
