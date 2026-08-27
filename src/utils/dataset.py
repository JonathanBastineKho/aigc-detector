"""The manifest: one index over every image in the project.

Sources are stored three incompatible ways (parquet blobs, zips, folders), so
everything downstream reads one table instead:

    image_id | path | label | y | source | generator | split

Buys two things: the leakage rule becomes one assertion rather than a convention
spread across a dozen files, and splits become a column (LOGO is a filter, not a
filesystem reshuffle).
"""

from __future__ import annotations

import os
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import transforms

# SID_Set ships three classes (SIDA paper): authentic, fully synthetic, and
# tampered -- a real photograph with an AI-edited region.
SID_REAL, SID_SYNTHETIC, SID_TAMPERED = 0, 1, 2

# Tampered is eval-only: the provided benchmark is whole-image real vs fake with
# no local edits, so training on them optimises for a distribution we are never
# scored on. As a slice it costs nothing and yields a free generalisation result.
SPLIT_TRAIN, SPLIT_VAL, SPLIT_TAMPERED, SPLIT_HELDOUT = (
    "train", "val", "eval_tampered", "heldout",
)


def data_root() -> Path:
    """Datasets never live in the repo. AIGCD_DATA_ROOT points at real storage."""
    return Path(os.environ.get("AIGCD_DATA_ROOT", Path(__file__).parents[2] / "data"))


# --------------------------------------------------------------------------
# Building the manifest
# --------------------------------------------------------------------------

def _scan_sid_set(root: Path, val_frac: float = 0.15, seed: int = 0) -> pd.DataFrame:
    """Index SID_Set's shards without decoding an image.

    Reading only (img_id, label) keeps this at seconds rather than minutes over
    16 GB of embedded bytes. Rows address as "<parquet>#<row>", decoded lazily.
    """
    shards = sorted((root / "sid_set" / "data").glob("validation-*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"No SID_Set parquets under {root / 'sid_set' / 'data'}. "
            "Run: ./scripts/download_data.sh stage2"
        )

    frames = []
    for shard in shards:
        table = pq.read_table(shard, columns=["img_id", "label"])
        rel = shard.relative_to(root)
        frames.append(pd.DataFrame({
            "image_id": [f"sid_{i}" for i in table.column("img_id").to_pylist()],
            "path": [f"{rel}#{i}" for i in range(table.num_rows)],
            "label": table.column("label").to_pylist(),
            "source": "SID_Set",
            "generator": pd.NA,   # SID_Set does not disclose per-image generators
        }))

    df = pd.concat(frames, ignore_index=True)

    # Tampered images are evaluation-only (see module docstring).
    df["split"] = SPLIT_TRAIN
    df.loc[df.label == SID_TAMPERED, "split"] = SPLIT_TAMPERED

    # LIMITATION: SID_Set exposes no generator labels, so this holdout is random
    # and therefore leaky -- same-generator images share fingerprints, so it
    # partly measures memorisation. Sanity check, not a generalisation number;
    # that comes from WildFake via split_by_generator().
    trainable = df.index[df.split == SPLIT_TRAIN]
    val_idx = (
        df.loc[trainable]
        .sample(frac=val_frac, random_state=seed)
        .index
    )
    df.loc[val_idx, "split"] = SPLIT_VAL
    return df


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# WildFake generator pool available for TRAINING.
WILDFAKE_GENERATORS = {
    "DDIM": "wildfake/Images/Diffusion_based/DDIM.zip",
    "DDPM": "wildfake/Images/Diffusion_based/DDPM.zip",
    "ADM":  "wildfake/Images/Diffusion_based/ADM.zip",
}
WILDFAKE_REAL = {
    "imagenet": "wildfake/Images/Real/imagenet.zip",
    "ffhq":     "wildfake/Images/Real/ffhq.zip",
}

# TikTok forbids training on COCO val2017 and DALL-E Advanced. Both live inside
# these archives alongside permitted images, so the whole archive is refused as
# a training source rather than trusting a filter to catch every path.
FORBIDDEN_ARCHIVES = ("coco.zip", "DALLE.zip")


def scan_wildfake_zip(
    zip_rel: str,
    generator: str,
    label: int,
    root: Path | None = None,
    limit: int | None = None,
    split: str = "eval_unseen",
) -> pd.DataFrame:
    """Index a WildFake archive without extracting it.

    Zips are 6-47 GB; lazy member reads avoid doubling disk. Paths use
    "<zip>::<member>", '::' so parquet row refs stay distinguishable.
    """
    root = root or data_root()
    with zipfile.ZipFile(root / zip_rel) as zf:
        members = [
            n for n in zf.namelist()
            if Path(n).suffix.lower() in IMAGE_SUFFIXES and not n.startswith("__MACOSX")
        ]
    members.sort()
    if limit:
        # Stride rather than truncate: zips are usually ordered by subfolder, so
        # taking the first N would sample one subdirectory instead of the archive.
        step = max(1, len(members) // limit)
        members = members[::step][:limit]

    return pd.DataFrame({
        "image_id": [f"wf_{generator}_{Path(m).stem}" for m in members],
        "path": [f"{zip_rel}::{m}" for m in members],
        "label": label,
        # Binary target, matching build_manifest. Set here rather than left to
        # the caller: a missing y silently becomes NaN on concat and only
        # surfaces later as an unrelated-looking metric error.
        "y": int(label != SID_REAL),
        "source": "WildFake",
        "generator": generator,
        "split": split,
    })


def scan_wildfake_pool(
    root: Path,
    generators: list[str] | None = None,
    per_generator: int = 4000,
    val_frac: float = 0.15,
    seed: int = 0,
) -> pd.DataFrame:
    """Build TRAIN rows from WildFake generators, balanced with WildFake reals.

    Reals come from WildFake too. Pairing WildFake fakes against SID_Set reals
    would let the model separate on photo provenance instead of generation --
    the same class of shortcut the bias control exists to remove.

    Missing archives are skipped, so a partial transfer still yields a usable
    pool rather than an error.
    """
    generators = generators or list(WILDFAKE_GENERATORS)
    frames, n_fake = [], 0

    for gen in generators:
        rel = WILDFAKE_GENERATORS.get(gen)
        if rel is None:
            raise ValueError(f"unknown generator {gen!r}; known: {list(WILDFAKE_GENERATORS)}")
        if any(bad in rel for bad in FORBIDDEN_ARCHIVES):
            raise ValueError(f"{rel} contains held-out data and may not be used for training")
        if not (root / rel).exists():
            print(f"  skipping {gen}: {rel} not found")
            continue
        df = scan_wildfake_zip(rel, gen, SID_SYNTHETIC, root, per_generator, SPLIT_TRAIN)
        frames.append(df)
        n_fake += len(df)

    if not frames:
        return pd.DataFrame()

    # Match the real count to the fake count so the pool stays balanced.
    for name, rel in WILDFAKE_REAL.items():
        if not (root / rel).exists():
            continue
        frames.append(scan_wildfake_zip(rel, name, SID_REAL, root, n_fake, SPLIT_TRAIN))
        break

    pool = pd.concat(frames, ignore_index=True)
    rng = np.random.default_rng(seed)
    val_idx = rng.choice(pool.index, size=int(len(pool) * val_frac), replace=False)
    pool.loc[val_idx, "split"] = SPLIT_VAL
    return pool


def build_manifest(
    root: Path | None = None,
    wildfake: bool = False,
    generators: list[str] | None = None,
    per_generator: int = 4000,
) -> pd.DataFrame:
    """Scan every available dataset and emit the project-wide index."""
    root = root or data_root()
    frames = [_scan_sid_set(root)]
    if wildfake:
        pool = scan_wildfake_pool(root, generators, per_generator)
        if len(pool):
            frames.append(pool)

    # WildFake and the held-out benchmark join here once stages 1 and 3 land.
    # Kept as an explicit gap rather than a silent one.
    heldout_dir = root / "heldout"
    if heldout_dir.exists() and any(heldout_dir.rglob("*.jpg")):
        frames.append(_scan_heldout(heldout_dir, root))

    df = pd.concat(frames, ignore_index=True)

    # Binary target: 1 = AI-generated, 0 = authentic. Tampered rows get y = 1 so
    # the eval slice is scoreable, but their split keeps them out of training.
    df["y"] = (df.label != SID_REAL).astype(int)

    return df[["image_id", "path", "label", "y", "source", "generator", "split"]]


def _scan_heldout(heldout_dir: Path, root: Path) -> pd.DataFrame:
    """Index the TikTok-specified benchmark. Never trainable, by construction."""
    rows = []
    for p in sorted(heldout_dir.rglob("*")):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        is_fake = "dalle" in str(p).lower() or "advanced" in str(p).lower()
        rows.append({
            "image_id": f"ho_{p.stem}",
            "path": str(p.relative_to(root)),
            "label": SID_SYNTHETIC if is_fake else SID_REAL,
            "source": "WildFake",
            "generator": "DALLE_Advanced" if is_fake else pd.NA,
            "split": SPLIT_HELDOUT,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Reading images
# --------------------------------------------------------------------------

def load_image(path: str, root: Path | None = None) -> Image.Image:
    """Single-image read. Fine for spot checks; use iter_images() for bulk."""
    root = root or data_root()
    if "::" in path:
        rel, member = path.split("::", 1)
        with zipfile.ZipFile(root / rel) as zf:
            return Image.open(BytesIO(zf.read(member))).convert("RGB")
    if "#" in path:
        rel, row = path.rsplit("#", 1)
        table = pq.read_table(root / rel, columns=["image"])
        blob = table.column("image")[int(row)].as_py()["bytes"]
        return Image.open(BytesIO(blob)).convert("RGB")
    return Image.open(root / path).convert("RGB")


def iter_images(df: pd.DataFrame, root: Path | None = None):
    """Yield (image_id, PIL.Image), grouped by shard/archive.

    load_image() per row decodes a ~500 MB column to fetch one image -- fine
    once, ruinous across 20k rows. Here each shard is read exactly once.
    """
    root = root or data_root()
    zip_rows = df[df.path.str.contains("::", na=False)]
    rest = df[~df.path.str.contains("::", na=False)]
    parquet_rows = rest[rest.path.str.contains("#", na=False)]
    plain_rows = rest[~rest.path.str.contains("#", na=False)]

    if len(zip_rows):
        archives = zip_rows.path.str.split("::", n=1).str[0]
        for archive, group in zip_rows.groupby(archives, sort=False):
            with zipfile.ZipFile(root / archive) as zf:
                for image_id, path in zip(group.image_id, group.path):
                    blob = zf.read(path.split("::", 1)[1])
                    yield image_id, Image.open(BytesIO(blob)).convert("RGB")

    if len(parquet_rows):
        shards = parquet_rows.path.str.rsplit("#", n=1).str[0]
        for shard, group in parquet_rows.groupby(shards, sort=False):
            col = pq.read_table(root / shard, columns=["image"]).column("image")
            for image_id, path in zip(group.image_id, group.path):
                blob = col[int(path.rsplit("#", 1)[1])].as_py()["bytes"]
                yield image_id, Image.open(BytesIO(blob)).convert("RGB")

    for image_id, path in zip(plain_rows.image_id, plain_rows.path):
        yield image_id, Image.open(root / path).convert("RGB")


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------

def split_by_generator(df: pd.DataFrame, holdout: str | list[str]) -> tuple:
    """Leave-one-generator-out: train on every generator except `holdout`.

    A random split measures memorisation -- same-generator images share
    fingerprints. Holding out a whole generator asks the question that matters.
    """
    holdout = [holdout] if isinstance(holdout, str) else holdout
    trainable = df[df.split.isin({SPLIT_TRAIN, SPLIT_VAL})]
    return (
        trainable[~trainable.generator.isin(holdout)],
        trainable[trainable.generator.isin(holdout)],
    )


def assert_no_heldout(df: pd.DataFrame) -> None:
    """Guard for training entry points. Cheap to call, expensive to omit."""
    leaked = df[df.split == SPLIT_HELDOUT]
    if len(leaked):
        raise AssertionError(
            f"{len(leaked)} held-out rows reached a training path. "
            "TikTok's brief forbids training on this data."
        )


class LaunderedPairs(Dataset):
    """Yields (clean, laundered, y, severity). The chain is resampled every
    epoch, so laundered views are deliberately not cacheable."""

    def __init__(self, df, preprocess, epoch: int = 0, max_epochs: int = 10,
                 root: Path | None = None, jpeg_q: int = 96, crop: int = 224,
                 pre_extracted: bool = False):
        self.rows = df.reset_index(drop=True)
        self.preprocess, self.epoch, self.max_epochs = preprocess, epoch, max_epochs
        self.root, self.jpeg_q, self.crop = root or data.data_root(), jpeg_q, crop
        self.pre_extracted = pre_extracted

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows.iloc[i]
        img = load_image(row.path, self.root)
        # Pre-extracted files are already cropped and recompressed. Aligning
        # again would add a second JPEG pass to every training sample -- an
        # artifact present in training but not at inference, and painful to find.
        if not self.pre_extracted:
            from .features import align_bias  # local: avoids a cycle
            img = align_bias(img, self.jpeg_q, self.crop)

        rng = np.random.default_rng((self.epoch << 32) ^ i)
        chain = transforms.sample_chain(self.epoch, rng, self.max_epochs)
        laundered, log = transforms.apply_chain(img, chain)

        return (
            self.preprocess(img),
            self.preprocess(laundered),
            torch.tensor(float(row.y)),
            torch.from_numpy(transforms.severity_vector(log)),
        )


class CellDataset(Dataset):
    """One battery cell applied deterministically -- for evaluation, no curriculum."""

    def __init__(self, df, preprocess, cell, root=None, pre_extracted=False):
        self.rows = df.reset_index(drop=True)
        self.preprocess = preprocess
        self.root = root or data_root()
        self.pre_extracted = pre_extracted
        op, kwargs = transforms.BATTERY[cell]
        self.chain = [] if op == "none" else [(op, kwargs)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows.iloc[i]
        img = load_image(row.path, self.root)
        if not self.pre_extracted:
            from .features import align_bias      # local: avoids a cycle
            img = align_bias(img)
        if self.chain:
            img, _ = transforms.apply_chain(img, self.chain)
        return self.preprocess(img), torch.tensor(float(row.y))
