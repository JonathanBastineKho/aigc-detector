#!/usr/bin/env python
"""Does the probe transfer to a generator it has never seen?

The probe scores 0.998 on SID_Set, but SID_Set is one generator family and an
old one. That number could mean "detects AI images" or it could mean "detects
old Stable Diffusion". Those are very different projects.

This swaps ONLY the fake side -- SID_Set reals against WildFake fakes -- so the
result isolates generator transfer rather than confounding it with a change of
real-image source.

    python scripts/eval_unseen_generator.py --zip Images/Diffusion_based/DDIM.zip --generator DDIM
"""
import argparse
import sys
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from aigcd import data, features as F  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="path under data/wildfake, e.g. Images/Diffusion_based/DDIM.zip")
    ap.add_argument("--generator", required=True)
    ap.add_argument("--n", type=int, default=1000, help="fake images to sample")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    root = data.data_root()
    zip_rel = f"wildfake/{args.zip}"
    if not (root / zip_rel).exists():
        raise SystemExit(f"not found: {root / zip_rel}")

    with zipfile.ZipFile(root / zip_rel) as zf:
        names = [n for n in zf.namelist() if Path(n).suffix.lower() in data.IMAGE_SUFFIXES]
    print(f"{args.zip}: {len(names)} images")
    print("  sample members:")
    for n in names[:4]:
        print(f"    {n}")

    fakes = data.scan_wildfake_zip(zip_rel, args.generator, data.SID_SYNTHETIC,
                                   root, limit=args.n)
    mf = data.build_manifest(root)
    reals = mf[(mf.split == "val") & (mf.y == 0)].head(args.n).copy()
    reals["generator"] = "SID_real"
    combined = pd.concat([reals, fakes], ignore_index=True)
    print(f"\nscoring {len(reals)} SID_Set reals vs {len(fakes)} {args.generator} fakes")

    bundle = joblib.load(root / "cache" / "probe.joblib")
    device = F.pick_device()
    model, prep = F.load_backbone(bundle["backbone"], device)
    ids, feats = F.extract(combined, model, prep, "clean", args.batch_size, device, root)

    y = combined.set_index("image_id").y.reindex(ids).to_numpy()
    p = bundle["clf"].predict_proba(feats)[:, 1]

    auc = roc_auc_score(y, p)
    print(f"\n{'':<22}{'AUC':>8}{'acc@0.5':>10}")
    print(f"{args.generator + ' (unseen)':<22}{auc:8.4f}{((p > .5) == y).mean():10.4f}")
    print(f"\n  mean p(AI):  real={p[y == 0].mean():.3f}   fake={p[y == 1].mean():.3f}")
    print(f"  fakes called real (miss rate): {(p[y == 1] <= .5).mean():.1%}")

    out = Path("results/tables/unseen_generator.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "generator": args.generator, "n_real": int((y == 0).sum()),
        "n_fake": int((y == 1).sum()), "auc": auc,
        "acc": float(((p > .5) == y).mean()),
        "miss_rate": float((p[y == 1] <= .5).mean()),
    }]).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
