#!/usr/bin/env python
"""Cross-generator evaluation on WildFake.

Answers two questions the SID_Set numbers cannot:

  1. TRANSFER  -- how far does a SID_Set-trained probe fall on each unseen
                  generator?
  2. LOGO      -- if we train on several generators and hold one out, does the
                  gap close? That is the generalisation axis, and it is the
                  question the PEFT ablation will later compete on.

Real and fake both come from WildFake so the comparison is not confounded by a
change of real-image source -- the mistake that would let a model separate
classes on photo provenance instead of generation.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from aigcd import data, features as F                      # noqa: E402
from aigcd.probe import fit                                # noqa: E402

GENERATORS = {
    "DDIM": "wildfake/Images/Diffusion_based/DDIM.zip",
    "DDPM": "wildfake/Images/Diffusion_based/DDPM.zip",
    "ADM":  "wildfake/Images/Diffusion_based/ADM.zip",
}
REAL_ZIP = "wildfake/Images/Real/imagenet.zip"


def build(root: Path, n: int) -> pd.DataFrame:
    frames = [data.scan_wildfake_zip(REAL_ZIP, "imagenet", data.SID_REAL, root,
                                     limit=n * len(GENERATORS))]
    for gen, rel in GENERATORS.items():
        frames.append(data.scan_wildfake_zip(rel, gen, data.SID_SYNTHETIC, root, limit=n))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800, help="images per generator")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    root = data.data_root()
    df = build(root, args.n)
    print(df.groupby("generator").size().to_string(), "\n")

    cache = root / "cache" / "wildfake_feats.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        ids, feats = d["image_ids"], d["features"]
        print(f"loaded cached features {feats.shape}")
    else:
        model, prep = F.load_backbone()
        ids, feats = F.extract(df, model, prep, "clean", args.batch_size, root=root)
        np.savez_compressed(cache, image_ids=ids, features=feats)
        print(f"cached {feats.shape} -> {cache.relative_to(root)}")

    meta = df.set_index("image_id").reindex(ids)
    y, gen = meta.y.to_numpy(), meta.generator.to_numpy()
    real = gen == "imagenet"

    rows = []

    # --- 1. transfer: the existing SID_Set-trained probe --------------------
    import joblib
    sid = joblib.load(root / "cache" / "probe.joblib")["clf"]
    for g in GENERATORS:
        m = real | (gen == g)
        p = sid.predict_proba(feats[m])[:, 1]
        rows.append({"setting": "SID-trained (transfer)", "test_generator": g,
                     "auc": roc_auc_score(y[m], p),
                     "acc": float(((p > .5) == y[m]).mean()),
                     "miss_rate": float((p[y[m] == 1] <= .5).mean())})

    # --- 2. LOGO: train on the other generators, test on the held-out one ---
    for g in GENERATORS:
        tr = (real | np.isin(gen, [x for x in GENERATORS if x != g]))
        te = real | (gen == g)
        # Disjoint real images between fit and test, else the real side leaks.
        idx = np.where(real)[0]
        half = len(idx) // 2
        tr_mask, te_mask = tr.copy(), te.copy()
        tr_mask[idx[half:]] = False
        te_mask[idx[:half]] = False

        clf = fit(feats[tr_mask], y[tr_mask])
        p = clf.predict_proba(feats[te_mask])[:, 1]
        rows.append({"setting": "LOGO (2 gens -> 1 unseen)", "test_generator": g,
                     "auc": roc_auc_score(y[te_mask], p),
                     "acc": float(((p > .5) == y[te_mask]).mean()),
                     "miss_rate": float((p[y[te_mask] == 1] <= .5).mean())})

    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    p = Path("results/tables/cross_generator.csv")
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
