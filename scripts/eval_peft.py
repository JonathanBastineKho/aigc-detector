#!/usr/bin/env python
"""Evaluate a trained arm; log results onto its own W&B run.

Produces the two headline numbers:
  robustness_gap   clean AUC - mean laundered AUC  (the §6.1 claim)
  unseen_adm_auc   generalisation, on ADM
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import wandb
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.utils import dataset as data, transforms as T                        # noqa: E402
from src.utils.dataset import CellDataset
from src.utils.features import pick_device             # noqa: E402
from src.models.detector import Detector                        # noqa: E402
from src.components.peft import apply_peft, count_params         # noqa: E402

ADM_ZIP = "wildfake/Images/Diffusion_based/ADM.zip"
REAL_ZIP = "wildfake/Images/Real/imagenet.zip"


@torch.no_grad()
def score(model, df, preprocess, cell, root, device, bs, workers, pre=False):
    dl = DataLoader(CellDataset(df, preprocess, cell, root, pre), batch_size=bs,
                    num_workers=workers)
    ps, ys = [], []
    for x, y in tqdm(dl, desc=f"{cell:12s}", unit="batch", leave=False):
        ps.append(torch.sigmoid(model(x.to(device))["logit"]).float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)


def load_model(ckpt, device):
    targs = ckpt["args"]
    bb = timm.create_model(targs["backbone"], pretrained=True, num_classes=0)
    bb = apply_peft(bb, mode=targs["arm"], r=targs["rank"])
    model = Detector(bb, dim=bb.num_features, conditioner=targs["conditioner"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval().to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--manifest", type=Path, help="pre-extracted manifest CSV")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-unseen", type=int, default=800)
    ap.add_argument("--device", default=None)
    ap.add_argument("--split", default="val", choices=["val", "heldout"],
                    help="'heldout' evaluates TikTok's benchmark, never trained on")
    ap.add_argument("--cells", nargs="+",
                    help="battery cells to score (default: all 17)")
    ap.add_argument("--no-chains", action="store_true",
                    help="skip composed chains (repost_x2, screenshot, ...)")
    ap.add_argument("--limit", type=int, help="cap images, for quick passes")
    ap.add_argument("--project", default="aigc-detector")
    args = ap.parse_args()

    device = args.device or pick_device()
    root = data.data_root()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = load_model(ckpt, device)

    cfg = timm.data.resolve_model_data_config(model.backbone)
    preprocess = timm.data.create_transform(**cfg, is_training=False)

    if args.manifest:
        mf = pd.read_csv(args.manifest)
        # Held-out images are extracted raw, so they still need bias alignment;
        # the training pool was already aligned on disk.
        pre = args.split != "heldout"
    else:
        mf, pre = data.build_manifest(root), False

    want = data.SPLIT_HELDOUT if args.split == "heldout" else data.SPLIT_VAL
    val = mf[mf.split == want]
    if args.limit:
        # Stratified: manifests are class-ordered, so head() takes one class and
        # every metric comes back NaN.
        val = (val.groupby("y", group_keys=False)
                  .apply(lambda g: g.head(max(1, args.limit // 2)))
                  .reset_index(drop=True))
    if not len(val):
        raise SystemExit(f"no rows with split=={want} in {args.manifest}")
    print(f"evaluating {len(val)} images from split={want}")

    # 1. robustness grid across every battery cell
    # Single transforms first, then composed chains. Nearly all published work
    # reports only single ops at one severity; real laundering is a sequence,
    # and chains are where detectors actually break.
    to_score = list(args.cells or T.BATTERY)
    if not args.no_chains and not args.cells:
        to_score += list(T.CHAIN_BATTERY)

    rows, raw_scores, raw_y = [], {}, None
    for cell in to_score:
        p, y = score(model, val, preprocess, cell, root, device,
                     args.batch_size, args.workers, pre)
        raw_scores[cell], raw_y = p, y
        kind = "chain" if cell in T.CHAIN_BATTERY else (
            "clean" if cell == "clean" else "single")
        rows.append({"cell": cell, "kind": kind, "auc": roc_auc_score(y, p),
                     "acc": float(((p > .5) == y).mean())})
        print(f"  {cell:12s} auc={rows[-1]['auc']:.4f}  acc={rows[-1]['acc']:.4f}")
    grid = pd.DataFrame(rows)

    clean = float(grid.loc[grid.cell == "clean", "auc"].iloc[0])
    laundered = grid[grid.kind == "single"]
    chains = grid[grid.kind == "chain"]

    # 2. unseen generator. ADM only -- DDIM and DDPM are near-siblings, so
    #    holding one out while training on the other is not a real test.
    #    Skipped rather than fatal if the archives are absent: the grid above
    #    costs minutes and must not be thrown away over a missing file.
    have_unseen = (args.split != "heldout"
                   and (root / ADM_ZIP).exists() and (root / REAL_ZIP).exists())
    if have_unseen:
        fakes = data.scan_wildfake_zip(ADM_ZIP, "ADM", data.SID_SYNTHETIC, root,
                                       limit=args.n_unseen)
        reals = data.scan_wildfake_zip(REAL_ZIP, "imagenet", data.SID_REAL, root,
                                       limit=args.n_unseen)
        unseen = pd.concat([reals, fakes], ignore_index=True)
        pu, yu = score(model, unseen, preprocess, "clean", root, device,
                       args.batch_size, args.workers)  # zip images never pre-extracted
    else:
        print(f"\n  skipping unseen-generator eval: {ADM_ZIP} or {REAL_ZIP} not found")
        pu = yu = None

    total, trainable = count_params(model)
    summary = {
        "clean_auc": clean,
        "laundered_auc_mean": float(laundered.auc.mean()),
        "laundered_auc_worst": float(laundered.auc.min()),
        "robustness_gap": clean - float(laundered.auc.mean()),
        **({"chain_auc_mean": float(chains.auc.mean()),
            "chain_auc_worst": float(chains.auc.min()),
            # Chains compose several ops, so this gap should exceed the
            # single-transform one. If it does not, the chains are too mild.
            "chain_gap": clean - float(chains.auc.mean())} if len(chains) else {}),
        **({"unseen_adm_auc": roc_auc_score(yu, pu),
            "unseen_adm_miss_rate": float((pu[yu == 1] <= .5).mean())}
           if have_unseen else {}),
        "n_params": total, "n_trainable": trainable,
        "gpu_hours": ckpt.get("gpu_hours", float("nan")),
    }
    print()
    for k, v in summary.items():
        print(f"  {k:22s} {v:,}" if isinstance(v, int) else f"  {k:22s} {v:.4f}")

    suffix = "" if args.split == "val" else f"_{args.split}"
    out = Path(f"results/tables/eval_{args.checkpoint.stem}{suffix}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.assign(**summary).to_csv(out, index=False)

    try:
        cache_dir = root / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_dir / f"scores_{args.checkpoint.stem}{suffix}.npz",
            **{f"p_{c}": v for c, v in raw_scores.items()}, y=raw_y,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  (score cache not written: {exc})")

    # Attach to the training run so config, GPU-hours and results share one row.
    rid = ckpt.get("wandb_run_id")
    if rid:
        wandb.init(project=args.project, id=rid, resume="allow")
        wandb.log({**summary, **{f"auc_{r.cell}": r.auc for r in grid.itertuples()}})
        wandb.finish()
        print(f"\nlogged to W&B run {rid}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
