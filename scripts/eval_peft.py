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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from aigcd import data, transforms as T                        # noqa: E402
from aigcd.features import align_bias, pick_device             # noqa: E402
from aigcd.models.heads import Detector                        # noqa: E402
from aigcd.models.peft import apply_peft, count_params         # noqa: E402

ADM_ZIP = "wildfake/Images/Diffusion_based/ADM.zip"
REAL_ZIP = "wildfake/Images/Real/imagenet.zip"


class Cell(Dataset):
    """One battery cell, applied deterministically. No curriculum, no sampling."""

    def __init__(self, df, preprocess, cell, root, pre_extracted=False):
        self.rows = df.reset_index(drop=True)
        self.preprocess, self.root, self.pre = preprocess, root, pre_extracted
        op, kwargs = T.BATTERY[cell]
        self.chain = [] if op == "none" else [(op, kwargs)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows.iloc[i]
        img = data.load_image(row.path, self.root)
        if not self.pre:
            img = align_bias(img)
        if self.chain:
            img, _ = T.apply_chain(img, self.chain)
        return self.preprocess(img), torch.tensor(float(row.y))


@torch.no_grad()
def score(model, df, preprocess, cell, root, device, bs, workers, pre=False):
    dl = DataLoader(Cell(df, preprocess, cell, root, pre), batch_size=bs,
                    num_workers=workers)
    ps, ys = [], []
    for x, y in dl:
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
    ap.add_argument("--project", default="aigc-detector")
    args = ap.parse_args()

    device = args.device or pick_device()
    root = data.data_root()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = load_model(ckpt, device)

    cfg = timm.data.resolve_model_data_config(model.backbone)
    preprocess = timm.data.create_transform(**cfg, is_training=False)

    if args.manifest:
        mf, pre = pd.read_csv(args.manifest), True
    else:
        mf, pre = data.build_manifest(root), False
    val = mf[mf.split == data.SPLIT_VAL]

    # 1. robustness grid across every battery cell
    rows = []
    for cell in T.BATTERY:
        p, y = score(model, val, preprocess, cell, root, device,
                     args.batch_size, args.workers, pre)
        rows.append({"cell": cell, "auc": roc_auc_score(y, p),
                     "acc": float(((p > .5) == y).mean())})
        print(f"  {cell:12s} auc={rows[-1]['auc']:.4f}  acc={rows[-1]['acc']:.4f}")
    grid = pd.DataFrame(rows)

    clean = float(grid.loc[grid.cell == "clean", "auc"].iloc[0])
    laundered = grid[grid.cell != "clean"]

    # 2. unseen generator. ADM only -- DDIM and DDPM are near-siblings, so
    #    holding one out while training on the other is not a real test.
    fakes = data.scan_wildfake_zip(ADM_ZIP, "ADM", data.SID_SYNTHETIC, root,
                                   limit=args.n_unseen)
    reals = data.scan_wildfake_zip(REAL_ZIP, "imagenet", data.SID_REAL, root,
                                   limit=args.n_unseen)
    unseen = pd.concat([reals, fakes], ignore_index=True)
    pu, yu = score(model, unseen, preprocess, "clean", root, device,
                   args.batch_size, args.workers)   # zip images are never pre-extracted

    total, trainable = count_params(model)
    summary = {
        "clean_auc": clean,
        "laundered_auc_mean": float(laundered.auc.mean()),
        "laundered_auc_worst": float(laundered.auc.min()),
        "robustness_gap": clean - float(laundered.auc.mean()),
        "unseen_adm_auc": roc_auc_score(yu, pu),
        "unseen_adm_miss_rate": float((pu[yu == 1] <= .5).mean()),
        "n_params": total, "n_trainable": trainable,
        "gpu_hours": ckpt.get("gpu_hours", float("nan")),
    }
    print()
    for k, v in summary.items():
        print(f"  {k:22s} {v:,}" if isinstance(v, int) else f"  {k:22s} {v:.4f}")

    out = Path(f"results/tables/eval_{args.checkpoint.stem}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.assign(**summary).to_csv(out, index=False)

    # Attach to the training run so config, GPU-hours and results share one row.
    rid = ckpt.get("wandb_run_id")
    if rid:
        wandb.init(project=args.project, id=rid, resume="must")
        wandb.log({**summary, **{f"auc_{r.cell}": r.auc for r in grid.itertuples()}})
        wandb.finish()
        print(f"\nlogged to W&B run {rid}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
