"""Training loop: PEFT arm + consistency loss + severity head, in one run.

    L = BCE(clean) + BCE(laundered)
        + 0.50 * KL(p || p~)       same verdict either way
        + 0.25 * MSE(h, h~)        same features either way
        + 0.50 * SmoothL1(s^, s)   severity, supervised for free

Coefficients are TeleAI's published NTIRE 2026 values. Both BCE terms carry the
SAME label -- laundering does not change whether an image was generated, and
teaching that invariance is the entire point.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import wandb
from torch.utils.data import DataLoader, Dataset

from .. import data, transforms as T
from ..features import DEFAULT_BACKBONE, align_bias, pick_device
from ..models.heads import Detector
from ..models.peft import apply_peft, count_params


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
        img = data.load_image(row.path, self.root)
        # Pre-extracted files are already cropped and recompressed. Aligning
        # again would add a second JPEG pass to every training sample -- an
        # artifact present in training but not at inference, and painful to find.
        if not self.pre_extracted:
            img = align_bias(img, self.jpeg_q, self.crop)

        rng = np.random.default_rng((self.epoch << 32) ^ i)
        chain = T.sample_chain(self.epoch, rng, self.max_epochs)
        laundered, log = T.apply_chain(img, chain)

        return (
            self.preprocess(img),
            self.preprocess(laundered),
            torch.tensor(float(row.y)),
            torch.from_numpy(T.severity_vector(log)),
        )


def build_model(arm: str, r: int, conditioner: str, backbone_name: str, device: str):
    backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
    backbone = apply_peft(backbone, mode=arm, r=r)
    model = Detector(backbone, dim=backbone.num_features, conditioner=conditioner)
    for p in list(model.severity.parameters()) + list(model.classifier.parameters()):
        p.requires_grad_(True)
    if model.conditioner.net is not None:
        for p in model.conditioner.parameters():
            p.requires_grad_(True)
    return model.to(device)


def loss_fn(out_c, out_l, y, s, a=0.5, b=0.25, c=0.5):
    bce = Fn.binary_cross_entropy_with_logits
    l_det = bce(out_c["logit"], y) + bce(out_l["logit"], y)

    # Symmetric KL between the two verdicts (DCPT / HIT-VIRLAB's DOCL).
    p_c = torch.stack([out_c["logit"], -out_c["logit"]], -1).log_softmax(-1)
    p_l = torch.stack([out_l["logit"], -out_l["logit"]], -1).log_softmax(-1)
    l_kl = 0.5 * (Fn.kl_div(p_l, p_c, log_target=True, reduction="batchmean")
                  + Fn.kl_div(p_c, p_l, log_target=True, reduction="batchmean"))

    l_feat = Fn.mse_loss(out_l["h"], out_c["h"].detach())
    # Severity is only defined for the laundered view; the clean view is all-zero
    # by construction and supervising it would just teach the head to output 0.
    l_sev = Fn.smooth_l1_loss(out_l["s_hat"], s)

    total = l_det + a * l_kl + b * l_feat + c * l_sev
    return total, {"det": l_det.item(), "kl": l_kl.item(),
                   "feat": l_feat.item(), "sev": l_sev.item()}


def main() -> None:
    ap = argparse.ArgumentParser(description="PEFT training with consistency + severity")
    ap.add_argument("--arm", default="svd", choices=["frozen", "full", "lora", "svd"])
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--conditioner", default="film", choices=["film", "gate", "concat", "none"])
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="rows per epoch, for smoke tests")
    ap.add_argument("--max-steps", type=int, help="stop early, for smoke tests")
    ap.add_argument("--manifest", type=Path,
                    help="pre-extracted manifest CSV (much faster than parquet)")
    ap.add_argument("--holdout", nargs="+",
                    help="generators to EXCLUDE from training (leave-one-generator-out)")
    ap.add_argument("--project", default="aigc-detector")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or pick_device()
    root = data.data_root()
    if args.manifest:
        mf = pd.read_csv(args.manifest)
        pre_extracted = True          # align_bias already applied on disk
    else:
        mf = data.build_manifest(root)
        pre_extracted = False
    train_df = mf[mf.split == data.SPLIT_TRAIN]
    if args.holdout:
        before = len(train_df)
        train_df = train_df[~train_df.generator.isin(args.holdout)]
        print(f"holding out {args.holdout}: {before} -> {len(train_df)} training rows")
    data.assert_no_heldout(train_df)
    if args.limit:
        train_df = train_df.head(args.limit)

    model = build_model(args.arm, args.rank, args.conditioner, args.backbone, device)
    total, trainable = count_params(model)
    print(f"arm={args.arm} rank={args.rank} cond={args.conditioner} device={device}")
    print(f"params: {total:,} total, {trainable:,} trainable ({100*trainable/total:.2f}%)")

    cfg = timm.data.resolve_model_data_config(model.backbone)
    preprocess = timm.data.create_transform(**cfg, is_training=False)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.02
    )

    run_name = f"{args.arm}_r{args.rank}_{args.conditioner}"
    if args.holdout:
        run_name += "_no" + "-".join(args.holdout)
    run = None if args.no_wandb else wandb.init(
        project=args.project, name=run_name,
        config={**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                # n_params is a Feasibility deliverable, not a curiosity -- log it
                # as a first-class value so the Pareto plot is a query later.
                "n_params": total, "n_trainable": trainable,
                "trainable_pct": round(100 * trainable / total, 3),
                "device": device, "n_train": len(train_df),
                "pre_extracted": pre_extracted},
    )

    gpu_seconds, global_step = 0.0, 0
    for epoch in range(args.epochs):
        ds = LaunderedPairs(train_df, preprocess, epoch, args.epochs, root,
                            pre_extracted=pre_extracted)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True)
        model.train()
        t0 = time.time()
        for step, (xc, xl, y, s) in enumerate(dl):
            xc, xl, y, s = xc.to(device), xl.to(device), y.to(device), s.to(device)
            total_loss, parts = loss_fn(model(xc), model(xl), y, s)
            opt.zero_grad(set_to_none=True)
            total_loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()

            if run:
                wandb.log({"loss": total_loss.item(), **parts, "epoch": epoch},
                          step=global_step)
            global_step += 1

            if step % 20 == 0:
                print(f"  ep{epoch} step{step:5d}  loss={total_loss.item():.4f}  "
                      + "  ".join(f"{k}={v:.4f}" for k, v in parts.items()))
            if args.max_steps and step + 1 >= args.max_steps:
                break
        gpu_seconds += time.time() - t0
        if run:
            wandb.log({"gpu_hours": gpu_seconds / 3600}, step=global_step)

    out = Path(args.out) / f"{run_name}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "args": {k: str(v) if isinstance(v, Path) else v
                         for k, v in vars(args).items()},
                "gpu_hours": gpu_seconds / 3600, "n_params": total,
                "wandb_run_id": run.id if run else None}, out)
    if run:
        wandb.log({"gpu_hours_total": gpu_seconds / 3600})
        wandb.finish()
    # Feasibility is scored on proportionate resource use -- so report it.
    print(f"\nsaved {out}   GPU-hours: {gpu_seconds/3600:.3f}")


if __name__ == "__main__":
    main()
