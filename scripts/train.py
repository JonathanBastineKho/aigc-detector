"""PEFT arm + consistency loss + severity head, in one training run.

    python scripts/train.py --arm svd --manifest data/extracted/manifest.csv

Arms: frozen | full | lora | svd. The comparison IS the contribution -- the
robustness gap (clean AUC minus laundered AUC) across arms tests whether
minor-direction adaptation preserves DINOv3's pretrained invariance.
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.components.peft import apply_peft, count_params     # noqa: E402
from src.losses import detection_loss                        # noqa: E402
from src.models.detector import Detector                     # noqa: E402
from src.utils import dataset as data                        # noqa: E402
from src.utils.dataset import LaunderedPairs                 # noqa: E402
from src.utils.features import DEFAULT_BACKBONE, pick_device  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(arm: str, rank: int, conditioner: str, backbone_name: str, device: str):
    backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
    backbone = apply_peft(backbone, mode=arm, r=rank)
    model = Detector(backbone, dim=backbone.num_features, conditioner=conditioner)
    # Heads are always trainable regardless of how the backbone is adapted.
    for module in (model.severity, model.classifier, model.conditioner):
        for p in module.parameters():
            p.requires_grad_(True)
    return model.to(device)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", default="svd", choices=["frozen", "full", "lora", "svd"])
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--conditioner", default="film",
                    choices=["film", "gate", "concat", "none"])
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    ap.add_argument("--manifest", type=Path,
                    help="pre-extracted manifest CSV (much faster than parquet)")
    ap.add_argument("--holdout", nargs="+",
                    help="generators to EXCLUDE from training (leave-one-generator-out)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, help="rows per epoch, for smoke tests")
    ap.add_argument("--max-steps", type=int, help="stop early, for smoke tests")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--project", default="aigc-detector")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device or pick_device()
    root = data.data_root()

    if args.manifest:
        manifest, pre_extracted = pd.read_csv(args.manifest), True
    else:
        manifest, pre_extracted = data.build_manifest(root), False

    train_df = manifest[manifest.split == data.SPLIT_TRAIN]
    if args.holdout:
        before = len(train_df)
        train_df = train_df[~train_df.generator.isin(args.holdout)]
        logger.info("holding out %s: %d -> %d training rows",
                    args.holdout, before, len(train_df))
    data.assert_no_heldout(train_df)

    model = build_model(args.arm, args.rank, args.conditioner, args.backbone, device)
    total, trainable = count_params(model)
    logger.info("arm=%s rank=%d cond=%s device=%s", args.arm, args.rank,
                args.conditioner, device)
    logger.info("params: %s total, %s trainable (%.2f%%)",
                f"{total:,}", f"{trainable:,}", 100 * trainable / total)

    cfg = timm.data.resolve_model_data_config(model.backbone)
    preprocess = timm.data.create_transform(**cfg, is_training=False)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.02,
    )

    run_name = f"{args.arm}_r{args.rank}_{args.conditioner}"
    if args.holdout:
        run_name += "_no" + "-".join(args.holdout)

    run = None if args.no_wandb else wandb.init(
        project=args.project, name=run_name,
        config={**{k: str(v) if isinstance(v, Path) else v
                   for k, v in vars(args).items()},
                # n_params is a Feasibility deliverable -- log it as a
                # first-class value so the Pareto plot is a query later.
                "n_params": total, "n_trainable": trainable,
                "trainable_pct": round(100 * trainable / total, 3),
                "device": device, "n_train": len(train_df),
                "pre_extracted": pre_extracted},
    )

    gpu_seconds, global_step = 0.0, 0
    for epoch in range(args.epochs):
        loader = DataLoader(
            LaunderedPairs(train_df, preprocess, epoch, args.epochs, root,
                           pre_extracted=pre_extracted),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, drop_last=True, pin_memory=(device == "cuda"),
        )
        model.train()
        epoch_start = __import__("time").time()

        for step, (x_clean, x_laundered, y, severity) in enumerate(loader):
            x_clean = x_clean.to(device, non_blocking=True)
            x_laundered = x_laundered.to(device, non_blocking=True)
            y, severity = y.to(device), severity.to(device)

            loss, parts = detection_loss(model(x_clean), model(x_laundered), y, severity)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()

            if run:
                wandb.log({"loss": loss.item(), **parts, "epoch": epoch}, step=global_step)
            global_step += 1

            if step % 20 == 0:
                logger.info("ep%d step%5d  loss=%.4f  %s", epoch, step, loss.item(),
                            "  ".join(f"{k}={v:.4f}" for k, v in parts.items()))
            if args.max_steps and step + 1 >= args.max_steps:
                break

        gpu_seconds += __import__("time").time() - epoch_start
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
    logger.info("saved %s   GPU-hours: %.3f", out, gpu_seconds / 3600)


if __name__ == "__main__":
    main()
