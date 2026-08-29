#!/usr/bin/env python
"""Strip a checkpoint down to what is actually ours.

A full checkpoint is ~1.2 GB, but 96% of that is DINOv3's frozen weights, which
timm downloads from HuggingFace anyway. Only the adapters and heads are trained
-- 53 MB. For deployment that is the difference between a container that ships a
model and one that fetches a public backbone and adds 53 MB on top.

Rebuilding is exact: apply_peft's SVD is deterministic given the same pretrained
weights, so the frozen base is reconstructed bit-for-bit.

    python scripts/export_slim.py --checkpoint checkpoints/svd_r32_film.pt
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

TRAINABLE = ("lora_A", "lora_B", "severity.", "conditioner.", "classifier.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    slim = {k: v for k, v in ck["state_dict"].items()
            if any(t in k for t in TRAINABLE)}

    out = args.out or args.checkpoint.with_name(args.checkpoint.stem + "_slim.pt")
    torch.save({"state_dict": slim, "args": ck["args"], "slim": True,
                "n_params": ck.get("n_params"),
                "gpu_hours": ck.get("gpu_hours")}, out)

    before = args.checkpoint.stat().st_size / 1e6
    after = out.stat().st_size / 1e6
    print(f"{before:.0f} MB -> {after:.0f} MB  ({100*after/before:.1f}%)   {out}")


if __name__ == "__main__":
    main()
