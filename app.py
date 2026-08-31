#!/usr/bin/env python
"""Demo: what laundering does to a detector.

Upload an image and drag the slider. Each stage adds one more operation --
reposted, cropped, thumbnailed, reposted again -- and the image is rescored
after every one.

The laundering calls the same apply_chain() the robustness table uses, so this
is the evaluation running live rather than a mock-up of it. Nothing is
pre-computed.

    python app.py --checkpoint checkpoints/lora_r32_film_resize_slim.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import timm
import torch

sys.path.insert(0, str(Path(__file__).parent))
from src.components.peft import apply_peft                      # noqa: E402
from src.models.detector import Detector                        # noqa: E402
from src.utils import transforms as T                           # noqa: E402
from src.utils.features import align_bias, pick_device          # noqa: E402

# Progressive laundering: each stage adds an operation, mirroring an image being
# reposted, screenshotted and re-encoded on its way across platforms.
STAGES = [
    ("clean", []),
    ("reposted once", [("jpeg", {"quality": 70})]),
    ("+ cropped", [("jpeg", {"quality": 70}), ("crop", {"frac": 0.8})]),
    ("+ thumbnailed", [("jpeg", {"quality": 70}), ("crop", {"frac": 0.8}),
                       ("resize", {"scale": 0.5})]),
    ("+ reposted again", [("jpeg", {"quality": 70}), ("crop", {"frac": 0.8}),
                          ("resize", {"scale": 0.5}), ("jpeg", {"quality": 30})]),
]

# The severity head's ABSOLUTE output does not transfer across image sources --
# calibrated on SID_Set it pins to the ceiling on DALL-E crops. But it is
# monotonic in laundering (r = 0.71), so we show the RISE relative to the same
# image untouched. That cancels the per-image offset and is what the head can
# actually support. 0.20 is its measured clean-to-worst span.
SEV_SPAN = 0.20

M = {}


def load(checkpoint: Path):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    targs, device = ck["args"], pick_device()

    bb = timm.create_model(targs["backbone"], pretrained=True, num_classes=0)
    bb = apply_peft(bb, mode=targs["arm"], r=targs["rank"])
    model = Detector(bb, dim=bb.num_features, conditioner=targs["conditioner"],
                     bounded_severity=not targs.get("unbounded_severity", False))
    model.load_state_dict(ck["state_dict"], strict=not ck.get("slim"))

    cfg = timm.data.resolve_model_data_config(bb)
    # Preprocess the way this checkpoint was trained. A resize-trained model fed
    # cropped inputs is a train/test mismatch that quietly degrades every
    # prediction -- the training manifest path records which was used.
    align = "resize" if "resize" in str(targs.get("manifest", "")) else "crop"
    M.update(model=model.eval().to(device), device=device,
             preprocess=timm.data.create_transform(**cfg, is_training=False),
             arm=targs["arm"], align=align)

    print(f"detector: {checkpoint.name} ({targs['arm']}, align={align})")


@torch.no_grad()
def score(img) -> tuple[float, float]:
    """p(AI) and the model's own estimate of how laundered the image is."""
    x = M["preprocess"](img).unsqueeze(0).to(M["device"])
    out = M["model"](x)
    ours = float(torch.sigmoid(out["logit"])[0])
    sev = float(out["s_hat"][0].float().cpu().numpy().max())

    return ours, sev


def panel(p: float | None) -> tuple[dict, str]:
    if p is None:
        return {}, "### —"
    label = "AI-generated" if p > 0.5 else "Authentic"
    conf = p if p > 0.5 else 1 - p
    other = "Authentic" if label == "AI-generated" else "AI-generated"
    return {label: conf, other: 1 - conf}, f"### {label}  ·  {conf:.0%}"


def run(img, stage_idx: int):
    if img is None:
        return None, {}, "### —", ""

    name, chain = STAGES[int(stage_idx)]

    # Launder the FULL image for display, so the viewer sees their own photo
    # degrading. Scoring uses the 224 crop the model was trained on -- showing
    # that crop instead made an untouched image look heavily transformed.
    shown, _ = T.apply_chain(img, chain)
    scored, _ = T.apply_chain(align_bias(img, 96, 224, M["align"]), chain)

    ours, sev = score(scored)
    # Baseline the estimate against this same image untouched.
    _, sev_clean = score(align_bias(img, 96, 224, M["align"]))
    rise = float(np.clip((sev - sev_clean) / SEV_SPAN, 0.0, 1.0))

    filled = int(round(rise * 8))
    bar = "▓" * filled + "░" * (8 - filled)
    level = ("none" if rise < .15 else "light" if rise < .45
             else "moderate" if rise < .75 else "heavy")

    note = f"**{name}**  \nlaundering detected: `{bar}` {level}"
    lbl, hdr = panel(ours)
    return shown, lbl, hdr, note


def build():
    with gr.Blocks(title="Laundered AI detector") as demo:
        gr.Markdown("## Laundered AI detector")
        with gr.Row():
            inp = gr.Image(type="pil", label="Upload an image", height=300)
            out_img = gr.Image(label="After laundering", height=300)

        slider = gr.Slider(
            0, len(STAGES) - 1, value=0, step=1,
            label="Laundering   clean → reposted → cropped → thumbnailed → reposted",
        )
        note = gr.Markdown("")
        hdr = gr.Markdown("### —")
        out = gr.Label(num_top_classes=2, show_label=False)

        for ev in (inp.change, slider.change):
            ev(fn=run, inputs=[inp, slider], outputs=[out_img, out, hdr, note])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("checkpoints/lora_r32_film_resize_slim.pt"))
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    load(args.checkpoint)
    build().launch(share=args.share)
