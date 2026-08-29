#!/usr/bin/env python
"""Demo: what laundering does to a detector.

Two real models, no simulation. The left panel is a frozen linear probe on
DINOv3 features -- the standard approach, and what most published baselines
amount to. The right is the same backbone with parameter-efficient adaptation
and laundering-augmented training.

Drag the slider and watch them diverge. We measured that 43% of AI images can be
laundered past the probe versus 24% past the adapted model; this shows one case
of it.

The laundering here calls the same apply_chain() the robustness table uses, so
this is the evaluation running live rather than a mock-up of it.

    python app.py --checkpoint checkpoints/lora_r32_film_slim.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import timm
import torch

sys.path.insert(0, str(Path(__file__).parent))
from src.components.peft import adapters_disabled, apply_peft   # noqa: E402
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

M = {}


def load(checkpoint: Path, probe_path: Path | None):
    """One backbone, two detectors. adapters_disabled() recovers the frozen
    features the probe was fitted on, so the baseline costs no extra memory."""
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    targs, device = ck["args"], pick_device()

    bb = timm.create_model(targs["backbone"], pretrained=True, num_classes=0)
    bb = apply_peft(bb, mode=targs["arm"], r=targs["rank"])
    model = Detector(bb, dim=bb.num_features, conditioner=targs["conditioner"])
    model.load_state_dict(ck["state_dict"], strict=not ck.get("slim"))

    cfg = timm.data.resolve_model_data_config(bb)
    M.update(model=model.eval().to(device), device=device,
             preprocess=timm.data.create_transform(**cfg, is_training=False),
             arm=targs["arm"])

    probe_path = probe_path or Path("data/cache/probe.joblib")
    M["probe"] = joblib.load(probe_path)["clf"] if probe_path.exists() else None
    print(f"detector: {checkpoint.name} ({targs['arm']})")
    print(f"baseline: {'frozen probe' if M['probe'] else 'NONE — left panel disabled'}")


@torch.no_grad()
def both_scores(img) -> tuple[float, float | None]:
    """p(AI) from the adapted detector and from the frozen probe."""
    x = M["preprocess"](img).unsqueeze(0).to(M["device"])
    ours = float(torch.sigmoid(M["model"](x)["logit"])[0])

    baseline = None
    if M["probe"] is not None:
        with adapters_disabled(M["model"]) as m:
            h = m.backbone(x).float().cpu().numpy()
        baseline = float(M["probe"].predict_proba(h)[0, 1])
    return ours, baseline


def panel(p: float | None) -> tuple[dict, str]:
    if p is None:
        return {}, "### —"
    label = "AI-generated" if p > 0.5 else "Authentic"
    conf = p if p > 0.5 else 1 - p
    other = "Authentic" if label == "AI-generated" else "AI-generated"
    return {label: conf, other: 1 - conf}, f"### {label}  ·  {conf:.0%}"


def run(img, stage_idx: int):
    if img is None:
        return None, {}, "### —", {}, "### —", ""

    name, chain = STAGES[int(stage_idx)]

    # Launder the FULL image for display, so the viewer sees their own photo
    # degrading. Scoring uses the 224 crop the model was trained on -- showing
    # that crop instead made an untouched image look heavily transformed.
    shown, _ = T.apply_chain(img, chain)
    scored, _ = T.apply_chain(align_bias(img, 96, 224), chain)

    ours, baseline = both_scores(scored)
    ours_lbl, ours_hdr = panel(ours)
    base_lbl, base_hdr = panel(baseline)

    note = f"**{name}**"
    if baseline is not None and (baseline > 0.5) != (ours > 0.5):
        note += "  —  the two detectors now disagree"
    return shown, base_lbl, base_hdr, ours_lbl, ours_hdr, note


def build():
    with gr.Blocks(title="Laundering and AIGC detection") as demo:
        gr.Markdown(
            "## Laundering an AI image past a detector\n"
            "Two real detectors on the same DINOv3 backbone. Left: a frozen "
            "linear probe. Right: parameter-efficient adaptation trained on "
            "laundered images. Drag the slider."
        )
        with gr.Row():
            inp = gr.Image(type="pil", label="Upload an image", height=300)
            out_img = gr.Image(label="After laundering", height=300)

        slider = gr.Slider(
            0, len(STAGES) - 1, value=0, step=1,
            label="Laundering   clean → reposted → cropped → thumbnailed → reposted",
        )
        note = gr.Markdown("")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Frozen probe  (baseline)")
                base_hdr = gr.Markdown("### —")
                base_out = gr.Label(num_top_classes=2, show_label=False)
            with gr.Column():
                gr.Markdown("### Adapted detector  (ours)")
                ours_hdr = gr.Markdown("### —")
                ours_out = gr.Label(num_top_classes=2, show_label=False)

        outputs = [out_img, base_out, base_hdr, ours_out, ours_hdr, note]
        for ev in (inp.change, slider.change):
            ev(fn=run, inputs=[inp, slider], outputs=outputs)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("checkpoints/lora_r32_film_slim.pt"))
    ap.add_argument("--probe", type=Path)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    load(args.checkpoint, args.probe)
    build().launch(share=args.share)
