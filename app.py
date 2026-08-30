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

# The severity head's ABSOLUTE output does not transfer across image sources --
# calibrated on SID_Set it pins to the ceiling on DALL-E crops. But it is
# monotonic in laundering (r = 0.71), so we show the RISE relative to the same
# image untouched. That cancels the per-image offset and is what the head can
# actually support. 0.20 is its measured clean-to-worst span.
SEV_SPAN = 0.20

M = {}


def load(checkpoint: Path, probe_path: Path | None):
    """One backbone, two detectors. adapters_disabled() recovers the frozen
    features the probe was fitted on, so the baseline costs no extra memory."""
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    targs, device = ck["args"], pick_device()

    bb = timm.create_model(targs["backbone"], pretrained=True, num_classes=0)
    bb = apply_peft(bb, mode=targs["arm"], r=targs["rank"])
    model = Detector(bb, dim=bb.num_features, conditioner=targs["conditioner"],
                     bounded_severity=not targs.get("unbounded_severity", False))
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
def both_scores(img) -> tuple[float, float | None, float]:
    """p(AI) from the adapted detector, from the frozen probe, and the model's
    own estimate of how laundered the image is."""
    x = M["preprocess"](img).unsqueeze(0).to(M["device"])
    out = M["model"](x)
    ours = float(torch.sigmoid(out["logit"])[0])
    sev = float(out["s_hat"][0].float().cpu().numpy().max())

    baseline = None
    if M["probe"] is not None:
        with adapters_disabled(M["model"]) as m:
            h = m.backbone(x).float().cpu().numpy()
        baseline = float(M["probe"].predict_proba(h)[0, 1])
    return ours, baseline, sev


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

    ours, baseline, sev = both_scores(scored)
    # Baseline the estimate against this same image untouched.
    _, _, sev_clean = both_scores(align_bias(img, 96, 224))
    rise = float(np.clip((sev - sev_clean) / SEV_SPAN, 0.0, 1.0))
    ours_lbl, ours_hdr = panel(ours)
    base_lbl, base_hdr = panel(baseline)

    filled = int(round(rise * 8))
    bar = "▓" * filled + "░" * (8 - filled)
    level = ("none" if rise < .15 else "light" if rise < .45
             else "moderate" if rise < .75 else "heavy")

    note = (f"**{name}**  \n"
            f"laundering detected by the model: `{bar}` {level}")
    if baseline is not None and (baseline > 0.5) != (ours > 0.5):
        note += "  \n\n**The two detectors now disagree.**"
    return shown, base_lbl, base_hdr, ours_lbl, ours_hdr, note


def build():
    with gr.Blocks(title="Laundering and AIGC detection") as demo:
        gr.Markdown(
            "## Laundering an AI image past a detector\n"
            "Two real detectors on the same DINOv3 backbone. Left: a frozen "
            "linear probe. Right: parameter-efficient adaptation trained on "
            "laundered images. Drag the slider.\n\n"
            "*The bar below shows the model's own estimate of how laundered the "
            "image is, inferred from the image alone — it is not told what we did.*"
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
                    default=Path("checkpoints/lora_r32_film_v2_slim.pt"))
    ap.add_argument("--probe", type=Path)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    load(args.checkpoint, args.probe)
    build().launch(share=args.share)
