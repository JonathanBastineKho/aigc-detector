#!/usr/bin/env python
"""Demo: watch a detector be laundered into being confidently wrong.

The point is not "upload an image, get a score" -- every team will build that.
It is the BEHAVIOUR under progressive laundering: a standard detector stays
confident all the way into being wrong, while ours holds, then refuses to answer.

The laundering here calls the same apply_chain() the robustness table uses, so
this is the evaluation running live, not a mock-up of it.

    python app.py [--share]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from src.utils import dataset as data, transforms as T          # noqa: E402
from src.utils.features import align_bias, load_backbone, pick_device  # noqa: E402

# Progressive laundering: each stage adds an op, mirroring an image being
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

# Measured on the val split (results/tables/probe_robustness.csv): the
# accuracy-optimal threshold falls from ~0.67 on clean images to ~0.00 under
# heavy laundering, because laundering drags every score toward "real".
ABSTAIN_SEVERITY = 0.55     # beyond this, evidence is too thin to trust
ABSTAIN_MARGIN = 0.30       # ...and the score is not decisive enough

MODEL = {}


def load(model_path: Path | None = None):
    root = data.data_root()
    bundle = joblib.load(model_path or root / "cache" / "probe.joblib")
    device = pick_device()
    backbone, preprocess = load_backbone(bundle["backbone"], device)
    MODEL.update(clf=bundle["clf"], backbone=backbone,
                 preprocess=preprocess, device=device)


@torch.no_grad()
def raw_score(img) -> float:
    x = MODEL["preprocess"](img).unsqueeze(0).to(MODEL["device"])
    feats = MODEL["backbone"](x).float().cpu().numpy()
    return float(MODEL["clf"].predict_proba(feats)[0, 1])


def severity_corrected(p: float, sev: np.ndarray) -> tuple[str, float, bool]:
    """Shift the decision threshold by how laundered the image is, and abstain
    when the evidence is too thin to place it either side.

    NOTE: severity is KNOWN here because the demo applied the transforms. In the
    deployed system the severity head estimates it from the image alone -- this
    is the one place the demo is ahead of the model.
    """
    total = float(sev.max())
    thr = 0.5 * (1.0 - total)                       # threshold falls with severity
    if total > ABSTAIN_SEVERITY and abs(p - thr) < ABSTAIN_MARGIN:
        return "ESCALATE", total, True
    return ("AI-generated" if p > thr else "Authentic"), total, False


def run(img, stage_idx: int):
    if img is None:
        return None, {}, "—", {}, "—", "upload an image to begin"

    label, chain = STAGES[int(stage_idx)]
    base_img = align_bias(img, 96, 224)             # same preprocessing as training
    laundered, log = T.apply_chain(base_img, chain)
    sev = T.severity_vector(log)

    p = raw_score(laundered)

    # Baseline: a fixed 0.5 threshold, no notion of degradation. This is what a
    # standard detector does, and it is what goes confidently wrong.
    base_label = "AI-generated" if p > 0.5 else "Authentic"
    base_conf = p if p > 0.5 else 1 - p

    ours_label, total_sev, abstained = severity_corrected(p, sev)
    ours_conf = 0.0 if abstained else abs(p - 0.5 * (1 - total_sev)) * 2

    sev_txt = "  ".join(
        f"{n}={v:.2f}" for n, v in zip(T.OP_NAMES, sev) if v > 0.01
    ) or "none"

    note = (f"**{label}** — estimated severity: {sev_txt}"
            + ("\n\n⚠️ laundered past reliable range — routed to human review"
               if abstained else ""))

    return (
        laundered,
        {base_label: base_conf, ("Authentic" if base_label == "AI-generated"
                                 else "AI-generated"): 1 - base_conf},
        f"### {base_label}",
        ({"ESCALATE": 1.0} if abstained
         else {ours_label: ours_conf, "uncertain": 1 - ours_conf}),
        f"### {'⚠️ ESCALATE' if abstained else ours_label}",
        note,
    )


def build():
    with gr.Blocks(title="Laundering-Aware AIGC Detection") as demo:
        gr.Markdown(
            "# You cannot launder your way past this system\n"
            "You can only launder your way into human review. "
            "Drag the slider and watch a standard detector become "
            "confidently wrong."
        )
        with gr.Row():
            inp = gr.Image(type="pil", label="Upload an image", height=280)
            out_img = gr.Image(label="After laundering", height=280)

        slider = gr.Slider(0, len(STAGES) - 1, value=0, step=1,
                           label="Laundering  (clean → reposted → cropped → thumbnailed → reposted)")
        note = gr.Markdown("upload an image to begin")

        with gr.Row():
            with gr.Column():
                gr.Markdown("## Standard detector")
                base_hdr = gr.Markdown("—")
                base_out = gr.Label(num_top_classes=2, show_label=False)
            with gr.Column():
                gr.Markdown("## Ours")
                ours_hdr = gr.Markdown("—")
                ours_out = gr.Label(num_top_classes=2, show_label=False)

        outputs = [out_img, base_out, base_hdr, ours_out, ours_hdr, note]
        for ev in (inp.change, slider.change):
            ev(fn=run, inputs=[inp, slider], outputs=outputs)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path)
    ap.add_argument("--share", action="store_true", help="public 72h tunnel")
    args = ap.parse_args()
    load(args.model)
    build().launch(share=args.share)
