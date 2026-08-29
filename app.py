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
import timm
import torch

sys.path.insert(0, str(Path(__file__).parent))
from src.components.peft import apply_peft                      # noqa: E402
from src.models.detector import Detector                        # noqa: E402
from src.utils import transforms as T                           # noqa: E402
from src.utils.features import align_bias, pick_device          # noqa: E402

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


def load(checkpoint: Path):
    """Build the detector. Slim checkpoints carry only adapters and heads --
    the frozen DINOv3 base is rebuilt from timm, which is exact because
    apply_peft's SVD is deterministic given the same pretrained weights."""
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    targs = ck["args"]
    device = pick_device()

    backbone = timm.create_model(targs["backbone"], pretrained=True, num_classes=0)
    backbone = apply_peft(backbone, mode=targs["arm"], r=targs["rank"])
    model = Detector(backbone, dim=backbone.num_features,
                     conditioner=targs["conditioner"])
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    if unexpected:
        raise SystemExit(f"unexpected keys in checkpoint: {unexpected[:3]}")
    if missing and not ck.get("slim"):
        raise SystemExit(f"missing keys in a full checkpoint: {missing[:3]}")

    cfg = timm.data.resolve_model_data_config(backbone)
    MODEL.update(model=model.eval().to(device), device=device,
                 preprocess=timm.data.create_transform(**cfg, is_training=False),
                 name=checkpoint.stem)

    # The jointly-trained severity head collapsed to predicting a constant
    # (0.396 correlation across a 0.025 range), so it cannot drive the
    # abstention rule. A ridge fitted separately on the same features reaches
    # ~0.66 with full range. Prefer it when available; fall back to the head so
    # the app still runs without one.
    stem = checkpoint.stem.replace("_slim", "")
    for cand in (checkpoint.parent.parent / "data" / "cache" / f"severity_{checkpoint.stem}.joblib",
                 Path("data/cache") / f"severity_{stem}.joblib",
                 Path("data/cache") / f"severity_{checkpoint.stem}.joblib"):
        if cand.exists():
            MODEL["severity"] = joblib.load(cand)["ridge"]
            print(f"severity estimator: {cand.name}")
            break
    else:
        MODEL["severity"] = None
        print("severity estimator: NONE -- falling back to the collapsed head; "
              "abstention will not fire")


@torch.no_grad()
def score(img) -> tuple[float, np.ndarray]:
    """Returns p(AI) and the model's ESTIMATED severity vector.

    s_hat is inferred from the image alone -- the demo does not tell the model
    what it did. That is the difference between showing the system work and
    showing a script.
    """
    x = MODEL["preprocess"](img).unsqueeze(0).to(MODEL["device"])
    out = MODEL["model"](x)
    p = float(torch.sigmoid(out["logit"])[0])

    if MODEL["severity"] is not None:
        h = out["h"].float().cpu().numpy()
        s_hat = np.clip(MODEL["severity"].predict(h)[0], 0.0, 1.0)
    else:
        s_hat = out["s_hat"][0].float().cpu().numpy()
    return p, s_hat


def severity_corrected(p: float, s_hat: np.ndarray) -> tuple[str, float, bool]:
    """Shift the decision threshold by how laundered the model THINKS it is,
    and abstain when the evidence is too thin to place it either side."""
    total = float(s_hat.max())
    thr = 0.5 * (1.0 - total)
    if total > ABSTAIN_SEVERITY and abs(p - thr) < ABSTAIN_MARGIN:
        return "ESCALATE", total, True
    return ("AI-generated" if p > thr else "Authentic"), total, False


def run(img, stage_idx: int):
    if img is None:
        return None, {}, "—", {}, "—", "upload an image to begin"

    label, chain = STAGES[int(stage_idx)]
    base_img = align_bias(img, 96, 224)
    laundered, log = T.apply_chain(base_img, chain)
    true_sev = T.severity_vector(log)

    p, s_hat = score(laundered)

    # Left panel: the same detector with a fixed 0.5 threshold and no
    # abstention -- what a standard system does.
    base_label = "AI-generated" if p > 0.5 else "Authentic"
    base_conf = p if p > 0.5 else 1 - p

    ours_label, total_sev, abstained = severity_corrected(p, s_hat)
    ours_conf = 0.0 if abstained else abs(p - 0.5 * (1 - total_sev)) * 2

    shown = "  ".join(f"{n}={v:.2f}" for n, v in zip(T.OP_NAMES, s_hat) if v > 0.05)
    actual = "  ".join(f"{n}={v:.2f}" for n, v in zip(T.OP_NAMES, true_sev) if v > 0.01)

    note = (f"**{label}**\n\n"
            f"model's estimate: `{shown or 'clean'}`  \n"
            f"actually applied: `{actual or 'nothing'}`"
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
                gr.Markdown("## Fixed threshold, no abstention")
                base_hdr = gr.Markdown("—")
                base_out = gr.Label(num_top_classes=2, show_label=False)
            with gr.Column():
                gr.Markdown("## Severity-aware + abstention")
                ours_hdr = gr.Markdown("—")
                ours_out = gr.Label(num_top_classes=2, show_label=False)

        outputs = [out_img, base_out, base_hdr, ours_out, ours_hdr, note]
        for ev in (inp.change, slider.change):
            ev(fn=run, inputs=[inp, slider], outputs=outputs)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("checkpoints/svd_r32_film_slim.pt"))
    ap.add_argument("--share", action="store_true", help="public 72h tunnel")
    args = ap.parse_args()
    load(args.checkpoint)
    build().launch(share=args.share)
