"""Laundering simulator: ops, evaluation battery, chain composition.

Shared by training augmentation, the robustness grid, and the attack-cost search
-- so all three agree on what "JPEG 70" means.

Every op returns (image, params). That log IS the severity supervision: because
we choose the transforms, the labels are exact and free.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# Order is fixed and load-bearing: it defines the layout of the severity vector
# that the severity head regresses against. Appending is safe; reordering is not.
OP_NAMES = ("jpeg", "blur", "resize", "noise", "jitter", "crop", "moire", "cast")

# Training samples severities from these ranges -- deliberately WIDER than the
# fixed grid in BATTERY. The brief says "a subset of the following", so the exact
# test severities may differ from the published table; training on the four
# published JPEG qualities would fit four points rather than the phenomenon.
SAMPLING_RANGES = {
    "jpeg": (25, 95),      # quality
    "blur": (0.3, 3.0),    # gaussian sigma
    "resize": (0.25, 0.9),  # downscale factor, then back up
    "noise": (0.01, 0.15),  # gaussian sigma, on [0,1] pixels
    "jitter": (0.05, 0.4),  # +/- fraction on brightness/contrast/saturation
    "crop": (0.6, 0.95),   # centre-crop fraction of the shorter side
    "moire": (0.2, 1.0),   # strength of the rescreen pattern
    "cast": (0.05, 0.35),  # strength of the colour cast
}


# --------------------------------------------------------------------------
# 1. Ops
# --------------------------------------------------------------------------

def jpeg(img: Image.Image, quality: float) -> tuple[Image.Image, dict]:
    """A REAL encode/decode round-trip, not a blur that looks like one.

    JPEG artifacts are 8x8 DCT block quantisation; no spatial filter reproduces
    them. Approximating this invalidates the entire robustness table.
    """
    q = int(round(quality))
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB"), {"op": "jpeg", "quality": q}


def blur(img: Image.Image, sigma: float) -> tuple[Image.Image, dict]:
    return img.filter(ImageFilter.GaussianBlur(radius=sigma)), {"op": "blur", "sigma": sigma}


def resize(img: Image.Image, scale: float) -> tuple[Image.Image, dict]:
    """Downscale then upscale back (thumbnail round-trip). Detail is lost on the
    way down and not recovered on the way up."""
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC), {"op": "resize", "scale": scale}


def noise(img: Image.Image, sigma: float) -> tuple[Image.Image, dict]:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr + np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8)), {"op": "noise", "sigma": sigma}


def jitter(img: Image.Image, strength: float) -> tuple[Image.Image, dict]:
    """Brightness/contrast/saturation, independently perturbed. Signs randomised
    so the model cannot learn "jitter means brighter"."""
    factors = {}
    out = img
    for name, enhancer in (
        ("brightness", ImageEnhance.Brightness),
        ("contrast", ImageEnhance.Contrast),
        ("saturation", ImageEnhance.Color),
    ):
        f = 1.0 + np.random.uniform(-strength, strength)
        out = enhancer(out).enhance(f)
        factors[name] = float(f)
    return out, {"op": "jitter", "strength": strength, **factors}


def crop(img: Image.Image, frac: float) -> tuple[Image.Image, dict]:
    """Centre crop to `frac` of each side, then resize back. Keeps batch shapes
    uniform, and a cropped repost gets platform-resized anyway."""
    w, h = img.size
    nw, nh = int(w * frac), int(h * frac)
    left, top = (w - nw) // 2, (h - nh) // 2
    out = img.crop((left, top, left + nw, top + nh)).resize((w, h), Image.BICUBIC)
    return out, {"op": "crop", "frac": frac}


def moire(img: Image.Image, strength: float) -> tuple[Image.Image, dict]:
    """Screenshot-of-a-screen. OURS, following TeleAI's NTIRE entry.

    Omitted from the official list but a real laundering path, and it ADDS
    structured high-frequency energy where JPEG removes it.
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    # Slight angle and non-integer period are what produce the beat pattern;
    # an axis-aligned integer grid would just alias away invisibly.
    theta = np.random.uniform(0.05, 0.35)
    period = np.random.uniform(2.5, 4.5)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    proj = xx * np.cos(theta) + yy * np.sin(theta)
    pattern = np.sin(2 * np.pi * proj / period)[..., None]
    arr = arr * (1.0 + 0.12 * strength * pattern)
    arr = np.clip(arr, 0, 255)
    return Image.fromarray(arr.astype(np.uint8)), {
        "op": "moire", "strength": strength, "theta": float(theta), "period": float(period),
    }


def cast(img: Image.Image, strength: float) -> tuple[Image.Image, dict]:
    """Global colour cast -- white-balance failure, warm/cool filters. OURS.

    Unlike jitter's saturation change this shifts channel BALANCE, which
    survives compression.
    """
    gains = 1.0 + np.random.uniform(-strength, strength, size=3).astype(np.float32)
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) * gains[None, None, :]
    arr = np.clip(arr, 0, 255)
    return Image.fromarray(arr.astype(np.uint8)), {
        "op": "cast", "strength": strength, "gains": [float(g) for g in gains],
    }


OPS = {
    "jpeg": jpeg, "blur": blur, "resize": resize, "noise": noise,
    "jitter": jitter, "crop": crop, "moire": moire, "cast": cast,
}

# The single scalar each op is parameterised by, used by severity_vector().
OP_PARAM = {
    "jpeg": "quality", "blur": "sigma", "resize": "scale", "noise": "sigma",
    "jitter": "strength", "crop": "frac", "moire": "strength", "cast": "strength",
}


# --------------------------------------------------------------------------
# 2. The battery -- fixed evaluation grid
# --------------------------------------------------------------------------
# The severities published in the brief (§5.2), plus our two additions. This is
# the grid every model and every ablation arm is scored against, so the rows of
# the robustness table are comparable. Changing it invalidates existing results
# in results/ -- treat it as an interface, not a config knob.

BATTERY: dict[str, tuple[str, dict]] = {
    "clean":     ("none",   {}),
    "jpeg_90":   ("jpeg",   {"quality": 90}),
    "jpeg_70":   ("jpeg",   {"quality": 70}),
    "jpeg_50":   ("jpeg",   {"quality": 50}),
    "jpeg_30":   ("jpeg",   {"quality": 30}),
    "blur_0.5":  ("blur",   {"sigma": 0.5}),
    "blur_1.0":  ("blur",   {"sigma": 1.0}),
    "blur_2.0":  ("blur",   {"sigma": 2.0}),
    "resize_0.5":  ("resize", {"scale": 0.5}),
    "resize_0.25": ("resize", {"scale": 0.25}),
    "noise_0.02": ("noise",  {"sigma": 0.02}),
    "noise_0.05": ("noise",  {"sigma": 0.05}),
    "noise_0.10": ("noise",  {"sigma": 0.10}),
    "jitter_0.2": ("jitter", {"strength": 0.2}),
    "crop_0.8":   ("crop",   {"frac": 0.8}),
    # Ours -- not in the brief's table.
    "moire_0.6":  ("moire",  {"strength": 0.6}),
    "cast_0.2":   ("cast",   {"strength": 0.2}),
}

# Composed chains. Nearly every paper tests one transform at one severity; real
# laundering is a sequence. These are the named chains the robustness table
# reports alongside the single-op cells.
CHAIN_BATTERY: dict[str, list[tuple[str, dict]]] = {
    "repost_x2": [("jpeg", {"quality": 70}), ("jpeg", {"quality": 50})],
    "screenshot": [("resize", {"scale": 0.5}), ("moire", {"strength": 0.6}),
                   ("jpeg", {"quality": 70})],
    "crop_repost": [("crop", {"frac": 0.8}), ("jpeg", {"quality": 60})],
    "heavy_launder": [("jpeg", {"quality": 70}), ("crop", {"frac": 0.8}),
                      ("resize", {"scale": 0.5}), ("blur", {"sigma": 0.5}),
                      ("jpeg", {"quality": 40})],
}


# --------------------------------------------------------------------------
# 3. Chains, curriculum, and the severity vector
# --------------------------------------------------------------------------

def apply_chain(
    img: Image.Image, chain: list[tuple[str, dict]]
) -> tuple[Image.Image, list[dict]]:
    """Apply ops in sequence, returning the image and the full parameter log."""
    log = []
    out = img
    for op_name, kwargs in chain:
        if op_name == "none":
            continue
        out, params = OPS[op_name](out, **kwargs)
        log.append(params)
    return out, log


def sample_chain(epoch: int, rng: np.random.Generator, max_epochs: int = 10) -> list:
    """Curriculum-ordered sampling: single ops early, compositions late (MICV,
    NTIRE 2026 1st).

    Starting on 6-op chains buries the signal so thoroughly the model never
    learns the artifact. ~25% stay clean throughout or clean accuracy decays.
    """
    progress = min(1.0, epoch / max(1, max_epochs - 1))
    max_ops = 1 + int(round(progress * 3))       # 1 op at epoch 0 -> 4 by the end

    # ~25% clean throughout, so clean accuracy does not decay.
    n_ops = rng.choice(
        [0] + list(range(1, max_ops + 1)),
        p=[0.25] + [0.75 / max_ops] * max_ops,
    )
    if n_ops == 0:
        return []

    chosen = rng.choice(list(OP_NAMES), size=int(n_ops), replace=False)
    chain = []
    for op_name in chosen:
        lo, hi = SAMPLING_RANGES[op_name]
        chain.append((op_name, {OP_PARAM[op_name]: float(rng.uniform(lo, hi))}))
    return chain


def severity_vector(log: list[dict]) -> np.ndarray:
    """Encode a parameter log as a fixed-width [0,1] vector. 0 = untouched.

    The severity head's regression target, and it is FREE -- we generated the
    transforms. (REAGVIS, CVPRW 2026, hand-computed 5 descriptors for lack of it.)

    All ops share one scale, else the loss is dominated by whichever uses the
    largest raw numbers. Repeats take the max, not the sum: two JPEG passes at
    q=70 leave an image about as damaged as one.
    """
    v = np.zeros(len(OP_NAMES), dtype=np.float32)
    idx = {name: i for i, name in enumerate(OP_NAMES)}

    for params in log:
        op = params["op"]
        i = idx[op]
        if op == "jpeg":       # lower quality = more severe
            s = (95.0 - params["quality"]) / 70.0
        elif op == "blur":
            s = params["sigma"] / 3.0
        elif op == "resize":   # smaller scale = more severe
            s = (1.0 - params["scale"]) / 0.75
        elif op == "noise":
            s = params["sigma"] / 0.15
        elif op == "jitter":
            s = params["strength"] / 0.4
        elif op == "crop":     # smaller fraction = more severe
            s = (1.0 - params["frac"]) / 0.4
        else:                  # moire, cast
            s = params["strength"]
        v[i] = max(v[i], float(np.clip(s, 0.0, 1.0)))

    return v
