"""Tests for the laundering simulator.

The JPEG round-trip test is the important one. A filter that merely *looks* like
compression would pass a visual check and silently invalidate every number in
the robustness table -- so we assert on the physical signature of DCT block
quantisation, not on how blurry the output is.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from aigcd import transforms as T  # noqa: E402


@pytest.fixture
def img():
    """Multi-scale structure, like a photograph.

    Detail matters here: flat colour compresses to nothing, and heavily blurred
    noise has no high frequencies left for `resize` to destroy, so either would
    make a working op look like a no-op. Mild blur over noise plus a low-
    frequency gradient gives content at both ends of the spectrum.
    """
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    base = Image.fromarray(noise).filter(ImageFilter.GaussianBlur(0.8))
    yy, xx = np.mgrid[0:256, 0:256].astype(np.float32)
    ramp = ((xx + yy) / 2.0)[..., None].repeat(3, axis=2)
    blended = 0.6 * np.asarray(base, dtype=np.float32) + 0.4 * ramp
    return Image.fromarray(blended.astype(np.uint8))


def test_jpeg_is_a_real_roundtrip(img):
    """Quantisation must appear on 8-pixel block boundaries and grow as q falls.

    A gaussian-blur approximation produces a flat ratio near 1.0 at every
    quality; genuine JPEG produces a rising one.
    """
    ratios = []
    for q in (90, 70, 50, 30):
        out, params = T.jpeg(img, q)
        assert params == {"op": "jpeg", "quality": q}
        arr = np.asarray(out.convert("L"), dtype=np.float32)
        steps = np.abs(np.diff(arr, axis=1))
        ratios.append(steps[:, 7::8].mean() / steps[:, 3::8].mean())

    assert ratios[0] > 1.0, "no block structure at q=90 -- not a real encode"
    assert ratios == sorted(ratios), f"blocking must worsen as quality falls: {ratios}"


def test_every_op_runs_and_changes_the_image(img):
    a0 = np.asarray(img, dtype=np.float32)
    for name in T.OP_NAMES:
        lo, hi = T.SAMPLING_RANGES[name]
        out, params = T.OPS[name](img, **{T.OP_PARAM[name]: (lo + hi) / 2})
        assert params["op"] == name
        assert out.size == img.size, f"{name} must preserve dimensions"
        assert np.abs(np.asarray(out, dtype=np.float32) - a0).mean() > 0.5, f"{name} is a no-op"


def test_severity_vector_encoding():
    assert T.severity_vector([]).sum() == 0.0

    heavy = T.severity_vector([{"op": "jpeg", "quality": 30}])
    mild = T.severity_vector([{"op": "jpeg", "quality": 90}])
    assert heavy[0] > mild[0], "lower JPEG quality must encode as MORE severe"
    assert 0.0 <= mild[0] <= heavy[0] <= 1.0

    # Repeated ops take the max: two q=70 passes leave an image about as damaged
    # as one, not twice as damaged.
    once = T.severity_vector([{"op": "jpeg", "quality": 70}])
    twice = T.severity_vector([{"op": "jpeg", "quality": 70}] * 2)
    assert np.allclose(once, twice)


def test_severity_vector_is_ordered_by_actual_damage(img):
    """The encoding must track real degradation, not just parameter magnitude."""
    prev_sev, prev_err = -1.0, -1.0
    a0 = np.asarray(img, dtype=np.float32)
    for q in (95, 70, 50, 30):
        out, log = T.apply_chain(img, [("jpeg", {"quality": q})])
        sev = T.severity_vector(log)[0]
        err = np.abs(np.asarray(out, dtype=np.float32) - a0).mean()
        assert sev > prev_sev and err > prev_err
        prev_sev, prev_err = sev, err


def test_curriculum_grows_but_keeps_clean_samples():
    rng = np.random.default_rng(0)
    means = []
    for epoch in (0, 3, 6, 9):
        lengths = [len(T.sample_chain(epoch, rng)) for _ in range(400)]
        means.append(np.mean(lengths))
        clean = sum(n == 0 for n in lengths) / len(lengths)
        assert 0.15 < clean < 0.35, f"epoch {epoch}: {clean:.0%} clean -- clean acc will suffer"

    assert means == sorted(means), f"chains must lengthen over training: {means}"


def test_chains_compose_and_log_every_op(img):
    for name, chain in T.CHAIN_BATTERY.items():
        out, log = T.apply_chain(img, chain)
        assert len(log) == len(chain), f"{name}: params log lost an op"
        assert out.size == img.size
        assert T.severity_vector(log).sum() > 0


def test_battery_covers_the_brief():
    """Every severity published in TikTok's table must have a battery cell."""
    required = [
        "clean", "jpeg_90", "jpeg_70", "jpeg_50", "jpeg_30",
        "blur_0.5", "blur_1.0", "blur_2.0", "resize_0.5", "resize_0.25",
        "noise_0.02", "noise_0.05", "noise_0.10", "jitter_0.2", "crop_0.8",
    ]
    missing = [c for c in required if c not in T.BATTERY]
    assert not missing, f"battery is missing cells from the brief: {missing}"
