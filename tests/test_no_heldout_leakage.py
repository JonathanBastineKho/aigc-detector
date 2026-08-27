"""The leakage guard.

TikTok's brief is explicit: "Do not use the following data during training",
naming COCO val2017 (4,998) and DALL-E Advanced (8,843). Those images live
inside WildFake archives alongside permitted ones, so the failure mode is not
malice -- it is downloading coco.zip and forgetting to filter.

This is a compliance check, not a style check. It runs in CI.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.utils import dataset as data  # noqa: E402


def test_assert_no_heldout_rejects_contaminated_frames():
    contaminated = pd.DataFrame({
        "image_id": ["a", "b"],
        "split": [data.SPLIT_TRAIN, data.SPLIT_HELDOUT],
    })
    with pytest.raises(AssertionError, match="held-out"):
        data.assert_no_heldout(contaminated)


def test_assert_no_heldout_passes_clean_frames():
    clean = pd.DataFrame({
        "image_id": ["a", "b"],
        "split": [data.SPLIT_TRAIN, data.SPLIT_VAL],
    })
    data.assert_no_heldout(clean)


@pytest.mark.skipif(
    not (data.data_root() / "sid_set" / "data").exists(),
    reason="SID_Set not downloaded",
)
def test_real_manifest_keeps_training_and_heldout_disjoint():
    df = data.build_manifest()
    trainable = df[df.split.isin({data.SPLIT_TRAIN, data.SPLIT_VAL})]
    data.assert_no_heldout(trainable)

    # Tampered images must never reach a training split -- they are an eval slice.
    assert not (trainable.label == data.SID_TAMPERED).any()

    # And no image_id may appear in both a trainable split and the held-out set.
    heldout_ids = set(df[df.split == data.SPLIT_HELDOUT].image_id)
    assert not (set(trainable.image_id) & heldout_ids)
