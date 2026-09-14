"""Determinism gate for the RGB-D post-training evaluation.

Skipped by default (needs GPU + released checkpoints + a staging root). Run with
``MAMBAGLUE_RGBD_EVAL=1`` and ``MAMBAGLUE_TOMATO_ROOT``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_ROOT_ENV = os.environ.get("MAMBAGLUE_TOMATO_ROOT")
ROOT = Path(_ROOT_ENV).expanduser() if _ROOT_ENV else Path("/nonexistent")
CONF = Path("mambaglue/training/configs/superpoint_rgbd+mambaglue_tomato.yaml")


@pytest.mark.skipif(
    os.environ.get("MAMBAGLUE_RGBD_EVAL") != "1"
    or not (ROOT / "dataset.json").is_file(),
    reason="set MAMBAGLUE_RGBD_EVAL=1 and MAMBAGLUE_TOMATO_ROOT to a staging root",
)
def test_eval_rgbd_is_deterministic(tmp_path):
    from mambaglue.training.eval_rgbd import run

    first = run(CONF, ROOT, checkpoint=None, num_pairs=2, output=tmp_path / "a.json")
    second = run(CONF, ROOT, checkpoint=None, num_pairs=2, output=tmp_path / "b.json")
    first.pop("conf"), second.pop("conf")
    assert first == second
