"""Real-data RGB-D smoke gate for the ``mambaglue-rgbd-smoke`` CLI.

Skipped by default: it downloads released checkpoints and needs a staged scene.
Set ``MAMBAGLUE_RGBD_SMOKE=1`` and ``MAMBAGLUE_RGBD_SCENE`` (a staged scene
directory) to run it, e.g.

    MAMBAGLUE_RGBD_SMOKE=1 MAMBAGLUE_RGBD_SCENE=/path/to/scene \
      uv run pytest tests/test_rgbd_smoke.py -q
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_ROOT_ENV = os.environ.get("MAMBAGLUE_RGBD_SCENE")
SCENE = Path(_ROOT_ENV).expanduser() if _ROOT_ENV else Path("/nonexistent")


@pytest.mark.skipif(
    os.environ.get("MAMBAGLUE_RGBD_SMOKE") != "1" or not SCENE.is_dir(),
    reason="set MAMBAGLUE_RGBD_SMOKE=1 and MAMBAGLUE_RGBD_SCENE to a staged scene",
)
def test_rgbd_smoke_cli_reports_matches(tmp_path):
    from mambaglue.rgbd_smoke import run

    summary = run(
        SCENE,
        0,
        1,
        output=tmp_path / "rgbd_smoke_test.json",
        resize=512,
    )
    assert summary["matches"] > 0
    assert summary["valid_matches"] > 0
    assert (tmp_path / "rgbd_smoke_test.json").is_file()
