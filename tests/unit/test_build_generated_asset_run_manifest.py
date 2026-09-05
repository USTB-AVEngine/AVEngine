"""Generated-asset run index must record action clips that were actually written."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "assets"))

import build_generated_asset_run_manifest as INDEX  # noqa: E402


def test_visual_review_records_action_dir_only_when_pngs_exist(tmp_path: Path) -> None:
    review = tmp_path / "review" / "siamese"
    review.mkdir(parents=True)
    (review / "front.png").write_bytes(b"png")
    (review / "action").mkdir()
    artifacts = INDEX.visual_review_artifacts(review)
    assert artifacts["front"].endswith("front.png")
    assert artifacts["action_dir"] is None

    (review / "action" / "000.png").write_bytes(b"png")
    artifacts = INDEX.visual_review_artifacts(review)
    assert artifacts["action_dir"] == str(review / "action")


def test_visual_review_prefers_manifest_action_dir(tmp_path: Path) -> None:
    review = tmp_path / "review" / "siamese"
    action = review / "action"
    action.mkdir(parents=True)
    (review / "front.png").write_bytes(b"png")
    (action / "000.png").write_bytes(b"png")
    manifest = {
        "schema": "avengine_generated_asset_visual_review_v1",
        "front": str(review / "front.png"),
        "action_dir": str(action),
        "turntable_dir": None,
    }
    (review / "review_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    artifacts = INDEX.visual_review_artifacts(review)
    assert artifacts["action_dir"] == str(action)
