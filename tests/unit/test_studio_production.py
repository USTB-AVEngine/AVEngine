"""Production surfaces: catalog freshness, board honesty, verdict lifecycle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from avengine.studio.production import (
    ProductionError,
    board_rows,
    load_sound_asset_catalog,
    read_human_verdict,
    review_queue,
    sound_asset_file,
    write_human_verdict,
)


def _asset_tree(tmp_path: Path) -> Path:
    root = tmp_path / "sound_source_assets_v1"
    asset_dir = root / "audio_playback" / "speaker" / "black"
    asset_dir.mkdir(parents=True)
    (asset_dir / "asset.json").write_text(
        json.dumps(
            {
                "asset_id": "speaker_black_v1",
                "category": "audio_playback",
                "entity_class": "rigid_static_object",
                "admission_state": "research_candidate",
                "identity": {"object_type": "speaker"},
                "geometry": {
                    "width_right_m": 0.2,
                    "depth_forward_m": 0.25,
                    "height_up_m": 0.35,
                    "resting_pose": {
                        "attachment_surface": "floor",
                        "attachment_surface_assumed": True,
                    },
                },
                "acceptance": {
                    "resting_pose_verdict": "level",
                    "base_normal_tilt_deg": 0.4,
                },
                "emitter": {"anchor_id": "a"},
            }
        )
    )
    (asset_dir / "finalized.glb").write_bytes(b"glTF fake")
    index = root / "index.json"
    index.write_text(json.dumps({"created_at": "2026-08-26", "assets": []}))
    return index


def test_catalog_reads_the_tree_not_the_index(tmp_path: Path) -> None:
    """A new asset must appear without an index rebuild - that is 实时."""

    index = _asset_tree(tmp_path)
    catalog = load_sound_asset_catalog(index)
    assert catalog["asset_count"] == 1
    entry = catalog["assets"][0]
    assert entry["asset_id"] == "speaker_black_v1"
    assert entry["resting_pose_verdict"] == "level"
    assert entry["attachment_surface"] == "floor"
    assert entry["attachment_surface_assumed"] is True
    assert entry["has_emitter"] is True
    assert entry["files"]["finalized.glb"]["path"].endswith("finalized.glb")

    second = index.parent / "cat" / "toy" / "red"
    second.mkdir(parents=True)
    (second / "asset.json").write_text(json.dumps({"asset_id": "toy_red"}))
    assert load_sound_asset_catalog(index)["asset_count"] == 2


def test_asset_files_are_sandboxed(tmp_path: Path) -> None:
    index = _asset_tree(tmp_path)
    served = sound_asset_file(index, "audio_playback/speaker/black/finalized.glb")
    assert served.name == "finalized.glb"
    with pytest.raises(ProductionError, match="escapes"):
        sound_asset_file(index, "../outside.glb")
    with pytest.raises(ProductionError, match="not a servable"):
        sound_asset_file(index, "audio_playback/speaker/black/finalized.exe")


def test_verdict_roundtrip_and_rejection(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    assert read_human_verdict(task_dir) is None
    written = write_human_verdict(task_dir, verdict="pass", note="clean")
    assert read_human_verdict(task_dir)["verdict"] == "pass"
    assert written["written_at"].endswith("Z")
    with pytest.raises(ProductionError, match="verdict must be one of"):
        write_human_verdict(task_dir, verdict="maybe")


def _episode_task(tmp_path: Path, name: str, *, with_mp4: bool) -> dict:
    task_dir = tmp_path / name
    output = task_dir / "output" / "render"
    (output / "audio_foa").mkdir(parents=True)
    (output / "receipt.json").write_text(json.dumps({"scene_id": "TEEsavR23oF"}))
    (output / "audio_foa" / "render_report.json").write_text(
        json.dumps(
            {
                "frames_rendered": 75,
                "frames_within_tolerance": 75,
                "direction_error_deg": {"median": 0.0, "p90": 0.0},
                "per_frame": [{"error_deg": 0.0}, {"error_deg": 7.5}],
            }
        )
    )
    if with_mp4:
        (output / "episode_binaural.mp4").write_bytes(b"mp4")
    return {
        "task_id": name,
        "template": "hm3d_episode",
        "status": "pass",
        "created_at": f"2026-08-27T0{name[-1]}:00:00Z",
        "task_dir": str(task_dir),
        "output_dir": str(output),
    }


def test_board_paints_from_verifier_artifacts_not_exit_codes(tmp_path: Path) -> None:
    """A task that exited 0 but left no deliverable must not be green."""

    honest = _episode_task(tmp_path, "task1", with_mp4=True)
    hollow = _episode_task(tmp_path, "task2", with_mp4=False)
    board = board_rows([honest])
    row = board["rows"][0]
    assert row["scene"] == "TEEsavR23oF"
    assert row["cells"]["episode"]["ok"] is True

    board = board_rows([hollow])
    cell = board["rows"][0]["cells"]["episode"]
    assert cell["task_status"] == "pass"
    assert cell["ok"] is False


def test_board_newest_task_wins_and_verdict_column_appears(tmp_path: Path) -> None:
    older = _episode_task(tmp_path, "task1", with_mp4=False)
    newer = _episode_task(tmp_path, "task2", with_mp4=True)
    write_human_verdict(Path(newer["task_dir"]), verdict="pass", note="sounds right")
    board = board_rows([older, newer])
    row = board["rows"][0]
    assert row["cells"]["episode"]["task_id"] == "task2"
    assert row["cells"]["episode"]["ok"] is True
    assert row["cells"]["verdict"]["ok"] is True
    assert "sounds right" in row["cells"]["verdict"]["summary"]


def test_review_queue_lists_only_episodes_with_receipts(tmp_path: Path) -> None:
    episode = _episode_task(tmp_path, "task1", with_mp4=True)
    other = {
        "task_id": "task9",
        "template": "hm3d_route_bank",
        "status": "pass",
        "task_dir": str(tmp_path / "task9"),
        "output_dir": str(tmp_path / "task9" / "output" / "render"),
    }
    queue = review_queue([episode, other])
    assert len(queue["episodes"]) == 1
    entry = queue["episodes"][0]
    assert entry["scene_id"] == "TEEsavR23oF"
    assert entry["artifacts"]["mp4"] == "episode_binaural.mp4"
    assert entry["error_deg_per_frame"] == [0.0, 7.5]
    assert entry["verdict"] is None
