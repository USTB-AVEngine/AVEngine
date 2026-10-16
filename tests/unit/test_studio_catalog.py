from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from avengine.studio.catalog import (
    annotate_room_registry,
    extract_commit_candidates,
    list_review_captures,
    main_branch_commit_shas,
    resolve_commit,
)
from avengine.studio.config import RoomPolicy


def _write_capture_dir(review_root: Path, name: str, receipt: dict) -> Path:
    capture_dir = review_root / name
    capture_dir.mkdir(parents=True)
    (capture_dir / "frame_records.json").write_text(
        json.dumps({"frame_records": []}), encoding="utf-8"
    )
    (capture_dir / "research_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    return capture_dir


def test_commit_candidates_skip_pure_digit_dates() -> None:
    assert extract_commit_candidates("capture_20260821T1400Z") == []
    assert extract_commit_candidates("visual_skeletal_2786897_20260821T1400Z") == [
        "2786897"
    ]
    assert extract_commit_candidates("natural_parallel_1fd3f5d_v1") == ["1fd3f5d"]


def test_resolve_commit_against_main_shas() -> None:
    main_shas = frozenset({"1fd3f5d" + "0" * 33, "2786897" + "f" * 33})
    assert resolve_commit("visual_skeletal_2786897_x", main_shas) == ("2786897", True)
    assert resolve_commit("visual_deadbee1_x", main_shas) == ("deadbee1", False)
    assert resolve_commit("capture_20260821T1400Z", main_shas) == (None, None)
    assert resolve_commit("visual_deadbee1_x", None) == ("deadbee1", None)


def test_room_registry_annotation_applies_owner_policy() -> None:
    registry = {
        "records": [
            {"room_id": "blender_custom_two_zone_v1", "provider_id": "blender_custom"},
            {"room_id": "skokloster_castle_room", "provider_id": "habitat_test_scene"},
            {"room_id": "habitat_mp3d_example_17DRP5sb8fy", "provider_id": "mp3d"},
        ]
    }
    policy = RoomPolicy(
        banned_provider_ids=frozenset({"blender_custom"}),
        excluded_room_id_substrings=frozenset({"skokloster"}),
    )
    annotated = annotate_room_registry(registry, policy)
    statuses = [record["studio_status"] for record in annotated["records"]]
    assert statuses == ["banned", "excluded", "available"]
    # the input registry is not mutated
    assert "studio_status" not in registry["records"][0]


def test_list_review_captures_scans_and_filters(tmp_path: Path) -> None:
    review_root = tmp_path / "review"
    _write_capture_dir(
        review_root,
        "room_visual_2786897_20260101T0000Z",
        {
            "schema": "capture_v1",
            "status": "research_only",
            "research_only": True,
            "capture": {"frame_count": 75, "modalities": ["rgb"]},
            "selected_room": {"room_id": "habitat_mp3d_example_17DRP5sb8fy"},
        },
    )
    _write_capture_dir(review_root, "room_visual_1fd3f5d_defect", {"status": "x"})
    # not a capture: no receipt
    (review_root / "not_a_capture").mkdir()
    (review_root / "not_a_capture" / "frame_records.json").write_text(
        "{}", encoding="utf-8"
    )

    main_shas = frozenset({"2786897" + "a" * 33})
    entries = list_review_captures(review_root, main_shas)
    assert [entry["name"] for entry in entries] == [
        "room_visual_1fd3f5d_defect",
        "room_visual_2786897_20260101T0000Z",
    ]
    by_name = {entry["name"]: entry for entry in entries}
    trusted = by_name["room_visual_2786897_20260101T0000Z"]
    assert trusted["trusted"] is True
    assert trusted["commit"] == "2786897"
    assert trusted["frame_count"] == 75
    assert trusted["room_id"] == "habitat_mp3d_example_17DRP5sb8fy"
    defect = by_name["room_visual_1fd3f5d_defect"]
    assert defect["commit_on_main"] is False
    assert defect["trusted"] is False


def test_main_branch_commit_shas_from_real_git_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    env_args = [
        "-c",
        "user.name=studio-test",
        "-c",
        "user.email=studio-test@example.invalid",
    ]
    subprocess.run(
        ["git", "-C", str(repo), "init", "-q", "-b", "main"], check=True
    )
    subprocess.run(
        ["git", "-C", str(repo), *env_args, "commit", "-q", "--allow-empty", "-m", "x"],
        check=True,
    )
    shas = main_branch_commit_shas(repo, branch="main")
    assert shas is not None and len(shas) == 1
    (sha,) = shas
    assert len(sha) == 40

    assert main_branch_commit_shas(tmp_path / "not_a_repo") is None


def test_main_branch_commit_shas_missing_branch_returns_none(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q", "-b", "main"], check=True)
    assert main_branch_commit_shas(repo, branch="main") is None


@pytest.mark.parametrize(
    "name,expected",
    [
        ("m7_habitat_mp3d_batch_installed_3404840_x", ["3404840"]),
        ("magnum_cp312_45811bb", ["45811bb"]),
    ],
)
def test_commit_candidates_examples(name: str, expected: list[str]) -> None:
    assert extract_commit_candidates(name) == expected
