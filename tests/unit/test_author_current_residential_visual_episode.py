from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
TOOL_PATH = REPOSITORY / "tools/m6z/author_current_residential_visual_episode.py"
SPEC = importlib.util.spec_from_file_location(
    "current_residential_visual_author", TOOL_PATH
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def _episode() -> dict[str, object]:
    visual_plan = {
        "schema": "legacy_visual_plan",
        "backend_role": "production_visual",
        "camera": {"horizontal_fov_deg": 90.0},
        "actors": [{"actor_id": "dog0"}, {"actor_id": "human0"}],
        "frames": [{"frame_index": index} for index in range(75)],
        "render": {"frame_count": 75},
        "source_logic": {"audio": "must_not_be_projected"},
        "qualification": {"source_center_gate_status": "pass"},
    }
    return {
        "scene": {"scene_id": "current_kujiale", "map_path": "/Game/Kujiale"},
        "review_lights": [{"light_id": "review0"}],
        "visual_plan": visual_plan,
        "timeline": {
            "audio": {"sample_count": 80_000},
            "audio_events": [{"audio_asset_sha256": "legacy"}],
        },
        "source_manifest": {"owner": "source_manifest"},
        "flags": {"owner": "flags"},
        "room_capsule": {"owner": "room_capsule"},
        "qualification": {"source_center_gate": "pass"},
        "acoustic_proxy": {"owner": "legacy_proxy"},
        "source_activity_by_frame": {"owner": "legacy_audio"},
    }


def test_author_needs_only_scene_profile_output_and_never_touches_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scene = {"scene": "current_kujiale"}
    profile = {"profile": "current_visual"}
    scene_path = tmp_path / "scene.json"
    profile_path = tmp_path / "profile.json"
    scene_path.write_text(json.dumps(scene), encoding="utf-8")
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    built = _episode()
    calls: list[tuple[dict[str, object], dict[str, object]]] = []

    def build_episode(
        *, scene_metadata: dict[str, object], profile: dict[str, object]
    ) -> dict[str, object]:
        calls.append((scene_metadata, profile))
        return built

    monkeypatch.setattr(TOOL, "build_residential_source_episode", build_episode)
    output = tmp_path / "fresh"
    receipt = TOOL.author(
        argparse.Namespace(
            scene_metadata=scene_path,
            profile=profile_path,
            output=output,
        )
    )

    assert calls == [(scene, profile)]
    assert receipt["status"] == "research_only"
    assert receipt["research_only"] is True
    assert receipt["episode_counted"] is False
    assert receipt["formal_dataset_count"] == 0
    assert receipt["qualification"] is False
    assert receipt["qualification_claim"] is False
    assert receipt["clock"] == {
        "frame_count": 75,
        "frame_rate_hz": 15,
        "ticks_per_frame": 3200,
    }
    assert receipt["audio"] == {"status": "not_requested"}
    assert receipt["rlr"] == {"status": "not_requested"}
    assert set(receipt["artifacts"]) == {
        "episode_plan",
        "visual_plan",
    }
    plan = json.loads((output / "episode_plan.json").read_text())
    assert set(plan) == {"status", "scene", "review_lights", "visual_plan"}
    assert plan["status"] == "research_only"
    assert plan["scene"] == built["scene"]
    assert plan["review_lights"] == built["review_lights"]
    assert set(plan["visual_plan"]) == {
        "backend_role",
        "camera",
        "actors",
        "frames",
    }
    assert json.loads((output / "visual_plan.json").read_text()) == plan["visual_plan"]
    assert json.loads((output / "research_receipt.json").read_text()) == receipt
    assert {path.name for path in output.iterdir()} == {
        "episode_plan.json",
        "visual_plan.json",
        "research_receipt.json",
    }
    for forbidden in (
        "timeline",
        "source_manifest",
        "flags",
        "room_capsule",
        "qualification",
        "acoustic_proxy",
        "source_activity_by_frame",
        "audio",
        "rlr",
    ):
        assert forbidden not in plan
    encoded = json.dumps(receipt, sort_keys=True).lower()
    assert all(
        word not in encoded for word in ("schema", "hash", "gate", "audio_claim")
    )
    with pytest.raises(FileExistsError, match="refusing to replace output"):
        TOOL.author(
            argparse.Namespace(
                scene_metadata=scene_path,
                profile=profile_path,
                output=output,
            )
        )


def test_parse_args_has_no_runtime_or_audio_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "author_current_residential_visual_episode.py",
            "--scene-metadata",
            "/tmp/scene.json",
            "--profile",
            "/tmp/profile.json",
            "--output",
            "/tmp/output",
        ],
    )

    parsed = TOOL.parse_args()

    assert vars(parsed) == {
        "scene_metadata": Path("/tmp/scene.json"),
        "profile": Path("/tmp/profile.json"),
        "output": Path("/tmp/output"),
    }
