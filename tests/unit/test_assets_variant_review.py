from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from avengine.contracts.json_io import canonical_json_sha256, sha256_file
from avengine.rooms.contracts import ValidatedM1Inputs
from avengine.assets.contracts import ValidatedM2Inputs
from avengine.assets.timeline import M2CanaryTrajectory
from avengine.assets import variant_review
from tools.assets import capture_animal_variant_review


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _schedule_request() -> dict[str, Any]:
    actions = ["idle"] * 15 + ["walk"] * 45 + ["idle"] * 15
    return {
        "view_ids": ["view0"],
        "modalities": ["rgb", "depth", "semantic"],
        "states": [
            {"frame_index": index, "action_id": action}
            for index, action in enumerate(actions)
        ],
    }


def test_room_presets_resolve_both_validated_m1_pairs_and_equal_paths() -> None:
    custom_manifest, custom_request, custom_path = variant_review.resolve_room_preset(
        "blender_custom"
    )
    native_manifest, native_request, native_path = variant_review.resolve_room_preset(
        "habitat_mp3d_example"
    )

    assert custom_manifest.name == "room_manifest.json"
    assert custom_request.parent.name == "blender_custom_articulated_review"
    assert custom_request.name == "capture_request.json"
    assert native_manifest.name == "room_manifest.json"
    assert native_request.name == "capture_request.json"
    custom_distance = math.dist(
        custom_path.start_translation_m, custom_path.end_translation_m
    )
    native_distance = math.dist(
        native_path.start_translation_m, native_path.end_translation_m
    )
    assert custom_distance == pytest.approx(0.8712)
    assert native_distance == pytest.approx(0.8712)
    assert native_path.start_translation_m == (-4.55, 0.072447, -3.56)
    assert native_path.end_translation_m == (-3.6788, 0.072447, -3.56)
    assert native_path.rotation_xyzw == (0.0, 0.0, 0.0, 1.0)


def test_unknown_room_preset_has_no_fallback() -> None:
    with pytest.raises(variant_review.VariantReviewError, match="unknown room preset"):
        variant_review.resolve_room_preset("maybe_a_room")


def test_explicit_trajectory_loader_is_exact_and_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.json"
    _write_json(
        path,
        {
            "start_translation_m": [1.0, 0.0, 2.0],
            "end_translation_m": [2.0, 0.0, 2.0],
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    )
    assert variant_review.load_trajectory(path) == M2CanaryTrajectory(
        start_translation_m=(1.0, 0.0, 2.0),
        end_translation_m=(2.0, 0.0, 2.0),
        rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )

    value = json.loads(path.read_text(encoding="utf-8"))
    value["implicit_fallback"] = True
    _write_json(path, value)
    with pytest.raises(
        variant_review.VariantReviewError,
        match="must contain exactly",
    ):
        variant_review.load_trajectory(path)


def test_schedule_names_one_view_and_co_located_modalities() -> None:
    request = _schedule_request()
    assert variant_review.validate_variant_review_schedule(request) == []

    request["view_ids"] = ["view0", "view1"]
    request["states"][59]["action_id"] = "idle"
    errors = variant_review.validate_variant_review_schedule(request)
    assert "review request view_ids must be exactly ['view0']" in errors
    assert "review action schedule must be Idle15/Walk45/Idle15" in errors


@pytest.mark.parametrize(
    ("admission_state", "expected_builder"),
    [
        ("research_candidate", "research"),
        ("canary_qualified", "formal"),
    ],
)
def test_request_builder_dispatches_without_weakening_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    admission_state: str,
    expected_builder: str,
) -> None:
    manifest = tmp_path / "asset_manifest.json"
    _write_json(
        manifest,
        {"asset_id": "animal_v1", "admission_state": admission_state, "files": []},
    )
    room_inputs = ValidatedM1Inputs(
        room_path=tmp_path / "room.json",
        request_path=tmp_path / "room_request.json",
        room={"room_id": "room_v1"},
        request={"seed": 17},
    )
    actions = object()
    contacts = object()
    observed: dict[str, Any] = {}

    monkeypatch.setattr(
        variant_review,
        "validate_animal_asset_package",
        lambda value, *, manifest_path: [],
    )
    monkeypatch.setattr(
        variant_review,
        "_load_baked_actions",
        lambda manifest_path, records: actions,
    )
    monkeypatch.setattr(
        variant_review,
        "_role_path",
        lambda manifest_path, records, role: tmp_path / "contacts.json",
    )
    monkeypatch.setattr(
        variant_review,
        "_load_contact_phases",
        lambda path, *, actions: contacts,
    )

    def builder(kind: str):
        def build(**kwargs: Any) -> dict[str, Any]:
            observed["kind"] = kind
            observed.update(kwargs)
            return _schedule_request()

        return build

    monkeypatch.setattr(
        variant_review,
        "build_m2_research_review_request",
        builder("research"),
    )
    monkeypatch.setattr(
        variant_review,
        "build_m2_capture_request",
        builder("formal"),
    )
    trajectory = M2CanaryTrajectory()

    result = variant_review.build_variant_review_request(
        asset_manifest=manifest,
        room_inputs=room_inputs,
        request_id="variant_review_v1",
        trajectory=trajectory,
    )

    assert result == _schedule_request()
    assert observed["kind"] == expected_builder
    assert observed["actions"] is actions
    assert observed["contact_phases"] is contacts
    assert observed["trajectory"] == trajectory
    assert observed["room_id"] == "room_v1"


@pytest.mark.parametrize(
    ("admission_state", "loader_name"),
    [
        ("research_candidate", "research"),
        ("canary_qualified", "formal"),
    ],
)
def test_review_loader_keeps_existing_admission_specific_authorities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    admission_state: str,
    loader_name: str,
) -> None:
    manifest = tmp_path / "asset.json"
    request = tmp_path / "request.json"
    _write_json(manifest, {"admission_state": admission_state})
    _write_json(request, _schedule_request())
    expected = SimpleNamespace(request=_schedule_request())
    observed: list[str] = []

    def load_research(asset_path: Path, request_path: Path) -> object:
        observed.append("research")
        return expected

    def load_formal(asset_path: Path, request_path: Path) -> object:
        observed.append("formal")
        return expected

    monkeypatch.setattr(variant_review, "load_research_review_inputs", load_research)
    monkeypatch.setattr(variant_review, "load_formal_m2_inputs", load_formal)

    assert variant_review.load_variant_review_inputs(manifest, request) is expected
    assert observed == [loader_name]


def _capture_inputs(tmp_path: Path) -> tuple[ValidatedM2Inputs, ValidatedM1Inputs]:
    asset_path = tmp_path / "asset.json"
    request_path = tmp_path / "request.json"
    room_path = tmp_path / "room.json"
    room_request_path = tmp_path / "room_request.json"
    asset = {
        "asset_id": "cat_variant_01",
        "admission_state": "research_candidate",
    }
    _write_json(asset_path, asset)
    request = _schedule_request()
    request.update(
        {
            "request_id": "cat_review_v1",
            "asset_id": "cat_variant_01",
            "asset_manifest_sha256": sha256_file(asset_path),
            "room_id": "blender_custom_two_zone_v1",
            "camera_rig_id": "camera_rig_0",
            "capture_policy": {
                "state_evaluation": "explicit_fixed_state",
                "free_running_animation": False,
            },
        }
    )
    _write_json(request_path, request)
    room = {"room_id": "blender_custom_two_zone_v1"}
    _write_json(room_path, room)
    room_request = {
        "room_id": "blender_custom_two_zone_v1",
        "primary_camera_rig": {
            "rig_id": "camera_rig_0",
            "view_id": "view0",
            "world_from_rig": {
                "translation_m": [0.0, 1.0, 2.0],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "shared_calibration": {"projection": "pinhole"},
            "modalities": [
                {"modality": "rgb", "sensor_uuid": "rig_rgb"},
                {"modality": "depth", "sensor_uuid": "rig_depth"},
                {"modality": "semantic", "sensor_uuid": "rig_semantic"},
            ],
        },
    }
    _write_json(room_request_path, room_request)
    inputs = ValidatedM2Inputs(
        asset_path=asset_path,
        request_path=request_path,
        asset=asset,
        request=request,
    )
    room_inputs = ValidatedM1Inputs(
        room_path=room_path,
        request_path=room_request_path,
        room=room,
        request=room_request,
    )
    return inputs, room_inputs


def _fake_core_capture(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True)
    (output / "arrays").mkdir()
    (output / "review_media").mkdir()
    array_artifacts: dict[str, Any] = {}
    videos: dict[str, Any] = {}
    for modality in ("rgb", "depth", "semantic"):
        array_path = output / "arrays" / f"{modality}.npy"
        array_path.write_bytes(f"array-{modality}".encode())
        video_path = output / "review_media" / f"view0_{modality}_review.mp4"
        video_path.write_bytes(f"video-{modality}".encode())
        array_artifacts[modality] = {
            "artifact": {
                "path": array_path.relative_to(output).as_posix(),
                "byte_size": array_path.stat().st_size,
                "sha256": sha256_file(array_path),
            }
        }
        videos[modality] = {
            "view_id": "view0",
            "review_only": True,
            "qualification_claim": False,
            "frame_count": 75,
            "frame_rate_hz": 15,
            "artifact": {
                "path": video_path.relative_to(output).as_posix(),
                "byte_size": video_path.stat().st_size,
                "sha256": sha256_file(video_path),
            },
        }
    input_root = output.parent
    request = json.loads((input_root / "request.json").read_text(encoding="utf-8"))
    room_request = json.loads(
        (input_root / "room_request.json").read_text(encoding="utf-8")
    )
    primary_rig = room_request["primary_camera_rig"]
    core: dict[str, Any] = {
        "schema": "avengine_m2_habitat_capture_evidence_v1",
        "status": "review_only",
        "review_only": True,
        "qualification_claim": False,
        "asset_id": "cat_variant_01",
        "asset_admission_state": "research_candidate",
        "request_id": "cat_review_v1",
        "room_id": "blender_custom_two_zone_v1",
        "formal_view_ids": [],
        "review_view_ids": ["view0"],
        "review_modalities": ["rgb", "depth", "semantic"],
        "frames": [
            {
                "frame_index": state["frame_index"],
                "action_id": state["action_id"],
            }
            for state in request["states"]
        ],
        "inputs": {
            "animal_asset_package": {
                "path": str((input_root / "asset.json").resolve()),
                "sha256": sha256_file(input_root / "asset.json"),
            },
            "m2_capture_request": {
                "path": str((input_root / "request.json").resolve()),
                "sha256": sha256_file(input_root / "request.json"),
            },
            "m1_room_manifest": {
                "path": str((input_root / "room.json").resolve()),
                "sha256": sha256_file(input_root / "room.json"),
            },
            "m1_camera_request": {
                "path": str((input_root / "room_request.json").resolve()),
                "sha256": sha256_file(input_root / "room_request.json"),
            },
        },
        "review_media": {"videos": videos},
        "array_artifacts": array_artifacts,
        "runtime_identity": {"runtime_commit_matches_lock": True},
        "sensor_contract": {
            "rig_id": primary_rig["rig_id"],
            "view_id": primary_rig["view_id"],
            "world_from_rig": primary_rig["world_from_rig"],
            "shared_calibration": primary_rig["shared_calibration"],
            "modality_to_sensor_uuid": {
                item["modality"]: item["sensor_uuid"]
                for item in primary_rig["modalities"]
            },
        },
        "runtime_application": {
            "state_evaluation": "explicit_fixed_state",
            "initial_world_time_seconds": 0.0,
            "final_world_time_seconds": 0.0,
        },
    }
    core["evidence_content_sha256"] = canonical_json_sha256(core)
    _write_json(output / "evidence.json", core)
    return core


def _rehash_document(value: dict[str, Any]) -> None:
    value.pop("evidence_content_sha256", None)
    value["evidence_content_sha256"] = canonical_json_sha256(value)


def _emit_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    inputs, room_inputs = _capture_inputs(tmp_path)
    output = tmp_path / "capture"
    monkeypatch.setattr(
        variant_review,
        "_reload_variant_context",
        lambda m2, m1: (m2, m1),
    )
    monkeypatch.setattr(
        variant_review,
        "_capture_m2_states",
        lambda m2, m1, destination, **kwargs: _fake_core_capture(destination),
    )
    variant_review.capture_variant_review(inputs, room_inputs, output)
    return (
        output / variant_review.VARIANT_REVIEW_EVIDENCE_FILENAME,
        output / "evidence.json",
    )


def _rewrite_wrapper(path: Path, value: dict[str, Any]) -> None:
    _rehash_document(value)
    _write_json(path, value)


def _rewrite_core_and_binding(
    wrapper_path: Path,
    core_path: Path,
    core: dict[str, Any],
    *,
    declared_path: str = "evidence.json",
) -> None:
    _rehash_document(core)
    _write_json(core_path, core)
    wrapper = json.loads(wrapper_path.read_text(encoding="utf-8"))
    wrapper["core_capture_evidence"] = {
        "path": declared_path,
        "byte_size": core_path.stat().st_size,
        "sha256": sha256_file(core_path),
        "evidence_content_sha256": core["evidence_content_sha256"],
    }
    _rewrite_wrapper(wrapper_path, wrapper)


def test_capture_is_review_only_hash_bound_and_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, room_inputs = _capture_inputs(tmp_path)
    output = tmp_path / "capture"

    monkeypatch.setattr(
        variant_review,
        "_reload_variant_context",
        lambda m2, m1: (m2, m1),
    )

    def capture(
        m2: ValidatedM2Inputs,
        m1: ValidatedM1Inputs,
        destination: Path,
        *,
        runtime_root: Path,
        review_only: bool,
    ) -> dict[str, Any]:
        assert m2 is inputs
        assert m1 is room_inputs
        assert destination == output.resolve()
        assert runtime_root == tmp_path / "runtime"
        assert review_only is True
        return _fake_core_capture(destination)

    monkeypatch.setattr(variant_review, "_capture_m2_states", capture)
    evidence = variant_review.capture_variant_review(
        inputs,
        room_inputs,
        output,
        runtime_root=tmp_path / "runtime",
    )

    assert evidence["status"] == "pass"
    assert evidence["review_only"] is True
    assert evidence["qualification_claim"] is False
    assert evidence["view_contract"] == {
        "view_ids": ["view0"],
        "camera_rig_id": "camera_rig_0",
        "modalities": ["rgb", "depth", "semantic"],
        "co_located_modalities": True,
        "camera_count": 1,
    }
    assert evidence["timeline"]["segments"] == [
        {"action_id": "idle", "start_frame": 0, "frame_count": 15},
        {"action_id": "walk", "start_frame": 15, "frame_count": 45},
        {"action_id": "idle", "start_frame": 60, "frame_count": 15},
    ]
    declared_hash = evidence.pop("evidence_content_sha256")
    assert declared_hash == canonical_json_sha256(evidence)
    saved_path = output / variant_review.VARIANT_REVIEW_EVIDENCE_FILENAME
    assert saved_path.is_file()
    assert variant_review.verify_variant_review_evidence(saved_path) == []

    rgb_video = output / "review_media/view0_rgb_review.mp4"
    rgb_video.write_bytes(b"tampered")
    assert "review_videos.rgb artifact bytes changed" in (
        variant_review.verify_variant_review_evidence(saved_path)
    )

    with pytest.raises(variant_review.VariantReviewError, match="refusing to replace"):
        variant_review.capture_variant_review(inputs, room_inputs, output)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("missing_record", "core_capture_evidence record is missing"),
        ("deleted", "core_capture_evidence artifact is missing"),
        ("swapped_path", "core_capture_evidence path is not canonical"),
        ("rewritten_core", "request_ids differ"),
    ],
)
def test_core_evidence_missing_deleted_swapped_or_rewritten_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected: str,
) -> None:
    wrapper_path, core_path = _emit_review(tmp_path, monkeypatch)
    wrapper = json.loads(wrapper_path.read_text(encoding="utf-8"))
    if mutation == "missing_record":
        wrapper.pop("core_capture_evidence")
        _rewrite_wrapper(wrapper_path, wrapper)
    elif mutation == "deleted":
        core_path.unlink()
    elif mutation == "swapped_path":
        swapped = core_path.with_name("other_core.json")
        core_path.rename(swapped)
        core = json.loads(swapped.read_text(encoding="utf-8"))
        _rewrite_core_and_binding(
            wrapper_path,
            swapped,
            core,
            declared_path="other_core.json",
        )
    else:
        core = json.loads(core_path.read_text(encoding="utf-8"))
        core["request_id"] = "detached_request"
        _rewrite_core_and_binding(wrapper_path, core_path, core)

    errors = variant_review.verify_variant_review_evidence(wrapper_path)
    assert any(expected in error for error in errors), errors


@pytest.mark.parametrize(
    ("filename", "owner"),
    [
        ("asset.json", "animal_asset_package"),
        ("request.json", "capture_request"),
        ("room.json", "room_manifest"),
        ("room_request.json", "room_request"),
    ],
)
def test_every_declared_input_is_rehashed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    owner: str,
) -> None:
    wrapper_path, _core_path = _emit_review(tmp_path, monkeypatch)
    input_path = tmp_path / filename
    value = json.loads(input_path.read_text(encoding="utf-8"))
    value["tampered_after_capture"] = True
    _write_json(input_path, value)

    errors = variant_review.verify_variant_review_evidence(wrapper_path)
    assert f"{owner} artifact bytes changed" in errors


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("asset_id", "asset identities differ"),
        ("timeline", "wrapper timeline contract differs"),
        ("runtime_identity", "runtime identity differs"),
        ("world_time", "world time differs"),
        ("view_contract", "wrapper view contract differs"),
        ("rgb_alias", "rgb_review_video differs"),
        ("review_video", "rgb review video differs"),
        ("array", "rgb array artifact differs"),
    ],
)
def test_rehashed_wrapper_redundancy_cannot_detach_from_core(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected: str,
) -> None:
    wrapper_path, _core_path = _emit_review(tmp_path, monkeypatch)
    wrapper = json.loads(wrapper_path.read_text(encoding="utf-8"))
    if mutation == "asset_id":
        wrapper["asset_id"] = "detached_asset"
    elif mutation == "timeline":
        wrapper["timeline"]["frame_count"] = 74
    elif mutation == "runtime_identity":
        wrapper["runtime_identity"] = {"detached": True}
    elif mutation == "world_time":
        wrapper["world_time_seconds"] = [1.0, 2.0]
    elif mutation == "view_contract":
        wrapper["view_contract"]["camera_count"] = 2
    elif mutation == "rgb_alias":
        wrapper["rgb_review_video"] = dict(wrapper["review_videos"]["depth"])
    elif mutation == "review_video":
        wrapper["review_videos"]["rgb"] = dict(wrapper["review_videos"]["depth"])
    else:
        wrapper["array_artifacts"]["rgb"] = dict(wrapper["array_artifacts"]["depth"])
    _rewrite_wrapper(wrapper_path, wrapper)

    errors = variant_review.verify_variant_review_evidence(wrapper_path)
    assert any(expected in error for error in errors), errors


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("schema", "invalid review claim"),
        ("frame", "core/request frame 15 action_id differs"),
        ("runtime_identity", "runtime identity differs"),
        ("runtime_application", "world time differs"),
        ("sensor", "wrapper/core view contract differs"),
        ("review_video", "rgb review video differs"),
        ("input", "wrapper/core capture_request input records differ"),
    ],
)
def test_fully_rehashed_core_redundancy_cannot_detach_from_wrapper_or_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected: str,
) -> None:
    wrapper_path, core_path = _emit_review(tmp_path, monkeypatch)
    core = json.loads(core_path.read_text(encoding="utf-8"))
    if mutation == "schema":
        core["schema"] = "detached_core_schema"
    elif mutation == "frame":
        core["frames"][15]["action_id"] = "idle"
    elif mutation == "runtime_identity":
        core["runtime_identity"] = {"detached": True}
    elif mutation == "runtime_application":
        core["runtime_application"]["final_world_time_seconds"] = 1.0
    elif mutation == "sensor":
        core["sensor_contract"]["view_id"] = "view1"
    elif mutation == "review_video":
        core["review_media"]["videos"]["rgb"]["artifact"] = dict(
            core["review_media"]["videos"]["depth"]["artifact"]
        )
    else:
        core["inputs"]["m2_capture_request"] = dict(
            core["inputs"]["animal_asset_package"]
        )
    _rewrite_core_and_binding(wrapper_path, core_path, core)

    errors = variant_review.verify_variant_review_evidence(wrapper_path)
    assert any(expected in error for error in errors), errors


def test_core_content_digest_must_match_wrapper_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper_path, _core_path = _emit_review(tmp_path, monkeypatch)
    wrapper = json.loads(wrapper_path.read_text(encoding="utf-8"))
    wrapper["core_capture_evidence"]["evidence_content_sha256"] = "0" * 64
    _rewrite_wrapper(wrapper_path, wrapper)

    assert "wrapper/core evidence content hashes differ" in (
        variant_review.verify_variant_review_evidence(wrapper_path)
    )


def test_cli_refuses_existing_capture_before_loading_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "capture"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep\n", encoding="utf-8")

    def unexpected_room(_preset: str) -> object:
        raise AssertionError("input loading must not start")

    monkeypatch.setattr(
        capture_animal_variant_review,
        "resolve_room_preset",
        unexpected_room,
    )
    with pytest.raises(
        variant_review.VariantReviewError,
        match="refusing to replace capture output",
    ):
        capture_animal_variant_review.main(
            [
                "--asset-manifest",
                str(tmp_path / "asset.json"),
                "--room-preset",
                "blender_custom",
                "--request-id",
                "review_v1",
                "--request-output",
                str(tmp_path / "request.json"),
                "--runtime-root",
                str(tmp_path / "runtime"),
                "--output",
                str(output),
            ]
        )
    assert marker.read_text(encoding="utf-8") == "keep\n"


def test_cli_help_routes_animated_glb_to_existing_baker() -> None:
    help_text = capture_animal_variant_review._parser().format_help()
    assert "tools/assets/bake_actions.py" in help_text
    assert "never plays a GLB animation clock" in help_text
