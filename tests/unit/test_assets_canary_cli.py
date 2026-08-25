from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from avengine.contracts.json_io import sha256_file
from tools.assets import build_canary_request, capture_canary


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _contact_report(path: Path) -> None:
    contact_order = [
        "paw_front_left",
        "paw_front_right",
        "paw_hind_left",
        "paw_hind_right",
    ]
    _write_json(
        path,
        {
            "contact_order": contact_order,
            "actions": [
                {
                    "semantic_action_id": action_id,
                    "frames": [
                        {
                            "contacts": [
                                {
                                    "contact_id": contact_id,
                                    "in_contact": action_id == "idle",
                                }
                                for contact_id in contact_order
                            ]
                        }
                    ],
                }
                for action_id in ("idle", "walk")
            ],
        },
    )


def _asset_fixture(tmp_path: Path) -> tuple[Path, dict[str, Any], Path]:
    visual = tmp_path / "visual.glb"
    actions = tmp_path / "actions.npz"
    contacts = tmp_path / "contacts.json"
    visual.write_bytes(b"visual")
    actions.write_bytes(b"actions")
    _contact_report(contacts)
    asset = {
        "asset_id": "dog_canary_v1",
        "admission_state": "canary_qualified",
        "files": [
            {
                "role": "visual",
                "path": visual.name,
                "sha256": sha256_file(visual),
            },
            {
                "role": "idle_poses",
                "path": actions.name,
                "sha256": sha256_file(actions),
            },
            {
                "role": "walk_poses",
                "path": actions.name,
                "sha256": sha256_file(actions),
            },
            {
                "role": "contact_phases",
                "path": contacts.name,
                "sha256": sha256_file(contacts),
            },
        ],
    }
    manifest = tmp_path / "asset_manifest.json"
    _write_json(manifest, asset)
    return manifest, asset, actions


def _world_contact_audit(
    path: Path,
    *,
    asset: dict[str, Any],
) -> None:
    records = {record["role"]: record for record in asset["files"]}
    _write_json(
        path,
        {
            "schema": "avengine_m2_world_contact_audit_v1",
            "status": "pass",
            "qualification_claim": False,
            "source_glb_sha256": records["visual"]["sha256"],
            "baked_actions_sha256": records["walk_poses"]["sha256"],
            "contact_phases_sha256": records["contact_phases"]["sha256"],
            "trajectory": {
                "start_translation_m": [-0.2, 0.02, 0.7],
                "end_translation_m": [-0.2, 0.02, -0.7],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                "walk_frame_count": 45,
                "sample_rate_hz": 15,
            },
            "gate": {"passed": True},
        },
    )


def test_world_contact_trajectory_requires_package_hash_bindings(
    tmp_path: Path,
) -> None:
    _, asset, _ = _asset_fixture(tmp_path)
    records = {record["role"]: record for record in asset["files"]}
    audit = tmp_path / "world_contact.json"
    _world_contact_audit(audit, asset=asset)

    trajectory = build_canary_request._trajectory_from_audit(
        audit,
        records=records,
    )

    assert trajectory.start_translation_m == (-0.2, 0.02, 0.7)
    assert trajectory.end_translation_m == (-0.2, 0.02, -0.7)
    value = json.loads(audit.read_text(encoding="utf-8"))
    value["contact_phases_sha256"] = "f" * 64
    _write_json(audit, value)
    with pytest.raises(
        build_canary_request.CanaryRequestCliError,
        match="bind package visual/actions/contacts",
    ):
        build_canary_request._trajectory_from_audit(audit, records=records)


def test_build_canary_request_writes_once_and_reloads_formal_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest, asset, actions_path = _asset_fixture(tmp_path)
    audit = tmp_path / "world_contact.json"
    _world_contact_audit(audit, asset=asset)
    room_manifest = tmp_path / "room.json"
    room_request = tmp_path / "room_request.json"
    output = tmp_path / "requests" / "canary.json"
    actions = object()
    observed: dict[str, Any] = {}

    monkeypatch.setattr(
        build_canary_request,
        "validate_animal_asset_package",
        lambda value, *, manifest_path: [],
    )
    monkeypatch.setattr(
        build_canary_request,
        "load_m1_inputs",
        lambda room, request: SimpleNamespace(
            room={"room_id": "custom_room_v1"},
            request={"seed": 17},
        ),
    )

    def read_actions(path: Path) -> object:
        assert path == actions_path.resolve()
        return actions

    monkeypatch.setattr(build_canary_request, "read_baked_actions_npz", read_actions)

    request_value = {
        "request_id": "formal_request_v1",
        "states": [{"frame_index": index} for index in range(75)],
        "view_ids": ["view0"],
        "modalities": ["rgb", "depth", "semantic"],
    }

    def build_request(**kwargs: Any) -> dict[str, Any]:
        observed.update(kwargs)
        return request_value

    monkeypatch.setattr(
        build_canary_request,
        "build_m2_capture_request",
        build_request,
    )

    def reload_inputs(asset_path: Path, request_path: Path) -> SimpleNamespace:
        assert asset_path == manifest.resolve()
        assert request_path == output.resolve()
        assert json.loads(request_path.read_text(encoding="utf-8")) == request_value
        return SimpleNamespace(asset=asset, request=request_value)

    monkeypatch.setattr(build_canary_request, "load_m2_inputs", reload_inputs)

    result = build_canary_request.main(
        [
            "--asset-manifest",
            str(manifest),
            "--room-manifest",
            str(room_manifest),
            "--room-request",
            str(room_request),
            "--world-contact-audit",
            str(audit),
            "--request-id",
            "formal_request_v1",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert observed["asset"] == asset
    assert observed["actions"] is actions
    assert observed["room_id"] == "custom_room_v1"
    assert observed["seed"] == 17
    assert observed["trajectory"].end_translation_m == (-0.2, 0.02, -0.7)
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "pass"
    assert summary["review_only"] is False
    assert summary["admission_state"] == "canary_qualified"
    assert summary["state_count"] == 75
    assert summary["request_sha256"] == sha256_file(output)


def test_build_canary_request_refuses_existing_output_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "canary.json"
    output.write_text("do not replace\n", encoding="utf-8")

    def unexpected_load(_path: Path) -> dict[str, Any]:
        raise AssertionError("input loading must not begin when output exists")

    monkeypatch.setattr(build_canary_request, "load_json", unexpected_load)
    with pytest.raises(
        build_canary_request.CanaryRequestCliError,
        match="refusing to replace request output",
    ):
        build_canary_request.main(
            [
                "--asset-manifest",
                str(tmp_path / "asset.json"),
                "--room-manifest",
                str(tmp_path / "room.json"),
                "--room-request",
                str(tmp_path / "room_request.json"),
                "--output",
                str(output),
            ]
        )
    assert output.read_text(encoding="utf-8") == "do not replace\n"


def test_capture_canary_uses_formal_loaders_and_prints_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    asset_path = tmp_path / "asset.json"
    request_path = tmp_path / "request.json"
    room_path = tmp_path / "room.json"
    room_request_path = tmp_path / "room_request.json"
    runtime_root = tmp_path / "runtime"
    output = tmp_path / "capture"
    inputs = object()
    room_inputs = object()

    def load_m2(asset: Path, request: Path) -> object:
        assert asset == asset_path
        assert request == request_path
        return inputs

    def load_m1(room: Path, request: Path) -> object:
        assert room == room_path
        assert request == room_request_path
        return room_inputs

    monkeypatch.setattr(capture_canary, "load_m2_inputs", load_m2)
    monkeypatch.setattr(capture_canary, "load_m1_inputs", load_m1)

    artifacts = {
        modality: {
            "artifact": {
                "path": f"arrays/{modality}.npy",
                "byte_size": index + 1,
                "sha256": str(index) * 64,
            }
        }
        for index, modality in enumerate(("rgb", "depth", "semantic"), start=1)
    }

    def capture(
        m2: object,
        m1: object,
        destination: Path,
        *,
        runtime_root: Path,
    ) -> dict[str, Any]:
        assert m2 is inputs
        assert m1 is room_inputs
        assert destination == output.resolve()
        assert runtime_root == tmp_path / "runtime"
        return {
            "status": "pass",
            "evidence_kind": "completed_formal_habitat_capture",
            "review_only": False,
            "asset_admission_state": "canary_qualified",
            "formal_view_ids": ["view0"],
            "review_view_ids": [],
            "formal_modalities": ["rgb", "depth", "semantic"],
            "frames": [{} for _ in range(75)],
            "runtime_application": {
                "initial_world_time_seconds": 0.0,
                "final_world_time_seconds": 0.0,
            },
            "array_artifacts": artifacts,
            "evidence_content_sha256": "e" * 64,
        }

    monkeypatch.setattr(capture_canary, "capture_m2_habitat", capture)

    result = capture_canary.main(
        [
            "--asset-manifest",
            str(asset_path),
            "--request",
            str(request_path),
            "--room-manifest",
            str(room_path),
            "--room-request",
            str(room_request_path),
            "--runtime-root",
            str(runtime_root),
            "--output",
            str(output),
        ]
    )

    assert result == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "pass"
    assert summary["review_only"] is False
    assert summary["asset_admission_state"] == "canary_qualified"
    assert summary["frame_count"] == 75
    assert summary["formal_view_ids"] == ["view0"]
    assert summary["review_view_ids"] == []
    assert summary["evidence"] == str(output.resolve() / "evidence.json")
    assert set(summary["array_artifacts"]) == {"rgb", "depth", "semantic"}
