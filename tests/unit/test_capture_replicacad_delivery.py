from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from avengine.contracts.json_io import canonical_json_sha256, load_json
from avengine.capture.delivery import M51DeliveryError
from avengine.capture.mp3d_delivery import (
    REPLICACAD_REQUIRED_GATE_IDS,
    validate_room_visual_gate,
)
from tools.review.build_mp3d_delivery import (
    _persistent_claim_boundary,
    _portable_dry_audio_metadata,
    _portable_source_program_audio,
    _runtime_navmesh_record,
)
from tools.review.render_review_acoustics import (
    _portableize_paths,
    _resolve_runtime_root,
    _yaw_orientation_wxyz,
)


REPOSITORY = Path(__file__).resolve().parents[2]


def test_replica_navmesh_locator_resolves_only_inside_declared_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "replica_cad"
    navmesh = root / "navmeshes/apt_0.navmesh"
    navmesh.parent.mkdir(parents=True)
    navmesh.write_bytes(b"navmesh")
    record = {
        "root_id": "AVENGINE_REPLICACAD_ROOT",
        "relative_path": "navmeshes/apt_0.navmesh",
        "byte_size": 7,
        "sha256": "a" * 64,
    }
    runtime = _runtime_navmesh_record(
        record, room_family="replicacad", replicacad_root=root
    )
    assert runtime == {
        "path": str(navmesh.resolve()),
        "byte_size": 7,
        "sha256": "a" * 64,
    }

    escaped = dict(record, relative_path="../elsewhere.navmesh")
    with pytest.raises(Exception, match="escapes"):
        _runtime_navmesh_record(
            escaped, room_family="replicacad", replicacad_root=root
        )


def test_replica_audio_records_replace_private_paths_with_root_locators() -> None:
    source = {
        "schema": "fixture",
        "sources": [
            {
                "source_id": "source0",
                "audio_provenance": {
                    "audio_assets": [
                        {
                            "asset_id": "speech",
                            "uri": "file:///data/datasets/LibriTTS/train/a.wav",
                        }
                    ]
                },
            },
            {
                "source_id": "source1",
                "audio_provenance": {
                    "audio_assets": [
                        {
                            "asset_id": "bark",
                            "uri": "file:///workspace/AVEngine/external/bark.wav",
                        }
                    ]
                },
            },
        ],
    }
    portable = _portable_source_program_audio(source)
    assets = [
        item
        for actor in portable["sources"]
        for item in actor["audio_provenance"]["audio_assets"]
    ]
    assert all("uri" not in item for item in assets)
    assert assets[0]["locator"] == {
        "root_id": "AVENGINE_LIBRITTS_ROOT",
        "relative_path": "train/a.wav",
    }
    assert assets[1]["locator"] == {
        "root_id": "AVENGINE_LEGACY_ROOT",
        "relative_path": "external/bark.wav",
    }
    assert "/data/" not in str(portable)
    assert len(portable["record_content_sha256"]) == 64

    dry = {
        "placement_receipts": [
            {
                "dry_asset": {
                    "asset_id": "speech",
                    "path": "/data/datasets/LibriTTS/train/a.wav",
                }
            }
        ]
    }
    portable_dry = _portable_dry_audio_metadata(dry)
    asset = portable_dry["placement_receipts"][0]["dry_asset"]
    assert "path" not in asset
    assert asset["locator"]["root_id"] == "AVENGINE_LIBRITTS_ROOT"
    declared = portable_dry["assembly_content_sha256"]
    unhashed = dict(portable_dry)
    unhashed.pop("assembly_content_sha256")
    assert declared == canonical_json_sha256(unhashed)


def test_replicacad_gate_is_exact_18_or_route_required_19() -> None:
    route = load_json(
        REPOSITORY / "examples/m5_1/replicacad_articulated_review/route_manifest.json"
    )
    required = set(REPLICACAD_REQUIRED_GATE_IDS)
    required.add("actor_rigid_object_center_clearance")
    gate = {
        "schema": "avengine_m5_1_replicacad_mixed_visual_gate_v1",
        "status": "pass",
        "qualification_claim": False,
        "room_id": route["room_id"],
        "request_id": route["request_id"],
        "route_id": route["route_id"],
        "gate_count": 19,
        "passed_gate_count": 19,
        "gates": {gate_id: True for gate_id in required},
    }
    assert len(validate_room_visual_gate(gate, route, room_family="replicacad")) == 19

    missing = dict(gate)
    missing["gates"] = dict(gate["gates"])
    missing["gates"].pop("actor_rigid_object_center_clearance")
    missing["gate_count"] = missing["passed_gate_count"] = 18
    with pytest.raises(M51DeliveryError, match="route-required"):
        validate_room_visual_gate(missing, route, room_family="replicacad")

    v1_route = dict(route)
    v1_route["placement_gate"] = dict(route["placement_gate"])
    v1_route["placement_gate"]["require_rigid_object_center_clearance"] = False
    v1_route["route_id"] = "replica_v1"
    base = dict(missing)
    base["route_id"] = "replica_v1"
    assert len(validate_room_visual_gate(base, v1_route, room_family="replicacad")) == 18


def test_rir_metadata_paths_become_declared_root_locators(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    repository.mkdir()
    runtime.mkdir()
    portable = _portableize_paths(
        {
            "module": {"path": str(repository / "src/module.py")},
            "binary": {"path": str(runtime / "lib/native.so")},
        },
        (("REPOSITORY", repository), ("RUNTIME", runtime)),
    )
    assert portable["module"]["path"] == "${REPOSITORY}/src/module.py"
    assert portable["binary"]["path"] == "${RUNTIME}/lib/native.so"
    assert str(tmp_path) not in str(portable)


def test_runtime_root_accepts_arbitrary_explicit_or_environment_checkout(
    tmp_path: Path,
) -> None:
    explicit = tmp_path / "external/runtime-explicit"
    environment = tmp_path / "elsewhere/runtime-from-env"
    explicit.mkdir(parents=True)
    environment.mkdir(parents=True)
    assert _resolve_runtime_root(
        explicit,
        environ={"AVENGINE_HABITAT_RUNTIME_ROOT": str(environment)},
    ) == explicit.resolve()
    assert _resolve_runtime_root(
        None,
        environ={"AVENGINE_HABITAT_RUNTIME_ROOT": str(environment)},
    ) == environment.resolve()


def test_standalone_replica_frames_burn_in_claim_boundary() -> None:
    frames = np.zeros((2, 480, 1280, 3), dtype=np.uint8)
    rendered = _persistent_claim_boundary(frames, room_family="replicacad")
    assert rendered.shape == frames.shape
    assert np.count_nonzero(rendered[:, 122:164]) > 0
    assert np.count_nonzero(rendered[:, -42:]) == 0
    assert np.array_equal(rendered[0], rendered[1])


def test_listener_yaw_hash_quaternion_canonicalizes_cardinal_angles() -> None:
    assert _yaw_orientation_wxyz(0.0) == (1.0, 0.0, 0.0, 0.0)
    assert _yaw_orientation_wxyz(180.0) == (0.0, 0.0, 1.0, 0.0)
