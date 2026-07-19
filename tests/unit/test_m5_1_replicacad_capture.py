from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

import avengine.m5_1.replicacad_capture as replicacad_capture_module
from avengine.m1.contracts import load_and_validate_inputs as load_m1_inputs
from avengine.m5_1.replicacad_capture import (
    ReplicaCADCaptureError,
    _assert_selected_closure,
    _bind_capture_identity_and_emitter_anchors,
    _camera_floor_record,
    _path_clearance_record,
    _rigid_object_center_clearance_record,
    capture_replicacad_route,
    derive_replicacad_route_paths,
    load_replicacad_route_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = REPOSITORY_ROOT / "examples/m5_1/replicacad_articulated_review"
ROUTE_PATH = EXAMPLE_ROOT / "route_manifest.json"
ROOM_PATH = EXAMPLE_ROOT / "room_manifest.json"
REQUEST_PATH = EXAMPLE_ROOT / "capture_request.json"


def test_example_route_is_parallel_clearance_review() -> None:
    route = load_replicacad_route_manifest(ROUTE_PATH)
    paths = derive_replicacad_route_paths(route)
    assert paths.human.shape == (270, 3)
    assert paths.beagle.shape == (270, 3)
    assert np.array_equal(paths.human[0], [1.0, 0.1193729192, 5.4])
    assert np.array_equal(paths.human[-1], [1.0, 0.1193729192, 6.6])
    assert np.array_equal(paths.beagle[0], [3.0, 0.1193729192, 5.4])
    assert np.array_equal(paths.beagle[-1], [3.0, 0.1193729192, 6.6])
    separation = np.linalg.norm(paths.human - paths.beagle, axis=1)
    assert np.allclose(separation, 2.0)
    assert route["placement_gate"]["minimum_navmesh_clearance_m"] == 0.25
    assert route["placement_gate"]["minimum_rigid_object_center_clearance_m"] == 0.02
    assert route["semantic_ids"] == {"human0": 62000, "dog0": 62001}


def test_route_loader_rejects_weakened_placement_contract(tmp_path: Path) -> None:
    route = load_replicacad_route_manifest(ROUTE_PATH)
    invalid = copy.deepcopy(route)
    invalid["placement_gate"]["require_camera_actor_line_of_sight"] = False
    path = tmp_path / "invalid.json"
    import json

    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ReplicaCADCaptureError, match="line_of_sight"):
        load_replicacad_route_manifest(path)

    invalid = copy.deepcopy(route)
    invalid["placement_gate"]["minimum_navmesh_clearance_m"] = 0.0
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ReplicaCADCaptureError, match="must be positive"):
        load_replicacad_route_manifest(path)

    invalid = copy.deepcopy(route)
    invalid["placement_gate"]["require_rigid_object_center_clearance"] = False
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ReplicaCADCaptureError, match="rigid_object"):
        load_replicacad_route_manifest(path)


class _PlacementPathFinder:
    def __init__(self, *, clearance: float = 0.8, camera_snap_x: float = 0.0):
        self.clearance = clearance
        self.camera_snap_x = camera_snap_x

    @staticmethod
    def is_navigable(point: np.ndarray, maximum_y_delta: float) -> bool:
        del point, maximum_y_delta
        return True

    def snap_point(self, point: np.ndarray) -> np.ndarray:
        result = np.asarray(point, dtype=np.float64).copy()
        result[0] += self.camera_snap_x
        return result

    def distance_to_closest_obstacle(
        self, point: np.ndarray, maximum_search_radius: float
    ) -> float:
        del point, maximum_search_radius
        return self.clearance


def test_camera_and_actor_clearance_gates_fail_closed() -> None:
    record = _camera_floor_record(
        _PlacementPathFinder(),
        camera_position_m=(2.6, 1.47, 3.4),
        route_floor_y_m=0.119,
        maximum_snap_error_m=0.05,
        minimum_clearance_m=0.25,
    )
    assert record["status"] == "pass"
    assert record["clearance_m"] == pytest.approx(0.8)

    with pytest.raises(ReplicaCADCaptureError, match="camera/listener"):
        _camera_floor_record(
            _PlacementPathFinder(camera_snap_x=0.051),
            camera_position_m=(2.6, 1.47, 3.4),
            route_floor_y_m=0.119,
            maximum_snap_error_m=0.05,
            minimum_clearance_m=0.25,
        )

    paths = derive_replicacad_route_paths(load_replicacad_route_manifest(ROUTE_PATH))
    clearance = _path_clearance_record(
        _PlacementPathFinder(clearance=0.8),
        paths,
        minimum_clearance_m=0.25,
    )
    assert clearance["human0"]["minimum_clearance_m"] == pytest.approx(0.8)
    assert clearance["dog0"]["query_count"] == 270
    assert clearance["dog0"]["failed_frame_indices"] == []
    assert len(clearance["dog0"]["clearance_sequence_sha256"]) == 64
    with pytest.raises(ReplicaCADCaptureError, match="route violates"):
        _path_clearance_record(
            _PlacementPathFinder(clearance=0.24),
            paths,
            minimum_clearance_m=0.25,
        )


class _Bounds:
    def __init__(self, minimum: tuple[float, float, float], maximum: tuple[float, float, float]):
        self.min = np.asarray(minimum, dtype=np.float64)
        self.max = np.asarray(maximum, dtype=np.float64)


class _InverseTransform:
    def __init__(self, center: tuple[float, float, float]):
        self.center = np.asarray(center, dtype=np.float64)

    def transform_point(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float64) - self.center


class _Transform:
    def __init__(self, center: tuple[float, float, float]):
        self.center = center

    def inverted(self) -> _InverseTransform:
        return _InverseTransform(self.center)

    def transform_point(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float64) + np.asarray(
            self.center, dtype=np.float64
        )


class _RigidObject:
    def __init__(self, handle: str, center: tuple[float, float, float]):
        self.handle = handle
        self.collision_shape_aabb = _Bounds(
            (-0.1, -0.1, -0.1), (0.1, 0.1, 0.1)
        )
        self.transformation = _Transform(center)


class _RigidObjectManager:
    def __init__(self, *objects: _RigidObject):
        self.objects = {value.handle: value for value in objects}

    def get_objects_by_handle_substring(self) -> dict[str, _RigidObject]:
        return self.objects


class _MN:
    @staticmethod
    def Vector3(value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float64)


def test_rigid_object_obb_clearance_retains_real_failed_frame_indices() -> None:
    paths = derive_replicacad_route_paths(load_replicacad_route_manifest(ROUTE_PATH))
    clear = _rigid_object_center_clearance_record(
        _RigidObjectManager(_RigidObject("far_table", (10.0, 0.0, 10.0))),
        _MN,
        paths,
        minimum_clearance_m=0.02,
    )
    assert clear["status"] == "pass"
    assert clear["required_minimum_clearance_m"] == pytest.approx(0.02)
    assert clear["actors"]["human0"]["query_count"] == 270
    assert clear["actors"]["human0"]["failed_frame_indices"] == []

    center = (1.0, 0.1193729192, 6.0)
    blocked = _rigid_object_center_clearance_record(
        _RigidObjectManager(_RigidObject("table_03", center)),
        _MN,
        paths,
        minimum_clearance_m=0.02,
    )
    local_human = paths.human - np.asarray(center)
    outside = np.maximum(np.abs(local_human) - 0.1, 0.0)
    expected_failed = np.flatnonzero(
        np.linalg.norm(outside, axis=1) < 0.02
    ).astype(int).tolist()
    assert expected_failed
    assert blocked["status"] == "fail"
    assert blocked["failed_actor_frame_indices"] == {
        "human0": expected_failed
    }
    assert blocked["actors"]["human0"]["failed_frame_indices"] == expected_failed
    assert blocked["actors"]["human0"]["first_failure"]["object_handle"] == "table_03"
    assert blocked["actors"]["dog0"]["status"] == "pass"


def test_capture_identity_and_emitter_anchor_binding_is_explicit() -> None:
    route = load_replicacad_route_manifest(ROUTE_PATH)
    evidence = {
        "anchor_order": [
            "human0.head",
            "human0.mouth_emitter",
            "dog0.mouth_emitter",
        ],
        "array_artifacts": {
            "anchor_positions_m": {
                "shape": [270, 3, 3],
                "readback_verified": True,
            }
        },
        "actors": [
            {"actor_id": "dog0", "emitter_link": "dog mouth"},
            {"actor_id": "human0", "emitter_link": "human mouth"},
        ],
    }
    _bind_capture_identity_and_emitter_anchors(evidence, route)
    actors = {item["actor_id"]: item for item in evidence["actors"]}
    assert actors["human0"]["emitter_anchor_index"] == 1
    assert actors["human0"]["emitter_anchor_id"] == "human0.mouth_emitter"
    assert actors["dog0"]["emitter_anchor_index"] == 2
    assert actors["dog0"]["emitter_anchor_id"] == "dog0.mouth_emitter"
    assert evidence["room_id"] == "replicacad_apt_0"
    assert evidence["request_id"] == route["request_id"]
    assert evidence["route_id"].endswith("_v2")

    evidence["anchor_order"][1] = "dog0.mouth_emitter"
    with pytest.raises(ReplicaCADCaptureError, match="anchor_order"):
        _bind_capture_identity_and_emitter_anchors(evidence, route)


def _write_human_runtime_fixture(tmp_path: Path):
    capture = tmp_path / ".capture.staging-fixture"
    human = capture / "runtime/human"
    human.mkdir(parents=True)
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    source = legacy / "runtime.glb"
    source.write_bytes(b"source")
    filenames = replicacad_capture_module._HUMAN_DERIVED_FILENAMES
    paths = {role: human / filename for role, filename in filenames.items()}
    for role, path in paths.items():
        if role != "rebase_report":
            path.write_bytes(
                b"{}\n" if path.suffix == ".json" else role.encode("ascii")
            )

    def absolute_record(path: Path):
        return {
            "path": str(path.resolve()),
            "byte_size": path.stat().st_size,
            "sha256": replicacad_capture_module.sha256_file(path),
        }

    rebase = {
        "schema": "avengine_m2_skin_root_rebase_local_tr_v2",
        "status": "pass",
        "source": absolute_record(paths["promoted_glb"]),
        "output": absolute_record(paths["visual_glb"]),
    }
    replicacad_capture_module.write_json(paths["rebase_report"], rebase)
    manifest = {
        "schema": "avengine_m5_1_rocketbox_human_runtime_v1",
        "status": "pass",
        "source": absolute_record(source),
        "derived": {
            role: absolute_record(path) for role, path in paths.items()
        },
    }
    manifest["manifest_content_sha256"] = (
        replicacad_capture_module.canonical_json_sha256(manifest)
    )
    manifest_path = human / "human_runtime_manifest.json"
    replicacad_capture_module.write_json(manifest_path, manifest)
    return capture, legacy, manifest_path, paths


def test_generated_human_json_portability_rebinds_hashes_structurally(
    tmp_path: Path,
) -> None:
    capture, legacy, manifest_path, paths = _write_human_runtime_fixture(tmp_path)
    old_manifest_sha = replicacad_capture_module.sha256_file(manifest_path)
    result = (
        replicacad_capture_module._portableize_generated_human_runtime_documents(
            capture,
            (
                ("AVENGINE_CAPTURE_ROOT", capture.resolve()),
                ("AVENGINE_LEGACY_ROOT", legacy.resolve()),
            ),
        )
    )
    manifest = replicacad_capture_module.load_json(manifest_path)
    content = dict(manifest)
    declared = content.pop("manifest_content_sha256")
    assert declared == replicacad_capture_module.canonical_json_sha256(content)
    assert manifest["source"]["root_id"] == "AVENGINE_LEGACY_ROOT"
    assert {
        item["root_id"] for item in manifest["derived"].values()
    } == {"AVENGINE_CAPTURE_ROOT"}
    rebase = replicacad_capture_module.load_json(paths["rebase_report"])
    assert rebase["source"]["root_id"] == "AVENGINE_CAPTURE_ROOT"
    assert rebase["output"]["root_id"] == "AVENGINE_CAPTURE_ROOT"
    assert manifest["derived"]["rebase_report"]["sha256"] == (
        replicacad_capture_module.sha256_file(paths["rebase_report"])
    )
    assert result["old_manifest_file_sha256"] == old_manifest_sha
    assert result["manifest_file"]["sha256"] != old_manifest_sha
    combined = manifest_path.read_text() + paths["rebase_report"].read_text()
    assert str(capture) not in combined
    assert str(legacy) not in combined
    assert ".staging-" not in combined
    replicacad_capture_module._readback_portable_human_runtime(capture)
    replicacad_capture_module._assert_portable_json_bundle(
        capture,
        declared_root_ids=("AVENGINE_CAPTURE_ROOT", "AVENGINE_LEGACY_ROOT"),
    )


def test_human_json_portability_rejects_stale_content_hash(tmp_path: Path) -> None:
    capture, legacy, manifest_path, _paths = _write_human_runtime_fixture(tmp_path)
    manifest = replicacad_capture_module.load_json(manifest_path)
    manifest["status"] = "tampered"
    replicacad_capture_module.write_json(manifest_path, manifest)
    with pytest.raises(ReplicaCADCaptureError, match="content hash differs"):
        replicacad_capture_module._portableize_generated_human_runtime_documents(
            capture,
            (
                ("AVENGINE_CAPTURE_ROOT", capture.resolve()),
                ("AVENGINE_LEGACY_ROOT", legacy.resolve()),
            ),
        )


def test_capture_evidence_rebinds_human_manifest_without_stale_sha() -> None:
    old_sha = "1" * 64
    new_sha = "2" * 64
    evidence = {
        "runtime": {
            "human_package_manifest": {
                "path": "runtime/human/human_runtime_manifest.json",
                "byte_size": 10,
                "sha256": old_sha,
            }
        },
        "heading_alignment": {
            "actors": [
                {
                    "actor_id": "human0",
                    "binding_source": {
                        "kind": "generated_human_runtime_manifest",
                        "path": "${AVENGINE_CAPTURE_ROOT}/runtime/human/human_runtime_manifest.json",
                        "sha256": old_sha,
                    },
                },
                {"actor_id": "dog0", "binding_source": {"kind": "fixture"}},
            ]
        },
    }
    portability = {
        "old_manifest_file_sha256": old_sha,
        "manifest_file": {
            "path": "runtime/human/human_runtime_manifest.json",
            "byte_size": 11,
            "sha256": new_sha,
        },
    }
    replicacad_capture_module._rebind_portable_human_runtime_references(
        evidence, portability
    )
    assert evidence["runtime"]["human_package_manifest"]["sha256"] == new_sha
    assert evidence["heading_alignment"]["actors"][0]["binding_source"][
        "sha256"
    ] == new_sha
    assert old_sha not in str(evidence)

    stale = copy.deepcopy(evidence)
    stale["runtime"]["human_package_manifest"]["sha256"] = old_sha
    stale["heading_alignment"]["actors"][0]["binding_source"]["sha256"] = old_sha
    stale["unexpected_stale_reference"] = old_sha
    with pytest.raises(ReplicaCADCaptureError, match="retains a stale"):
        replicacad_capture_module._rebind_portable_human_runtime_references(
            stale, portability
        )


@pytest.mark.parametrize(
    "payload, match",
    [
        ({"path": "/data/private/source.glb"}, "absolute path"),
        ({"path": "runtime/.capture.staging-leak/file.json"}, "staging path"),
        (
            {"root_id": "UNDECLARED_ROOT", "relative_path": "asset.glb"},
            "undeclared portable root",
        ),
    ],
)
def test_bundle_json_portability_scan_fails_closed(
    tmp_path: Path, payload: dict[str, str], match: str
) -> None:
    replicacad_capture_module.write_json(tmp_path / "payload.json", payload)
    with pytest.raises(ReplicaCADCaptureError, match=match):
        replicacad_capture_module._assert_portable_json_bundle(
            tmp_path, declared_root_ids=("AVENGINE_CAPTURE_ROOT",)
        )


def _fake_staging_result(output_dir: Path):
    output_dir.mkdir(parents=True)
    (output_dir / "marker.txt").write_text("complete", encoding="utf-8")
    (output_dir / replicacad_capture_module.GATE_EVIDENCE_NAME).write_text(
        "{}\n", encoding="utf-8"
    )
    (output_dir / replicacad_capture_module.CONTACT_SHEET_NAME).write_bytes(b"png")
    empty = np.empty((0,), dtype=np.float64)
    capture = replicacad_capture_module.MixedCaptureResult(
        output_dir=output_dir,
        rgb=empty,
        semantic=empty,
        actor_world_matrices=empty,
        skin_root_world_matrices=empty,
        anchor_positions_m=empty,
        semantic_visibility_pixels=empty,
        records=(),
        evidence={},
    )
    return replicacad_capture_module.ReplicaCADCaptureResult(
        capture=capture,
        gate_evidence_path=(
            output_dir / replicacad_capture_module.GATE_EVIDENCE_NAME
        ),
        gate_evidence={},
        contact_sheet_path=(
            output_dir / replicacad_capture_module.CONTACT_SHEET_NAME
        ),
    )


def test_capture_publishes_only_after_staging_readback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "capture"
    events: list[str] = []

    def build(**kwargs):
        events.append("build")
        return _fake_staging_result(Path(kwargs["output_dir"]))

    def readback(result):
        events.append("readback")
        assert result.capture.output_dir.name.startswith(".capture.staging-")
        assert not destination.exists()

    monkeypatch.setattr(
        replicacad_capture_module, "_capture_replicacad_route_in_staging", build
    )
    monkeypatch.setattr(
        replicacad_capture_module, "_readback_replicacad_capture_result", readback
    )
    result = capture_replicacad_route(
        route_manifest_path="unused",
        room_manifest_path="unused",
        m1_request_path="unused",
        human_runtime_glb_path="unused",
        beagle_animal_manifest_path="unused",
        beagle_m2_request_path="unused",
        output_dir=destination,
        replicacad_root="unused",
    )
    assert events == ["build", "readback"]
    assert result.capture.output_dir == destination
    assert result.gate_evidence_path == destination / "replicacad_gate_evidence.json"
    assert (destination / "marker.txt").read_text(encoding="utf-8") == "complete"
    assert list(tmp_path.glob(".capture.staging-*")) == []


def test_capture_late_failure_cleans_staging_and_same_path_can_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "capture"

    def build(**kwargs):
        return _fake_staging_result(Path(kwargs["output_dir"]))

    monkeypatch.setattr(
        replicacad_capture_module, "_capture_replicacad_route_in_staging", build
    )
    monkeypatch.setattr(
        replicacad_capture_module,
        "_readback_replicacad_capture_result",
        lambda _result: (_ for _ in ()).throw(
            ReplicaCADCaptureError("late readback failure")
        ),
    )
    with pytest.raises(ReplicaCADCaptureError, match="late readback"):
        capture_replicacad_route(
            route_manifest_path="unused",
            room_manifest_path="unused",
            m1_request_path="unused",
            human_runtime_glb_path="unused",
            beagle_animal_manifest_path="unused",
            beagle_m2_request_path="unused",
            output_dir=destination,
            replicacad_root="unused",
        )
    assert not destination.exists()
    assert list(tmp_path.glob(".capture.staging-*")) == []

    monkeypatch.setattr(
        replicacad_capture_module,
        "_readback_replicacad_capture_result",
        lambda _result: None,
    )
    retried = capture_replicacad_route(
        route_manifest_path="unused",
        room_manifest_path="unused",
        m1_request_path="unused",
        human_runtime_glb_path="unused",
        beagle_animal_manifest_path="unused",
        beagle_m2_request_path="unused",
        output_dir=destination,
        replicacad_root="unused",
    )
    assert retried.capture.output_dir == destination


def test_capture_rejects_existing_final_before_native_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "capture"
    destination.mkdir()
    called = False

    def build(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("native work must not start")

    monkeypatch.setattr(
        replicacad_capture_module, "_capture_replicacad_route_in_staging", build
    )
    with pytest.raises(ReplicaCADCaptureError, match="refusing to replace"):
        capture_replicacad_route(
            route_manifest_path="unused",
            room_manifest_path="unused",
            m1_request_path="unused",
            human_runtime_glb_path="unused",
            beagle_animal_manifest_path="unused",
            beagle_m2_request_path="unused",
            output_dir=destination,
            replicacad_root="unused",
        )
    assert called is False


def test_selected_closure_uses_environment_root_without_hardcoded_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "replica_cad"
    relative_paths = (
        "replicaCAD.scene_dataset_config.json",
        "configs/scenes/apt_0.scene_instance.json",
        "configs/stages/frl_apartment_stage.stage_config.json",
        "stages/frl_apartment_stage.glb",
        "navmeshes/apt_0.navmesh",
        "configs/lighting/frl_apartment_stage.lighting_config.json",
    )
    for relative in relative_paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}" if path.suffix == ".json" else b"fixture")
    monkeypatch.setenv("AVENGINE_REPLICACAD_ROOT", str(root))
    inputs = load_m1_inputs(ROOM_PATH, REQUEST_PATH)
    closure = _assert_selected_closure(
        room_inputs=inputs,
        runtime=tmp_path / "runtime",
        root=root,
    )
    assert closure["scene_instance"] == (
        root / "configs/scenes/apt_0.scene_instance.json"
    ).resolve()
    assert closure["navmesh"] == (root / "navmeshes/apt_0.navmesh").resolve()
    assert all(str(path).startswith(str(root.resolve())) for path in closure.values())
