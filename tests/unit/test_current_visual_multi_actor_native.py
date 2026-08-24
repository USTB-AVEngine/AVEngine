"""CPU mock tests for the native two-actor visual boundary."""

import json

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from avengine import cli
from avengine.contracts.json_io import write_json
from avengine.m5.current_visual import _object_id_pixel_record
from avengine.m5.current_visual_multi_actor_native import (
    ARRAY_ARTIFACTS,
    NativeMultiActorVisualError,
    _camera_pose,
    _capture_m1_request_path,
    _fresh_output_directory,
    _load_actor_templates,
    _require_camera_and_sensor_pose_match,
    _require_no_case_semantic_collision,
    _runtime_contact_api,
    _save_array,
    compare_capture_outputs,
)
from tools.m5 import capture_mp3d_region_case as capture_tool


def _pose(translation: list[float] | None = None) -> dict[str, list[float]]:
    return {
        "translation_m": translation or [0.0, 0.0, 0.0],
        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }


def test_camera_pose_calls_quaternion_factory_on_module() -> None:
    calls: list[tuple[float, float, float, float]] = []

    class QuaternionModule:
        @staticmethod
        def quaternion(w: float, x: float, y: float, z: float) -> str:
            calls.append((w, x, y, z))
            return "rotation"

    room = SimpleNamespace(
        request={"primary_camera_rig": {"world_from_rig": _pose([1.0, 2.0, 3.0])}}
    )
    position, rotation = _camera_pose(
        room, SimpleNamespace(quaternion=QuaternionModule)
    )
    assert np.array_equal(position, np.asarray([1.0, 2.0, 3.0]))
    assert rotation == "rotation"
    assert calls == [(1.0, 0.0, 0.0, 0.0)]


def test_camera_and_every_sensor_pose_must_match_m1_rig() -> None:
    rig = _pose()
    _require_camera_and_sensor_pose_match(
        rig,
        {"agent": _pose(), "sensors": {"rgb": _pose(), "object_id": _pose()}},
        frame_index=0,
    )
    with pytest.raises(NativeMultiActorVisualError, match="camera/sensor"):
        _require_camera_and_sensor_pose_match(
            rig,
            {
                "agent": _pose(),
                "sensors": {"rgb": _pose(), "object_id": _pose([0.01, 0.0, 0.0])},
            },
            frame_index=3,
        )


def test_semantic_preflight_uses_case_actor_ids() -> None:
    semantic = np.asarray([[0, 310], [0, 0]], dtype=np.int32)
    _require_no_case_semantic_collision(semantic, [210, 211])
    with pytest.raises(NativeMultiActorVisualError, match="310"):
        _require_no_case_semantic_collision(semantic, [310, 311])


def test_capture_camera_can_be_rebound_without_mutating_case(tmp_path: Path) -> None:
    declared = tmp_path / "declared.json"
    declared.write_text("{}")
    alternate = tmp_path / "alternate.json"
    alternate.write_text("{}")
    case = SimpleNamespace(camera_m1_request_path=declared.resolve())
    assert _capture_m1_request_path(case, None) == declared.resolve()
    assert _capture_m1_request_path(case, alternate) == alternate.resolve()
    link = tmp_path / "alternate-link.json"
    link.symlink_to(alternate)
    with pytest.raises(NativeMultiActorVisualError, match="symlink"):
        _capture_m1_request_path(case, link)


def test_runtime_contact_api_requires_both_native_methods() -> None:
    class Runtime:
        def perform_discrete_collision_detection(self) -> None:
            pass

        def get_physics_contact_points(self) -> list[object]:
            return []

    detector, reader = _runtime_contact_api(Runtime())
    assert callable(detector)
    assert callable(reader)
    with pytest.raises(NativeMultiActorVisualError, match="lacks"):
        _runtime_contact_api(SimpleNamespace())


def test_gpu_guard_checks_only_the_explicit_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(stdout="123\n456\n")

    monkeypatch.setattr(capture_tool.subprocess, "run", fake_run)
    assert capture_tool._gpu_compute_pids(2) == [123, 456]
    assert calls == [
        [
            "nvidia-smi",
            "--id=2",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ]
    ]
    with pytest.raises(ValueError, match="nonnegative"):
        capture_tool._gpu_compute_pids(-1)


def test_each_track_loads_its_own_template_path() -> None:
    calls: list[str] = []

    class Manager:
        def load_configs(self, path: str) -> list[int]:
            calls.append(path)
            return [1]

        def get_template_handles(self, prefix: str) -> list[str]:
            return [f"handle:{prefix}"]

    simulator = SimpleNamespace(
        metadata_mediator=SimpleNamespace(ao_template_manager=Manager())
    )
    tracks = [
        SimpleNamespace(
            actor_id="actor0",
            bundle=SimpleNamespace(
                paths_by_role={"habitat_ao_config": Path("dog-a.ao_config")}
            ),
        ),
        SimpleNamespace(
            actor_id="actor1",
            bundle=SimpleNamespace(
                paths_by_role={"habitat_ao_config": Path("dog-b.ao_config")}
            ),
        ),
    ]
    handles = _load_actor_templates(simulator, tracks)
    assert handles == ["handle:dog-a", "handle:dog-b"]
    assert calls == ["dog-a.ao_config", "dog-b.ao_config"]


def test_zero_object_id_pixels_are_retained_without_a_gate() -> None:
    record = _object_id_pixel_record(np.zeros((4, 4), dtype=np.int32), 123)
    assert record["pixel_count"] == 0
    assert record["bbox_xyxy"] is None


def test_no_clobber_output_directory(tmp_path: Path) -> None:
    first = _fresh_output_directory(tmp_path / "capture")
    assert first.is_dir()
    with pytest.raises(NativeMultiActorVisualError, match="existing"):
        _fresh_output_directory(first)


def _make_capture(root: Path, *, altered: bool = False) -> None:
    arrays = root / "arrays"
    arrays.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(ARRAY_ARTIFACTS):
        value = np.asarray([index, 1 if altered and index == 0 else 0], dtype=np.int64)
        np.save(arrays / name, value)
    write_json(
        root / "frame_records.json", {"frames": [{"frame_index": 0, "x": altered}]}
    )


def test_repeat_capture_comparison_checks_all_arrays_and_records(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _make_capture(first)
    _make_capture(second)
    assert compare_capture_outputs(first, second) == {
        "status": "pass",
        "differing_artifacts": [],
    }
    _make_capture(second, altered=True)
    result = compare_capture_outputs(first, second)
    assert result["status"] == "fail"
    assert "arrays/rgb.npy" in result["differing_artifacts"]
    assert "frame_records.json" in result["differing_artifacts"]


def test_per_actor_joint_arrays_can_have_different_lengths(tmp_path: Path) -> None:
    arrays = tmp_path / "arrays"
    arrays.mkdir()
    _save_array(arrays, "actual_joint_readback_actor0.npy", [np.zeros(2), np.ones(2)])
    _save_array(arrays, "actual_joint_readback_actor1.npy", [np.zeros(3), np.ones(3)])
    assert np.load(arrays / "actual_joint_readback_actor0.npy").shape == (2, 2)
    assert np.load(arrays / "actual_joint_readback_actor1.npy").shape == (2, 3)


def test_cli_exposes_native_region_multi_actor_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: dict[str, object] = {}

    def fake_capture(**kwargs: object) -> dict[str, object]:
        calls.update(kwargs)
        return {
            "status": "research_only",
            "research_only": True,
            "episode_counted": False,
            "capture": {"frame_count": 75, "actor_pair_collision_free": True},
        }

    monkeypatch.setattr(cli, "capture_mp3d_region_case", fake_capture)
    output = tmp_path / "capture"
    argv = [
        "m5",
        "capture-mp3d-region-multi-actor-native",
        "--case-manifest",
        "case.json",
        "--room-manifest",
        "room.json",
        "--m1-request",
        "camera.json",
        "--runtime-prefix",
        "prefix",
        "--mp3d-root",
        "mp3d",
        "--magnum-python-site",
        "magnum",
        "--rlr-sdk-root",
        "rlr",
        "--gpu-device-id",
        "2",
        "--output",
        str(output),
    ]
    parsed = cli.build_parser().parse_args(argv)
    assert parsed.m5_command == "capture-mp3d-region-multi-actor-native"
    assert parsed.gpu_device_id == 2
    assert cli.main(argv) == 0
    assert calls == {
        "case_manifest_path": "case.json",
        "room_manifest_path": "room.json",
        "m1_request_path": "camera.json",
        "runtime_prefix": "prefix",
        "mp3d_root": "mp3d",
        "magnum_python_site": "magnum",
        "output_directory": output.resolve(),
        "rlr_sdk_root": "rlr",
        "gpu_device_id": 2,
    }
    assert json.loads(capsys.readouterr().out) == {
        "actor_pair_collision_free": True,
        "episode_counted": False,
        "frame_count": 75,
        "output": str(output.resolve()),
        "receipt": str(output.resolve() / "research_receipt.json"),
        "research_only": True,
        "status": "research_only",
    }
