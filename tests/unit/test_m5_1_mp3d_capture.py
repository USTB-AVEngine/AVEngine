from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from avengine.m5_1.legacy_route import FRAME_COUNT
from avengine.m5_1.mp3d_capture import (
    MP3DCaptureError,
    _assert_route_geometry,
    _pathfinder_path_record,
    capture_mp3d_route,
    derive_mp3d_route_paths,
    load_mp3d_route_manifest,
    validate_mp3d_paths_with_declared_navmesh,
    write_mp3d_contact_sheet,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ROUTE_PATH = (
    REPOSITORY_ROOT
    / "examples"
    / "m5_1"
    / "mp3d_articulated_review"
    / "route_manifest.json"
)
CLI_PATH = REPOSITORY_ROOT / "tools/m5_1/capture_human_beagle_mp3d.py"
CLI_SPEC = importlib.util.spec_from_file_location(
    "capture_human_beagle_mp3d_cli", CLI_PATH
)
assert CLI_SPEC is not None and CLI_SPEC.loader is not None
CLI = importlib.util.module_from_spec(CLI_SPEC)
CLI_SPEC.loader.exec_module(CLI)


def _mp3d_cli_argv(
    tmp_path: Path, *, include_runtime_root: bool = True
) -> list[str]:
    argv = [
        "--route-manifest",
        str(tmp_path / "route.json"),
        "--room-manifest",
        str(tmp_path / "room.json"),
        "--m1-request",
        str(tmp_path / "m1.json"),
        "--human-runtime-glb",
        str(tmp_path / "human.glb"),
        "--beagle-manifest",
        str(tmp_path / "beagle.json"),
        "--beagle-m2-request",
        str(tmp_path / "beagle_m2.json"),
        "--pbr-asset-root",
        str(tmp_path / "pbr-assets"),
        "--output",
        str(tmp_path / "output"),
    ]
    if include_runtime_root:
        argv.extend(["--runtime-root", str(tmp_path / "runtime")])
    return argv


def _fake_mp3d_capture_result(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        capture=SimpleNamespace(output_dir=tmp_path / "output"),
        gate_evidence={
            "status": "pass",
            "frame_count": FRAME_COUNT,
            "mixed_capture": {
                "semantic_visible_frame_count": {
                    "human0": FRAME_COUNT,
                    "dog0": FRAME_COUNT,
                }
            },
            "pathfinder": {
                "geometry": {"minimum_center_separation_m": 0.9}
            },
            "evidence_content_sha256": "fixture",
        },
        contact_sheet_path=tmp_path / "contact_sheet.png",
    )


def test_mp3d_cli_rejects_prefix_and_root_together(tmp_path: Path) -> None:
    parser = CLI._parser()
    argv = _mp3d_cli_argv(tmp_path, include_runtime_root=False)
    argv.extend(
        [
            "--runtime-prefix",
            str(tmp_path / "prefix"),
            "--runtime-root",
            str(tmp_path / "root"),
        ]
    )
    with pytest.raises(SystemExit):
        parser.parse_args(argv)


def test_mp3d_cli_requires_pbr_root_before_capture_output(tmp_path: Path) -> None:
    parser = CLI._parser()
    argv = _mp3d_cli_argv(tmp_path)
    option_index = argv.index("--pbr-asset-root")
    del argv[option_index : option_index + 2]

    with pytest.raises(SystemExit):
        parser.parse_args(argv)
    assert not (tmp_path / "output").exists()

def test_mp3d_cli_forwards_runtime_root_alias(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []

    def fake_capture(**kwargs):
        calls.append(kwargs)
        return _fake_mp3d_capture_result(tmp_path)

    monkeypatch.setattr(CLI, "capture_mp3d_route", fake_capture)

    assert CLI.main(_mp3d_cli_argv(tmp_path)) == 0
    assert calls[0]["runtime_prefix"] is None
    assert calls[0]["runtime_root"] == tmp_path / "runtime"
    assert calls[0]["mp3d_root"] is None
    assert calls[0]["pbr_asset_root"] == tmp_path / "pbr-assets"
    assert calls[0]["magnum_python_site"] is None


def test_mp3d_cli_forwards_installed_prefix_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []

    def fake_capture(**kwargs):
        calls.append(kwargs)
        return _fake_mp3d_capture_result(tmp_path)

    monkeypatch.setattr(CLI, "capture_mp3d_route", fake_capture)
    argv = _mp3d_cli_argv(tmp_path, include_runtime_root=False)
    argv.extend(
        [
            "--runtime-prefix",
            str(tmp_path / "prefix"),
            "--mp3d-root",
            str(tmp_path / "mp3d"),
            "--magnum-python-site",
            str(tmp_path / "magnum"),
        ]
    )

    assert CLI.main(argv) == 0
    assert calls[0]["runtime_prefix"] == tmp_path / "prefix"
    assert calls[0]["runtime_root"] is None
    assert calls[0]["mp3d_root"] == tmp_path / "mp3d"
    assert calls[0]["pbr_asset_root"] == tmp_path / "pbr-assets"
    assert calls[0]["magnum_python_site"] == tmp_path / "magnum"


def test_mp3d_cli_defers_runtime_selection_to_installed_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setenv("AVENGINE_HABITAT_RUNTIME_PREFIX", str(tmp_path / "prefix"))
    monkeypatch.setenv("AVENGINE_MP3D_ROOT", str(tmp_path / "mp3d"))
    monkeypatch.setenv(
        "AVENGINE_HABITAT_MAGNUM_PYTHON_SITE", str(tmp_path / "magnum")
    )

    def fake_capture(**kwargs):
        calls.append(kwargs)
        return _fake_mp3d_capture_result(tmp_path)

    monkeypatch.setattr(CLI, "capture_mp3d_route", fake_capture)

    assert CLI.main(_mp3d_cli_argv(tmp_path, include_runtime_root=False)) == 0
    assert calls[0]["runtime_prefix"] is None
    assert calls[0]["runtime_root"] is None
    assert calls[0]["mp3d_root"] is None
    assert calls[0]["pbr_asset_root"] == tmp_path / "pbr-assets"
    assert calls[0]["magnum_python_site"] is None


def test_mp3d_route_requires_pbr_root_before_runtime_or_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def unexpected_prepare(**_kwargs: object) -> object:
        raise AssertionError("missing PBR root must fail before runtime prepare")

    monkeypatch.setattr(
        "avengine.m5_1.mp3d_capture.prepare_installed_habitat_runtime",
        unexpected_prepare,
    )
    output = tmp_path / "output"
    with pytest.raises(MP3DCaptureError, match="explicit pbr_asset_root"):
        capture_mp3d_route(
            route_manifest_path=ROUTE_PATH,
            room_manifest_path=tmp_path / "room.json",
            m1_request_path=tmp_path / "request.json",
            human_runtime_glb_path=tmp_path / "human.glb",
            beagle_animal_manifest_path=tmp_path / "beagle.json",
            beagle_m2_request_path=tmp_path / "beagle_request.json",
            output_dir=output,
        )
    assert not output.exists()


def test_mp3d_route_environment_runtime_reaches_mixed_capture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    installed_runtime = SimpleNamespace(
        mp3d_root=tmp_path / "mp3d",
        pbr_asset_root=tmp_path / "pbr-assets",
    )
    prepare_calls: list[dict[str, object]] = []
    mixed_calls: list[dict[str, object]] = []
    monkeypatch.setenv("AVENGINE_HABITAT_RUNTIME_PREFIX", str(tmp_path / "prefix"))
    monkeypatch.setenv("AVENGINE_MP3D_ROOT", str(tmp_path / "mp3d"))
    monkeypatch.setenv(
        "AVENGINE_HABITAT_MAGNUM_PYTHON_SITE", str(tmp_path / "magnum")
    )

    def fake_prepare(**kwargs: object) -> object:
        prepare_calls.append(kwargs)
        return installed_runtime

    paths = SimpleNamespace(
        human=np.zeros((FRAME_COUNT, 3), dtype=np.float64),
        beagle=np.zeros((FRAME_COUNT, 3), dtype=np.float64),
    )
    navigation = SimpleNamespace(
        paths=paths,
        record={
            "routes": {
                "human0": {"trajectory_sha256": "human"},
                "dog0": {"trajectory_sha256": "dog"},
            }
        },
    )

    def stop_after_mixed_capture_receives_runtime(**kwargs: object) -> object:
        mixed_calls.append(kwargs)
        raise RuntimeError("mixed capture received installed runtime")

    monkeypatch.setattr(
        "avengine.m5_1.mp3d_capture.prepare_installed_habitat_runtime",
        fake_prepare,
    )
    monkeypatch.setattr(
        "avengine.m5_1.mp3d_capture.validate_mp3d_paths_with_declared_navmesh",
        lambda **kwargs: navigation,
    )
    monkeypatch.setattr(
        "avengine.m5_1.mp3d_capture.capture_human_beagle_paths",
        stop_after_mixed_capture_receives_runtime,
    )

    with pytest.raises(RuntimeError, match="mixed capture received installed runtime"):
        capture_mp3d_route(
            route_manifest_path=ROUTE_PATH,
            room_manifest_path=tmp_path / "room.json",
            m1_request_path=tmp_path / "request.json",
            human_runtime_glb_path=tmp_path / "human.glb",
            beagle_animal_manifest_path=tmp_path / "beagle.json",
            beagle_m2_request_path=tmp_path / "beagle_request.json",
            output_dir=tmp_path / "output",
            pbr_asset_root=tmp_path / "pbr-assets",
        )

    assert prepare_calls == [
        {
            "runtime_prefix": None,
            "runtime_root": None,
            "mp3d_root": None,
            "pbr_asset_root": tmp_path / "pbr-assets",
            "magnum_python_site": None,
        }
    ]
    assert mixed_calls[0]["installed_runtime"] is installed_runtime


def test_mp3d_route_reuses_injected_installed_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    installed_runtime = SimpleNamespace(
        mp3d_root=tmp_path / "mp3d",
        pbr_asset_root=tmp_path / "pbr-assets",
    )
    paths = SimpleNamespace(
        human=np.zeros((FRAME_COUNT, 3), dtype=np.float64),
        beagle=np.zeros((FRAME_COUNT, 3), dtype=np.float64),
    )
    navigation = SimpleNamespace(
        paths=paths,
        record={
            "routes": {
                "human0": {"trajectory_sha256": "human"},
                "dog0": {"trajectory_sha256": "dog"},
            }
        },
    )
    mixed_calls: list[dict[str, object]] = []

    def fail_if_runtime_is_prepared(**_kwargs: object) -> object:
        raise AssertionError("injected installed runtime must be reused")

    def stop_after_mixed_capture_receives_runtime(**kwargs: object) -> object:
        mixed_calls.append(kwargs)
        raise RuntimeError("mixed capture received injected runtime")

    monkeypatch.setattr(
        "avengine.m5_1.mp3d_capture.prepare_installed_habitat_runtime",
        fail_if_runtime_is_prepared,
    )
    monkeypatch.setattr(
        "avengine.m5_1.mp3d_capture.validate_mp3d_paths_with_declared_navmesh",
        lambda **_kwargs: navigation,
    )
    monkeypatch.setattr(
        "avengine.m5_1.mp3d_capture.capture_human_beagle_paths",
        stop_after_mixed_capture_receives_runtime,
    )

    with pytest.raises(RuntimeError, match="mixed capture received injected runtime"):
        capture_mp3d_route(
            route_manifest_path=ROUTE_PATH,
            room_manifest_path=tmp_path / "room.json",
            m1_request_path=tmp_path / "request.json",
            human_runtime_glb_path=tmp_path / "human.glb",
            beagle_animal_manifest_path=tmp_path / "beagle.json",
            beagle_m2_request_path=tmp_path / "beagle_request.json",
            output_dir=tmp_path / "output",
            installed_runtime=installed_runtime,
        )

    assert mixed_calls[0]["installed_runtime"] is installed_runtime

    with pytest.raises(
        MP3DCaptureError,
        match="installed_runtime cannot be combined with runtime path arguments",
    ):
        capture_mp3d_route(
            route_manifest_path=ROUTE_PATH,
            room_manifest_path=tmp_path / "room.json",
            m1_request_path=tmp_path / "request.json",
            human_runtime_glb_path=tmp_path / "human.glb",
            beagle_animal_manifest_path=tmp_path / "beagle.json",
            beagle_m2_request_path=tmp_path / "beagle_request.json",
            output_dir=tmp_path / "output",
            runtime_prefix=tmp_path / "prefix",
            installed_runtime=installed_runtime,
        )


def test_pathfinder_preflight_preserves_visual_sensors_without_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    route = load_mp3d_route_manifest(ROUTE_PATH)
    navmesh = tmp_path / "scene.navmesh"
    navmesh.write_bytes(b"navmesh")
    physics = tmp_path / "default.physics_config.json"
    physics.write_text("{}\n", encoding="utf-8")
    inputs = SimpleNamespace(
        room={"room_id": route["room_id"]},
        request={"request_id": route["request_id"]},
    )
    configuration_calls: list[dict[str, object]] = []
    visual_modalities = {
        "rgb": "rig_rgb",
        "depth": "rig_depth",
        "semantic": "rig_semantic",
    }

    def fake_make_configuration(
        _inputs: object,
        runtime_root: object,
        output_dir: object,
        **kwargs: object,
    ) -> tuple[object, dict[str, str], str, dict[str, object]]:
        configuration_calls.append(
            {
                "runtime_root": runtime_root,
                "output_dir": output_dir,
                **kwargs,
            }
        )
        return (
            {"sensor_uuids": list(visual_modalities.values())},
            visual_modalities,
            "listener0",
            {"navmesh": navmesh},
        )

    class FakeHabitat:
        @staticmethod
        def Simulator(configuration: object) -> object:
            assert configuration == {
                "sensor_uuids": ["rig_rgb", "rig_depth", "rig_semantic"]
            }
            raise RuntimeError("visual-only configuration reached Simulator")

    monkeypatch.setattr(
        "avengine.m5_1.mp3d_capture.load_m1_inputs",
        lambda *_args: inputs,
    )
    monkeypatch.setattr(
        "avengine.m5_1.mp3d_capture._resolved_scene",
        lambda *_args, **_kwargs: {"navmesh": navmesh},
    )
    monkeypatch.setattr(
        "avengine.m5_1.mp3d_capture._make_configuration",
        fake_make_configuration,
    )
    installed_runtime = SimpleNamespace(
        mp3d_root=tmp_path / "mp3d",
        quaternion=object(),
        habitat_sim=FakeHabitat,
        physics_config_path=physics,
    )

    with pytest.raises(RuntimeError, match="visual-only configuration"):
        validate_mp3d_paths_with_declared_navmesh(
            route=route,
            room_manifest_path=tmp_path / "room.json",
            m1_request_path=tmp_path / "request.json",
            installed_runtime=installed_runtime,
        )

    assert configuration_calls == [
        {
            "runtime_root": None,
            "output_dir": tmp_path / ".mp3d_preflight_not_retained",
            "mp3d_root": tmp_path / "mp3d",
            "include_audio_sensor": False,
            "physics_config_path": physics,
        }
    ]


def test_example_route_is_exact_parallel_separated_motion() -> None:
    route = load_mp3d_route_manifest(ROUTE_PATH)
    paths = derive_mp3d_route_paths(route)
    assert paths.human.shape == (FRAME_COUNT, 3)
    assert paths.beagle.shape == (FRAME_COUNT, 3)
    assert np.array_equal(paths.human[0], np.asarray([-4.6, 0.072447, -2.7]))
    assert np.array_equal(paths.human[-1], np.asarray([-4.6, 0.072447, -3.8]))
    assert np.array_equal(paths.beagle[0], np.asarray([-3.7, 0.072447, -2.7]))
    assert np.array_equal(paths.beagle[-1], np.asarray([-3.7, 0.072447, -3.8]))

    geometry = _assert_route_geometry(route, paths)
    assert geometry["minimum_center_separation_m"] == pytest.approx(0.9)
    assert geometry["maximum_center_separation_m"] == pytest.approx(0.9)
    for actor_id in ("human0", "dog0"):
        assert geometry["movement"][actor_id]["path_length_m"] == pytest.approx(1.1)
        assert geometry["movement"][actor_id][
            "endpoint_displacement_m"
        ] == pytest.approx(1.1)
        assert geometry["movement"][actor_id][
            "maximum_center_step_m"
        ] == pytest.approx(1.1 / 269.0)


def test_route_geometry_rejects_center_overlap() -> None:
    route = load_mp3d_route_manifest(ROUTE_PATH)
    invalid = copy.deepcopy(route)
    invalid["routes"]["dog0"] = copy.deepcopy(invalid["routes"]["human0"])
    paths = derive_mp3d_route_paths(invalid)
    with pytest.raises(MP3DCaptureError, match="minimum separation"):
        _assert_route_geometry(invalid, paths)


class _FakePathFinder:
    def __init__(self, *, failed_frame: int | None = None, step_offset: float = 0.0):
        self.failed_frame = failed_frame
        self.step_offset = step_offset
        self.calls = 0

    def is_navigable(self, point: np.ndarray, maximum_y_delta: float) -> bool:
        del point
        assert maximum_y_delta == pytest.approx(1.0e-4)
        index = self.calls
        self.calls += 1
        return index != self.failed_frame

    @staticmethod
    def snap_point(point: np.ndarray) -> np.ndarray:
        return np.asarray(point, dtype=np.float64)

    @staticmethod
    def get_island(point: np.ndarray) -> int:
        del point
        return 1

    def try_step_no_sliding(
        self, start: np.ndarray, end: np.ndarray
    ) -> np.ndarray:
        del start
        result = np.asarray(end, dtype=np.float64).copy()
        result[0] += self.step_offset
        return result


def test_pathfinder_record_requires_every_point_and_no_sliding_segment() -> None:
    route = load_mp3d_route_manifest(ROUTE_PATH)
    path = derive_mp3d_route_paths(route).human
    record = _pathfinder_path_record(
        _FakePathFinder(),
        path,
        owner="human0",
        maximum_snap_error_m=1.0e-5,
        maximum_y_delta_m=1.0e-4,
        maximum_step_endpoint_error_m=1.0e-5,
    )
    assert record["navigable_frame_count"] == FRAME_COUNT
    assert record["island_id"] == 1
    assert record["no_sliding_passed_segment_count"] == FRAME_COUNT - 1
    assert record["maximum_snap_error_m"] == 0.0
    assert record["maximum_step_endpoint_error_m"] == 0.0

    with pytest.raises(MP3DCaptureError, match="non-navigable center frames"):
        _pathfinder_path_record(
            _FakePathFinder(failed_frame=17),
            path,
            owner="human0",
            maximum_snap_error_m=1.0e-5,
            maximum_y_delta_m=1.0e-4,
            maximum_step_endpoint_error_m=1.0e-5,
        )
    with pytest.raises(MP3DCaptureError, match="no-sliding segment failures"):
        _pathfinder_path_record(
            _FakePathFinder(step_offset=2.0e-5),
            path,
            owner="human0",
            maximum_snap_error_m=1.0e-5,
            maximum_y_delta_m=1.0e-4,
            maximum_step_endpoint_error_m=1.0e-5,
        )


def test_contact_sheet_retains_nine_semantic_box_frames(tmp_path: Path) -> None:
    rgb = np.zeros((FRAME_COUNT, 8, 10, 3), dtype=np.uint8)
    rgb[..., 1] = 48
    semantic = np.zeros((FRAME_COUNT, 8, 10), dtype=np.int32)
    semantic[:, 1:5, 1:4] = 62000
    semantic[:, 4:7, 6:9] = 62001
    destination = tmp_path / "sheet.jpg"
    record = write_mp3d_contact_sheet(
        rgb=rgb,
        semantic=semantic,
        semantic_ids={"human0": 62000, "dog0": 62001},
        output_path=destination,
    )
    assert destination.is_file()
    assert record["readback_verified"] is True
    assert record["file"]["path"] == "sheet.jpg"
    assert record["size_wh"] == [30, 96]
    assert len(record["selected_frames"]) == 9
    assert all(
        item["visible_pixels"] == {"human0": 12, "dog0": 9}
        for item in record["selected_frames"]
    )
