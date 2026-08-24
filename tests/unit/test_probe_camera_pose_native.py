from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.m1 import probe_camera_pose_native as probe


class _AgentState:
    position = None
    rotation = None


class _Pathfinder:
    def __init__(self) -> None:
        self.loaded = []

    def load_nav_mesh(self, path: str) -> bool:
        self.loaded.append(path)
        return True

    def snap_point(self, point) -> np.ndarray:
        value = np.asarray(point, dtype=np.float64).copy()
        value[1] = 0.271
        return value

    def is_navigable(self, _point, _maximum_y_delta_m) -> bool:
        return True

    def get_island(self, _point) -> int:
        return 7


class _Simulator:
    def __init__(self, configuration) -> None:
        self.configuration = configuration
        self.pathfinder = _Pathfinder()
        self.sensors = {name: object() for name in ("rgb0", "depth0", "semantic0")}
        self.seed_value = None
        self.agent_state = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def seed(self, value: int) -> None:
        self.seed_value = value

    def initialize_agent(self, _agent_id: int, state: _AgentState) -> None:
        self.agent_state = state

    def render_sensors(self, _wrappers):
        return {
            "rgb0": np.full((4, 5, 4), 127, dtype=np.uint8),
            "depth0": np.full((4, 5), 2.5, dtype=np.float32),
            "semantic0": np.full((4, 5), 9, dtype=np.int32),
        }


def _inputs() -> SimpleNamespace:
    return SimpleNamespace(
        room={"room_id": "apartment-room"},
        request={
            "request_id": "view-request",
            "seed": 17,
            "primary_camera_rig": {
                "rig_id": "camera_rig_0",
                "view_id": "view0",
                "world_from_rig": {
                    "translation_m": [-0.7, 1.471, 0.65],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
            "listener": {
                "rig_from_listener": {
                    "translation_m": [0, 0, 0],
                    "rotation_xyzw": [0, 0, 0, 1],
                }
            },
        },
    )


def test_probe_passes_explicit_installed_runtime_inputs_and_is_no_clobber(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    navmesh = tmp_path / "room.navmesh"
    navmesh.write_bytes(b"navmesh")
    runtime_prefix = tmp_path / "runtime"
    magnum_site = tmp_path / "magnum"
    rlr_sdk_root = tmp_path / "rlr"
    pbr_asset_root = tmp_path / "pbr"
    for directory in (runtime_prefix, magnum_site, rlr_sdk_root, pbr_asset_root):
        directory.mkdir()
    physics = runtime_prefix / "default.physics_config.json"
    physics.write_text("{}\n", encoding="utf-8")
    calls: dict[str, object] = {"prepare": [], "configuration": []}
    simulators = []

    class _Habitat:
        AgentState = _AgentState

        @staticmethod
        def Simulator(configuration):
            simulator = _Simulator(configuration)
            simulators.append(simulator)
            return simulator

    runtime = SimpleNamespace(
        prefix=runtime_prefix.resolve(),
        mp3d_root=None,
        pbr_asset_root=pbr_asset_root.resolve(),
        magnum_python_site=magnum_site.resolve(),
        physics_config_path=physics.resolve(),
        quaternion=object(),
        habitat_sim=_Habitat,
    )
    monkeypatch.setattr(probe, "load_and_validate_inputs", lambda *_args: _inputs())
    monkeypatch.setattr(
        probe,
        "prepare_installed_habitat_runtime",
        lambda **kwargs: calls["prepare"].append(kwargs) or runtime,
    )

    def fake_configuration(*args, **kwargs):
        calls["configuration"].append((args, kwargs))
        return (
            "configuration",
            {"rgb": "rgb0", "depth": "depth0", "semantic": "semantic0"},
            "listener0",
            {"navmesh": navmesh},
        )

    monkeypatch.setattr(probe, "_make_configuration", fake_configuration)
    monkeypatch.setattr(probe, "_numpy_quaternion", lambda value, qt: (value, qt))
    output = tmp_path / "fresh-probe"
    args = probe.parse_args(
        [
            "--room",
            str(tmp_path / "room.json"),
            "--request",
            str(tmp_path / "request.json"),
            "--runtime-prefix",
            str(runtime_prefix),
            "--magnum-python-site",
            str(magnum_site),
            "--rlr-sdk-root",
            str(rlr_sdk_root),
            "--pbr-asset-root",
            str(pbr_asset_root),
            "--expected-floor-point-m",
            "-0.7",
            "0.271",
            "0.65",
            "--expected-floor-island-id",
            "7",
            "--output",
            str(output),
        ]
    )

    receipt_path = probe.run_probe(args)

    assert calls["prepare"] == [
        {
            "runtime_prefix": runtime_prefix,
            "magnum_python_site": magnum_site,
            "rlr_sdk_root": rlr_sdk_root,
            "pbr_asset_root": pbr_asset_root,
            "mp3d_root": None,
            "allow_mp3d_environment": False,
        }
    ]
    configuration_args, configuration_kwargs = calls["configuration"][0]
    assert configuration_args[1:] == (None, output)
    assert configuration_kwargs == {
        "mp3d_root": None,
        "include_audio_sensor": False,
        "physics_config_path": physics.resolve(),
    }
    assert simulators[0].pathfinder.loaded == [str(navmesh)]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "pass"
    assert receipt["classification"] == "native_interactive_probe_not_release_evidence"
    assert receipt["navigation"]["floor_authority"] == (
        "explicit_builder_expected_floor_point"
    )
    assert receipt["navigation"]["expected_floor_passed"] is True
    assert receipt["navigation"]["floor_island_id"] == 7
    assert receipt["installed_runtime_inputs"] == {
        "runtime_prefix": str(runtime_prefix.resolve()),
        "magnum_python_site": str(magnum_site.resolve()),
        "rlr_sdk_root": str(rlr_sdk_root.resolve()),
        "pbr_asset_root": str(pbr_asset_root.resolve()),
        "mp3d_root": None,
        "physics_config_path": str(physics.resolve()),
        "include_audio_sensor": False,
    }
    assert set(receipt["observations"]) == {"rgb", "depth", "semantic"}

    with pytest.raises(FileExistsError, match="refusing to replace output"):
        probe.run_probe(args)
    assert len(calls["prepare"]) == 1


def test_cli_accepts_explicit_optional_mp3d_root(tmp_path: Path) -> None:
    args = probe.parse_args(
        [
            "--room",
            "room.json",
            "--request",
            "request.json",
            "--runtime-prefix",
            "runtime",
            "--magnum-python-site",
            "magnum",
            "--rlr-sdk-root",
            "rlr",
            "--pbr-asset-root",
            "pbr",
            "--mp3d-root",
            str(tmp_path / "mp3d"),
            "--output",
            str(tmp_path / "output"),
        ]
    )

    assert args.magnum_python_site == Path("magnum")
    assert args.rlr_sdk_root == Path("rlr")
    assert args.pbr_asset_root == Path("pbr")
    assert args.mp3d_root == tmp_path / "mp3d"


@pytest.mark.parametrize(
    ("status", "expected_code", "marker"),
    [("pass", 0, "PROBE_OK"), ("fail", 2, "PROBE_FAIL")],
)
def test_main_exit_code_follows_receipt_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    status: str,
    expected_code: int,
    marker: str,
) -> None:
    receipt = tmp_path / f"{status}.json"
    receipt.write_text(json.dumps({"status": status}), encoding="utf-8")
    monkeypatch.setattr(probe, "parse_args", lambda _argv: SimpleNamespace())
    monkeypatch.setattr(probe, "run_probe", lambda _args: receipt)

    assert probe.main([]) == expected_code
    assert marker in capsys.readouterr().out
