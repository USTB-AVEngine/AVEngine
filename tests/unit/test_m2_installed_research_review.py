from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import avengine.m1.contracts as m1_contracts
import avengine.m1.habitat_capture as m1_habitat_capture
import avengine.m2.habitat_capture as habitat_capture
from avengine.m2.habitat_capture import HabitatCaptureError
from tools.m2 import capture_installed_research_review as installed_cli


def _array_artifacts() -> dict[str, dict[str, Any]]:
    return {
        modality: {
            "artifact": {
                "path": f"arrays/{modality}.npy",
                "byte_size": 1,
                "sha256": "a" * 64,
            }
        }
        for modality in ("rgb", "depth", "semantic")
    }


def _review_media() -> dict[str, Any]:
    return {
        "videos": {
            modality: {
                "artifact": {
                    "path": f"review_media/view0_{modality}_review.mp4",
                    "byte_size": 1,
                    "sha256": "b" * 64,
                }
            }
            for modality in ("rgb", "depth", "semantic")
        }
    }


def test_installed_entry_prepares_explicit_runtime_without_legacy_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = object()
    room_inputs = SimpleNamespace(room={"room_kind": "blender_custom"})
    runtime_prefix = tmp_path / "prefix"
    magnum_site = tmp_path / "magnum"
    installed_runtime = SimpleNamespace(
        habitat_sim=SimpleNamespace(built_with_bullet=True)
    )
    result = {"status": "research_only"}
    prepare_calls: list[dict[str, object]] = []
    capture_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        habitat_capture,
        "_reload_research_review_context",
        lambda *_args: (inputs, room_inputs),
    )

    def fake_prepare(**kwargs: object) -> object:
        prepare_calls.append(kwargs)
        return installed_runtime

    def fake_capture(*args: object, **kwargs: object) -> dict[str, object]:
        assert args == (inputs, room_inputs, tmp_path / "output")
        capture_calls.append(kwargs)
        return result

    monkeypatch.setattr(
        m1_habitat_capture,
        "prepare_installed_habitat_runtime",
        fake_prepare,
    )
    monkeypatch.setattr(habitat_capture, "_capture_m2_states", fake_capture)

    assert (
        habitat_capture.capture_m2_installed_research_review(
            inputs,
            room_inputs,
            tmp_path / "output",
            runtime_prefix=runtime_prefix,
            magnum_python_site=magnum_site,
        )
        == result
    )
    assert prepare_calls == [
        {
            "runtime_prefix": runtime_prefix,
            "magnum_python_site": magnum_site,
        }
    ]
    assert capture_calls == [
        {
            "runtime_root": None,
            "review_only": True,
            "installed_runtime": installed_runtime,
        }
    ]


def test_installed_entry_requires_bullet_before_capture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = object()
    room_inputs = SimpleNamespace(room={"room_kind": "blender_custom"})
    no_bullet_runtime = SimpleNamespace(
        habitat_sim=SimpleNamespace(built_with_bullet=False)
    )
    monkeypatch.setattr(
        habitat_capture,
        "_reload_research_review_context",
        lambda *_args: (inputs, room_inputs),
    )
    monkeypatch.setattr(
        m1_habitat_capture,
        "prepare_installed_habitat_runtime",
        lambda **_kwargs: no_bullet_runtime,
    )
    monkeypatch.setattr(
        habitat_capture,
        "_capture_m2_states",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("no-Bullet runtime reached capture")
        ),
    )

    with pytest.raises(HabitatCaptureError, match="requires a Bullet-enabled"):
        habitat_capture.capture_m2_installed_research_review(
            inputs,
            room_inputs,
            tmp_path / "output",
            runtime_prefix=tmp_path / "prefix",
            magnum_python_site=tmp_path / "magnum",
        )


def test_installed_entry_rejects_non_blender_room_before_runtime_activation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = object()
    room_inputs = SimpleNamespace(room={"room_kind": "mp3d"})
    monkeypatch.setattr(
        habitat_capture,
        "_reload_research_review_context",
        lambda *_args: (inputs, room_inputs),
    )

    def unexpected_prepare(**_kwargs: object) -> object:
        raise AssertionError("non-Blender room activated an installed runtime")

    monkeypatch.setattr(
        m1_habitat_capture,
        "prepare_installed_habitat_runtime",
        unexpected_prepare,
    )

    with pytest.raises(HabitatCaptureError, match="only blender_custom rooms"):
        habitat_capture.capture_m2_installed_research_review(
            inputs,
            room_inputs,
            tmp_path / "output",
            runtime_prefix=tmp_path / "prefix",
            magnum_python_site=tmp_path / "magnum",
        )


def test_installed_entry_wraps_git_checkout_prefix_rejection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = object()
    room_inputs = SimpleNamespace(room={"room_kind": "blender_custom"})
    checkout = tmp_path / "old-habitat-checkout"
    checkout.mkdir()
    (checkout / ".git").mkdir()
    monkeypatch.setattr(
        habitat_capture,
        "_reload_research_review_context",
        lambda *_args: (inputs, room_inputs),
    )

    with pytest.raises(HabitatCaptureError, match="must not be inside a Git checkout"):
        habitat_capture.capture_m2_installed_research_review(
            inputs,
            room_inputs,
            tmp_path / "output",
            runtime_prefix=checkout,
            magnum_python_site=tmp_path / "magnum",
        )


def test_installed_capture_injects_prefix_physics_and_writes_plain_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prefix = tmp_path / "installed-prefix"
    module_path = prefix / "habitat_sim" / "__init__.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("", encoding="utf-8")
    magnum_site = tmp_path / "magnum-site"
    physics_config = prefix / "config" / "default.physics_config.json"
    physics_config.parent.mkdir(parents=True)
    physics_config.write_text("{}\n", encoding="utf-8")
    configuration_calls: list[dict[str, object]] = []
    resolved_calls: list[tuple[object, object]] = []
    static_calls: list[tuple[object, object]] = []
    loaded_calls: list[tuple[object, object]] = []

    class FakeAgentState:
        pass

    class FakeSimulator:
        def __init__(self, _configuration: object) -> None:
            self.pathfinder = SimpleNamespace()
            self.sensors = {
                "rig_rgb": object(),
                "rig_depth": object(),
                "rig_semantic": object(),
            }

        def __enter__(self) -> FakeSimulator:
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

        def seed(self, _seed: int) -> None:
            return None

        def initialize_agent(self, _agent_id: int, _state: object) -> None:
            return None

        def get_world_time(self) -> float:
            return 0.0

    habitat_sim = SimpleNamespace(
        __file__=str(module_path),
        AgentState=FakeAgentState,
        Simulator=lambda configuration: FakeSimulator(configuration),
    )
    installed_runtime = SimpleNamespace(
        prefix=prefix,
        mp3d_root=None,
        magnum_python_site=magnum_site,
        physics_config_path=physics_config,
        quaternion=SimpleNamespace(quaternion=lambda *values: values),
        habitat_sim=habitat_sim,
        magnum=object(),
    )
    inputs = SimpleNamespace(
        request={"seed": 17},
        asset_path=tmp_path / "asset_manifest.json",
        request_path=tmp_path / "request.json",
    )
    room_inputs = SimpleNamespace(
        room={"room_id": "blender_custom_two_zone_v1", "room_kind": "blender_custom"},
        room_path=tmp_path / "room_manifest.json",
        request_path=tmp_path / "room_request.json",
        request={
            "primary_camera_rig": {
                "world_from_rig": {
                    "translation_m": [0.0, 0.0, 0.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            }
        },
    )
    bundle = SimpleNamespace(
        paths_by_role={"visual": tmp_path / "animal.glb"},
        semantic_id=200,
    )

    def fake_resolved_assets(
        _room_inputs: object,
        runtime_root: object,
        *,
        mp3d_root: object = None,
    ) -> list[dict[str, object]]:
        resolved_calls.append((runtime_root, mp3d_root))
        return []

    def fake_static_graph(
        _room_inputs: object,
        runtime_root: object,
        *,
        mp3d_root: object = None,
    ) -> list[str]:
        static_calls.append((runtime_root, mp3d_root))
        return []

    def fake_loaded_graph(
        _room_inputs: object,
        runtime_root: object,
        _simulator: object,
        *,
        declared_navmesh_loaded: bool,
        mp3d_root: object = None,
    ) -> tuple[list[str], dict[str, object]]:
        assert declared_navmesh_loaded is False
        loaded_calls.append((runtime_root, mp3d_root))
        return [], {"loaded": "blender_custom"}

    def fake_make_configuration(
        _room_inputs: object,
        runtime_root: object,
        output: Path,
        *,
        mp3d_root: object = None,
        include_audio_sensor: bool = True,
        physics_config_path: object = None,
    ) -> tuple[object, dict[str, str], str, dict[str, object]]:
        configuration_calls.append(
            {
                "runtime_root": runtime_root,
                "output": output,
                "mp3d_root": mp3d_root,
                "include_audio_sensor": include_audio_sensor,
                "physics_config_path": physics_config_path,
            }
        )
        return (
            object(),
            {"rgb": "rig_rgb", "depth": "rig_depth", "semantic": "rig_semantic"},
            "listener0",
            {"navmesh": None},
        )

    monkeypatch.setattr(
        habitat_capture, "load_runtime_asset_bundle", lambda _inputs: bundle
    )
    monkeypatch.setattr(
        habitat_capture, "compile_frame_applications", lambda *_args: [object()]
    )
    monkeypatch.setattr(
        m1_habitat_capture,
        "discover_runtime_root",
        lambda *_args: (_ for _ in ()).throw(AssertionError("legacy runtime used")),
    )
    monkeypatch.setattr(m1_habitat_capture, "_resolved_assets", fake_resolved_assets)
    monkeypatch.setattr(m1_contracts, "validate_scene_asset_graph", fake_static_graph)
    monkeypatch.setattr(
        m1_contracts,
        "validate_loaded_scene_asset_graph",
        fake_loaded_graph,
    )
    monkeypatch.setattr(
        m1_habitat_capture, "_make_configuration", fake_make_configuration
    )
    monkeypatch.setattr(
        habitat_capture,
        "_instantiate_articulated_object",
        lambda *_args, **_kwargs: (
            SimpleNamespace(get_link_name=lambda _link_id: "root"),
            SimpleNamespace(to_json_data=lambda: {"runtime_joint_order": []}),
            [],
        ),
    )
    monkeypatch.setattr(
        habitat_capture,
        "apply_and_capture_fixed_frame",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        habitat_capture, "save_capture_arrays", lambda *_args: _array_artifacts()
    )
    monkeypatch.setattr(
        habitat_capture,
        "write_research_review_media",
        lambda *_args: _review_media(),
    )

    output = tmp_path / "output"
    receipt = habitat_capture._capture_m2_states(
        inputs,
        room_inputs,
        output,
        runtime_root=None,
        review_only=True,
        installed_runtime=installed_runtime,
    )

    assert resolved_calls == [(None, None)]
    assert static_calls == [(None, None)]
    assert loaded_calls == [(None, None)]
    assert configuration_calls == [
        {
            "runtime_root": None,
            "output": output.resolve(),
            "mp3d_root": None,
            "include_audio_sensor": False,
            "physics_config_path": physics_config,
        }
    ]
    assert receipt["status"] == "research_only"
    assert receipt["qualification_claim"] is False
    assert receipt["formal_admission"] is False
    assert receipt["runtime"]["mode"] == "non_git_installed_prefix"
    assert receipt["runtime"]["physics_config_path"] == str(physics_config)
    assert receipt["artifacts"]["arrays"]["rgb"] == "arrays/rgb.npy"
    assert receipt["artifacts"]["review_media"]["semantic"] == (
        "review_media/view0_semantic_review.mp4"
    )
    encoded = json.dumps(receipt, sort_keys=True)
    assert "schema" not in encoded
    assert "sha256" not in encoded
    assert "runtime_identity" not in encoded
    assert (
        json.loads((output / "research_receipt.json").read_text(encoding="utf-8"))
        == receipt
    )
    assert not (output / "evidence.json").exists()


def test_cli_uses_explicit_prefix_and_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prefix = tmp_path / "installed-prefix"
    prefix.mkdir()
    output = tmp_path / "output"
    inputs = object()
    room_inputs = object()
    calls: list[dict[str, object]] = []
    receipt = {
        "status": "research_only",
        "research_only": True,
        "qualification_claim": False,
        "formal_admission": False,
        "capture": {"frame_count": 75, "review_view_ids": ["view0"]},
    }
    monkeypatch.setattr(
        installed_cli, "load_research_review_inputs", lambda *_args: inputs
    )
    monkeypatch.setattr(installed_cli, "load_m1_inputs", lambda *_args: room_inputs)

    def fake_capture(*args: object, **kwargs: object) -> dict[str, object]:
        assert args == (inputs, room_inputs, output.resolve())
        calls.append(kwargs)
        return receipt

    monkeypatch.setattr(
        installed_cli, "capture_m2_installed_research_review", fake_capture
    )

    assert (
        installed_cli.main(
            [
                "--asset-manifest",
                str(tmp_path / "asset.json"),
                "--request",
                str(tmp_path / "request.json"),
                "--room-manifest",
                str(tmp_path / "room.json"),
                "--room-request",
                str(tmp_path / "room_request.json"),
                "--runtime-prefix",
                str(prefix),
                "--magnum-python-site",
                str(tmp_path / "magnum"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert calls == [
        {
            "runtime_prefix": prefix.resolve(),
            "magnum_python_site": tmp_path / "magnum",
        }
    ]
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "formal_admission": False,
        "frame_count": 75,
        "output": str(output.resolve()),
        "qualification_claim": False,
        "receipt": str(output.resolve() / "research_receipt.json"),
        "research_only": True,
        "review_view_ids": ["view0"],
        "status": "research_only",
    }


def test_cli_reports_bullet_capability_failure_as_usage_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prefix = tmp_path / "installed-prefix"
    prefix.mkdir()
    monkeypatch.setattr(
        installed_cli,
        "resolve_installed_runtime_prefix",
        lambda value: value.resolve(),
    )
    monkeypatch.setattr(
        installed_cli,
        "load_research_review_inputs",
        lambda *_args: object(),
    )
    monkeypatch.setattr(installed_cli, "load_m1_inputs", lambda *_args: object())
    monkeypatch.setattr(
        installed_cli,
        "capture_m2_installed_research_review",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            HabitatCaptureError(
                "installed-prefix M2 Blender-custom research requires a "
                "Bullet-enabled Habitat runtime"
            )
        ),
    )

    with pytest.raises(SystemExit) as raised:
        installed_cli.main(
            [
                "--asset-manifest",
                str(tmp_path / "asset.json"),
                "--request",
                str(tmp_path / "request.json"),
                "--room-manifest",
                str(tmp_path / "room.json"),
                "--room-request",
                str(tmp_path / "room_request.json"),
                "--runtime-prefix",
                str(prefix),
                "--magnum-python-site",
                str(tmp_path / "magnum"),
                "--output",
                str(tmp_path / "output"),
            ]
        )

    assert raised.value.code == 2
    assert "Bullet-enabled Habitat runtime" in capsys.readouterr().err


def test_cli_rejects_checkout_prefix_before_loading_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkout = tmp_path / "old-habitat-checkout"
    checkout.mkdir()
    (checkout / ".git").mkdir()

    with pytest.raises(SystemExit) as raised:
        installed_cli.main(
            [
                "--asset-manifest",
                str(tmp_path / "asset.json"),
                "--request",
                str(tmp_path / "request.json"),
                "--room-manifest",
                str(tmp_path / "room.json"),
                "--room-request",
                str(tmp_path / "room_request.json"),
                "--runtime-prefix",
                str(checkout),
                "--magnum-python-site",
                str(tmp_path / "magnum"),
                "--output",
                str(tmp_path / "output"),
            ]
        )

    assert raised.value.code == 2
    assert "must not be inside a Git checkout" in capsys.readouterr().err


def test_cli_does_not_accept_the_legacy_runtime_root_option() -> None:
    with pytest.raises(SystemExit) as raised:
        installed_cli.main(["--runtime-root", "/old/habitat-checkout"])

    assert raised.value.code == 2
