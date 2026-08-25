from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from avengine.contracts.json_io import load_json, write_json

import avengine.cli as cli
from avengine.cli import build_parser, main
from avengine.acoustics.runtime import RuntimeUnavailableError


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REQUEST = REPOSITORY_ROOT / "examples/m3/blender_custom/canary_request.json"
M3_EXAMPLE = REPOSITORY_ROOT / "examples/m3/blender_custom"


def test_acoustics_compile_validate_and_verify_cli(tmp_path: Path, capsys) -> None:
    output = tmp_path / "canary"

    assert (
        main(
            [
                "acoustics",
                "compile-canary",
                "--request",
                str(REQUEST),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "acoustics",
                "validate-package",
                str(output / "low_absorption/manifest.json"),
            ]
        )
        == 0
    )
    assert main(["acoustics", "verify-compile", str(output / "compile_evidence.json")]) == 0
    rendered = capsys.readouterr().out
    assert '"status": "pass"' in rendered


def test_acoustics_current_installed_help_requires_explicit_runtime_paths(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["acoustics", "run-canary", "--help"])
    assert exit_info.value.code == 0
    rendered = " ".join(capsys.readouterr().out.split())
    assert (
        "Explicit installed non-checkout Habitat prefix required with "
        "--runtime-mode current-installed"
    ) in rendered
    assert (
        "Explicit external non-checkout RLRAudioPropagationPkg required with "
        "--runtime-mode current-installed"
    ) in rendered
    assert "AVENGINE_HABITAT_RUNTIME_PREFIX" not in rendered
    assert "AVENGINE_RLR_SDK_ROOT" not in rendered


def test_acoustics_research_commands_are_exposed_as_separate_cli_paths() -> None:
    parser = build_parser()

    proposed = parser.parse_args(
        [
            "acoustics",
            "propose-visual-slots",
            "--room",
            "room.json",
            "--transform-profile",
            "identity_y_up",
            "--output",
            "/tmp/proposal",
        ]
    )
    compiled = parser.parse_args(
        [
            "acoustics",
            "compile-explicit-research",
            "--room",
            "room.json",
            "--mapping",
            "mapping.json",
            "--materials",
            "materials.json",
            "--output",
            "/tmp/package",
        ]
    )
    semantic = parser.parse_args(
        [
            "acoustics",
            "compile-mp3d-semantic",
            "--room",
            "room.json",
            "--rules",
            "rules.json",
            "--output",
            "/tmp/semantic",
        ]
    )
    soundspaces = parser.parse_args(
        [
            "acoustics",
            "compile-mp3d-rlr-materials",
            "--room",
            "room.json",
            "--materials",
            "mp3d_material_config.json",
            "--database-id",
            "soundspaces_mp3d_v1",
            "--source-description",
            "SoundSpaces/RLR MP3D material config",
            "--output",
            "/tmp/soundspaces",
        ]
    )
    usd_semantic = parser.parse_args(
        [
            "acoustics",
            "compile-usd-snapshot-semantic",
            "--room",
            "room.json",
            "--rules",
            "rules.json",
            "--output",
            "/tmp/usd-semantic",
        ]
    )
    leakage = parser.parse_args(
        [
            "acoustics",
            "inspect-mesh-leakage",
            "--package",
            "manifest.json",
            "--origin",
            "0",
            "1",
            "0",
            "--output",
            "/tmp/leakage.json",
        ]
    )
    canary = parser.parse_args(
        [
            "acoustics",
            "run-canary",
            "--request",
            "request.json",
            "--compile-evidence",
            "compile.json",
            "--output",
            "/tmp/canary",
        ]
    )
    verified = parser.parse_args(
        ["acoustics", "verify-canary", "/tmp/canary/canary_evidence.json"]
    )

    assert proposed.m3_command == "propose-visual-slots"
    assert compiled.m3_command == "compile-explicit-research"
    assert semantic.m3_command == "compile-mp3d-semantic"
    assert soundspaces.m3_command == "compile-mp3d-rlr-materials"
    assert usd_semantic.m3_command == "compile-usd-snapshot-semantic"
    assert leakage.m3_command == "inspect-mesh-leakage"
    assert canary.m3_command == "run-canary"
    assert verified.m3_command == "verify-canary"


def test_acoustics_static_writers_use_explicit_non_git_mp3d_and_reject_legacy_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mp3d = tmp_path / "mp3d"
    (mp3d / "scene_datasets").mkdir(parents=True)
    seen: dict[str, object] = {}

    def fake_proposal(**kwargs: object) -> tuple[Path, Path, Path]:
        seen.update(kwargs)
        return (
            tmp_path / "mapping.json",
            tmp_path / "materials.json",
            tmp_path / "report.json",
        )

    monkeypatch.setattr(cli, "propose_visual_slot_research_materials", fake_proposal)
    monkeypatch.setenv("AVENGINE_HABITAT_RUNTIME_ROOT", "/old/habitat-checkout")
    monkeypatch.setenv("AVENGINE_MP3D_ROOT", "/old/mp3d-checkout")
    arguments = [
        "acoustics",
        "propose-visual-slots",
        "--room",
        "room.json",
        "--transform-profile",
        "identity_y_up",
        "--mp3d-root",
        str(mp3d),
        "--output",
        str(tmp_path / "output"),
    ]
    assert main(arguments) == 0
    environment = seen["environment"]
    assert isinstance(environment, dict)
    assert environment["AVENGINE_MP3D_ROOT"] == str(mp3d.resolve())
    assert "AVENGINE_HABITAT_RUNTIME_ROOT" not in environment

    assert main([*arguments, "--runtime-root", str(tmp_path / "old")]) == 2
    assert "--runtime-root is retired" in capsys.readouterr().out


def test_acoustics_static_writers_do_not_inherit_mp3d_root_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_proposal(**kwargs: object) -> tuple[Path, Path, Path]:
        seen.update(kwargs)
        return (
            tmp_path / "mapping.json",
            tmp_path / "materials.json",
            tmp_path / "report.json",
        )

    monkeypatch.setattr(cli, "propose_visual_slot_research_materials", fake_proposal)
    monkeypatch.setenv("AVENGINE_MP3D_ROOT", "/old/mp3d-checkout")
    assert (
        main(
            [
                "acoustics",
                "propose-visual-slots",
                "--room",
                "relative-glb-room.json",
                "--transform-profile",
                "identity_y_up",
                "--output",
                str(tmp_path / "output"),
            ]
        )
        == 0
    )
    environment = seen["environment"]
    assert isinstance(environment, dict)
    assert "AVENGINE_MP3D_ROOT" not in environment


def test_acoustics_all_legacy_root_writers_expose_explicit_mp3d_option() -> None:
    parser = build_parser()
    commands = (
        (
            "propose-visual-slots",
            [
                "--room",
                "room",
                "--transform-profile",
                "identity_y_up",
                "--output",
                "/tmp/out",
            ],
        ),
        (
            "compile-explicit-research",
            [
                "--room",
                "room",
                "--mapping",
                "mapping",
                "--materials",
                "materials",
                "--output",
                "/tmp/out",
            ],
        ),
        (
            "compile-mp3d-semantic",
            ["--room", "room", "--rules", "rules", "--output", "/tmp/out"],
        ),
        (
            "compile-mp3d-rlr-materials",
            [
                "--room",
                "room",
                "--materials",
                "materials",
                "--database-id",
                "db",
                "--source-description",
                "source",
                "--output",
                "/tmp/out",
            ],
        ),
        (
            "compile-usd-snapshot-semantic",
            ["--room", "room", "--rules", "rules", "--output", "/tmp/out"],
        ),
        (
            "compile-visual-slots-semantic",
            [
                "--room",
                "room",
                "--rules",
                "rules",
                "--transform-profile",
                "identity_y_up",
                "--output",
                "/tmp/out",
            ],
        ),
        (
            "compile-registered-scene",
            [
                "--room",
                "room",
                "--room-id",
                "room-id",
                "--room-revision",
                "r1",
                "--output",
                "/tmp/out",
            ],
        ),
    )
    for command, arguments in commands:
        parsed = parser.parse_args(["acoustics", command, *arguments])
        assert hasattr(parsed, "mp3d_root")
        assert hasattr(parsed, "runtime_root")
        assert parsed.mp3d_root is None
        assert parsed.runtime_root is None


def test_acoustics_static_writers_reject_git_backed_mp3d_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkout = tmp_path / "checkout"
    (checkout / "scene_datasets").mkdir(parents=True)
    (checkout / ".git").mkdir()
    assert (
        main(
            [
                "acoustics",
                "propose-visual-slots",
                "--room",
                "room.json",
                "--transform-profile",
                "identity_y_up",
                "--mp3d-root",
                str(checkout),
                "--output",
                str(tmp_path / "output"),
            ]
        )
        == 2
    )
    assert "must resolve outside a Git checkout" in capsys.readouterr().out


def test_acoustics_native_runtime_unavailable_is_blocked_and_flags_are_scoped(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    arguments = [
        "acoustics",
        "run-canary",
        "--request",
        "request.json",
        "--compile-evidence",
        "compile.json",
        "--runtime-prefix",
        "/external/habitat-prefix",
        "--rlr-sdk-root",
        "/external/rlr-sdk",
        "--output",
        str(tmp_path / "output"),
    ]
    parsed = build_parser().parse_args(arguments)
    assert parsed.runtime_prefix == "/external/habitat-prefix"
    assert parsed.rlr_sdk_root == "/external/rlr-sdk"
    seen: dict[str, str | None] = {}

    def unavailable(*_args, **_kwargs):
        seen["prefix"] = os.environ.get("AVENGINE_HABITAT_RUNTIME_PREFIX")
        seen["sdk"] = os.environ.get("AVENGINE_RLR_SDK_ROOT")
        raise RuntimeUnavailableError("RLR adapter is unavailable")

    monkeypatch.setattr("avengine.cli.run_material_activation_canary", unavailable)
    assert main(arguments) == 3
    assert seen == {
        "prefix": "/external/habitat-prefix",
        "sdk": "/external/rlr-sdk",
    }
    assert json.loads(capsys.readouterr().out) == {
        "error": "RLR adapter is unavailable",
        "status": "blocked",
    }


def test_acoustics_material_profile_cli_resolves_and_compiles(tmp_path: Path, capsys) -> None:
    resolved = tmp_path / "resolved"

    assert (
        main(
            [
                "acoustics",
                "resolve-materials",
                "--mapping",
                str(M3_EXAMPLE / "mapping.json"),
                "--base-materials",
                str(M3_EXAMPLE / "materials_low.json"),
                "--profile",
                str(M3_EXAMPLE / "material_profile_example.json"),
                "--output",
                str(resolved),
            ]
        )
        == 0
    )
    report = load_json(resolved / "resolution_report.json")
    database = load_json(resolved / "materials.json")
    by_key = {item["material_key"]: item for item in database["materials"]}
    assert report["status"] == "pass"
    assert report["precedence"] == [
        "base_database",
        "global_override",
        "material_override",
    ]
    assert by_key["floor_extreme_c91f"]["absorption"] == [
        0.08,
        0.24,
        0.57,
        0.69,
    ]
    assert by_key["wall_extreme_d42a"]["absorption"] == [0.2] * 4
    assert by_key["doorframe_extreme_f84c"]["absorption"] == [0.1] * 4

    copied_inputs = tmp_path / "copied_inputs"
    copied_inputs.mkdir()
    copied_names = (
        "mapping.json",
        "materials_low.json",
        "material_profile_example.json",
    )
    for filename in copied_names:
        (copied_inputs / filename).write_bytes((M3_EXAMPLE / filename).read_bytes())

    replay = tmp_path / "resolved_replay"
    assert (
        main(
            [
                "acoustics",
                "resolve-materials",
                "--mapping",
                str(copied_inputs / "mapping.json"),
                "--base-materials",
                str(copied_inputs / "materials_low.json"),
                "--profile",
                str(copied_inputs / "material_profile_example.json"),
                "--output",
                str(replay),
            ]
        )
        == 0
    )
    for filename in ("mapping.json", "materials.json", "resolution_report.json"):
        assert (resolved / filename).read_bytes() == (replay / filename).read_bytes()

    package = tmp_path / "package"
    assert (
        main(
            [
                "acoustics",
                "compile-custom",
                "--room",
                str(
                    REPOSITORY_ROOT
                    / "examples/m1/rooms/blender_custom/room_manifest.json"
                ),
                "--mapping",
                str(resolved / "mapping.json"),
                "--materials",
                str(resolved / "materials.json"),
                "--output",
                str(package),
            ]
        )
        == 0
    )
    assert main(["acoustics", "validate-package", str(package / "manifest.json")]) == 0
    leakage = tmp_path / "leakage.json"
    assert (
        main(
            [
                "acoustics",
                "inspect-mesh-leakage",
                "--package",
                str(package / "manifest.json"),
                "--origin",
                "-2.5",
                "1.55",
                "0.0",
                "--directions",
                "8",
                "--output",
                str(leakage),
            ]
        )
        == 0
    )
    leakage_report = load_json(leakage)
    assert leakage_report["status"] == "diagnostic_complete"
    assert leakage_report["source_package"]["room_id"] == ("blender_custom_two_zone_v1")
    assert leakage_report["ray_count"] == 8
    assert '"status": "pass"' in capsys.readouterr().out


def test_acoustics_material_profile_cli_refuses_unknown_selector(
    tmp_path: Path, capsys
) -> None:
    profile = load_json(M3_EXAMPLE / "material_profile_example.json")
    profile["material_overrides"][0]["selector"] = {
        "source_material_name": "MissingSlot"
    }
    profile_path = tmp_path / "profile.json"
    write_json(profile_path, profile)
    assert (
        main(
            [
                "acoustics",
                "resolve-materials",
                "--mapping",
                str(M3_EXAMPLE / "mapping.json"),
                "--base-materials",
                str(M3_EXAMPLE / "materials_low.json"),
                "--profile",
                str(profile_path),
                "--output",
                str(tmp_path / "resolved"),
            ]
        )
        == 2
    )
    assert "unknown source_material_name" in capsys.readouterr().out


def test_acoustics_current_installed_cli_requires_and_forwards_all_runtime_inputs(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    arguments = [
        "acoustics",
        "run-canary",
        "--request",
        "request.json",
        "--compile-evidence",
        "compile.json",
        "--runtime-mode",
        "current-installed",
        "--runtime-prefix",
        "/external/habitat-prefix",
        "--rlr-sdk-root",
        "/external/rlr-sdk",
        "--magnum-python-site",
        "/external/magnum-site",
        "--output",
        str(tmp_path / "output"),
    ]
    parsed = build_parser().parse_args(arguments)
    assert parsed.runtime_mode == "current-installed"
    assert parsed.magnum_python_site == "/external/magnum-site"
    seen: dict[str, object] = {}

    def unavailable(*_args, **kwargs):
        seen["kwargs"] = kwargs
        seen["magnum_env"] = os.environ.get("AVENGINE_HABITAT_MAGNUM_PYTHON_SITE")
        raise RuntimeUnavailableError("current RLR adapter is unavailable")

    monkeypatch.setattr("avengine.cli.run_material_activation_canary", unavailable)
    assert main(arguments) == 3
    assert seen == {
        "kwargs": {
            "runtime_mode": "current-installed",
            "runtime_prefix": "/external/habitat-prefix",
            "rlr_sdk_root": "/external/rlr-sdk",
            "magnum_python_site": "/external/magnum-site",
        },
        "magnum_env": "/external/magnum-site",
    }
    assert json.loads(capsys.readouterr().out) == {
        "error": "current RLR adapter is unavailable",
        "status": "blocked",
    }

    assert (
        main(
            [
                "acoustics",
                "run-canary",
                "--request",
                "request.json",
                "--compile-evidence",
                "compile.json",
                "--runtime-mode",
                "current-installed",
                "--output",
                str(tmp_path / "missing-output"),
            ]
        )
        == 2
    )
    assert "current-installed mode requires explicit" in capsys.readouterr().out
