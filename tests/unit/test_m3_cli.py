from __future__ import annotations

from pathlib import Path

from avengine.contracts.json_io import load_json, write_json

from avengine.cli import build_parser, main


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REQUEST = REPOSITORY_ROOT / "examples/m3/blender_custom/canary_request.json"
M3_EXAMPLE = REPOSITORY_ROOT / "examples/m3/blender_custom"


def test_m3_compile_validate_and_verify_cli(tmp_path: Path, capsys) -> None:
    output = tmp_path / "canary"

    assert (
        main(
            [
                "m3",
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
                "m3",
                "validate-package",
                str(output / "low_absorption/manifest.json"),
            ]
        )
        == 0
    )
    assert main(["m3", "verify-compile", str(output / "compile_evidence.json")]) == 0
    rendered = capsys.readouterr().out
    assert '"status": "pass"' in rendered


def test_m3_research_commands_are_exposed_as_separate_cli_paths() -> None:
    parser = build_parser()

    proposed = parser.parse_args(
        [
            "m3",
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
            "m3",
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
    canary = parser.parse_args(
        [
            "m3",
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
        ["m3", "verify-canary", "/tmp/canary/canary_evidence.json"]
    )

    assert proposed.m3_command == "propose-visual-slots"
    assert compiled.m3_command == "compile-explicit-research"
    assert canary.m3_command == "run-canary"
    assert verified.m3_command == "verify-canary"


def test_m3_material_profile_cli_resolves_and_compiles(tmp_path: Path, capsys) -> None:
    resolved = tmp_path / "resolved"

    assert (
        main(
            [
                "m3",
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
                "m3",
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
                "m3",
                "compile-custom",
                "--room",
                str(REPOSITORY_ROOT / "examples/m1/rooms/blender_custom/room_manifest.json"),
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
    assert main(["m3", "validate-package", str(package / "manifest.json")]) == 0
    assert '"status": "pass"' in capsys.readouterr().out


def test_m3_material_profile_cli_refuses_unknown_selector(
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
                "m3",
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
