from __future__ import annotations

from pathlib import Path

from avengine.cli import build_parser, main


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REQUEST = REPOSITORY_ROOT / "examples/m3/blender_custom/canary_request.json"


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
