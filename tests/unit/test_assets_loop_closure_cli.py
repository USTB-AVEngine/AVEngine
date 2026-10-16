from __future__ import annotations

from pathlib import Path

import pytest

from tools.assets import append_loop_closure as cli


def _arguments(source: Path, output: Path, report: Path) -> list[str]:
    return [
        "--input",
        str(source),
        "--output",
        str(output),
        "--report",
        str(report),
    ]


def _fake_compile(source: Path, output: Path) -> dict[str, object]:
    output.write_bytes(b"loop-closed")
    return {
        "schema": cli.SCHEMA,
        "status": "pass",
        "qualification_state": "research_candidate",
        "qualification_claim": False,
        "source": {"path": str(source)},
        "output": {"path": str(output)},
    }


@pytest.mark.parametrize("occupied", ["output", "report"])
@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_cli_preflights_both_paired_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    occupied: str,
    kind: str,
) -> None:
    source = tmp_path / "source.glb"
    source.write_bytes(b"source")
    output = tmp_path / "closed.glb"
    report = tmp_path / "report.json"
    path = output if occupied == "output" else report
    if kind == "file":
        path.write_bytes(b"sentinel")
    else:
        path.symlink_to(tmp_path / f"dangling-{occupied}")
    monkeypatch.setattr(cli, "compile_loop_closure", _fake_compile)

    with pytest.raises(SystemExit):
        cli.main(_arguments(source, output, report))

    counterpart = report if occupied == "output" else output
    assert not counterpart.exists()
    assert not counterpart.is_symlink()
    if kind == "file":
        assert path.read_bytes() == b"sentinel"
    else:
        assert path.is_symlink()


def test_cli_rolls_back_output_when_report_creation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.glb"
    source.write_bytes(b"source")
    output = tmp_path / "closed.glb"
    report = tmp_path / "report.json"
    monkeypatch.setattr(cli, "compile_loop_closure", _fake_compile)

    def fail_report(_path: Path, _payload: bytes) -> None:
        raise OSError("injected report failure")

    monkeypatch.setattr(cli, "_write_exclusive", fail_report)
    with pytest.raises(SystemExit):
        cli.main(_arguments(source, output, report))

    assert not output.exists()
    assert not report.exists()
