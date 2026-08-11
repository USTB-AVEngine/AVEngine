from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
TOOL_PATH = REPOSITORY / "tools/qa/materialize_strict_two_human_dynamic_canary.py"
SPEC = importlib.util.spec_from_file_location("dynamic_materializer", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def test_audio_program_has_exact_declared_activity_window(tmp_path: Path) -> None:
    TOOL._copy_audio_contracts(TOOL.BASE_AUDIO, tmp_path, "dynamic_test_episode")
    result = TOOL._validate_audio_contracts(tmp_path)
    program = json.loads(
        (tmp_path / "controlled_audio_program/audio_program.json").read_text()
    )

    assert result["status"] == "pass"
    assert result["speech_frame_window_inclusive"] == [7, 31]
    assert result["dry_bus_activity_checks"] == {
        "frame_6_silent": True,
        "frame_7_active": True,
        "frame_31_active": True,
        "frame_32_silent": True,
        "source2_all_zero": True,
    }
    assert program["events"][0]["start_sample"] == 7595
    assert program["events"][0]["end_sample_exclusive"] == 33221


def test_materializer_publishes_only_failure_receipt_on_error(tmp_path: Path) -> None:
    output = tmp_path / "failed_materialization"
    with pytest.raises(FileNotFoundError):
        TOOL.materialize(
            preflight_path=tmp_path / "missing_preflight.json",
            canary_index=1,
            base_suite_path=tmp_path / "missing_suite.json",
            audio_template=tmp_path / "missing_audio",
            output=output,
        )

    assert sorted(path.name for path in output.iterdir()) == ["FAILED.json"]
    failure = json.loads((output / "FAILED.json").read_text())
    assert failure["status"] == "failed"
    assert failure["formal"] is False
    assert failure["qualification_claim"] is False
    assert not list(tmp_path.glob(".failed_materialization.staging.*"))
