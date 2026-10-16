from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
TOOL_PATH = REPOSITORY / "tools/qa/validate_strict_two_human_publication_plan.py"
SPEC = importlib.util.spec_from_file_location("strict8_publication_validator", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)
PLAN = REPOSITORY / "examples/qa/native_strict_two_human_publication_v1.json"


def test_strict8_publication_plan_validates_real_evidence() -> None:
    result = TOOL.validate(PLAN)
    assert result["status"] == "pass"
    assert result["counted_sparse_scene_count"] == 8
    assert result["target_side_counts"] == {"left": 4, "right": 4}
    assert result["target_identity_counts"] == {"M": 3, "F": 3, "C": 2}
    assert result["formal_scene_count"] == 0


def test_strict8_publication_rejects_unbalanced_or_counted_history(tmp_path: Path) -> None:
    value = json.loads(PLAN.read_text(encoding="utf-8"))
    value["rows"][0]["target_side"] = "left"
    invalid = tmp_path / "unbalanced.json"
    invalid.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="target side balance"):
        TOOL.validate(invalid)

    value = json.loads(PLAN.read_text(encoding="utf-8"))
    value["excluded_attempts"][0]["counted"] = True
    invalid = tmp_path / "counted_history.json"
    invalid.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="excluded history"):
        TOOL.validate(invalid)


def test_strict8_publication_rejects_weakened_target_threshold(tmp_path: Path) -> None:
    value = json.loads(PLAN.read_text(encoding="utf-8"))
    value["visibility_contract"]["target_visible_fraction_minimum"] = 0.5
    invalid = tmp_path / "weak_target.json"
    invalid.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="visibility contract"):
        TOOL.validate(invalid)

_RETAINED_TMP_WORKSPACE = Path(__file__).resolve().parents[2] / "tmp"
if not _RETAINED_TMP_WORKSPACE.exists():
    pytest.skip(
        "retained strict-two-human evidence workspace (repository tmp "
        "symlink) is not present in this checkout",
        allow_module_level=True,
    )
