from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

REPOSITORY = Path(__file__).resolve().parents[2]
TOOL_PATH = REPOSITORY / "tools/qa/publish_strict_two_human_review.py"
SPEC = importlib.util.spec_from_file_location("strict8_review_publisher", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)
PLAN = REPOSITORY / "examples/qa/native_strict_two_human_publication_v1.json"


def test_build_ledger_closes_eight_real_sparse_rows() -> None:
    ledger = TOOL.build_ledger(PLAN, TOOL.DEFAULT_MEDIA_BASE_URL)
    assert ledger["status"] == "pass"
    assert ledger["sparse_pass_count"] == 8
    assert ledger["formal_scene_count"] == 0
    assert ledger["qualification_claim"] is False
    assert [row["row_index"] for row in ledger["rows"]] == list(range(1, 9))
    assert all(row["status"] == "pass_sparse_f15" for row in ledger["rows"])
    assert ledger["rows"][6]["attempt_id"] == "v2"
    assert ledger["excluded_attempts"][0]["counted"] is False


def test_publish_verify_and_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "strict8_review"
    result = TOOL.publish(PLAN, output, TOOL.DEFAULT_MEDIA_BASE_URL)
    assert result["status"] == "pass"
    assert (output / "ledger.json").is_file()
    assert (output / "index.html").read_text(encoding="utf-8").count(
        "data-counted-row"
    ) == 8
    with Image.open(output / "contact_sheet.png") as contact_sheet:
        assert contact_sheet.size == (1040, 1384)
    with pytest.raises(FileExistsError, match="Refusing to clobber"):
        TOOL.publish(PLAN, output, TOOL.DEFAULT_MEDIA_BASE_URL)


def test_verify_rejects_claim_promotion(tmp_path: Path) -> None:
    output = tmp_path / "strict8_review"
    TOOL.publish(PLAN, output, TOOL.DEFAULT_MEDIA_BASE_URL)
    ledger_path = output / "ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["formal_scene_count"] = 8
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    with pytest.raises(RuntimeError, match="claim boundary"):
        TOOL.verify(output)

_RETAINED_TMP_WORKSPACE = Path(__file__).resolve().parents[2] / "tmp"
# Guarding on tmp/ existing was wrong: running the engine in a
# checkout creates tmp/spear_instance_*, which made this look
# mounted and sent 49 tests into a run without their data.  The
# evidence mount signature is a lead_* workspace.
if not any(_RETAINED_TMP_WORKSPACE.glob("lead_*")):
    pytest.skip(
        "no lead_* evidence workspace under the repository tmp "
        "directory, so this checkout does not carry the retained "
        "strict-two-human evidence",
        allow_module_level=True,
    )
