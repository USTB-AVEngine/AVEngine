"""Tests for Gate-B gold and representative precert rules."""

from __future__ import annotations

import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

from finalize_qa_v3_gateb_precheck import pixel_case  # noqa: E402
from recompute_qa_v3_gateb_gold import compute  # noqa: E402


def test_event_count_gateb_preserves_gold():
    fact = {
        "mcq": {"truth_option": 4},
        "open": {"truth_value": 4},
    }
    program = {"events": [{}, {}, {}, {}]}
    result = compute(
        "card15b", {"id": "card15b"}, fact,
        {}, {}, program, {})
    assert result["status"] == "pass"
    assert result["gateb_gold"] == 4


def test_card15a_gateb_pixel_rejects_gold_outside_main_options(tmp_path):
    pixel = {
        "per_instance": {
            "source1": {"frames": [{"frame_index": 30, "state": "visible_clear"}]},
            "source2": {"frames": [{"frame_index": 30, "state": "visible_clear"}]},
            "source3": {"frames": [{"frame_index": 30, "state": "visible_clear"}]},
            "source4": {"frames": [{"frame_index": 30, "state": "out_of_view"}]},
        }
    }
    fact = {
        "open": {"truth_value": [4, 2]},
        "mcq": {"options_space": [[4, 1], [4, 2], [4, 3], [4, 4]]},
    }
    pixel_path = tmp_path / "pixel.json"
    fact_path = tmp_path / "fact.json"
    pixel_path.write_text(json.dumps(pixel))
    fact_path.write_text(json.dumps(fact))
    result = pixel_case(
        "card15a",
        {"pixel_truth": str(pixel_path), "main_fact": str(fact_path)})
    assert result["status"] == "reject"
    assert result["gateb_gold"] == [3, 2]
    assert result["rejection_reasons"] == [
        "gateb_gold_outside_main_mcq_option_space"]
