"""Tests for Gate-B gold and representative precert rules."""

from __future__ import annotations

import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

from finalize_qa_v3_gateb_precheck import audio_pair, pixel_case  # noqa: E402
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


def _audio_root(path, payload):
    mixture = path / "audio" / "binaural" / "mixture.wav"
    mixture.parent.mkdir(parents=True)
    mixture.write_bytes(payload)
    (path / "research_receipt.json").write_text(json.dumps({
        "status": "pass", "research_only": True,
    }))
    return path


def test_canonical_appearance_audio_requires_identical_rerenders(tmp_path):
    main = _audio_root(tmp_path / "main", b"same")
    gateb = _audio_root(tmp_path / "gateb", b"same")
    result = audio_pair({
        "main_audio": str(main), "gateb_audio": str(gateb),
        "policy": "appearance_canonical_anchor_audio_must_be_identical",
    })
    assert result["rerender_mixtures_identical"] is True
    assert result["decision"] == "pass_canonical_anchor_audio_identical"
    (gateb / "audio" / "binaural" / "mixture.wav").write_bytes(b"changed")
    import pytest
    with pytest.raises(RuntimeError, match="unexpectedly changed"):
        audio_pair({
            "main_audio": str(main), "gateb_audio": str(gateb),
            "policy": "appearance_canonical_anchor_audio_must_be_identical",
        })
