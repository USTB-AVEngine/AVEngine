from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from test_qa_question_spec import _controlled_inputs

from avengine.qa.question_protocol import (
    COVERAGE_SCHEMA,
    DELIVERY_SCHEMA,
    QuestionProtocolError,
    _qa_case_matches_canary,
    enumerate_episode_specs,
    validate_compiled_delivery,
    validate_episode_catalog,
    validate_protocol,
)
from avengine.qa.question_spec import question_type_catalog

REPOSITORY = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPOSITORY / "examples/qa/question_spec_paper_protocol_v1.json"
EPISODE_CATALOG_PATH = (
    REPOSITORY / "examples/qa/native_question_episode_catalog_v1.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_protocol_keeps_api_indices_separate_from_0807_semantic_order() -> None:
    protocol = _load(PROTOCOL_PATH)
    validate_protocol(protocol)
    definitions = {item["question_type"]: item for item in protocol["question_types"]}
    assert definitions["appearance_to_spoken_content"]["catalog_index"] == 12
    assert definitions["appearance_to_spoken_content"]["original_0807_order"] == 2
    assert {
        item["question_type"]
        for item in protocol["question_types"]
        if item["historical_origin"] == "post_0807_extension"
    } == {"reappeared_after_occlusion", "became_clear_after_partial_occlusion"}
    assert [item["catalog_index"] for item in protocol["question_types"]] == list(
        range(1, 13)
    )


def test_protocol_rejects_historical_relabeling_and_catalog_drift() -> None:
    protocol = _load(PROTOCOL_PATH)
    changed = copy.deepcopy(protocol)
    changed["question_types"][11]["historical_origin"] = "post_0807_extension"
    changed["question_types"][11]["original_0807_order"] = None
    with pytest.raises(QuestionProtocolError, match="semantic order|extension set"):
        validate_protocol(changed)
    changed = copy.deepcopy(protocol)
    changed["question_types"][0]["name_zh"] = "漂移"
    with pytest.raises(QuestionProtocolError, match="catalog signature"):
        validate_protocol(changed)


def test_episode_catalog_has_exact_five_canaries_and_native_roles() -> None:
    protocol = _load(PROTOCOL_PATH)
    catalog = _load(EPISODE_CATALOG_PATH)
    validate_episode_catalog(catalog, protocol)
    assert len(catalog["episodes"]) == 6
    assert {
        "paper_balance_stationary_first",
        "paper_balance_right_entry",
    } <= {item["episode_key"] for item in catalog["episodes"]}
    assert {item["canary_id"] for item in catalog["visual_canaries"]} == set(
        protocol["visual_canary_contract"]["required_canary_ids"]
    )
    assert all(
        set(item["native_role_pointers"])
        == set(protocol["visual_canary_contract"]["required_native_roles"])
        for item in catalog["episodes"]
    )


def test_episode_enumerator_uses_only_observed_selectors_and_all_live_types() -> None:
    facts, _assets, _sounds, bindings = _controlled_inputs()
    specs = enumerate_episode_specs(facts, bindings)
    assert [spec["spec_id"] for spec in specs] == [
        f"QS-{index:03d}" for index in range(1, len(specs) + 1)
    ]
    assert {spec["question_type"] for spec in specs} == {
        item["question_type"] for item in question_type_catalog()
    }
    observed_sounds = set(bindings.values())
    observed_instances = {item["instance_id"] for item in facts["instances"]}
    frame_count = facts["time"]["frame_count"]
    for spec in specs:
        selectors = spec["selectors"]
        if "sound_asset_id" in selectors:
            assert selectors["sound_asset_id"] in observed_sounds
        if "sound_asset_ids" in selectors:
            assert set(selectors["sound_asset_ids"]) <= observed_sounds
        if "target_instance_id" in selectors:
            assert selectors["target_instance_id"] in observed_instances
        if "frame_index" in selectors:
            assert 0 <= selectors["frame_index"] < frame_count


def test_pixel_canary_qa_must_match_the_selected_frame_target_and_state() -> None:
    case = {
        "question_type": "occlusion_while_speaking",
        "status": "pass",
        "selectors": {"sound_asset_id": "dog_bark", "frame_index": 43},
        "answer": {"value": "out_of_view"},
        "evidence": {"instance_id": "source1", "frame_index": 43},
    }
    assert _qa_case_matches_canary(
        case,
        question_type="occlusion_while_speaking",
        target_instance_id="source1",
        frame_indices=[43],
        expected_states=["out_of_view"],
    )
    assert not _qa_case_matches_canary(
        case,
        question_type="occlusion_while_speaking",
        target_instance_id="source1",
        frame_indices=[0],
        expected_states=["visible_clear"],
    )


def test_compiled_validator_accepts_minimum_and_keeps_paper_gate_separate(
    tmp_path: Path,
) -> None:
    output = tmp_path / "delivery"
    output.mkdir()
    protocol = _load(PROTOCOL_PATH)
    (output / "protocol_snapshot.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    coverage = {
        "schema": COVERAGE_SCHEMA,
        "minimum_protocol_status": "pass",
        "visual_canary_status": "pass",
        "paper_balance_status": "gap",
        "episode_count": 1,
        "candidate_case_count": 12,
        "question_type_coverage": [
            {"minimum_status": "pass", "question_type": item["question_type"]}
            for item in question_type_catalog()
        ],
        "visual_canaries": [
            {"status": "pass", "canary_id": canary_id}
            for canary_id in protocol["visual_canary_contract"]["required_canary_ids"]
        ],
    }
    (output / "coverage.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files = []
    for name in ("coverage.json", "protocol_snapshot.json"):
        path = output / name
        files.append(
            {"path": name, "size_bytes": path.stat().st_size, "sha256": _sha(path)}
        )
    (output / "manifest.json").write_text(
        json.dumps(
            {"schema": DELIVERY_SCHEMA, "status": "pass", "files": files},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    assert validate_compiled_delivery(output)["minimum_protocol_status"] == "pass"
    with pytest.raises(QuestionProtocolError, match="paper-balance"):
        validate_compiled_delivery(output, require_paper_ready=True)
