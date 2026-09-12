from copy import deepcopy
import json

import pytest

from avengine.qa.binding_catalog import derive_binding_catalog, whole_degree_display
from avengine.qa.binding_groups import BindingGroupError


def test_display_rounding_retains_numeric_truth_and_query_time():
    original = {"question": "At 9.5 s, compare 12.7 degrees and -4.5°.",
                "truth": 12.7, "evidence": {"query_time_s": 9.5}}
    shown = whole_degree_display(original)
    assert shown["question"] == "At 9.5 s, compare 13 degrees and -5°."
    assert shown["truth"] == 12.7
    assert shown["evidence"]["query_time_s"] == 9.5
    assert original["question"].endswith("-4.5°.")


def test_structure_only_groups_cannot_produce_catalog(tmp_path):
    source = tmp_path/"groups.json"
    source.write_text(json.dumps({"validation": "structure_only", "groups": []}))
    with pytest.raises(BindingGroupError, match="media-checked"):
        derive_binding_catalog([source], output=tmp_path/"out", qa_sampling={})


def test_clipped_core_observation_cannot_supply_full_catalog(tmp_path):
    source = tmp_path/"groups.json"
    source.write_text(json.dumps({"validation": "media_checked", "groups": [{
        "group_id": "g", "members": [{"member_id": "m", "media_clock": {
            "exported": {"status": "pass", "observation_cutoff_s": 9}}}]}]}))
    with pytest.raises(BindingGroupError, match="full-duration"):
        derive_binding_catalog([source], output=tmp_path/"out", qa_sampling={})


def test_episode_catalog_request_is_validated(tmp_path) -> None:
    from avengine.qa.binding_catalog import EPISODE_SPEC_SCHEMA, _episode_specs
    from avengine.qa.binding_groups import BindingGroupError

    with pytest.raises(BindingGroupError, match="has no episodes"):
        _episode_specs([])

    with pytest.raises(BindingGroupError, match="is missing"):
        _episode_specs([{"episode_id": "a"}])

    with pytest.raises(BindingGroupError, match="needs media.video_path"):
        _episode_specs(
            [
                {
                    "episode_id": "a",
                    "facts_path": "facts.json",
                    "room_family": "mp3d",
                    "world_id": "w",
                    "media": {"audio_path": "a.wav"},
                }
            ]
        )

    entry = {
        "episode_id": "a",
        "facts_path": "facts.json",
        "room_family": "mp3d",
        "world_id": "w",
        "media": {"video_path": "v.mp4", "audio_path": "a.wav"},
    }
    with pytest.raises(BindingGroupError, match="duplicate episode_id"):
        _episode_specs([entry, dict(entry)])

    request = tmp_path / "request.json"
    request.write_text(
        json.dumps({"schema": "wrong_schema_v1", "episodes": [entry]}), encoding="utf-8"
    )
    with pytest.raises(BindingGroupError, match="schema must be"):
        _episode_specs(request)

    request.write_text(
        json.dumps({"schema": EPISODE_SPEC_SCHEMA, "episodes": [entry]}), encoding="utf-8"
    )
    (tmp_path / "facts.json").write_text("{}", encoding="utf-8")
    (tmp_path / "v.mp4").write_bytes(b"video")
    (tmp_path / "a.wav").write_bytes(b"audio")
    normalised = _episode_specs(request)
    assert len(normalised) == 1
    assert normalised[0]["facts_path"] == (tmp_path / "facts.json").resolve()
    assert normalised[0]["video_path"] == (tmp_path / "v.mp4").resolve()


def test_episode_catalog_refuses_a_non_pass_or_wrong_schema_facts(tmp_path) -> None:
    from avengine.qa.binding_catalog import EPISODE_SPEC_SCHEMA, derive_episode_catalog
    from avengine.qa.binding_groups import BindingGroupError

    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps({"schema": "other_v1", "status": "pass"}), encoding="utf-8")
    (tmp_path / "v.mp4").write_bytes(b"video")
    (tmp_path / "a.wav").write_bytes(b"audio")
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "schema": EPISODE_SPEC_SCHEMA,
                "episodes": [
                    {
                        "episode_id": "a",
                        "facts_path": "facts.json",
                        "room_family": "mp3d",
                        "world_id": "w",
                        "media": {"video_path": "v.mp4", "audio_path": "a.wav"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(BindingGroupError, match="facts schema is not"):
        derive_episode_catalog(
            request,
            output=tmp_path / "out_a",
            qa_sampling={"time_display_precision": 0},
        )

    facts.write_text(
        json.dumps(
            {"schema": "avengine_qa_unified_episode_facts_v1", "status": "blocked"}
        ),
        encoding="utf-8",
    )
    with pytest.raises(BindingGroupError, match="facts status is"):
        derive_episode_catalog(
            request,
            output=tmp_path / "out_b",
            qa_sampling={"time_display_precision": 0},
        )
