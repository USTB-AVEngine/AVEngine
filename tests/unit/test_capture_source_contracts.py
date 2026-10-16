from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from avengine.contracts.json_io import canonical_json_sha256, load_json
from avengine.capture.source_contracts import (
    ALL_FLAG_IDS,
    PAIR_FLAG_IDS,
    SOURCE_FLAG_IDS,
    SourceContractError,
    bind_source_manifest_hashes,
    load_source_manifest,
    sample_boundary,
    validate_source_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "schemas" / "m5_1_source_manifest_v1.schema.json"
EXAMPLE = (
    ROOT
    / "examples"
    / "capture"
    / "legacy_apartment"
    / "source_manifest.json"
)


def _example() -> dict:
    return load_json(EXAMPLE)


def _bind_outer(value: dict) -> None:
    value.pop("manifest_content_sha256", None)
    value["manifest_content_sha256"] = canonical_json_sha256(value)


def _active_event(source: dict, frame_index: int) -> str | None:
    active = [
        event["event_id"]
        for event in source["event_windows"]
        if event["start_frame"]
        <= frame_index
        < event["end_frame_exclusive"]
    ]
    assert len(active) <= 1
    return active[0] if active else None


def _make_75_frame_variant() -> dict:
    value = _example()
    value["manifest_id"] = "short_contract_variant_v1"
    value["clip"].update(
        {
            "clip_id": "short_contract_clip",
            "duration_ticks": 240000,
            "frame_count": 75,
            "sample_count": 80000,
        }
    )
    human, dog = value["sources"]
    human["trajectory"]["keyframes"] = human["trajectory"]["keyframes"][:75]
    dog["trajectory"]["keyframes"] = dog["trajectory"]["keyframes"][:75]
    human["trajectory"]["route_binding"]["point_count"] = 75
    dog["trajectory"]["route_binding"]["point_count"] = 75
    human_event = human["event_windows"][0]
    human_event.update(
        {
            "start_frame": 6,
            "end_frame_exclusive": 30,
            "start_sample": sample_boundary(6),
            "end_sample_exclusive": sample_boundary(30),
        }
    )
    human_event["audio_program"].update(
        {
            "source_start_sample": 0,
            "source_end_sample_exclusive": 36000,
            "resampled_content_sample_count": 24000,
            "event_sample_count": 25600,
            "tail_padding_samples": 1600,
        }
    )
    human["event_windows"] = [human_event]
    dog_event = dog["event_windows"][0]
    dog_event.update(
        {
            "start_frame": 15,
            "end_frame_exclusive": 25,
            "start_sample": sample_boundary(15),
            "end_sample_exclusive": sample_boundary(25),
        }
    )
    dog_event["audio_program"].update(
        {
            "event_sample_count": 10667,
            "tail_padding_samples": 5867,
        }
    )
    dog["event_windows"] = [dog_event]
    value["frame_event_state"] = [
        {
            "frame_index": frame_index,
            "pts_ticks": frame_index * 3200,
            "sample_start": sample_boundary(frame_index),
            "sample_end": sample_boundary(frame_index + 1),
            "current_event_by_source": {
                human["source_id"]: _active_event(human, frame_index),
                dog["source_id"]: _active_event(dog, frame_index),
            },
        }
        for frame_index in range(75)
    ]
    value["relationships"][0]["event_overlap_windows"] = [
        {
            "event_ids": [human_event["event_id"], dog_event["event_id"]],
            "start_frame": 15,
            "end_frame_exclusive": 25,
        }
    ]
    value["relationships"][0]["flags"]["sources_pass_each_other"].update(
        {"status": "absent", "value": False}
    )
    value["clip_flags"]["sources_pass_each_other"].update(
        {"status": "absent", "value": False}
    )
    return bind_source_manifest_hashes(value)


def test_schema_is_valid_draft_2020_12() -> None:
    Draft202012Validator.check_schema(load_json(SCHEMA))


def test_checked_in_18_second_example_is_strictly_valid() -> None:
    value = load_source_manifest(EXAMPLE)
    assert value["research_only"] is True
    assert value["qualification_claim"] is False
    assert value["dataset_admission"] == "not_requested"
    assert value["clip"]["frame_count"] == 270
    assert value["clip"]["duration_ticks"] == 864000
    assert value["clip"]["sample_count"] == 288000
    assert len(value["frame_event_state"]) == 270
    assert validate_source_manifest(value) == []


def test_example_has_rocketbox_adult_male_and_beagle_taxonomies() -> None:
    human, dog = _example()["sources"]
    assert human["asset_class"] == "human"
    assert human["human_demographic"]["age_group"] == "adult"
    assert human["human_demographic"]["sex_or_gender_label"] == "male"
    assert human["voice_taxonomy"]["vocalization_type"] == "speech"
    assert human["emitter"]["semantic_anchor_id"] == "mouth"
    assert human["emitter"]["link_name"] == "Bip01 MJaw"
    assert (
        human["emitter"]["position_authority"]
        == "route_root_plus_nominal_emitter_offset"
    )
    assert human["provenance"]["visual_asset"]["origin"] == "rocketbox"

    assert dog["asset_class"] == "animal"
    assert dog["animal_identity"]["species_common_name"] == "domestic dog"
    assert dog["animal_identity"]["breed"] == "Beagle"
    assert dog["call_taxonomy"]["call_type"] == "bark"
    assert dog["emitter"]["semantic_anchor_id"] == "muzzle"
    assert dog["emitter"]["link_name"] == "beagle Xtra Mouth"


def test_all_legacy_flags_keep_scope_state_reason_and_evidence() -> None:
    value = _example()
    assert tuple(value["sources"][0]["flags"]) == SOURCE_FLAG_IDS
    assert tuple(value["sources"][1]["flags"]) == SOURCE_FLAG_IDS
    assert tuple(value["relationships"][0]["flags"]) == PAIR_FLAG_IDS
    assert tuple(value["clip_flags"]) == ALL_FLAG_IDS

    for source in value["sources"]:
        for assessment in source["flags"].values():
            assert assessment["scope"] == "per_source"
            assert assessment["status"] in {"present", "absent", "not_evaluated"}
            assert "value" in assessment
            assert assessment["reason"]
            assert assessment["evidence"]
    pair_flag = value["relationships"][0]["flags"]["sources_pass_each_other"]
    assert pair_flag["scope"] == "pairwise"
    assert all(flag["scope"] == "clip" for flag in value["clip_flags"].values())
    assert value["clip_flags"]["occluded_by_wall"]["status"] == "not_evaluated"
    assert value["clip_flags"]["occluded_by_wall"]["value"] is None
    assert value["clip_flags"]["stationary"]["status"] == "absent"
    assert value["clip_flags"]["steady_walk"]["status"] == "present"
    assert value["clip_flags"]["sources_pass_each_other"]["status"] == "present"


def test_route_and_real_audio_provenance_are_hash_bound_without_placeholders() -> None:
    value = _example()
    route_sha = "703a7a60fead9a6b489d73071ae5e1cd160e6ca4bef91fb35de47691509be92e"
    for source in value["sources"]:
        trajectory = source["trajectory"]
        assert len(trajectory["keyframes"]) == 270
        assert trajectory["route_binding"]["authority_file_sha256"] == route_sha
        assert source["emitter"]["path_sha256"] == trajectory[
            "trajectory_content_sha256"
        ]
        assert "0" * 64 not in {
            trajectory["trajectory_content_sha256"],
            source["emitter"]["path_sha256"],
        }

    human_audio = value["sources"][0]["provenance"]["audio_assets"][0]
    assert human_audio["sha256"] == (
        "ea738922e4f4e8a0cd30ecfd1b4ebf82296a83dfa55bb12456d156b5e787d055"
    )
    assert human_audio["rights_status"] == "licensed"
    assert human_audio["rights_evidence_sha256"] == (
        "70279f4c750c20909fd1e2ba9cdc8ab379b7229aa47ed25ead16aa15af21c385"
    )
    human_event = value["sources"][0]["event_windows"][0]
    assert (human_event["start_frame"], human_event["end_frame_exclusive"]) == (
        75,
        171,
    )
    assert human_event["audio_program"]["tail_padding_samples"] == 963

    dog_audio = value["sources"][1]["provenance"]["audio_assets"][0]
    assert dog_audio["sha256"] == (
        "12d9b3a2c9cd81852ddeb76d1abeef41ef623868b6731ff91ed511d474d2c634"
    )
    assert dog_audio["rights_status"] == "unresolved_item_level_review"
    assert dog_audio["rights_evidence_sha256"] == (
        "7b7d2455073d73e02ab055fc4399befc1b6a615b451166568a377287cc05a27b"
    )
    assert [
        (event["start_frame"], event["end_frame_exclusive"])
        for event in value["sources"][1]["event_windows"]
    ] == [(90, 95), (120, 125), (150, 155)]
    assert all(
        event["audio_program"]["source_start_sample"] == 3200
        and event["audio_program"]["source_end_sample_exclusive"] == 8000
        for event in value["sources"][1]["event_windows"]
    )


def test_frame_state_is_authoritative_for_current_events_and_overlap() -> None:
    value = _example()
    human, dog = value["sources"]
    states = value["frame_event_state"]
    assert states[0]["current_event_by_source"] == {
        human["source_id"]: None,
        dog["source_id"]: None,
    }
    assert states[90]["current_event_by_source"] == {
        human["source_id"]: "event_human_speech_001",
        dog["source_id"]: "event_beagle_bark_001",
    }
    assert states[269]["current_event_by_source"] == {
        human["source_id"]: None,
        dog["source_id"]: None,
    }
    assert len(value["relationships"][0]["event_overlap_windows"]) == 3
    for frame_index, state in enumerate(states):
        assert state["current_event_by_source"] == {
            source["source_id"]: _active_event(source, frame_index)
            for source in value["sources"]
        }


def test_exact_sample_boundaries_cover_18_seconds_without_gap() -> None:
    assert sample_boundary(0) == 0
    assert sample_boundary(1) == 1067
    assert sample_boundary(2) == 2133
    assert sample_boundary(270) == 288000
    states = _example()["frame_event_state"]
    assert all(
        left["sample_end"] == right["sample_start"]
        for left, right in zip(states, states[1:])
    )
    with pytest.raises(ValueError):
        sample_boundary(-1)


def test_contract_also_supports_the_short_75_frame_m5_timebase() -> None:
    value = _make_75_frame_variant()
    assert len(value["frame_event_state"]) == 75
    assert value["frame_event_state"][-1]["sample_end"] == 80000
    assert validate_source_manifest(value) == []


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["sources"][0]["event_windows"][0].__setitem__(
                "start_sample",
                value["sources"][0]["event_windows"][0]["start_sample"] + 1,
            ),
            "start_sample does not match exact boundary",
        ),
        (
            lambda value: value["frame_event_state"][90][
                "current_event_by_source"
            ].__setitem__("source1", None),
            "current_event_by_source does not match event windows",
        ),
        (
            lambda value: value["relationships"][0][
                "event_overlap_windows"
            ].pop(),
            "event_overlap_windows do not match source events",
        ),
        (
            lambda value: value["sources"][0]["event_windows"][0].__setitem__(
                "dry_audio_asset_id", "missing_audio_asset"
            ),
            "dry_audio_asset_id must resolve exactly once",
        ),
        (
            lambda value: value["sources"][0]["event_windows"][0][
                "audio_program"
            ].__setitem__("tail_padding_samples", 964),
            "tail padding does not close window",
        ),
        (
            lambda value: value["sources"][0]["trajectory"][
                "route_binding"
            ].__setitem__("authority_file_sha256", "f" * 64),
            "authority_file_sha256 differs from the route file",
        ),
        (
            lambda value: value["relationships"][0]["flags"][
                "sources_pass_each_other"
            ].__setitem__("scope", "per_source"),
            "scope must be 'pairwise'",
        ),
        (
            lambda value: value["sources"][0]["flags"]["steady_walk"].__setitem__(
                "value", False
            ),
            "value must match status 'present'",
        ),
        (
            lambda value: value["clip_flags"]["stationary"].update(
                {"status": "present", "value": True}
            ),
            "must equal the legacy OR aggregate",
        ),
        (
            lambda value: value["sources"][1]["flags"][
                "occluded_by_wall"
            ]["evidence"][0].__setitem__("kind", "metric"),
            "not_evaluated requires missing_dependency",
        ),
    ],
)
def test_cross_field_tampering_fails_closed(mutate, message: str) -> None:
    value = deepcopy(_example())
    mutate(value)
    _bind_outer(value)
    assert any(message in error for error in validate_source_manifest(value))


def test_overlapping_events_are_rejected() -> None:
    value = _example()
    second = value["sources"][1]["event_windows"][1]
    second["start_frame"] = 92
    second["start_sample"] = sample_boundary(92)
    _bind_outer(value)
    assert any("event_windows must not overlap" in error for error in validate_source_manifest(value))


def test_trajectory_hash_and_clip_endpoint_are_both_authoritative() -> None:
    value = _example()
    value["sources"][1]["trajectory"]["keyframes"][-1]["frame_index"] = 268
    value = bind_source_manifest_hashes(value)
    assert any("must end at clip frame 269" in error for error in validate_source_manifest(value))

    value = _example()
    value["sources"][1]["trajectory"]["keyframes"][1]["position_m"][0] += 0.1
    _bind_outer(value)
    assert any("trajectory_content_sha256" in error for error in validate_source_manifest(value))


def test_outer_hash_and_loader_fail_on_tamper() -> None:
    value = _example()
    value["claim_boundary"] += " tampered"
    assert any("manifest_content_sha256" in error for error in validate_source_manifest(value))

    bad_path = EXAMPLE.parent / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        load_source_manifest(bad_path)


def test_binder_is_pure_and_returns_a_valid_detached_copy() -> None:
    original = _make_75_frame_variant()
    stale = deepcopy(original)
    stale["sources"][0]["trajectory"]["trajectory_content_sha256"] = "0" * 64
    stale["sources"][0]["emitter"]["path_sha256"] = "0" * 64
    stale["manifest_content_sha256"] = "0" * 64
    before = deepcopy(stale)
    rebound = bind_source_manifest_hashes(stale)
    assert stale == before
    assert rebound is not stale
    assert validate_source_manifest(rebound) == []


def test_source_contract_error_preserves_individual_errors() -> None:
    error = SourceContractError(["first", "second"])
    assert error.errors == ("first", "second")
    assert str(error) == "first; second"
