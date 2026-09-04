from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from avengine.contracts.json_io import load_json
from avengine.timeline.audio_program import (
    AudioProgramError,
    bind_audio_program_hash,
    compile_audio_program_variant,
    compile_audio_program,
    load_audio_program,
    materialize_audio_program_variant,
    validate_audio_program,
)
from avengine.registry.sources import load_sound_asset_registry, load_source_endpoint_registry


ROOT = Path(__file__).resolve().parents[2]
REGISTRIES = ROOT / "examples" / "registry" / "registries"


def _inputs() -> tuple[dict, dict, dict]:
    endpoints = load_source_endpoint_registry(REGISTRIES / "source_endpoints_v1.json")
    sounds = load_sound_asset_registry(REGISTRIES / "sound_assets_v1.json")
    program = load_json(REGISTRIES / "one_active_of_n_program_v1.json")
    return endpoints, sounds, program


def test_one_active_of_n_compiles_exact_half_open_timeline_and_silent_endpoint() -> (
    None
):
    endpoints, sounds, _ = _inputs()
    program = load_audio_program(
        REGISTRIES / "one_active_of_n_program_v1.json",
        source_endpoint_registry=endpoints,
        sound_asset_registry=sounds,
    )
    compiled = compile_audio_program(
        program,
        source_endpoint_registry=endpoints,
        sound_asset_registry=sounds,
    )
    assert compiled.active_source_endpoint_ids == ("beagle_0_muzzle",)
    assert compiled.silent_source_endpoint_ids == ("beagle_1_muzzle",)
    assert len(compiled.events) == 6
    event = compiled.events[0]
    assert (event.start_tick, event.end_tick_exclusive) == (19200, 33600)
    assert (event.start_sample, event.end_sample_exclusive) == (6400, 11200)
    assert (event.source_start_sample, event.source_end_sample_exclusive) == (
        3200,
        8000,
    )
    assert event.linear_gain == 0.18
    assert event.fade_samples == 80
    assert compiled.current_event_by_source(5) == {
        "beagle_0_muzzle": None,
        "beagle_1_muzzle": None,
    }
    assert (
        compiled.current_event_by_source(6)["beagle_0_muzzle"]
        == "m5_source0_simultaneous0"
    )
    assert (
        compiled.current_event_by_source(10)["beagle_0_muzzle"]
        == "m5_source0_simultaneous0"
    )
    assert compiled.current_event_by_source(11)["beagle_0_muzzle"] is None


def test_program_reuses_all_six_authoritative_m5_source0_sample_windows() -> None:
    _, _, program = _inputs()
    upstream = load_json(
        ROOT
        / "examples"
        / "timeline"
        / "blender_custom"
        / "two_dog_simultaneous_counterfactual_request.json"
    )
    expected = [
        (item["start_sample"], item["end_sample"])
        for item in upstream["audio_program"]["simultaneous_windows"]
    ]
    actual = [
        (item["start_sample"], item["end_sample_exclusive"])
        for item in program["events"]
    ]
    assert actual == expected
    assert all(
        (item["source_start_sample"], item["source_end_sample_exclusive"])
        == (3200, 8000)
        for item in program["events"]
    )
    assert program["source_program_provenance"]["upstream_source_id"] == "source0"


def test_silent_negative_retains_both_registered_endpoint_capabilities() -> None:
    endpoints, sounds, program = _inputs()
    program["program_id"] = "beagle_silent_negative_v1"
    program["mode"] = "silent_negative"
    program["events"] = []
    program = bind_audio_program_hash(program)
    assert (
        validate_audio_program(
            program,
            source_endpoint_registry=endpoints,
            sound_asset_registry=sounds,
        )
        == []
    )
    compiled = compile_audio_program(
        program,
        source_endpoint_registry=endpoints,
        sound_asset_registry=sounds,
    )
    assert compiled.active_source_endpoint_ids == ()
    assert compiled.silent_source_endpoint_ids == (
        "beagle_0_muzzle",
        "beagle_1_muzzle",
    )


def test_one_active_of_n_rejects_events_on_two_endpoints() -> None:
    endpoints, sounds, program = _inputs()
    second = deepcopy(program["events"][0])
    second["event_id"] = "bark_event_1"
    second["source_endpoint_id"] = "beagle_1_muzzle"
    program["events"].append(second)
    program = bind_audio_program_hash(program)
    errors = validate_audio_program(
        program,
        source_endpoint_registry=endpoints,
        sound_asset_registry=sounds,
    )
    assert "one_active_of_n requires events on exactly one candidate endpoint" in errors


def test_audio_program_rejects_non_authoritative_tick_or_sample_boundaries() -> None:
    endpoints, sounds, program = _inputs()
    program["events"][0]["start_tick"] += 1
    program = bind_audio_program_hash(program)
    errors = validate_audio_program(
        program,
        source_endpoint_registry=endpoints,
        sound_asset_registry=sounds,
    )
    assert "events[0].start_tick must equal 19200" in errors


def test_audio_program_rejects_sound_class_not_supported_by_endpoint() -> None:
    endpoints, sounds, program = _inputs()
    sounds = deepcopy(sounds)
    dog_sound = next(
        item
        for item in sounds["sound_assets"]
        if item["sound_asset_id"] == "dog_beagle_v2_scheduled_dry"
    )
    dog_sound["semantic_sound_class"] = "speech"
    from avengine.registry.registry import bind_content_hash

    sounds = bind_content_hash(sounds)
    errors = validate_audio_program(
        program,
        source_endpoint_registry=endpoints,
        sound_asset_registry=sounds,
    )
    assert any("sound class 'speech' is not allowed" in item for item in errors)


def test_audio_program_rejects_sound_event_usage_not_permitted_by_asset() -> None:
    endpoints, sounds, program = _inputs()
    sounds = deepcopy(sounds)
    dog_sound = next(
        item
        for item in sounds["sound_assets"]
        if item["sound_asset_id"] == "dog_beagle_v2_scheduled_dry"
    )
    dog_sound["permitted_event_usage"] = ["simultaneous_subset"]
    from avengine.registry.registry import bind_content_hash

    sounds = bind_content_hash(sounds)
    errors = validate_audio_program(
        program,
        source_endpoint_registry=endpoints,
        sound_asset_registry=sounds,
    )
    assert any(
        "does not permit AudioProgram mode 'one_active_of_n'" in item
        for item in errors
    )


def test_all_declared_audio_program_modes_have_enforced_semantics() -> None:
    endpoints, sounds, base = _inputs()

    simultaneous = deepcopy(base)
    simultaneous["program_id"] = "beagle_simultaneous_subset_v1"
    simultaneous["mode"] = "simultaneous_subset"
    overlap = deepcopy(simultaneous["events"][0])
    overlap["event_id"] = "m5_source1_simultaneous0"
    overlap["source_endpoint_id"] = "beagle_1_muzzle"
    simultaneous["events"].append(overlap)
    simultaneous["events"].sort(
        key=lambda item: (
            item["start_sample"],
            item["source_endpoint_id"],
            item["event_id"],
        )
    )
    simultaneous = bind_audio_program_hash(simultaneous)
    assert (
        validate_audio_program(
            simultaneous,
            source_endpoint_registry=endpoints,
            sound_asset_registry=sounds,
        )
        == []
    )

    sequential = deepcopy(base)
    sequential["program_id"] = "beagle_sequential_sources_v1"
    sequential["mode"] = "sequential_sources"
    sequential["events"][1]["source_endpoint_id"] = "beagle_1_muzzle"
    sequential = bind_audio_program_hash(sequential)
    assert (
        validate_audio_program(
            sequential,
            source_endpoint_registry=endpoints,
            sound_asset_registry=sounds,
        )
        == []
    )

    intermittent = deepcopy(base)
    intermittent["program_id"] = "beagle_intermittent_events_v1"
    intermittent["mode"] = "intermittent_events"
    intermittent = bind_audio_program_hash(intermittent)
    assert (
        validate_audio_program(
            intermittent,
            source_endpoint_registry=endpoints,
            sound_asset_registry=sounds,
        )
        == []
    )


def test_counterfactual_route_swap_changes_only_declared_endpoint_routes() -> None:
    endpoints, sounds, program = _inputs()
    program["program_id"] = "beagle_counterfactual_route_swap_v1"
    program["mode"] = "counterfactual_route_swap"
    program["counterfactual"] = {
        "operation": "swap_source_endpoint_routing",
        "variants": ["A", "B"],
        "reference_variant": "A",
        "mapped_variant": "B",
        "endpoint_permutation": {
            "beagle_0_muzzle": "beagle_1_muzzle",
            "beagle_1_muzzle": "beagle_0_muzzle",
        },
        "allowed_changed_fields": ["events[*].source_endpoint_id"],
    }
    program = bind_audio_program_hash(program)
    assert (
        validate_audio_program(
            program,
            source_endpoint_registry=endpoints,
            sound_asset_registry=sounds,
        )
        == []
    )

    variant_a = materialize_audio_program_variant(
        program,
        "A",
        source_endpoint_registry=endpoints,
        sound_asset_registry=sounds,
    )
    variant_b = materialize_audio_program_variant(
        program,
        "B",
        source_endpoint_registry=endpoints,
        sound_asset_registry=sounds,
    )
    assert variant_a == program
    assert variant_b["program_content_sha256"] != program["program_content_sha256"]
    for event_a, event_b in zip(variant_a["events"], variant_b["events"]):
        assert event_b["source_endpoint_id"] == "beagle_1_muzzle"
        assert {
            key: value for key, value in event_a.items() if key != "source_endpoint_id"
        } == {
            key: value for key, value in event_b.items() if key != "source_endpoint_id"
        }
    assert compile_audio_program_variant(
        program,
        "B",
        source_endpoint_registry=endpoints,
        sound_asset_registry=sounds,
    ).active_source_endpoint_ids == ("beagle_1_muzzle",)


def test_audio_program_modes_fail_closed_when_their_required_pattern_is_absent() -> (
    None
):
    endpoints, sounds, base = _inputs()
    for mode, expected in (
        (
            "simultaneous_subset",
            "simultaneous_subset requires overlapping events on at least two endpoints",
        ),
        (
            "sequential_sources",
            "sequential_sources requires at least two active endpoints",
        ),
    ):
        program = deepcopy(base)
        program["mode"] = mode
        program = bind_audio_program_hash(program)
        assert expected in validate_audio_program(
            program,
            source_endpoint_registry=endpoints,
            sound_asset_registry=sounds,
        )

    no_gap = deepcopy(base)
    no_gap["mode"] = "intermittent_events"
    no_gap["events"] = no_gap["events"][:1]
    no_gap = bind_audio_program_hash(no_gap)
    assert (
        "intermittent_events requires a positive silent gap between events on at least one endpoint"
        in validate_audio_program(
            no_gap,
            source_endpoint_registry=endpoints,
            sound_asset_registry=sounds,
        )
    )



def test_non_counterfactual_variant_error_is_one_message():
    base_program = _inputs()[2]
    with pytest.raises(AudioProgramError) as error:
        materialize_audio_program_variant(base_program, "not-a-variant")
    assert error.value.errors == (
        "non-counterfactual AudioProgram supports only variant 'A'",)
