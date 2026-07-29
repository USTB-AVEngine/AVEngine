from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from avengine.m7.asset_bound_audio import AssetBoundAudioError
from tools.m7.render_asset_bound_binaural_batch import (
    AudioProgramSpec,
    _prepare_audio_program_variants,
    audio_program_specs,
)


ROOT = Path(__file__).resolve().parents[2]
PROGRAMS = ROOT / "examples/m6x/fixed_apartment/audio_programs"
REGISTRIES = ROOT / "examples/m6/registries"


def test_program_specs_default_to_a_and_require_aligned_variants() -> None:
    paths = (Path("first.json"), Path("second.json"))

    assert [item.variant_id for item in audio_program_specs(paths, ())] == [
        "A",
        "A",
    ]
    assert [item.variant_id for item in audio_program_specs(paths, ("A", "B"))] == [
        "A",
        "B",
    ]
    with pytest.raises(AssetBoundAudioError, match="count"):
        audio_program_specs(paths, ("A",))


def test_m7_prepares_sequential_m6_program_as_exact_slot_buses() -> None:
    prepared, library = _prepare_audio_program_variants(
        specs=(
            AudioProgramSpec(
                PROGRAMS / "m6x_s5_los_nlos_sequential_v1.json"
            ),
        ),
        source_endpoint_registry_path=REGISTRIES / "source_endpoints_v1.json",
        sound_asset_registry_path=REGISTRIES / "sound_assets_v1.json",
        endpoint_to_source_slot={
            "m6x_world_los_speaker": "source1",
            "m6x_world_nlos_speaker": "source2",
        },
        sound_audio={
            "directional_chime_v1": str(
                ROOT / "examples/m6x/assets/directional_chime_16k.wav"
            ),
            "unused_library_entry": "/not-read.wav",
        },
    )

    item = prepared[0]
    source1 = item.dry_by_source_slot["source1"]
    source2 = item.dry_by_source_slot["source2"]
    assert not np.any(source1[:4_000])
    assert np.any(source1[4_080:31_920])
    assert not np.any(source1[32_000:])
    assert not np.any(source2[:44_000])
    assert np.any(source2[44_080:71_920])
    assert not np.any(source2[72_000:])
    assert item.source_activity_summary == {
        "active_source_slots": ["source1", "source2"],
        "silent_source_slots": [],
        "active_sample_count_by_source_slot": {
            "source1": 28_000,
            "source2": 28_000,
        },
        "simultaneous_active_sample_count": 0,
        "both_sources_have_events": True,
        "both_sources_active": False,
    }
    assert item.audio_program_binding[
        "source_endpoint_to_source_slot"
    ] == {
        "m6x_world_los_speaker": "source1",
        "m6x_world_nlos_speaker": "source2",
    }
    assert (
        item.instance_record["dry_audio_assembly"][
            "assembly_content_sha256"
        ]
        == item.audio_program_binding[
            "dry_audio_assembly_content_sha256"
        ]
    )
    assert library["schema"] == "avengine_m7_m6_audio_program_dry_bus_library_v1"


def test_m7_prepares_counterfactual_b_as_exact_endpoint_and_slot_bus_swap() -> None:
    prepared, _library = _prepare_audio_program_variants(
        specs=(
            AudioProgramSpec(
                PROGRAMS / "m6x_s1_front_rear_route_swap_v1.json",
                variant_id="A",
            ),
            AudioProgramSpec(
                PROGRAMS / "m6x_s1_front_rear_route_swap_v1.json",
                variant_id="B",
            ),
        ),
        source_endpoint_registry_path=REGISTRIES / "source_endpoints_v1.json",
        sound_asset_registry_path=REGISTRIES / "sound_assets_v1.json",
        endpoint_to_source_slot={
            "m6x_marker_front_speaker": "source1",
            "m6x_marker_rear_speaker": "source2",
        },
        sound_audio={
            "directional_chime_v1": str(
                ROOT / "examples/m6x/assets/directional_chime_16k.wav"
            )
        },
    )

    variant_a = prepared[0]
    variant_b = prepared[1]
    event_a = variant_a.instance_record["materialized_audio_program"]["events"][0]
    event_b = variant_b.instance_record["materialized_audio_program"]["events"][0]
    assert event_a["source_endpoint_id"] == "m6x_marker_front_speaker"
    assert event_b["source_endpoint_id"] == "m6x_marker_rear_speaker"
    assert variant_a.audio_program_binding["variant_id"] == "A"
    assert variant_b.audio_program_binding["variant_id"] == "B"
    assert np.array_equal(
        variant_a.dry_by_source_slot["source1"],
        variant_b.dry_by_source_slot["source2"],
    )
    assert np.array_equal(
        variant_a.dry_by_source_slot["source2"],
        variant_b.dry_by_source_slot["source1"],
    )


def test_m7_prepares_silent_negative_as_two_exact_zero_slot_buses() -> None:
    prepared, _library = _prepare_audio_program_variants(
        specs=(
            AudioProgramSpec(
                PROGRAMS / "m6x_s2_silent_negative_v1.json"
            ),
        ),
        source_endpoint_registry_path=REGISTRIES / "source_endpoints_v1.json",
        sound_asset_registry_path=REGISTRIES / "sound_assets_v1.json",
        endpoint_to_source_slot={
            "m6x_dog0_muzzle": "source1",
            "m6x_human0_mouth": "source2",
        },
        sound_audio={},
    )

    item = prepared[0]
    assert all(not np.any(bus) for bus in item.dry_by_source_slot.values())
    assert item.source_activity_summary["active_source_slots"] == []
    assert item.source_activity_summary["silent_source_slots"] == [
        "source1",
        "source2",
    ]
    assert item.source_activity_summary["both_sources_active"] is False
