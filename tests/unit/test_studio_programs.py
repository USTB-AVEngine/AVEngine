from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from avengine.studio.programs import (
    StudioProgramError,
    build_turn_taking_program,
    persist_program,
    pick_energetic_slice,
)

REPOSITORY = Path(__file__).resolve().parents[2]
ENDPOINTS = REPOSITORY / "examples/m6/registries/source_endpoints_v1.json"
SOUNDS = REPOSITORY / "examples/m6/registries/sound_assets_v1.json"


def test_pick_energetic_slice_finds_the_burst() -> None:
    samples = np.zeros(16000, dtype=np.float64)
    samples[9000:9800] = 0.9
    start, end = pick_energetic_slice(samples, 800)
    assert 8800 <= start <= 9100
    assert end - start == 800


def test_build_turn_taking_program_binds_and_validates(tmp_path: Path) -> None:
    program = build_turn_taking_program(
        program_id="studio_test_program_v1",
        candidate_source_endpoint_ids=["m6x_marker_front_speaker", "m6x_human0_mouth"],
        sound_by_endpoint={
            "m6x_marker_front_speaker": "directional_chime_v1",
            "m6x_human0_mouth": "human_speech_libritts_1594_16k_v1",
        },
        source_endpoint_registry_path=ENDPOINTS,
        sound_asset_registry_path=SOUNDS,
        repository_root=REPOSITORY,
        event_count=6,
        event_samples=4800,
    )
    assert program["program_content_sha256"]
    assert program["admission_state"] == "research"
    events = program["events"]
    assert len(events) == 6
    sources = [event["source_endpoint_id"] for event in events]
    assert all(sources[i] != sources[i + 1] for i in range(len(sources) - 1))
    spans = [(e["start_sample"], e["end_sample_exclusive"]) for e in events]
    assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))
    assert spans[-1][1] <= 80000
    for event in events:
        assert event["end_tick_exclusive"] == event["end_sample_exclusive"] * 3
        assert (
            event["source_end_sample_exclusive"] - event["source_start_sample"]
            == event["end_sample_exclusive"] - event["start_sample"]
        )

    path = persist_program(program, tmp_path / "programs")
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["program_id"] == "studio_test_program_v1"
    with pytest.raises(StudioProgramError, match="fresh/no-clobber"):
        persist_program(program, tmp_path / "programs")


def test_unknown_sound_is_rejected() -> None:
    with pytest.raises(StudioProgramError, match="unknown sound asset"):
        build_turn_taking_program(
            program_id="x",
            candidate_source_endpoint_ids=["m6x_dog0_muzzle"],
            sound_by_endpoint={"m6x_dog0_muzzle": "nope"},
            source_endpoint_registry_path=ENDPOINTS,
            sound_asset_registry_path=SOUNDS,
            repository_root=REPOSITORY,
        )
