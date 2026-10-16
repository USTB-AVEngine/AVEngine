from __future__ import annotations

import numpy as np
import pytest

from avengine.qa.intermittent import (
    EDGE_MARGIN_SAMPLES,
    MAX_EVENT_SAMPLES,
    MIN_EVENT_SAMPLES,
    MIN_GAP_SAMPLES,
    SAMPLE_COUNT,
    QAIntermittentError,
    event_records,
    frame_window,
    gating_envelope,
    plan_slot_windows,
    validate_windows,
)


def test_planner_is_deterministic_and_respects_constraints() -> None:
    for episode_index in range(25):
        episode_id = f"episode_{episode_index:04d}"
        for slot_id in ("source1", "source2"):
            windows = plan_slot_windows(
                seed="s", episode_id=episode_id, slot_id=slot_id
            )
            again = plan_slot_windows(
                seed="s", episode_id=episode_id, slot_id=slot_id
            )
            assert windows == again
            assert 2 <= len(windows) <= 3
            previous_end = None
            for start, end in windows:
                assert EDGE_MARGIN_SAMPLES <= start < end
                assert end <= SAMPLE_COUNT - EDGE_MARGIN_SAMPLES
                assert MIN_EVENT_SAMPLES <= end - start <= MAX_EVENT_SAMPLES
                if previous_end is not None:
                    assert start - previous_end >= MIN_GAP_SAMPLES
                previous_end = end
    different = plan_slot_windows(seed="s2", episode_id="episode_0000", slot_id="source1")
    baseline = plan_slot_windows(seed="s", episode_id="episode_0000", slot_id="source1")
    assert different != baseline


def test_gating_envelope_shape() -> None:
    windows = [(8000, 20000), (40000, 52000)]
    envelope = gating_envelope(windows, fade_samples=80)
    assert envelope.shape == (SAMPLE_COUNT,)
    assert float(envelope[:8000].max()) == 0.0
    assert float(envelope[20000:40000].max()) == 0.0
    assert float(envelope[52000:].max()) == 0.0
    core = envelope[8080:20000 - 80]
    assert float(core.min()) == 1.0
    fade_in = envelope[8000:8080]
    assert np.all(np.diff(fade_in) > 0)
    assert 0.0 < float(fade_in[0]) < 0.05
    fade_out = envelope[20000 - 80 : 20000]
    assert np.all(np.diff(fade_out) < 0)


def test_event_records_use_audio_program_vocabulary() -> None:
    records = event_records(slot_id="source1", windows=[(16000, 28800)])
    (record,) = records
    assert record["event_id"] == "source1_event_000"
    assert record["start_tick"] == 48000
    assert record["end_tick_exclusive"] == 86400
    assert record["source_end_sample_exclusive"] - record["source_start_sample"] == (
        record["end_sample_exclusive"] - record["start_sample"]
    )
    assert frame_window(record["start_tick"], record["end_tick_exclusive"]) == (15, 27)


def test_validate_windows_rejects_bad_plans() -> None:
    with pytest.raises(QAIntermittentError):
        validate_windows([])
    with pytest.raises(QAIntermittentError):
        validate_windows([(0, 100_000)])
    with pytest.raises(QAIntermittentError):
        validate_windows([(1000, 1100)], fade_samples=80)
    with pytest.raises(QAIntermittentError):
        validate_windows([(8000, 20000), (20100, 30000)])
    with pytest.raises(QAIntermittentError):
        frame_window(10, 10)
