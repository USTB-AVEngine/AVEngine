from __future__ import annotations

from pathlib import Path
import wave

import numpy as np
import pytest

from avengine.m7.asset_bound_audio import (
    AssetBoundAudioError,
    bind_endpoint_buses_to_source_slots,
    float32_stems_and_exact_mix,
    prepare_dry_audio,
    render_asset_bound_binaural,
)
from avengine.m5.audio import raised_cosine_partition
from tools.m7.render_asset_bound_binaural_batch import variant_start_samples


def _write_pcm16(path: Path, samples: np.ndarray, sample_rate_hz: int) -> None:
    values = np.asarray(samples, dtype="<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(values.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(sample_rate_hz)
        handle.writeframes(values.tobytes())


def test_prepares_explicit_stereo_downmix_without_normalization(tmp_path: Path) -> None:
    path = tmp_path / "cat.wav"
    _write_pcm16(
        path,
        np.asarray([[16384, 32767]] * 100, dtype=np.int16),
        sample_rate_hz=16_000,
    )

    prepared = prepare_dry_audio(
        path,
        channel_policy="equal_weight_downmix",
        output_sample_count=100,
        fade_samples=0,
    )

    assert prepared.samples.shape == (100,)
    assert prepared.samples[0] == pytest.approx((0.5 + 32767 / 32768) / 2)
    assert prepared.record["input"]["source_channel_count"] == 2
    assert prepared.record["channel_policy"] == "equal_weight_downmix"
    assert prepared.record["normalization"] is False
    assert prepared.record["looping"] is False


def test_stereo_requires_an_explicit_downmix_policy(tmp_path: Path) -> None:
    path = tmp_path / "dog.wav"
    _write_pcm16(path, np.asarray([[1000, 2000]] * 100, dtype=np.int16), 16_000)

    with pytest.raises(AssetBoundAudioError, match="requires equal_weight_downmix"):
        prepare_dry_audio(
            path,
            channel_policy="require_mono",
            output_sample_count=100,
            fade_samples=0,
        )


def test_renders_two_active_sources_as_exact_binaural_stem_sum() -> None:
    dry = {
        "source1": np.full(80_000, 0.25, dtype=np.float64),
        "source2": np.full(80_000, 0.125, dtype=np.float64),
    }
    rirs = np.zeros((1, 2, 2, 1), dtype=np.float64)
    rirs[0, 0, :, 0] = (1.0, 0.5)
    rirs[0, 1, :, 0] = (0.25, 1.0)

    stems, mixture = render_asset_bound_binaural(
        dry,
        rir_samples=rirs,
        rir_lengths=np.ones((1, 2), dtype=np.uint32),
        source_ids=("source1", "source2"),
        keyframe_samples=(0,),
    )

    assert stems["source1"].episode.shape == (2, 80_000)
    assert stems["source2"].episode.shape == (2, 80_000)
    assert np.array_equal(mixture, stems["source1"].episode + stems["source2"].episode)
    assert mixture[:, 0] == pytest.approx((0.28125, 0.25))


def test_renders_with_a_reused_precomputed_partition() -> None:
    dry = {
        "source1": np.full(80_000, 0.25, dtype=np.float64),
        "source2": np.full(80_000, 0.125, dtype=np.float64),
    }
    rirs = np.zeros((1, 2, 2, 1), dtype=np.float64)
    rirs[:, 0, :, 0] = (1.0, 0.5)
    rirs[:, 1, :, 0] = (0.25, 1.0)
    partition = raised_cosine_partition((0,), 80_000)

    _stems, mixture = render_asset_bound_binaural(
        dry,
        rir_samples=rirs,
        rir_lengths=np.ones((1, 2), dtype=np.uint32),
        source_ids=("source1", "source2"),
        keyframe_samples=(0,),
        partition_weights=partition,
    )

    assert mixture[:, 0] == pytest.approx((0.28125, 0.25))


def test_persisted_float32_mixture_is_the_exact_float32_stem_sum() -> None:
    class _Stem:
        def __init__(self, value: float) -> None:
            self.episode = np.full((2, 80_000), value, dtype=np.float64)

    stored, mixture = float32_stems_and_exact_mix(
        {"source1": _Stem(0.1), "source2": _Stem(0.2)},  # type: ignore[arg-type]
        source_ids=("source1", "source2"),
    )

    assert mixture.dtype == np.float32
    assert np.array_equal(mixture, stored["source1"] + stored["source2"])


def test_audio_program_endpoint_buses_require_an_explicit_slot_bijection() -> None:
    endpoint_buses = {
        "dog_muzzle": np.full(80_000, 0.25),
        "human_mouth": np.full(80_000, 0.125),
    }

    bound = bind_endpoint_buses_to_source_slots(
        endpoint_buses,
        endpoint_to_source_slot={
            "human_mouth": "source1",
            "dog_muzzle": "source2",
        },
        source_slots=("source1", "source2"),
    )

    assert tuple(bound) == ("source1", "source2")
    assert np.array_equal(bound["source1"], endpoint_buses["human_mouth"])
    assert np.array_equal(bound["source2"], endpoint_buses["dog_muzzle"])

    with pytest.raises(AssetBoundAudioError, match="bijection"):
        bind_endpoint_buses_to_source_slots(
            endpoint_buses,
            endpoint_to_source_slot={
                "human_mouth": "source1",
                "dog_muzzle": "source1",
            },
            source_slots=("source1", "source2"),
        )


def test_batch_variants_use_distinct_full_duration_source_windows() -> None:
    starts = variant_start_samples(
        source_sample_rate_hz=16_000,
        source_sample_count=11 * 16_000,
        variant_count=10,
    )

    assert starts[0] == 0
    assert starts[-1] == 6 * 16_000
    assert len(set(starts)) == 10


def test_batch_variants_refuse_to_loop_a_short_source() -> None:
    with pytest.raises(AssetBoundAudioError, match="shorter than one five-second"):
        variant_start_samples(
            source_sample_rate_hz=16_000,
            source_sample_count=79_999,
            variant_count=1,
        )
