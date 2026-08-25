from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from avengine.cli import build_parser
from avengine.contracts.json_io import write_json
from avengine.spatial_audio.audio import generate_sine_wave, read_float32_wav, write_float32_wav
from avengine.m5.current_m1_research_audio import (
    CurrentM1ResearchAudioError,
    load_current_m1_research_audio_inputs,
    render_current_m1_research_audio,
)


SOURCES = ("source0", "source1")


def _runtime_identity(root: Path) -> dict[str, Any]:
    return {
        "identity_schema": "avengine_current_installed_rlr_runtime_v1",
        "mode": "current-installed",
        "habitat_runtime_prefix": str(root / "habitat"),
        "habitat_sim_module": str(root / "habitat/habitat_sim/__init__.py"),
        "habitat_sim_binding": str(root / "habitat/habitat_sim/bindings.so"),
        "magnum_python_site": str(root / "magnum"),
        "rlr_sdk_root": str(root / "rlr"),
        "rlr_sdk_header": str(root / "rlr/headers/RLRAudioPropagation.h"),
        "rlr_sdk_library": str(root / "rlr/libs/libRLRAudioPropagation.so"),
        "rlr_adapter_enabled": True,
        "binding_api": "habitat_sim.RLRAcousticContext_v1",
    }


def _write_pair_receipt(
    root: Path,
    *,
    layout: str,
    reverse_pairs: bool = False,
    sample_rate_hz: int = 16_000,
    sidecar_metadata_mutation: Callable[[dict[str, Any]], None] | None = None,
    spatial_format_mutation: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Path, dict[str, np.ndarray]]:
    channel_count = 4 if layout == "foa" else 2
    render_mode = (
        "native_rlr_foa_from_static_m1"
        if layout == "foa"
        else "native_rlr_binaural_from_static_m1"
    )
    audio_role = (
        "native_rlr_foa_pair_ir" if layout == "foa" else "native_rlr_binaural_pair_ir"
    )
    if layout == "foa":
        spatial_format = {
            "format_id": "rlr_foa_acn_n3d_world_v1",
            "ambisonic_order": 1,
            "channel_count": 4,
            "raw_channel_order": ["W", "Y", "Z", "X"],
            "acn_indices": [0, 1, 2, 3],
            "normalization": "N3D",
            "coordinate_frame": "avengine_world",
            "handedness": "right",
            "axes": {
                "right": "+X",
                "up": "+Y",
                "back": "+Z",
                "forward": "-Z",
            },
            "raw_array_layout": "channel_major_[channels,samples]",
            "dtype": "float32_le",
        }
    else:
        spatial_format = {
            "channel_layout": {
                "type": "binaural",
                "channel_count": 2,
                "channel_order": ["left", "right"],
            },
            "hrtf_policy": "explicit_hash_and_license_required",
            "normalization_policy": "forbidden",
            "limiter_policy": "forbidden",
            "avengine_resampling_policy": "forbidden",
            "native_rate_adaptation_policy": "explicit_current_binary_only",
            "renderer": "RLR native binaural listener",
            "rendering_method": "rlr_native_binaural_v1",
        }
    if spatial_format_mutation is not None:
        spatial_format_mutation(spatial_format)
    if layout == "foa":
        pair_layout_metadata = {
            "spatial_format": copy.deepcopy(spatial_format),
            "hrtf_used": False,
            "amplitude_normalization": "not_applied",
        }
    else:
        pair_layout_metadata = {
            **copy.deepcopy(spatial_format),
            "hrtf_asset_id": "fixture_hrtf",
        }
    pairs: list[dict[str, Any]] = []
    expected: dict[str, np.ndarray] = {}
    for source_index, source_id in enumerate(SOURCES):
        samples = np.zeros((channel_count, 3), dtype=np.float32)
        samples[:, 0] = np.arange(1, channel_count + 1) * (source_index + 1) / 10
        samples[:, 1] = (source_index + 1) / 100
        metadata = {
            "audio_role": audio_role,
            "render_mode": render_mode,
            "listener_id": "listener0",
            "source_id": source_id,
            "source_authority": "static_m1_world_from_source",
            "dynamic_actor_anchor_claim": False,
            "m2_anchor_evidence_claim": False,
            "resampling": "not_applied",
            **copy.deepcopy(pair_layout_metadata),
        }
        if sidecar_metadata_mutation is not None:
            sidecar_metadata_mutation(metadata)
        artifact = write_float32_wav(
            root / "raw_ir" / layout / f"{source_id}.wav",
            samples,
            sample_rate_hz,
            metadata=metadata,
        )
        pairs.append(
            {
                "listener_id": "listener0",
                "source_id": source_id,
                "wav": artifact.audio_path.relative_to(root).as_posix(),
                "sidecar": artifact.sidecar_path.relative_to(root).as_posix(),
                "sample_rate_hz": sample_rate_hz,
                "channel_count": channel_count,
            }
        )
        expected[source_id] = samples
    if reverse_pairs:
        pairs.reverse()

    shared = root.parent / "shared"
    receipt: dict[str, Any] = {
        "status": "pass",
        "research_status": "research_candidate",
        "research_only": True,
        "episode_counted": False,
        "formal_dataset_count": 0,
        "qualification": False,
        "qualification_claim": False,
        "runtime_mode": "current-installed",
        "request_id": "current_m1_fixture",
        "inputs": {
            "m1_request": str(shared / "m1.json"),
            "simulation_request": str(shared / "simulation.json"),
            "package_manifest": str(shared / "package/manifest.json"),
        },
        "authority": {
            "room_id": "room0",
            "m1_request": str(shared / "m1.json"),
            "listener": {
                "listener_id": "listener0",
                "position_m": [1.0, 2.0, 3.0],
                "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
            "sources": [
                {
                    "source_id": "source0",
                    "position_m": [-1.0, 0.5, 2.0],
                    "motion": "static",
                },
                {
                    "source_id": "source1",
                    "position_m": [2.0, 0.5, -1.0],
                    "motion": "static",
                },
            ],
            "static_m1_sources": True,
            "dynamic_actor_anchor_claim": False,
            "m2_anchor_evidence_claim": False,
        },
        "claims": {
            "static_m1_source_positions": True,
            "dynamic_actor_motion": False,
            "m2_anchor_evidence": False,
            "formal_m4_qualification": False,
        },
        "render": {
            "mode": render_mode,
            "sample_rate_hz": sample_rate_hz,
            "declared_m1_source_order": ["source1", "source0"],
            "canonical_native_source_order": ["source0", "source1"],
            "listener_id": "listener0",
            "native_pair_count": 2,
            "propagation": {
                "direct": True,
                "indirect": True,
                "diffraction": True,
                "transmission": True,
            },
        },
        "acoustic_package": {
            "manifest": str(shared / "package/manifest.json"),
            "package_id": "package0",
            "source_room_id": "room0",
            "package_mode": "research_candidate",
            "nonpassing_research_qa_allowed": True,
            "qualification_claim": False,
        },
        "research_package_qa": {
            "admission": "nonpassing_research_override",
            "statuses": {"geometry_report": "fail", "ray_leakage": "not_run"},
            "formal_qualification": False,
            "dataset_admission": False,
        },
        "runtime_identity": _runtime_identity(shared),
        "pairs": pairs,
    }
    if layout == "foa":
        receipt.update(
            {
                "hrtf_used": False,
                "spatial_format": spatial_format,
            }
        )
    else:
        receipt.update(
            {
                "hrtf_preflight": {
                    "status": "pass",
                    **copy.deepcopy(spatial_format),
                    "render_sample_rate_hz": 16_000,
                    "hrtf": {"asset_id": "fixture_hrtf", "sample_rate_hz": 16_000},
                    "sample_rate_binding": {
                        "render_sample_rate_hz": 16_000,
                        "hrtf_input_sample_rate_hz": 16_000,
                        "policy": "strict_match",
                        "native_rate_adaptation": "not_required",
                        "avengine_resampling_performed": False,
                    },
                },
                "sofa_native_compatibility": {
                    "status": "pass",
                    "data_sampling_rate_hz": 16_000,
                    "data_ir_shape": [4, 2, 8],
                },
                "spatial_format": spatial_format,
            }
        )
    receipt_path = root / "research_receipt.json"
    write_json(receipt_path, receipt)
    return receipt_path, expected


def _fixture_pair(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, dict[str, np.ndarray]]]:
    foa, foa_samples = _write_pair_receipt(tmp_path / "foa", layout="foa")
    binaural, binaural_samples = _write_pair_receipt(
        tmp_path / "binaural", layout="binaural", reverse_pairs=True
    )
    return foa, binaural, {"foa": foa_samples, "binaural": binaural_samples}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_pair_join_uses_source_and_sidecar_not_receipt_order(tmp_path: Path) -> None:
    foa, binaural, expected = _fixture_pair(tmp_path)
    loaded = load_current_m1_research_audio_inputs(foa, binaural)

    assert tuple(loaded.foa.pair_samples) == SOURCES
    assert tuple(loaded.binaural.pair_samples) == SOURCES
    for layout in ("foa", "binaural"):
        observed = getattr(loaded, layout).pair_samples
        for source_id in SOURCES:
            assert np.array_equal(observed[source_id], expected[layout][source_id])

    value = _read_json(binaural)
    value["pairs"][0]["source_id"], value["pairs"][1]["source_id"] = (
        value["pairs"][1]["source_id"],
        value["pairs"][0]["source_id"],
    )
    write_json(binaural, value)
    with pytest.raises(CurrentM1ResearchAudioError, match="sidecar identity"):
        load_current_m1_research_audio_inputs(foa, binaural)


@pytest.mark.parametrize(
    ("layout", "mutation"),
    [
        (
            "foa",
            lambda value: value["spatial_format"].update(
                raw_channel_order=["W", "X", "Z", "Y"]
            ),
        ),
        (
            "foa",
            lambda value: value.update(amplitude_normalization="applied"),
        ),
        (
            "binaural",
            lambda value: value["channel_layout"].update(
                channel_order=["right", "left"]
            ),
        ),
        (
            "binaural",
            lambda value: value.update(hrtf_asset_id="different_hrtf"),
        ),
        (
            "binaural",
            lambda value: value.update(normalization_policy="applied"),
        ),
    ],
)
def test_pair_join_rejects_authenticated_sidecar_semantic_rebind(
    tmp_path: Path,
    layout: str,
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    foa, _ = _write_pair_receipt(
        tmp_path / "foa",
        layout="foa",
        sidecar_metadata_mutation=mutation if layout == "foa" else None,
    )
    binaural, _ = _write_pair_receipt(
        tmp_path / "binaural",
        layout="binaural",
        sidecar_metadata_mutation=mutation if layout == "binaural" else None,
    )
    with pytest.raises(CurrentM1ResearchAudioError, match="sidecar semantics differ"):
        load_current_m1_research_audio_inputs(foa, binaural)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(handedness="left"),
        lambda value: value["axes"].update(forward="+Z", back="-Z"),
        lambda value: value.update(acn_indices=[0, 3, 2, 1]),
    ],
)
def test_foa_rejects_self_consistent_wrong_direction_semantics(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    foa, _ = _write_pair_receipt(
        tmp_path / "foa",
        layout="foa",
        spatial_format_mutation=mutation,
    )
    binaural, _ = _write_pair_receipt(tmp_path / "binaural", layout="binaural")
    with pytest.raises(CurrentM1ResearchAudioError, match="invalid FOA semantics"):
        load_current_m1_research_audio_inputs(foa, binaural)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda v: v["inputs"].update(m1_request="/different/m1.json"), "different M1"),
        (
            lambda v: v["inputs"].update(simulation_request="/different/sim.json"),
            "different current M1",
        ),
        (
            lambda v: v["acoustic_package"].update(package_id="other"),
            "different current M1",
        ),
        (
            lambda v: v["authority"]["listener"].update(position_m=[9.0, 2.0, 3.0]),
            "different current M1",
        ),
        (
            lambda v: v["authority"]["sources"][0].update(position_m=[9.0, 0.5, 2.0]),
            "different current M1",
        ),
        (
            lambda v: v["render"]["propagation"].update(diffraction=False),
            "different current M1",
        ),
        (lambda v: v.update(research_only=False), "research-only"),
        (
            lambda v: v["runtime_identity"].update(rlr_sdk_root="/other/sdk"),
            "runtime identity",
        ),
        (
            lambda v: v["research_package_qa"]["statuses"].update(
                geometry_report="pass"
            ),
            "research package QA",
        ),
        (
            lambda v: v["sofa_native_compatibility"].update(status="fail"),
            "native SOFA preflight",
        ),
        (
            lambda v: v["hrtf_preflight"]["hrtf"].update(sample_rate_hz=44_100),
            "HRTF/SOFA rate",
        ),
        (
            lambda v: v["sofa_native_compatibility"].update(
                data_sampling_rate_hz=44_100
            ),
            "HRTF/SOFA rate",
        ),
        (
            lambda v: v["hrtf_preflight"]["sample_rate_binding"].update(
                policy="implicit_adaptation",
                native_rate_adaptation="performed",
                avengine_resampling_performed=True,
            ),
            "HRTF/SOFA rate",
        ),
        (
            lambda v: v["hrtf_preflight"]["channel_layout"].update(
                channel_order=["right", "left"]
            ),
            "preflight layout/policies",
        ),
        (
            lambda v: v["hrtf_preflight"].update(normalization_policy="applied"),
            "preflight layout/policies",
        ),
        (lambda v: v.update(research_only=1), "research-only current receipt"),
        (lambda v: v.update(episode_counted=0), "research-only current receipt"),
        (
            lambda v: v.update(formal_dataset_count=False),
            "research-only current receipt",
        ),
        (lambda v: v.update(qualification=0), "research-only current receipt"),
        (
            lambda v: v.update(qualification_claim=0),
            "research-only current receipt",
        ),
        (
            lambda v: v["claims"].update(static_m1_source_positions=False),
            "static-M1 research claims",
        ),
        (
            lambda v: v["claims"].update(dynamic_actor_motion=True),
            "static-M1 research claims",
        ),
        (
            lambda v: v["claims"].update(m2_anchor_evidence=True),
            "static-M1 research claims",
        ),
        (
            lambda v: v["claims"].update(formal_m4_qualification=True),
            "static-M1 research claims",
        ),
    ],
)
def test_join_rejects_mismatched_receipts(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    foa, binaural, _expected = _fixture_pair(tmp_path)
    value = _read_json(binaural)
    mutation(value)
    write_json(binaural, value)
    with pytest.raises(CurrentM1ResearchAudioError, match=message):
        load_current_m1_research_audio_inputs(foa, binaural)


def test_join_rejects_escape_and_non_16k_sidecar(tmp_path: Path) -> None:
    foa, binaural, _expected = _fixture_pair(tmp_path)
    value = _read_json(foa)
    value["pairs"][0]["wav"] = "../outside.wav"
    write_json(foa, value)
    with pytest.raises(CurrentM1ResearchAudioError, match="escapes its receipt root"):
        load_current_m1_research_audio_inputs(foa, binaural)

    other = tmp_path / "rate"
    foa_rate, _ = _write_pair_receipt(
        other / "foa", layout="foa", sample_rate_hz=48_000
    )
    binaural_rate, _ = _write_pair_receipt(other / "binaural", layout="binaural")
    with pytest.raises(CurrentM1ResearchAudioError, match="exactly 16 kHz"):
        load_current_m1_research_audio_inputs(foa_rate, binaural_rate)


def test_join_rejects_same_relative_identity_strings_from_different_roots(
    tmp_path: Path,
) -> None:
    foa, binaural, _expected = _fixture_pair(tmp_path)
    for path in (foa, binaural):
        value = _read_json(path)
        value["inputs"] = {
            "m1_request": "m1.json",
            "simulation_request": "simulation.json",
            "package_manifest": "package/manifest.json",
        }
        value["authority"]["m1_request"] = "m1.json"
        value["acoustic_package"]["manifest"] = "package/manifest.json"
        write_json(path, value)
    with pytest.raises(CurrentM1ResearchAudioError, match="absolute identity path"):
        load_current_m1_research_audio_inputs(foa, binaural)


def test_join_rejects_relative_runtime_identity_path(tmp_path: Path) -> None:
    foa, binaural, _expected = _fixture_pair(tmp_path)
    for path in (foa, binaural):
        value = _read_json(path)
        value["runtime_identity"]["rlr_sdk_root"] = "runtime/rlr-sdk"
        write_json(path, value)
    with pytest.raises(CurrentM1ResearchAudioError, match="absolute identity path"):
        load_current_m1_research_audio_inputs(foa, binaural)


def test_join_rejects_two_receipts_that_both_claim_formal_package(
    tmp_path: Path,
) -> None:
    foa, binaural, _expected = _fixture_pair(tmp_path)
    for path in (foa, binaural):
        value = _read_json(path)
        value["acoustic_package"].update(
            package_mode="formal", qualification_claim=True
        )
        write_json(path, value)
    with pytest.raises(
        CurrentM1ResearchAudioError, match="research nonqualification boundary"
    ):
        load_current_m1_research_audio_inputs(foa, binaural)


def test_join_allows_all_pass_research_qa_without_nonpassing_override(
    tmp_path: Path,
) -> None:
    foa, binaural, _expected = _fixture_pair(tmp_path)
    for path in (foa, binaural):
        value = _read_json(path)
        value["acoustic_package"].pop("nonpassing_research_qa_allowed")
        value["research_package_qa"].update(
            admission="passing_research",
            statuses={"geometry_report": "pass", "ray_leakage": "pass"},
        )
        write_json(path, value)
    loaded = load_current_m1_research_audio_inputs(foa, binaural)
    assert loaded.room_id == "room0"


def test_render_writes_eight_exact_episode_wavs_and_reconstructs(
    tmp_path: Path,
) -> None:
    foa, binaural, pair_samples = _fixture_pair(tmp_path)
    output = tmp_path / "output"
    receipt = render_current_m1_research_audio(foa, binaural, output)

    assert receipt["research_only"] is True
    assert receipt["episode_counted"] is False
    assert receipt["formal_dataset_count"] == 0
    assert receipt["qualification"] is False
    assert receipt["qualification_claim"] is False
    assert "schema" not in receipt
    assert len(list(output.rglob("*.wav"))) == 8
    assert len(list(output.rglob("*.wav.json"))) == 8
    for path in output.rglob("*.wav.json"):
        metadata = _read_json(path)["metadata"]
        assert metadata["resampling"] == "not_applied"
        assert metadata["amplitude_normalization"] == "not_applied"
        assert metadata["limiter"] == "not_applied"

    dry: dict[str, np.ndarray] = {}
    for source_id, frequency, phase in (
        ("source0", 440.0, 0.0),
        ("source1", 660.0, math.pi / 2.0),
    ):
        wav = read_float32_wav(output / receipt["audio"]["dry"][source_id]["wav"])
        assert wav.sample_rate_hz == 16_000
        assert wav.samples.shape == (1, 80_000)
        expected = np.asarray(
            generate_sine_wave(
                16_000,
                80_000,
                frequency,
                amplitude=0.25,
                phase_radians=phase,
            ),
            dtype=np.float32,
        )
        assert np.array_equal(wav.samples[0], expected)
        dry[source_id] = wav.samples[0]

    input_receipts = {"foa": _read_json(foa), "binaural": _read_json(binaural)}
    for layout, channel_count in (("foa", 4), ("binaural", 2)):
        layout_record = receipt["audio"]["layouts"][layout]
        expected_spatial = input_receipts[layout]["spatial_format"]
        assert layout_record["spatial_format"] == expected_spatial
        if layout == "foa":
            assert layout_record["hrtf_used"] is False
        else:
            assert layout_record["hrtf_asset_id"] == "fixture_hrtf"
        stem_samples: list[np.ndarray] = []
        for source_id in SOURCES:
            stem = read_float32_wav(output / layout_record["stems"][source_id]["wav"])
            assert stem.sample_rate_hz == 16_000
            assert stem.samples.shape == (channel_count, 80_000)
            assert stem.sidecar is not None
            assert stem.sidecar["metadata"]["spatial_format"] == expected_spatial
            if layout == "foa":
                assert stem.sidecar["metadata"]["hrtf_used"] is False
            else:
                assert stem.sidecar["metadata"]["hrtf_asset_id"] == "fixture_hrtf"
            expected_channels = np.stack(
                [
                    np.convolve(
                        dry[source_id].astype(np.float64),
                        channel.astype(np.float64),
                    )[:80_000]
                    for channel in pair_samples[layout][source_id]
                ]
            ).astype(np.float32)
            assert np.allclose(stem.samples, expected_channels, rtol=0.0, atol=2e-7)
            stem_samples.append(stem.samples)
            assert layout_record["full_tail_frame_count_by_source"][source_id] == 80_002
        mix = read_float32_wav(output / layout_record["mix"]["wav"])
        assert mix.samples.shape == (channel_count, 80_000)
        assert mix.sidecar is not None
        assert mix.sidecar["metadata"]["spatial_format"] == expected_spatial
        if layout == "foa":
            assert mix.sidecar["metadata"]["hrtf_used"] is False
        else:
            assert mix.sidecar["metadata"]["hrtf_asset_id"] == "fixture_hrtf"
        assert np.allclose(
            mix.samples, stem_samples[0] + stem_samples[1], rtol=0.0, atol=2e-7
        )

    with pytest.raises(CurrentM1ResearchAudioError, match="refusing to replace"):
        render_current_m1_research_audio(foa, binaural, output)


def test_pair_list_order_does_not_change_rendered_samples(tmp_path: Path) -> None:
    foa, binaural, _expected = _fixture_pair(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    receipt_a = render_current_m1_research_audio(foa, binaural, first)
    for path in (foa, binaural):
        value = _read_json(path)
        value["pairs"].reverse()
        write_json(path, value)
    receipt_b = render_current_m1_research_audio(foa, binaural, second)
    for relative in sorted(path.relative_to(first) for path in first.rglob("*.wav")):
        left = read_float32_wav(first / relative).samples
        right = read_float32_wav(second / relative).samples
        assert np.array_equal(left, right)
    assert receipt_a["audio"]["layouts"] == receipt_b["audio"]["layouts"]


def test_cli_has_only_receipt_and_output_inputs() -> None:
    parser = build_parser()
    command = [
        "m5",
        "render-current-m1-research-audio",
        "--foa-receipt",
        "foa.json",
        "--binaural-receipt",
        "binaural.json",
        "--output",
        "/external/output",
    ]
    parsed = parser.parse_args(command)
    assert parsed.m5_command == "render-current-m1-research-audio"
    for forbidden in (
        "runtime_prefix",
        "runtime_root",
        "rlr_sdk_root",
        "magnum_python_site",
        "hrtf",
        "native",
    ):
        assert not hasattr(parsed, forbidden)
    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args([*command, "--runtime-prefix", "/unexpected"])
    assert exit_info.value.code == 2
