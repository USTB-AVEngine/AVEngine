"""Offline five-second M5 assembly from matching current-M1 pair receipts.

This research-only path reads already-rendered FOA and binaural pair IR WAVs.
It never activates Habitat, RLR, an HRTF, Unreal, or another native runtime.
Pair identity comes from ``source_id`` plus each retained WAV sidecar rather
than receipt-list order.  The only output crop is the explicit five-second
``[0, 80000)`` episode window; full convolution lengths remain in metadata.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from avengine.contracts.json_io import load_json, write_json
from avengine.spatial_audio.audio import (
    AudioContractError,
    Float32WavArtifact,
    generate_sine_wave,
    read_float32_wav,
    write_float32_wav,
)
from avengine.timeline.audio import (
    M5_AUDIO_SAMPLE_COUNT,
    M5_AUDIO_SAMPLE_RATE_HZ,
    DynamicStemResult,
    render_dynamic_stems_and_mix,
)


CURRENT_M1_RESEARCH_SOURCE_IDS = ("source0", "source1")
_LAYOUT_CHANNELS = {"foa": 4, "binaural": 2}
_LAYOUT_RENDER_MODES = {
    "foa": "native_rlr_foa_from_static_m1",
    "binaural": "native_rlr_binaural_from_static_m1",
}
_LAYOUT_AUDIO_ROLES = {
    "foa": "native_rlr_foa_pair_ir",
    "binaural": "native_rlr_binaural_pair_ir",
}
_FOA_SPATIAL_FORMAT = {
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


class CurrentM1ResearchAudioError(ValueError):
    """Current pair receipts cannot safely form one offline research result."""


@dataclass(frozen=True)
class CurrentM1PairLayout:
    """One validated, source-addressed pair-IR layout."""

    layout: str
    receipt_path: Path
    receipt: Mapping[str, Any]
    listener_id: str
    pair_samples: Mapping[str, np.ndarray]
    pair_wavs: Mapping[str, Path]
    pair_sidecars: Mapping[str, Path]
    spatial_format: Mapping[str, Any]
    hrtf_asset_id: str | None


@dataclass(frozen=True)
class CurrentM1ResearchAudioInputs:
    """The matching FOA/binaural inputs for one offline assembly."""

    foa: CurrentM1PairLayout
    binaural: CurrentM1PairLayout
    request_id: str
    room_id: str
    listener_id: str
    source_positions_m: Mapping[str, tuple[float, float, float]]


def _mapping(value: Any, *, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CurrentM1ResearchAudioError(f"{owner} must be a JSON object")
    return value


def _text(value: Any, *, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CurrentM1ResearchAudioError(f"{owner} must be non-empty text")
    return value


def _vector(value: Any, *, owner: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise CurrentM1ResearchAudioError(
            f"{owner} must contain exactly {length} numeric values"
        )
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise CurrentM1ResearchAudioError(f"{owner} must be numeric")
        number = float(item)
        if not math.isfinite(number):
            raise CurrentM1ResearchAudioError(f"{owner} must be finite")
        result.append(number)
    return tuple(result)


def _declared_identity_path(value: Any, *, owner: str) -> str:
    raw = _text(value, owner=owner)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise CurrentM1ResearchAudioError(
            f"{owner} must declare an absolute identity path"
        )
    return str(path.resolve())


def _confined_file(root: Path, value: Any, *, owner: str) -> Path:
    raw = Path(_text(value, owner=owner)).expanduser()
    candidate = (raw if raw.is_absolute() else root / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise CurrentM1ResearchAudioError(
            f"{owner} escapes its receipt root: {candidate}"
        ) from error
    if not candidate.is_file():
        raise CurrentM1ResearchAudioError(
            f"{owner} is not a readable file: {candidate}"
        )
    return candidate


def _research_boundary(receipt: Mapping[str, Any], *, owner: str) -> tuple[Any, ...]:
    expected_text = {
        "status": "pass",
        "research_status": "research_candidate",
        "runtime_mode": "current-installed",
    }
    expected_bool = {
        "research_only": True,
        "episode_counted": False,
        "qualification": False,
        "qualification_claim": False,
    }
    mismatches = [
        name
        for name, expected_value in expected_text.items()
        if type(receipt.get(name)) is not str or receipt.get(name) != expected_value
    ]
    mismatches.extend(
        name
        for name, expected_value in expected_bool.items()
        if receipt.get(name) is not expected_value
    )
    formal_count = receipt.get("formal_dataset_count")
    if type(formal_count) is not int or formal_count != 0:
        mismatches.append("formal_dataset_count")
    if mismatches:
        raise CurrentM1ResearchAudioError(
            f"{owner} is not a passing research-only current receipt: "
            + ", ".join(mismatches)
        )
    return (
        *(receipt[name] for name in expected_text),
        *(receipt[name] for name in expected_bool),
        formal_count,
    )


def _claims_identity(receipt: Mapping[str, Any], *, owner: str) -> tuple[bool, ...]:
    claims = _mapping(receipt.get("claims"), owner=f"{owner}.claims")
    expected = {
        "static_m1_source_positions": True,
        "dynamic_actor_motion": False,
        "m2_anchor_evidence": False,
        "formal_m4_qualification": False,
    }
    mismatches = [
        name
        for name, expected_value in expected.items()
        if claims.get(name) is not expected_value
    ]
    if mismatches:
        raise CurrentM1ResearchAudioError(
            f"{owner} changes the static-M1 research claims: " + ", ".join(mismatches)
        )
    return tuple(claims[name] for name in expected)


def _authority_identity(
    receipt: Mapping[str, Any], *, owner: str
) -> tuple[str, str, tuple[float, ...], dict[str, tuple[float, float, float]]]:
    authority = _mapping(receipt.get("authority"), owner=f"{owner}.authority")
    room_id = _text(authority.get("room_id"), owner=f"{owner}.authority.room_id")
    listener = _mapping(authority.get("listener"), owner=f"{owner}.authority.listener")
    listener_id = _text(
        listener.get("listener_id"), owner=f"{owner}.authority.listener.listener_id"
    )
    listener_pose = (
        *_vector(
            listener.get("position_m"),
            owner=f"{owner}.authority.listener.position_m",
            length=3,
        ),
        *_vector(
            listener.get("orientation_wxyz"),
            owner=f"{owner}.authority.listener.orientation_wxyz",
            length=4,
        ),
    )
    raw_sources = authority.get("sources")
    if not isinstance(raw_sources, list) or len(raw_sources) != 2:
        raise CurrentM1ResearchAudioError(
            f"{owner}.authority.sources must contain source0 and source1"
        )
    sources: dict[str, tuple[float, float, float]] = {}
    for index, raw_source in enumerate(raw_sources):
        source = _mapping(raw_source, owner=f"{owner}.authority.sources[{index}]")
        source_id = _text(
            source.get("source_id"),
            owner=f"{owner}.authority.sources[{index}].source_id",
        )
        if source_id in sources:
            raise CurrentM1ResearchAudioError(
                f"{owner} repeats source_id {source_id!r}"
            )
        if source.get("motion") != "static":
            raise CurrentM1ResearchAudioError(
                f"{owner} source {source_id!r} is not a static M1 endpoint"
            )
        position = _vector(
            source.get("position_m"),
            owner=f"{owner}.authority.sources[{index}].position_m",
            length=3,
        )
        sources[source_id] = (position[0], position[1], position[2])
    if set(sources) != set(CURRENT_M1_RESEARCH_SOURCE_IDS):
        raise CurrentM1ResearchAudioError(
            f"{owner} authority must contain exactly source0 and source1"
        )
    if (
        authority.get("static_m1_sources") is not True
        or authority.get("dynamic_actor_anchor_claim") is not False
        or authority.get("m2_anchor_evidence_claim") is not False
    ):
        raise CurrentM1ResearchAudioError(
            f"{owner} changes the static-M1 research authority boundary"
        )
    return room_id, listener_id, listener_pose, sources


def _input_identity(receipt: Mapping[str, Any], *, owner: str) -> dict[str, str]:
    inputs = _mapping(receipt.get("inputs"), owner=f"{owner}.inputs")
    result = {
        name: _declared_identity_path(inputs.get(name), owner=f"{owner}.inputs.{name}")
        for name in ("m1_request", "simulation_request", "package_manifest")
    }
    authority = _mapping(receipt.get("authority"), owner=f"{owner}.authority")
    if (
        _declared_identity_path(
            authority.get("m1_request"), owner=f"{owner}.authority.m1_request"
        )
        != result["m1_request"]
    ):
        raise CurrentM1ResearchAudioError(
            f"{owner} binds different M1 paths in inputs and authority"
        )
    package = _mapping(
        receipt.get("acoustic_package"), owner=f"{owner}.acoustic_package"
    )
    if (
        _declared_identity_path(
            package.get("manifest"), owner=f"{owner}.acoustic_package.manifest"
        )
        != result["package_manifest"]
    ):
        raise CurrentM1ResearchAudioError(
            f"{owner} binds different package manifests internally"
        )
    return result


def _package_identity(receipt: Mapping[str, Any], *, owner: str) -> tuple[Any, ...]:
    package = _mapping(
        receipt.get("acoustic_package"), owner=f"{owner}.acoustic_package"
    )
    if (
        package.get("package_mode") != "research_candidate"
        or package.get("qualification_claim") is not False
    ):
        raise CurrentM1ResearchAudioError(
            f"{owner} acoustic package must retain the research nonqualification boundary"
        )
    return (
        _declared_identity_path(
            package.get("manifest"), owner=f"{owner}.acoustic_package.manifest"
        ),
        _text(package.get("package_id"), owner=f"{owner}.acoustic_package.package_id"),
        _text(
            package.get("source_room_id"),
            owner=f"{owner}.acoustic_package.source_room_id",
        ),
        package.get("package_mode"),
        package.get("nonpassing_research_qa_allowed"),
        package.get("qualification_claim"),
    )


def _propagation_identity(
    receipt: Mapping[str, Any], *, owner: str
) -> tuple[tuple[str, bool], ...]:
    render = _mapping(receipt.get("render"), owner=f"{owner}.render")
    propagation = _mapping(
        render.get("propagation"), owner=f"{owner}.render.propagation"
    )
    expected_keys = ("direct", "indirect", "diffraction", "transmission")
    result: list[tuple[str, bool]] = []
    for name in expected_keys:
        value = propagation.get(name)
        if not isinstance(value, bool):
            raise CurrentM1ResearchAudioError(
                f"{owner}.render.propagation.{name} must be boolean"
            )
        result.append((name, value))
    return tuple(result)


def _runtime_identity(receipt: Mapping[str, Any], *, owner: str) -> dict[str, Any]:
    identity = dict(
        _mapping(receipt.get("runtime_identity"), owner=f"{owner}.runtime_identity")
    )
    text_fields = ("identity_schema", "mode", "binding_api")
    path_fields = (
        "habitat_runtime_prefix",
        "habitat_sim_module",
        "habitat_sim_binding",
        "magnum_python_site",
        "rlr_sdk_root",
        "rlr_sdk_header",
        "rlr_sdk_library",
    )
    for name in text_fields:
        _text(identity.get(name), owner=f"{owner}.runtime_identity.{name}")
    for name in path_fields:
        identity[name] = _declared_identity_path(
            identity.get(name), owner=f"{owner}.runtime_identity.{name}"
        )
    if identity.get("mode") != "current-installed":
        raise CurrentM1ResearchAudioError(
            f"{owner}.runtime_identity.mode must be current-installed"
        )
    if identity.get("rlr_adapter_enabled") is not True:
        raise CurrentM1ResearchAudioError(
            f"{owner}.runtime_identity.rlr_adapter_enabled must be true"
        )
    return identity


def _research_package_qa(receipt: Mapping[str, Any], *, owner: str) -> dict[str, Any]:
    value = dict(
        _mapping(
            receipt.get("research_package_qa"),
            owner=f"{owner}.research_package_qa",
        )
    )
    if (
        value.get("formal_qualification") is not False
        or value.get("dataset_admission") is not False
    ):
        raise CurrentM1ResearchAudioError(
            f"{owner}.research_package_qa changes the non-formal boundary"
        )
    _mapping(value.get("statuses"), owner=f"{owner}.research_package_qa.statuses")
    return value


def _validate_layout_declaration(
    receipt: Mapping[str, Any], *, owner: str, layout: str, listener_id: str
) -> None:
    render = _mapping(receipt.get("render"), owner=f"{owner}.render")
    if render.get("mode") != _LAYOUT_RENDER_MODES[layout]:
        raise CurrentM1ResearchAudioError(f"{owner} has the wrong {layout} render mode")
    if render.get("sample_rate_hz") != M5_AUDIO_SAMPLE_RATE_HZ:
        raise CurrentM1ResearchAudioError(f"{owner} must be exactly 16 kHz")
    if render.get("listener_id") != listener_id:
        raise CurrentM1ResearchAudioError(
            f"{owner} render listener differs from M1 authority"
        )
    if render.get("native_pair_count") != 2:
        raise CurrentM1ResearchAudioError(f"{owner} must retain exactly two pairs")
    if render.get("canonical_native_source_order") != list(
        CURRENT_M1_RESEARCH_SOURCE_IDS
    ):
        raise CurrentM1ResearchAudioError(
            f"{owner} lacks canonical source0/source1 native identity"
        )
    declared = render.get("declared_m1_source_order")
    if (
        not isinstance(declared, list)
        or set(declared) != set(CURRENT_M1_RESEARCH_SOURCE_IDS)
        or len(declared) != 2
    ):
        raise CurrentM1ResearchAudioError(
            f"{owner} M1 declaration must contain source0 and source1"
        )
    spatial = _mapping(receipt.get("spatial_format"), owner=f"{owner}.spatial_format")
    if layout == "foa":
        if dict(spatial) != _FOA_SPATIAL_FORMAT:
            raise CurrentM1ResearchAudioError(f"{owner} has invalid FOA semantics")
    else:
        channel_layout = _mapping(
            spatial.get("channel_layout"),
            owner=f"{owner}.spatial_format.channel_layout",
        )
        if channel_layout != {
            "type": "binaural",
            "channel_count": 2,
            "channel_order": ["left", "right"],
        }:
            raise CurrentM1ResearchAudioError(
                f"{owner} has invalid binaural left/right semantics"
            )
    if layout == "foa":
        if receipt.get("hrtf_used") is not False:
            raise CurrentM1ResearchAudioError(f"{owner} must record hrtf_used=false")
    else:
        hrtf_preflight = _mapping(
            receipt.get("hrtf_preflight"), owner=f"{owner}.hrtf_preflight"
        )
        hrtf = _mapping(
            hrtf_preflight.get("hrtf"), owner=f"{owner}.hrtf_preflight.hrtf"
        )
        rate_binding = _mapping(
            hrtf_preflight.get("sample_rate_binding"),
            owner=f"{owner}.hrtf_preflight.sample_rate_binding",
        )
        sofa = _mapping(
            receipt.get("sofa_native_compatibility"),
            owner=f"{owner}.sofa_native_compatibility",
        )
        if hrtf_preflight.get("status") != "pass" or sofa.get("status") != "pass":
            raise CurrentM1ResearchAudioError(
                f"{owner} lacks passing HRTF and native SOFA preflight"
            )
        sofa_shape = sofa.get("data_ir_shape")
        if (
            hrtf_preflight.get("render_sample_rate_hz") != M5_AUDIO_SAMPLE_RATE_HZ
            or hrtf.get("sample_rate_hz") != M5_AUDIO_SAMPLE_RATE_HZ
            or rate_binding.get("render_sample_rate_hz") != M5_AUDIO_SAMPLE_RATE_HZ
            or rate_binding.get("hrtf_input_sample_rate_hz") != M5_AUDIO_SAMPLE_RATE_HZ
            or rate_binding.get("policy") != "strict_match"
            or rate_binding.get("native_rate_adaptation") != "not_required"
            or rate_binding.get("avengine_resampling_performed") is not False
            or sofa.get("data_sampling_rate_hz") != M5_AUDIO_SAMPLE_RATE_HZ
            or not isinstance(sofa_shape, list)
            or len(sofa_shape) != 3
            or sofa_shape[1] != 2
        ):
            raise CurrentM1ResearchAudioError(
                f"{owner} HRTF/SOFA rate or strict no-resampling binding is invalid"
            )
        expected_spatial = {
            "hrtf_policy": "explicit_hash_and_license_required",
            "normalization_policy": "forbidden",
            "limiter_policy": "forbidden",
            "avengine_resampling_policy": "forbidden",
            "renderer": "RLR native binaural listener",
            "rendering_method": "rlr_native_binaural_v1",
        }
        if any(spatial.get(name) != value for name, value in expected_spatial.items()):
            raise CurrentM1ResearchAudioError(
                f"{owner} has invalid binaural rendering/processing policy"
            )
        _text(
            spatial.get("native_rate_adaptation_policy"),
            owner=f"{owner}.spatial_format.native_rate_adaptation_policy",
        )
        _text(hrtf.get("asset_id"), owner=f"{owner}.hrtf_preflight.hrtf.asset_id")
        preflight_layout = _mapping(
            hrtf_preflight.get("channel_layout"),
            owner=f"{owner}.hrtf_preflight.channel_layout",
        )
        policy_fields = (
            "hrtf_policy",
            "normalization_policy",
            "limiter_policy",
            "avengine_resampling_policy",
            "native_rate_adaptation_policy",
            "renderer",
            "rendering_method",
        )
        if dict(preflight_layout) != dict(channel_layout) or any(
            hrtf_preflight.get(name) != spatial.get(name) for name in policy_fields
        ):
            raise CurrentM1ResearchAudioError(
                f"{owner} HRTF preflight layout/policies differ from spatial format"
            )


def _validate_pair_sidecar_metadata(
    metadata: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any],
    owner: str,
    layout: str,
    source_id: str,
    listener_id: str,
) -> None:
    if (
        metadata.get("source_id") != source_id
        or metadata.get("listener_id") != listener_id
        or metadata.get("audio_role") != _LAYOUT_AUDIO_ROLES[layout]
        or metadata.get("render_mode") != _LAYOUT_RENDER_MODES[layout]
        or metadata.get("source_authority") != "static_m1_world_from_source"
        or metadata.get("dynamic_actor_anchor_claim") is not False
        or metadata.get("m2_anchor_evidence_claim") is not False
        or metadata.get("resampling") != "not_applied"
    ):
        raise CurrentM1ResearchAudioError(
            f"{owner} pair {source_id!r} sidecar identity/processing metadata differs"
        )
    spatial = _mapping(receipt.get("spatial_format"), owner=f"{owner}.spatial_format")
    if layout == "foa":
        pair_spatial = _mapping(
            metadata.get("spatial_format"),
            owner=f"{owner} pair {source_id!r} metadata.spatial_format",
        )
        if (
            dict(pair_spatial) != dict(spatial)
            or metadata.get("hrtf_used") is not False
            or metadata.get("amplitude_normalization") != "not_applied"
            or metadata.get("limiter", "not_applied") != "not_applied"
        ):
            raise CurrentM1ResearchAudioError(
                f"{owner} pair {source_id!r} FOA sidecar semantics differ"
            )
        return

    pair_layout = _mapping(
        metadata.get("channel_layout"),
        owner=f"{owner} pair {source_id!r} metadata.channel_layout",
    )
    expected_layout = _mapping(
        spatial.get("channel_layout"), owner=f"{owner}.spatial_format.channel_layout"
    )
    hrtf_preflight = _mapping(
        receipt.get("hrtf_preflight"), owner=f"{owner}.hrtf_preflight"
    )
    hrtf = _mapping(hrtf_preflight.get("hrtf"), owner=f"{owner}.hrtf_preflight.hrtf")
    policy_fields = (
        "hrtf_policy",
        "normalization_policy",
        "limiter_policy",
        "avengine_resampling_policy",
        "native_rate_adaptation_policy",
        "renderer",
        "rendering_method",
    )
    if (
        dict(pair_layout) != dict(expected_layout)
        or metadata.get("hrtf_asset_id") != hrtf.get("asset_id")
        or any(metadata.get(name) != spatial.get(name) for name in policy_fields)
        or metadata.get("amplitude_normalization", "not_applied") != "not_applied"
        or metadata.get("limiter", "not_applied") != "not_applied"
    ):
        raise CurrentM1ResearchAudioError(
            f"{owner} pair {source_id!r} binaural sidecar semantics differ"
        )


def _load_layout(receipt_value: str | Path, *, layout: str) -> CurrentM1PairLayout:
    owner = f"{layout} receipt"
    receipt_path = Path(receipt_value).expanduser().resolve()
    if not receipt_path.is_file():
        raise CurrentM1ResearchAudioError(
            f"{owner} is not a readable JSON file: {receipt_path}"
        )
    try:
        receipt = _mapping(load_json(receipt_path), owner=owner)
    except (OSError, ValueError) as error:
        raise CurrentM1ResearchAudioError(f"cannot load {owner}: {error}") from error
    _research_boundary(receipt, owner=owner)
    _claims_identity(receipt, owner=owner)
    _input_identity(receipt, owner=owner)
    room_id, listener_id, _listener_pose, _sources = _authority_identity(
        receipt, owner=owner
    )
    package = _mapping(
        receipt.get("acoustic_package"), owner=f"{owner}.acoustic_package"
    )
    if package.get("source_room_id") != room_id:
        raise CurrentM1ResearchAudioError(
            f"{owner} package room differs from M1 authority room"
        )
    _validate_layout_declaration(
        receipt, owner=owner, layout=layout, listener_id=listener_id
    )

    raw_pairs = receipt.get("pairs")
    if not isinstance(raw_pairs, list) or len(raw_pairs) != 2:
        raise CurrentM1ResearchAudioError(
            f"{owner}.pairs must contain exactly source0 and source1"
        )
    root = receipt_path.parent.resolve()
    samples_by_source: dict[str, np.ndarray] = {}
    wavs_by_source: dict[str, Path] = {}
    sidecars_by_source: dict[str, Path] = {}
    for index, raw_pair in enumerate(raw_pairs):
        pair = _mapping(raw_pair, owner=f"{owner}.pairs[{index}]")
        source_id = _text(
            pair.get("source_id"), owner=f"{owner}.pairs[{index}].source_id"
        )
        if source_id in samples_by_source:
            raise CurrentM1ResearchAudioError(
                f"{owner} repeats pair source_id {source_id!r}"
            )
        if source_id not in CURRENT_M1_RESEARCH_SOURCE_IDS:
            raise CurrentM1ResearchAudioError(
                f"{owner} contains unexpected pair source_id {source_id!r}"
            )
        if pair.get("listener_id") != listener_id:
            raise CurrentM1ResearchAudioError(
                f"{owner} pair {source_id!r} listener differs from authority"
            )
        if pair.get("sample_rate_hz") != M5_AUDIO_SAMPLE_RATE_HZ:
            raise CurrentM1ResearchAudioError(
                f"{owner} pair {source_id!r} must be exactly 16 kHz"
            )
        if pair.get("channel_count") != _LAYOUT_CHANNELS[layout]:
            raise CurrentM1ResearchAudioError(
                f"{owner} pair {source_id!r} has the wrong channel count"
            )
        wav_path = _confined_file(
            root, pair.get("wav"), owner=f"{owner}.pairs[{index}].wav"
        )
        sidecar_path = _confined_file(
            root, pair.get("sidecar"), owner=f"{owner}.pairs[{index}].sidecar"
        )
        try:
            wav = read_float32_wav(
                wav_path, sidecar_path=sidecar_path, verify_sidecar=True
            )
        except AudioContractError as error:
            raise CurrentM1ResearchAudioError(
                f"{owner} pair {source_id!r} WAV/sidecar is invalid: {error}"
            ) from error
        if (
            wav.sample_rate_hz != M5_AUDIO_SAMPLE_RATE_HZ
            or wav.channel_count != _LAYOUT_CHANNELS[layout]
            or wav.sidecar is None
        ):
            raise CurrentM1ResearchAudioError(
                f"{owner} pair {source_id!r} decoded dimensions differ from receipt"
            )
        metadata = _mapping(
            wav.sidecar.get("metadata"), owner=f"{owner} pair {source_id!r} metadata"
        )
        _validate_pair_sidecar_metadata(
            metadata,
            receipt=receipt,
            owner=owner,
            layout=layout,
            source_id=source_id,
            listener_id=listener_id,
        )
        samples_by_source[source_id] = np.ascontiguousarray(wav.samples)
        wavs_by_source[source_id] = wav_path
        sidecars_by_source[source_id] = sidecar_path
    if set(samples_by_source) != set(CURRENT_M1_RESEARCH_SOURCE_IDS):
        raise CurrentM1ResearchAudioError(
            f"{owner} does not bind both source0 and source1"
        )
    samples_by_source = {
        source_id: samples_by_source[source_id]
        for source_id in CURRENT_M1_RESEARCH_SOURCE_IDS
    }
    wavs_by_source = {
        source_id: wavs_by_source[source_id]
        for source_id in CURRENT_M1_RESEARCH_SOURCE_IDS
    }
    sidecars_by_source = {
        source_id: sidecars_by_source[source_id]
        for source_id in CURRENT_M1_RESEARCH_SOURCE_IDS
    }
    return CurrentM1PairLayout(
        layout=layout,
        receipt_path=receipt_path,
        receipt=receipt,
        listener_id=listener_id,
        pair_samples=samples_by_source,
        pair_wavs=wavs_by_source,
        pair_sidecars=sidecars_by_source,
        spatial_format=copy.deepcopy(dict(receipt["spatial_format"])),
        hrtf_asset_id=(
            None
            if layout == "foa"
            else str(receipt["hrtf_preflight"]["hrtf"]["asset_id"])
        ),
    )


def load_current_m1_research_audio_inputs(
    foa_receipt: str | Path,
    binaural_receipt: str | Path,
) -> CurrentM1ResearchAudioInputs:
    """Load and cross-check the two current pair receipts without rendering."""

    foa = _load_layout(foa_receipt, layout="foa")
    binaural = _load_layout(binaural_receipt, layout="binaural")
    left_owner = "FOA receipt"
    right_owner = "binaural receipt"

    comparisons = (
        (
            "research boundary",
            _research_boundary(foa.receipt, owner=left_owner),
            _research_boundary(binaural.receipt, owner=right_owner),
        ),
        (
            "static-M1 claims",
            _claims_identity(foa.receipt, owner=left_owner),
            _claims_identity(binaural.receipt, owner=right_owner),
        ),
        (
            "request_id",
            _text(foa.receipt.get("request_id"), owner=f"{left_owner}.request_id"),
            _text(
                binaural.receipt.get("request_id"),
                owner=f"{right_owner}.request_id",
            ),
        ),
        (
            "M1/simulation/package inputs",
            _input_identity(foa.receipt, owner=left_owner),
            _input_identity(binaural.receipt, owner=right_owner),
        ),
        (
            "acoustic package",
            _package_identity(foa.receipt, owner=left_owner),
            _package_identity(binaural.receipt, owner=right_owner),
        ),
        (
            "room/listener/source authority",
            _authority_identity(foa.receipt, owner=left_owner),
            _authority_identity(binaural.receipt, owner=right_owner),
        ),
        (
            "propagation",
            _propagation_identity(foa.receipt, owner=left_owner),
            _propagation_identity(binaural.receipt, owner=right_owner),
        ),
        (
            "runtime identity",
            _runtime_identity(foa.receipt, owner=left_owner),
            _runtime_identity(binaural.receipt, owner=right_owner),
        ),
        (
            "research package QA",
            _research_package_qa(foa.receipt, owner=left_owner),
            _research_package_qa(binaural.receipt, owner=right_owner),
        ),
    )
    mismatches = [name for name, left, right in comparisons if left != right]
    if mismatches:
        raise CurrentM1ResearchAudioError(
            "FOA and binaural receipts describe different current M1 inputs: "
            + ", ".join(mismatches)
        )

    room_id, listener_id, _listener_pose, sources = _authority_identity(
        foa.receipt, owner=left_owner
    )
    return CurrentM1ResearchAudioInputs(
        foa=foa,
        binaural=binaural,
        request_id=str(foa.receipt["request_id"]),
        room_id=room_id,
        listener_id=listener_id,
        source_positions_m=sources,
    )


def _fresh_output_path(value: str | Path) -> Path:
    if not str(value).strip():
        raise CurrentM1ResearchAudioError("offline current M1 audio requires --output")
    requested = Path(value).expanduser()
    output = requested.resolve()
    if os.path.lexists(requested) or os.path.lexists(output):
        raise CurrentM1ResearchAudioError(
            f"refusing to replace current M1 research audio output: {requested}"
        )
    return output


def _deterministic_dry() -> dict[str, np.ndarray]:
    definitions = {
        "source0": (440.0, 0.0),
        "source1": (660.0, math.pi / 2.0),
    }
    return {
        source_id: np.ascontiguousarray(
            generate_sine_wave(
                M5_AUDIO_SAMPLE_RATE_HZ,
                M5_AUDIO_SAMPLE_COUNT,
                frequency_hz,
                amplitude=0.25,
                phase_radians=phase_radians,
            ),
            dtype=np.float32,
        )
        for source_id, (frequency_hz, phase_radians) in definitions.items()
    }


def _artifact_record(artifact: Float32WavArtifact, *, output: Path) -> dict[str, Any]:
    return {
        "wav": artifact.audio_path.relative_to(output).as_posix(),
        "sidecar": artifact.sidecar_path.relative_to(output).as_posix(),
        "sample_rate_hz": artifact.sample_rate_hz,
        "frame_count": artifact.frame_count,
        "channel_count": artifact.channel_count,
    }


def _render_layout(
    *,
    output: Path,
    inputs: CurrentM1ResearchAudioInputs,
    layout: CurrentM1PairLayout,
    dry_by_source: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    channel_count = _LAYOUT_CHANNELS[layout.layout]
    spatial_format = copy.deepcopy(dict(layout.spatial_format))
    layout_identity_metadata: dict[str, Any] = {
        "spatial_format": spatial_format,
    }
    if layout.layout == "foa":
        layout_identity_metadata["hrtf_used"] = False
    else:
        layout_identity_metadata["hrtf_asset_id"] = layout.hrtf_asset_id
    maximum_ir_count = max(
        int(layout.pair_samples[source_id].shape[1])
        for source_id in CURRENT_M1_RESEARCH_SOURCE_IDS
    )
    rirs = np.zeros((1, 2, channel_count, maximum_ir_count), dtype=np.float64)
    rir_lengths = np.zeros((1, 2), dtype=np.uint32)
    for source_index, source_id in enumerate(CURRENT_M1_RESEARCH_SOURCE_IDS):
        pair = layout.pair_samples[source_id]
        length = int(pair.shape[1])
        rirs[0, source_index, :, :length] = pair
        rir_lengths[0, source_index] = length

    stems, mixture = render_dynamic_stems_and_mix(
        dry_by_source,
        rirs,
        rir_lengths,
        source_ids=CURRENT_M1_RESEARCH_SOURCE_IDS,
        keyframe_samples=(0,),
        output_sample_count=M5_AUDIO_SAMPLE_COUNT,
    )
    stem_records: dict[str, dict[str, Any]] = {}
    full_tail_counts: dict[str, int] = {}
    for source_id in CURRENT_M1_RESEARCH_SOURCE_IDS:
        result: DynamicStemResult = stems[source_id]
        full_tail_count = int(result.full_tail.shape[1])
        full_tail_counts[source_id] = full_tail_count
        artifact = write_float32_wav(
            output / layout.layout / "stems" / f"{source_id}.wav",
            result.episode,
            M5_AUDIO_SAMPLE_RATE_HZ,
            channel_axis=0,
            metadata={
                "audio_role": f"current_m1_research_{layout.layout}_stem",
                "layout": layout.layout,
                "listener_id": inputs.listener_id,
                "source_id": source_id,
                "pair_receipt": str(layout.receipt_path),
                "pair_wav": layout.pair_wavs[source_id]
                .relative_to(layout.receipt_path.parent)
                .as_posix(),
                "full_tail_frame_count": full_tail_count,
                "episode_crop": [0, M5_AUDIO_SAMPLE_COUNT],
                "resampling": "not_applied",
                "amplitude_normalization": "not_applied",
                "limiter": "not_applied",
                **copy.deepcopy(layout_identity_metadata),
            },
        )
        stem_records[source_id] = _artifact_record(artifact, output=output)
    mixture_artifact = write_float32_wav(
        output / layout.layout / "mix.wav",
        mixture,
        M5_AUDIO_SAMPLE_RATE_HZ,
        channel_axis=0,
        metadata={
            "audio_role": f"current_m1_research_{layout.layout}_mix",
            "layout": layout.layout,
            "listener_id": inputs.listener_id,
            "source_order": list(CURRENT_M1_RESEARCH_SOURCE_IDS),
            "episode_crop": [0, M5_AUDIO_SAMPLE_COUNT],
            "full_tail_frame_count_by_source": full_tail_counts,
            "resampling": "not_applied",
            "amplitude_normalization": "not_applied",
            "limiter": "not_applied",
            **copy.deepcopy(layout_identity_metadata),
        },
    )
    return {
        "channel_count": channel_count,
        "source_order": list(CURRENT_M1_RESEARCH_SOURCE_IDS),
        "keyframe_samples": [0],
        "full_tail_frame_count_by_source": full_tail_counts,
        "episode_crop": [0, M5_AUDIO_SAMPLE_COUNT],
        **layout_identity_metadata,
        "stems": stem_records,
        "mix": _artifact_record(mixture_artifact, output=output),
    }


def render_current_m1_research_audio(
    foa_receipt: str | Path,
    binaural_receipt: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Write dry, FOA and binaural five-second WAVs using only retained IRs."""

    output = _fresh_output_path(output_directory)
    inputs = load_current_m1_research_audio_inputs(foa_receipt, binaural_receipt)
    dry_by_source = _deterministic_dry()

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.mkdir()
    except FileExistsError as error:
        raise CurrentM1ResearchAudioError(
            f"refusing to replace current M1 research audio output: {output}"
        ) from error

    dry_records: dict[str, dict[str, Any]] = {}
    definitions = {
        "source0": {"frequency_hz": 440.0, "amplitude": 0.25, "phase_radians": 0.0},
        "source1": {
            "frequency_hz": 660.0,
            "amplitude": 0.25,
            "phase_radians": math.pi / 2.0,
        },
    }
    try:
        for source_id in CURRENT_M1_RESEARCH_SOURCE_IDS:
            artifact = write_float32_wav(
                output / "dry" / f"{source_id}.wav",
                dry_by_source[source_id][np.newaxis, :],
                M5_AUDIO_SAMPLE_RATE_HZ,
                channel_axis=0,
                metadata={
                    "audio_role": "deterministic_current_m1_dry",
                    "source_id": source_id,
                    **definitions[source_id],
                    "resampling": "not_applied",
                    "amplitude_normalization": "not_applied",
                    "limiter": "not_applied",
                },
            )
            dry_records[source_id] = _artifact_record(artifact, output=output)

        layouts = {
            "foa": _render_layout(
                output=output,
                inputs=inputs,
                layout=inputs.foa,
                dry_by_source=dry_by_source,
            ),
            "binaural": _render_layout(
                output=output,
                inputs=inputs,
                layout=inputs.binaural,
                dry_by_source=dry_by_source,
            ),
        }
        receipt = {
            "status": "pass",
            "research_status": "research_candidate",
            "research_only": True,
            "episode_counted": False,
            "formal_dataset_count": 0,
            "qualification": False,
            "qualification_claim": False,
            "request_id": inputs.request_id,
            "room_id": inputs.room_id,
            "listener_id": inputs.listener_id,
            "inputs": {
                "foa_receipt": str(inputs.foa.receipt_path),
                "binaural_receipt": str(inputs.binaural.receipt_path),
            },
            "source_positions_m": {
                source_id: list(inputs.source_positions_m[source_id])
                for source_id in CURRENT_M1_RESEARCH_SOURCE_IDS
            },
            "audio": {
                "sample_rate_hz": M5_AUDIO_SAMPLE_RATE_HZ,
                "sample_count": M5_AUDIO_SAMPLE_COUNT,
                "duration_seconds": 5.0,
                "processing": {
                    "resampling": "not_applied",
                    "amplitude_normalization": "not_applied",
                    "limiter": "not_applied",
                    "tail_policy": "full_length_recorded_episode_crop_written",
                },
                "dry": dry_records,
                "layouts": layouts,
            },
        }
        write_json(output / "research_receipt.json", receipt)
    except (AudioContractError, OSError, ValueError) as error:
        if isinstance(error, CurrentM1ResearchAudioError):
            raise
        raise CurrentM1ResearchAudioError(str(error)) from error
    return receipt


__all__ = [
    "CURRENT_M1_RESEARCH_SOURCE_IDS",
    "CurrentM1PairLayout",
    "CurrentM1ResearchAudioError",
    "CurrentM1ResearchAudioInputs",
    "load_current_m1_research_audio_inputs",
    "render_current_m1_research_audio",
]
