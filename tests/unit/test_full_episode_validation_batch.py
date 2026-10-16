from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from avengine.qa.full_episode_validation_batch import (
    ARTIFACT_ROLES,
    BATCH_SCHEMA,
    DYNAMIC_FINALIZATION_SCHEMA,
    REQUEST_SCHEMA,
    ROOM_READINESS_SCHEMA,
    SEMANTIC_AUTHORITY_SCHEMA,
    STATIC_FINALIZATION_SCHEMA,
    FullEpisodeValidationError,
    build_full_episode_validation_batch,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = REPO_ROOT / "tools" / "qa" / "build_full_episode_validation_batch.py"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_bytes(path: Path, value: bytes = b"fixture") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def _ref(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve(strict=True))}


def _semantic_fixture(root: Path, episode_id: str, mechanism: str) -> tuple[Path, str]:
    authority = root / "semantic_authority.json"
    target_side = "left"
    distractor_side = "right"
    scene = {
        "scene_id": f"scene_{episode_id}",
        "room_id": "fixture_room",
        "room_variant_id": f"variant_{episode_id}",
        "map_id": "fixture_map",
    }
    readiness_value = {
        "schema": ROOM_READINESS_SCHEMA,
        "status": "pass",
        "full75_capture_ready": True,
        **scene,
    }
    readiness_authority = root / "room_readiness_authority.json"
    _write_json(readiness_authority, {"records": [readiness_value]})
    scene["room_readiness"] = {
        **readiness_value,
        "authority_ref": _ref(readiness_authority),
        "authority_selector": "/records/0",
    }
    selected = {
        "schema": SEMANTIC_AUTHORITY_SCHEMA,
        "episode_id": episode_id,
        "mechanism": mechanism,
        "scene": scene,
        "camera": {
            "camera_cluster_id": f"camera_{episode_id}",
        },
        "actors": [
            {
                "source_slot_id": "source1",
                "role": "target",
                "side": target_side,
                "identity_id": "adult_target",
                "asset_id": "adult_target_asset",
                "asset_revision": "revision_1",
                "voice_id": "voice_1",
                "content_id": "content_1",
                "sound_asset_id": "speech_1",
            },
            {
                "source_slot_id": "source2",
                "role": "distractor",
                "side": distractor_side,
                "identity_id": "adult_distractor",
                "asset_id": "adult_distractor_asset",
                "asset_revision": "revision_2",
                "voice_policy": "silent",
            },
        ],
        "timeline": {
            "frame_count": 75,
            "frame_rate_hz": 15,
            "duration_seconds": 5,
        },
        "rir_job_ids": [f"rir_{episode_id}_source1", f"rir_{episode_id}_source2"],
        "question": {
            "prompt": "Who spoke?",
            "options": ["Left", "Right"],
            "correct_index": 0,
            "option_order_id": "left-right-v1",
        },
        "formal_episode_count": 0,
        "qualification_claim": False,
    }
    _write_json(
        authority,
        {"policies": [{"policy/name": {"selected~value": selected}}]},
    )
    selector = "/policies/0/policy~1name/selected~0value"
    return authority, selector


def _entry(
    *,
    finalization: Path,
    semantic_authority: Path,
    semantic_selector: str,
    artifacts: dict[str, Path],
) -> dict[str, Any]:
    finalization_ref = _ref(finalization)
    return {
        "finalization_ref": finalization_ref,
        "semantic_authority_ref": _ref(semantic_authority),
        "semantic_selector": semantic_selector,
        "artifacts": {
            "source_finalization": finalization_ref,
            **{role: _ref(path) for role, path in artifacts.items()},
        },
    }


def _static_entry(
    root: Path, episode_id: str = "arbitrary_static_episode"
) -> dict[str, Any]:
    capture_manifest = root / "capture" / "manifest.json"
    audiovisual = root / "capture" / "native_rgb_binaural.mp4"
    binaural = root / "audio" / "mixture.wav"
    pixels = root / "capture" / "pixel_visibility_truth.json"
    readbacks = root / "capture" / "runtime_readbacks.json"
    for path in (capture_manifest, pixels, readbacks):
        _write_json(path, {"fixture": path.name})
    _write_bytes(audiovisual, b"mp4-static")
    _write_bytes(binaural, b"wav-static")
    finalization = root / "static_finalization.json"
    _write_json(
        finalization,
        {
            "schema": STATIC_FINALIZATION_SCHEMA,
            "status": "pass",
            "episode_id": episode_id,
            "full75_canary_pass": True,
            "captured_frame_count": 75,
            "formal_episode_count": 0,
            "qualification_claim": False,
            "artifacts": {
                "capture_manifest": str(capture_manifest.resolve()),
                "binaural_video": str(audiovisual.resolve()),
                "binaural_wav": str(binaural.resolve()),
                "pixel_visibility_truth": str(pixels.resolve()),
                "runtime_readbacks": str(readbacks.resolve()),
            },
        },
    )
    authority, selector = _semantic_fixture(root, episode_id, "both_static")
    return _entry(
        finalization=finalization,
        semantic_authority=authority,
        semantic_selector=selector,
        artifacts={
            "capture_manifest": capture_manifest,
            "audiovisual_mp4": audiovisual,
            "binaural_wav": binaural,
            "pixel_visibility_truth": pixels,
            "runtime_readbacks": readbacks,
        },
    )


def _dynamic_entry(
    root: Path, episode_id: str = "arbitrary_dynamic_episode"
) -> dict[str, Any]:
    capture_root = root / "capture"
    capture_manifest = capture_root / "manifest.json"
    audiovisual = capture_root / "native_rgb_binaural.mp4"
    pixels = capture_root / "pixel_visibility_truth.json"
    readbacks = capture_root / "runtime_readbacks.json"
    for path in (capture_manifest, pixels, readbacks):
        _write_json(path, {"fixture": path.name})
    _write_bytes(audiovisual, b"mp4-dynamic")

    materialization_root = root / "materialization"
    sample_id = f"{episode_id}__v00"
    binaural = (
        materialization_root / "binaural_v1" / "audio" / "binaural" / f"{sample_id}.wav"
    )
    _write_bytes(binaural, b"wav-dynamic")
    audio_program_binding = {
        "audio_program_ref": {
            "program_id": "strict_two_human_audio_v1",
            "revision": "v1",
        },
        "source_endpoint_to_source_slot": {
            "source1_mouth": "source1",
            "source2_mouth": "source2",
        },
        "variant_id": "A",
    }
    acoustic_binding = {
        "schema": "avengine_rir_cache_acoustic_selection_binding_v1",
        "binding_id": None,
        "selection_mode": "explicit_legacy",
        "profile_ref": None,
        "room_ref": None,
        "registry_selection_applied": False,
    }
    program_instance_path = "labels/audio_program_instances/v00.json"
    variant_index = 0
    sidecar = binaural.with_name(f"{binaural.name}.json")
    sidecar_value = {
        "api_array_layout": "channel_major",
        "audio_file": binaural.name,
        "bits_per_sample": 32,
        "channel_count": 2,
        "container": "RIFF/WAVE",
        "endianness": "little",
        "file_interleave": "frame_major",
        "format_tag": 3,
        "frame_count": 80_000,
        "metadata": {
            "audio_program_binding": audio_program_binding,
            "audio_program_instance_path": program_instance_path,
            "audio_program_mode": True,
            "episode_id": episode_id,
            "limiting": False,
            "mixture": "exact_source1_plus_source2_stem_sum",
            "normalization": False,
            "role": "m7_asset_bound_binaural_training_mixture",
            "sample_id": sample_id,
            "variant_index": variant_index,
        },
        "sample_encoding": "IEEE_FLOAT",
        "sample_rate_hz": 16_000,
        "schema": "avengine_float32_wav_sidecar_v1",
    }
    _write_json(sidecar, sidecar_value)
    delivery = materialization_root / "binaural_v1" / "delivery.json"
    _write_json(
        delivery,
        {
            "schema": "avengine_m7_asset_bound_binaural_batch_delivery_v1",
            "status": "pass",
            "acoustic_selection_binding": acoustic_binding,
            "binaural_layout": "native_RLR_HRTF_binaural_left_right",
            "both_sources_active": False,
            "episode_count": 1,
            "sample_count": 1,
            "source_activity_contract": "m6_audio_program_event_windows_v1",
            "variants_per_episode": 1,
            "qualification_claim": False,
        },
    )
    samples = materialization_root / "binaural_v1" / "samples.json"
    _write_json(
        samples,
        {
            "schema": "avengine_m7_asset_bound_binaural_training_samples_v1",
            "status": "pass",
            "acoustic_selection_binding": acoustic_binding,
            "sample_count": 1,
            "samples": [
                {
                    "asset_ids_by_source_slot": {
                        "source1": "adult_target_asset",
                        "source2": "adult_distractor_asset",
                    },
                    "audio": {
                        "channel_count": 2,
                        "layout": "native_RLR_HRTF_binaural_left_right",
                        "mixture": {
                            "path": binaural.name,
                            "peak_absolute": 0.5,
                            "sidecar_path": sidecar.name,
                        },
                        "mixture_is_exact_stem_sum_before_delivery": True,
                        "peak_absolute": 0.5,
                        "sample_count": 80_000,
                        "sample_rate_hz": 16_000,
                        "stems": {},
                        "stems_retained": True,
                    },
                    "audio_program_binding": audio_program_binding,
                    "audio_program_instance_path": program_instance_path,
                    "both_sources_active": False,
                    "episode_id": episode_id,
                    "sample_id": sample_id,
                    "sensor_rig_trajectory": {},
                    "source_activity_contract": "m6_audio_program_event_windows_v1",
                    "source_activity_summary": {},
                    "variant_index": variant_index,
                }
            ],
        },
    )
    finalization = root / "dynamic_finalization.json"
    _write_json(
        finalization,
        {
            "schema": DYNAMIC_FINALIZATION_SCHEMA,
            "status": "pass",
            "episode_id": episode_id,
            "mechanism": "target_moves",
            "dynamic_full75_canary_pass": True,
            "cpu_pre_capture_gate_pass": True,
            "formal": False,
            "formal_episode_count": 0,
            "qualification_claim": False,
            "capture": {"status": "pass", "captured_frame_count": 75},
            "artifacts": {
                "capture_root": str(capture_root.resolve()),
                "materialization_root": str(materialization_root.resolve()),
                "binaural_delivery": str(delivery.resolve()),
            },
        },
    )
    authority, selector = _semantic_fixture(root, episode_id, "target_moves")
    entry = _entry(
        finalization=finalization,
        semantic_authority=authority,
        semantic_selector=selector,
        artifacts={
            "capture_manifest": capture_manifest,
            "audiovisual_mp4": audiovisual,
            "binaural_wav": binaural,
            "pixel_visibility_truth": pixels,
            "runtime_readbacks": readbacks,
        },
    )
    entry["dynamic_audio_authority"] = {
        "binaural_delivery_ref": _ref(delivery),
        "binaural_samples_ref": _ref(samples),
        "binaural_wav_sidecar_ref": _ref(sidecar),
    }
    return entry


def _expand_dynamic_authority_to_multi_episode_multi_variant(
    entry: dict[str, Any],
) -> None:
    samples_path = Path(
        entry["dynamic_audio_authority"]["binaural_samples_ref"]["path"]
    )
    delivery_path = Path(
        entry["dynamic_audio_authority"]["binaural_delivery_ref"]["path"]
    )
    audio_root = samples_path.parent / "audio" / "binaural"
    samples_document = json.loads(samples_path.read_text(encoding="utf-8"))
    base_row = samples_document["samples"][0]
    base_episode_id = base_row["episode_id"]
    rows: list[dict[str, Any]] = []
    for episode_id in (base_episode_id, "other_batch_episode"):
        for variant_index, variant_id in enumerate(("A", "B")):
            sample_id = f"{episode_id}__v{variant_index:02d}"
            wav = audio_root / f"{sample_id}.wav"
            sidecar_path = wav.with_name(f"{wav.name}.json")
            _write_bytes(wav, f"wav-{episode_id}-{variant_index}".encode())
            row = copy.deepcopy(base_row)
            row["episode_id"] = episode_id
            row["sample_id"] = sample_id
            row["variant_index"] = variant_index
            row["audio_program_binding"]["variant_id"] = variant_id
            row["audio_program_instance_path"] = (
                f"labels/audio_program_instances/v{variant_index:02d}.json"
            )
            row["audio"]["mixture"]["path"] = wav.name
            row["audio"]["mixture"]["sidecar_path"] = sidecar_path.name
            sidecar = {
                "api_array_layout": "channel_major",
                "audio_file": wav.name,
                "bits_per_sample": 32,
                "channel_count": 2,
                "container": "RIFF/WAVE",
                "endianness": "little",
                "file_interleave": "frame_major",
                "format_tag": 3,
                "frame_count": 80_000,
                "metadata": {
                    "audio_program_binding": row["audio_program_binding"],
                    "audio_program_instance_path": row["audio_program_instance_path"],
                    "audio_program_mode": True,
                    "episode_id": episode_id,
                    "limiting": False,
                    "mixture": "exact_source1_plus_source2_stem_sum",
                    "normalization": False,
                    "role": "m7_asset_bound_binaural_training_mixture",
                    "sample_id": sample_id,
                    "variant_index": variant_index,
                },
                "sample_encoding": "IEEE_FLOAT",
                "sample_rate_hz": 16_000,
                "schema": "avengine_float32_wav_sidecar_v1",
            }
            _write_json(sidecar_path, sidecar)
            rows.append(row)

    samples_document["sample_count"] = len(rows)
    samples_document["samples"] = rows
    _write_json(samples_path, samples_document)
    delivery = json.loads(delivery_path.read_text(encoding="utf-8"))
    delivery["episode_count"] = 2
    delivery["sample_count"] = len(rows)
    delivery["variants_per_episode"] = 2
    _write_json(delivery_path, delivery)


def _request(*entries: dict[str, Any]) -> dict[str, Any]:
    return {"schema": REQUEST_SCHEMA, "episodes": list(entries)}


def _selected_semantic(entry: dict[str, Any]) -> dict[str, Any]:
    authority = json.loads(
        Path(entry["semantic_authority_ref"]["path"]).read_text(encoding="utf-8")
    )
    return authority["policies"][0]["policy/name"]["selected~value"]


def _replace_selected_semantic(entry: dict[str, Any], selected: dict[str, Any]) -> None:
    authority_path = Path(entry["semantic_authority_ref"]["path"])
    _write_json(
        authority_path,
        {"policies": [{"policy/name": {"selected~value": selected}}]},
    )


def test_normalizes_explicit_static_and_dynamic_entries_without_scanning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(
        _static_entry(tmp_path / "static"),
        _dynamic_entry(tmp_path / "dynamic"),
    )

    def forbidden_scan(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("directory scanning is forbidden")

    monkeypatch.setattr(Path, "glob", forbidden_scan)
    monkeypatch.setattr(Path, "rglob", forbidden_scan)
    monkeypatch.setattr(Path, "iterdir", forbidden_scan)
    first = build_full_episode_validation_batch(request)
    second = build_full_episode_validation_batch(request)

    assert first == second
    assert first["schema"] == BATCH_SCHEMA
    assert first["status"] == "pass"
    assert first["episode_count"] == 2
    assert first["formal_episode_count"] == 0
    assert first["qualification_claim"] is False
    assert [row["source_kind"] for row in first["episodes"]] == [
        "static",
        "dynamic",
    ]
    assert [row["mechanism"] for row in first["episodes"]] == [
        "both_static",
        "target_moves",
    ]
    assert all(row["captured_frame_count"] == 75 for row in first["episodes"])
    assert all(
        set(row["artifacts"]) == set(ARTIFACT_ROLES) for row in first["episodes"]
    )
    assert all(
        row["semantic"]["schema"] == SEMANTIC_AUTHORITY_SCHEMA
        for row in first["episodes"]
    )
    assert all(
        row["semantic"]["independence"]["excluded_fields"]
        == ["rir_job_ids", "question", "target_audio"]
        for row in first["episodes"]
    )
    assert all(
        set(ref) == {"path"}
        for row in first["episodes"]
        for ref in row["artifacts"].values()
    )
    assert set(first) == {
        "schema",
        "status",
        "episode_count",
        "formal_episode_count",
        "qualification_claim",
        "episodes",
    }


def test_selects_one_bound_variant_from_a_real_multi_episode_batch(
    tmp_path: Path,
) -> None:
    entry = _dynamic_entry(tmp_path, "selected_episode")
    _expand_dynamic_authority_to_multi_episode_multi_variant(entry)

    result = build_full_episode_validation_batch(_request(entry))

    selected = result["episodes"][0]["dynamic_audio_authority"]
    assert selected["sample_id"] == "selected_episode__v00"
    assert selected["variant_index"] == 0
    assert selected["audio_program"]["variant_id"] == "A"
    assert selected["audio_program"]["program_id"] == "strict_two_human_audio_v1"
    assert selected["acoustic_selection"]["schema"] == (
        "avengine_rir_cache_acoustic_selection_binding_v1"
    )
    assert set(selected["delivery_ref"]) == {"path"}
    assert set(selected["samples_ref"]) == {"path"}
    assert set(selected["sidecar_ref"]) == {"path"}


@pytest.mark.parametrize("role", ARTIFACT_ROLES)
def test_rejects_each_missing_artifact_role(tmp_path: Path, role: str) -> None:
    entry = _static_entry(tmp_path)
    del entry["artifacts"][role]
    with pytest.raises(FullEpisodeValidationError, match="artifact roles drifted"):
        build_full_episode_validation_batch(_request(entry))


def test_rejects_seventh_artifact_role(tmp_path: Path) -> None:
    entry = _static_entry(tmp_path)
    entry["artifacts"]["contact_sheet"] = entry["artifacts"]["capture_manifest"]
    with pytest.raises(FullEpisodeValidationError, match="artifact roles drifted"):
        build_full_episode_validation_batch(_request(entry))


def test_rejects_non_path_semantic_authority_binding(tmp_path: Path) -> None:
    entry = _static_entry(tmp_path)
    entry["semantic_authority_ref"]["digest"] = "legacy-file-digest"
    with pytest.raises(FullEpisodeValidationError, match="binding fields drifted"):
        build_full_episode_validation_batch(_request(entry))


def test_rejects_legacy_selected_value_digest_field(tmp_path: Path) -> None:
    entry = _static_entry(tmp_path)
    entry["selected_value_digest"] = "legacy-selected-value-digest"
    with pytest.raises(FullEpisodeValidationError, match="fields drifted"):
        build_full_episode_validation_batch(_request(entry))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("episode_id", "other_episode", "does not match finalization"),
        ("mechanism", "camera_pan", "does not match finalization"),
    ],
)
def test_rejects_semantic_identity_drift_from_finalization(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    entry = _static_entry(tmp_path)
    selected = _selected_semantic(entry)
    selected[field] = value
    _replace_selected_semantic(entry, selected)
    with pytest.raises(FullEpisodeValidationError, match=message):
        build_full_episode_validation_batch(_request(entry))


def test_rejects_unready_room_binding(tmp_path: Path) -> None:
    entry = _static_entry(tmp_path)
    selected = _selected_semantic(entry)
    selected["scene"]["room_readiness"]["full75_capture_ready"] = False
    _replace_selected_semantic(entry, selected)
    with pytest.raises(FullEpisodeValidationError, match="room is not ready"):
        build_full_episode_validation_batch(_request(entry))


def test_rejects_bare_or_drifted_room_readiness_authority(tmp_path: Path) -> None:
    bare = _static_entry(tmp_path / "bare")
    selected = _selected_semantic(bare)
    readiness = selected["scene"]["room_readiness"]
    for field in ("authority_ref", "authority_selector"):
        del readiness[field]
    _replace_selected_semantic(bare, selected)
    with pytest.raises(FullEpisodeValidationError, match="fields drifted"):
        build_full_episode_validation_batch(_request(bare))

    drifted = _static_entry(tmp_path / "drifted")
    selected = _selected_semantic(drifted)
    readiness_path = Path(selected["scene"]["room_readiness"]["authority_ref"]["path"])
    authority = json.loads(readiness_path.read_text(encoding="utf-8"))
    authority["records"][0]["status"] = "drifted"
    _write_json(readiness_path, authority)
    with pytest.raises(FullEpisodeValidationError, match="does not bind the room"):
        build_full_episode_validation_batch(_request(drifted))


def test_rejects_missing_target_or_distractor_role(tmp_path: Path) -> None:
    entry = _static_entry(tmp_path)
    selected = _selected_semantic(entry)
    selected["actors"][1]["role"] = "target"
    selected["actors"][1].update(
        {"voice_id": "voice_2", "content_id": "content_2", "sound_asset_id": "speech_2"}
    )
    _replace_selected_semantic(entry, selected)
    with pytest.raises(
        FullEpisodeValidationError, match="one target and one distractor"
    ):
        build_full_episode_validation_batch(_request(entry))


def test_rejects_duplicate_geometry_even_when_audio_and_question_differ(
    tmp_path: Path,
) -> None:
    first = _static_entry(tmp_path / "first", "episode_first")
    second = _static_entry(tmp_path / "second", "episode_second")
    first_semantic = _selected_semantic(first)
    second_semantic = copy.deepcopy(first_semantic)
    second_semantic["episode_id"] = "episode_second"
    target = next(
        actor for actor in second_semantic["actors"] if actor["role"] == "target"
    )
    target.update(
        {
            "voice_id": "different_voice",
            "content_id": "different_content",
            "sound_asset_id": "different_sound",
        }
    )
    second_semantic["rir_job_ids"] = ["different_rir_1", "different_rir_2"]
    second_semantic["question"] = {
        "prompt": "Which person spoke?",
        "options": ["Right", "Left"],
        "correct_index": 1,
        "option_order_id": "right-left-v2",
    }
    _replace_selected_semantic(second, second_semantic)
    with pytest.raises(FullEpisodeValidationError, match="independence units"):
        build_full_episode_validation_batch(_request(first, second))


@pytest.mark.parametrize(
    "selector",
    ["policies/0", "/policies/00", "/policies/0/bad~2escape"],
)
def test_rejects_invalid_or_unresolved_json_pointer(
    tmp_path: Path, selector: str
) -> None:
    entry = _static_entry(tmp_path)
    entry["semantic_selector"] = selector
    with pytest.raises(FullEpisodeValidationError):
        build_full_episode_validation_batch(_request(entry))


def test_rejects_artifact_not_declared_by_finalization(tmp_path: Path) -> None:
    entry = _dynamic_entry(tmp_path)
    substitute = tmp_path / "substitute.mp4"
    _write_bytes(substitute, b"other-mp4")
    entry["artifacts"]["audiovisual_mp4"] = _ref(substitute)
    with pytest.raises(FullEpisodeValidationError, match="not declared"):
        build_full_episode_validation_batch(_request(entry))


def test_rejects_relative_or_non_regular_path_refs(tmp_path: Path) -> None:
    relative = _static_entry(tmp_path / "relative")
    relative["semantic_authority_ref"] = {"path": "semantic_authority.json"}
    with pytest.raises(FullEpisodeValidationError, match="not absolute"):
        build_full_episode_validation_batch(_request(relative))

    non_regular = _static_entry(tmp_path / "non_regular")
    non_regular["semantic_authority_ref"] = {"path": str(tmp_path.resolve())}
    with pytest.raises(FullEpisodeValidationError, match="regular file"):
        build_full_episode_validation_batch(_request(non_regular))


def test_rejects_dynamic_asset_or_acoustic_semantic_drift(tmp_path: Path) -> None:
    asset_drift = _dynamic_entry(tmp_path / "asset_drift")
    samples_path = Path(
        asset_drift["dynamic_audio_authority"]["binaural_samples_ref"]["path"]
    )
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    samples["samples"][0]["asset_ids_by_source_slot"]["source1"] = "other_asset"
    _write_json(samples_path, samples)
    with pytest.raises(FullEpisodeValidationError, match="semantic actors"):
        build_full_episode_validation_batch(_request(asset_drift))

    acoustic_drift = _dynamic_entry(tmp_path / "acoustic_drift")
    samples_path = Path(
        acoustic_drift["dynamic_audio_authority"]["binaural_samples_ref"]["path"]
    )
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    samples["acoustic_selection_binding"]["selection_mode"] = "different"
    _write_json(samples_path, samples)
    with pytest.raises(FullEpisodeValidationError, match="acoustic selections"):
        build_full_episode_validation_batch(_request(acoustic_drift))


def test_rejects_unbound_dynamic_samples_or_exact6_wav(tmp_path: Path) -> None:
    samples_drift = _dynamic_entry(tmp_path / "samples_drift")
    samples_path = Path(
        samples_drift["dynamic_audio_authority"]["binaural_samples_ref"]["path"]
    )
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    samples["samples"][0]["sample_id"] = "unbound_sample"
    _write_json(samples_path, samples)
    with pytest.raises(FullEpisodeValidationError, match="select exactly one"):
        build_full_episode_validation_batch(_request(samples_drift))

    wav_drift = _dynamic_entry(tmp_path / "wav_drift")
    substitute = tmp_path / "wav_drift" / "substitute.wav"
    _write_bytes(substitute, b"different-wav")
    wav_drift["artifacts"]["binaural_wav"] = _ref(substitute)
    with pytest.raises(FullEpisodeValidationError, match="select exactly one"):
        build_full_episode_validation_batch(_request(wav_drift))

    program_drift = _dynamic_entry(tmp_path / "program_drift")
    samples_path = Path(
        program_drift["dynamic_audio_authority"]["binaural_samples_ref"]["path"]
    )
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    samples["samples"][0]["audio_program_binding"]["variant_id"] = "B"
    _write_json(samples_path, samples)
    with pytest.raises(FullEpisodeValidationError, match="audio-program binding"):
        build_full_episode_validation_batch(_request(program_drift))


def test_rejects_freshly_rebound_sidecar_with_episode_metadata_drift(
    tmp_path: Path,
) -> None:
    entry = _dynamic_entry(tmp_path)
    sidecar_path = Path(
        entry["dynamic_audio_authority"]["binaural_wav_sidecar_ref"]["path"]
    )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["metadata"]["episode_id"] = "different_episode"
    _write_json(sidecar_path, sidecar)
    with pytest.raises(FullEpisodeValidationError, match="Episode mixture"):
        build_full_episode_validation_batch(_request(entry))


def test_rejects_missing_or_finalization_unbound_dynamic_audio_authority(
    tmp_path: Path,
) -> None:
    missing = _dynamic_entry(tmp_path / "missing")
    del missing["dynamic_audio_authority"]
    with pytest.raises(FullEpisodeValidationError, match="authority is missing"):
        build_full_episode_validation_batch(_request(missing))

    unbound = _dynamic_entry(tmp_path / "unbound")
    substitute = tmp_path / "unbound" / "other_delivery.json"
    _write_json(
        substitute,
        {
            "status": "pass",
            "episode_count": 1,
            "sample_count": 1,
            "qualification_claim": False,
        },
    )
    finalization_path = Path(unbound["finalization_ref"]["path"])
    finalization = json.loads(finalization_path.read_text(encoding="utf-8"))
    finalization["artifacts"]["binaural_delivery"] = str(substitute.resolve())
    _write_json(finalization_path, finalization)
    finalization_ref = _ref(finalization_path)
    unbound["finalization_ref"] = finalization_ref
    unbound["artifacts"]["source_finalization"] = finalization_ref
    with pytest.raises(FullEpisodeValidationError, match="not declared"):
        build_full_episode_validation_batch(_request(unbound))


def test_rejects_source_finalization_alias(tmp_path: Path) -> None:
    entry = _static_entry(tmp_path)
    entry["artifacts"]["source_finalization"] = entry["semantic_authority_ref"]
    with pytest.raises(FullEpisodeValidationError, match="does not equal"):
        build_full_episode_validation_batch(_request(entry))


def test_rejects_duplicate_derived_episode_id(tmp_path: Path) -> None:
    request = _request(
        _static_entry(tmp_path / "first", "duplicate"),
        _dynamic_entry(tmp_path / "second", "duplicate"),
    )
    with pytest.raises(FullEpisodeValidationError, match="duplicate Episode IDs"):
        build_full_episode_validation_batch(request)


def test_cli_writes_once_and_refuses_overwrite(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "batch.json"
    _write_json(request_path, _request(_static_entry(tmp_path / "static")))
    spec = importlib.util.spec_from_file_location("validation_batch_cli", CLI_PATH)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    assert cli.main(["--request", str(request_path), "--output", str(output_path)]) == 0
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["schema"] == BATCH_SCHEMA
    assert result["episode_count"] == 1
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        cli.main(["--request", str(request_path), "--output", str(output_path)])
