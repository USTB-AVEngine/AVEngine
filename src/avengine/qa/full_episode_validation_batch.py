"""Build stable validation batches from explicit full-Episode bindings only."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from avengine.contracts.json_io import load_json

REQUEST_SCHEMA = "avengine_full_episode_validation_batch_request_v1"
BATCH_SCHEMA = "avengine_full_episode_validation_batch_v1"
SEMANTIC_AUTHORITY_SCHEMA = "avengine_full_episode_semantic_authority_v1"
ROOM_READINESS_SCHEMA = "avengine_room_readiness_binding_v1"
STATIC_FINALIZATION_SCHEMA = (
    "avengine_native_strict_two_human_full75_canary_finalization_v1"
)
DYNAMIC_FINALIZATION_SCHEMA = (
    "avengine_native_strict_two_human_dynamic_full75_finalization_v1"
)
ARTIFACT_ROLES = (
    "source_finalization",
    "capture_manifest",
    "audiovisual_mp4",
    "binaural_wav",
    "pixel_visibility_truth",
    "runtime_readbacks",
)

_REF_FIELDS = {"path"}
_SELECTED_AUTHORITY_FIELDS = {
    "authority_ref",
    "authority_selector",
}
_ROOM_READINESS_CORE_FIELDS = {
    "schema",
    "status",
    "full75_capture_ready",
    "scene_id",
    "room_id",
    "room_variant_id",
    "map_id",
}
_ENTRY_FIELDS = {
    "finalization_ref",
    "semantic_authority_ref",
    "semantic_selector",
    "artifacts",
}
_DYNAMIC_ENTRY_FIELD = "dynamic_audio_authority"
_DYNAMIC_AUXILIARY_FIELDS = {
    "binaural_delivery_ref",
    "binaural_samples_ref",
    "binaural_wav_sidecar_ref",
}
_SAMPLE_ROW_REQUIRED_FIELDS = {
    "asset_ids_by_source_slot",
    "audio",
    "audio_program_binding",
    "audio_program_instance_path",
    "both_sources_active",
    "episode_id",
    "sample_id",
    "sensor_rig_trajectory",
    "source_activity_contract",
    "source_activity_summary",
    "variant_index",
}
_SAMPLE_AUDIO_REQUIRED_FIELDS = {
    "channel_count",
    "layout",
    "mixture",
    "mixture_is_exact_stem_sum_before_delivery",
    "peak_absolute",
    "sample_count",
    "sample_rate_hz",
    "stems",
    "stems_retained",
}
_SAMPLE_MIXTURE_FIELDS = {
    "path",
    "sidecar_path",
}
_WAV_SIDECAR_REQUIRED_FIELDS = {
    "api_array_layout",
    "audio_file",
    "bits_per_sample",
    "channel_count",
    "container",
    "endianness",
    "file_interleave",
    "format_tag",
    "frame_count",
    "metadata",
    "sample_encoding",
    "sample_rate_hz",
    "schema",
}
_WAV_METADATA_REQUIRED_FIELDS = {
    "audio_program_binding",
    "audio_program_instance_path",
    "audio_program_mode",
    "episode_id",
    "limiting",
    "mixture",
    "normalization",
    "role",
    "sample_id",
    "variant_index",
}
_AUDIO_PROGRAM_BINDING_REQUIRED_FIELDS = {
    "audio_program_ref",
    "source_endpoint_to_source_slot",
    "variant_id",
}
_DELIVERY_SCHEMA = "avengine_m7_asset_bound_binaural_batch_delivery_v1"
_SAMPLES_SCHEMA = "avengine_m7_asset_bound_binaural_training_samples_v1"
_WAV_SIDECAR_SCHEMA = "avengine_float32_wav_sidecar_v1"
_WAV_MIXTURE_ROLE = "m7_asset_bound_binaural_training_mixture"
_ACOUSTIC_BINDING_SCHEMA = "avengine_rir_cache_acoustic_selection_binding_v1"


class FullEpisodeValidationError(ValueError):
    """An explicit path, semantic authority, or Episode gate drifted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FullEpisodeValidationError(message)


def _file_ref(path: str | Path) -> dict[str, Any]:
    raw_path = Path(path)
    _require(raw_path.is_absolute(), f"artifact path is not absolute: {raw_path}")
    try:
        resolved = raw_path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise FullEpisodeValidationError(
            f"artifact does not exist: {raw_path}"
        ) from error
    _require(resolved.is_file(), f"artifact is not a regular file: {resolved}")
    return {"path": str(resolved)}


def _validate_declared_ref(value: object, *, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} binding is missing")
    _require(set(value) == _REF_FIELDS, f"{label} binding fields drifted")
    path = value.get("path")
    _require(isinstance(path, str) and bool(path), f"{label} path is invalid")
    return _file_ref(path)


def _decode_pointer_token(raw_token: str) -> str:
    decoded: list[str] = []
    index = 0
    while index < len(raw_token):
        character = raw_token[index]
        if character != "~":
            decoded.append(character)
            index += 1
            continue
        _require(index + 1 < len(raw_token), "invalid JSON pointer escape")
        escaped = raw_token[index + 1]
        _require(escaped in {"0", "1"}, "invalid JSON pointer escape")
        decoded.append("~" if escaped == "0" else "/")
        index += 2
    return "".join(decoded)


def _select_json_pointer(document: object, selector: str) -> object:
    _require(isinstance(selector, str), "semantic selector must be a string")
    _require(selector == "" or selector.startswith("/"), "invalid JSON pointer")
    current = document
    if not selector:
        return current
    for raw_token in selector[1:].split("/"):
        token = _decode_pointer_token(raw_token)
        if isinstance(current, Mapping):
            _require(token in current, f"semantic selector misses key {token!r}")
            current = current[token]
        elif isinstance(current, list):
            _require(
                token == "0" or (token.isdigit() and not token.startswith("0")),
                "semantic list selector is not a canonical index",
            )
            index = int(token)
            _require(index < len(current), "semantic selector index is out of range")
            current = current[index]
        else:
            raise FullEpisodeValidationError("semantic selector traverses a scalar")
    return current


def _selected_authority(
    value: Mapping[str, Any], *, label: str
) -> tuple[dict[str, Any], str, object]:
    ref = _validate_declared_ref(
        value.get("authority_ref"), label=f"{label}.authority_ref"
    )
    selector = value.get("authority_selector")
    _require(isinstance(selector, str), f"{label}.authority_selector is invalid")
    authority = load_json(ref["path"])
    selected = _select_json_pointer(authority, selector)
    return ref, selector, selected


def _string(value: object, *, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} is missing")
    return value


def _integer(value: object, *, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool), f"{label} is invalid"
    )
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _semantic_actor(value: object, *, label: str) -> dict[str, Any]:
    actor = _mapping(value, label=label)
    role = _string(actor.get("role"), label=f"{label}.role")
    _require(role in {"target", "distractor"}, f"{label}.role is invalid")
    side = _string(actor.get("side"), label=f"{label}.side")
    _require(side in {"left", "right"}, f"{label}.side is invalid")
    normalized = {
        "source_slot_id": _string(
            actor.get("source_slot_id"), label=f"{label}.source_slot_id"
        ),
        "role": role,
        "side": side,
        "identity_id": _string(actor.get("identity_id"), label=f"{label}.identity_id"),
        "asset_id": _string(actor.get("asset_id"), label=f"{label}.asset_id"),
        "asset_revision": _string(
            actor.get("asset_revision"), label=f"{label}.asset_revision"
        ),
    }
    if role == "target":
        normalized.update(
            {
                "voice_id": _string(actor.get("voice_id"), label=f"{label}.voice_id"),
                "content_id": _string(
                    actor.get("content_id"), label=f"{label}.content_id"
                ),
                "sound_asset_id": _string(
                    actor.get("sound_asset_id"), label=f"{label}.sound_asset_id"
                ),
            }
        )
    else:
        _require(
            actor.get("voice_policy") == "silent",
            f"{label}.voice_policy must be silent",
        )
        normalized["voice_policy"] = "silent"
    return normalized


def _validate_semantic_value(
    value: object, *, normalized_finalization: Mapping[str, Any], label: str
) -> dict[str, Any]:
    semantic = _mapping(value, label=label)
    _require(
        semantic.get("schema") == SEMANTIC_AUTHORITY_SCHEMA,
        f"{label}.schema drifted",
    )
    episode_id = _string(semantic.get("episode_id"), label=f"{label}.episode_id")
    _require(
        episode_id == normalized_finalization["episode_id"],
        f"{label}.episode_id does not match finalization",
    )
    mechanism = _string(semantic.get("mechanism"), label=f"{label}.mechanism")
    _require(
        mechanism == normalized_finalization["mechanism"],
        f"{label}.mechanism does not match finalization",
    )
    _require(
        semantic.get("qualification_claim") is False,
        f"{label}.qualification_claim must remain false",
    )
    _require(
        _integer(
            semantic.get("formal_episode_count"),
            label=f"{label}.formal_episode_count",
        )
        == 0,
        f"{label}.formal_episode_count must remain zero",
    )

    scene = _mapping(semantic.get("scene"), label=f"{label}.scene")
    normalized_scene = {
        key: _string(scene.get(key), label=f"{label}.scene.{key}")
        for key in ("scene_id", "room_id", "room_variant_id", "map_id")
    }
    readiness = _mapping(
        scene.get("room_readiness"), label=f"{label}.scene.room_readiness"
    )
    _require(
        set(readiness) == _ROOM_READINESS_CORE_FIELDS | _SELECTED_AUTHORITY_FIELDS,
        f"{label}.scene.room_readiness fields drifted",
    )
    _require(
        readiness.get("schema") == ROOM_READINESS_SCHEMA,
        f"{label}.scene.room_readiness.schema drifted",
    )
    _require(
        readiness.get("status") == "pass"
        and readiness.get("full75_capture_ready") is True,
        f"{label}.scene room is not ready for full75 capture",
    )
    for key in ("scene_id", "room_id", "room_variant_id", "map_id"):
        _require(
            readiness.get(key) == normalized_scene[key],
            f"{label}.scene.room_readiness.{key} drifted",
        )
    readiness_ref, readiness_selector, selected_readiness = _selected_authority(
        readiness,
        label=f"{label}.scene.room_readiness",
    )
    selected_readiness_mapping = _mapping(
        selected_readiness,
        label=f"{label}.scene.room_readiness.selected_authority_value",
    )
    _require(
        set(selected_readiness_mapping) == _ROOM_READINESS_CORE_FIELDS,
        f"{label}.scene.room_readiness selected authority fields drifted",
    )
    readiness_core = {
        "schema": ROOM_READINESS_SCHEMA,
        "status": "pass",
        "full75_capture_ready": True,
        **{key: normalized_scene[key] for key in normalized_scene},
    }
    _require(
        dict(selected_readiness_mapping) == readiness_core,
        f"{label}.scene.room_readiness selected authority does not bind the room",
    )
    normalized_scene["room_readiness"] = {
        **readiness_core,
        "authority_ref": readiness_ref,
        "authority_selector": readiness_selector,
    }

    camera = _mapping(semantic.get("camera"), label=f"{label}.camera")
    normalized_camera = {
        "camera_cluster_id": _string(
            camera.get("camera_cluster_id"),
            label=f"{label}.camera.camera_cluster_id",
        ),
    }

    actors_raw = semantic.get("actors")
    _require(
        isinstance(actors_raw, Sequence) and not isinstance(actors_raw, (str, bytes)),
        f"{label}.actors must be a list",
    )
    _require(len(actors_raw) == 2, f"{label}.actors must contain exactly two actors")
    actors = [
        _semantic_actor(actor, label=f"{label}.actors[{index}]")
        for index, actor in enumerate(actors_raw)
    ]
    roles = [actor["role"] for actor in actors]
    slots = [actor["source_slot_id"] for actor in actors]
    _require(
        set(roles) == {"target", "distractor"} and len(set(roles)) == 2,
        f"{label}.actors must bind one target and one distractor",
    )
    _require(len(set(slots)) == 2, f"{label}.actors source slots must be unique")
    actors_by_role = {actor["role"]: actor for actor in actors}
    _require(
        actors_by_role["target"]["side"] != actors_by_role["distractor"]["side"],
        f"{label}.actors target and distractor share a side",
    )
    normalized_actors = [actors_by_role["target"], actors_by_role["distractor"]]

    timeline = _mapping(semantic.get("timeline"), label=f"{label}.timeline")
    _require(
        _integer(timeline.get("frame_count"), label=f"{label}.timeline.frame_count")
        == 75,
        f"{label}.timeline must contain 75 frames",
    )
    frame_rate = timeline.get("frame_rate_hz")
    duration = timeline.get("duration_seconds")
    _require(
        isinstance(frame_rate, (int, float))
        and not isinstance(frame_rate, bool)
        and frame_rate == 15,
        f"{label}.timeline.frame_rate_hz must be 15",
    )
    _require(
        isinstance(duration, (int, float))
        and not isinstance(duration, bool)
        and duration == 5,
        f"{label}.timeline.duration_seconds must be 5",
    )
    normalized_timeline = {
        "frame_count": 75,
        "frame_rate_hz": 15,
        "duration_seconds": 5,
    }

    rir_job_ids = semantic.get("rir_job_ids")
    _require(
        isinstance(rir_job_ids, Sequence)
        and not isinstance(rir_job_ids, (str, bytes))
        and len(rir_job_ids) == 2
        and all(isinstance(value, str) and bool(value) for value in rir_job_ids)
        and len(set(rir_job_ids)) == 2,
        f"{label}.rir_job_ids must contain two distinct IDs",
    )

    question = _mapping(semantic.get("question"), label=f"{label}.question")
    prompt = _string(question.get("prompt"), label=f"{label}.question.prompt")
    options = question.get("options")
    _require(
        isinstance(options, Sequence)
        and not isinstance(options, (str, bytes))
        and len(options) >= 2
        and all(isinstance(option, str) and bool(option) for option in options)
        and len(set(options)) == len(options),
        f"{label}.question.options are invalid",
    )
    correct_index = question.get("correct_index")
    _require(
        isinstance(correct_index, int)
        and not isinstance(correct_index, bool)
        and 0 <= correct_index < len(options),
        f"{label}.question.correct_index is invalid",
    )
    normalized_question = {
        "prompt": prompt,
        "options": list(options),
        "correct_index": correct_index,
        "option_order_id": _string(
            question.get("option_order_id"),
            label=f"{label}.question.option_order_id",
        ),
    }

    independence_fields = {
        "scene": {
            key: normalized_scene[key]
            for key in ("scene_id", "room_id", "room_variant_id", "map_id")
        },
        "mechanism": mechanism,
        "camera": {
            "camera_cluster_id": normalized_camera["camera_cluster_id"],
        },
        "actors": [
            {
                key: actor[key]
                for key in (
                    "source_slot_id",
                    "role",
                    "identity_id",
                    "asset_id",
                    "asset_revision",
                )
            }
            for actor in normalized_actors
        ],
    }
    return {
        "schema": SEMANTIC_AUTHORITY_SCHEMA,
        "episode_id": episode_id,
        "mechanism": mechanism,
        "scene": normalized_scene,
        "camera": normalized_camera,
        "actors": normalized_actors,
        "timeline": normalized_timeline,
        "rir_job_ids": list(rir_job_ids),
        "question": normalized_question,
        "formal_episode_count": 0,
        "qualification_claim": False,
        "independence": {
            "canonical_fields": independence_fields,
            "excluded_fields": ["rir_job_ids", "question", "target_audio"],
        },
    }


def _static_normalization(finalization: Mapping[str, Any]) -> dict[str, Any]:
    _require(finalization.get("status") == "pass", "static finalization did not pass")
    _require(
        finalization.get("full75_canary_pass") is True,
        "static full75 gate did not pass",
    )
    frame_count = _integer(
        finalization.get("captured_frame_count"), label="static captured frame count"
    )
    _require(frame_count == 75, "static finalization is not full75")
    _require(
        finalization.get("qualification_claim") is False,
        "static qualification claim must remain false",
    )
    formal_count = _integer(
        finalization.get("formal_episode_count"), label="static formal episode count"
    )
    _require(formal_count == 0, "static validation cannot admit formal Episodes")
    declared = finalization.get("artifacts")
    _require(isinstance(declared, Mapping), "static artifact declarations are missing")
    paths = {
        "capture_manifest": declared.get("capture_manifest"),
        "audiovisual_mp4": declared.get("binaural_video"),
        "binaural_wav": declared.get("binaural_wav"),
        "pixel_visibility_truth": declared.get("pixel_visibility_truth"),
        "runtime_readbacks": declared.get("runtime_readbacks"),
    }
    _require(
        all(isinstance(value, str) and value for value in paths.values()),
        "static artifact declaration drifted",
    )
    return {
        "source_kind": "static",
        "episode_id": _string(finalization.get("episode_id"), label="episode_id"),
        "mechanism": "both_static",
        "captured_frame_count": frame_count,
        "formal_episode_count": formal_count,
        "qualification_claim": False,
        "declared_artifact_paths": paths,
    }


def _dynamic_binaural_authority(
    auxiliary: object,
    *,
    finalization: Mapping[str, Any],
    episode_id: str,
    binaural_wav_ref: Mapping[str, Any],
    semantic: Mapping[str, Any],
) -> dict[str, Any]:
    bound = _mapping(auxiliary, label="dynamic auxiliary authority")
    _require(
        set(bound) == _DYNAMIC_AUXILIARY_FIELDS,
        "dynamic auxiliary authority fields drifted",
    )
    artifacts = finalization.get("artifacts")
    _require(
        isinstance(artifacts, Mapping), "dynamic artifact declarations are missing"
    )
    samples_ref = _validate_declared_ref(
        bound.get("binaural_samples_ref"),
        label="dynamic binaural samples authority",
    )
    delivery_ref = _validate_declared_ref(
        bound.get("binaural_delivery_ref"),
        label="dynamic binaural delivery authority",
    )
    sidecar_ref = _validate_declared_ref(
        bound.get("binaural_wav_sidecar_ref"),
        label="dynamic binaural WAV sidecar authority",
    )
    materialization_root_raw = artifacts.get("materialization_root")
    _require(
        isinstance(materialization_root_raw, str)
        and bool(materialization_root_raw)
        and Path(materialization_root_raw).is_absolute(),
        "dynamic materialization root is missing or not absolute",
    )
    try:
        materialization_root = Path(materialization_root_raw).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise FullEpisodeValidationError(
            "dynamic materialization root does not exist"
        ) from error
    _require(
        materialization_root.is_dir(),
        "dynamic materialization root is not a directory",
    )
    expected_authority_root = materialization_root / "binaural_v1"
    _require(
        Path(delivery_ref["path"]).name == "delivery.json"
        and Path(samples_ref["path"]).name == "samples.json"
        and Path(delivery_ref["path"]).resolve().parent
        == Path(samples_ref["path"]).resolve().parent,
        "dynamic delivery and samples authorities are not a sibling pair",
    )
    _require(
        Path(delivery_ref["path"]).resolve().parent == expected_authority_root,
        "dynamic audio authorities are outside the finalization materialization root",
    )
    declared_delivery = artifacts.get("binaural_delivery")
    _require(
        isinstance(declared_delivery, str)
        and Path(declared_delivery).is_absolute()
        and Path(delivery_ref["path"]).resolve() == Path(declared_delivery).resolve(),
        "dynamic auxiliary delivery is not declared by finalization",
    )
    delivery = _mapping(
        load_json(delivery_ref["path"]), label="dynamic binaural delivery"
    )
    _require(
        delivery.get("schema") == _DELIVERY_SCHEMA
        and delivery.get("status") == "pass"
        and delivery.get("qualification_claim") is False,
        "dynamic binaural delivery contract drifted",
    )
    samples = _mapping(
        load_json(samples_ref["path"]), label="dynamic binaural samples authority"
    )
    rows = samples.get("samples")
    _require(
        samples.get("schema") == _SAMPLES_SCHEMA
        and samples.get("status") == "pass"
        and isinstance(rows, list)
        and bool(rows),
        "dynamic binaural samples authority drifted",
    )
    _require(
        all(
            isinstance(row, Mapping)
            and isinstance(row.get("episode_id"), str)
            and bool(row.get("episode_id"))
            for row in rows
        ),
        "dynamic samples rows have invalid Episode identities",
    )
    episode_ids = {row["episode_id"] for row in rows}
    delivery_episode_count = _integer(
        delivery.get("episode_count"), label="dynamic delivery episode_count"
    )
    delivery_sample_count = _integer(
        delivery.get("sample_count"), label="dynamic delivery sample_count"
    )
    samples_sample_count = _integer(
        samples.get("sample_count"), label="dynamic samples sample_count"
    )
    _require(
        delivery_episode_count == len(episode_ids)
        and delivery_sample_count == len(rows)
        and samples_sample_count == len(rows),
        "dynamic delivery counts do not match the samples authority",
    )
    variants_per_episode = delivery.get("variants_per_episode")
    _require(
        isinstance(variants_per_episode, int)
        and not isinstance(variants_per_episode, bool)
        and variants_per_episode > 0,
        "dynamic variants_per_episode is invalid",
    )
    episode_row_counts = {
        candidate_episode_id: sum(
            1
            for row in rows
            if isinstance(row, Mapping)
            and row.get("episode_id") == candidate_episode_id
        )
        for candidate_episode_id in episode_ids
    }
    _require(
        all(count == variants_per_episode for count in episode_row_counts.values()),
        "dynamic variants_per_episode does not match samples rows",
    )

    wav_path = Path(binaural_wav_ref["path"])
    _require(
        wav_path.suffix.lower() == ".wav" and bool(wav_path.stem),
        "dynamic exact6 WAV path is invalid",
    )
    selected_sample_id = wav_path.stem
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("episode_id") == episode_id
        and row.get("sample_id") == selected_sample_id
    ]
    _require(
        len(matches) == 1,
        "dynamic WAV must select exactly one Episode/sample row",
    )
    sample = matches[0]
    sample_id = sample.get("sample_id")
    _require(
        isinstance(sample_id, str)
        and bool(sample_id)
        and Path(sample_id).name == sample_id,
        "dynamic binaural sample_id is invalid",
    )
    _require(
        _SAMPLE_ROW_REQUIRED_FIELDS.issubset(sample),
        "dynamic binaural sample row lacks required authority fields",
    )
    semantic_actors = semantic.get("actors")
    _require(
        isinstance(semantic_actors, list),
        "dynamic semantic actors are missing",
    )
    expected_assets = {
        actor["source_slot_id"]: actor["asset_id"] for actor in semantic_actors
    }
    sample_assets = _mapping(
        sample.get("asset_ids_by_source_slot"),
        label="dynamic sample assets by source slot",
    )
    _require(
        dict(sample_assets) == expected_assets,
        "dynamic sample assets do not match semantic actors",
    )
    audio = _mapping(sample.get("audio"), label="dynamic binaural sample audio")
    _require(
        _SAMPLE_AUDIO_REQUIRED_FIELDS.issubset(audio),
        "dynamic binaural sample audio lacks required fields",
    )
    mixture = _mapping(audio.get("mixture"), label="dynamic binaural sample mixture")
    _require(
        _SAMPLE_MIXTURE_FIELDS.issubset(mixture),
        "dynamic binaural sample mixture lacks required paths",
    )
    mixture_path = _string(
        mixture.get("path"), label="dynamic binaural sample mixture.path"
    )
    sidecar_name = _string(
        mixture.get("sidecar_path"),
        label="dynamic binaural sample mixture.sidecar_path",
    )
    _require(
        Path(mixture_path).name == mixture_path and mixture_path == f"{sample_id}.wav",
        "dynamic binaural sample mixture path is not its bound basename",
    )
    _require(
        Path(sidecar_name).name == sidecar_name
        and sidecar_name == f"{mixture_path}.json",
        "dynamic binaural sample sidecar path is not its bound basename",
    )
    expected_audio_root = expected_authority_root / "audio" / "binaural"
    _require(
        Path(binaural_wav_ref["path"]).resolve() == expected_audio_root / mixture_path,
        "dynamic exact6 WAV is not selected by samples authority",
    )
    _require(
        Path(sidecar_ref["path"]).resolve() == expected_audio_root / sidecar_name,
        "dynamic sidecar ref is not selected by samples authority",
    )

    sidecar = _mapping(
        load_json(sidecar_ref["path"]), label="dynamic binaural WAV sidecar"
    )
    _require(
        _WAV_SIDECAR_REQUIRED_FIELDS.issubset(sidecar),
        "dynamic binaural WAV sidecar lacks required fields",
    )
    _require(
        sidecar.get("schema") == _WAV_SIDECAR_SCHEMA,
        "dynamic binaural WAV sidecar schema drifted",
    )
    _require(
        sidecar.get("audio_file") == mixture_path,
        "dynamic WAV sidecar audio_file does not match samples mixture",
    )
    for sidecar_field, sample_field in (
        ("frame_count", "sample_count"),
        ("sample_rate_hz", "sample_rate_hz"),
        ("channel_count", "channel_count"),
    ):
        _require(
            sidecar.get(sidecar_field) == audio.get(sample_field),
            f"dynamic WAV sidecar {sidecar_field} does not match samples audio",
        )

    metadata = _mapping(
        sidecar.get("metadata"), label="dynamic binaural WAV sidecar.metadata"
    )
    _require(
        _WAV_METADATA_REQUIRED_FIELDS.issubset(metadata),
        "dynamic binaural WAV sidecar metadata lacks required fields",
    )
    _require(
        metadata.get("episode_id") == episode_id
        and metadata.get("sample_id") == sample_id
        and metadata.get("role") == _WAV_MIXTURE_ROLE
        and metadata.get("audio_program_mode") is True
        and isinstance(metadata.get("mixture"), str)
        and bool(metadata.get("mixture")),
        "dynamic WAV sidecar metadata does not bind the Episode mixture",
    )
    for field in (
        "audio_program_instance_path",
        "variant_index",
    ):
        _require(
            metadata.get(field) == sample.get(field),
            f"dynamic WAV sidecar metadata {field} does not match samples row",
        )
    delivery_acoustic = _mapping(
        delivery.get("acoustic_selection_binding"),
        label="dynamic delivery acoustic_selection_binding",
    )
    samples_acoustic = _mapping(
        samples.get("acoustic_selection_binding"),
        label="dynamic samples acoustic_selection_binding",
    )
    acoustic_fields = (
        "schema",
        "binding_id",
        "selection_mode",
        "profile_ref",
        "room_ref",
        "registry_selection_applied",
    )
    acoustic_semantics = {key: delivery_acoustic.get(key) for key in acoustic_fields}
    _require(
        delivery_acoustic.get("schema") == _ACOUSTIC_BINDING_SCHEMA
        and {key: samples_acoustic.get(key) for key in acoustic_fields}
        == acoustic_semantics,
        "dynamic delivery and samples acoustic selections do not match",
    )
    sample_program = _mapping(
        sample.get("audio_program_binding"),
        label="dynamic sample audio_program_binding",
    )
    metadata_program = _mapping(
        metadata.get("audio_program_binding"),
        label="dynamic WAV sidecar metadata.audio_program_binding",
    )
    _require(
        _AUDIO_PROGRAM_BINDING_REQUIRED_FIELDS.issubset(sample_program)
        and _AUDIO_PROGRAM_BINDING_REQUIRED_FIELDS.issubset(metadata_program),
        "dynamic audio-program binding lacks required semantic fields",
    )
    program_ref = _mapping(
        sample_program.get("audio_program_ref"),
        label="dynamic sample audio_program_binding.audio_program_ref",
    )
    program_id = _string(
        program_ref.get("program_id"),
        label="dynamic sample audio program ID",
    )
    program_revision = _string(
        program_ref.get("revision"),
        label="dynamic sample audio program revision",
    )
    program_variant_id = _string(
        sample_program.get("variant_id"),
        label="dynamic sample audio program variant",
    )
    endpoint_slots = _mapping(
        sample_program.get("source_endpoint_to_source_slot"),
        label="dynamic sample audio endpoint-to-slot binding",
    )
    _require(
        bool(endpoint_slots)
        and all(
            isinstance(endpoint, str)
            and bool(endpoint)
            and isinstance(slot, str)
            and bool(slot)
            for endpoint, slot in endpoint_slots.items()
        ),
        "dynamic sample audio endpoint-to-slot binding is invalid",
    )
    metadata_program_ref = _mapping(
        metadata_program.get("audio_program_ref"),
        label="dynamic sidecar audio_program_binding.audio_program_ref",
    )
    metadata_program_semantics = {
        "program_id": metadata_program_ref.get("program_id"),
        "revision": metadata_program_ref.get("revision"),
        "variant_id": metadata_program.get("variant_id"),
        "source_endpoint_to_source_slot": dict(
            _mapping(
                metadata_program.get("source_endpoint_to_source_slot"),
                label="dynamic sidecar audio endpoint-to-slot binding",
            )
        ),
    }
    sample_program_semantics = {
        "program_id": program_id,
        "revision": program_revision,
        "variant_id": program_variant_id,
        "source_endpoint_to_source_slot": dict(endpoint_slots),
    }
    _require(
        metadata_program_semantics == sample_program_semantics,
        "dynamic WAV sidecar audio-program binding does not match samples row",
    )
    program_instance_path = sample.get("audio_program_instance_path")
    instance_relative_path = (
        Path(program_instance_path) if isinstance(program_instance_path, str) else None
    )
    _require(
        isinstance(sample.get("variant_index"), int)
        and not isinstance(sample.get("variant_index"), bool)
        and 0 <= sample["variant_index"] < variants_per_episode
        and instance_relative_path is not None
        and not instance_relative_path.is_absolute()
        and ".." not in instance_relative_path.parts
        and bool(instance_relative_path.parts)
        and metadata.get("audio_program_instance_path") == program_instance_path,
        "dynamic sample variant or audio-program instance path is invalid",
    )
    _require(
        audio.get("layout") == delivery.get("binaural_layout")
        and sample.get("source_activity_contract")
        == delivery.get("source_activity_contract")
        and sample.get("both_sources_active") == delivery.get("both_sources_active"),
        "dynamic sample audio semantics do not match delivery",
    )
    return {
        "samples_ref": samples_ref,
        "delivery_ref": delivery_ref,
        "sidecar_ref": sidecar_ref,
        "sample_id": sample_id,
        "binaural_wav_ref": dict(binaural_wav_ref),
        "variant_index": sample["variant_index"],
        "audio_program": {
            "program_id": program_id,
            "revision": program_revision,
            "variant_id": program_variant_id,
            "instance_path": program_instance_path,
            "source_endpoint_to_source_slot": dict(endpoint_slots),
        },
        "acoustic_selection": acoustic_semantics,
    }


def _dynamic_normalization(finalization: Mapping[str, Any]) -> dict[str, Any]:
    _require(finalization.get("status") == "pass", "dynamic finalization did not pass")
    _require(
        finalization.get("dynamic_full75_canary_pass") is True,
        "dynamic full75 gate did not pass",
    )
    _require(
        finalization.get("cpu_pre_capture_gate_pass") is True,
        "dynamic CPU gate did not pass",
    )
    _require(finalization.get("formal") is False, "dynamic formal flag must be false")
    _require(
        finalization.get("qualification_claim") is False,
        "dynamic qualification claim must remain false",
    )
    formal_count = _integer(
        finalization.get("formal_episode_count"), label="dynamic formal episode count"
    )
    _require(formal_count == 0, "dynamic validation cannot admit formal Episodes")
    capture = finalization.get("capture")
    _require(
        isinstance(capture, Mapping) and capture.get("status") == "pass",
        "dynamic capture gate did not pass",
    )
    frame_count = _integer(
        capture.get("captured_frame_count"), label="dynamic captured frame count"
    )
    _require(frame_count == 75, "dynamic finalization is not full75")
    episode_id = _string(finalization.get("episode_id"), label="episode_id")
    artifacts = finalization.get("artifacts")
    _require(
        isinstance(artifacts, Mapping), "dynamic artifact declarations are missing"
    )
    capture_root_raw = artifacts.get("capture_root")
    _require(
        isinstance(capture_root_raw, str) and bool(capture_root_raw),
        "dynamic capture root is missing",
    )
    capture_root = Path(capture_root_raw)
    _require(capture_root.is_absolute(), "dynamic capture root is not absolute")
    capture_root = capture_root.resolve()
    paths = {
        "capture_manifest": str(capture_root / "manifest.json"),
        "audiovisual_mp4": str(capture_root / "native_rgb_binaural.mp4"),
        "binaural_wav": None,
        "pixel_visibility_truth": str(capture_root / "pixel_visibility_truth.json"),
        "runtime_readbacks": str(capture_root / "runtime_readbacks.json"),
    }
    return {
        "source_kind": "dynamic",
        "episode_id": episode_id,
        "mechanism": _string(finalization.get("mechanism"), label="mechanism"),
        "captured_frame_count": frame_count,
        "formal_episode_count": formal_count,
        "qualification_claim": False,
        "declared_artifact_paths": paths,
    }


def _normalize_finalization(finalization: Mapping[str, Any]) -> dict[str, Any]:
    schema = finalization.get("schema")
    if schema == STATIC_FINALIZATION_SCHEMA:
        normalized = _static_normalization(finalization)
    elif schema == DYNAMIC_FINALIZATION_SCHEMA:
        normalized = _dynamic_normalization(finalization)
    else:
        raise FullEpisodeValidationError(f"unsupported finalization schema: {schema!r}")
    normalized["source_finalization_schema"] = schema
    return normalized


def _validate_entry(value: object, *, index: int) -> dict[str, Any]:
    label = f"episodes[{index}]"
    _require(isinstance(value, Mapping), f"{label} is not an object")
    entry_fields = set(value)
    _require(
        entry_fields
        in (
            _ENTRY_FIELDS,
            _ENTRY_FIELDS | {_DYNAMIC_ENTRY_FIELD},
        ),
        f"{label} fields drifted",
    )
    finalization_ref = _validate_declared_ref(
        value.get("finalization_ref"), label=f"{label}.finalization_ref"
    )
    finalization = load_json(finalization_ref["path"])
    normalized = _normalize_finalization(finalization)

    semantic_ref = _validate_declared_ref(
        value.get("semantic_authority_ref"),
        label=f"{label}.semantic_authority_ref",
    )
    selector = value.get("semantic_selector")
    _require(isinstance(selector, str), f"{label}.semantic_selector is invalid")
    semantic_authority = load_json(semantic_ref["path"])
    selected_value = _select_json_pointer(semantic_authority, selector)
    normalized_semantic = _validate_semantic_value(
        selected_value,
        normalized_finalization=normalized,
        label=f"{label}.selected_semantic_value",
    )

    artifacts = value.get("artifacts")
    _require(isinstance(artifacts, Mapping), f"{label}.artifacts is missing")
    _require(set(artifacts) == set(ARTIFACT_ROLES), f"{label} artifact roles drifted")
    artifact_refs = {
        role: _validate_declared_ref(
            artifacts.get(role), label=f"{label}.artifacts.{role}"
        )
        for role in ARTIFACT_ROLES
    }
    _require(
        artifact_refs["source_finalization"] == finalization_ref,
        f"{label} source_finalization does not equal finalization_ref",
    )
    if normalized["source_kind"] == "dynamic":
        _require(
            _DYNAMIC_ENTRY_FIELD in entry_fields,
            f"{label} dynamic audio authority is missing",
        )
        audio_authority = _dynamic_binaural_authority(
            value.get(_DYNAMIC_ENTRY_FIELD),
            finalization=finalization,
            episode_id=normalized["episode_id"],
            binaural_wav_ref=artifact_refs["binaural_wav"],
            semantic=normalized_semantic,
        )
        normalized["declared_artifact_paths"]["binaural_wav"] = audio_authority[
            "binaural_wav_ref"
        ]["path"]
        normalized["dynamic_audio_authority"] = audio_authority
    else:
        _require(
            _DYNAMIC_ENTRY_FIELD not in entry_fields,
            f"{label} static entry cannot carry dynamic audio authority",
        )
    for role, declared_path in normalized.pop("declared_artifact_paths").items():
        _require(
            Path(artifact_refs[role]["path"]).resolve()
            == Path(declared_path).resolve(),
            f"{label} {role} is not declared by its finalization",
        )
    row = {
        **normalized,
        "finalization_ref": finalization_ref,
        "semantic_authority_ref": semantic_ref,
        "semantic_selector": selector,
        "semantic": normalized_semantic,
        "artifacts": artifact_refs,
    }
    return row


def build_full_episode_validation_batch(request: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize explicitly selected source schemas into one stable batch."""

    _require(set(request) == {"schema", "episodes"}, "request fields drifted")
    _require(request.get("schema") == REQUEST_SCHEMA, "request schema drifted")
    rows = request.get("episodes")
    _require(
        isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)),
        "episodes missing",
    )
    _require(bool(rows), "episodes must not be empty")
    normalized_rows = [
        _validate_entry(row, index=index) for index, row in enumerate(rows)
    ]
    episode_ids = [row["episode_id"] for row in normalized_rows]
    _require(len(set(episode_ids)) == len(episode_ids), "duplicate Episode IDs")
    finalization_paths = [row["finalization_ref"]["path"] for row in normalized_rows]
    _require(
        len(set(finalization_paths)) == len(finalization_paths),
        "duplicate finalization refs",
    )
    independence_units = [
        row["semantic"]["independence"]["canonical_fields"] for row in normalized_rows
    ]
    for index, unit in enumerate(independence_units):
        _require(
            unit not in independence_units[:index],
            "duplicate semantic independence units",
        )
    return {
        "schema": BATCH_SCHEMA,
        "status": "pass",
        "episode_count": len(normalized_rows),
        "formal_episode_count": 0,
        "qualification_claim": False,
        "episodes": normalized_rows,
    }
