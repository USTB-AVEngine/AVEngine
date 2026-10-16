"""Produce approved full-Episode semantics from explicit runtime authorities."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from avengine.contracts.json_io import load_json

REQUEST_SCHEMA = "avengine_full_episode_semantic_authority_request_v1"
AUTHORITY_SCHEMA = "avengine_full_episode_semantic_authority_collection_v1"
SEMANTIC_SCHEMA = "avengine_full_episode_semantic_authority_v1"
ADAPTER_REGISTRY_SCHEMA = "avengine_full_episode_source_adapter_registry_v1"
LABEL_SCHEMA = "avengine_full_episode_approved_semantic_label_v1"
ROOM_READINESS_SCHEMA = "avengine_room_readiness_binding_v1"

STATIC_FINALIZATION_SCHEMA = (
    "avengine_native_strict_two_human_full75_canary_finalization_v1"
)
DYNAMIC_FINALIZATION_SCHEMA = (
    "avengine_native_strict_two_human_dynamic_full75_finalization_v1"
)
STATIC_CANARY_PLAN_SCHEMA = "avengine_native_strict_two_human_full75_canary_plan_v1"
STATIC_IDENTITY_PLAN_SCHEMA = "avengine_native_strict_two_human_expansion_plan_v1"
SUITE_SCHEMA = "avengine_optional_spear_apartment_suite_v1"
SCENARIO_SCHEMA = "avengine_optional_spear_apartment_scenario_v1"
AUDIO_SAMPLES_SCHEMA = "avengine_m7_asset_bound_binaural_training_samples_v1"
AUDIO_DELIVERY_SCHEMA = "avengine_m7_asset_bound_binaural_batch_delivery_v1"
AUDIO_PROGRAM_SCHEMA = "avengine_m6_audio_program_v1"
SOUND_REGISTRY_SCHEMA = "avengine_m6_sound_asset_registry_v1"
ENDPOINT_REGISTRY_SCHEMA = "avengine_m6_source_endpoint_registry_v1"
RIR_PLAN_SCHEMA = "avengine_room_rir_job_plan_v2"

_SELECTED_REF_FIELDS = {"authority_ref", "authority_selector"}
_FILE_REF_FIELDS = {"path"}
_STATIC_EPISODE_FIELDS = {
    "finalization",
    "planning",
    "suite",
    "audio_sample",
    "audio_delivery",
    "audio_program",
    "audio_event",
    "sound_asset",
    "source_endpoints",
    "rir_jobs",
    "semantic_label",
    "identity_binding",
}
_SCENE_ID_FIELDS = ("scene_id", "room_id", "room_variant_id", "map_id")


class FullEpisodeSemanticAuthorityError(ValueError):
    """A selected runtime authority is absent, unsupported, or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FullEpisodeSemanticAuthorityError(message)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{label} must be a list",
    )
    return value


def _string(value: object, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} is missing")
    return value


def _integer(value: object, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{label} is invalid",
    )
    return value


def _file_ref(value: object, label: str) -> dict[str, str]:
    ref = _mapping(value, label)
    _require(set(ref) == _FILE_REF_FIELDS, f"{label} fields drifted")
    path = Path(_string(ref.get("path"), f"{label}.path"))
    _require(path.is_absolute(), f"{label}.path is not absolute")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise FullEpisodeSemanticAuthorityError(
            f"{label}.path does not exist"
        ) from error
    _require(resolved.is_file(), f"{label}.path is not a regular file")
    return {"path": str(resolved)}


def _declared_file(value: object, label: str) -> str:
    path = Path(_string(value, label))
    _require(path.is_absolute(), f"{label} is not absolute")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise FullEpisodeSemanticAuthorityError(f"{label} does not exist") from error
    _require(resolved.is_file(), f"{label} is not a regular file")
    return str(resolved)


def _decode_pointer_token(token: str) -> str:
    output: list[str] = []
    index = 0
    while index < len(token):
        if token[index] != "~":
            output.append(token[index])
            index += 1
            continue
        _require(index + 1 < len(token), "invalid JSON pointer escape")
        escaped = token[index + 1]
        _require(escaped in {"0", "1"}, "invalid JSON pointer escape")
        output.append("~" if escaped == "0" else "/")
        index += 2
    return "".join(output)


def _select(document: object, selector: str) -> object:
    _require(selector == "" or selector.startswith("/"), "invalid JSON pointer")
    current = document
    if selector == "":
        return current
    for raw_token in selector[1:].split("/"):
        token = _decode_pointer_token(raw_token)
        if isinstance(current, Mapping):
            _require(token in current, f"JSON pointer misses key {token!r}")
            current = current[token]
        elif isinstance(current, list):
            _require(
                token == "0" or (token.isdigit() and not token.startswith("0")),
                "non-canonical JSON pointer index",
            )
            index = int(token)
            _require(index < len(current), "JSON pointer index is out of range")
            current = current[index]
        else:
            raise FullEpisodeSemanticAuthorityError("JSON pointer traverses a scalar")
    return current


def _selected(
    value: object, label: str
) -> tuple[dict[str, str], str, Mapping[str, Any], Mapping[str, Any]]:
    binding = _mapping(value, label)
    _require(set(binding) == _SELECTED_REF_FIELDS, f"{label} fields drifted")
    ref = _file_ref(binding.get("authority_ref"), f"{label}.authority_ref")
    selector = binding.get("authority_selector")
    _require(isinstance(selector, str), f"{label}.authority_selector is invalid")
    document = _mapping(load_json(ref["path"]), f"{label}.document")
    selected = _mapping(_select(document, selector), f"{label}.selected")
    return ref, selector, selected, document


def _normalized_ref(ref: Mapping[str, str], selector: str) -> dict[str, Any]:
    return {
        "authority_ref": dict(ref),
        "authority_selector": selector,
    }


def _adapter_registry(value: object) -> dict[str, str]:
    _, _, registry, _ = _selected(value, "adapter_registry")
    _require(
        registry.get("schema") == ADAPTER_REGISTRY_SCHEMA,
        "adapter registry schema drifted",
    )
    adapters = _sequence(registry.get("adapters"), "adapter_registry.adapters")
    normalized: dict[str, str] = {}
    for index, adapter_value in enumerate(adapters):
        row = _mapping(adapter_value, f"adapter_registry.adapters[{index}]")
        _require(
            set(row) == {"finalization_schema", "source_kind", "planning_schema"},
            "adapter registry entry fields drifted",
        )
        schema = _string(row.get("finalization_schema"), "finalization_schema")
        kind = _string(row.get("source_kind"), "source_kind")
        planning_schema = _string(row.get("planning_schema"), "planning_schema")
        _require(
            schema in ADAPTERS
            and ADAPTERS[schema][0] == kind
            and ADAPTERS[schema][2] == planning_schema,
            "adapter registry declares an unsupported adapter",
        )
        _require(schema not in normalized, "duplicate adapter schema")
        normalized[schema] = kind
    expected = {STATIC_FINALIZATION_SCHEMA: "static"}
    _require(normalized == expected, "registry must bind the static adapter")
    return normalized


def _static_finalization(
    value: Mapping[str, Any], ref: Mapping[str, str]
) -> dict[str, Any]:
    _require(value.get("status") == "pass", "static finalization did not pass")
    _require(
        value.get("full75_canary_pass") is True,
        "static full75 machine gate did not pass",
    )
    _require(
        _integer(value.get("captured_frame_count"), "captured_frame_count") == 75,
        "static finalization is not full75",
    )
    _require(
        value.get("formal_episode_count") == 0
        and value.get("qualification_claim") is False,
        "static finalization must remain formal zero",
    )
    acoustics = _mapping(value.get("acoustics"), "static finalization.acoustics")
    _require(
        acoustics.get("status") == "pass_exact_two_source_rir_target_only_binaural"
        and acoustics.get("rir_job_count") == 2
        and acoustics.get("target_active") is True
        and acoustics.get("distractor_silent") is True
        and acoustics.get("channel_count") == 2
        and acoustics.get("sample_count") == 80_000
        and acoustics.get("sample_rate_hz") == 16_000,
        "static finalization acoustics drifted",
    )
    pixels = _mapping(value.get("pixels"), "static finalization.pixels")
    target_side = _string(pixels.get("target_side"), "static pixels target_side")
    _require(
        pixels.get("status") == "pass" and target_side in {"left", "right"},
        "static finalization pixel side drifted",
    )
    artifacts = _mapping(value.get("artifacts"), "static finalization.artifacts")
    artifact_paths = {
        key: _declared_file(artifacts.get(key), f"static {key}")
        for key in (
            "capture_manifest",
            "binaural_video",
            "binaural_wav",
            "pixel_visibility_truth",
            "runtime_readbacks",
        )
    }
    return {
        "source_kind": "static",
        "episode_id": _string(value.get("episode_id"), "finalization.episode_id"),
        "mechanism": "both_static",
        "capture_manifest": artifact_paths["capture_manifest"],
        "finalization_path": ref["path"],
        "target_side": target_side,
    }


ADAPTERS: dict[
    str,
    tuple[
        str,
        Callable[[Mapping[str, Any], Mapping[str, str]], dict[str, Any]],
        str,
    ],
] = {
    STATIC_FINALIZATION_SCHEMA: (
        "static",
        _static_finalization,
        STATIC_CANARY_PLAN_SCHEMA,
    ),
}


def _timeline(value: object, label: str) -> dict[str, int]:
    timeline = _mapping(value, label)
    normalized = {
        "frame_count": _integer(timeline.get("frame_count"), f"{label}.frame_count"),
        "frame_rate_hz": _integer(
            timeline.get("frame_rate_hz"), f"{label}.frame_rate_hz"
        ),
        "duration_seconds": _integer(
            timeline.get("duration_seconds"), f"{label}.duration_seconds"
        ),
    }
    _require(
        normalized == {"frame_count": 75, "frame_rate_hz": 15, "duration_seconds": 5},
        f"{label} is not full75/5s",
    )
    return normalized


def _static_planning(
    value: object, identity_value: object
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    ref, selector, row, document = _selected(value, "planning")
    _require(
        document.get("schema") == STATIC_CANARY_PLAN_SCHEMA,
        "static canary planning schema drifted",
    )
    _require(
        document.get("status") == "ready_pending_gpu1_idle_gate"
        and document.get("full_batch_authorized") is False,
        "static canary plan boundary drifted",
    )
    canaries = _sequence(document.get("canaries"), "planning.canaries")
    _require(
        selector.startswith("/canaries/")
        and selector.count("/") == 2
        and any(candidate == row for candidate in canaries),
        "planning selector is not a canary row",
    )

    identity_ref, identity_selector, identity_row, identity_document = _selected(
        identity_value, "identity_binding"
    )
    _require(
        identity_document.get("schema") == STATIC_IDENTITY_PLAN_SCHEMA,
        "static identity plan schema drifted",
    )
    identity_rows = _sequence(identity_document.get("rows"), "identity plan rows")
    _require(
        identity_selector.startswith("/rows/")
        and identity_selector.count("/") == 2
        and any(candidate == identity_row for candidate in identity_rows),
        "identity selector is not an expansion row",
    )
    _require(
        identity_row.get("episode_id") == row.get("episode_id")
        and identity_row.get("row_id") == row.get("row_id")
        and identity_row.get("target_expected_screen_side") == row.get("target_side"),
        "static identity row drifted from canary row",
    )
    actors = _sequence(identity_row.get("actors"), "static identity actors")
    _require(len(actors) == 2, "static identity row must contain two actors")
    by_role = {
        _string(actor.get("role"), "static identity actor role"): actor
        for actor in actors
        if isinstance(actor, Mapping)
    }
    _require(set(by_role) == {"target", "distractor"}, "static identity roles drifted")
    _require(
        by_role["target"].get("identity_key") == row.get("target_identity_key")
        and by_role["distractor"].get("identity_key")
        == row.get("distractor_identity_key")
        and by_role["target"].get("expected_screen_side") == row.get("target_side")
        and by_role["distractor"].get("voice_policy") == "silent",
        "static identity keys, side, or roles drifted",
    )
    catalog = _mapping(
        identity_document.get("approved_identity_catalog"),
        "approved identity catalog",
    )
    identity_timeline = _mapping(identity_document.get("timeline"), "identity timeline")
    _require(
        identity_timeline.get("frame_count") == 75
        and identity_timeline.get("frame_rate_hz") == 15,
        "static identity timeline is not full75/5s",
    )
    normalized: dict[str, Any] = {
        "episode_id": _string(row.get("episode_id"), "canary episode_id"),
        "mechanism": "both_static",
        "timeline": {
            "frame_count": 75,
            "frame_rate_hz": 15,
            "duration_seconds": 5,
        },
        "expected_suite_path": _declared_file(
            row.get("suite_plan"), "canary suite_plan"
        ),
        "expected_audio_wav": _declared_file(row.get("audio_wav"), "canary audio_wav"),
    }
    camera = _mapping(identity_row.get("camera_pose"), "static identity camera_pose")
    camera_translation = _sequence(
        camera.get("translation_m"), "static identity camera translation"
    )
    _require(
        len(camera_translation) == 3
        and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in camera_translation
        ),
        "static identity camera translation is invalid",
    )
    normalized["camera_position_m"] = [float(value) for value in camera_translation]
    camera_yaw = camera.get("habitat_yaw_deg")
    _require(
        isinstance(camera_yaw, (int, float)) and not isinstance(camera_yaw, bool),
        "static identity camera yaw is invalid",
    )
    normalized["camera_yaw_deg"] = float(camera_yaw)
    output_root = Path(_string(row.get("output_root"), "canary output_root"))
    _require(output_root.is_absolute(), "canary output_root is not absolute")
    try:
        output_root = output_root.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise FullEpisodeSemanticAuthorityError(
            "canary output_root does not exist"
        ) from error
    _require(output_root.is_dir(), "canary output_root is not a directory")
    normalized["output_root"] = str(output_root)
    acoustic_evidence = _mapping(
        row.get("acoustic_evidence"), "canary acoustic_evidence"
    )
    normalized["expected_delivery_path"] = _declared_file(
        acoustic_evidence.get("binaural_delivery"), "canary binaural_delivery"
    )
    normalized["expected_rir_plan_path"] = _declared_file(
        acoustic_evidence.get("exact_rir_plan"), "canary exact_rir_plan"
    )
    for role in ("target", "distractor"):
        actor = by_role[role]
        identity_key = _string(actor.get("identity_key"), f"{role} identity key")
        catalog_entry = _mapping(catalog.get(identity_key), f"catalog[{identity_key}]")
        normalized[role] = {
            "source_slot_id": _string(actor.get("source_slot_id"), f"{role} slot"),
            "role": role,
            "side": _string(actor.get("expected_screen_side"), f"{role} side"),
            "identity_id": _string(
                catalog_entry.get("original_identity_id"), f"{role} identity"
            ),
            "runtime_asset_id": _string(
                catalog_entry.get("runtime_asset_id"), f"{role} runtime asset"
            ),
            "runtime_revision": _string(
                catalog_entry.get("runtime_revision"), f"{role} runtime revision"
            ),
            "sound_asset_id": catalog_entry.get("sound_asset_id"),
            "voice_policy": actor.get("voice_policy"),
        }
    return (
        normalized,
        _normalized_ref(ref, selector),
        _normalized_ref(identity_ref, identity_selector),
    )


def _suite(value: object) -> tuple[Mapping[str, Any], dict[str, Any]]:
    ref, selector, scenario, document = _selected(value, "suite")
    _require(document.get("schema") == SUITE_SCHEMA, "suite schema drifted")
    _require(scenario.get("schema") == SCENARIO_SCHEMA, "scenario schema drifted")
    scenarios = _sequence(document.get("scenarios"), "suite.scenarios")
    _require(
        selector.startswith("/scenarios/")
        and selector.count("/") == 2
        and any(candidate == scenario for candidate in scenarios),
        "suite selector is not a scenario",
    )
    return scenario, _normalized_ref(ref, selector)


def _member_selection(
    value: object,
    *,
    label: str,
    document_schema: str,
    collection_field: str,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    ref, selector, selected, document = _selected(value, label)
    _require(document.get("schema") == document_schema, f"{label} schema drifted")
    rows = _sequence(document.get(collection_field), f"{label}.{collection_field}")
    _require(
        selector.startswith(f"/{collection_field}/")
        and selector.count("/") == 2
        and any(candidate == selected for candidate in rows),
        f"{label} selector is not a {collection_field} member",
    )
    return selected, _normalized_ref(ref, selector)


def _audio_evidence(
    entry: Mapping[str, Any],
    *,
    episode_id: str,
    actors_by_slot: Mapping[str, Mapping[str, Any]],
    expected_audio_wav: str | None,
    expected_delivery_path: str,
) -> tuple[dict[str, str], list[str], dict[str, Any]]:
    sample, sample_ref = _member_selection(
        entry.get("audio_sample"),
        label="audio_sample",
        document_schema=AUDIO_SAMPLES_SCHEMA,
        collection_field="samples",
    )
    _require(sample.get("episode_id") == episode_id, "audio sample Episode drifted")
    _require(
        sample.get("both_sources_active") is False, "audio sample is not target-only"
    )
    assets = _mapping(sample.get("asset_ids_by_source_slot"), "audio sample assets")
    expected_assets = {
        slot: actor["asset_id"] for slot, actor in actors_by_slot.items()
    }
    _require(dict(assets) == expected_assets, "audio sample actor assets drifted")
    sample_audio = _mapping(sample.get("audio"), "audio sample audio")
    mixture = _mapping(sample_audio.get("mixture"), "audio sample mixture")
    mixture_name = _string(mixture.get("path"), "audio sample mixture.path")
    _require(
        Path(mixture_name).name == mixture_name,
        "audio sample mixture path must be a basename",
    )
    sample_document_path = Path(sample_ref["authority_ref"]["path"])
    selected_audio_wav = (
        sample_document_path.parent / "audio" / "binaural" / mixture_name
    ).resolve()
    _require(
        selected_audio_wav.is_file(),
        "selected audio sample WAV does not exist",
    )
    if expected_audio_wav is not None:
        _require(
            selected_audio_wav == Path(expected_audio_wav).resolve(),
            "audio sample WAV drifted from planning row",
        )

    delivery_ref, delivery_selector, delivery, _ = _selected(
        entry.get("audio_delivery"), "audio_delivery"
    )
    _require(delivery_selector == "", "audio delivery must select its document root")
    _require(
        delivery_ref["path"] == expected_delivery_path,
        "audio delivery path drifted from source authority",
    )
    _require(
        delivery.get("schema") == AUDIO_DELIVERY_SCHEMA
        and delivery.get("status") == "pass"
        and delivery.get("qualification_claim") is False
        and isinstance(delivery.get("episode_count"), int)
        and delivery.get("episode_count") >= 1,
        "audio delivery boundary drifted",
    )
    delivery_outputs = _mapping(delivery.get("outputs"), "audio delivery outputs")
    samples_relative = Path(
        _string(delivery_outputs.get("samples"), "audio delivery samples output")
    )
    _require(
        not samples_relative.is_absolute(),
        "audio delivery samples output must be relative",
    )
    delivery_root = Path(delivery_ref["path"]).parent
    selected_samples_path = (delivery_root / samples_relative).resolve()
    try:
        selected_samples_path.relative_to(delivery_root)
    except ValueError as error:
        raise FullEpisodeSemanticAuthorityError(
            "audio delivery samples output escapes its delivery root"
        ) from error
    _require(
        selected_samples_path == Path(sample_ref["authority_ref"]["path"]),
        "audio delivery samples output drifted from selected samples authority",
    )

    binding = _mapping(sample.get("audio_program_binding"), "audio program binding")
    endpoint_slots = _mapping(
        binding.get("source_endpoint_to_source_slot"), "endpoint-to-slot binding"
    )
    _require(
        set(endpoint_slots.values()) == set(actors_by_slot)
        and len(endpoint_slots) == 2,
        "endpoint-to-slot binding does not cover both actors",
    )
    program_ref = _mapping(binding.get("audio_program_ref"), "audio program ref")
    program_selected_ref, program_selector, program, _ = _selected(
        entry.get("audio_program"), "audio_program"
    )
    _require(
        program.get("schema") == AUDIO_PROGRAM_SCHEMA, "audio program schema drifted"
    )
    _require(program_selector == "", "audio program must select its document root")
    _require(
        program.get("program_id") == program_ref.get("program_id")
        and program.get("revision") == program_ref.get("revision"),
        "audio program binding drifted",
    )
    program_timeline = _mapping(program.get("timeline"), "audio program timeline")
    _require(
        program_timeline.get("frame_count") == 75
        and program_timeline.get("video_fps") == 15
        and program_timeline.get("sample_count") == 80_000
        and program_timeline.get("sample_rate_hz") == 16_000,
        "audio program timeline drifted",
    )
    events = _sequence(program.get("events"), "audio program.events")
    _require(
        program.get("mode") == "one_active_of_n" and len(events) == 1,
        "audio program is not exactly one active event",
    )
    event_ref, event_selector, event, event_document = _selected(
        entry.get("audio_event"), "audio_event"
    )
    _require(
        event_document == program, "audio event is not selected from the bound program"
    )
    _require(
        event_selector.startswith("/events/")
        and event_selector.count("/") == 2
        and any(candidate == event for candidate in events),
        "audio event selector is not an event",
    )
    target_endpoint = _string(event.get("source_endpoint_id"), "audio event endpoint")
    target_slot = _string(endpoint_slots.get(target_endpoint), "target source slot")
    distractor_slots = set(actors_by_slot) - {target_slot}
    _require(len(distractor_slots) == 1, "audio target slot is ambiguous")
    distractor_slot = next(iter(distractor_slots))
    _require(
        actors_by_slot[target_slot]["role"] == "target"
        and actors_by_slot[distractor_slot]["role"] == "distractor",
        "audio active and silent slots do not bind target and distractor roles",
    )

    sound, sound_ref = _member_selection(
        entry.get("sound_asset"),
        label="sound_asset",
        document_schema=SOUND_REGISTRY_SCHEMA,
        collection_field="sound_assets",
    )
    sound_asset_id = _string(sound.get("sound_asset_id"), "sound asset ID")
    _require(
        event.get("sound_asset_id") == sound_asset_id, "audio event sound asset drifted"
    )
    _require(
        sound.get("semantic_sound_class") == "human_speech",
        "target sound is not human speech",
    )

    endpoint_bindings = _sequence(entry.get("source_endpoints"), "source_endpoints")
    _require(len(endpoint_bindings) == 2, "exactly two source endpoints are required")
    endpoints: dict[str, Mapping[str, Any]] = {}
    endpoint_refs: list[dict[str, Any]] = []
    for index, endpoint_binding in enumerate(endpoint_bindings):
        endpoint, endpoint_ref = _member_selection(
            endpoint_binding,
            label=f"source_endpoints[{index}]",
            document_schema=ENDPOINT_REGISTRY_SCHEMA,
            collection_field="source_endpoints",
        )
        endpoint_id = _string(endpoint.get("source_endpoint_id"), "source endpoint ID")
        _require(endpoint_id not in endpoints, "duplicate source endpoint")
        endpoints[endpoint_id] = endpoint
        endpoint_refs.append(endpoint_ref)
    _require(
        set(endpoints) == set(endpoint_slots),
        "selected endpoints drift from audio sample",
    )
    for endpoint_id, slot_value in endpoint_slots.items():
        slot = _string(slot_value, "source endpoint slot")
        endpoint_binding = _mapping(
            endpoints[endpoint_id].get("binding"), "endpoint binding"
        )
        actor = actors_by_slot[slot]
        _require(
            endpoint_binding.get("entity_instance_id") == slot
            and endpoint_binding.get("entity_asset_id") == actor["asset_id"]
            and endpoint_binding.get("entity_asset_revision") == actor["asset_revision"]
            and endpoint_binding.get("emitter_anchor_id") == "mouth",
            "source endpoint actor binding drifted",
        )

    activity = _mapping(
        sample.get("source_activity_summary"), "source activity summary"
    )
    _require(
        activity.get("active_source_slots") == [target_slot]
        and activity.get("silent_source_slots") == [distractor_slot],
        "audio sample activity does not bind target and silent distractor",
    )
    target_audio = {
        "voice_id": _string(sound.get("instance_lineage_id"), "target voice ID"),
        "sound_asset_id": sound_asset_id,
    }
    provenance = {
        "audio_sample": sample_ref,
        "audio_delivery": _normalized_ref(delivery_ref, delivery_selector),
        "audio_program": _normalized_ref(program_selected_ref, program_selector),
        "audio_event": _normalized_ref(event_ref, event_selector),
        "sound_asset": sound_ref,
        "source_endpoints": endpoint_refs,
    }
    return target_audio, [target_slot, distractor_slot], provenance


def _rir_evidence(
    value: object,
    *,
    episode_id: str,
    source_slots: Sequence[str],
    expected_plan_path: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    bindings = _sequence(value, "rir_jobs")
    _require(len(bindings) == 2, "exactly two RIR job selections are required")
    job_ids: list[str] = []
    refs: list[dict[str, Any]] = []
    observed_slots: list[str] = []
    for index, binding in enumerate(bindings):
        job, ref = _member_selection(
            binding,
            label=f"rir_jobs[{index}]",
            document_schema=RIR_PLAN_SCHEMA,
            collection_field="jobs",
        )
        job_id = _string(job.get("job_id"), "RIR job ID")
        _require(
            ref["authority_ref"]["path"] == expected_plan_path,
            "RIR plan path drifted from source authority",
        )
        uses = _sequence(job.get("uses"), "RIR job uses")
        matching_slots = {
            use.get("source_slot_id")
            for use in uses
            if isinstance(use, Mapping) and use.get("episode_id") == episode_id
        }
        _require(
            len(matching_slots) == 1, "RIR job does not bind one Episode source slot"
        )
        slot = next(iter(matching_slots))
        _require(isinstance(slot, str), "RIR source slot is invalid")
        job_ids.append(job_id)
        observed_slots.append(slot)
        refs.append(ref)
    _require(len(set(job_ids)) == 2, "RIR job IDs must be unique")
    _require(
        set(observed_slots) == set(source_slots),
        "RIR jobs do not cover both source slots",
    )
    return job_ids, refs


def _question(value: object) -> dict[str, Any]:
    question = _mapping(value, "semantic_label.question")
    _require(
        set(question) == {"prompt", "options", "correct_index", "option_order_id"},
        "semantic label question fields drifted",
    )
    options = _sequence(question.get("options"), "semantic label question.options")
    _require(len(options) >= 2, "semantic label question needs at least two options")
    semantics: list[dict[str, str]] = []
    for index, option_value in enumerate(options):
        option = _mapping(option_value, f"question.options[{index}]")
        _require(
            set(option) == {"semantic_id", "text"}, "question option fields drifted"
        )
        semantics.append(
            {
                "semantic_id": _string(option.get("semantic_id"), "option semantic_id"),
                "text": _string(option.get("text"), "option text"),
            }
        )
    _require(
        len({option["semantic_id"] for option in semantics}) == len(semantics)
        and len({option["text"] for option in semantics}) == len(semantics),
        "question options are not unique",
    )
    correct_index = _integer(question.get("correct_index"), "question.correct_index")
    _require(
        0 <= correct_index < len(semantics), "question.correct_index is out of range"
    )
    return {
        "prompt": _string(question.get("prompt"), "question.prompt"),
        "options": [option["text"] for option in semantics],
        "option_semantics": semantics,
        "correct_index": correct_index,
        "option_order_id": _string(question.get("option_order_id"), "option_order_id"),
    }


def _approved_label(value: object) -> tuple[dict[str, Any], dict[str, Any]]:
    ref, selector, label, _ = _selected(value, "semantic_label")
    expected_fields = {
        "schema",
        "approval",
        "episode_id",
        "mechanism",
        "scene",
        "camera",
        "actors",
        "timeline",
        "rir_job_ids",
        "question",
        "formal_episode_count",
        "qualification_claim",
        "independence_claim",
    }
    _require(set(label) == expected_fields, "semantic label fields drifted")
    _require(label.get("schema") == LABEL_SCHEMA, "semantic label schema drifted")
    approval = _mapping(label.get("approval"), "semantic label approval")
    _require(
        set(approval) == {"status", "approved_by", "approved_at"}
        and approval.get("status") == "approved",
        "semantic label is not explicitly approved",
    )
    _string(approval.get("approved_by"), "semantic label approved_by")
    _string(approval.get("approved_at"), "semantic label approved_at")
    _require(
        label.get("formal_episode_count") == 0
        and label.get("qualification_claim") is False
        and label.get("independence_claim") is False,
        "semantic label must remain formal zero and non-independent",
    )
    scene = _mapping(label.get("scene"), "semantic label scene")
    actors = _sequence(label.get("actors"), "semantic label actors")
    _require(len(actors) == 2, "semantic label must contain two actors")
    normalized_actors: list[dict[str, Any]] = []
    for index, actor_value in enumerate(actors):
        actor = _mapping(actor_value, f"semantic label actors[{index}]")
        common = {
            field: _string(actor.get(field), f"semantic label actor.{field}")
            for field in (
                "source_slot_id",
                "role",
                "side",
                "identity_id",
                "asset_id",
                "asset_revision",
            )
        }
        _require(
            common["role"] in {"target", "distractor"}, "semantic actor role is invalid"
        )
        _require(common["side"] in {"left", "right"}, "semantic actor side is invalid")
        if common["role"] == "target":
            common.update(
                {
                    key: _string(actor.get(key), f"semantic label actor.{key}")
                    for key in ("voice_id", "content_id", "sound_asset_id")
                }
            )
        else:
            _require(
                actor.get("voice_policy") == "silent",
                "semantic distractor must be silent",
            )
            common["voice_policy"] = "silent"
        normalized_actors.append(common)
    by_role = {actor["role"]: actor for actor in normalized_actors}
    _require(set(by_role) == {"target", "distractor"}, "semantic actor roles drifted")
    question = _question(label.get("question"))
    _require(
        question["option_semantics"][question["correct_index"]]["semantic_id"]
        == by_role["target"]["side"],
        "approved answer conflicts with approved target side",
    )
    normalized = {
        "episode_id": _string(label.get("episode_id"), "semantic label episode_id"),
        "mechanism": _string(label.get("mechanism"), "semantic label mechanism"),
        "scene": {
            field: _string(scene.get(field), f"semantic label scene.{field}")
            for field in _SCENE_ID_FIELDS
        },
        "camera": {
            "camera_cluster_id": _string(
                _mapping(label.get("camera"), "semantic label camera").get(
                    "camera_cluster_id"
                ),
                "semantic label camera_cluster_id",
            )
        },
        "actors": [by_role["target"], by_role["distractor"]],
        "timeline": _timeline(label.get("timeline"), "semantic label timeline"),
        "rir_job_ids": [
            _string(value, "semantic label RIR job ID")
            for value in _sequence(label.get("rir_job_ids"), "semantic label RIR jobs")
        ],
        "question": question,
    }
    _require(
        len(normalized["rir_job_ids"]) == 2, "semantic label must bind two RIR jobs"
    )
    return normalized, _normalized_ref(ref, selector)


def _build_episode(
    value: object,
    *,
    index: int,
    registered: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    entry = _mapping(value, f"episodes[{index}]")
    final_ref, final_selector, finalization, _ = _selected(
        entry.get("finalization"), f"episodes[{index}].finalization"
    )
    _require(
        final_selector == "",
        f"episodes[{index}].finalization must select its document root",
    )
    final_schema = _string(finalization.get("schema"), "finalization.schema")
    _require(final_schema in registered, "finalization schema has no selected adapter")
    source = ADAPTERS[final_schema][1](finalization, final_ref)

    _require(
        source["source_kind"] == "static",
        "dynamic semantic production is not supported",
    )
    _require(
        set(entry) == _STATIC_EPISODE_FIELDS,
        f"episodes[{index}] static fields drifted",
    )
    plan, plan_ref, identity_ref = _static_planning(
        entry.get("planning"), entry.get("identity_binding")
    )
    _require(
        Path(source["capture_manifest"]).parent == Path(plan["output_root"]).resolve(),
        "static capture manifest is outside canary output_root",
    )
    static_artifacts = _mapping(
        finalization.get("artifacts"), "static finalization artifacts"
    )
    _require(
        Path(_string(static_artifacts.get("binaural_wav"), "static WAV")).resolve()
        == Path(plan["expected_audio_wav"]).resolve(),
        "static finalization WAV drifted from canary row",
    )
    _require(plan.get("episode_id") == source["episode_id"], "planning Episode drifted")
    _require(plan.get("mechanism") == source["mechanism"], "planning mechanism drifted")
    _require(
        plan["target"]["side"] == source["target_side"],
        "static pixel target side drifted from planning authority",
    )
    plan_timeline = _timeline(plan.get("timeline"), "planning timeline")
    scenario, suite_ref = _suite(entry.get("suite"))
    expected_suite_path = plan.get(
        "expected_suite_path", source.get("expected_suite_path")
    )
    _require(
        isinstance(expected_suite_path, str)
        and suite_ref["authority_ref"]["path"] == expected_suite_path,
        "selected suite path drifted from planning authority",
    )
    scenario_plan = _mapping(scenario.get("plan"), "suite scenario plan")
    _require(
        scenario.get("scenario_id") == source["episode_id"],
        "suite scenario_id does not match Episode",
    )
    suite_camera = _mapping(scenario_plan.get("camera"), "suite camera")
    suite_camera_position = _sequence(
        suite_camera.get("habitat_position_m"), "suite camera position"
    )
    _require(
        suite_camera.get("dynamic") is False
        and list(suite_camera_position) == plan["camera_position_m"]
        and suite_camera.get("habitat_yaw_deg") == plan["camera_yaw_deg"],
        "suite camera pose drifted from identity authority",
    )
    camera_cluster_id = _string(
        suite_camera.get("sensor_rig_trajectory_id"),
        "suite camera sensor_rig_trajectory_id",
    )
    render = _mapping(scenario_plan.get("render"), "suite render")
    frames = _sequence(scenario_plan.get("frames"), "suite frames")
    _require(
        render.get("frame_count") == 75
        and render.get("fps_num") == 15
        and render.get("fps_den") == 1
        and len(frames) == 75
        and [frame.get("frame_index") for frame in frames if isinstance(frame, Mapping)]
        == list(range(75)),
        "suite is not a contiguous full75 timeline",
    )
    room = _mapping(scenario_plan.get("room"), "suite room")
    scene_provenance = _mapping(
        room.get("source_scene_provenance"), "suite source scene provenance"
    )
    native_scene = _mapping(scenario.get("native_scene"), "suite native scene")
    scene = {
        "scene_id": _string(scene_provenance.get("scene_id"), "suite scene_id"),
        "room_id": _string(room.get("room_id"), "suite room_id"),
        "room_variant_id": (
            f"{_string(room.get('room_capsule_id'), 'suite room capsule')}@"
            f"{_string(room.get('room_capsule_revision'), 'suite room revision')}"
        ),
        "map_id": _string(native_scene.get("map"), "suite map"),
    }
    capture_manifest = _mapping(
        load_json(source["capture_manifest"]), "capture manifest"
    )
    capture_request = _mapping(
        capture_manifest.get("authoritative_capture_request"),
        "capture manifest authoritative_capture_request",
    )
    capture_audio = _mapping(capture_manifest.get("audio"), "capture manifest audio")
    frame_contract = _mapping(
        capture_manifest.get("frame_contract"), "capture manifest frame_contract"
    )
    _require(
        capture_manifest.get("schema") == "avengine_qa_native_spear_pixel_episode_v1"
        and capture_manifest.get("status") == "pass"
        and capture_manifest.get("scenario_id") == source["episode_id"]
        and capture_manifest.get("native_map") == scene["map_id"],
        "capture manifest does not bind the Episode and suite map",
    )
    _require(
        frame_contract.get("frame_count") == 75
        and frame_contract.get("frame_rate_hz") == 15
        and frame_contract.get("captured_frame_indices") == list(range(75)),
        "capture frame contract is not contiguous full75",
    )
    _require(
        capture_request.get("episode_id") == source["episode_id"]
        and capture_request.get("scenario_type")
        in {
            "strict_two_human_static_canary",
            "strict_two_human_expansion_static",
        }
        and capture_request.get("target_source_slot_id")
        == plan["target"]["source_slot_id"],
        "capture request does not bind the planned active target",
    )
    _require(
        _declared_file(
            capture_audio.get("authoritative_wav"),
            "capture manifest authoritative WAV",
        )
        == plan["expected_audio_wav"],
        "capture audio drifted from planning authority",
    )
    suite_actors = {
        _string(actor.get("actor_id"), "suite actor ID"): actor
        for actor in _sequence(scenario_plan.get("actors"), "suite actors")
        if isinstance(actor, Mapping)
    }
    _require(len(suite_actors) == 2, "suite must contain exactly two actors")
    actors: list[dict[str, Any]] = []
    actors_by_slot: dict[str, dict[str, Any]] = {}
    for role in ("target", "distractor"):
        planned = _mapping(plan.get(role), f"planning {role}")
        slot = _string(planned.get("source_slot_id"), f"planning {role} slot")
        matches = [
            (actor_id, actor)
            for actor_id, actor in suite_actors.items()
            if _mapping(
                actor.get("runtime_asset_expectation", {}),
                f"suite {role} runtime expectation",
            ).get("source_slot_id")
            == slot
        ]
        _require(len(matches) == 1, f"suite {role} source slot is ambiguous")
        actor_id = matches[0][0]
        _require(
            actor_id in suite_actors, f"planning {role} actor is absent from suite"
        )
        suite_actor = suite_actors[actor_id]
        actor = {
            "source_slot_id": slot,
            "role": role,
            "side": _string(planned.get("side"), f"{role} side"),
            "identity_id": _string(planned.get("identity_id"), f"{role} identity"),
            "asset_id": _string(suite_actor.get("asset_id"), f"{role} asset"),
            "asset_revision": _string(
                suite_actor.get("asset_revision"), f"{role} asset revision"
            ),
        }
        _require(
            planned.get("runtime_asset_id") == actor["asset_id"]
            and planned.get("runtime_revision") == actor["asset_revision"],
            f"planning {role} asset drifted from suite",
        )
        if role == "distractor":
            _require(
                planned.get("voice_policy") == "silent",
                "planned distractor is not silent",
            )
            actor["voice_policy"] = "silent"
        actors.append(actor)
        actors_by_slot[actor["source_slot_id"]] = actor
    _require(
        len(actors_by_slot) == 2 and actors[0]["side"] != actors[1]["side"],
        "actor slots or sides are not unique",
    )

    target_audio, source_slots, audio_refs = _audio_evidence(
        entry,
        episode_id=source["episode_id"],
        actors_by_slot=actors_by_slot,
        expected_audio_wav=plan.get("expected_audio_wav"),
        expected_delivery_path=plan.get(
            "expected_delivery_path", source.get("expected_delivery_path")
        ),
    )
    target = actors[0]
    planned_target = _mapping(plan.get("target"), "planning target")
    _require(
        planned_target.get("sound_asset_id") == target_audio["sound_asset_id"],
        "planning target sound asset drifted from audio authorities",
    )
    rir_job_ids, rir_refs = _rir_evidence(
        entry.get("rir_jobs"),
        episode_id=source["episode_id"],
        source_slots=source_slots,
        expected_plan_path=plan.get(
            "expected_rir_plan_path", source.get("expected_rir_plan_path")
        ),
    )
    label, label_ref = _approved_label(entry.get("semantic_label"))
    approved_target = label["actors"][0]
    target.update({**target_audio, "content_id": approved_target["content_id"]})
    source_facts = {
        "episode_id": source["episode_id"],
        "mechanism": source["mechanism"],
        "scene": scene,
        "camera": {
            "camera_cluster_id": camera_cluster_id,
        },
        "actors": actors,
        "timeline": plan_timeline,
        "rir_job_ids": rir_job_ids,
    }
    for field in (
        "episode_id",
        "mechanism",
        "scene",
        "camera",
        "actors",
        "timeline",
        "rir_job_ids",
    ):
        _require(
            label[field] == source_facts[field],
            f"approved label {field} drifted from runtime evidence",
        )

    record = {
        "schema": SEMANTIC_SCHEMA,
        "source_kind": source["source_kind"],
        **source_facts,
        "scene": scene,
        "target_audio": {
            **target_audio,
            "content_id": approved_target["content_id"],
        },
        "distractor_audio": {"voice_policy": "silent"},
        "question": label["question"],
        "formal_episode_count": 0,
        "qualification_claim": False,
        "independence_claim": False,
    }
    provenance = {
        "finalization": _normalized_ref(final_ref, final_selector),
        "planning": plan_ref,
        "suite": suite_ref,
        **audio_refs,
        "rir_jobs": rir_refs,
        "semantic_label": label_ref,
    }
    if identity_ref is not None:
        provenance["identity_binding"] = identity_ref
    readiness_row = {
        "source_kind": source["source_kind"],
        "scene_key": tuple(scene[field] for field in _SCENE_ID_FIELDS),
        "episode_id": source["episode_id"],
        "finalization_path": source["finalization_path"],
        "capture_manifest": source["capture_manifest"],
    }
    return record, provenance, readiness_row


def _readiness(
    rows: Sequence[Mapping[str, Any]], *, authority_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[tuple[str, ...], int]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["source_kind"] == "static":
            groups[row["scene_key"]].append(row)
    ready_groups: list[dict[str, Any]] = []
    readiness_records: list[dict[str, Any]] = []
    record_indices: dict[tuple[str, ...], int] = {}
    for scene_key, group in groups.items():
        episode_ids = [row["episode_id"] for row in group]
        finalizations = [row["finalization_path"] for row in group]
        captures = [row["capture_manifest"] for row in group]
        _require(
            len(set(episode_ids)) == len(group),
            "static readiness Episode IDs are not unique",
        )
        _require(
            len(set(finalizations)) == len(group),
            "static readiness finalizations are not unique",
        )
        _require(
            len(set(captures)) == len(group),
            "static readiness capture paths are not unique",
        )
        if len(group) >= 4:
            record_indices[scene_key] = len(readiness_records)
            readiness_records.append(
                {
                    "schema": ROOM_READINESS_SCHEMA,
                    "status": "pass",
                    "full75_capture_ready": True,
                    **dict(zip(_SCENE_ID_FIELDS, scene_key, strict=True)),
                }
            )
            ready_groups.append(
                {
                    "scene": dict(zip(_SCENE_ID_FIELDS, scene_key, strict=True)),
                    "static_machine_full75_evidence_count": len(group),
                    "episode_ids": episode_ids,
                    "evidence": [
                        {
                            "episode_id": row["episode_id"],
                            "finalization_ref": {"path": row["finalization_path"]},
                            "capture_manifest_ref": {"path": row["capture_manifest"]},
                        }
                        for row in group
                    ],
                }
            )
    status = "pass" if ready_groups else "not_ready"
    return (
        {
            "status": status,
            "minimum_unique_static_machine_full75_per_scene": 4,
            "ready_scene_count": len(ready_groups),
            "ready_groups": ready_groups,
        },
        readiness_records,
        record_indices,
    )


def build_full_episode_semantic_authority(
    request: Mapping[str, Any],
    *,
    authority_path: str | Path,
) -> dict[str, Any]:
    """Cross-check selected real authorities and emit only approved semantics."""
    _require(
        set(request) == {"schema", "adapter_registry", "episodes"},
        "request fields drifted",
    )
    _require(request.get("schema") == REQUEST_SCHEMA, "request schema drifted")
    registered = _adapter_registry(request.get("adapter_registry"))
    episodes = _sequence(request.get("episodes"), "episodes")
    _require(bool(episodes), "episodes must not be empty")
    records: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    readiness_rows: list[dict[str, Any]] = []
    for index, value in enumerate(episodes):
        record, source_refs, readiness_row = _build_episode(
            value,
            index=index,
            registered=registered,
        )
        records.append(record)
        provenance.append(source_refs)
        readiness_rows.append(readiness_row)
    episode_ids = [record["episode_id"] for record in records]
    _require(len(set(episode_ids)) == len(records), "duplicate Episode IDs")
    authority_raw = Path(authority_path)
    _require(authority_raw.is_absolute(), "authority_path must be absolute")
    authority = authority_raw.resolve()
    _require(
        authority.parent.exists() and authority.parent.is_dir(),
        "authority_path parent must be an existing directory",
    )
    _require(not authority.exists(), "authority_path must not already exist")
    readiness, readiness_records, readiness_indices = _readiness(
        readiness_rows, authority_path=authority
    )
    for record in records:
        scene_key = tuple(record["scene"][field] for field in _SCENE_ID_FIELDS)
        readiness_index = readiness_indices.get(scene_key)
        if readiness_index is None:
            record["scene"]["room_readiness"] = {
                "schema": ROOM_READINESS_SCHEMA,
                "status": "not_ready",
                "full75_capture_ready": False,
                **{field: record["scene"][field] for field in _SCENE_ID_FIELDS},
            }
        else:
            record["scene"]["room_readiness"] = {
                **readiness_records[readiness_index],
                "authority_ref": {"path": str(authority)},
                "authority_selector": f"/room_readiness_records/{readiness_index}",
            }
    return {
        "schema": AUTHORITY_SCHEMA,
        "status": readiness["status"],
        "episode_count": len(records),
        "formal_episode_count": 0,
        "qualification_claim": False,
        "independence_claim": False,
        "readiness": readiness,
        "room_readiness_records": readiness_records,
        "ready_record_selectors": [
            {
                "episode_id": record["episode_id"],
                "authority_ref": {"path": str(authority)},
                "authority_selector": f"/records/{index}",
            }
            for index, record in enumerate(records)
            if record["scene"]["room_readiness"]["status"] == "pass"
        ],
        "records": records,
        "provenance": provenance,
    }
