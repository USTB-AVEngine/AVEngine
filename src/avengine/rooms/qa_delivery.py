"""Finish one captured research Episode without duplicating its media per QA."""
from __future__ import annotations

import json
from copy import deepcopy
import os
from pathlib import Path
import time
import wave
import subprocess
import sys
from typing import Any, Mapping, Sequence

from avengine.capture.neutral_readback import validate_clock, validate_neutral_readback
from avengine.qa.unified_catalog import (
    generate_unified_questions,
    normalize_episode_bundle,
    with_derived_sound_class_answer_domain,
)
from avengine.rooms.evidence_contract import validate_evidence_contract
from avengine.rooms.qa_episode import read_json, write_json
from avengine.rooms.qa_evidence import (
    acquire_shared_visual_evidence,
    annotate_pixel_visibility_semantics,
    audit_native_structural_clearance, build_pixel_appearance_review, derive_actor_occluders,
    nonhuman_appearance_placeholder_thresholds,
    review_imported_pose_clearance,
    shared_visual_pack_root,
)



def _declared_audio_gain(request: Mapping[str, Any], plan: Mapping[str, Any]) -> float | None:
    from avengine.timeline.current_mp3d_dynamic_audio import validate_post_assembly_convolution_gain

    runtime = request.get("runtime")
    for source in (request, runtime, plan):
        if isinstance(source, Mapping) and source.get("post_assembly_convolution_gain") is not None:
            return validate_post_assembly_convolution_gain(source["post_assembly_convolution_gain"])
    return None


def _declared_source_context_policy(request: Mapping[str, Any], plan: Mapping[str, Any]) -> str:
    value = request.get("source_context_policy", plan.get("request", {}).get("source_context_policy", "joint"))
    if value not in {"joint", "independent_states"}:
        raise ValueError("source_context_policy must be joint or independent_states")
    return value


def _declared_audio_render_options(
    request: Mapping[str, Any], plan: Mapping[str, Any],
) -> tuple[str, str]:
    """Resolve the existing audio delivery declaration for both renderers."""
    sources: list[Mapping[str, Any]] = [request]
    request_runtime = request.get("runtime")
    if isinstance(request_runtime, Mapping):
        sources.append(request_runtime)
    plan_request = plan.get("request")
    if isinstance(plan_request, Mapping):
        sources.append(plan_request)
    sources.append(plan)
    plan_runtime = plan.get("runtime")
    if isinstance(plan_runtime, Mapping):
        sources.append(plan_runtime)

    layouts_value: Any = None
    for source in sources:
        for key in ("audio_layouts", "layouts"):
            if key in source and source[key] is not None:
                layouts_value = source[key]
                break
        if layouts_value is not None:
            break

    if layouts_value is None:
        layout_names = ("binaural",)
    elif isinstance(layouts_value, str):
        layout_names = tuple(item.strip() for item in layouts_value.split(",") if item.strip())
    elif isinstance(layouts_value, Mapping):
        layout_names = tuple(str(item).strip() for item in layouts_value if str(item).strip())
    elif isinstance(layouts_value, Sequence) and not isinstance(layouts_value, (str, bytes)):
        names: list[str] = []
        for index, item in enumerate(layouts_value):
            if isinstance(item, Mapping):
                value = item.get("type", item.get("layout_type"))
            else:
                value = item
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"audio_layouts[{index}] must declare a layout type")
            names.append(value.strip())
        layout_names = tuple(names)
    else:
        raise ValueError("audio_layouts must be a sequence, mapping or comma-separated string")
    if not layout_names:
        raise ValueError("audio_layouts must declare at least one layout")
    unsupported = [item for item in layout_names if item not in {"binaural", "ambisonics"}]
    if unsupported:
        raise ValueError(
            "audio render does not support declared layouts "
            f"{unsupported}; choose binaural or ambisonics"
        )
    if len(set(layout_names)) != len(layout_names):
        raise ValueError("audio_layouts must not repeat a layout")

    normalization: Any = None
    for source in sources:
        if "foa_normalization" in source and source["foa_normalization"] is not None:
            normalization = source["foa_normalization"]
            break
    if normalization is None:
        normalization = "native_n3d"
    if normalization not in {"native_n3d", "sn3d"}:
        raise ValueError("foa_normalization must be native_n3d or sn3d")
    return ",".join(layout_names), str(normalization)


def build_audio_command(
    request: Mapping[str, Any], plan: Mapping[str, Any], episode_root: Path,
    audio_root: Path, *, repository: Path,
    capture_root: Path | None = None, plan_root: Path | None = None,
) -> list[str]:
    runtime = request["runtime"]
    source_policy = _declared_source_context_policy(request, plan)
    layouts, foa_normalization = _declared_audio_render_options(request, plan)
    selected_capture = Path(capture_root) if capture_root is not None else episode_root / "capture"
    selected_plan = Path(plan_root) if plan_root is not None else episode_root / "plan"
    cache_value = request.get("rir_cache") or runtime.get("rir_cache") or plan.get("rir_cache")
    cache_path = Path(cache_value).expanduser().resolve() if cache_value else audio_root.parent / (audio_root.name + "_rir_cache")
    if source_policy != "joint" and cache_value:
        raise ValueError("independent source contexts cannot reuse a declared joint RIR cache")
    if cache_value and not cache_path.is_dir():
        raise FileNotFoundError(f"declared existing RIR cache is unavailable: {cache_path}")
    command = [
        sys.executable, str(repository / "tools/acoustics/render_frame_readback_sequential_speech.py"),
        "--frame-readbacks", str(selected_capture / "frame_readbacks.json"),
        "--package-manifest", str(plan["resources"]["acoustic_package"]),
        "--voice-binding", str(selected_plan / "voice_bindings.json"),
        "--audio-plan", str(selected_plan / "episode_plan.json"),
        "--output", str(audio_root),
        "--runtime-prefix", str(runtime["runtime_prefix"]),
        "--rlr-sdk-root", str(runtime["rlr_sdk_root"]),
        "--magnum-python-site", str(runtime["magnum_python_site"]),
        "--rir-stride", str(request.get("rir_stride", 3)),
        "--layouts", layouts,
        "--foa-normalization", foa_normalization,
    ]
    if source_policy == "joint":
        command += ["--rir-cache", str(cache_path)]
    else:
        command += ["--source-context-policy", source_policy]
    gain = _declared_audio_gain(request, plan)
    if gain is not None:
        command += ["--post-assembly-convolution-gain", str(gain)]
    if runtime.get("hrtf"):
        command += ["--hrtf", str(runtime["hrtf"])]
    neutral = (
        request.get("neutral_readback")
        or runtime.get("neutral_readback")
        or plan.get("neutral_readback")
        or (selected_capture / "neutral_readback.json" if (selected_capture / "neutral_readback.json").is_file() else None)
    )
    if neutral:
        command += ["--neutral-readback", str(neutral)]
    prepared_manifest = (
        request.get("prepared_manifest")
        or runtime.get("prepared_manifest")
        or plan.get("prepared_manifest")
    )
    if prepared_manifest:
        command += ["--prepared-manifest", str(prepared_manifest)]
    if "diffraction" in request:
        command += [("--diffraction" if request["diffraction"] else "--no-diffraction")]
    elif "diffraction" in runtime:
        command += [("--diffraction" if runtime["diffraction"] else "--no-diffraction")]
    elif "diffraction" in plan:
        command += [("--diffraction" if plan["diffraction"] else "--no-diffraction")]
    max_order = request.get(
        "max_diffraction_order",
        runtime.get("max_diffraction_order", plan.get("max_diffraction_order")),
    )
    if max_order is not None:
        command += ["--max-diffraction-order", str(max_order)]
    simulation_request = (
        request.get("simulation_request")
        or runtime.get("simulation_request")
        or plan.get("simulation_request")
    )
    if simulation_request:
        command += ["--simulation-request", str(simulation_request)]
    simulation: dict[str, Any] = {}
    for block in (plan.get("simulation"), runtime.get("simulation"), request.get("simulation")):
        if isinstance(block, Mapping):
            for key in (
                "direct_sh_order",
                "indirect_sh_order",
                "indirect_ray_depth",
                "max_ir_seconds",
            ):
                if key in block and block[key] is not None:
                    simulation[key] = block[key]
    if "direct_sh_order" in simulation:
        command += ["--direct-sh-order", str(simulation["direct_sh_order"])]
    if "indirect_sh_order" in simulation:
        command += ["--indirect-sh-order", str(simulation["indirect_sh_order"])]
    if "indirect_ray_depth" in simulation:
        command += ["--indirect-depth", str(simulation["indirect_ray_depth"])]
    if "max_ir_seconds" in simulation:
        command += ["--max-ir-seconds", str(simulation["max_ir_seconds"])]
    return command


def _read_optional(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    value = read_json(path)
    return value if isinstance(value, Mapping) else None


def _capture_root(root: Path) -> Path:
    """Find a contract-bearing capture directory without assuming a renderer."""
    candidates = [root, root / "capture_retry_v2", root / "capture"]
    if root.is_dir():
        candidates.extend(sorted(path for path in root.iterdir() if path.is_dir()))
    scored: list[tuple[tuple[int, int, int, int], Path]] = []
    for candidate in candidates:
        if candidate in {entry[1] for entry in scored}:
            continue
        if not (
            (candidate / "pixel_visibility_truth.json").is_file()
            and (candidate / "native_pixel_masks_depth_authority_v1.npz").is_file()
        ):
            continue
        score = (
            int((candidate / "research_receipt.json").is_file()),
            int((candidate / "neutral_readback.json").is_file()),
            int((candidate / "frame_readbacks.json").is_file() or (candidate / "frame_records.json").is_file()),
            int(candidate.stat().st_mtime_ns),
        )
        scored.append((score, candidate))
    if not scored:
        raise FileNotFoundError(
            f"episode has no EvidenceContract capture directory: {root}"
        )
    return max(scored, key=lambda item: item[0])[1].resolve()


def _find_plan_path(root: Path, capture_root: Path) -> Path | None:
    candidates = [
        root / "plan/episode_plan.json",
        capture_root.parent / "plan/episode_plan.json",
        root.parent / "plan/episode_plan.json",
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return None


def _find_request_path(root: Path, capture_root: Path) -> Path | None:
    candidates = [root / "request.json", capture_root.parent / "request.json", root.parent / "request.json"]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return None


def _is_habitat_plan(plan: Mapping[str, Any]) -> bool:
    resources = plan.get("resources")
    resources = resources if isinstance(resources, Mapping) else {}
    package = resources.get("room_package")
    renderer = (
        (package.get("renderer") if isinstance(package, Mapping) else None)
        or plan.get("renderer_backend")
        or resources.get("backend")
    )
    if renderer == "habitat":
        return True
    if renderer in {"ue_spear", "spear_unreal", "spear_unreal_native"}:
        return False
    if plan.get("plan_coordinates") == "renderer_neutral":
        raise ValueError("neutral plan has no declared room renderer for automatic audio")
    return False  # Retained UE plans predate renderer-neutral room packages.


def _authoritative_endpoint_bindings(
    plan: Mapping[str, Any],
    neutral_readback: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Resolve actor/slot IDs only from explicit endpoint-bearing records."""
    endpoint_by_actor: dict[str, str] = {}
    identities = (
        neutral_readback.get("entity_identities")
        if isinstance(neutral_readback, Mapping)
        else None
    )
    if isinstance(identities, Mapping):
        for slot, identity in identities.items():
            if not isinstance(identity, Mapping):
                continue
            endpoint = identity.get("source_endpoint_id")
            if not isinstance(endpoint, str) or not endpoint:
                continue
            endpoint_by_actor[str(slot)] = endpoint
            actor_id = identity.get("actor_id")
            if isinstance(actor_id, str) and actor_id:
                endpoint_by_actor[actor_id] = endpoint
    materialized_tracks = []
    for key in ("actor_tracks", "materialized_tracks", "tracks"):
        value = plan.get(key)
        if isinstance(value, list):
            materialized_tracks.extend(value)
    for track in materialized_tracks:
        if not isinstance(track, Mapping):
            continue
        actor_id = (
            track.get("actor_id")
            or track.get("source_slot_id")
            or track.get("entity_id")
        )
        endpoint = track.get("source_endpoint_id")
        binding = track.get("emitter_binding")
        if not isinstance(endpoint, str) and isinstance(binding, Mapping):
            endpoint = binding.get("source_endpoint_id")
        if isinstance(actor_id, str) and actor_id and isinstance(endpoint, str) and endpoint:
            endpoint_by_actor.setdefault(actor_id, endpoint)
    return endpoint_by_actor



def select_habitat_audio_program_mode(
    events: Sequence[Mapping[str, Any]],
    candidate_ids: Sequence[str],
) -> str:
    """Select an AudioProgram mode from event overlap, not plan.audio_mode.

    ``plan["audio_mode"]`` is an event_relation (sequential/overlap/repeat).
    Habitat must use the same mode rules as the UE program writer.
    """
    records = [event for event in events if isinstance(event, Mapping)]
    if not records:
        raise ValueError("Habitat audio plan has no audio_events")
    overlaps = any(
        str(left.get("source_endpoint_id")) != str(right.get("source_endpoint_id"))
        and max(int(left["start_sample"]), int(right["start_sample"]))
        < min(int(left["end_sample_exclusive"]), int(right["end_sample_exclusive"]))
        for left_index, left in enumerate(records)
        for right in records[left_index + 1 :]
    )
    active_count = len({str(item["source_endpoint_id"]) for item in records})
    if active_count == 1:
        if len(list(candidate_ids)) >= 2:
            return "one_active_of_n"
        ordered = sorted(
            records,
            key=lambda item: (
                int(item["start_sample"]),
                str(item.get("source_endpoint_id")),
                str(item.get("event_id")),
            ),
        )
        if len(ordered) > 1 and any(
            int(right["start_sample"]) > int(left["end_sample_exclusive"])
            for left, right in zip(ordered, ordered[1:])
        ):
            return "intermittent_events"
        raise ValueError("a one-event plan must declare at least two candidate endpoints")
    return "simultaneous_subset" if overlaps else "sequential_sources"


def _write_habitat_audio_program(
    plan: Mapping[str, Any],
    output_path: Path,
    *,
    neutral_readback: Mapping[str, Any] | None = None,
) -> Path:
    """Materialize common-plan events using authoritative endpoint bindings."""
    from avengine.timeline.audio_program import bind_audio_program_hash, validate_audio_program

    clock = plan.get("clock")
    if not isinstance(clock, Mapping):
        raise ValueError("Habitat audio plan has no clock")
    raw_events = plan.get("audio_events")
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError("Habitat audio plan has no audio_events")
    endpoint_by_actor = _authoritative_endpoint_bindings(plan, neutral_readback)
    actors = _plan_actor_list(plan)
    actor_to_endpoint: dict[str, str] = {}
    endpoints: list[str] = []
    for actor in actors:
        actor_id = actor.get("actor_id")
        binding = actor.get("emitter_binding")
        binding = binding if isinstance(binding, Mapping) else {}
        endpoint = actor.get("source_endpoint_id") or binding.get("source_endpoint_id")
        if not isinstance(endpoint, str) or not endpoint:
            endpoint = endpoint_by_actor.get(str(actor_id)) if actor_id is not None else None
        if not isinstance(endpoint, str) or not endpoint:
            raise ValueError(
                f"Habitat actor {actor_id!r} has no authoritative source endpoint "
                "(plan, materialized track, or neutral entity_identities)"
            )
        if isinstance(actor_id, str) and isinstance(endpoint, str):
            actor_to_endpoint[actor_id] = endpoint
        if isinstance(endpoint, str) and endpoint not in endpoints:
            endpoints.append(endpoint)
    events: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(raw_events):
        if not isinstance(raw, Mapping):
            raise ValueError(f"Habitat audio event {ordinal} must be an object")
        actor_id = raw.get("actor_id")
        endpoint = raw.get("source_endpoint_id") or (actor_to_endpoint.get(str(actor_id)) if actor_id is not None else None)
        if not isinstance(endpoint, str) or not endpoint:
            raise ValueError(
                f"Habitat audio event {ordinal} has no authoritative source endpoint "
                "(event, actor, materialized track, or neutral entity_identities)"
            )
        if endpoint not in endpoints:
            endpoints.append(endpoint)
        start_sample = raw.get("start_sample", raw.get("start_sample_index"))
        end_sample = raw.get("end_sample_exclusive", raw.get("end_sample", raw.get("end_sample_index")))
        if not isinstance(start_sample, (int, float)) or not isinstance(end_sample, (int, float)):
            raise ValueError(f"Habitat audio event {ordinal} has no sample interval")
        start_sample = int(start_sample)
        end_sample = int(end_sample)
        start_tick = raw.get("start_tick", raw.get("start_ticks", start_sample * 3))
        end_tick = raw.get("end_tick_exclusive", raw.get("end_tick", raw.get("end_ticks", end_sample * 3)))
        source_start = int(raw.get("source_start_sample", 0))
        source_end = raw.get("source_end_sample_exclusive", raw.get("source_end_sample"))
        if source_end is None:
            source_end = source_start + int(raw.get("sample_count", end_sample - start_sample))
        event = {
            "event_id": str(raw.get("event_id") or f"event_{ordinal:04d}"),
            "source_endpoint_id": endpoint,
            "sound_asset_id": str(raw.get("sound_asset_id") or raw.get("prepared_audio_id") or ""),
            "start_tick": int(start_tick),
            "end_tick_exclusive": int(end_tick),
            "start_sample": start_sample,
            "end_sample_exclusive": end_sample,
            "source_start_sample": source_start,
            "source_end_sample_exclusive": int(source_end),
            "linear_gain": float(raw.get("linear_gain", 1.0)),
            "fade_samples": int(raw.get("fade_samples", 0)),
            "render_source_stem": True,
            "normalization_policy": raw.get("normalization_policy", "use_sound_asset_policy"),
        }
        if not event["sound_asset_id"]:
            raise ValueError(f"Habitat audio event {event['event_id']} has no sound asset")
        events.append(event)
    candidate_ids = sorted(set(endpoints))
    timeline = {
        "time_base_hz": int(clock["time_base_hz"]),
        "ticks_per_frame": int(clock["ticks_per_frame"]),
        "video_fps": clock["frame_rate_hz"],
        "frame_count": int(clock["frame_count"]),
        "sample_rate_hz": int(clock["sample_rate_hz"]),
        "ticks_per_sample": 3,
        "sample_count": int(clock["sample_count"]),
    }
    program = bind_audio_program_hash({
        "schema": "avengine_m6_audio_program_v1",
        "program_id": str(plan.get("program_id", f"{plan.get('episode_id', 'habitat')}_audio_program")),
        "revision": str(plan.get("revision", "v1")),
        "mode": select_habitat_audio_program_mode(events, candidate_ids),
        "timeline": timeline,
        "candidate_source_endpoint_ids": candidate_ids,
        "events": events,
        "source_specific_stems": True,
        "admission_state": "research",
    })
    errors = validate_audio_program(program)
    if errors:
        raise ValueError("AudioProgram validation failed: " + "; ".join(errors))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, program)
    return output_path.resolve()


def _build_habitat_audio_command(
    request: Mapping[str, Any],
    plan: Mapping[str, Any],
    episode_root: Path,
    capture_root: Path,
    audio_root: Path,
    audio_program_path: Path,
    *,
    repository: Path,
) -> list[str]:
    runtime = request.get("runtime") if isinstance(request.get("runtime"), Mapping) else {}
    layouts, foa_normalization = _declared_audio_render_options(request, plan)
    resources = plan.get("resources") if isinstance(plan.get("resources"), Mapping) else {}
    m1_value = resources.get("m1_request") or resources.get("m1_request_path")
    if not isinstance(m1_value, str):
        candidate = episode_root / "plan/habitat_execution/m1_capture_request.json"
        m1_value = str(candidate)
    package_value = resources.get("acoustic_package")
    if not isinstance(package_value, str):
        package = resources.get("room_package")
        package_value = package.get("acoustic_package") if isinstance(package, Mapping) else None
    if not isinstance(package_value, str):
        raise ValueError("Habitat audio plan lacks acoustic package manifest")
    simulation_value = (
        request.get("simulation_request")
        or runtime.get("simulation_request")
        or plan.get("simulation_request")
        or resources.get("simulation_request")
        or str(repository / "examples/runtime/rir_cache_simulation_request_v2.json")
    )
    bindings = plan.get("voice_bindings")
    bindings = bindings if isinstance(bindings, list) else []
    event_bindings = plan.get("audio_events")
    event_bindings = event_bindings if isinstance(event_bindings, list) else []
    all_bindings = [*bindings, *event_bindings]
    sound_paths: dict[str, str] = {}
    for value in all_bindings:
        if not isinstance(value, Mapping):
            continue
        sound_id = value.get("sound_asset_id") or value.get("prepared_audio_id")
        path = value.get("path") or value.get("prepared") or value.get("audio_path")
        if isinstance(sound_id, str) and sound_id and isinstance(path, str) and path:
            resolved_path = str(Path(path).expanduser().resolve())
            if sound_id in sound_paths and sound_paths[sound_id] != resolved_path:
                raise ValueError(f"conflicting dry PCM paths for sound asset {sound_id!r}")
            sound_paths[sound_id] = resolved_path
    if not sound_paths:
        raise ValueError("Habitat audio plan lacks explicit dry asset bindings")
    hrtf = runtime.get("hrtf") or "/usr/share/libmysofa/MIT_KEMAR_normal_pinna.sofa"
    # Invoke the installed AVEngine CLI directly; ``python -m`` is represented
    # as separate argv entries below so no shell path assumptions are needed.
    command = [
        sys.executable,
        "-m",
        "avengine.cli",
        "m5",
        "render-current-mp3d-dynamic-audio",
        "--rir-stride-frames", str(request.get("rir_stride", 3)),
        "--visual-capture-dir", str(capture_root),
        "--m1-request", str(Path(m1_value).expanduser().resolve()),
        "--simulation-request", str(Path(simulation_value).expanduser().resolve()),
        "--package-manifest", str(Path(package_value).expanduser().resolve()),
        "--audio-program", str(audio_program_path.resolve()),
        "--hrtf", str(hrtf),
        "--runtime-prefix", str(runtime.get("runtime_prefix", "")),
        "--rlr-sdk-root", str(runtime.get("rlr_sdk_root", "")),
        "--output", str(audio_root),
        "--layouts", layouts,
        "--foa-normalization", foa_normalization,
    ]
    source_policy = _declared_source_context_policy(request, plan)
    if source_policy != "joint":
        command += ["--source-context-policy", source_policy]
    gain = _declared_audio_gain(request, plan)
    if gain is not None:
        command += ["--post-assembly-convolution-gain", str(gain)]
    if "diffraction" in request:
        command += ["--diffraction" if request["diffraction"] else "--no-diffraction"]
    elif "diffraction" in runtime:
        command += ["--diffraction" if runtime["diffraction"] else "--no-diffraction"]
    max_order = request.get("max_diffraction_order", runtime.get("max_diffraction_order"))
    if max_order is not None:
        command += ["--max-diffraction-order", str(max_order)]
    if runtime.get("magnum_python_site"):
        command += ["--magnum-python-site", str(runtime["magnum_python_site"])]
    neutral = capture_root / "neutral_readback.json"
    if neutral.is_file():
        command += ["--neutral-readback", str(neutral)]
    prepared = request.get("prepared_manifest") or runtime.get("prepared_manifest") or plan.get("prepared_manifest")
    if isinstance(prepared, str) and prepared:
        command += ["--prepared-manifest", str(Path(prepared).expanduser().resolve())]
    beagle_path = sound_paths.get("dog_beagle_v2_scheduled_dry")
    if isinstance(beagle_path, str) and beagle_path:
        command += ["--beagle-audio", beagle_path]
    for sound_id, path in sorted(sound_paths.items()):
        if sound_id != "dog_beagle_v2_scheduled_dry":
            command += ["--asset-binding", f"{sound_id}={path}"]
    return command


def _find_audio_report(
    root: Path, capture_root: Path, supplied: Path | None,
) -> Path:
    candidates = [
        supplied,
        root / "research_report.json",
        capture_root / "research_report.json",
        root / "capture/research_report.json",
    ]
    for path in candidates:
        if path is not None and path.is_file():
            return path.resolve()
    raise FileNotFoundError("finalization requires an existing P6 research_report.json")


def _asset_registry(repository: Path, registry_path: str | Path | None = None) -> dict[str, Mapping[str, Any]]:
    path = Path(registry_path).expanduser() if registry_path is not None else (
        repository / "examples/runtime/source_asset_runtime_profiles.json"
    )
    if not path.is_absolute():
        path = repository / path
    if not path.is_file():
        return {}
    value = read_json(path)
    records = value.get("assets") if isinstance(value, Mapping) else value
    if not isinstance(records, list):
        return {}
    return {
        str(record["asset_id"]): record
        for record in records
        if isinstance(record, Mapping) and isinstance(record.get("asset_id"), str)
    }


def _registry_attributes(
    asset_id: Any, registry: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = registry.get(str(asset_id)) if isinstance(asset_id, str) else None
    if not isinstance(record, Mapping):
        return {}, {}
    attrs = record.get("realized_attributes")
    identity = record.get("identity")
    return (
        deepcopy(dict(attrs)) if isinstance(attrs, Mapping) else {},
        deepcopy(dict(identity)) if isinstance(identity, Mapping) else {},
    )


def _decorate_actor(
    actor: Mapping[str, Any], registry: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    value = deepcopy(dict(actor))
    asset_id = value.get("asset_id") or value.get("entity_asset_id")
    attrs, identity = _registry_attributes(asset_id, registry)
    if not attrs and isinstance(value.get("realized_attributes"), Mapping):
        attrs = deepcopy(dict(value["realized_attributes"]))
    if attrs:
        value["registered_appearance"] = deepcopy(attrs)
        value["realized_attributes"] = attrs
        from avengine.runtime_profiles import PIXEL_APPEARANCE_VALUE_VOCABULARY
        candidates = [(field, attrs[field]) for field in
                      ("finish", "surface_finish", "body_color", "top_color")
                      if isinstance(attrs.get(field), str) and attrs[field].strip()]
        coat = attrs.get("coat_profile")
        if isinstance(coat, Mapping) and isinstance(coat.get("value"), str):
            candidates.append(("coat_profile.value", coat["value"]))
        supported = [(field, appearance) for field, appearance in candidates
                     if appearance.casefold() in PIXEL_APPEARANCE_VALUE_VOCABULARY]
        # A supplementary unsupported finish must not hide an existing,
        # classifier-supported colour. Keep the old first field as an honest
        # unavailable observation when no registered candidate is supported.
        if candidates:
            field, appearance = (supported or candidates)[0]
            value["appearance"] = {"field": field, "value": appearance,
                                   "label": value.get("display_label", asset_id or "source")}
    if identity:
        value.setdefault("identity", identity)
        if identity.get("species_id") is not None:
            value.setdefault("species_id", identity["species_id"])
    return value


def _plan_actor_list(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    visual = plan.get("visual_plan")
    values = visual.get("actors") if isinstance(visual, Mapping) else None
    if values is None:
        values = plan.get("actors")
    if isinstance(values, Mapping):
        return [dict(value, actor_id=str(key)) for key, value in values.items() if isinstance(value, Mapping)]
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        return [dict(value) for value in values if isinstance(value, Mapping)]
    return []


def _build_habitat_plan(
    capture_root: Path,
    capture_receipt: Mapping[str, Any],
    truth: Mapping[str, Any],
    audio_report: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    inputs = capture_receipt.get("inputs")
    inputs = inputs if isinstance(inputs, Mapping) else {}
    case_value = inputs.get("case_manifest")
    case_path = Path(str(case_value)).expanduser().resolve() if case_value else None
    case = _read_optional(case_path)
    case = case or {}
    track_by_slot: dict[str, Mapping[str, Any]] = {}
    tracks = case.get("actor_tracks")
    if isinstance(tracks, list):
        for track in tracks:
            if isinstance(track, Mapping) and isinstance(track.get("source_slot_id"), str):
                track_by_slot[str(track["source_slot_id"])] = track
    receipt_by_slot: dict[str, Mapping[str, Any]] = {}
    actors_value = capture_receipt.get("actors")
    if isinstance(actors_value, list):
        for actor in actors_value:
            if isinstance(actor, Mapping) and isinstance(actor.get("source_slot_id"), str):
                receipt_by_slot[str(actor["source_slot_id"])] = actor
    instances = truth.get("per_instance")
    instances = instances if isinstance(instances, Mapping) else {}
    actors: list[dict[str, Any]] = []
    for slot in sorted(set(track_by_slot) | set(receipt_by_slot) | set(instances)):
        track = track_by_slot.get(slot, {})
        observed = receipt_by_slot.get(slot, {})
        asset_id = observed.get("asset_id") or track.get("asset_id")
        registered = registry.get(str(asset_id)) if isinstance(asset_id, str) else None
        attrs, identity = _registry_attributes(asset_id, registry)
        entity_class = (
            observed.get("entity_class")
            or track.get("entity_class")
            or (registered.get("entity_class") if isinstance(registered, Mapping) else None)
            or "unknown"
        )
        semantic_id = instances.get(slot, {}).get("semantic_id") if isinstance(instances.get(slot), Mapping) else observed.get("semantic_id")
        actor: dict[str, Any] = {
            "actor_id": slot,
            "native_actor_id": observed.get("actor_id"),
            "source_slot_id": slot,
            "source_endpoint_id": track.get("source_endpoint_id") or observed.get("source_endpoint_id"),
            "asset_id": asset_id,
            "entity_class": entity_class,
            "semantic_id": semantic_id,
            "display_label": registered.get("display_label", slot) if isinstance(registered, Mapping) else slot,
            "realized_attributes": attrs,
            "registered_appearance": attrs,
            "identity": identity,
        }
        if identity.get("species_id") is not None:
            actor["species_id"] = identity["species_id"]
        actors.append(actor)
    room_manifest = inputs.get("room_manifest")
    if not isinstance(room_manifest, str) or not Path(room_manifest).is_file():
        raise ValueError("Habitat capture lacks an actual room manifest identity")
    manifest = _read_optional(Path(room_manifest).resolve())
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("room_id"), str) or not manifest["room_id"].strip():
        raise ValueError("Habitat room manifest lacks a readable room_id")
    room_id = manifest["room_id"].strip()
    clock = case.get("clock")
    if not isinstance(clock, Mapping):
        planned_value = case.get("planned_timeline_path")
        planned_path = Path(str(planned_value)).expanduser().resolve() if isinstance(planned_value, str) else None
        planned = _read_optional(planned_path)
        clock = planned.get("render") if isinstance(planned, Mapping) else None
    if not isinstance(clock, Mapping):
        raise ValueError("Habitat case lacks an authoritative case/planned clock")
    plan: dict[str, Any] = {
        "kind": "avengine_habitat_contract_episode",
        "plan_coordinates": "renderer_neutral",
        "episode_id": capture_root.name,
        "seed": 0,
        "clock": deepcopy(dict(clock)),
        "scene": {"scene_id": room_id, "room_id": room_id},
        "actors": actors,
        "visual_plan": {
            "backend_role": "production_visual",
            "actors": actors,
            "frames": [],
            "render": deepcopy(dict(clock)),
            "authority": {"actor_state": "native_habitat_readback", "camera_listener": "native_habitat_readback", "backend_may_replan": False},
        },
        "resources": {
            "manifest": room_manifest,
            "acoustic_package": (
                audio_report.get("inputs", {}).get("package_manifest", {}).get("path")
                if isinstance(audio_report.get("inputs"), Mapping)
                and isinstance(audio_report["inputs"].get("package_manifest"), Mapping)
                else None
            ),
            "backend": "habitat_native",
        },
        "source_endpoint_bindings": [
            {
                "source_endpoint_id": actor.get("source_endpoint_id"),
                "actor_id": actor["actor_id"],
            }
            for actor in actors
            if isinstance(actor.get("source_endpoint_id"), str)
        ],
        "status": "research_only",
    }
    return plan


def _normal_frame_readbacks(
    neutral: Mapping[str, Any], actors: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    entities = neutral.get("entities")
    if not isinstance(entities, Mapping):
        raise ValueError("NeutralReadback has no entities")
    identities = neutral.get("entity_identities")
    identities = identities if isinstance(identities, Mapping) else {}
    actor_frames: dict[str, list[dict[str, Any]]] = {}
    emitter_frames: dict[str, list[dict[str, Any]]] = {}
    for slot, rows in entities.items():
        if not isinstance(rows, list):
            raise ValueError(f"NeutralReadback entity rows are invalid: {slot}")
        actor_id = str(slot)
        identity = identities.get(slot) if isinstance(identities.get(slot), Mapping) else {}
        actor = actors.get(actor_id, {})
        actor_frames[actor_id] = []
        emitter_frames[actor_id] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"NeutralReadback entity row is invalid: {slot}")
            index = int(row["frame_index"])
            base = {
                "frame_index": index,
                "pts_ticks": row.get("pts_ticks"),
                "actor_id": actor_id,
                "native_actor_id": identity.get("actor_id"),
                "asset_id": actor.get("asset_id"),
                "moving": row.get("moving"),
            }
            actor_frames[actor_id].append({**base, "position_m": list(row["root"])})
            emitter_frames[actor_id].append({
                **base,
                "position_m": list(row["emitter"]),
                "source_endpoint_id": actor.get("source_endpoint_id"),
            })
    camera = [dict(row) for row in neutral.get("camera", []) if isinstance(row, Mapping)]
    return {
        "clock": deepcopy(dict(neutral["clock"])),
        "camera": camera,
        "listener": camera,
        "actors": actor_frames,
        "emitters": emitter_frames,
        "coordinate_frame": deepcopy(dict(neutral.get("coordinate_frame", {}))),
    }


def _source_endpoint_index(actors: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    return {
        str(record["source_endpoint_id"]): actor_id
        for actor_id, record in actors.items()
        if isinstance(record.get("source_endpoint_id"), str)
    }


def _contract_audio_report(
    report: Mapping[str, Any],
    endpoint_to_actor: Mapping[str, str],
    output_path: Path,
) -> dict[str, Any]:
    value = deepcopy(dict(report))
    events_value = value.get("events")
    if not isinstance(events_value, list):
        raise ValueError("audio research report requires an events list")
    events: list[dict[str, Any]] = []
    for event in events_value:
        if not isinstance(event, Mapping):
            raise ValueError("audio research report event must be an object")
        row = dict(event)
        endpoint = row.get("source_endpoint_id")
        actor_id = row.get("actor_id") or row.get("voice_binding_actor_id")
        if not isinstance(actor_id, str) or not actor_id:
            actor_id = endpoint_to_actor.get(str(endpoint)) if endpoint is not None else None
        if not isinstance(actor_id, str) or not actor_id:
            raise ValueError(f"audio event cannot bind source endpoint: {endpoint!r}")
        row["actor_id"] = actor_id
        events.append(row)
    value["events"] = events
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, value)
    return value


def _audio_program_path(report: Mapping[str, Any]) -> Path:
    value = report.get("audio_program")
    if isinstance(value, Mapping):
        path = value.get("path")
    elif isinstance(value, str):
        path = value
    else:
        path = None
    path = path or report.get("audio_program_path")
    if not isinstance(path, str) or not path:
        raise ValueError("audio report lacks an actual audio_program path")
    result = Path(path).expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(f"audio_program is unavailable: {result}")
    return result


def _audio_media_info(path: Path) -> dict[str, int]:
    """Read either P6 float32 WAVE or historical PCM WAVE metadata."""
    try:
        from avengine.spatial_audio.audio import read_float32_wav
        value = read_float32_wav(path)
        return {
            "channel_count": value.channel_count,
            "sample_rate_hz": value.sample_rate_hz,
            "sample_count": value.frame_count,
            "sample_width": 4,
        }
    except Exception:
        with wave.open(str(path), "rb") as stream:
            return {
                "channel_count": stream.getnchannels(),
                "sample_rate_hz": stream.getframerate(),
                "sample_count": stream.getnframes(),
                "sample_width": stream.getsampwidth(),
            }


def _appearance_registry_for_plan(
    plan: Mapping[str, Any], registry: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    return {
        str(actor["actor_id"]): _decorate_actor(actor, registry)
        for actor in _plan_actor_list(plan)
        if isinstance(actor.get("actor_id"), str)
    }


def _find_visual_media(capture_root: Path, report: Mapping[str, Any]) -> Path | None:
    candidates = [
        capture_root / "ue_visual_only.mp4",
        capture_root / "visual.mp4",
        Path(str(report.get("video_path"))).expanduser() if isinstance(report.get("video_path"), str) else None,
    ]
    for path in candidates:
        if path is not None and path.is_file():
            return path.resolve()
    return None


def _probe_video(path: Path) -> dict[str, Any]:
    """Read native video frame/rate metadata from an actual file."""
    command = [
        "ffprobe", "-v", "error", "-count_frames",
        "-show_entries", "stream=codec_type,nb_read_frames,r_frame_rate,width,height",
        "-of", "json", str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") if isinstance(payload, Mapping) else None
    video = next((row for row in streams or () if row.get("codec_type") == "video"), None)
    if not isinstance(video, Mapping):
        raise ValueError(f"visual video has no video stream: {path}")
    raw_frames = video.get("nb_read_frames")
    if raw_frames in (None, "N/A"):
        raise ValueError(f"ffprobe did not count visual video frames: {path}")
    rate_value = str(video.get("r_frame_rate", "0/1"))
    numerator, denominator = rate_value.split("/", 1)
    frame_rate = float(numerator) / float(denominator)
    return {
        "path": str(path.resolve()),
        "frame_count": int(raw_frames),
        "frame_rate_hz": frame_rate,
        "width": int(video.get("width", 0)),
        "height": int(video.get("height", 0)),
    }


def _rawvideo_encoder_command(
    *, output_path: Path, width: int, height: int, frame_rate_hz: float, frame_count: int,
) -> list[str]:
    """Reuse the established raw RGB24 to H.264 encoder contract."""
    from avengine.optional_backends.spear_apartment import build_rawvideo_encode_command

    if float(frame_rate_hz).is_integer():
        return build_rawvideo_encode_command(
            output_path=output_path,
            width=int(width),
            height=int(height),
            frame_rate_hz=int(frame_rate_hz),
            frame_count=int(frame_count),
            pixel_format="rgb24",
        )
    return [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{int(width)}x{int(height)}", "-framerate", str(frame_rate_hz),
        "-i", "pipe:0", "-frames:v", str(int(frame_count)), "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path),
    ]


def _encode_rgb_frames_to_video(
    frames: Any, *, output_path: Path, frame_rate_hz: float, expected_frame_count: int,
) -> dict[str, Any]:
    import numpy as np

    values = np.asarray(frames)
    if values.ndim != 4 or values.shape[-1] != 3 or values.dtype != np.uint8:
        raise ValueError("native RGB frames must be uint8 [frame,height,width,3]")
    if values.shape[0] != int(expected_frame_count):
        raise ValueError("native RGB frame count differs from the episode clock")
    height, width = (int(values.shape[1]), int(values.shape[2]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = _rawvideo_encoder_command(
        output_path=output_path,
        width=width,
        height=height,
        frame_rate_hz=float(frame_rate_hz),
        frame_count=int(expected_frame_count),
    )
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        assert process.stdin is not None
        for frame in values:
            process.stdin.write(np.ascontiguousarray(frame).tobytes(order="C"))
        process.stdin.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        return_code = process.wait()
    except BaseException:
        if process.poll() is None:
            process.kill()
            process.wait()
        raise
    if return_code != 0:
        raise RuntimeError(
            "RGB rawvideo encode failed: " + stderr.decode("utf-8", errors="replace").strip()
        )
    probe = _probe_video(output_path)
    if probe["frame_count"] != int(expected_frame_count):
        raise ValueError(
            f"encoded RGB video frame count differs from clock: {probe['frame_count']} != {expected_frame_count}"
        )
    return {
        "status": "pass",
        "path": str(output_path.resolve()),
        "source_frame_count": int(values.shape[0]),
        "resolution_hw": [height, width],
        "frame_rate_hz": float(frame_rate_hz),
        "probe": probe,
        "encoding": "actual_rgb24_frames_to_h264_video",
    }


SHARED_VISUAL_EVIDENCE_DIRECTORY = "shared_visual_evidence"
SHARED_VISUAL_MASTER_NAME = "visual_rgb.mp4"


def _capture_is_shared(root: Path, capture_root: Path) -> dict[str, Any]:
    """Report whether this episode reuses a visual capture owned elsewhere.

    `materialize_audio_variant` links a member's `capture` at the group's one
    rendered visual episode and records that reuse in `native_linkage.json`.
    Either signal means several audio members read the same pixels, which is
    exactly when the visual evidence is worth sharing.
    """
    link = root / "capture"
    reasons: list[str] = []
    if link.is_symlink():
        reasons.append("capture is a symlink to a visual episode owned elsewhere")
    linkage = _read_optional(root / "native_linkage.json")
    if isinstance(linkage, Mapping) and linkage.get("native_capture_reused") is True:
        reasons.append("native_linkage.json declares native_capture_reused")
    return {
        "shared": bool(reasons),
        "reasons": reasons,
        "capture_root": str(Path(capture_root).resolve()),
        "member_id": linkage.get("member_id") if isinstance(linkage, Mapping) else None,
    }


def _resolve_shared_visual_root(
    root: Path,
    derived: Path,
    capture_root: Path,
    *,
    shared_visual_root: Path | str | None,
    visual_evidence_reuse: bool | None,
) -> tuple[Path | None, dict[str, Any]]:
    """Decide where this episode's shared visual evidence lives.

    An explicit root always wins. Otherwise reuse turns itself on only for a
    declared shared capture, and the default location is the directory that
    holds the members, so one group's members share and unrelated episodes do
    not.
    """
    scope = _capture_is_shared(root, capture_root)
    if visual_evidence_reuse is False:
        return None, {**scope, "enabled": False, "source": "caller_disabled_reuse"}
    if shared_visual_root is not None:
        return Path(shared_visual_root).expanduser().resolve(), {
            **scope, "enabled": True, "source": "caller_declared_shared_visual_root",
        }
    if visual_evidence_reuse is None and not scope["shared"]:
        return None, {
            **scope, "enabled": False,
            "source": "capture_is_not_declared_shared_between_audio_members",
        }
    default = (derived.parent.parent / SHARED_VISUAL_EVIDENCE_DIRECTORY).resolve()
    return default, {
        **scope, "enabled": True,
        "source": "default_shared_root_beside_the_member_roots",
    }


def _publish_visual_master(
    encoded_path: Path, publish_to: Path | None, encoded: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Publish a finished encode without replacing an existing shared master."""
    if publish_to is None:
        return encoded_path.resolve(), encoded
    try:
        # A hard-link create is atomic and has no-replace semantics on the
        # same filesystem. This closes the first-creator race between members
        # that resolve to one capture.
        os.link(encoded_path, publish_to)
    except FileExistsError:
        existing_probe = _probe_video(publish_to)
        encoded_probe = encoded.get("probe")
        if not isinstance(encoded_probe, Mapping):
            raise ValueError("encoded shared visual master lacks a video probe")
        if (
            existing_probe["frame_count"] != encoded_probe["frame_count"]
            or abs(existing_probe["frame_rate_hz"] - encoded_probe["frame_rate_hz"]) > 1.0e-3
        ):
            raise ValueError("existing shared visual master has a different clock")
        encoded["path"] = str(publish_to.resolve())
        encoded["published_shared_master"] = False
        encoded["reused"] = True
        encoded["source"] = "shared_visual_master_existing"
        encoded["source_encode_path"] = str(encoded_path)
        encoded["probe"] = existing_probe
        try:
            encoded_path.unlink()
        except FileNotFoundError:
            pass
        return publish_to.resolve(), encoded
    encoded["path"] = str(publish_to.resolve())
    encoded["published_shared_master"] = True
    encoded["source_encode_path"] = str(encoded_path)
    try:
        encoded_path.unlink()
    except FileNotFoundError:
        pass
    return publish_to.resolve(), encoded


def _prepare_visual_video(
    capture_root: Path, *, clock: Mapping[str, Any], output_path: Path,
    shared_master_path: Path | None = None,
) -> tuple[Path | None, dict[str, Any]]:
    expected_frames = int(clock["frame_count"])
    expected_rate = float(clock["frame_rate_hz"])
    publish_to: Path | None = None
    existing = _find_visual_media(capture_root, {})
    if existing is not None:
        probe = _probe_video(existing)
        if probe["frame_count"] != expected_frames or abs(probe["frame_rate_hz"] - expected_rate) > 1.0e-3:
            raise ValueError("existing visual video does not match the episode clock")
        return existing, {"status": "pass", "source": "existing_native_video", "probe": probe}
    if shared_master_path is not None:
        # Members of one group share a capture, so they share its encode. The
        # master is probed against this episode's own clock before it is used,
        # and a mismatch falls through to a private encode instead of silently
        # accepting a video that belongs to a different capture.
        master = Path(shared_master_path)
        if master.is_file():
            probe = _probe_video(master)
            if probe["frame_count"] == expected_frames and abs(probe["frame_rate_hz"] - expected_rate) <= 1.0e-3:
                return master.resolve(), {
                    "status": "pass",
                    "source": "shared_visual_master",
                    "reused": True,
                    "path": str(master.resolve()),
                    "probe": probe,
                }
        else:
            # Encode to a private name and publish with no-replace semantics,
            # so a member arriving mid-encode sees no master or a complete one.
            master.parent.mkdir(parents=True, exist_ok=True)
            publish_to = master
            output_path = master.parent / f".{master.name}.{os.getpid()}.{time.time_ns()}.tmp.mp4"
    rgb_path = capture_root / "rgb.npy"
    if rgb_path.is_file():
        import numpy as np
        values = np.load(rgb_path, mmap_mode="r", allow_pickle=False)
        encoded = _encode_rgb_frames_to_video(
            values,
            output_path=output_path,
            frame_rate_hz=expected_rate,
            expected_frame_count=expected_frames,
        )
        encoded["source"] = str(rgb_path.resolve())
        return _publish_visual_master(output_path, publish_to, encoded)
    png_paths = sorted((capture_root / "frames").glob("frame_*.png"))
    if png_paths:
        import cv2
        import numpy as np
        if len(png_paths) != expected_frames:
            raise ValueError("native PNG frame count differs from the episode clock")
        frames = []
        for path in png_paths:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"native RGB frame unavailable: {path}")
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        encoded = _encode_rgb_frames_to_video(
            np.stack(frames),
            output_path=output_path,
            frame_rate_hz=expected_rate,
            expected_frame_count=expected_frames,
        )
        encoded["source"] = str((capture_root / "frames").resolve())
        return _publish_visual_master(output_path, publish_to, encoded)
    return None, {
        "status": "not_run",
        "reason": "no native visual video, rgb.npy, or complete PNG frame sequence was supplied",
    }


def _mux_visual_and_audio(
    visual_path: Path, audio_path: Path, *, output_path: Path, clock: Mapping[str, Any],
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
        "-i", str(visual_path), "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
        "-ac", "2", "-ar", str(int(clock["sample_rate_hz"])),
        "-movflags", "+faststart", str(output_path),
    ]
    subprocess.run(command, check=True)
    streams_command = [
        "ffprobe", "-v", "error", "-count_frames",
        "-show_entries", "stream=codec_type,nb_read_frames,r_frame_rate,channels,sample_rate,duration",
        "-of", "json", str(output_path),
    ]
    completed = subprocess.run(streams_command, capture_output=True, text=True, check=True)
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") if isinstance(payload, Mapping) else None
    video = next((row for row in streams or () if row.get("codec_type") == "video"), None)
    audio = next((row for row in streams or () if row.get("codec_type") == "audio"), None)
    if not isinstance(video, Mapping) or not isinstance(audio, Mapping):
        raise ValueError("muxed preview lacks video or audio stream")
    raw_frames = video.get("nb_read_frames")
    if raw_frames in (None, "N/A") or int(raw_frames) != int(clock["frame_count"]):
        raise ValueError("muxed visual video frame count differs from the episode clock")
    numerator, denominator = str(video.get("r_frame_rate", "0/1")).split("/", 1)
    video_rate = float(numerator) / float(denominator)
    if abs(video_rate - float(clock["frame_rate_hz"])) > 1.0e-3:
        raise ValueError("muxed visual video frame rate differs from the episode clock")
    if int(audio.get("channels", 0)) != 2 or int(audio.get("sample_rate", 0)) != int(clock["sample_rate_hz"]):
        raise ValueError("muxed audio is not the required two-channel episode rate")
    return {
        "status": "pass",
        "path": str(output_path.resolve()),
        "video": {"frame_count": int(raw_frames), "frame_rate_hz": video_rate},
        "audio": {
            "channels": int(audio["channels"]),
            "sample_rate_hz": int(audio["sample_rate"]),
            "duration_seconds": float(audio.get("duration", 0.0)),
        },
        "method": "encode_visual_then_independent_audio_mux_without_shortest",
    }


def _build_export_if_possible(
    *,
    derived: Path,
    plan: Mapping[str, Any],
    plan_path: Path | None,
    actual_path: Path,
    capture_root: Path,
    video_path: Path | None,
    mixture: Path,
    questions_path: Path,
    evidence: list[dict[str, Any]],
) -> tuple[str | None, dict[str, Any]]:
    if video_path is None:
        return None, {"status": "not_run", "reason": "native video master is unavailable for export"}
    if plan_path is None:
        return None, {"status": "not_run", "reason": "a plan file is required for export"}
    resources = plan.get("resources")
    resources = resources if isinstance(resources, Mapping) else {}
    room_manifest = resources.get("manifest") or resources.get("room_manifest")
    acoustic_package = resources.get("acoustic_package")
    shared: list[dict[str, Any]] = []
    for path_value, role in ((room_manifest, "room_manifest"), (acoustic_package, "shared_acoustic_geometry_and_materials")):
        if isinstance(path_value, str) and Path(path_value).is_file():
            shared.append({"path": str(Path(path_value).resolve()), "role": role})
    if not shared:
        return None, {"status": "not_run", "reason": "room shared resources are unavailable for export"}
    from avengine.dataset.episode_export import export_episode_bundle
    export_request = {
        "schema": "avengine_episode_export_request_v1",
        "evidence_retention": {"mode": "extended", "include_extended": True, "preserve_required": True},
        "rooms": [{"room_id": plan.get("scene", {}).get("room_id", plan.get("episode_id", capture_root.name)),
                   "shared_resources": shared,
                   "episodes": [{
                       "episode_id": plan.get("episode_id", capture_root.name),
                       "status": "research_only",
                       "plan": str(plan_path),
                       "actual": str(actual_path),
                       "media": {"video_master": str(video_path), "stereo_wav": str(mixture)},
                       "qa": str(questions_path),
                       "evidence": evidence,
                   }]}],
        "model_evaluations": [],
    }
    export_request_path = derived / "export_request.json"
    write_json(export_request_path, export_request)
    export_root = derived / "export"
    manifest = export_episode_bundle(request_path=export_request_path, output_root=export_root, gzip_json=True)
    return str(export_root), {"status": "pass", "manifest": manifest}



def _rendered_sound_registry(request, audio_program, audio_report, *, repository):
    """Bind sound semantics to the pool entries used by the completed audio."""
    inputs = audio_report.get("inputs") or {}
    rendered = inputs.get("dry_assets") if isinstance(inputs, Mapping) else None
    pool_value = request.get("sound_pool")
    if not pool_value or not isinstance(rendered, Mapping):
        return {"sounds": [], "status": "unavailable",
                "reason": "sound pool or rendered dry-asset references are absent"}
    pool_path = Path(pool_value).expanduser()
    if not pool_path.is_absolute():
        pool_path = Path(repository) / pool_path
    pool_path = pool_path.resolve()
    payload = read_json(pool_path)
    rows = payload.get("sounds", []) if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list):
        raise ValueError("sound pool must contain a list of sounds")
    by_id = {row.get("sound_asset_id"): row for row in rows if isinstance(row, Mapping)}
    used_ids = {
        event.get("sound_asset_id") for event in audio_program.get("events", [])
        if isinstance(event, Mapping) and event.get("sound_asset_id")
    }
    records = []
    for sound_id in sorted(used_ids):
        source = by_id.get(sound_id)
        observed = rendered.get(sound_id)
        if not isinstance(source, Mapping) or not isinstance(observed, Mapping):
            raise ValueError(f"rendered sound {sound_id!r} is absent from the declared pool or audio inputs")
        source_path = Path(source["path"]).expanduser()
        if not source_path.is_absolute():
            source_path = pool_path.parent / source_path
        observed_path = Path(observed["path"]).expanduser()
        if not observed_path.is_absolute():
            observed_path = Path(repository) / observed_path
        if source_path.resolve() != observed_path.resolve():
            raise ValueError(f"sound pool PCM differs from the rendered input for {sound_id}")
        record = deepcopy(dict(source))
        record["semantic_sound_class"] = source.get("sound_class") or source.get("event_class")
        record["rendered_pcm_path"] = str(observed_path.resolve())
        record["rendered_pcm_sha256"] = observed.get("sha256")
        records.append(record)
    return {"sounds": records, "status": "matched_rendered_inputs",
            "source_pool": str(pool_path), "sound_count": len(records)}


def _reviewed_occluder_registry(review, actors):
    """Reuse the question renderer's readable labels for reviewed occluders."""
    from avengine.qa.unified_catalog import _Deferred, _appearance_phrases

    result = {}
    reviews = review.get("actors")
    if not isinstance(reviews, Mapping):
        return result
    for actor_id, record in reviews.items():
        if not isinstance(record, Mapping) or record.get("status") not in {
            "pass", "reviewed", "astra_reviewed"
        }:
            continue
        actor = actors.get(str(actor_id), {})
        label = actor.get("display_label")
        if not isinstance(label, str) or not label.strip():
            continue
        value = record.get("value") or record.get("attribute_value")
        appearance = {
            "field": record.get("attribute_field") or record.get("appearance_field_used"),
            "value": value, "label": label,
        }
        try:
            label_en, label_zh = _appearance_phrases(appearance)
        except _Deferred:
            continue
        result[str(actor_id)] = {
            "display_label": label_en, "display_label_zh": label_zh,
            "appearance_value": value,
            "entity_kind": str(record.get("entity_kind") or actor.get("entity_class") or "entity"),
        }
    return result


def _ancillary_audio_outputs(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Collect non-canonical audio renders the report declares beside the mix.

    The two-channel mixture stays the canonical delivery and keeps its own
    clock and channel checks. An ambisonic or first-order render is carried
    through as an extra, optional artifact so a consumer can find it without
    any of the binaural contract changing. Nothing is inferred from a filename
    the report did not declare.
    """
    outputs: list[dict[str, Any]] = []
    seen: set[str] = set()
    candidates: list[tuple[Any, str]] = []
    audio_record = report.get("audio")
    layout_delivery = (audio_record.get("layout_delivery")
                       if isinstance(audio_record, Mapping) else None)
    if isinstance(layout_delivery, Mapping):
        clock = report.get("clock") if isinstance(report.get("clock"), Mapping) else {}
        for layout_name, declaration in layout_delivery.items():
            if not isinstance(declaration, Mapping):
                raise ValueError(f"audio.layout_delivery.{layout_name} must be an object")
            layout_type = str(declaration.get("layout_type") or layout_name)
            if layout_type == "binaural":
                continue
            mixture = declaration.get("mixture")
            value = mixture.get("path") if isinstance(mixture, Mapping) else declaration.get("mixture_path")
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"audio.layout_delivery.{layout_name} lacks a mixture path")
            path = Path(value).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"declared ancillary mixture is unavailable: {path}")
            actual = _audio_media_info(path)
            for key in ("channel_count", "sample_rate_hz", "sample_count"):
                expected = declaration.get(key)
                if (isinstance(expected, bool) or not isinstance(expected, int)
                        or expected <= 0 or actual[key] != expected):
                    raise ValueError(f"ancillary {layout_type} {key} differs from its declared audio layout")
            for key in ("sample_rate_hz", "sample_count"):
                expected = clock.get(key)
                if expected is not None and actual[key] != expected:
                    raise ValueError(f"ancillary {layout_type} {key} differs from the episode clock")
            labels = declaration.get("channel_labels")
            if (not isinstance(labels, (list, tuple)) or len(labels) != actual["channel_count"]
                    or any(not isinstance(label, str) or not label.strip() for label in labels)):
                raise ValueError(f"ancillary {layout_type} lacks matching channel labels")
            for key in ("layout_id", "channel_order", "normalization", "coordinate_frame"):
                if not isinstance(declaration.get(key), str) or not declaration[key].strip():
                    raise ValueError(f"ancillary {layout_type} lacks an explicit {key}")
            resolved = str(path)
            if resolved in seen:
                continue
            seen.add(resolved)
            outputs.append({
                "path": resolved, "role": f"ancillary_audio_{layout_type}",
                "required": False, "canonical": False,
                "source": f"audio.layout_delivery.{layout_name}",
                "layout_type": layout_type,
                **{key: deepcopy(declaration[key]) for key in (
                    "layout_id", "channel_count", "channel_labels", "channel_order",
                    "normalization", "coordinate_frame", "sample_rate_hz", "sample_count",
                )},
                "foa_normalization": deepcopy(declaration.get("foa_normalization")),
                "media_readback": actual,
            })
    for key in ("ambisonic_path", "foa_path", "ambisonics_path"):
        candidates.append((report.get(key), key))
    for container_key in ("ancillary_outputs", "additional_outputs", "auxiliary_audio"):
        container = report.get(container_key)
        if isinstance(container, Sequence) and not isinstance(container, (str, bytes)):
            for row in container:
                if isinstance(row, Mapping):
                    candidates.append((row.get("path"), str(row.get("role") or container_key)))
        elif isinstance(container, Mapping):
            for role, value in container.items():
                candidates.append((value.get("path") if isinstance(value, Mapping) else value, str(role)))
    audio = report.get("audio")
    if isinstance(audio, Mapping):
        for key in ("ambisonic_path", "foa_path", "ambisonics_path"):
            candidates.append((audio.get(key), key))
    for value, role in candidates:
        if not isinstance(value, str) or not value:
            continue
        path = Path(value).expanduser()
        if not path.is_file():
            continue
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        outputs.append({
            "path": resolved,
            "role": f"ancillary_audio_{role}",
            "required": False,
            "canonical": False,
        })
    return outputs


def _sampling_for_delivery(plan: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    """Carry named targets across the instance-to-native-actor boundary."""
    original = plan.get("request") or {}
    sampling = deepcopy(request.get("qa_sampling") or original.get("qa_sampling") or {})
    targets = request.get("qa_targets", original.get("qa_targets"))
    if targets is None:
        return sampling
    actor_ids = {}
    for row in plan.get("entity_instances", ()):
        actor_id = row.get("actor_id")
        if actor_id:
            actor_ids[str(actor_id)] = str(actor_id)
            if row.get("entity_instance_id"):
                actor_ids[str(row["entity_instance_id"])] = str(actor_id)
    normalized = []
    for raw in targets:
        target = deepcopy(dict(raw))
        ids = target.get("target_instance_ids") or ()
        missing = [str(value) for value in ids if str(value) not in actor_ids]
        if missing:
            raise ValueError(f"QA target instances have no native actor mapping: {missing}")
        target["target_actor_ids"] = [actor_ids[str(value)] for value in ids]
        normalized.append(target)
    sampling["qa_targets"] = normalized
    return sampling


def finalize_qa_episode(
    episode_root: Path, derived_root: Path, *, repository: Path,
    request: Mapping[str, Any] | None = None, audio_report: Path | None = None,
    appearance_review: Path | None = None,
    shared_visual_root: Path | str | None = None,
    visual_evidence_reuse: bool | None = None,
    visual_reuse_verification_frames: int = 2,
) -> dict[str, Any]:
    """Finalize UE or Habitat evidence through the shared contract bundle.

    When several audio members bind to one rendered visual episode, the
    appearance review, actor occluders, visibility annotation and the encoded
    visual master depend only on that shared capture. `shared_visual_root`
    names where those products are published so later members reuse them;
    leaving it unset turns the sharing on only for a capture that declares
    itself reused, and `visual_evidence_reuse=False` always turns it off. The
    audio program, mixture, muxed preview, facts and questions stay per member.
    """
    from avengine.timeline.unified_audio_receipt import validate_unified_audio_receipt

    root, derived = Path(episode_root).expanduser().resolve(), Path(derived_root).expanduser().resolve()
    if derived.exists():
        raise FileExistsError(f"refusing existing derived output: {derived}")
    capture_root = _capture_root(root)
    stage_timings: dict[str, float] = {}
    finalize_started = time.monotonic()
    plan_path = _find_plan_path(root, capture_root)
    plan = read_json(plan_path) if plan_path is not None else None
    capture_receipt = _read_optional(capture_root / "research_receipt.json") or {}
    request_value = dict(request) if isinstance(request, Mapping) else (_read_optional(_find_request_path(root, capture_root)) or {})
    commands: dict[str, Any] = {}
    if audio_report is None:
        if plan is None or plan_path is None:
            raise FileNotFoundError(
                "Habitat contract finalization requires an existing P6 audio_report; "
                "automatic audio rendering needs a plan file"
            )
        derived.mkdir(parents=True)
        audio_root = derived / "audio"
        repository_path = Path(repository).resolve()
        if _is_habitat_plan(plan):
            neutral_for_audio = _read_optional(capture_root / "neutral_readback.json")
            generated_program = _write_habitat_audio_program(
                plan,
                derived / "audio_program.json",
                neutral_readback=neutral_for_audio,
            )
            command = _build_habitat_audio_command(
                request_value, plan, root, capture_root, audio_root,
                generated_program, repository=repository_path,
            )
        else:
            command_root = root if (root / "capture").is_dir() else plan_path.parent.parent
            command = build_audio_command(
                request_value, plan, command_root, audio_root, repository=repository_path,
                capture_root=capture_root, plan_root=plan_path.parent,
            )
        commands["audio"] = command
        write_json(derived / "commands.json", commands)
        with (derived / "audio.log").open("x") as log:
            subprocess.run(command, cwd=repository_path, stdout=log, stderr=subprocess.STDOUT, check=True)
        audio_report = audio_root / "research_report.json"
        if not audio_report.is_file():
            audio_report = audio_root / "research_receipt.json"
    report_path = _find_audio_report(root, capture_root, Path(audio_report).expanduser() if audio_report is not None else None)
    report = read_json(report_path)
    if not isinstance(report, Mapping):
        raise ValueError("audio report must be an object")
    truth_path = capture_root / "pixel_visibility_truth.json"
    masks_path = capture_root / "native_pixel_masks_depth_authority_v1.npz"
    truth = read_json(truth_path)
    if not isinstance(truth, Mapping):
        raise ValueError("pixel visibility truth must be an object")
    raw_truth = truth
    registry = _asset_registry(
        Path(repository).resolve(),
        request_value.get("source_registry") or (
            (plan.get("request") or {}).get("source_registry") if isinstance(plan, Mapping) else None
        ),
    )
    if plan is None:
        plan = _build_habitat_plan(capture_root, capture_receipt, truth, report, registry)
        plan_path = Path(capture_receipt.get("inputs", {}).get("case_manifest", capture_root / "native_case_manifest.json")).expanduser().resolve() if isinstance(capture_receipt.get("inputs"), Mapping) and capture_receipt["inputs"].get("case_manifest") else None
    else:
        plan = deepcopy(dict(plan))
    if not isinstance(plan.get("clock"), Mapping):
        raise ValueError("finalization plan lacks the authoritative clock")
    plan_clock = validate_clock(plan["clock"])
    report_clock = report.get("clock")
    if not isinstance(report_clock, Mapping):
        raise ValueError("audio report lacks the authoritative clock")
    report_clock = validate_clock(report_clock)
    for key in ("frame_count", "frame_rate_hz", "sample_rate_hz", "sample_count", "time_base_hz", "ticks_per_frame"):
        if plan_clock[key] != report_clock[key]:
            raise ValueError(f"audio report clock differs from plan: {key}")
    neutral_value = capture_root / "neutral_readback.json"
    if not neutral_value.is_file():
        neutral_input = report.get("input_neutral_readback")
        neutral_value = Path(neutral_input.get("path")).expanduser().resolve() if isinstance(neutral_input, Mapping) and isinstance(neutral_input.get("path"), str) else neutral_value
    if not neutral_value.is_file():
        # Older UE captures predate the P1 sidecar. Derive it from the actual
        # centimeter readback into the fresh derived tree, preserving the
        # source file as the authority and validating the written contract.
        source = capture_root / "frame_readbacks.json"
        if source.is_file() and plan_path is not None:
            from avengine.capture.ue_neutral_readback import neutral_from_ue_readbacks
            from avengine.capture.neutral_readback import write_neutral_readback
            neutral_value = derived / "neutral_readback.json"
            derived.mkdir(parents=True, exist_ok=True)
            neutral_data = neutral_from_ue_readbacks(
                read_json(source), plan, source_readbacks=str(source.resolve())
            )
            write_neutral_readback(neutral_value, neutral_data, plan=plan)
    if not neutral_value.is_file():
        raise FileNotFoundError("finalization requires an actual P1 neutral_readback.json")
    neutral = read_json(neutral_value)
    validate_neutral_readback(neutral, plan=plan)
    endpoint_by_actor = _authoritative_endpoint_bindings(plan, neutral)
    if endpoint_by_actor:
        for key in ("actors", "visual_plan"):
            container = plan.get(key)
            if key == "visual_plan" and isinstance(container, Mapping):
                container = container.get("actors")
            if isinstance(container, Mapping):
                values = [
                    dict(actor, actor_id=str(actor_id))
                    for actor_id, actor in container.items()
                    if isinstance(actor, Mapping)
                ]
            elif isinstance(container, Sequence) and not isinstance(container, (str, bytes)):
                values = [dict(actor) for actor in container if isinstance(actor, Mapping)]
            else:
                continue
            for actor in values:
                actor_id = actor.get("actor_id")
                if not isinstance(actor_id, str) or actor.get("source_endpoint_id"):
                    continue
                endpoint = endpoint_by_actor.get(actor_id)
                if isinstance(endpoint, str) and endpoint:
                    actor["source_endpoint_id"] = endpoint
            if key == "visual_plan" and isinstance(plan.get(key), Mapping):
                plan[key] = dict(plan[key], actors=values)
            else:
                plan[key] = values
    actor_declarations = _appearance_registry_for_plan(plan, registry)
    if actor_declarations:
        visual = plan.get("visual_plan")
        if isinstance(visual, Mapping):
            plan["visual_plan"] = dict(visual, actors=list(actor_declarations.values()))
        plan["actors"] = list(actor_declarations.values())
    actors = {
        str(actor["actor_id"]): actor
        for actor in _plan_actor_list(plan)
        if isinstance(actor.get("actor_id"), str)
    }
    if not actors:
        raise ValueError("finalization plan has no actor declarations")
    frame_source = capture_root / "frame_readbacks.json"
    if not frame_source.is_file():
        frame_source = capture_root / "frame_records.json"
    if not frame_source.is_file():
        source_values = neutral.get("producer", {}).get("source_readbacks") if isinstance(neutral.get("producer"), Mapping) else None
        if isinstance(source_values, list) and source_values and isinstance(source_values[0], str) and Path(source_values[0]).is_file():
            frame_source = Path(source_values[0]).resolve()
        else:
            frame_source = neutral_value
    frame_readbacks = _normal_frame_readbacks(neutral, actors)
    endpoint_to_actor = _source_endpoint_index(actors)
    for endpoint, actor_id in list(endpoint_to_actor.items()):
        if not actor_id:
            raise ValueError(f"source endpoint has no actor binding: {endpoint}")
    derived.mkdir(parents=True, exist_ok=True)
    contract_report_path = derived / "research_report.json"
    contract_report = _contract_audio_report(report, endpoint_to_actor, contract_report_path)
    if contract_report.get("schema") == "avengine_unified_audio_receipt_v1":
        validate_unified_audio_receipt(contract_report)
    program_path = _audio_program_path(contract_report)
    mixture_value = contract_report.get("mixture_path")
    if not isinstance(mixture_value, str) or not mixture_value:
        raise ValueError("audio report lacks an actual mixture_path")
    mixture = Path(mixture_value).expanduser().resolve()
    if not mixture.is_file():
        raise FileNotFoundError(f"audio mixture is unavailable: {mixture}")
    media_info = _audio_media_info(mixture)
    clock = plan_clock
    if media_info["channel_count"] != 2 or media_info["sample_rate_hz"] != int(clock["sample_rate_hz"]) or media_info["sample_count"] != int(clock["sample_count"]):
        raise ValueError("actual audio mixture does not match the plan clock and binaural contract")
    # A QA-10 pre-audio check may already have prepared this capture's visual
    # evidence. Reuse the existing content-checked cache in the same output root.
    prepared_visual = root / "native_visibility_visual_evidence"
    if shared_visual_root is None and visual_evidence_reuse is not False and prepared_visual.is_dir():
        shared_visual_root = prepared_visual
    shared_root, shared_scope = _resolve_shared_visual_root(
        root, derived, capture_root,
        shared_visual_root=shared_visual_root,
        visual_evidence_reuse=visual_evidence_reuse,
    )
    started = time.monotonic()
    shared_visual = acquire_shared_visual_evidence(
        capture_root, plan, raw_truth,
        shared_root=shared_root,
        asset_registry=registry,
        frame_stride=1,
        thresholds=nonhuman_appearance_placeholder_thresholds(),
        verification_frames=int(visual_reuse_verification_frames),
    )
    stage_timings["shared_visual_evidence_s"] = time.monotonic() - started
    stage_timings.update(shared_visual.get("stage_timings_s", {}))
    visual_reuse = dict(shared_visual.get("reuse", {}))
    visual_reuse["scope"] = shared_scope
    truth = shared_visual["annotated_pixel_visibility_truth"]
    # The shared pack is what applies the visibility semantics now. Fail closed
    # if a pack ever arrives without them rather than finalizing questions on
    # un-annotated pixel truth.
    if truth.get("visibility_semantics_authority") != "qa_evidence.annotate_pixel_visibility_semantics":
        raise ValueError(
            "shared visual evidence did not apply annotate_pixel_visibility_semantics"
        )
    review = read_json(appearance_review) if appearance_review is not None else None
    if review is None:
        review = shared_visual["appearance_review"]
        pack_dir = visual_reuse.get("pack_dir")
        if isinstance(pack_dir, str) and (Path(pack_dir) / "appearance_review.json").is_file():
            # The shared pack already holds this exact review; point the
            # evidence at it instead of writing another copy per member.
            appearance_path = (Path(pack_dir) / "appearance_review.json").resolve()
        else:
            appearance_path = derived / "appearance_review.json"
            write_json(appearance_path, review)
    else:
        appearance_path = Path(appearance_review).expanduser().resolve()
    if not isinstance(review, Mapping):
        raise ValueError("appearance review must be an object")
    occluders = shared_visual["actor_occluders"]
    pack_dir = visual_reuse.get("pack_dir")
    if isinstance(pack_dir, str) and (Path(pack_dir) / "actor_occluders.json").is_file():
        occluder_path = (Path(pack_dir) / "actor_occluders.json").resolve()
    else:
        occluder_path = derived / "actor_occluders.json"
        write_json(occluder_path, occluders)
    occluder_registry = _reviewed_occluder_registry(review, actors)
    occluder_registry_path = derived / "occluder_registry.json"
    write_json(occluder_registry_path, occluder_registry)
    audio_program = read_json(program_path)
    sound_registry = _rendered_sound_registry(
        request_value, audio_program, report, repository=repository,
    )
    sound_registry_path = derived / "rendered_sound_registry.json"
    write_json(sound_registry_path, sound_registry)
    report_audio = contract_report.get("audio") if isinstance(contract_report.get("audio"), Mapping) else {}
    report_hrtf = contract_report.get("hrtf") if isinstance(contract_report.get("hrtf"), Mapping) else {}
    audio_readback = {
        "channel_count": media_info["channel_count"],
        "sample_rate_hz": media_info["sample_rate_hz"],
        "sample_count": media_info["sample_count"],
        "channel_order": ["left", "right"],
        "proof": "actual_lossless_stereo_WAV_readback",
        "source_path": str(mixture),
        "hrtf_id": report_hrtf.get("id") or contract_report.get("hrtf_id"),
        "source_mix": report_audio.get("source_mix") or contract_report.get("source_mix"),
        "wet_tail_intervals": contract_report.get("wet_tail_intervals", []),
        "event_segmentation": report_audio.get("event_segmentation"),
    }
    voice_bindings_value = plan.get("voice_bindings")
    if not isinstance(voice_bindings_value, (list, Mapping)):
        voice_bindings_value = [
            {
                "actor_id": event["actor_id"],
                "source_endpoint_id": event.get("source_endpoint_id"),
                "sound_asset_id": event.get("sound_asset_id"),
                "transcript": event.get("transcript"),
                "sound_class": event.get("sound_class"),
            }
            for event in contract_report.get("events", [])
            if isinstance(event, Mapping) and isinstance(event.get("actor_id"), str)
        ]
    started = time.monotonic()
    visual_video, visual_video_status = _prepare_visual_video(
        capture_root, clock=clock, output_path=derived / SHARED_VISUAL_MASTER_NAME,
        shared_master_path=(
            shared_visual_pack_root(shared_root, capture_root) / SHARED_VISUAL_MASTER_NAME
            if shared_root is not None else None
        ),
    )
    stage_timings["prepare_visual_video_s"] = time.monotonic() - started
    raw = {
        "episode_id": str(plan.get("episode_id") or capture_root.name),
        "plan": plan,
        "sampling": _sampling_for_delivery(plan, request_value),
        "sampling_policy": request_value.get("sampling_policy") or (plan.get("request") or {}).get("sampling_policy"),
        "actors": actors,
        "frame_readbacks": frame_readbacks,
        "pixel_visibility_truth": truth,
        "audio_program": audio_program,
        "audio_readback": audio_readback,
        "research_report": contract_report,
        "voice_bindings": voice_bindings_value,
        "sound_registry": sound_registry,
        "sound_registry_path": str(sound_registry_path.resolve()),
        "source_endpoint_bindings": [
            {"source_endpoint_id": endpoint, "actor_id": actor_id}
            for endpoint, actor_id in endpoint_to_actor.items()
        ],
        "appearance_review": review,
        "occluder_evidence": occluders,
        "occluder_registry": occluder_registry,
        "plan_path": str(plan_path) if plan_path is not None else None,
        "video_path": None,
        "frame_readbacks_path": str(frame_source.resolve()),
        "pixel_visibility_truth_path": str(truth_path.resolve()),
        "audio_program_path": str(program_path),
        "audio_readback_path": str(mixture),
        "research_report_path": str(contract_report_path.resolve()),
        "appearance_review_path": str(appearance_path.resolve()),
        "occluder_evidence_path": str(occluder_path.resolve()),
        "occluder_registry_path": str(occluder_registry_path.resolve()),
    }
    preview = None
    preview_status: dict[str, Any]
    final_video = None
    if visual_video is not None:
        preview = derived / "preview.mp4"
        mux_status = _mux_visual_and_audio(
            visual_video, mixture, output_path=preview, clock=clock
        )
        commands["preview"] = {
            "method": mux_status["method"],
            "visual_input": str(visual_video),
            "audio_input": str(mixture),
            "output": str(preview.resolve()),
        }
        final_video = preview.resolve()
        preview_status = {"status": "pass", "visual": visual_video_status, "mux": mux_status}
    else:
        preview_status = {
            "status": "not_run",
            "visual": visual_video_status,
            "reason": "native video master is unavailable; facts/questions use actual frame readback and RGB evidence",
        }
    raw["video_path"] = str(final_video) if final_video is not None else None
    contract_files = {
        "pixel_visibility_truth.json": truth_path,
        "native_pixel_masks_depth_authority_v1.npz": masks_path,
        "appearance_review.json": appearance_path,
        "actor_occluders.json": occluder_path,
        "research_report.json": contract_report_path,
    }
    declared_frames = truth.get("frame_indices") if isinstance(truth, Mapping) else None
    full_frame_indices = list(range(int(clock["frame_count"])))
    contract_validation = validate_evidence_contract(
        contract_files,
        clock=clock,
        require_complete=declared_frames == full_frame_indices,
    )
    write_json(derived / "evidence_contract_validation.json", contract_validation)
    facts_path = derived / "facts.json"
    questions_path = derived / "questions.json"
    from avengine.qa.angular_questions import camera_calibration_from_capture
    raw["camera_calibration"] = camera_calibration_from_capture(capture_root)
    facts = with_derived_sound_class_answer_domain(
        normalize_episode_bundle(raw)
    )
    facts.setdefault("audio", {})["path"] = str(mixture)
    facts["audio"]["actual_path"] = str(mixture)
    facts.setdefault("source_paths", {})["neutral_readback"] = str(neutral_value.resolve())
    facts["source_paths"]["mixture_audio"] = str(mixture)
    facts["source_paths"]["audio_readback"] = str(mixture)
    facts["source_paths"]["research_report"] = str(contract_report_path.resolve())
    facts["source_paths"]["appearance_review"] = str(appearance_path.resolve())
    facts["source_paths"]["occluder_evidence"] = str(occluder_path.resolve())
    facts["source_paths"]["occluder_registry"] = str(occluder_registry_path.resolve())
    facts["source_paths"]["sound_registry"] = str(sound_registry_path.resolve())
    if final_video is not None:
        facts["source_paths"]["video"] = str(final_video)
    qa_ids = request_value.get("qa_ids") if isinstance(request_value, Mapping) else None
    questions = generate_unified_questions(
        facts, qa_ids=qa_ids, seed=str(plan.get("seed", capture_root.name)),
        items_per_type=int(request_value.get("items_per_type", 1)),
    )
    questions.pop("input_facts", None)
    questions["normalized_facts_path"] = str(facts_path.resolve())
    try:
        write_json(facts_path, facts)
        write_json(questions_path, questions)
        from avengine.qa.unified_catalog import model_input_questions
        write_json(derived / "model_inputs.json", model_input_questions(questions))
    except Exception:
        facts_path.unlink(missing_ok=True)
        questions_path.unlink(missing_ok=True)
        raise
    evidence: list[dict[str, Any]] = [
        {"path": str(neutral_value.resolve()), "role": "p1_neutral_readback", "required": True},
        {"path": str(frame_source.resolve()), "role": "native_frame_readback_source", "required": True},
        {"path": str(truth_path.resolve()), "role": "question_pixel_evidence", "required": True},
        {"path": str(masks_path.resolve()), "role": "native_pixel_masks_depth_authority_v1", "required": True},
        {"path": str(contract_report_path.resolve()), "role": "audio_render_report", "required": True},
        {"path": str(program_path), "role": "actual_audio_program", "required": True},
        {"path": str(appearance_path.resolve()), "role": "registered_appearance_review", "required": True},
        {"path": str(occluder_path.resolve()), "role": "actor_occluder_evidence", "required": True},
        {"path": str(occluder_registry_path.resolve()), "role": "actor_occluder_display_registry", "required": True},
        {"path": str(sound_registry_path.resolve()), "role": "rendered_sound_semantic_registry", "required": True},
        {"path": str(facts_path.resolve()), "role": "private_normalized_facts", "required": True},
        {"path": str(questions_path.resolve()), "role": "question_output", "required": True},
    ]
    evidence.append({"path": str((derived / "evidence_contract_validation.json").resolve()), "role": "evidence_contract_validation", "required": True})
    ancillary_audio = _ancillary_audio_outputs(contract_report)
    evidence.extend(ancillary_audio)
    if visual_video is not None:
        evidence.append({"path": str(visual_video.resolve()), "role": "native_visual_video_or_rgb_encode", "required": True})
        evidence.append({"path": str(preview.resolve()), "role": "AAC_preview_with_complete_lossless_audio_source", "required": True})
    export, export_status = _build_export_if_possible(
        derived=derived, plan=plan, plan_path=plan_path, actual_path=frame_source.resolve(), capture_root=capture_root, video_path=final_video, mixture=mixture, questions_path=questions_path, evidence=evidence,
    )
    result = {
        "status": "research_only",
        "episode_id": str(plan.get("episode_id") or capture_root.name),
        "questions": questions.get("counts"),
        "coverage": questions.get("coverage"),
        "facts": str(facts_path.resolve()),
        "questions_path": str(questions_path.resolve()),
        "appearance_review": str(appearance_path.resolve()),
        "occluder_evidence": str(occluder_path.resolve()),
        "evidence_contract": str((derived / "evidence_contract_validation.json").resolve()),
        "visual_video": str(visual_video.resolve()) if visual_video is not None else None,
        "visual_video_status": visual_video_status,
        "shared_visual_evidence": visual_reuse,
        "ancillary_audio_outputs": ancillary_audio,
        "canonical_audio_delivery": "two_channel_binaural_mixture",
        "preview": str(preview.resolve()) if preview is not None else None,
        "preview_status": preview_status,
        "lossless_stereo_wav": str(mixture),
        "export": export,
        "export_status": export_status,
        "contract_bundle": {
            "capture_root": str(capture_root),
            "neutral_readback": str(neutral_value.resolve()),
            "frame_readback_source": str(frame_source.resolve()),
            "audio_report_source": str(report_path),
        },
        "model_evaluation": "not_run",
        "formal_admission": False,
    }
    if questions.get("qa_target_results"):
        result["qa_target_results"] = deepcopy(questions["qa_target_results"])
        result["qa_targets_met"] = all(row["status"] == "met" for row in questions["qa_target_results"])
    stage_timings["finalize_total_s"] = time.monotonic() - finalize_started
    timings_record = {
        "schema": "avengine_qa_finalize_stage_timings_v1",
        "episode_id": result["episode_id"],
        "capture_root": str(capture_root),
        "stage_timings_s": stage_timings,
        "shared_visual_evidence": visual_reuse,
        "visual_video_source": visual_video_status.get("source"),
        "claim_boundary": "wall-clock stage timing on a shared host; not an isolated benchmark",
    }
    write_json(derived / "visual_stage_timings.json", timings_record)
    result["stage_timings"] = timings_record
    result["stage_timings_path"] = str((derived / "visual_stage_timings.json").resolve())
    commands["contract_validation"] = ["validate_evidence_contract", str(derived / "evidence_contract_validation.json")]
    write_json(derived / "input_refs.json", {
        "plan": str(plan_path) if plan_path is not None else None,
        "frame_readbacks": str(frame_source.resolve()),
        "neutral_readback": str(neutral_value.resolve()),
        "pixel_visibility_truth": str(truth_path.resolve()),
        "audio_program": str(program_path),
        "audio_report": str(report_path),
        "contract_audio_report": str(contract_report_path.resolve()),
        "appearance_review": str(appearance_path.resolve()),
        "occluder_evidence": str(occluder_path.resolve()),
        "shared_visual_evidence_pack": visual_reuse.get("pack_dir"),
        "visual_video": str(visual_video.resolve()) if visual_video is not None else None,
    })
    write_json(derived / "commands.json", commands)
    write_json(derived / "result.json", result)
    return result
