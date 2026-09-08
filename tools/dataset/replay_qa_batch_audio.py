#!/usr/bin/env python3
"""Replay QA audio or finalize retained audio in fresh attempts, preserving native captures and history."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import wave

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def code_producer():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(REPOSITORY), *args], text=True).strip()
    import avengine
    changes = git("status", "--porcelain").splitlines()
    return {"cwd": os.getcwd(), "avengine_source": str(Path(avengine.__file__).resolve()), "repository": str(REPOSITORY), "git_commit": git("rev-parse", "HEAD"),
            "code_state": "working_tree" if changes else "committed", "working_tree_changes": changes,
            "python": sys.executable, "argv": sys.argv, "created_at": datetime.now(timezone.utc).isoformat()}


def source_episode(row):
    if row.get("episode_output_root"):
        root = Path(row["episode_output_root"])
    elif row.get("attempt_root"):
        root = Path(row["attempt_root"]) / "episode"
    else:
        raise ValueError(f"no native capture source for {row.get('episode_id')}")
    if not (root / "capture").is_dir() or not (root / "plan/episode_plan.json").is_file():
        raise FileNotFoundError(f"incomplete retained capture: {root}")
    return root.resolve()


def load_audio_rebind_inputs(path):
    """Load a batch-pool map and its target sound rows for an audio replay."""
    input_path = Path(path).expanduser().resolve()
    payload = read_json(input_path)
    map_value = payload.get("batch_sound_asset_id_map")
    if not map_value:
        raise ValueError("audio input manifest lacks batch_sound_asset_id_map")
    map_path = Path(map_value).expanduser().resolve()
    mapping_payload = read_json(map_path)
    rows = mapping_payload.get("mappings")
    if not isinstance(rows, list) or not rows:
        raise ValueError("audio sound-ID map has no mappings")
    mapping = {}
    for row in rows:
        old_id = row.get("old_sound_asset_id")
        new_id = row.get("new_sound_asset_id")
        if not isinstance(old_id, str) or not old_id or not isinstance(new_id, str) or not new_id:
            raise ValueError("audio sound-ID map contains an invalid mapping")
        if old_id in mapping:
            raise ValueError(f"audio sound-ID map repeats {old_id}")
        mapping[old_id] = deepcopy(row)
    batch_prepare = payload.get("batch_prepare_10s")
    target_value = (
        batch_prepare.get("batch_sound_pool")
        if isinstance(batch_prepare, dict)
        else None
    ) or payload.get("target_batch_sound_pool") or payload.get("batch_sound_pool")
    if not target_value:
        raise ValueError("audio input manifest lacks target batch sound pool")
    target_pool_path = Path(target_value).expanduser().resolve()
    target_pool = read_json(target_pool_path)
    sounds = target_pool.get("sounds")
    if not isinstance(sounds, list) or not sounds:
        raise ValueError("target batch sound pool has no sounds")
    target_sounds = {}
    for sound in sounds:
        sound_id = sound.get("sound_asset_id")
        if not isinstance(sound_id, str) or not sound_id or sound_id in target_sounds:
            raise ValueError("target batch sound pool has invalid or duplicate sound IDs")
        target_sounds[sound_id] = sound
    header_cache = {}
    for old_id, row in mapping.items():
        new_id = row["new_sound_asset_id"]
        target = target_sounds.get(new_id)
        if target is None:
            raise ValueError(f"target pool lacks mapped sound {old_id}->{new_id}")
        for key in ("sample_count", "sample_rate_hz"):
            map_key = "old_" + key
            if row.get(map_key) is not None and target.get(key) != row["new_" + key]:
                raise ValueError(f"mapped sound {old_id}->{new_id} changes {key}")
        for key in ("sound_class", "source_origin"):
            if row.get(key) is not None and target.get(key) != row[key]:
                raise ValueError(f"mapped sound {old_id}->{new_id} changes {key}")
        if row.get("new_pool_path") is not None and target.get("path") != row["new_pool_path"]:
            raise ValueError(f"target pool path differs for {old_id}->{new_id}")
        old_duration = row.get("old_active_duration_s")
        new_duration = row.get("new_active_duration_s")
        if old_duration is not None and new_duration is not None and not math.isclose(
            float(old_duration), float(new_duration), abs_tol=1e-9
        ):
            raise ValueError(f"mapped sound {old_id}->{new_id} changes active duration")
        old_path = row.get("old_pool_path")
        new_path = target.get("path") or row.get("new_pool_path")
        _validate_audio_headers(
            old_path,
            new_path,
            row,
            target,
            header_cache=header_cache,
            require_files=True,
        )
    return {
        "input_manifest_path": str(input_path),
        "input_manifest": payload,
        "mapping_path": str(map_path),
        "mapping": mapping,
        "target_pool_path": str(target_pool_path),
        "target_sounds": target_sounds,
        "audio_header_cache": header_cache,
    }


def _read_audio_header(path, *, header_cache=None):
    """Read only the WAV header, caching each canonical path in this attempt."""
    if not path:
        return None
    canonical = str(Path(path).expanduser().resolve())
    cache = header_cache if header_cache is not None else {}
    if canonical in cache:
        return cache[canonical]
    try:
        with wave.open(canonical, "rb") as audio:
            header = {
                "sample_count": int(audio.getnframes()),
                "sample_rate_hz": int(audio.getframerate()),
                "channels": int(audio.getnchannels()),
            }
    except (OSError, EOFError, wave.Error) as exc:
        raise ValueError(f"audio file header is unreadable: {canonical}: {exc}") from exc
    cache[canonical] = header
    return header


def _validate_audio_headers(
    old_path,
    new_path,
    map_row,
    target,
    *,
    header_cache=None,
    require_files=False,
):
    """Verify actual old/new WAV headers against the rebind contract."""
    if not old_path or not new_path:
        if require_files:
            raise ValueError("audio mapping lacks old/new PCM paths for header validation")
        return
    old_file = Path(old_path).expanduser()
    new_file = Path(new_path).expanduser()
    if not old_file.is_file() or not new_file.is_file():
        if require_files:
            missing = old_file if not old_file.is_file() else new_file
            raise ValueError(f"audio mapping PCM path is unavailable: {missing}")
        return
    old_header = _read_audio_header(old_file, header_cache=header_cache)
    new_header = _read_audio_header(new_file, header_cache=header_cache)
    for label, header in (("old", old_header), ("new", new_header)):
        if header["channels"] != 1:
            raise ValueError(
                f"audio {label} PCM must be mono; got {header['channels']} channels"
            )
    for label, header, prefix in (
        ("old", old_header, "old_"),
        ("new", new_header, "new_"),
    ):
        expected_count = map_row.get(prefix + "sample_count")
        expected_rate = map_row.get(prefix + "sample_rate_hz")
        if expected_count is not None and header["sample_count"] != int(expected_count):
            raise ValueError(
                f"audio {label} header sample_count differs from mapping: "
                f"{header['sample_count']} != {expected_count}"
            )
        if expected_rate is not None and header["sample_rate_hz"] != int(expected_rate):
            raise ValueError(
                f"audio {label} header sample_rate_hz differs from mapping: "
                f"{header['sample_rate_hz']} != {expected_rate}"
            )
    if target.get("sample_count") is not None and new_header["sample_count"] != int(target["sample_count"]):
        raise ValueError(
            f"audio new header sample_count differs from target pool: "
            f"{new_header['sample_count']} != {target['sample_count']}"
        )
    if target.get("sample_rate_hz") is not None and new_header["sample_rate_hz"] != int(target["sample_rate_hz"]):
        raise ValueError(
            f"audio new header sample_rate_hz differs from target pool: "
            f"{new_header['sample_rate_hz']} != {target['sample_rate_hz']}"
        )
    if old_header["sample_count"] != new_header["sample_count"]:
        raise ValueError(
            f"audio replay changes actual sample_count: "
            f"{old_header['sample_count']} != {new_header['sample_count']}"
        )
    if old_header["sample_rate_hz"] != new_header["sample_rate_hz"]:
        raise ValueError(
            f"audio replay changes actual sample_rate_hz: "
            f"{old_header['sample_rate_hz']} != {new_header['sample_rate_hz']}"
        )


def _rebind_sound_id(value, context):
    if not isinstance(value, str) or not value:
        raise ValueError("audio sound_asset_id must be nonempty text")
    row = context["mapping"].get(value)
    if row is None:
        if value in context["target_sounds"]:
            return value
        raise ValueError(f"audio sound ID is absent from replay map: {value}")
    return row["new_sound_asset_id"]


_AUDIO_REBOUND_FIELDS = (
    "path", "sample_count", "sample_rate_hz", "active_duration_s",
    "source_activity_intervals_samples", "audible_start_sample",
    "audible_end_sample_exclusive", "source_origin", "source_origin_aliases",
    "source_pcm_path", "source_relative", "source_sha256", "source_metadata_path",
    "source_metadata_manifest", "source_event_registry", "source_normalization",
    "normalization_applied", "event_class", "sound_class", "species_id",
    "sound_identity_id", "sound_identity_keys", "compatible_asset_ids",
    "prepared_audio_id",
)


def _rebind_optional_sound_id(value, context):
    if value is None:
        return None
    return _rebind_sound_id(value, context)


def _mark_offline_normalization(record, context):
    source_normalization = record.get("source_normalization")
    input_normalization = context.get("input_manifest", {}).get("normalization")
    if not isinstance(source_normalization, dict) or not isinstance(input_normalization, dict):
        return
    policy = source_normalization.get("policy")
    if not isinstance(policy, dict):
        return
    if any(
        policy.get(key) != input_normalization.get(key)
        for key in ("mode", "target_dbfs")
    ) or not input_normalization.get("applied_once_at"):
        return
    source_normalization["normalization_applied"] = True
    source_normalization["applied_once_at"] = input_normalization["applied_once_at"]


def _rebind_audio_record(record, context):
    sound_id = record.get("sound_asset_id")
    if sound_id is None:
        return record
    new_id = _rebind_sound_id(sound_id, context)
    map_row = context["mapping"].get(sound_id)
    target = context["target_sounds"].get(new_id)
    if target is None:
        raise ValueError(f"target pool lacks mapped sound {sound_id}->{new_id}")
    old_path = record.get("path") or (map_row or {}).get("old_pool_path")
    new_path = target.get("path") or (map_row or {}).get("new_pool_path")
    _validate_audio_headers(
        old_path,
        new_path,
        map_row or {},
        target,
        header_cache=context.get("audio_header_cache"),
    )
    if map_row is not None:
        for key in ("sample_count", "sample_rate_hz"):
            record_value = record.get(key)
            old_value = map_row.get("old_" + key)
            new_value = map_row.get("new_" + key)
            if record_value is not None and old_value is not None and int(record_value) != int(old_value):
                raise ValueError(f"audio record {sound_id} has incompatible {key}")
            if old_value is not None and new_value is not None and int(old_value) != int(new_value):
                raise ValueError(f"audio replay changes {key} for {sound_id}")
        record_duration = record.get("active_duration_s")
        old_duration = map_row.get("old_active_duration_s")
        new_duration = map_row.get("new_active_duration_s")
        if record_duration is not None and old_duration is not None and not math.isclose(
            float(record_duration), float(old_duration), abs_tol=1e-9
        ):
            raise ValueError(f"audio record {sound_id} has incompatible active duration")
        if old_duration is not None and new_duration is not None and not math.isclose(
            float(old_duration), float(new_duration), abs_tol=1e-9
        ):
            raise ValueError(f"audio replay changes active duration for {sound_id}")
    record["sound_asset_id"] = new_id
    for key in _AUDIO_REBOUND_FIELDS:
        if key in target:
            record[key] = deepcopy(target[key])
    _mark_offline_normalization(record, context)
    return record


def rebind_audio_payload(value, context, *, _parent_key=None):
    """Rebind sound IDs and PCM metadata while leaving timing and runtime gain intact."""
    if isinstance(value, list):
        if _parent_key in {
            "sound_asset_ids",
            "preallocated_sound_asset_ids",
            "selected_sound_asset_ids",
        }:
            return [_rebind_sound_id(item, context) for item in value]
        return [rebind_audio_payload(item, context, _parent_key=_parent_key) for item in value]
    if not isinstance(value, dict):
        return deepcopy(value)
    result = {}
    for key, item in value.items():
        if key in {
            "preallocated_sound_asset_ids_by_actor",
            "selected_sound_asset_ids_by_actor",
        } and isinstance(item, dict):
            result[key] = {
                actor_id: [_rebind_sound_id(sound_id, context) for sound_id in sound_ids]
                if isinstance(sound_ids, list)
                else rebind_audio_payload(sound_ids, context, _parent_key=actor_id)
                for actor_id, sound_ids in item.items()
            }
        elif key == "prepared_audio_id" and isinstance(item, str):
            result[key] = _rebind_sound_id(item, context)
        else:
            result[key] = rebind_audio_payload(item, context, _parent_key=key)
    return _rebind_audio_record(result, context)


def rebind_request(request, context):
    rebound = rebind_audio_payload(request, context)
    rebound["sound_pool"] = context["target_pool_path"]
    return rebound


def _resolve_source_registry(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY / path
    return path.resolve()


def _effective_source_registry(request):
    value = request.get("source_registry")
    if value:
        path = _resolve_source_registry(value)
        if not path.is_file():
            raise FileNotFoundError(f"explicit source registry is unavailable: {path}")
        return str(path)
    return str((REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json").resolve())


def validate_replay_request_compatibility(source_request, requested_request, source_plan, catalog_path):
    """Reject request changes that would require a new visual capture or plan."""
    immutable = (
        "schema", "room_id", "source_asset_ids", "frame_count", "frame_rate_hz",
        "sample_rate_hz", "camera", "entities", "profile", "sampling_policy",
    )
    for key in immutable:
        if source_request.get(key) != requested_request.get(key):
            raise ValueError(f"replay request changes {key}; recapture is required")
    if requested_request.get("source_registry"):
        _effective_source_registry(requested_request)
    source_catalog = source_request.get("room_catalog")
    if source_catalog and Path(source_catalog).expanduser().resolve() != Path(catalog_path).resolve():
        raise ValueError("replay request changes room_catalog; recapture is required")
    clock = source_plan.get("clock") if isinstance(source_plan, dict) else None
    if isinstance(clock, dict):
        for key, default in (("frame_count", 240), ("frame_rate_hz", 15), ("sample_rate_hz", 16000)):
            if float(clock.get(key, default)) != float(source_request.get(key, default)):
                raise ValueError(f"captured plan clock differs at {key}; recapture is required")


def _assert_replay_plan_visual_identity(source_plan, replay_plan):
    for key in (
        "clock", "scene", "condition_profile", "planned_conditions", "activity_plan",
        "camera_condition_sampling", "visual_plan", "room_capabilities",
    ):
        if source_plan.get(key) != replay_plan.get(key):
            raise ValueError(f"replay changed captured visual/condition field {key}")


def rebind_replay_plan(source_plan, request, audio_rebind=None):
    """Copy a retained plan and update only its current audio-facing fields."""
    replay_plan = deepcopy(source_plan)
    replay_plan["request"] = deepcopy(request)
    if audio_rebind is not None:
        for key in ("audio_events", "voice_bindings"):
            if key in source_plan:
                replay_plan[key] = rebind_audio_payload(
                    source_plan[key], audio_rebind
                )
    return replay_plan


def rebind_manifest_entry(entry, audio_rebind):
    """Rebind current request/audio assignments while preserving history fields."""
    rebound = deepcopy(entry)
    if isinstance(rebound.get("request"), dict):
        rebound["request"] = rebind_request(rebound["request"], audio_rebind)
    assignments = rebound.get("source_assignments")
    if isinstance(assignments, list):
        rebound["source_assignments"] = [
            rebind_audio_payload(assignment, audio_rebind)
            if isinstance(assignment, dict) else assignment
            for assignment in assignments
        ]
    for key in ("audio_events", "voice_bindings"):
        if key in rebound:
            rebound[key] = rebind_audio_payload(rebound[key], audio_rebind)
    return rebound


def _report_audio_event_records(report):
    events = report.get("events")
    if not isinstance(events, list):
        raise ValueError("retained audio report lacks rendered event inputs")
    return events


def _report_audio_path(event, report):
    path = event.get("path")
    if path:
        return str(Path(path).expanduser().resolve())
    sound_id = event.get("sound_asset_id") or event.get("prepared_audio_id")
    dry_assets = (report.get("inputs") or {}).get("dry_assets")
    if isinstance(dry_assets, dict) and sound_id in dry_assets:
        value = dry_assets[sound_id]
        if isinstance(value, dict) and value.get("path"):
            return str(Path(value["path"]).expanduser().resolve())
    return None


def validate_reusable_audio_inputs(audio_events, report, gain):
    """Require rendered audio to use the same IDs, files, and event gains."""
    rendered = _report_audio_event_records(report)
    current = [
        event for event in audio_events
        if isinstance(event, dict) and event.get("sound_asset_id")
    ]
    if len(current) != len(rendered):
        raise ValueError(
            "retained audio event count differs from requested audio inputs"
        )
    rendered_by_key = {}
    for event in rendered:
        key = (
            event.get("event_id"),
            event.get("voice_binding_actor_id") or event.get("actor_id"),
        )
        if key in rendered_by_key:
            raise ValueError(f"retained audio report repeats event key {key!r}")
        rendered_by_key[key] = event
    for event in current:
        key = (event.get("event_id"), event.get("actor_id"))
        recorded = rendered_by_key.get(key)
        if recorded is None:
            raise ValueError(f"retained audio lacks event input {key!r}")
        requested_id = event.get("sound_asset_id")
        recorded_id = recorded.get("sound_asset_id") or recorded.get("prepared_audio_id")
        if requested_id != recorded_id:
            raise ValueError(
                f"retained audio sound ID differs for {key!r}: "
                f"{recorded_id!r} != {requested_id!r}"
            )
        requested_path = event.get("path")
        recorded_path = _report_audio_path(recorded, report)
        if requested_path and recorded_path:
            requested_path = str(Path(requested_path).expanduser().resolve())
            if requested_path != recorded_path:
                raise ValueError(
                    f"retained audio path differs for {key!r}: "
                    f"{recorded_path!r} != {requested_path!r}"
                )
        elif requested_path or recorded_path:
            raise ValueError(f"retained audio path is missing for {key!r}")
        requested_gain = event.get("linear_gain")
        recorded_gain = recorded.get("linear_gain")
        if recorded_gain is None:
            recorded_gain = (recorded.get("gain_application") or {}).get("linear_gain")
        if requested_gain != recorded_gain:
            raise ValueError(
                f"retained audio event gain differs for {key!r}: "
                f"{recorded_gain!r} != {requested_gain!r}"
            )
    recorded_gain = report.get("gain_application", {}).get(
        "post_assembly_convolution_gain"
    )
    if recorded_gain != gain:
        raise ValueError(
            f"retained audio gain {recorded_gain!r} differs from declared gain {gain!r}"
        )


def prepare_attempt(row, entry, output, attempt, gain, catalog_path, producer, *, reuse_audio=False, audio_rebind=None):
    from avengine.rooms.room_package import write_room_package_plan_snapshot
    source = source_episode(row)
    source_plan = read_json(source / "plan/episode_plan.json")
    source_request_path = source / "request.json"
    source_request = read_json(source_request_path) if source_request_path.is_file() else deepcopy(entry["request"])
    requested_request = deepcopy(entry["request"])
    validate_replay_request_compatibility(
        source_request, requested_request, source_plan, catalog_path
    )
    request = (
        rebind_request(requested_request, audio_rebind)
        if audio_rebind is not None
        else requested_request
    )
    retained_report = None
    if reuse_audio:
        refs = read_json(source / "delivery/input_refs.json")
        retained_report = Path(refs["audio_report"]).resolve()
        report = read_json(retained_report)
        source_audio_events = deepcopy(source_plan.get("audio_events", []))
        if audio_rebind is not None:
            source_audio_events = rebind_audio_payload(
                source_audio_events, audio_rebind
            )
        validate_reusable_audio_inputs(source_audio_events, report, gain)
    target = Path(output) / "episodes" / row["episode_id"] / attempt / "episode"
    target.mkdir(parents=True, exist_ok=False)
    # Finalization only reads capture. Keep the actual pixel/readback identity intact.
    (target / "capture").symlink_to((source / "capture").resolve(), target_is_directory=True)
    shutil.copytree(source / "plan", target / "plan", symlinks=False)
    catalog = read_json(catalog_path)
    request["room_catalog"] = str(Path(catalog_path).resolve())
    request["source_registry"] = _effective_source_registry(request)
    request["post_assembly_convolution_gain"] = gain
    if row.get("rir_cache"):
        cache = Path(row["rir_cache"]).resolve()
        if not cache.is_dir():
            raise FileNotFoundError(f"selected RIR cache is unavailable: {cache}")
        request["rir_cache"] = str(cache)
    runtime = request.setdefault("runtime", {})
    runtime["path_bindings"] = {**catalog.get("path_bindings", {}), **runtime.get("path_bindings", {})}

    replay_plan = read_json(target / "plan/episode_plan.json")
    replay_plan = rebind_replay_plan(
        replay_plan, request, audio_rebind
    )
    if audio_rebind is not None:
        for relative in ("plan/audio_events.json", "plan/voice_bindings.json"):
            source_payload = read_json(source / relative)
            write_json(target / relative, rebind_audio_payload(source_payload, audio_rebind))
        replay_plan["audio_events"] = read_json(target / "plan/audio_events.json")
        replay_plan["voice_bindings"] = read_json(target / "plan/voice_bindings.json")
    _assert_replay_plan_visual_identity(source_plan, replay_plan)
    write_json(target / "plan/episode_plan.json", replay_plan)
    write_json(target / "request.json", request)
    package = read_json(target / "plan/room_package.json")
    write_room_package_plan_snapshot(target / "plan", package,
        path_bindings=runtime["path_bindings"], catalog_path=Path(catalog_path))
    rebind_info = None
    if audio_rebind is not None:
        changed = sum(
            row.get("old_sound_asset_id") != row.get("new_sound_asset_id")
            for row in audio_rebind["mapping"].values()
        )
        rebind_info = {
            "input_manifest_path": audio_rebind["input_manifest_path"],
            "mapping_path": audio_rebind["mapping_path"],
            "target_pool_path": audio_rebind["target_pool_path"],
            "mapping_count": len(audio_rebind["mapping"]),
            "changed_sound_asset_id": changed,
        }
    write_json(target / "plan/audio_replay.json", {
        "source_episode_root": str(source), "captured_plan": str(source / "plan/episode_plan.json"),
        "post_assembly_convolution_gain": gain, "capture_reused": True,
        "rir_cache": request.get("rir_cache"),
        "audio_reused": reuse_audio,
        "retained_audio_report": str(retained_report) if retained_report else None,
        "audio_rebind": rebind_info,
        "capture_producer": str(source / "producer_version.json") if (source / "producer_version.json").is_file() else None,
        "capture_receipt": str((source / "capture/research_receipt.json").resolve()), "producer": producer,
    })
    write_json(target / "producer_version.json", producer)
    prepared = deepcopy(entry)
    if audio_rebind is not None:
        prepared = rebind_manifest_entry(prepared, audio_rebind)
    prepared["request"] = request
    prepared["request_path"] = str(target / "request.json")
    prepared["historical_request_path"] = entry.get("request_path")
    prepared["controller_entrypoint"] = str(REPOSITORY / "tools/studio/run_qa_episode.py")
    if rebind_info is not None:
        prepared["audio_rebind"] = rebind_info
    if retained_report is not None:
        prepared["audio_report_path"] = str(retained_report)
    return target, prepared


def runner_module():
    path = REPOSITORY / "tools/dataset/run_qa_batch.py"
    spec = importlib.util.spec_from_file_location("qa_audio_replay_runner", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def replay_one(target_string, entry, producer):
    from avengine.qa.batch_delivery import finalize_batch_episode
    target = Path(target_string)
    attempt_root = target.parent
    started = time.monotonic()
    command = [sys.executable, str(REPOSITORY / "tools/studio/run_qa_episode.py"),
        "--resume", "--request", str(target / "request.json"), "--output", str(target),
        "--derived-output", str(target / "delivery")]
    if entry.get("audio_report_path"):
        command += ["--audio-report", entry["audio_report_path"]]
    env = dict(os.environ, PYTHONPATH=f"{REPOSITORY / 'src'}:{REPOSITORY / 'tmp/native_python_addons_v1'}",
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    rec = {"episode_id": entry["episode_id"], "attempt": attempt_root.name,
        "attempt_root": str(attempt_root), "episode_output_root": str(target),
        "request_path": str(target / "request.json"), "command": command, "producer": producer,
        "stdout_log": str(attempt_root / "stdout.log"), "stderr_log": str(attempt_root / "stderr.log"),
        "room_id": entry["room_id"], "room_family": entry.get("room_family"),
        "asset_ids": [a["asset_id"] for a in entry["source_assignments"]]}
    with (attempt_root / "stdout.log").open("x") as out, (attempt_root / "stderr.log").open("x") as err:
        process = subprocess.run(command, cwd=REPOSITORY, env=env, stdout=out, stderr=err)
    rec["controller_returncode"] = process.returncode
    if process.returncode:
        rec.update(status="failed", **runner_module().classify_controller_failure(
            episode_output_root=target, stderr_path=attempt_root / "stderr.log",
            stdout_path=attempt_root / "stdout.log", returncode=process.returncode))
    else:
        try:
            review = finalize_batch_episode(target, entry, repository=REPOSITORY)
            rec["review"] = review
            rec["status"] = "delivered" if review["status"] == "delivered" else "failed"
            rec["facts_path"] = review["facts_path"]
            rec["questions_path"] = review["questions_path"]
            if rec["status"] != "delivered":
                rec.update(failure_stage="finalize", reason_code="review_failed",
                           failure_reason=f"review status: {review['status']}")
                rec["gap_state"] = runner_module().gap_state_for_failure(
                    failure_stage="finalize", reason=rec["failure_reason"], reason_code="review_failed")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            write_json(attempt_root / "review_failure.json", {"error": error, "traceback": traceback.format_exc()})
            rec.update(status="failed", failure_stage="finalize", reason_code="review_exception",
                       failure_reason=error,
                       gap_state=runner_module().gap_state_for_failure(failure_stage="finalize", reason=error))
    rec["duration_seconds"] = time.monotonic() - started
    rec["outcome_path"] = str(attempt_root / "outcome.json")
    write_json(attempt_root / "outcome.json", rec)
    return rec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=REPOSITORY / "examples/rooms/packages/catalog.json")
    parser.add_argument("--attempt", default="attempt_04")
    parser.add_argument("--convolution-gain", type=float, required=True)
    parser.add_argument(
        "--audio-inputs", "--audio-input-manifest", dest="audio_inputs", type=Path,
        default=None, help="Fresh audio pool and old-to-new sound-ID map manifest")
    parser.add_argument(
        "--prepare-only", action="store_true",
        help="Prepare fresh audio-replay requests and plans without running audio")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--reuse-audio", action="store_true",
                        help="reuse matching retained gain audio and rebuild facts/questions/reviews only")
    parser.add_argument("--episode-id", action="append", help="Optional explicit subset for a bounded verification")
    args = parser.parse_args()
    if not math.isfinite(args.convolution_gain) or args.convolution_gain <= 0:
        parser.error("convolution gain must be positive and finite")
    if Path(args.attempt).name != args.attempt or args.attempt in {".", ".."}:
        parser.error("attempt must be a single directory name")
    if args.max_parallel < 1:
        parser.error("max-parallel must be positive")
    output = args.output.resolve()
    audio_rebind = load_audio_rebind_inputs(args.audio_inputs) if args.audio_inputs else None
    audio_rebind_summary = None
    if audio_rebind is not None:
        audio_rebind_summary = {
            "input_manifest_path": audio_rebind["input_manifest_path"],
            "mapping_path": audio_rebind["mapping_path"],
            "target_pool_path": audio_rebind["target_pool_path"],
            "mapping_count": len(audio_rebind["mapping"]),
            "changed_sound_asset_id": sum(
                row.get("old_sound_asset_id") != row.get("new_sound_asset_id")
                for row in audio_rebind["mapping"].values()
            ),
        }
    output.mkdir(parents=True, exist_ok=False)
    producer = code_producer()
    manifest = read_json(args.manifest)
    entries = {e["episode_id"]: e for e in manifest["episodes"]}
    rows = [r for r in read_json(args.source_index)["episodes"]
            if r.get("status") == "delivered" or r.get("capture_reusable") is True]
    if args.episode_id:
        requested = set(args.episode_id)
        rows = [r for r in rows if r["episode_id"] in requested]
        if {r["episode_id"] for r in rows} != requested:
            raise ValueError("some requested Episodes have no reusable native source")
    if not rows:
        raise ValueError("no reusable native Episodes were selected for replay")
    prepared_entries, jobs = {}, []
    for row in rows:
        target, entry = prepare_attempt(row, entries[row["episode_id"]], output,
            args.attempt, args.convolution_gain, args.catalog.resolve(), producer,
            reuse_audio=args.reuse_audio, audio_rebind=audio_rebind)
        prepared_entries[row["episode_id"]] = entry
        jobs.append((str(target), entry, producer))
    final_manifest = deepcopy(manifest)
    final_entries = []
    catalog = read_json(args.catalog.resolve())
    for source_entry in manifest["episodes"]:
        eid = source_entry["episode_id"]
        entry = prepared_entries.get(eid)
        if entry is None:
            entry = deepcopy(source_entry)
            if audio_rebind is not None:
                entry = rebind_manifest_entry(entry, audio_rebind)
            request = deepcopy(entry["request"])
            request["room_catalog"] = str(args.catalog.resolve())
            request["source_registry"] = _effective_source_registry(request)
            request["post_assembly_convolution_gain"] = args.convolution_gain
            runtime = request.setdefault("runtime", {})
            runtime["path_bindings"] = {**catalog.get("path_bindings", {}), **runtime.get("path_bindings", {})}
            request_path = output / "requests" / (eid + ".json")
            write_json(request_path, request)
            entry.update(request=request, request_path=str(request_path),
                         historical_request_path=source_entry.get("request_path"),
                         controller_entrypoint=str(REPOSITORY / "tools/studio/run_qa_episode.py"))
        final_entries.append(entry)
    final_manifest["episodes"] = final_entries
    final_manifest["producer"] = producer
    final_manifest["audio_replay_source_index"] = str(args.source_index.resolve())
    final_manifest["post_assembly_convolution_gain"] = args.convolution_gain
    final_manifest["audio_rebind_input_manifest"] = (
        audio_rebind["input_manifest_path"] if audio_rebind is not None else None
    )
    final_manifest["audio_rebind_mapping"] = (
        audio_rebind["mapping_path"] if audio_rebind is not None else None
    )
    final_manifest["audio_rebind_target_pool"] = (
        audio_rebind["target_pool_path"] if audio_rebind is not None else None
    )
    write_json(output / "manifest.json", final_manifest)
    if args.prepare_only:
        records = {}
        for row in rows:
            entry = prepared_entries[row["episode_id"]]
            attempt_root = Path(entry["request_path"]).parent
            records[row["episode_id"]] = {
                "episode_id": row["episode_id"],
                "attempt": args.attempt,
                "attempt_root": str(attempt_root),
                "episode_output_root": str(attempt_root / "episode"),
                "request_path": entry["request_path"],
                "status": "prepared",
                "capture_reused": True,
                "audio_replay": "not_run",
                "producer": producer,
            }
        snapshot = {
            "producer": producer,
            "manifest_path": str(output / "manifest.json"),
            "episodes": [records[row["episode_id"]] for row in rows],
            "complete": True,
            "selected_episode_count": len(rows),
            "prepare_only": True,
            "audio_rebind": audio_rebind_summary,
        }
        write_json(output / "outcomes.json", snapshot)
        write_json(output / "prepare_only.json", {
            "schema": "avengine_qa_audio_replay_prepare_v1",
            "status": "prepared",
            "native_audio_execution": "not_run",
            "selected_episode_count": len(rows),
            "audio_rebind": audio_rebind_summary,
            "manifest_path": str(output / "manifest.json"),
            "outcomes_path": str(output / "outcomes.json"),
        })
        print(json.dumps({
            "status": "prepared", "native_audio_execution": "not_run",
            "selected_episode_count": len(rows), "output": str(output),
        }))
        return 0
    records = {}
    with ProcessPoolExecutor(max_workers=args.max_parallel) as pool:
        futures = {pool.submit(replay_one, *job): job[1]["episode_id"] for job in jobs}
        for future in as_completed(futures):
            eid = futures[future]
            try:
                records[eid] = future.result()
            except Exception as exc:
                failure_root = output / "episodes" / eid / args.attempt
                failure_details = failure_root / "worker_failure.json"
                write_json(failure_details, {"error": f"{type(exc).__name__}: {exc}",
                                            "traceback": traceback.format_exc()})
                records[eid] = {"episode_id": eid, "status": "failed", "failure_stage": "launch",
                    "attempt_root": str(failure_root), "episode_output_root": str(failure_root / "episode"),
                    "reason_code": "replay_worker_exception", "failure_reason": f"{type(exc).__name__}: {exc}",
                    "gap_state": "evidence_missing_or_unsampled", "classification_unknown": True,
                    "classification_status": "unclassified", "failure_details": str(failure_details)}
            snapshot = {"producer": producer, "manifest_path": str(output / "manifest.json"),
                "episodes": [records[r["episode_id"]] for r in rows if r["episode_id"] in records],
                "complete": len(records) == len(rows), "selected_episode_count": len(rows)}
            write_json(output / "outcomes.json", snapshot)
            print(json.dumps({"episode_id": eid, "status": records[eid]["status"],
                "completed": len(records), "total": len(rows)}), flush=True)
    return int(any(r["status"] != "delivered" for r in records.values()))


if __name__ == "__main__":
    raise SystemExit(main())
