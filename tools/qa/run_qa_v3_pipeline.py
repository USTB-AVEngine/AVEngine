#!/usr/bin/env python3
"""Run one declared QA-v3 request through design, media and verification.

This is an orchestration entry point. It delegates design to the room/profile
scheduler, capture and audio to their existing batch runners, media to the
current review clip builder, and question publication to the released-item
adapter. It records each stage independently, including design quota
shortfalls and unavailable runtime resources. It never reads model scores or
uses evaluation results to choose a profile.

A normal run needs a runtime configuration with explicit capture and audio
resources. The resume-only mode is useful for checking retained design roots
and assembly state without starting UE, RLR or any media command. Missing later
artifacts remain pending and cannot make the whole pipeline complete.

Request fields are scene_configs, profiles, params, optional requested_profiles,
question_budget or cells_per_pair, answer_forms and audio_variants. A request
may name an existing_design_root for a read-only continuation. Runtime fields
are python, capture, audio, media, scene_profiles and finite stage timeouts;
scene_profiles can override resources for a particular scene/profile pair.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))

from build_batch_review_page import _clip_matches_receipt, receipt_summary
from qa_v3_request import (
    batch_point_ids,
    normalize_answer_forms,
    plan_room_questions,
    read_qa_params,
)
from run_qa_v3_audio_batch import (
    normalize_layouts as normalize_audio_layouts,
    point_state as audio_point_state,
)
from run_qa_v3_capture_batch import (
    point_state as capture_point_state,
    resolve_capture_inputs,
)


SCHEDULER = HERE / "run_qa_v3_room_profile_scheduler.py"
ASSEMBLER = HERE / "assemble_qa_v3_room_pilot.py"
CAPTURE_BATCH = HERE / "run_qa_v3_capture_batch.py"
AUDIO_BATCH = HERE / "run_qa_v3_audio_batch.py"
VISUAL_VERIFY = HERE / "verify_qa_v3_visual_batch.py"
AUDIO_VERIFY = HERE / "verify_qa_v3_audio_batch.py"
RELEASED_ITEMS = HERE / "build_qa_v3_released_probe_items.py"
REVIEW_CLIP = REPO / "tools/review/build_current_mp3d_dynamic_review_clip.py"

DEFAULT_TIMEOUTS = {
    "design": 3600.0,
    "capture": 7200.0,
    "audio": 7200.0,
    "media": 1800.0,
    "verification": 900.0,
}

STAGE_ORDER = (
    "design",
    "assemble",
    "capture",
    "audio",
    "media",
    "verify",
    "questions",
)


class PipelineError(ValueError):
    """A declared pipeline request or runtime input cannot be used safely."""


def _source_state() -> dict[str, Any]:
    """Record the Git and interpreter source used by this invocation.

    This is provenance only.  A dirty documentation file does not block a
    run, and this record is not used as a hash gate.
    """

    def git(*args: str, required: bool = True) -> str | None:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(REPO),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            if required:
                raise PipelineError(
                    f"cannot inspect AVEngine Git source: "
                    f"{completed.stderr.strip() or args}"
                )
            return None
        return completed.stdout.strip()

    repository = git("rev-parse", "--show-toplevel")
    commit = git("rev-parse", "HEAD")
    branch = git("symbolic-ref", "--quiet", "--short", "HEAD", required=False)
    status = git("status", "--short", "--untracked-files=no") or ""
    return {
        "repository": str(Path(repository).resolve()),
        "entrypoint": str(Path(__file__).resolve()),
        "git_commit": commit,
        "git_branch": branch,
        "tracked_worktree_changes": status.splitlines(),
        "python_executable": str(Path(sys.executable).resolve()),
    }


def _normalize_through_stage(value: Any) -> str:
    if value is None:
        return "questions"
    if not isinstance(value, str) or value not in STAGE_ORDER:
        raise PipelineError(
            f"through_stage must be one of {list(STAGE_ORDER)}, got {value!r}"
        )
    return value


def _stage_enabled(through_stage: str, stage: str) -> bool:
    return STAGE_ORDER.index(stage) <= STAGE_ORDER.index(through_stage)


def _deferred_stage(stage: str, root: Path, through_stage: str) -> dict[str, Any]:
    return {
        "status": "pending",
        "root": str(root.resolve()),
        "run": {
            "status": "not_run",
            "through_stage": through_stage,
        },
        "detail": (
            f"through-stage={through_stage} stopped before the {stage} stage"
        ),
    }


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PipelineError(f"cannot read JSON {path}: {exc}") from exc


def _write_json(path: Path, value: Any, *, replace: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise PipelineError(f"refusing to overwrite pipeline artifact: {path}")
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise PipelineError(f"temporary pipeline artifact already exists: {temporary}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolve_path(value: Any, *, base: Path, owner: str, must_exist: bool = True) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise PipelineError(f"{owner} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise PipelineError(f"{owner} is missing: {path}")
    return path


def _safe_component(value: Any, *, owner: str) -> str:
    text = str(value)
    if (
        not text
        or text in {".", ".."}
        or "/" in text
        or "\\" in text
        or text != Path(text).name
    ):
        raise PipelineError(f"{owner} is not a single path component: {text!r}")
    return text


def _positive_int(value: Any, *, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PipelineError(f"{owner} must be a positive integer")
    return int(value)


def _finite_positive(value: Any, *, owner: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PipelineError(f"{owner} must be finite and positive") from exc
    if not result > 0.0 or result != result or result in {float("inf"), float("-inf")}:
        raise PipelineError(f"{owner} must be finite and positive")
    return result


def _mapping(value: Any, *, owner: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise PipelineError(f"{owner} must be an object")
    return dict(value)


def _resolve_scene_entries(request: Mapping[str, Any], base: Path) -> list[Path]:
    raw = request.get("scene_configs", request.get("scenes"))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise PipelineError("request.scene_configs must be a non-empty list")
    result = []
    seen = set()
    for index, item in enumerate(raw):
        value = item.get("path", item.get("config")) if isinstance(item, Mapping) else item
        path = _resolve_path(value, base=base, owner=f"scene_configs[{index}]")
        if path in seen:
            raise PipelineError(f"duplicate scene config: {path}")
        seen.add(path)
        result.append(path)
    return result


def _load_request(path: Path) -> dict[str, Any]:
    raw = _read_json(path)
    if not isinstance(raw, Mapping):
        raise PipelineError("request must be a JSON object")
    base = path.parent
    scenes = _resolve_scene_entries(raw, base)
    profiles = _resolve_path(raw.get("profiles"), base=base, owner="request.profiles")
    params = _resolve_path(raw.get("params"), base=base, owner="request.params")
    profile_params_raw = raw.get("profile_params", raw.get("params_by_profile", {}))
    if profile_params_raw is None:
        profile_params_raw = {}
    if not isinstance(profile_params_raw, Mapping):
        raise PipelineError("request.profile_params must be an object")
    profile_params = {}
    for profile_id, spec in profile_params_raw.items():
        profile_id = _safe_component(profile_id, owner="request.profile_params")
        if isinstance(spec, (str, Path)):
            source = _resolve_path(
                spec, base=base, owner=f"request.profile_params.{profile_id}")
            overrides = {}
        elif isinstance(spec, Mapping):
            source_value = spec.get("params", params)
            source = _resolve_path(
                source_value, base=base,
                owner=f"request.profile_params.{profile_id}.params")
            overrides = spec.get("overrides", {})
            if not isinstance(overrides, Mapping):
                raise PipelineError(
                    f"request.profile_params.{profile_id}.overrides must be an object")
            overrides = dict(overrides)
        else:
            raise PipelineError(
                f"request.profile_params.{profile_id} must be a path or object")
        profile_params[profile_id] = {
            "params": source,
            "overrides": overrides,
        }
    requested = raw.get("requested_profiles")
    if requested is not None:
        if not isinstance(requested, Sequence) or isinstance(requested, (str, bytes)):
            raise PipelineError("requested_profiles must be a list")
        requested = [_safe_component(value, owner="requested_profiles") for value in requested]
        if len(requested) != len(set(requested)):
            raise PipelineError("requested_profiles must be unique")
    answer_forms = raw.get("answer_forms")
    if answer_forms is not None:
        answer_forms = normalize_answer_forms(answer_forms)
    for key in ("question_budget", "cells_per_pair"):
        if key in raw and raw[key] is not None:
            _positive_int(raw[key], owner=f"request.{key}")
    if raw.get("question_budget") is not None and raw.get("cells_per_pair") is not None:
        raise PipelineError(
            "request.question_budget and request.cells_per_pair are mutually exclusive"
        )
    seed = raw.get("seed", "qa-v3-pipeline")
    if not isinstance(seed, str) or not seed.strip():
        raise PipelineError("request.seed must be a non-empty string")
    variants = raw.get("audio_variants", ["main", "gateA"])
    if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes)) or not variants:
        raise PipelineError("request.audio_variants must be a non-empty list")
    variants = [_safe_component(value, owner="audio_variants") for value in variants]
    if "main" not in variants:
        raise PipelineError("request.audio_variants must include main")
    if len(variants) != len(set(variants)):
        raise PipelineError("request.audio_variants must be unique")
    snapshot = raw.get("snapshot_content")
    if snapshot is not None:
        snapshot = str(_resolve_path(
            snapshot, base=base, owner="request.snapshot_content"))
    existing_design = raw.get("existing_design_root")
    if existing_design is not None:
        existing_design = _resolve_path(
            existing_design, base=base, owner="request.existing_design_root")
    weights = raw.get("profile_weights")
    if isinstance(weights, (str, Path)):
        weights = _resolve_path(weights, base=base, owner="request.profile_weights")
    elif weights is not None and not isinstance(weights, Mapping):
        raise PipelineError("request.profile_weights must be a path or object")
    return {
        "raw": dict(raw),
        "path": path.resolve(),
        "scene_configs": scenes,
        "profiles": profiles,
        "params": params,
        "profile_params": profile_params,
        "requested_profiles": requested,
        "answer_forms": answer_forms,
        "question_budget": raw.get("question_budget"),
        "cells_per_pair": raw.get("cells_per_pair"),
        "profile_weights": weights,
        "seed": seed.strip(),
        "audio_variants": variants,
        "snapshot_content": snapshot,
        "existing_design_root": existing_design,
    }


def _resolve_config_paths(
    config: Mapping[str, Any],
    *,
    base: Path,
    path_fields: Sequence[str],
    required: bool,
    owner: str,
) -> dict[str, Any]:
    result = dict(config)
    for field in path_fields:
        if field not in result or result[field] is None:
            if required:
                raise PipelineError(f"{owner}.{field} is required")
            continue
        result[field] = str(_resolve_path(
            result[field], base=base, owner=f"{owner}.{field}").resolve())
    return result


def _load_runtime(path: Path, *, resume_only: bool) -> dict[str, Any]:
    raw = _read_json(path)
    if not isinstance(raw, Mapping):
        raise PipelineError("runtime config must be a JSON object")
    base = path.parent
    top = dict(raw)
    timeouts = top.get("timeouts") or {}
    if not isinstance(timeouts, Mapping):
        raise PipelineError("runtime.timeouts must be an object")
    for stage, value in timeouts.items():
        if stage not in DEFAULT_TIMEOUTS:
            raise PipelineError(f"runtime.timeouts has unknown stage {stage!r}")
        _finite_positive(value, owner=f"runtime.timeouts.{stage}")
    python_value = top.get("python")
    if python_value is not None:
        python_path = _resolve_path(
            python_value, base=base, owner="runtime.python", must_exist=not resume_only)
        top["python"] = str(python_path)
    elif not resume_only:
        raise PipelineError("runtime.python is required")
    capture = _resolve_config_paths(
        _mapping(top.get("capture"), owner="runtime.capture"),
        base=base,
        path_fields=("python", "spear_ext", "closure_report", "stage_root",
                     "spear_executable", "source_asset_registry",
                     "capture_warmup_config"),
        required=False,
        owner="runtime.capture",
    )
    audio = _resolve_config_paths(
        _mapping(top.get("audio"), owner="runtime.audio"),
        base=base,
        path_fields=("python", "repo", "m1_request", "simulation_request",
                     "package_manifest", "source_endpoint_registry",
                     "sound_asset_registry", "hrtf", "runtime_prefix",
                     "rlr_sdk_root", "magnum_python_site", "source_asset_registry",
                     "sound_asset_map"),
        required=False,
        owner="runtime.audio",
    )
    media = _mapping(top.get("media"), owner="runtime.media")
    scene_profiles = _mapping(top.get("scene_profiles"), owner="runtime.scene_profiles")
    top["capture"] = capture
    top["audio"] = audio
    top["media"] = media
    top["scene_profiles"] = scene_profiles
    top["_path"] = path.resolve()
    return top


def _stage_config(runtime: Mapping[str, Any], section: str,
                  scene_id: str, profile_id: str) -> dict[str, Any]:
    base = _mapping(runtime.get(section), owner=f"runtime.{section}")
    by_scene = runtime.get("scene_profiles") or {}
    override = {}
    if isinstance(by_scene, Mapping):
        scene = by_scene.get(scene_id, {})
        if isinstance(scene, Mapping):
            override = scene.get(profile_id, {})
            if isinstance(override, Mapping):
                override = override.get(section, override)
    result = dict(base)
    result.update(_mapping(
        override,
        owner=f"runtime.scene_profiles.{scene_id}.{profile_id}.{section}"))
    path_fields = {
        "capture": (
            "python", "spear_ext", "closure_report", "stage_root",
            "spear_executable", "source_asset_registry",
            "capture_warmup_config",
        ),
        "audio": (
            "python", "repo", "m1_request", "simulation_request",
            "package_manifest", "source_endpoint_registry",
            "sound_asset_registry", "hrtf", "runtime_prefix",
            "rlr_sdk_root", "magnum_python_site", "source_asset_registry",
            "sound_asset_map",
        ),
        "media": ("python",),
    }.get(section, ())
    if path_fields:
        result = _resolve_config_paths(
            result,
            base=Path(runtime.get("_path", REPO)).parent,
            path_fields=path_fields,
            required=False,
            owner=f"runtime.{section}.{scene_id}.{profile_id}",
        )
    return result


def _python_for_runtime(runtime: Mapping[str, Any]) -> str:
    return str(runtime.get("python") or sys.executable)


def _timeout(runtime: Mapping[str, Any], stage: str) -> float:
    values = runtime.get("timeouts") or {}
    value = values.get(stage, DEFAULT_TIMEOUTS[stage]) if isinstance(values, Mapping) else DEFAULT_TIMEOUTS[stage]
    return _finite_positive(value, owner=f"runtime.timeouts.{stage}")


def _run_logged(label: str, command: Sequence[str], log_path: Path,
                *, timeout: float, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    if log_path.exists():
        raise PipelineError(f"refusing to overwrite stage log: {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    launch_env = dict(os.environ)
    if env:
        launch_env.update({str(key): str(value) for key, value in env.items()})
    existing_pythonpath = launch_env.get("PYTHONPATH")
    launch_env["PYTHONPATH"] = (
        f"{REPO / 'src'}:{existing_pythonpath}"
        if existing_pythonpath else str(REPO / "src")
    )
    started = datetime.now(timezone.utc).isoformat()
    with log_path.open("x", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(
                [str(value) for value in command],
                cwd=str(REPO),
                env=launch_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            return {
                "status": "failed",
                "label": label,
                "error": f"{type(exc).__name__}: {exc}",
                "log": str(log_path.resolve()),
            }
        pid = process.pid
        timed_out = False
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            code = -signal.SIGTERM
    return {
        "status": "failed" if code != 0 else "complete",
        "label": label,
        "pid": pid,
        "exit_code": code,
        "timed_out": timed_out,
        "timeout_seconds": timeout,
        "started_at": started,
        "log": str(log_path.resolve()),
    }


def _snapshot(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    if path.exists():
        old = _read_json(path)
        if old != dict(value):
            raise PipelineError(f"pipeline snapshot differs from current configuration: {path}")
        return {"path": str(path.resolve()), "status": "reused"}
    _write_json(path, dict(value))
    return {"path": str(path.resolve()), "status": "written"}


def _snapshot_request(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    """Compare only stable request declarations across resume runs.

    Stage and execution mode are per-run controls recorded in the pipeline
    manifest, so changing either must not look like a request mutation.
    """
    if path.exists():
        old = _read_json(path)
        if not isinstance(old, Mapping):
            raise PipelineError(f"request snapshot is not an object: {path}")
        stable_old = dict(old)
        for key in ("through_stage", "mode", "run_control"):
            stable_old.pop(key, None)
        if stable_old != dict(value):
            raise PipelineError(
                f"pipeline request snapshot differs from current declaration: {path}"
            )
        return {"path": str(path.resolve()), "status": "reused"}
    _write_json(path, dict(value))
    return {"path": str(path.resolve()), "status": "written"}


def _validate_resume_stage(out_root: Path, through_stage: str) -> None:
    manifest_path = out_root / "pipeline_manifest.json"
    if not manifest_path.is_file():
        return
    old = _read_json(manifest_path)
    if not isinstance(old, Mapping):
        raise PipelineError(f"existing pipeline manifest is not an object: {manifest_path}")
    previous = old.get("through_stage")
    if previous is None:
        control = old.get("run_control")
        if isinstance(control, Mapping):
            previous = control.get("through_stage")
    if previous is None:
        return
    if not isinstance(previous, str) or previous not in STAGE_ORDER:
        raise PipelineError(
            f"existing pipeline manifest has invalid through_stage: {previous!r}"
        )
    if STAGE_ORDER.index(through_stage) < STAGE_ORDER.index(previous):
        raise PipelineError(
            f"cannot resume through-stage={through_stage} after "
            f"through-stage={previous}; stage scope cannot move backward"
        )


def _profile_catalog(path: Path) -> list[dict[str, Any]]:
    value = _read_json(path)
    if isinstance(value, Mapping):
        value = [dict(value)]
    if not isinstance(value, list):
        raise PipelineError("request.profiles must contain a JSON list or profile object")
    result = []
    for index, profile in enumerate(value):
        if not isinstance(profile, Mapping) or not profile.get("id"):
            raise PipelineError(
                f"profile catalog entry {index} needs a non-empty id"
            )
        result.append(dict(profile))
    if not result:
        raise PipelineError("profile catalog is empty")
    ids = [str(profile["id"]) for profile in result]
    if len(ids) != len(set(ids)):
        raise PipelineError("profile catalog ids must be unique")
    return result


def _normalized_profiles_path(request: Mapping[str, Any], out_root: Path) -> Path:
    raw = _read_json(request["profiles"])
    if isinstance(raw, list):
        _profile_catalog(request["profiles"])
        return Path(request["profiles"])
    profiles = _profile_catalog(request["profiles"])
    path = out_root / "request" / "profiles.json"
    if path.exists():
        if _read_json(path) != profiles:
            raise PipelineError(
                f"normalized profile snapshot differs from existing output: {path}"
            )
    else:
        _write_json(path, profiles)
    return path


def _stable_profile_overrides(
    source: Path, overrides: Mapping[str, Any]
) -> dict[str, Any]:
    result = dict(overrides)
    for key in ("SOUND_EVENT_POOL", "SPEECH_SELECTION_POLICY"):
        value = result.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise PipelineError(
                f"profile_params override {key} must be a non-empty path"
            )
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = source.parent / path
        result[key] = str(path.resolve())
    return result


def _stable_request_snapshot(
    request: Mapping[str, Any], params_summary: Mapping[str, Any]
) -> dict[str, Any]:
    scene_paths = [Path(path).resolve() for path in request["scene_configs"]]
    profile_catalog = _profile_catalog(request["profiles"])
    params_path = Path(request["params"]).resolve()
    profile_params = {}
    for profile_id, spec in (request.get("profile_params") or {}).items():
        source = Path(spec["params"]).resolve()
        profile_params[str(profile_id)] = {
            "params": str(source),
            "params_content": read_qa_params(source),
            "overrides": _stable_profile_overrides(
                source, spec.get("overrides") or {}
            ),
        }
    weights = request.get("profile_weights")
    if isinstance(weights, Path):
        weights_snapshot = {
            "path": str(weights.resolve()),
            "content": _read_json(weights),
        }
    elif isinstance(weights, Mapping):
        weights_snapshot = {"value": dict(weights)}
    else:
        weights_snapshot = None
    return {
        "source": str(Path(request["path"]).resolve()),
        "scene_configs": [str(path) for path in scene_paths],
        "scene_config_content": [_read_json(path) for path in scene_paths],
        "profiles": str(Path(request["profiles"]).resolve()),
        "profile_catalog": profile_catalog,
        "params": str(params_path),
        "params_content": read_qa_params(params_path),
        "profile_params": profile_params,
        "requested_profiles": request["requested_profiles"],
        "answer_forms": request["answer_forms"],
        "question_budget": request["question_budget"],
        "cells_per_pair": request["cells_per_pair"],
        "profile_weights": weights_snapshot,
        "snapshot_content": request["snapshot_content"],
        "existing_design_root": (
            str(Path(request["existing_design_root"]).resolve())
            if request["existing_design_root"] is not None else None
        ),
        "audio_variants": request["audio_variants"],
        "seed": request["seed"],
        "params_summary": dict(params_summary),
    }


def _params_summary(params_path: Path) -> dict[str, Any]:
    params = read_qa_params(params_path)
    summary = {}
    for key in (
        "PAIR_KIND", "SOUND_SOURCE_MODE", "SOUND_ASSET", "SOUND_EVENT_POOL",
        "CLIP_SECONDS", "FRAME_COUNT", "VIDEO_FPS", "SAMPLE_RATE_HZ",
        "SAMPLE_COUNT", "ANSWER_FORMS_DEFAULT",
    ):
        if key in params:
            summary[key] = params[key]
    if "PAIR_KIND" not in summary:
        raise PipelineError("params must declare PAIR_KIND")
    if "SOUND_SOURCE_MODE" not in summary:
        raise PipelineError("params must declare SOUND_SOURCE_MODE")
    return summary


def _request_profile_ids(request: Mapping[str, Any]) -> list[str]:
    requested = request.get("requested_profiles")
    if requested is not None:
        return list(requested)
    profiles = _profile_catalog(request["profiles"])
    return [
        _safe_component(profile["id"], owner="profile catalog id")
        for profile in profiles
    ]


def _effective_profile_params(
    request: Mapping[str, Any],
    profile_id: str,
    *,
    common_forms: Sequence[str],
    common_budget: int,
    output_root: Path,
) -> tuple[Path, dict[str, Any]]:
    configured = request.get("profile_params") or {}
    spec = configured.get(profile_id, {
        "params": request["params"],
        "overrides": {},
    })
    source = Path(spec["params"])
    params = read_qa_params(source)
    overrides = dict(spec.get("overrides") or {})
    forbidden = {"ITEMS_PER_ROOM_DEFAULT", "ANSWER_FORMS_DEFAULT"} & set(overrides)
    if forbidden:
        raise PipelineError(
            f"profile_params.{profile_id}.overrides cannot change common "
            f"budget/forms: {sorted(forbidden)}")
    path_fields = {"SOUND_EVENT_POOL", "SPEECH_SELECTION_POLICY"}
    for key in path_fields:
        value = overrides.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise PipelineError(
                f"profile_params.{profile_id}.overrides.{key} must be a path")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = source.parent / path
        overrides[key] = str(path.resolve())
    params.update(overrides)
    params["ANSWER_FORMS_DEFAULT"] = list(common_forms)
    params["ITEMS_PER_ROOM_DEFAULT"] = common_budget
    summary = _params_summary_from_value(params)
    snapshot = output_root / "request" / "profile_params" / (
        _safe_component(profile_id, owner="profile id") + ".json")
    _snapshot(snapshot, params)
    return snapshot, summary


def _params_summary_from_value(params: Mapping[str, Any]) -> dict[str, Any]:
    summary = {}
    for key in (
        "PAIR_KIND", "SOUND_SOURCE_MODE", "SOUND_ASSET", "SOUND_EVENT_POOL",
        "CLIP_SECONDS", "FRAME_COUNT", "VIDEO_FPS", "SAMPLE_RATE_HZ",
        "SAMPLE_COUNT", "ANSWER_FORMS_DEFAULT",
    ):
        if key in params:
            summary[key] = params[key]
    if "PAIR_KIND" not in summary:
        raise PipelineError("params must declare PAIR_KIND")
    if "SOUND_SOURCE_MODE" not in summary:
        raise PipelineError("params must declare SOUND_SOURCE_MODE")
    return summary


def _common_request_plan(request: Mapping[str, Any]) -> tuple[list[str], int, dict[str, int]]:
    profile_ids = _request_profile_ids(request)
    default_params = read_qa_params(request["params"])
    forms = request.get("answer_forms")
    if forms is None:
        forms = normalize_answer_forms(default_params.get(
            "ANSWER_FORMS_DEFAULT", ["mcq", "open"]))
    else:
        forms = list(forms)
    if request.get("question_budget") is not None:
        budget = int(request["question_budget"])
        weights = request.get("profile_weights")
        if isinstance(weights, Path):
            weights = _read_json(weights)
        plan_params = {"ITEMS_PER_ROOM_DEFAULT": budget,
                       "ANSWER_FORMS_DEFAULT": forms}
        plan = plan_room_questions(
            profile_ids, plan_params, question_budget=budget,
            answer_forms=forms, profile_weights=weights)
        return forms, budget, dict(plan["cells"])
    if request.get("cells_per_pair") is not None:
        return forms, int(request["cells_per_pair"]), {
            profile_id: int(request["cells_per_pair"]) for profile_id in profile_ids
        }
    budget = default_params.get("ITEMS_PER_ROOM_DEFAULT")
    if budget is None:
        raise PipelineError(
            "params or request must declare ITEMS_PER_ROOM_DEFAULT/question_budget")
    budget = _positive_int(budget, owner="common question budget")
    weights = request.get("profile_weights")
    if isinstance(weights, Path):
        weights = _read_json(weights)
    plan_params = {"ITEMS_PER_ROOM_DEFAULT": budget,
                   "ANSWER_FORMS_DEFAULT": forms}
    plan = plan_room_questions(
        profile_ids, plan_params, question_budget=budget,
        answer_forms=forms, profile_weights=weights)
    return forms, budget, dict(plan["cells"])


def _validate_design_matrix(
    matrix: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    expected_profiles: Sequence[str] | None = None,
) -> str | None:
    """Check that a retained matrix belongs to this request declaration."""
    scene_rows = matrix.get("scenes")
    if not isinstance(scene_rows, list):
        return "design matrix has no scene declarations"
    matrix_scene_ids = {
        str(row.get("scene_id"))
        for row in scene_rows
        if isinstance(row, Mapping) and row.get("scene_id") is not None
    }
    request_scene_ids = set()
    for path in request["scene_configs"]:
        value = _read_json(path)
        if not isinstance(value, Mapping) or not value.get("scene_id"):
            return f"scene config has no scene_id: {path}"
        request_scene_ids.add(str(value["scene_id"]))
    if matrix_scene_ids != request_scene_ids:
        return (
            f"retained design scenes differ: matrix={sorted(matrix_scene_ids)} "
            f"request={sorted(request_scene_ids)}"
        )
    profile_ids = (
        list(expected_profiles)
        if expected_profiles is not None
        else request.get("requested_profiles")
    )
    if profile_ids is None:
        try:
            profiles = _profile_catalog(request["profiles"])
        except PipelineError as exc:
            return str(exc)
        profile_ids = [str(profile["id"]) for profile in profiles]
    matrix_profiles = matrix.get("requested_profiles")
    if not isinstance(matrix_profiles, list):
        return "design matrix has no requested_profiles declaration"
    if set(map(str, matrix_profiles)) != set(map(str, profile_ids)):
        return (
            f"retained design profiles differ: matrix={sorted(map(str, matrix_profiles))} "
            f"request={sorted(map(str, profile_ids))}"
        )
    requested_forms = request.get("answer_forms")
    matrix_request = matrix.get("question_request")
    matrix_forms = (
        matrix_request.get("answer_forms")
        if isinstance(matrix_request, Mapping) else None
    )
    if requested_forms is not None and matrix_forms is not None:
        try:
            if normalize_answer_forms(matrix_forms) != list(requested_forms):
                return (
                    f"retained design answer forms differ: "
                    f"matrix={matrix_forms} request={requested_forms}"
                )
        except (TypeError, ValueError) as exc:
            return f"retained design answer forms are invalid: {exc}"
    return None


def _stub_profile_matrix(
    scene_paths: Sequence[Path],
    profile_id: str,
    *,
    cells: int,
    answer_forms: Sequence[str],
    params_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    scenes = []
    rows = []
    for scene_path in scene_paths:
        value = _read_json(scene_path)
        scene_id = value.get("scene_id") if isinstance(value, Mapping) else None
        if not scene_id:
            raise PipelineError(f"scene config has no scene_id: {scene_path}")
        scenes.append({"scene_id": str(scene_id)})
        rows.append({
            "scene_id": str(scene_id),
            "profile_id": profile_id,
            "attempt_status": "not_scheduled",
            "requested_cells": cells,
            "geometry_candidates": 0,
            "quota_shortfall": cells,
            "batch_manifest": None,
        })
    return {
        "kind": "qa_v3_pipeline_design_group",
        "status": "completed",
        "qualification_claim": False,
        "inputs": {
            "scene_configs": [str(path.resolve()) for path in scene_paths],
            "params": str(params_path.resolve()),
            "requested_profiles": [profile_id],
        },
        "scenes": scenes,
        "requested_profiles": [profile_id],
        "matrix": rows,
        "question_request": {
            "answer_forms": list(answer_forms),
            "forms_per_candidate": len(answer_forms),
        },
        "counts": {
            "cells_requested": len(rows) * cells,
            "geometry_candidates": 0,
            "rejected": 0,
        },
    }


def _design_one_profile(
    request: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    profile_id: str,
    cells: int,
    forms: Sequence[str],
    params_path: Path,
    params_summary: Mapping[str, Any],
    out_root: Path,
    resume_only: bool,
) -> dict[str, Any]:
    design_root = out_root / "design" / _safe_component(
        profile_id, owner="profile id")
    matrix_path = design_root / "scene_profile_matrix.json"
    if matrix_path.is_file():
        matrix = _read_json(matrix_path)
        mismatch = _validate_design_matrix(
            matrix, request, expected_profiles=[profile_id])
        if mismatch is not None:
            return {
                "status": "failed",
                "profile_id": profile_id,
                "root": str(design_root.resolve()),
                "matrix_path": str(matrix_path.resolve()),
                "detail": mismatch,
                "matrix": matrix,
            }
        return {
            "status": "complete" if matrix.get("status") == "completed" else "partial",
            "profile_id": profile_id,
            "root": str(design_root.resolve()),
            "matrix_path": str(matrix_path.resolve()),
            "matrix": matrix,
            "params_path": str(params_path.resolve()),
            "params_summary": dict(params_summary),
            "run": {"status": "reused_existing"},
        }
    if design_root.exists():
        return {
            "status": "failed",
            "profile_id": profile_id,
            "root": str(design_root.resolve()),
            "detail": "design group exists without scene_profile_matrix.json",
        }
    if cells == 0:
        design_root.mkdir(parents=True)
        matrix = _stub_profile_matrix(
            request["scene_configs"], profile_id, cells=cells,
            answer_forms=forms, params_path=params_path,
            output_root=design_root)
        _write_json(matrix_path, matrix)
        return {
            "status": "complete",
            "profile_id": profile_id,
            "root": str(design_root.resolve()),
            "matrix_path": str(matrix_path.resolve()),
            "matrix": matrix,
            "params_path": str(params_path.resolve()),
            "params_summary": dict(params_summary),
            "run": {"status": "stub_not_scheduled"},
        }
    if resume_only:
        return {
            "status": "pending",
            "profile_id": profile_id,
            "root": str(design_root.resolve()),
            "params_path": str(params_path.resolve()),
            "params_summary": dict(params_summary),
            "detail": "resume-only mode does not launch the scheduler",
        }
    command = [
        runtime["python"], str(SCHEDULER),
        "--profiles", str(request["profiles"]),
        "--params", str(params_path),
        "--seed", request["seed"] + "|" + profile_id,
        "--out-root", str(design_root),
        "--requested-profile", profile_id,
        "--cells-per-pair", str(cells),
    ]
    for scene in request["scene_configs"]:
        command.extend(["--scene-config", str(scene)])
    for form in forms:
        command.extend(["--answer-form", form])
    if request.get("snapshot_content") is not None:
        command.extend(["--snapshot-content", request["snapshot_content"]])
    run = _run_logged(
        f"design:{profile_id}", command,
        out_root / "logs" / f"design_{_safe_component(profile_id, owner='profile id')}.log",
        timeout=_timeout(runtime, "design"))
    if not matrix_path.is_file():
        run["status"] = "failed"
        run["detail"] = "scheduler returned without scene_profile_matrix.json"
        return {
            "status": "failed",
            "profile_id": profile_id,
            "root": str(design_root.resolve()),
            "params_path": str(params_path.resolve()),
            "params_summary": dict(params_summary),
            "run": run,
        }
    matrix = _read_json(matrix_path)
    mismatch = _validate_design_matrix(
        matrix, request, expected_profiles=[profile_id])
    if mismatch is not None:
        return {
            "status": "failed",
            "profile_id": profile_id,
            "root": str(design_root.resolve()),
            "matrix_path": str(matrix_path.resolve()),
            "matrix": matrix,
            "params_path": str(params_path.resolve()),
            "params_summary": dict(params_summary),
            "run": run,
            "detail": mismatch,
        }
    status = (
        "complete"
        if run["status"] == "complete" and matrix.get("status") == "completed"
        else "partial"
    )
    return {
        "status": status,
        "profile_id": profile_id,
        "root": str(design_root.resolve()),
        "matrix_path": str(matrix_path.resolve()),
        "matrix": matrix,
        "params_path": str(params_path.resolve()),
        "params_summary": dict(params_summary),
        "run": run,
    }


def _design_per_profile(
    request: Mapping[str, Any],
    runtime: Mapping[str, Any],
    out_root: Path,
    *,
    resume_only: bool,
) -> dict[str, Any]:
    if request.get("existing_design_root") is not None:
        raise PipelineError(
            "profile_params cannot be combined with existing_design_root")
    profile_ids = _request_profile_ids(request)
    configured = request.get("profile_params") or {}
    unknown = set(configured) - set(profile_ids)
    if unknown:
        raise PipelineError(
            f"request.profile_params names unrequested profiles: {sorted(unknown)}")
    forms, common_budget, cells = _common_request_plan(request)
    groups = []
    for profile_id in profile_ids:
        params_path, summary = _effective_profile_params(
            request, profile_id, common_forms=forms,
            common_budget=common_budget, output_root=out_root)
        groups.append(_design_one_profile(
            request, runtime, profile_id=profile_id, cells=cells[profile_id],
            forms=forms, params_path=params_path, params_summary=summary,
            out_root=out_root, resume_only=resume_only))
    statuses = [group.get("status") for group in groups]
    status = (
        "failed" if any(value == "failed" for value in statuses)
        else "partial" if any(value in {"partial", "pending"} for value in statuses)
        else "complete"
    )
    return {
        "status": status,
        "groups": groups,
        "common_answer_forms": forms,
        "common_question_budget": common_budget,
        "cells_by_profile": cells,
        "profile_params": {
            profile_id: str(group["params_path"])
            for profile_id, group in zip(profile_ids, groups)
            if group.get("params_path")
        },
    }


def _design(request: Mapping[str, Any], runtime: Mapping[str, Any],
            out_root: Path, *, resume_only: bool) -> dict[str, Any]:
    if request.get("profile_params"):
        return _design_per_profile(
            request, runtime, out_root, resume_only=resume_only)
    existing = request.get("existing_design_root")
    if existing is not None:
        matrix_path = Path(existing) / "scene_profile_matrix.json"
        if not matrix_path.is_file():
            return {
                "status": "failed",
                "root": str(Path(existing).resolve()),
                "detail": f"existing design root has no matrix: {matrix_path}",
            }
        matrix = _read_json(matrix_path)
        mismatch = _validate_design_matrix(matrix, request)
        if mismatch is not None:
            return {
                "status": "failed",
                "root": str(Path(existing).resolve()),
                "matrix_path": str(matrix_path.resolve()),
                "detail": mismatch,
                "matrix": matrix,
            }
        return {
            "status": "complete" if matrix.get("status") == "completed" else "partial",
            "root": str(Path(existing).resolve()),
            "matrix_path": str(matrix_path.resolve()),
            "matrix": matrix,
            "run": {"status": "reused_existing"},
        }
    design_root = out_root / "design"
    matrix_path = design_root / "scene_profile_matrix.json"
    if matrix_path.is_file():
        matrix = _read_json(matrix_path)
        mismatch = _validate_design_matrix(matrix, request)
        if mismatch is not None:
            return {
                "status": "failed",
                "root": str(design_root.resolve()),
                "matrix_path": str(matrix_path.resolve()),
                "detail": mismatch,
                "matrix": matrix,
            }
        return {
            "status": "complete" if matrix.get("status") == "completed" else "partial",
            "root": str(design_root.resolve()),
            "matrix_path": str(matrix_path.resolve()),
            "matrix": matrix,
            "run": {"status": "reused_existing"},
        }
    if design_root.exists():
        return {
            "status": "failed",
            "root": str(design_root.resolve()),
            "detail": "design output exists without scene_profile_matrix.json",
        }
    if resume_only:
        return {
            "status": "pending",
            "root": str(design_root.resolve()),
            "detail": "resume-only mode does not launch the scheduler",
        }
    design_root.parent.mkdir(parents=True, exist_ok=True)
    command = [
        runtime["python"], str(SCHEDULER),
        "--profiles", str(request["profiles"]),
        "--params", str(request["params"]),
        "--seed", request["seed"],
        "--out-root", str(design_root),
    ]
    for scene in request["scene_configs"]:
        command.extend(["--scene-config", str(scene)])
    requested = request.get("requested_profiles")
    for profile_id in requested or []:
        command.extend(["--requested-profile", profile_id])
    forms = request.get("answer_forms")
    for form in forms or []:
        command.extend(["--answer-form", form])
    if request.get("question_budget") is not None:
        command.extend(["--question-budget", str(request["question_budget"])])
    if request.get("cells_per_pair") is not None:
        command.extend(["--cells-per-pair", str(request["cells_per_pair"])])
    weights = request.get("profile_weights")
    if weights is not None:
        weight_path = out_root / "request" / "profile_weights.json"
        if isinstance(weights, Mapping):
            _write_json(weight_path, weights)
        else:
            weight_path = Path(weights)
        command.extend(["--profile-weights", str(weight_path)])
    if request.get("snapshot_content") is not None:
        command.extend(["--snapshot-content", request["snapshot_content"]])
    run = _run_logged(
        "design", command, out_root / "logs" / "design.log",
        timeout=_timeout(runtime, "design"))
    if not matrix_path.is_file():
        run["status"] = "failed"
        run["detail"] = "scheduler returned without scene_profile_matrix.json"
        return {
            "status": "failed",
            "root": str(design_root.resolve()),
            "run": run,
        }
    matrix = _read_json(matrix_path)
    mismatch = _validate_design_matrix(matrix, request)
    if mismatch is not None:
        return {
            "status": "failed",
            "root": str(design_root.resolve()),
            "matrix_path": str(matrix_path.resolve()),
            "matrix": matrix,
            "run": run,
            "detail": mismatch,
        }
    return {
        "status": "complete" if run["status"] == "complete" and matrix.get("status") == "completed" else "partial",
        "root": str(design_root.resolve()),
        "matrix_path": str(matrix_path.resolve()),
        "matrix": matrix,
        "run": run,
    }


def _design_rows(matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = matrix.get("matrix")
    if not isinstance(rows, list):
        raise PipelineError("scene profile matrix has no matrix list")
    result = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise PipelineError(f"scene profile matrix row {index} is not an object")
        result.append(dict(row))
    return result


def _assemble(request: Mapping[str, Any], runtime: Mapping[str, Any],
              design: Mapping[str, Any], out_root: Path,
              *, resume_only: bool, through_stage: str) -> dict[str, Any]:
    assembly_root = out_root / "assembly"
    pilot_path = assembly_root / "pilot_manifest.json"
    if not _stage_enabled(through_stage, "assemble"):
        return _deferred_stage("assemble", assembly_root, through_stage)
    if pilot_path.is_file():
        pilot = _read_json(pilot_path)
        return {
            "status": "complete" if pilot.get("status") == "research_candidate" else "partial",
            "root": str(assembly_root.resolve()),
            "pilot_path": str(pilot_path.resolve()),
            "pilot": pilot,
            "run": {"status": "reused_existing"},
        }
    if assembly_root.exists():
        return {
            "status": "failed",
            "root": str(assembly_root.resolve()),
            "detail": "assembly output exists without pilot_manifest.json",
        }
    if design.get("status") == "failed":
        return {
            "status": "pending",
            "root": str(assembly_root.resolve()),
            "detail": "design matrix failed request validation",
        }
    design_groups = design.get("groups")
    if isinstance(design_groups, list):
        design_roots = [
            Path(group["root"]).resolve()
            for group in design_groups
            if (
                isinstance(group, Mapping)
                and group.get("root")
                and (Path(group["root"]) / "scene_profile_matrix.json").is_file()
            )
        ]
    elif design.get("matrix") is not None:
        design_roots = [Path(design["root"]).resolve()]
    else:
        design_roots = []
    if not design_roots:
        return {
            "status": "pending",
            "root": str(assembly_root.resolve()),
            "detail": "design matrix is unavailable",
        }
    profiles_path = _normalized_profiles_path(request, out_root)
    assembly_root.parent.mkdir(parents=True, exist_ok=True)
    command = [
        _python_for_runtime(runtime), str(ASSEMBLER),
        "--matrix-root", str(design_roots[0]),
        "--profiles", str(profiles_path),
        "--output-root", str(assembly_root),
    ]
    for root in design_roots[1:]:
        command.extend(["--supplement-root", str(root)])
    run = _run_logged(
        "assembly", command, out_root / "logs" / "assembly.log",
        timeout=_timeout(runtime, "verification"))
    if not pilot_path.is_file():
        return {
            "status": "failed",
            "root": str(assembly_root.resolve()),
            "run": run,
            "detail": "assembler returned without pilot_manifest.json",
        }
    pilot = _read_json(pilot_path)
    return {
        "status": "complete" if run["status"] == "complete" and pilot.get("status") == "research_candidate" else "partial",
        "root": str(assembly_root.resolve()),
        "pilot_path": str(pilot_path.resolve()),
        "pilot": pilot,
        "run": run,
    }


def _pair_key(scene_id: Any, profile_id: Any) -> tuple[str, str]:
    return str(scene_id), str(profile_id)


def _pair_root(out_root: Path, scene_id: str, profile_id: str) -> Path:
    return out_root / "pairs" / _safe_component(scene_id, owner="scene_id") / _safe_component(profile_id, owner="profile_id")


def _point_state_summary(
    point_ids: Sequence[str],
    state_fn,
) -> tuple[str, dict[str, str]]:
    states = {pid: state_fn(pid) for pid in point_ids}
    values = set(states.values())
    if point_ids and values == {"complete"}:
        return "complete", states
    if "failed" in values or "partial" in values:
        return "failed", states
    if values == {"missing"}:
        return "pending", states
    if "missing" in values or "pending" in values:
        return "partial", states
    return "failed", states


def _capture_states(batch_root: Path, capture_root: Path,
                    point_ids: Sequence[str]) -> tuple[str, dict[str, str], str | None]:
    def state(pid: str) -> str:
        point = batch_root / pid
        out = capture_root / pid
        try:
            selection, timeline, _ = resolve_capture_inputs(point)
            if not out.exists():
                return "missing"
            return capture_point_state(
                out, timeline_path=timeline, selection_path=selection)
        except (OSError, ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError) as exc:
            return f"failed:{type(exc).__name__}:{exc}"
    raw_status, raw_states = _point_state_summary(
        point_ids, state)
    if any(value.startswith("failed:") for value in raw_states.values()):
        return "failed", raw_states, "capture input or receipt inspection failed"
    return raw_status, raw_states, None


def _capture_command(cfg: Mapping[str, Any], batch_root: Path,
                     capture_root: Path, *, resume: bool) -> list[str]:
    required = ("python", "spear_ext", "closure_report", "stage_root", "spear_executable")
    missing = [key for key in required if not cfg.get(key)]
    if missing:
        raise PipelineError(f"capture runtime config missing {missing}")
    command = [
        str(cfg["python"]), str(CAPTURE_BATCH),
        "--inputs-root", str(batch_root),
        "--output-root", str(capture_root),
        "--python", str(cfg["python"]),
        "--spear-ext", str(cfg["spear_ext"]),
        "--closure-report", str(cfg["closure_report"]),
        "--stage-root", str(cfg["stage_root"]),
        "--spear-executable", str(cfg["spear_executable"]),
    ]
    optional = (
        ("source_asset_registry", "--source-asset-registry"),
        ("capture_warmup_config", "--capture-warmup-config"),
        ("graphics_adapter", "--graphics-adapter"),
        ("rpc_port", "--rpc-port"),
    )
    for key, flag in optional:
        if cfg.get(key) is not None:
            command.extend([flag, str(cfg[key])])
    if resume:
        command.append("--resume")
    return command


def _run_capture(request: Mapping[str, Any], runtime: Mapping[str, Any],
                 scene_id: str, profile_id: str, batch_root: Path,
                 pair_root: Path, point_ids: Sequence[str],
                 *, resume_only: bool) -> dict[str, Any]:
    capture_root = pair_root / "capture"
    status, states, detail = _capture_states(batch_root, capture_root, point_ids)
    if status == "complete":
        return {"status": "complete", "root": str(capture_root.resolve()),
                "points": states, "run": {"status": "reused_existing"}}
    if status == "failed":
        return {"status": "failed", "root": str(capture_root.resolve()),
                "points": states, "detail": detail}
    if resume_only:
        return {"status": "pending", "root": str(capture_root.resolve()),
                "points": states, "detail": "resume-only mode did not launch capture"}
    cfg = _stage_config(runtime, "capture", scene_id, profile_id)
    if not cfg.get("python") and runtime.get("python"):
        cfg["python"] = runtime["python"]
    command = _capture_command(cfg, batch_root, capture_root, resume=capture_root.exists())
    run = _run_logged(
        f"capture:{scene_id}/{profile_id}", command,
        pair_root / "capture.log", timeout=_timeout(runtime, "capture"))
    status, states, detail = _capture_states(batch_root, capture_root, point_ids)
    if status != "complete":
        return {"status": "failed", "root": str(capture_root.resolve()),
                "points": states, "run": run,
                "detail": detail or "capture did not complete every point"}
    return {"status": "complete", "root": str(capture_root.resolve()),
            "points": states, "run": run}


def _configured_audio_layouts(cfg: Mapping[str, Any]) -> tuple[str, ...]:
    try:
        return normalize_audio_layouts(cfg.get("layouts", ("binaural",)))
    except ValueError as error:
        raise PipelineError(f"invalid audio layouts: {error}") from error


def _audio_states(audio_root: Path, point_ids: Sequence[str],
                  variants: Sequence[str], layouts: Sequence[str],
                  ) -> tuple[str, dict[str, str]]:
    states = {}
    for pid in point_ids:
        for variant in variants:
            name = pid if variant == "main" else f"{pid}_{variant}"
            path = audio_root / name
            states[f"{pid}:{variant}"] = (
                "missing" if not path.exists()
                else audio_point_state(path, layouts=layouts))
    raw, compact = _point_state_summary(list(states), lambda key: states[key])
    return raw, states


def _audio_command(cfg: Mapping[str, Any], batch_root: Path,
                   capture_root: Path, audio_root: Path,
                   variants: Sequence[str], point_ids: Sequence[str],
                   *, resume: bool) -> list[str]:
    layouts = _configured_audio_layouts(cfg)
    required = (
        "python", "repo", "simulation_request",
        "package_manifest", "sound_asset_registry",
        "runtime_prefix", "rlr_sdk_root", "magnum_python_site",
        "source_asset_registry",
    )
    missing = [key for key in required if not cfg.get(key)]
    if "binaural" in layouts and not cfg.get("hrtf"):
        missing.append("hrtf")
    if missing:
        raise PipelineError(f"audio runtime config missing {missing}")
    command = [
        str(cfg["python"]), str(AUDIO_BATCH),
        "--inputs-root", str(batch_root),
        "--captures-root", str(capture_root),
        "--output-root", str(audio_root),
        "--config", str(cfg["_snapshot_path"]),
        "--variants", ",".join(variants),
        "--layouts", ",".join(layouts),
        "--points", ",".join(point_ids),
    ]
    if resume:
        command.append("--resume")
    return command


def _run_audio(request: Mapping[str, Any], runtime: Mapping[str, Any],
               scene_id: str, profile_id: str, batch_root: Path,
               pair_root: Path, point_ids: Sequence[str],
               *, resume_only: bool) -> dict[str, Any]:
    capture_root = pair_root / "capture"
    audio_root = pair_root / "audio"
    cfg = _stage_config(runtime, "audio", scene_id, profile_id)
    layouts = _configured_audio_layouts(cfg)
    status, states = _audio_states(
        audio_root, point_ids, request["audio_variants"], layouts)
    if status == "complete":
        return {"status": "complete", "root": str(audio_root.resolve()),
                "points": states, "run": {"status": "reused_existing"}}
    if status == "failed":
        return {"status": "failed", "root": str(audio_root.resolve()),
                "points": states, "detail": "an audio point is partial"}
    capture_status, _, _ = _capture_states(batch_root, capture_root, point_ids)
    if capture_status != "complete":
        return {"status": "pending", "root": str(audio_root.resolve()),
                "points": states, "detail": "capture is not complete"}
    if resume_only:
        return {"status": "pending", "root": str(audio_root.resolve()),
                "points": states, "detail": "resume-only mode did not launch audio"}
    if not cfg.get("python") and runtime.get("python"):
        cfg["python"] = runtime["python"]
    if not cfg.get("repo"):
        cfg["repo"] = str(REPO)
    snapshot_path = pair_root / "audio_config.json"
    cfg["_snapshot_path"] = str(snapshot_path.resolve())
    snapshot_value = {key: value for key, value in cfg.items() if key != "_snapshot_path"}
    snapshot = _snapshot(snapshot_path, snapshot_value)
    command = _audio_command(
        cfg, batch_root, capture_root, audio_root,
        request["audio_variants"], point_ids, resume=audio_root.exists())
    run = _run_logged(
        f"audio:{scene_id}/{profile_id}", command,
        pair_root / "audio.log", timeout=_timeout(runtime, "audio"))
    status, states = _audio_states(
        audio_root, point_ids, request["audio_variants"], layouts)
    if status != "complete":
        return {"status": "failed", "root": str(audio_root.resolve()),
                "points": states, "run": run,
                "detail": "audio did not complete every requested variant",
                "config": snapshot}
    return {"status": "complete", "root": str(audio_root.resolve()),
            "points": states, "run": run, "config": snapshot}


def _capture_receipt(capture_root: Path, pid: str) -> dict[str, Any] | None:
    path = capture_root / pid / "research_receipt.json"
    if not path.is_file():
        return None
    value = _read_json(path)
    return receipt_summary(value)


def _channel_order(media_cfg: Mapping[str, Any], capture_root: Path,
                   pid: str) -> tuple[str | None, str | None]:
    configured = media_cfg.get("channel_order")
    receipt = _read_json(capture_root / pid / "research_receipt.json")
    declared = ((receipt.get("capture") or {}).get("rgb_channel_order")
                if isinstance(receipt, Mapping) else None)
    if configured is not None and configured not in {"rgb", "bgr"}:
        return None, "media.channel_order must be rgb or bgr"
    if declared is not None and declared not in {"rgb", "bgr"}:
        return None, "capture receipt rgb_channel_order must be rgb or bgr"
    if configured is not None and declared is not None and configured != declared:
        return None, f"media channel order {configured} conflicts with receipt {declared}"
    selected = configured if configured is not None else declared
    if selected is None:
        return None, "channel order is absent from runtime config and capture receipt"
    return selected, None


def _media_state(media_root: Path, capture_root: Path,
                 audio_root: Path, pid: str) -> tuple[str, str | None]:
    output = media_root / f"{pid}.mp4"
    if not output.is_file():
        return "missing", None
    receipt = _capture_receipt(capture_root, pid)
    if receipt is None:
        return "failed", "media exists but capture receipt is missing"
    try:
        valid = _clip_matches_receipt(output, receipt)
    except (OSError, ValueError, RuntimeError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return "failed", f"cannot inspect existing review clip: {exc}"
    if valid is False:
        return "failed", "existing review clip clock differs from capture receipt"
    mixture = audio_root / pid / "audio" / "binaural" / "mixture.wav"
    if not mixture.is_file():
        return "failed", "review clip exists but main mixture is missing"
    return "complete", None


def _run_media(runtime: Mapping[str, Any], scene_id: str, profile_id: str,
               batch_root: Path, pair_root: Path, point_ids: Sequence[str],
               *, resume_only: bool) -> dict[str, Any]:
    capture_root = pair_root / "capture"
    audio_root = pair_root / "audio"
    media_root = pair_root / "media"
    audio_cfg = _stage_config(runtime, "audio", scene_id, profile_id)
    layouts = _configured_audio_layouts(audio_cfg)
    statuses = {}
    for pid in point_ids:
        state, detail = _media_state(media_root, capture_root, audio_root, pid)
        statuses[pid] = state if detail is None else f"{state}:{detail}"
    if all(value == "complete" for value in statuses.values()):
        return {"status": "complete", "root": str(media_root.resolve()),
                "points": statuses, "run": {"status": "reused_existing"}}
    if any(value.startswith("failed") for value in statuses.values()):
        return {"status": "failed", "root": str(media_root.resolve()),
                "points": statuses}
    if "binaural" not in layouts:
        return {
            "status": "failed",
            "root": str(media_root.resolve()),
            "points": statuses,
            "detail": "review media requires requested binaural audio",
        }
    audio_status, _ = _audio_states(
        audio_root, point_ids, ["main"], layouts)
    capture_status, _, _ = _capture_states(batch_root=batch_root,
                                           capture_root=capture_root,
                                           point_ids=point_ids)
    if audio_status != "complete" or capture_status != "complete":
        return {"status": "pending", "root": str(media_root.resolve()),
                "points": statuses, "detail": "capture or main audio is not complete"}
    if resume_only:
        return {"status": "pending", "root": str(media_root.resolve()),
                "points": statuses, "detail": "resume-only mode did not launch media"}
    media_cfg = _stage_config(runtime, "media", scene_id, profile_id)
    media_python = media_cfg.get("python") or _python_for_runtime(runtime)
    if not media_python:
        return {"status": "failed", "root": str(media_root.resolve()),
                "points": statuses, "detail": "media runtime has no python"}
    media_root.mkdir(parents=True, exist_ok=True)
    run_records = []
    for pid in point_ids:
        channel_order, detail = _channel_order(media_cfg, capture_root, pid)
        if detail:
            statuses[pid] = "failed:" + detail
            return {"status": "failed", "root": str(media_root.resolve()),
                    "points": statuses}
        output = media_root / f"{pid}.mp4"
        command = [
            str(media_python), str(REVIEW_CLIP),
            "--visual-capture-dir", str(capture_root / pid),
            "--mixture-wav", str(audio_root / pid / "audio" / "binaural" / "mixture.wav"),
            "--channel-order", str(channel_order),
            "--output", str(output),
        ]
        log_path = pair_root / f"media_{pid}.log"
        run = _run_logged(
            f"media:{scene_id}/{profile_id}/{pid}", command, log_path,
            timeout=_timeout(runtime, "media"))
        run_records.append(run)
        state, detail = _media_state(media_root, capture_root, audio_root, pid)
        if state != "complete":
            return {"status": "failed", "root": str(media_root.resolve()),
                    "points": statuses, "runs": run_records,
                    "detail": detail or "review clip did not validate"}
        statuses[pid] = "complete"
    return {"status": "complete", "root": str(media_root.resolve()),
            "points": statuses, "runs": run_records}


def _verification_selection(path: Path, point_ids: Sequence[str]) -> dict[str, Any]:
    selected = {"selected": [{"point_id": pid} for pid in point_ids]}
    if path.is_file():
        old = _read_json(path)
        if old != selected:
            raise PipelineError(f"verification selection differs: {path}")
        return old
    _write_json(path, selected)
    return selected


def _verification_report_passed(
    name: str,
    value: Any,
    *,
    expected_audio_variants: Sequence[str] | None = None,
    expected_audio_layouts: Sequence[str] | None = None,
) -> bool:
    """Interpret the declared result contract of each maintained verifier."""

    if not isinstance(value, Mapping):
        return False
    if name == "visual":
        return (
            value.get("schema") == "qa_v3_visual_batch_verification_v1"
            and value.get("status") == "pass"
            and int((value.get("counts") or {}).get("failures", 0)) == 0
        )
    if name == "audio":
        failures = value.get("failures")
        if not (
            value.get("schema") == "qa_v3_audio_batch_verification_v1"
            and value.get("status") == "research_candidate"
            and isinstance(failures, list)
            and not failures
        ):
            return False
        reported = value.get("expected_variants")
        if not isinstance(reported, list) or not reported:
            return False
        expected = list(expected_audio_variants or reported)
        if reported != expected or "main" not in expected:
            return False
        reported_layouts = value.get("expected_layouts")
        if not isinstance(reported_layouts, list) or not reported_layouts:
            return False
        expected_layouts = list(expected_audio_layouts or reported_layouts)
        if reported_layouts != expected_layouts:
            return False
        complete = value.get("complete_render_point_ids")
        if not isinstance(complete, Mapping):
            return False
        main_ids = complete.get("main")
        if not isinstance(main_ids, list) or not main_ids:
            return False
        if int(value.get("checked_renders", 0)) < len(main_ids) * len(expected):
            return False
        if "gateA" not in expected:
            return True
        gatea_ids = complete.get("gateA")
        pair_count = value.get("complete_pair_count")
        execution = value.get("execution_variant_verification")
        return (
            isinstance(gatea_ids, list)
            and sorted(gatea_ids) == sorted(main_ids)
            and isinstance(pair_count, int)
            and not isinstance(pair_count, bool)
            and pair_count == len(main_ids)
            and value.get("audio_variant_waveform_nonidentity_pairs") == pair_count
            and value.get("gatea_semantic_flip_pairs") == pair_count
            and isinstance(execution, Mapping)
            and execution.get("status") == "verified"
        )
    raise PipelineError(f"unknown verification report kind: {name}")


def _run_verifications(runtime: Mapping[str, Any], request: Mapping[str, Any],
                       scene_id: str, profile_id: str, batch_root: Path,
                       pair_root: Path, point_ids: Sequence[str],
                       *, params_path: Path, resume_only: bool) -> dict[str, Any]:
    capture_root = pair_root / "capture"
    audio_root = pair_root / "audio"
    verification_root = pair_root / "verification"
    visual_path = verification_root / "visual.json"
    audio_path = verification_root / "audio.json"
    selection_path = verification_root / "selection.json"
    audio_cfg = _stage_config(runtime, "audio", scene_id, profile_id)
    layouts = _configured_audio_layouts(audio_cfg)
    records = {}
    if visual_path.is_file():
        records["visual"] = _read_json(visual_path)
    if audio_path.is_file():
        records["audio"] = _read_json(audio_path)
    invalid_existing = [
        name for name, value in records.items()
        if not _verification_report_passed(
            name,
            value,
            expected_audio_variants=request["audio_variants"],
            expected_audio_layouts=layouts,
        )
    ]
    if invalid_existing:
        return {
            "status": "failed",
            "root": str(verification_root.resolve()),
            "reports": records,
            "detail": f"existing verifier reports failed: {invalid_existing}",
        }
    if set(records) == {"visual", "audio"}:
        return {"status": "complete", "root": str(verification_root.resolve()),
                "reports": records, "run": {"status": "reused_existing"}}
    capture_status, _, _ = _capture_states(batch_root, capture_root, point_ids)
    audio_status, _ = _audio_states(
        audio_root, point_ids, request["audio_variants"], layouts)
    if capture_status != "complete" or audio_status != "complete":
        return {"status": "pending", "root": str(verification_root.resolve()),
                "reports": records, "detail": "capture or audio is not complete"}
    verification_root.mkdir(parents=True, exist_ok=True)
    _verification_selection(selection_path, point_ids)
    if resume_only:
        return {"status": "pending", "root": str(verification_root.resolve()),
                "reports": records, "detail": "resume-only mode did not launch verifiers"}
    python_value = _python_for_runtime(runtime)
    commands = []
    if "visual" not in records:
        commands.append(("visual", [
            str(python_value), str(VISUAL_VERIFY),
            "--selection-manifest", str(selection_path),
            "--visual-root", str(capture_root),
            "--out", str(visual_path),
        ]))
    if "audio" not in records:
        commands.append(("audio", [
            str(python_value), str(AUDIO_VERIFY),
            "--design-root", str(batch_root),
            "--audio-root", str(audio_root),
            "--params", str(params_path.resolve()),
            "--variants", ",".join(request["audio_variants"]),
            "--layouts", ",".join(layouts),
            "--out", str(audio_path),
        ]))
    runs = {}
    for name, command in commands:
        run = _run_logged(
            f"verification:{scene_id}/{profile_id}/{name}", command,
            verification_root / f"{name}.log",
            timeout=_timeout(runtime, "verification"))
        runs[name] = run
        if not Path(visual_path if name == "visual" else audio_path).is_file():
            return {"status": "failed", "root": str(verification_root.resolve()),
                    "reports": records, "runs": runs,
                    "detail": f"{name} verifier wrote no report"}
        records[name] = _read_json(Path(visual_path if name == "visual" else audio_path))
        if (
            run["status"] != "complete"
            or not _verification_report_passed(
                name,
                records[name],
                expected_audio_variants=request["audio_variants"],
                expected_audio_layouts=layouts,
            )
        ):
            return {"status": "failed", "root": str(verification_root.resolve()),
                    "reports": records, "runs": runs,
                    "detail": f"{name} verifier did not produce a passing report"}
    return {"status": "complete", "root": str(verification_root.resolve()),
            "reports": records, "runs": runs}


def _link_no_replace(alias: Path, target: Path) -> None:
    target = target.resolve()
    if not target.is_file():
        raise PipelineError(f"release alias target is missing: {target}")
    alias.parent.mkdir(parents=True, exist_ok=True)
    if alias.is_symlink() or alias.exists():
        if alias.resolve() != target:
            raise PipelineError(f"release alias points elsewhere: {alias}")
        return
    alias.symlink_to(target)


def _pilot_candidates(pilot: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for scene_id, room in (pilot.get("rooms") or {}).items():
        for profile_id, profile in (room.get("profiles") or {}).items():
            if profile.get("status") != "selected":
                continue
            for candidate in profile.get("candidates", []):
                result.append({
                    "scene_id": str(scene_id),
                    "profile_id": str(profile_id),
                    "candidate": dict(candidate),
                })
    return result


def _run_questions(request: Mapping[str, Any], runtime: Mapping[str, Any],
                   assembly: Mapping[str, Any], pairs: Mapping[tuple[str, str], dict],
                   out_root: Path, *, resume_only: bool,
                   through_stage: str) -> dict[str, Any]:
    questions_root = out_root / "questions"
    released_path = questions_root / "released_items.json"
    if not _stage_enabled(through_stage, "questions"):
        return _deferred_stage("questions", questions_root, through_stage)
    pilot = assembly.get("pilot")
    if not isinstance(pilot, Mapping):
        return {"status": "pending", "root": str(questions_root.resolve()),
                "detail": "assembly manifest is unavailable"}
    candidates = _pilot_candidates(pilot)
    if released_path.is_file():
        items = _read_json(released_path)
        if not isinstance(items, list):
            return {"status": "failed", "root": str(questions_root.resolve()),
                    "detail": "released items output is not a list"}
        expected = int(pilot.get("question_count", 0))
        return {
            "status": "complete" if len(items) == expected else "failed",
            "root": str(questions_root.resolve()),
            "released_items": str(released_path.resolve()),
            "item_count": len(items),
            "task_types": dict(Counter(str(item.get("task_type")) for item in items)),
            "detail": None if len(items) == expected else f"expected {expected}, got {len(items)}",
            "run": {"status": "reused_existing"},
        }
    if not candidates:
        return {"status": "pending", "root": str(questions_root.resolve()),
                "detail": "assembly has no selected candidates"}
    aliases_root = out_root / "released"
    audio_root = aliases_root / "audio"
    media_root = aliases_root / "media"
    try:
        for entry in candidates:
            scene_id = entry["scene_id"]
            profile_id = entry["profile_id"]
            candidate = entry["candidate"]
            key = (scene_id, profile_id)
            pair = pairs.get(key)
            if pair is None:
                raise PipelineError(f"no pipeline pair for assembled {scene_id}/{profile_id}")
            pid = _safe_component(
                candidate["source_point_id"], owner="assembled source_point_id")
            pilot_id = _safe_component(
                candidate["pilot_id"], owner="assembled pilot_id")
            capture = pair["capture"]
            audio = pair["audio"]
            media = pair["media"]
            if capture.get("status") != "complete" or audio.get("status") != "complete" or media.get("status") != "complete":
                return {"status": "pending", "root": str(questions_root.resolve()),
                        "detail": f"media chain is incomplete for {scene_id}/{profile_id}/{pid}"}
            source_wav = Path(audio["root"]) / pid / "audio" / "binaural" / "mixture.wav"
            source_base = Path(media["root"]) / f"{pid}.base.mp4"
            _link_no_replace(
                audio_root / pilot_id / "audio" / "binaural" / "mixture.wav",
                source_wav,
            )
            _link_no_replace(
                media_root / pilot_id / "video_only.mp4",
                source_base,
            )
    except PipelineError as exc:
        return {"status": "failed", "root": str(questions_root.resolve()),
                "detail": str(exc)}
    if resume_only:
        return {"status": "pending", "root": str(questions_root.resolve()),
                "detail": "resume-only mode did not launch released adapter"}
    python_value = _python_for_runtime(runtime)
    questions_root.mkdir(parents=True, exist_ok=True)
    command = [
        str(python_value), str(RELEASED_ITEMS),
        "--selection-manifest", str(assembly["pilot_path"]),
        "--audio-root", str(audio_root),
        "--media-root", str(media_root),
        "--output", str(released_path),
        "--params", str(request["params"]),
    ]
    run = _run_logged(
        "released-items", command, questions_root / "released_items.log",
        timeout=_timeout(runtime, "verification"))
    if not released_path.is_file():
        return {"status": "failed", "root": str(questions_root.resolve()),
                "run": run, "detail": "released adapter wrote no items"}
    items = _read_json(released_path)
    if not isinstance(items, list):
        return {"status": "failed", "root": str(questions_root.resolve()),
                "run": run, "detail": "released items output is not a list"}
    expected = int(pilot.get("question_count", 0))
    status = "complete" if run["status"] == "complete" and len(items) == expected else "failed"
    return {
        "status": status,
        "root": str(questions_root.resolve()),
        "released_items": str(released_path.resolve()),
        "item_count": len(items),
        "task_types": dict(Counter(str(item.get("task_type")) for item in items)),
        "run": run,
        "detail": None if status == "complete" else f"expected {expected}, got {len(items)}",
    }


def _blocked_stage_after_failure(
    stage: str, root: Path, failed_stage: str
) -> dict[str, Any]:
    return {
        "status": "pending",
        "root": str(root.resolve()),
        "detail": f"{failed_stage} failed; fail-fast stopped before {stage}",
    }


def _pair_record(request: Mapping[str, Any], runtime: Mapping[str, Any],
                 row: Mapping[str, Any], design_root: Path, out_root: Path,
                 *, params_path: Path, resume_only: bool,
                 through_stage: str) -> dict[str, Any]:
    scene_id, profile_id = _pair_key(row.get("scene_id"), row.get("profile_id"))
    pair_root = _pair_root(out_root, scene_id, profile_id)
    batch_raw = row.get("batch_manifest")
    if not batch_raw:
        return {
            "scene_id": scene_id, "profile_id": profile_id,
            "status": str(row.get("attempt_status")),
            "design": dict(row),
            "detail": "generated row has no batch_manifest",
        }
    batch_path = Path(batch_raw).expanduser()
    if not batch_path.is_absolute():
        batch_path = design_root / batch_path
    batch_path = batch_path.resolve()
    if not batch_path.is_file():
        return {
            "scene_id": scene_id, "profile_id": profile_id,
            "status": "failed", "design": dict(row),
            "detail": f"batch manifest is missing: {batch_path}",
        }
    batch_root = batch_path.parent
    try:
        point_ids = batch_point_ids(batch_root)
    except (OSError, ValueError, TypeError) as exc:
        return {
            "scene_id": scene_id, "profile_id": profile_id,
            "status": "failed", "design": dict(row),
            "detail": f"cannot select generated points: {exc}",
        }
    if not point_ids:
        return {
            "scene_id": scene_id, "profile_id": profile_id,
            "status": "partial", "design": dict(row), "points": [],
            "detail": "generated row has no completed points",
        }
    pair_root.mkdir(parents=True, exist_ok=True)
    if _stage_enabled(through_stage, "capture"):
        try:
            capture = _run_capture(
                request, runtime, scene_id, profile_id, batch_root, pair_root, point_ids,
                resume_only=resume_only)
        except PipelineError as exc:
            capture = {
                "status": "failed",
                "root": str((pair_root / "capture").resolve()),
                "detail": str(exc),
            }
    else:
        capture = _deferred_stage(
            "capture", pair_root / "capture", through_stage)
    if capture.get("status") == "failed":
        audio = _blocked_stage_after_failure(
            "audio", pair_root / "audio", "capture")
    elif _stage_enabled(through_stage, "audio"):
        try:
            audio = _run_audio(
                request, runtime, scene_id, profile_id, batch_root, pair_root, point_ids,
                resume_only=resume_only)
        except PipelineError as exc:
            audio = {
                "status": "failed",
                "root": str((pair_root / "audio").resolve()),
                "detail": str(exc),
            }
    else:
        audio = _deferred_stage("audio", pair_root / "audio", through_stage)
    if audio.get("status") == "failed":
        media = _blocked_stage_after_failure(
            "media", pair_root / "media", "audio")
    elif _stage_enabled(through_stage, "media"):
        try:
            media = _run_media(
                runtime, scene_id, profile_id, batch_root, pair_root, point_ids,
                resume_only=resume_only)
        except PipelineError as exc:
            media = {
                "status": "failed",
                "root": str((pair_root / "media").resolve()),
                "detail": str(exc),
            }
    else:
        media = _deferred_stage("media", pair_root / "media", through_stage)
    if media.get("status") == "failed":
        verifications = _blocked_stage_after_failure(
            "verification", pair_root / "verification", "media")
    elif _stage_enabled(through_stage, "verify"):
        try:
            verifications = _run_verifications(
                runtime, request, scene_id, profile_id, batch_root, pair_root, point_ids,
                params_path=params_path, resume_only=resume_only)
        except PipelineError as exc:
            verifications = {
                "status": "failed",
                "root": str((pair_root / "verification").resolve()),
                "detail": str(exc),
            }
    else:
        verifications = _deferred_stage(
            "verify", pair_root / "verification", through_stage)
    stage_values = [
        stage.get("status")
        for stage in (capture, audio, media, verifications)
    ]
    pair_status = (
        "failed" if "failed" in stage_values
        else "complete" if all(value == "complete" for value in stage_values)
        else "partial"
    )
    return {
        "scene_id": scene_id, "profile_id": profile_id,
        "status": pair_status,
        "design": dict(row),
        "params_path": str(params_path.resolve()),
        "batch_root": str(batch_root),
        "points": point_ids,
        "capture": capture,
        "audio": audio,
        "media": media,
        "verification": verifications,
    }


def _overall_status(design: Mapping[str, Any], assembly: Mapping[str, Any],
                    pairs: Sequence[Mapping[str, Any]],
                    questions: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    failures = []
    shortfalls = []
    if design.get("status") == "failed":
        failures.append({"stage": "design", "detail": design.get("detail")})
    if assembly.get("status") == "failed":
        failures.append({"stage": "assembly", "detail": assembly.get("detail")})
    for pair in pairs:
        row = pair.get("design") or {}
        if pair.get("status") == "failed":
            failures.append({
                "stage": "pair",
                "scene_id": pair.get("scene_id"),
                "profile_id": pair.get("profile_id"),
                "detail": pair.get("detail"),
            })
        if row.get("attempt_status") not in {"generated", None} or int(row.get("quota_shortfall", 0)) > 0:
            shortfalls.append({
                "stage": "design", "scene_id": pair.get("scene_id"),
                "profile_id": pair.get("profile_id"),
                "requested_cells": row.get("requested_cells"),
                "geometry_candidates": row.get("geometry_candidates"),
                "attempt_status": row.get("attempt_status"),
                "quota_shortfall": row.get("quota_shortfall"),
            })
        for stage in ("capture", "audio", "media", "verification"):
            value = pair.get(stage) or {}
            if value.get("status") == "failed":
                failures.append({
                    "stage": stage, "scene_id": pair.get("scene_id"),
                    "profile_id": pair.get("profile_id"),
                    "detail": value.get("detail"),
                })
    if questions.get("status") == "failed":
        failures.append({"stage": "questions", "detail": questions.get("detail")})
    if failures:
        return "failed", failures, shortfalls
    pending = (
        design.get("status") in {"pending", "partial"}
        or assembly.get("status") in {"pending", "partial"}
        or any(pair.get("status") != "complete" for pair in pairs)
        or questions.get("status") in {"pending", "partial"}
        or bool(shortfalls)
    )
    return ("partial" if pending else "complete"), failures, shortfalls


def run_pipeline(request_path: str | Path, runtime_config_path: str | Path,
                 out_root: str | Path, *, resume: bool = False,
                 resume_only: bool = False,
                 through_stage: str = "questions") -> dict[str, Any]:
    through_stage = _normalize_through_stage(through_stage)
    source_state = _source_state()
    request = _load_request(Path(request_path).expanduser().resolve())
    out_root = Path(out_root).expanduser().resolve()
    if out_root.exists() and not resume:
        raise PipelineError(f"output root exists; pass resume to continue: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    _validate_resume_stage(out_root, through_stage)
    runtime = _load_runtime(
        Path(runtime_config_path).expanduser().resolve(),
        resume_only=resume_only)
    params_summary = _params_summary(request["params"])
    request_snapshot = _stable_request_snapshot(request, params_summary)
    _snapshot_request(out_root / "request.json", request_snapshot)
    runtime_snapshot = {
        key: value for key, value in runtime.items()
        if not str(key).startswith("_")
    }
    _snapshot(out_root / "runtime.json", runtime_snapshot)
    design = _design(request, runtime, out_root, resume_only=resume_only)
    assembly = _assemble(
        request, runtime, design, out_root,
        resume_only=resume_only, through_stage=through_stage)
    pairs = []
    matrix_sources = []
    if design.get("status") != "failed":
        groups = design.get("groups")
        if isinstance(groups, list):
            matrix_sources.extend(
                (
                    Path(group["root"]),
                    group.get("matrix"),
                    Path(group.get("params_path") or request["params"]).resolve(),
                )
                for group in groups
                if isinstance(group, Mapping)
                and group.get("root")
                and isinstance(group.get("matrix"), Mapping)
            )
        elif isinstance(design.get("matrix"), Mapping):
            matrix_sources.append(
                (Path(design["root"]), design["matrix"],
                 Path(request["params"]).resolve())
            )
    failed_pair: tuple[str, str] | None = None
    for design_root, matrix, params_path in matrix_sources:
        for row in _design_rows(matrix):
            scene_id, profile_id = _pair_key(
                row.get("scene_id"), row.get("profile_id"))
            if row.get("attempt_status") == "generated":
                if failed_pair is not None:
                    pair_root = _pair_root(out_root, scene_id, profile_id)
                    detail = (
                        f"fail-fast stopped after {failed_pair[0]}/"
                        f"{failed_pair[1]} failed"
                    )
                    pairs.append({
                        "scene_id": scene_id,
                        "profile_id": profile_id,
                        "status": "pending",
                        "design": dict(row),
                        "params_path": str(params_path.resolve()),
                        "points": [],
                        "capture": {
                            "status": "pending",
                            "root": str((pair_root / "capture").resolve()),
                            "detail": detail,
                        },
                        "audio": {
                            "status": "pending",
                            "root": str((pair_root / "audio").resolve()),
                            "detail": detail,
                        },
                        "media": {
                            "status": "pending",
                            "root": str((pair_root / "media").resolve()),
                            "detail": detail,
                        },
                        "verification": {
                            "status": "pending",
                            "root": str((pair_root / "verification").resolve()),
                            "detail": detail,
                        },
                        "detail": detail,
                    })
                    continue
                pair = _pair_record(
                    request, runtime, row, design_root, out_root,
                    params_path=params_path, resume_only=resume_only,
                    through_stage=through_stage)
                pairs.append(pair)
                if pair.get("status") == "failed":
                    failed_pair = (scene_id, profile_id)
            else:
                pairs.append({
                    "scene_id": scene_id,
                    "profile_id": profile_id,
                    "status": str(row.get("attempt_status")),
                    "design": dict(row),
                    "detail": row.get("detail"),
                })
    pair_map = {
        (str(pair["scene_id"]), str(pair["profile_id"])): pair
        for pair in pairs
    }
    questions = _run_questions(
        request, runtime, assembly, pair_map, out_root,
        resume_only=resume_only, through_stage=through_stage)
    status, failures, shortfalls = _overall_status(
        design, assembly, pairs, questions)
    point_count = sum(len(pair.get("points", [])) for pair in pairs)
    pipeline = {
        "kind": "qa_v3_pipeline_run",
        "status": status,
        "through_stage": through_stage,
        "source": source_state,
        "run_control": {
            "mode": "resume_only" if resume_only else "execute",
            "resume": bool(resume),
            "resume_only": bool(resume_only),
            "through_stage": through_stage,
        },
        "qualification_claim": False,
        "request": request_snapshot,
        "runtime": {
            "path": str(Path(runtime_config_path).expanduser().resolve()),
            "configured_sections": sorted(
                key for key, value in runtime.items()
                if key in {"capture", "audio", "media", "scene_profiles"}
                and value),
        },
        "stages": {
            "design": {
                key: value for key, value in design.items() if key != "matrix"
            },
            "assembly": {
                key: value for key, value in assembly.items() if key != "pilot"
            },
            "pairs": pairs,
            "questions": questions,
        },
        "counts": {
            "scene_count": len(request["scene_configs"]),
            "pair_count": len(pairs),
            "point_count": point_count,
            "complete_pairs": sum(1 for pair in pairs if pair.get("status") == "complete"),
            "question_count": questions.get("item_count", 0),
            "transcript_wer_items": sum(
                count for task, count in (questions.get("task_types") or {}).items()
                if task == "transcript_wer"),
        },
        "evaluation": {
            "status": "pending",
            "reason": "No model or human answer results were supplied; this run records production inputs and verification only.",
        },
        "failures": failures,
        "shortfalls": shortfalls,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _write_json(out_root / "pipeline_manifest.json", pipeline, replace=True)
    return pipeline


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--runtime-config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-only", action="store_true")
    parser.add_argument(
        "--through-stage",
        choices=STAGE_ORDER,
        default="questions",
        help="stop after this stage; later stages remain pending",
    )
    args = parser.parse_args(argv)
    try:
        result = run_pipeline(
            args.request, args.runtime_config, args.output_root,
            resume=args.resume, resume_only=args.resume_only,
            through_stage=args.through_stage)
    except PipelineError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "output": str(Path(args.output_root).expanduser().resolve()),
        "status": result["status"],
        "counts": result["counts"],
        "failures": len(result["failures"]),
        "shortfalls": len(result["shortfalls"]),
    }, ensure_ascii=False))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
