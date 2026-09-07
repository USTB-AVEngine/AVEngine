#!/usr/bin/env python3
"""Execute a fresh QA batch with bounded independent controller processes.

The manifest is an offline allocation.  Each saved request is copied once into
its own Episode directory and passed unchanged to ``run_qa_episode.py``.  A
single controller failure, resource diagnostic, or review failure is retained
while independent Episodes continue.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Mapping, Sequence
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed

REPOSITORY = Path(__file__).resolve().parents[2]
CONTROLLER = REPOSITORY / "tools/studio/run_qa_episode.py"
DEFAULT_MAX_PARALLEL = 4
DEFAULT_MIN_FREE_GPU_MB = 8192


class ManifestError(ValueError):
    """The manifest cannot be executed without changing its meaning."""


class ResourceBlocked(RuntimeError):
    """A declared GPU or RPC resource is unavailable before launch."""

    def __init__(self, code: str, message: str, **details: Any):
        super().__init__(message)
        self.code = code
        self.details = details


class Job:
    """One immutable manifest row plus its exact saved request."""

    __slots__ = ("episode_id", "entry", "request", "episode_root",
                 "attempt_root", "episode_output_root", "request_path", "gpu")

    def __init__(self, *, episode_id: str, entry: dict[str, Any], request: dict[str, Any],
                 episode_root: Path, attempt_root: Path, episode_output_root: Path,
                 request_path: Path, gpu: int):
        self.episode_id = episode_id
        self.entry = entry
        self.request = request
        self.episode_root = episode_root
        self.attempt_root = attempt_root
        self.episode_output_root = episode_output_root
        self.request_path = request_path
        self.gpu = gpu


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json_write_new(path: Path, value: Any) -> None:
    """Write a fresh JSON file without replacing an existing artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _json_write_atomic(path: Path, value: Any) -> None:
    """Atomically replace a mutable batch status snapshot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _append_event(path: Path, event: Mapping[str, Any], lock: threading.Lock) -> None:
    """Append one durable progress event while serializing worker writes."""
    line = json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read JSON {path}: {exc}") from exc


def _resolve_path(raw: Any, *, base: Path, owner: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ManifestError(f"{owner} must be a nonempty path")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _safe_episode_id(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError("episode_id must be nonempty text")
    value = value.strip()
    if value in {".", ".."} or Path(value).name != value or "/" in value or "\\" in value:
        raise ManifestError(f"episode_id is not a single safe path component: {value!r}")
    return value


def _request_from_row(row: Mapping[str, Any], *, manifest_path: Path) -> dict[str, Any]:
    inline = row.get("request")
    request_path_value = row.get("request_path")
    from_file = None
    if request_path_value is not None:
        request_path = _resolve_path(request_path_value, base=manifest_path.parent,
                                     owner="episode.request_path")
        if not request_path.is_file():
            raise ManifestError(f"episode request is missing: {request_path}")
        from_file = _read_json(request_path)
        if not isinstance(from_file, dict):
            raise ManifestError(f"episode request must be a JSON object: {request_path}")
    if inline is not None:
        if not isinstance(inline, dict):
            raise ManifestError("episode.request must be a JSON object")
        if from_file is not None and inline != from_file:
            raise ManifestError("inline request differs from request_path; refusing substitution")
        request = deepcopy(inline)
    elif from_file is not None:
        request = deepcopy(from_file)
    else:
        raise ManifestError("episode requires inline request or request_path")
    return request


def load_manifest(path: Path, episode_ids: Sequence[str] | None = None) -> tuple[dict[str, Any], list[tuple[dict[str, Any], dict[str, Any]]]]:
    """Load and validate rows before creating the fresh output root."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ManifestError(f"manifest is missing: {path}")
    manifest = _read_json(path)
    if not isinstance(manifest, dict):
        raise ManifestError("manifest must be a JSON object")
    rows = manifest.get("episodes")
    if not isinstance(rows, list) or not rows:
        raise ManifestError("manifest.episodes must be a nonempty list")
    wanted = None
    if episode_ids is not None:
        wanted = [_safe_episode_id(value) for value in episode_ids]
        if len(set(wanted)) != len(wanted):
            raise ManifestError("--episode-id contains duplicates")
        wanted_set = set(wanted)
    seen: set[str] = set()
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for raw_row in rows:
        if not isinstance(raw_row, dict):
            raise ManifestError("manifest episode rows must be objects")
        episode_id = _safe_episode_id(raw_row.get("episode_id"))
        if episode_id in seen:
            raise ManifestError(f"duplicate episode_id: {episode_id}")
        seen.add(episode_id)
        if wanted is not None and episode_id not in wanted_set:
            continue
        request = _request_from_row(raw_row, manifest_path=path)
        request_episode_id = request.get("episode_id")
        if request_episode_id is not None and _safe_episode_id(request_episode_id) != episode_id:
            raise ManifestError(f"request episode_id differs for {episode_id}")
        selected.append((deepcopy(raw_row), request))
    if wanted is not None:
        selected_by_id = {pair[0]["episode_id"]: pair for pair in selected}
        missing = [value for value in wanted if value not in selected_by_id]
        if missing:
            raise ManifestError(f"--episode-id is absent from manifest: {missing}")
        selected = [selected_by_id[value] for value in wanted]
    if not selected:
        raise ManifestError("episode selection is empty")
    return manifest, selected


def _producer_metadata(*, manifest_path: Path, repository: Path) -> dict[str, Any]:
    def git(args: list[str]) -> str | None:
        try:
            result = subprocess.run(["git", *args], cwd=repository,
                                    capture_output=True, text=True, check=False)
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    status = git(["status", "--porcelain"])
    return {
        "repository": str(repository),
        "cwd": str(repository),
        "invocation_cwd": str(Path.cwd()),
        "git_commit": git(["rev-parse", "HEAD"]),
        "working_tree_changes_at_launch": status.splitlines() if status is not None and status else [],
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "manifest_path": str(manifest_path),
        "executor_pid": os.getpid(),
    }


def _gpu_from_request(request: Mapping[str, Any]) -> int:
    runtime = request.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ResourceBlocked("missing_runtime", "request.runtime is missing")
    value = runtime.get("graphics_adapter")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResourceBlocked("invalid_graphics_adapter",
                              "request.runtime.graphics_adapter must be a nonnegative integer",
                              value=value)
    return value


def _query_free_gpu(gpu: int) -> dict[str, Any]:
    """Read only free VRAM; utilization is intentionally not a scheduling gate."""
    command = ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise ResourceBlocked("nvidia_smi_unavailable", f"cannot run nvidia-smi: {exc}") from exc
    if result.returncode != 0:
        raise ResourceBlocked("nvidia_smi_failed",
                              f"nvidia-smi failed with exit {result.returncode}",
                              stderr=(result.stderr or "").strip()[-1000:])
    for raw in (result.stdout or "").splitlines():
        fields = [field.strip() for field in raw.split(",")]
        if len(fields) < 2:
            continue
        try:
            index = int(fields[0])
            free_mb = int(re.search(r"-?\d+", fields[1]).group(0))
        except (AttributeError, ValueError):
            continue
        if index == gpu:
            return {"gpu": gpu, "free_memory_mb": free_mb, "command": command}
    raise ResourceBlocked("gpu_not_reported", f"nvidia-smi did not report GPU {gpu}",
                          stdout=(result.stdout or "").strip()[-1000:])


def _probe_rpc_port(port: Any) -> dict[str, Any]:
    """Bind then release one loopback port; never probe by connecting or scanning."""
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ResourceBlocked("invalid_rpc_port", "runtime.rpc_port must be an integer in [1, 65535]",
                              value=port)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
    except OSError as exc:
        raise ResourceBlocked("rpc_port_unavailable", f"RPC port {port} is unavailable: {exc}",
                              port=port) from exc
    return {"host": "127.0.0.1", "port": port, "probe": "bind_then_release"}


def _check_resources(request: Mapping[str, Any], *, min_free_gpu_mb: int | None,
                     require_rpc_port: bool = True) -> dict[str, Any]:
    gpu = _gpu_from_request(request)
    gpu_info = _query_free_gpu(gpu)
    if min_free_gpu_mb is not None and gpu_info["free_memory_mb"] < min_free_gpu_mb:
        raise ResourceBlocked("insufficient_free_vram",
                              f"GPU {gpu} has {gpu_info['free_memory_mb']} MiB free; "
                              f"required {min_free_gpu_mb} MiB",
                              gpu=gpu, free_memory_mb=gpu_info["free_memory_mb"],
                              minimum_free_memory_mb=min_free_gpu_mb)
    runtime = request.get("runtime", {})
    port_info = None
    if require_rpc_port and isinstance(runtime, Mapping) and runtime.get("rpc_port") is not None:
        port_info = _probe_rpc_port(runtime["rpc_port"])
    return {"gpu": gpu_info, "rpc_port": port_info}


def _controller_result(attempt_root: Path, stdout_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    result_path = attempt_root / "episode_result.json"
    if result_path.is_file():
        try:
            value = _read_json(result_path)
        except ManifestError as exc:
            return None, str(exc)
        if isinstance(value, dict):
            return value, None
        return None, "episode_result.json is not an object"
    try:
        text = stdout_path.read_text(encoding="utf-8", errors="replace")[-4_000_000:]
    except OSError as exc:
        return None, f"cannot read stdout log: {exc}"
    decoder = json.JSONDecoder()
    starts = [match.start() for match in re.finditer(r"\{", text)]
    for start in reversed(starts):
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and ("status" in value or "episode_id" in value):
            return value, None
    return None, "controller emitted no JSON result"


def _captured_delivery_status(result: Mapping[str, Any] | None) -> Any:
    if not isinstance(result, Mapping):
        return None
    delivery = result.get("delivery")
    if isinstance(delivery, Mapping):
        return delivery.get("status") or delivery.get("export_status") or deepcopy(dict(delivery))
    return result.get("status")



def _read_text_tail(path: Path | None, limit: int = 400_000) -> str:
    if path is None:
        return ""
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")[-limit:]
    except OSError:
        return ""


def _first_useful_error_line(text: str) -> str | None:
    """Pick a reader-facing error without requiring a traceback grep."""
    if not isinstance(text, str) or not text.strip():
        return None
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, Mapping):
        for key in ("error", "message", "reason", "failure_reason"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:2000]
    useful: list[str] = []
    for raw in stripped.splitlines():
        line = raw.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("traceback") or lower.startswith("file "):
            continue
        if line.startswith("^") or re.fullmatch(r"~+", line):
            continue
        useful.append(line)
    for line in reversed(useful):
        lower = line.lower()
        if any(token in lower for token in ("error", "exception", "failed", "exhausted", "refused")):
            return line[:2000]
    return useful[-1][:2000] if useful else None


_INTERFACE_EXCEPTION_TYPES = frozenset({
    "ModuleNotFoundError",
    "ImportError",
    "NotImplementedError",
    "CalledProcessError",
    "UnifiedAudioReceiptError",
    "EvidenceContractError",
    "TypeError",
    "AttributeError",
    "FileNotFoundError",
    "RuntimeError",
    "SystemExit",
    "AssertionError",
    "KeyError",
    "ValueError",
    "OSError",
    "JSONDecodeError",
    "AudioProgramError",
    "SubprocessError",
})
_PLANNING_EXHAUSTION_MARKERS = (
    "conditionedplanningfailure",
    "fixed condition profile exhausted",
)


def _exception_type(reason: str) -> str | None:
    """Return the last TypeError-style token in ``Type: message`` form."""
    if not isinstance(reason, str) or not reason.strip():
        return None
    match = None
    for match in re.finditer(
        r"(?:^|\n|[^A-Za-z0-9_.])(?:[A-Za-z_][\w]*\.)*([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|Failure))\s*:",
        reason,
    ):
        pass
    return match.group(1) if match else None


def _is_planning_exhaustion(
    reason: str,
    *,
    histogram: Mapping[str, Any] | None = None,
    reason_code: str | None = None,
) -> bool:
    if reason_code == "planning_exhausted":
        return True
    if isinstance(histogram, Mapping) and histogram:
        return True
    lower = (reason or "").lower()
    return any(marker in lower for marker in _PLANNING_EXHAUSTION_MARKERS)


def _is_preallocation_deficit(reason: str, *, reason_code: str | None = None) -> bool:
    if reason_code == "preallocation_gap":
        return True
    lower = (reason or "").lower()
    return "preallocation_gap" in lower or "preallocation gap" in lower


def _is_cli_interface_error(reason: str) -> bool:
    lower = (reason or "").lower()
    return (
        "unrecognized arguments" in lower
        or "the following arguments are required" in lower
        or "no such option" in lower
        or "missing cli" in lower
        or lower.lstrip().startswith("error: argument")
    )


def _is_code_exception(reason: str) -> bool:
    """True when the reason names a code exception or an audio/contract interface fault."""
    if _exception_type(reason) in _INTERFACE_EXCEPTION_TYPES:
        return True
    if _is_cli_interface_error(reason):
        return True
    lower = (reason or "").lower()
    return (
        "audioprogram validation" in lower
        or "unifiedaudioreceipt" in lower
        or "validate_evidence_contract" in lower
        or "evidencecontracterror" in lower
        or "not implemented" in lower
        or "interface_not_implemented" in lower
    )


def gap_state_for_failure(
    *,
    failure_stage: str,
    reason: str = "",
    reason_code: str | None = None,
    histogram: Mapping[str, Any] | None = None,
) -> str:
    """Owner rule 4: planning exhaustion / preallocation → evidence; code exceptions → interface."""
    if _is_preallocation_deficit(reason, reason_code=reason_code):
        return "evidence_missing_or_unsampled"
    if _is_planning_exhaustion(reason, histogram=histogram, reason_code=reason_code):
        return "evidence_missing_or_unsampled"
    stage = failure_stage or "unknown"
    if stage == "planning":
        if _is_code_exception(reason):
            return "interface_not_implemented"
        return "evidence_missing_or_unsampled"
    if stage == "audio":
        return "interface_not_implemented"
    if stage in {"capture", "finalize"}:
        if _is_code_exception(reason):
            return "interface_not_implemented"
        return "evidence_missing_or_unsampled"
    if stage == "launch":
        return "interface_not_implemented"
    if _is_code_exception(reason):
        return "interface_not_implemented"
    return "evidence_missing_or_unsampled"


def _looks_like_interface_defect(reason: str, *, failure_stage: str = "finalize") -> bool:
    """Backward-compatible wrapper around the stage/exception-type classifier."""
    return gap_state_for_failure(failure_stage=failure_stage, reason=reason) == "interface_not_implemented"


def classify_controller_failure(
    *,
    episode_output_root: Path,
    stderr_path: Path | None = None,
    stdout_path: Path | None = None,
    process_error: str | None = None,
    returncode: int | None = None,
) -> dict[str, str]:
    """Map controller artifacts onto failure_stage / gap_state / a useful reason."""
    if process_error:
        return {
            "failure_stage": "launch",
            "failure_reason": process_error,
            "gap_state": gap_state_for_failure(failure_stage="launch", reason=process_error),
            "reason_code": "controller_launch_failed",
        }
    root = Path(episode_output_root)
    planning: dict[str, Any] = {}
    planning_path = root / "planning_result.json"
    if planning_path.is_file():
        try:
            value = json.loads(planning_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            value = None
        if isinstance(value, Mapping):
            planning = dict(value)
    stderr = _read_text_tail(stderr_path)
    stdout = _read_text_tail(stdout_path)
    audio_log = _read_text_tail(root / "delivery" / "audio.log")
    capture_log = _read_text_tail(root / "capture.log")
    has_execution = (root / "execution_commands.json").is_file()
    has_capture = (root / "capture" / "neutral_readback.json").is_file()
    has_audio_report = (
        (root / "delivery" / "audio" / "research_report.json").is_file()
        or (root / "delivery" / "audio" / "research_receipt.json").is_file()
        or (root / "delivery" / "research_report.json").is_file()
    )
    histogram = planning.get("failure_histogram") if isinstance(planning.get("failure_histogram"), Mapping) else None
    planning_failed = (
        planning.get("status") == "failed"
        or not has_execution
    )
    combined = "\n".join(part for part in (audio_log, capture_log, stderr, stdout) if part)

    if planning_failed and not has_execution:
        if isinstance(histogram, Mapping) and histogram:
            reason = (
                "ConditionedPlanningFailure: fixed condition profile exhausted "
                + json.dumps(dict(histogram), ensure_ascii=False, sort_keys=True)
            )
            return {
                "failure_stage": "planning",
                "failure_reason": reason,
                "gap_state": "evidence_missing_or_unsampled",
                "reason_code": "planning_exhausted",
            }
        reason = (
            _first_useful_error_line(stderr)
            or _first_useful_error_line(stdout)
            or f"planning failed (exit {returncode})"
        )
        exhausted = _is_planning_exhaustion(reason)
        return {
            "failure_stage": "planning",
            "failure_reason": reason,
            "gap_state": gap_state_for_failure(failure_stage="planning", reason=reason),
            "reason_code": "planning_exhausted" if exhausted else "planning_failed",
        }

    audio_reason = _first_useful_error_line(audio_log)
    if audio_reason or (has_capture and not has_audio_report and (root / "delivery" / "audio.log").is_file()):
        reason = audio_reason or _first_useful_error_line(stderr) or f"audio render failed (exit {returncode})"
        return {
            "failure_stage": "audio",
            "failure_reason": reason,
            "gap_state": gap_state_for_failure(failure_stage="audio", reason=reason),
            "reason_code": "audio_failed",
        }

    if has_execution and not has_capture:
        reason = (
            _first_useful_error_line(capture_log)
            or _first_useful_error_line(stderr)
            or f"capture failed (exit {returncode})"
        )
        return {
            "failure_stage": "capture",
            "failure_reason": reason,
            "gap_state": gap_state_for_failure(failure_stage="capture", reason=reason),
            "reason_code": "capture_failed",
        }

    reason = (
        _first_useful_error_line(stderr)
        or _first_useful_error_line(combined)
        or f"finalize failed (exit {returncode})"
    )
    return {
        "failure_stage": "finalize",
        "failure_reason": reason,
        "gap_state": gap_state_for_failure(failure_stage="finalize", reason=reason),
        "reason_code": "finalize_failed",
    }


def _finalize_delivery(attempt_root: Path, manifest_entry: Mapping[str, Any], *, repository: Path) -> dict[str, Any]:
    """Lazy import so unit tests and manifest preparation need no delivery deps."""
    from avengine.qa.batch_delivery import finalize_batch_episode
    return finalize_batch_episode(attempt_root, manifest_entry, repository=repository)


def _finalize_batch_outputs(output_root: Path, manifest: Mapping[str, Any],
                           execution_summary: Mapping[str, Any], *,
                           repository: Path) -> dict[str, Any]:
    """Lazy import for the full-denominator aggregate after all jobs finish."""
    from avengine.qa.batch_delivery import finalize_batch_outputs
    return finalize_batch_outputs(output_root, manifest, execution_summary, repository=repository)


def _summary_paths(batch_summary: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(batch_summary, Mapping):
        return {}
    keys = ("summary_root", "coverage_outputs", "five_clip_listening",
            "failed_episodes", "audio_levels_and_activity", "grouped_splits")
    return {key: deepcopy(batch_summary[key]) for key in keys if key in batch_summary}


class BatchExecutor:
    def __init__(self, *, manifest_path: Path, manifest: Mapping[str, Any], jobs: Sequence[Job],
                 output_root: Path, max_parallel: int, min_free_gpu_mb: int | None,
                 repository: Path = REPOSITORY):
        if max_parallel <= 0:
            raise ValueError("max_parallel must be positive")
        if min_free_gpu_mb is not None and min_free_gpu_mb < 0:
            raise ValueError("min_free_gpu_mb must be nonnegative")
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = deepcopy(dict(manifest))
        self.jobs = list(jobs)
        self.output_root = Path(output_root).expanduser().resolve()
        self.max_parallel = max_parallel
        self.min_free_gpu_mb = min_free_gpu_mb
        self.repository = Path(repository).resolve()
        self.progress_path = self.output_root / "progress.json"
        self.events_path = self.output_root / "events.jsonl"
        self.outcomes_path = self.output_root / "outcomes.json"
        self._state_lock = threading.Lock()
        self._event_lock = threading.Lock()
        self._gpu_locks: dict[int, threading.Lock] = defaultdict(threading.Lock)
        self._progress: dict[str, Any] = {}
        self._outcomes: dict[str, dict[str, Any]] = {}

    def _counts(self) -> dict[str, int]:
        counts = Counter(row.get("status", "unknown") for row in self._progress.get("episodes", {}).values())
        return {key: int(counts.get(key, 0)) for key in
                ("queued", "running", "delivered", "failed", "blocked", "review_failed")}

    def _transition(self, episode_id: str, status: str, *, event_type: str = "status",
                    **fields: Any) -> None:
        with self._state_lock:
            row = self._progress["episodes"][episode_id]
            row.update(fields)
            row["status"] = status
            row["updated_at"] = _utc_now()
            self._progress["counts"] = self._counts()
            event = {"event": event_type, "episode_id": episode_id,
                     "status": status, "timestamp": row["updated_at"], **fields}
            _json_write_atomic(self.progress_path, self._progress)
            _append_event(self.events_path, event, self._event_lock)

    def _prepare(self) -> None:
        if self.output_root.exists():
            raise FileExistsError(f"refusing existing output: {self.output_root}")
        self.output_root.parent.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir()
        (self.output_root / "episodes").mkdir()
        _json_write_new(self.output_root / "manifest.json", {
            "manifest_path": str(self.manifest_path),
            "manifest": self.manifest,
            "selected_episode_ids": [job.episode_id for job in self.jobs],
        })
        _json_write_new(self.output_root / "producer.json",
                        _producer_metadata(manifest_path=self.manifest_path, repository=self.repository))
        self.events_path.touch(exist_ok=False)
        episodes: dict[str, Any] = {}
        for job in self.jobs:
            job.episode_root.mkdir()
            job.attempt_root.mkdir()
            _json_write_new(job.request_path, job.request)
            episodes[job.episode_id] = {
                "status": "queued", "episode_root": str(job.episode_root),
                "attempt_root": str(job.attempt_root),
                "episode_output_root": str(job.episode_output_root),
                "request_path": str(job.request_path),
                "gpu": job.gpu, "queued_at": _utc_now(),
            }
        self._progress = {
            "schema": "avengine_qa_batch_execution_progress_v1",
            "status": "running", "batch_id": self.manifest.get("batch_id"),
            "manifest_path": str(self.manifest_path), "output_root": str(self.output_root),
            "max_parallel": self.max_parallel, "min_free_gpu_mb": self.min_free_gpu_mb,
            "selected_episode_ids": [job.episode_id for job in self.jobs],
            "episodes": episodes, "updated_at": _utc_now(),
        }
        self._progress["counts"] = self._counts()
        _json_write_atomic(self.progress_path, self._progress)
        for job in self.jobs:
            _append_event(self.events_path, {
                "event": "queued", "episode_id": job.episode_id,
                "status": "queued", "timestamp": episodes[job.episode_id]["queued_at"],
                "request_path": str(job.request_path), "attempt_root": str(job.attempt_root),
            }, self._event_lock)

    def _write_outcome(self, job: Job, outcome: dict[str, Any]) -> dict[str, Any]:
        _json_write_atomic(job.attempt_root / "outcome.json", outcome)
        self._outcomes[job.episode_id] = outcome
        return outcome

    def _blocked(self, job: Job, *, code: str, message: str,
                 details: Mapping[str, Any] | None = None,
                 preserve_logs: bool = False) -> dict[str, Any]:
        now = _utc_now()
        preallocation = code == "preallocation_gap"
        outcome = {
            "schema": "avengine_qa_batch_episode_outcome_v1", "episode_id": job.episode_id,
            "status": "blocked", "reason_code": code, "reason": message,
            "failure_stage": "planning" if preallocation else "launch",
            "failure_reason": message,
            "gap_state": "evidence_missing_or_unsampled" if preallocation else "interface_not_implemented",
            "diagnostic": deepcopy(dict(details or {})), "pid": None,
            "started_at": None, "finished_at": now, "duration_seconds": 0.0,
            "request_path": str(job.request_path), "attempt_root": str(job.attempt_root),
            "episode_output_root": str(job.episode_output_root),
            "stdout_log": str(job.attempt_root / "stdout.log"),
            "stderr_log": str(job.attempt_root / "stderr.log"),
            "diagnostic_log": str(job.attempt_root / "diagnostic.log"), "command": None,
        }
        diagnostic_path = job.attempt_root / "diagnostic.log"
        with diagnostic_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{code}: {message}\n")
        if not preserve_logs:
            if not (job.attempt_root / "stdout.log").exists():
                (job.attempt_root / "stdout.log").touch(exist_ok=False)
            if not (job.attempt_root / "stderr.log").exists():
                (job.attempt_root / "stderr.log").write_text(
                    f"{code}: {message}\n", encoding="utf-8")
        outcome["diagnostic_log"] = str(diagnostic_path)
        self._write_outcome(job, outcome)
        self._transition(job.episode_id, "blocked", event_type="blocked",
                         reason_code=code, reason=message, outcome_path=str(job.attempt_root / "outcome.json"))
        return outcome

    def _run_job(self, job: Job) -> dict[str, Any]:
        gaps = job.entry.get("preallocation_gaps", [])
        if gaps:
            return self._blocked(job, code="preallocation_gap",
                                 message="manifest row has a known preallocation gap",
                                 details={"preallocation_gaps": deepcopy(gaps)})
        lock = self._gpu_locks[job.gpu]
        wait_started = time.monotonic()
        if not lock.acquire(timeout=60.0 * 60.0):
            return self._blocked(job, code="gpu_lock_timeout",
                                 message="bounded wait for per-GPU lock expired",
                                 details={"gpu": job.gpu, "wait_seconds": time.monotonic() - wait_started})
        overall_started = time.monotonic()
        controller_started_at = controller_finished_at = None
        review_started_at = review_finished_at = None
        controller_duration = review_duration = None
        try:
            renderer = job.entry.get("renderer")
            require_rpc_port = renderer not in {"habitat", "habitat_sim", "mp3d"}
            try:
                resources = _check_resources(job.request, min_free_gpu_mb=self.min_free_gpu_mb,
                                             require_rpc_port=require_rpc_port)
            except ResourceBlocked as exc:
                return self._blocked(job, code=exc.code, message=str(exc), details=exc.details)
            controller_started_at = _utc_now()
            controller_started = time.monotonic()
            command = [sys.executable, str(CONTROLLER), "--request", str(job.request_path),
                       "--output", str(job.episode_output_root)]
            self._transition(job.episode_id, "running", event_type="launching",
                             started_at=controller_started_at,
                             episode_output_root=str(job.episode_output_root),
                             gpu_resources=resources, command=command)
            stdout_path = job.attempt_root / "stdout.log"
            stderr_path = job.attempt_root / "stderr.log"
            pid = None
            returncode = None
            process_error = None
            try:
                with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open("x", encoding="utf-8") as stderr:
                    process = subprocess.Popen(command, cwd=self.repository, stdout=stdout,
                                               stderr=stderr, stdin=subprocess.DEVNULL,
                                               start_new_session=True, text=True)
                    pid = process.pid
                    self._transition(job.episode_id, "running", event_type="started", pid=pid)
                    returncode = process.wait()
            except (OSError, subprocess.SubprocessError) as exc:
                process_error = f"controller launch/wait failed: {exc}"
            controller_finished_at = _utc_now()
            controller_duration = time.monotonic() - controller_started
            controller_result = None
            parse_error = None
            if process_error is None:
                controller_result, parse_error = _controller_result(job.episode_output_root, stdout_path)
            base = {
                "schema": "avengine_qa_batch_episode_outcome_v1", "episode_id": job.episode_id,
                "request_path": str(job.request_path), "attempt_root": str(job.attempt_root),
                "episode_output_root": str(job.episode_output_root),
                "stdout_log": str(stdout_path), "stderr_log": str(stderr_path),
                "diagnostic_log": str(job.attempt_root / "diagnostic.log"),
                "command": command, "pid": pid,
                "controller_started_at": controller_started_at,
                "controller_finished_at": controller_finished_at,
                "controller_duration_seconds": controller_duration,
                "controller_returncode": returncode, "controller_result": controller_result,
                "controller_result_parse_error": parse_error,
                "captured_delivery_status": _captured_delivery_status(controller_result),
                "gpu_resources": resources,
            }
            if process_error is not None or returncode != 0:
                classified = classify_controller_failure(
                    episode_output_root=job.episode_output_root,
                    stderr_path=stderr_path,
                    stdout_path=stdout_path,
                    process_error=process_error,
                    returncode=returncode,
                )
                base.update(
                    status="failed",
                    reason_code=classified["reason_code"],
                    reason=classified["failure_reason"],
                    failure_stage=classified["failure_stage"],
                    failure_reason=classified["failure_reason"],
                    gap_state=classified["gap_state"],
                )
                base["finished_at"] = _utc_now()
                base["duration_seconds"] = time.monotonic() - overall_started
                self._write_outcome(job, base)
                self._transition(job.episode_id, "failed", event_type="completed",
                                 pid=pid, returncode=returncode,
                                 duration_seconds=base["duration_seconds"],
                                 controller_duration_seconds=controller_duration,
                                 reason_code=base["reason_code"], reason=base["reason"],
                                 outcome_path=str(job.attempt_root / "outcome.json"))
                return base
            if isinstance(controller_result, Mapping) and controller_result.get("episode_id") not in {None, job.episode_id}:
                base.update(
                    status="failed",
                    reason_code="controller_episode_mismatch",
                    reason="controller result episode_id differs from manifest",
                    failure_stage="finalize",
                    failure_reason="controller result episode_id differs from manifest",
                    gap_state="interface_not_implemented",
                )
                base["finished_at"] = _utc_now()
                base["duration_seconds"] = time.monotonic() - overall_started
                self._write_outcome(job, base)
                self._transition(job.episode_id, "failed", event_type="completed",
                                 pid=pid, returncode=returncode,
                                 duration_seconds=base["duration_seconds"],
                                 controller_duration_seconds=controller_duration,
                                 reason_code=base["reason_code"], reason=base["reason"],
                                 outcome_path=str(job.attempt_root / "outcome.json"))
                return base
            review_started_at = _utc_now()
            review_started = time.monotonic()
            try:
                review = _finalize_delivery(job.episode_output_root, job.entry, repository=self.repository)
                if not isinstance(review, Mapping):
                    raise ValueError("finalize_batch_episode returned a non-object")
                base["review"] = deepcopy(dict(review))
                base["review_status"] = review.get("status")
                if review.get("status") == "delivered":
                    base["status"] = "delivered"
                    transition = "delivered"
                else:
                    base["status"] = "review_failed"
                    base["reason_code"] = "review_failed"
                    base["reason"] = "batch delivery review did not pass"
                    base["failure_stage"] = "finalize"
                    base["failure_reason"] = "batch delivery review did not pass"
                    base["gap_state"] = "evidence_missing_or_unsampled"
                    transition = "review_failed"
            except Exception as exc:  # preserve captured delivery and continue siblings
                review_reason = f"{type(exc).__name__}: {exc}"
                base.update(status="review_failed", review_status="review_failed",
                            reason_code="review_failed", reason=review_reason,
                            review_error=review_reason,
                            failure_stage="finalize", failure_reason=review_reason,
                            gap_state=gap_state_for_failure(failure_stage="finalize", reason=review_reason))
                transition = "review_failed"
            finally:
                review_finished_at = _utc_now()
                review_duration = time.monotonic() - review_started
            base.update(review_started_at=review_started_at, review_finished_at=review_finished_at,
                        review_duration_seconds=review_duration, finished_at=_utc_now(),
                        duration_seconds=time.monotonic() - overall_started)
            self._write_outcome(job, base)
            self._transition(job.episode_id, transition, event_type="completed", pid=pid,
                             returncode=returncode, duration_seconds=base["duration_seconds"],
                             controller_duration_seconds=controller_duration,
                             review_duration_seconds=review_duration,
                             review_status=base.get("review_status"),
                             captured_delivery_status=base.get("captured_delivery_status"),
                             reason_code=base.get("reason_code"), reason=base.get("reason"),
                             outcome_path=str(job.attempt_root / "outcome.json"))
            return base
        finally:
            lock.release()


    def _run_lane(self, lane: Sequence[Job]) -> None:
        for job in lane:
            try:
                self._run_job(job)
            except Exception as exc:  # preserve this Episode's logs and continue the lane
                self._blocked(job, code="executor_internal_error",
                              message=f"{type(exc).__name__}: {exc}",
                              preserve_logs=True)

    def execute(self) -> dict[str, Any]:
        self._prepare()
        lanes_by_gpu: dict[int, list[Job]] = defaultdict(list)
        for job in self.jobs:
            lanes_by_gpu[job.gpu].append(job)
        lanes = list(lanes_by_gpu.values())
        worker_count = min(self.max_parallel, len(lanes))
        with ThreadPoolExecutor(max_workers=worker_count,
                                thread_name_prefix="qa-gpu-lane") as pool:
            futures = {pool.submit(self._run_lane, lane): lane for lane in lanes}
            for future, lane in ((future, futures[future]) for future in as_completed(futures)):
                try:
                    future.result()
                except Exception as exc:  # lane-level failure: retain every unstarted Episode
                    for job in lane:
                        if job.episode_id not in self._outcomes:
                            self._blocked(job, code="executor_internal_error",
                                          message=f"{type(exc).__name__}: {exc}",
                                          preserve_logs=True)
        outcomes = [self._outcomes[job.episode_id] for job in self.jobs]
        outcome_counts = dict(Counter(item["status"] for item in outcomes))
        operational_status = ("complete" if outcome_counts.get("delivered", 0) == len(outcomes)
                              else "completed_with_diagnostics")
        summary = {
            "schema": "avengine_qa_batch_execution_v1",
            "status": operational_status,
            "operational_status": operational_status,
            "aggregate_status": "pending",
            "batch_id": self.manifest.get("batch_id"),
            "manifest_path": str(self.manifest_path),
            "output_root": str(self.output_root),
            "episode_denominator": len(outcomes),
            "selected_episode_ids": [job.episode_id for job in self.jobs],
            "outcome_counts": outcome_counts, "episodes": outcomes,
            "producer_path": str(self.output_root / "producer.json"),
            "progress_path": str(self.progress_path), "events_path": str(self.events_path),
        }
        # Persist every per-Episode result before starting the aggregate. The
        # aggregate can fail without losing any native/controller outcome.
        _json_write_atomic(self.outcomes_path, summary)
        _append_event(self.events_path, {
            "event": "batch_complete", "status": operational_status,
            "timestamp": _utc_now(), "outcome_counts": outcome_counts,
            "aggregate_status": "pending",
        }, self._event_lock)
        with self._state_lock:
            self._progress["operational_status"] = operational_status
            self._progress["aggregate_status"] = "pending"
            self._progress["counts"] = self._counts()
            _json_write_atomic(self.progress_path, self._progress)
        selected_ids = {job.episode_id for job in self.jobs}
        aggregate_manifest = deepcopy(self.manifest)
        if isinstance(aggregate_manifest.get("episodes"), list):
            aggregate_manifest["episodes"] = [
                row for row in aggregate_manifest["episodes"]
                if isinstance(row, Mapping) and row.get("episode_id") in selected_ids
            ]
        batch_summary = None
        aggregate_error_path = None
        try:
            batch_summary = _finalize_batch_outputs(
                self.output_root, aggregate_manifest, summary, repository=self.repository)
            if not isinstance(batch_summary, Mapping):
                raise ValueError("finalize_batch_outputs returned a non-object")
            if batch_summary.get("status") != "machine_artifacts_complete":
                raise ValueError(
                    f"aggregate returned status {batch_summary.get('status')!r}")
            summary["aggregate_status"] = "complete"
            summary["batch_summary"] = deepcopy(dict(batch_summary))
            summary["batch_summary_paths"] = _summary_paths(batch_summary)
            summary["status"] = operational_status
        except Exception as exc:  # retain all jobs; never rerun native work
            aggregate_error_path = self.output_root / "aggregate_error.log"
            with aggregate_error_path.open("x", encoding="utf-8") as stream:
                stream.write(traceback.format_exc())
                stream.flush()
                os.fsync(stream.fileno())
            summary["status"] = "aggregate_failed"
            summary["aggregate_status"] = "aggregate_failed"
            summary["aggregate_error"] = f"{type(exc).__name__}: {exc}"
            summary["aggregate_error_path"] = str(aggregate_error_path)
            summary["batch_summary"] = None
            summary["batch_summary_paths"] = {}
        _json_write_atomic(self.outcomes_path, summary)
        with self._state_lock:
            self._progress["status"] = summary["status"]
            self._progress["operational_status"] = operational_status
            self._progress["aggregate_status"] = summary["aggregate_status"]
            self._progress["batch_summary"] = deepcopy(summary.get("batch_summary"))
            self._progress["batch_summary_paths"] = deepcopy(summary.get("batch_summary_paths", {}))
            if aggregate_error_path is not None:
                self._progress["aggregate_error_path"] = str(aggregate_error_path)
            self._progress["finished_at"] = _utc_now()
            self._progress["counts"] = self._counts()
            _json_write_atomic(self.progress_path, self._progress)
        _append_event(self.events_path, {
            "event": "aggregate_complete" if summary["aggregate_status"] == "complete" else "aggregate_failed",
            "status": summary["aggregate_status"], "timestamp": _utc_now(),
            "operational_status": operational_status,
            "batch_summary_paths": summary.get("batch_summary_paths", {}),
            "error_path": str(aggregate_error_path) if aggregate_error_path else None,
        }, self._event_lock)
        return summary


def _build_jobs(manifest_path: Path, manifest: Mapping[str, Any], selected: Sequence[tuple[dict[str, Any], dict[str, Any]]], output_root: Path, *, attempt_name: str = "attempt_01", force_gpu: int | None = None) -> list[Job]:
    episodes_root = output_root / "episodes"
    jobs = []
    if not isinstance(attempt_name, str) or not attempt_name or "/" in attempt_name or "\\" in attempt_name:
        raise ValueError("attempt_name must be a single path component")
    for entry, request in selected:
        episode_id = _safe_episode_id(entry["episode_id"])
        copied = deepcopy(request)
        if force_gpu is not None:
            if not isinstance(force_gpu, int) or isinstance(force_gpu, bool) or force_gpu < 0:
                raise ValueError("force_gpu must be a nonnegative integer")
            runtime = copied.get("runtime")
            runtime = dict(runtime) if isinstance(runtime, Mapping) else {}
            runtime["graphics_adapter"] = force_gpu
            copied["runtime"] = runtime
            gpu = force_gpu
        else:
            try:
                gpu = _gpu_from_request(copied)
            except ResourceBlocked:
                # Preserve the row so the executor records a diagnostic rather than
                # silently dropping it.  A sentinel lets the worker fail closed.
                runtime = copied.get("runtime")
                value = runtime.get("graphics_adapter") if isinstance(runtime, Mapping) else None
                gpu = value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else -1
        episode_root = episodes_root / episode_id
        attempt_root = episode_root / attempt_name
        jobs.append(Job(episode_id=episode_id, entry=deepcopy(entry), request=copied,
                        episode_root=episode_root, attempt_root=attempt_root,
                        episode_output_root=attempt_root / "episode",
                        request_path=episode_root / "request.json", gpu=gpu))
    return jobs


def execute_batch(manifest_path: Path, output_root: Path, *, max_parallel: int = DEFAULT_MAX_PARALLEL,
                  min_free_gpu_mb: int | None = DEFAULT_MIN_FREE_GPU_MB,
                  episode_ids: Sequence[str] | None = None,
                  repository: Path = REPOSITORY,
                  attempt_name: str = "attempt_01",
                  force_gpu: int | None = None) -> dict[str, Any]:
    manifest, selected = load_manifest(Path(manifest_path), episode_ids=episode_ids)
    output_root = Path(output_root).expanduser().resolve()
    jobs = _build_jobs(Path(manifest_path).expanduser().resolve(), manifest, selected, output_root,
                       attempt_name=attempt_name, force_gpu=force_gpu)
    return BatchExecutor(manifest_path=Path(manifest_path), manifest=manifest, jobs=jobs,
                         output_root=output_root, max_parallel=max_parallel,
                         min_free_gpu_mb=min_free_gpu_mb, repository=repository).execute()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-parallel", type=int, default=DEFAULT_MAX_PARALLEL)
    parser.add_argument("--min-free-gpu-mb", type=int, default=DEFAULT_MIN_FREE_GPU_MB)
    parser.add_argument("--episode-id", action="append", default=None,
                        help="run a selected original manifest row; repeat for multiple IDs")
    parser.add_argument("--attempt-name", default="attempt_01",
                        help="attempt directory name under each episode; G-E reruns use attempt_02")
    parser.add_argument("--force-gpu", type=int, default=None,
                        help="override request.runtime.graphics_adapter for every selected row")
    args = parser.parse_args(argv)
    try:
        summary = execute_batch(args.manifest, args.output, max_parallel=args.max_parallel,
                                min_free_gpu_mb=args.min_free_gpu_mb, episode_ids=args.episode_id,
                                attempt_name=args.attempt_name, force_gpu=args.force_gpu)
    except (ManifestError, FileExistsError, OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": summary["status"], "output": summary["output_root"],
                      "outcome_counts": summary["outcome_counts"]}, ensure_ascii=False))
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
