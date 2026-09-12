"""Run V1 production: shared units, real outputs, interruption recovery.

An ordinary worker program drives this. It reads a prepared batch manifest,
recognises which rows are core group members, asks the stage protocol which
shared units may run now, leases the machine for each one, executes it in a
fresh interpreter, reads the artifacts back, and files a stage result. No step
waits for a person or a language model, and no step is "print a command for an
agent to run next".

Three things are deliberately separate here:

* **What the protocol says may run** comes from `avengine.dataset.production_spec`
  through `avengine.qa.batch_manifest`. This module never invents a schedule.
* **What actually ran** is decided by reading the files a unit produced and
  comparing them with the upstream work item they were supposed to come from.
  A `StageResult` saying `pass` is a claim; `verify_stage_outputs` is the check.
* **What the machine can hold** comes from `avengine.dataset.production_resources`.
  Leases are submitted, pumped, rechecked immediately before launch, bound to
  the pid that actually holds the memory, and released in one step.

Recovery adopts the workers of this run that are still alive, using the
allocator's own `restore_from_snapshot`, and rebuilds the schedule from the one
valid round rather than from every result ever filed.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
import contextlib
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time
from typing import Any
from uuid import uuid4

from avengine.dataset.production_spec import (
    STAGE_PUBLISHED_FACTS,
    ProductionSpecError,
    ResourceRequest,
    RetryPolicy,
    StageResult,
    StageWorkItem,
    recipe_for_task_family,
    retry_stage_work_item,
    work_item_id,
    fresh_output_relative,
)
from avengine.dataset.production_resources import (
    Lease,
    ResourceAllocator,
    ResourcePolicy,
    ResourcePolicyError,
    ResourceUnavailable,
    WorkerCompatibility,
    backend_profiles,
    lease_request_for_work_item,
    worker_launch_plan,
)
from avengine.qa.batch_manifest import (
    core_group_from_manifest,
    group_blockers_for_group,
    stage_work_items_for_group,
    stage_work_items_for_row,
)
from avengine.qa.failure_accounting import classify_failure

SCHEMA = "avengine_v1_production_run_v1"
REPOSITORY = Path(__file__).resolve().parents[3]
# Both entries are required. `soundfile` lives in the addon path, not in the
# base Habitat environment, so dropping the second one reads as a missing
# package and tempts a worker onto another interpreter.
DEFAULT_PYTHON_PATH = ("src", "tmp/native_python_addons_v1")
DEFAULT_MAX_WAVES = 200
DEFAULT_LEASE_WAIT_S = 900.0
DEFAULT_LEASE_POLL_S = 5.0
DEFAULT_CANDIDATE_ROTATIONS = 2

# A path-shaped fact must actually exist. These are the published facts whose
# value names a file the next stage reads.
PATH_FACT_SUFFIXES = ("_path", "_paths")
# An attempt that was interrupted produced no result and is not a failure of
# the work. Recovering it does not spend the stage retry budget.
INTERRUPTED_REASON_CODE = "attempt_interrupted"
# A planning unit occupies no port. The recipe's own request builder still
# demands one, and the capture unit replaces it with the leased port, so the
# value a plan carries is a placeholder and is recorded as one.
RECIPE_PLACEHOLDER_RPC_PORT = 39782
DEFAULT_INTERRUPTED_RETRIES = 2


class ProductionRunError(RuntimeError):
    """The run cannot continue as written."""

    reason_code = "production_run_failed"

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.details = details


class StageExecutorMissing(ProductionRunError):
    """No real per-unit entry point is wired for this recipe unit yet."""

    reason_code = "stage_executor_not_implemented"


class StageOutputMissing(ProductionRunError):
    """A unit reported facts whose artifacts are absent or do not match."""

    reason_code = "stage_output_missing"


class CandidateRejected(ProductionRunError):
    """A legal candidate was refused; another candidate may be tried."""

    reason_code = "candidate_rejected"


class NativeBudgetExhausted(ProductionRunError):
    """This run has used its authorised number of new native visual worlds."""

    reason_code = "native_visual_world_budget_exhausted"


def _sampling_candidate_index_from_payload(payload: Mapping[str, Any] | None) -> int | None:
    """Read the retry selection carried by a stage work item.

    candidate_index is the scheduler's historical payload name. The
    sampler-facing name is explicit so the worker task cannot silently record
    a rotation while handing the original request to the callee. Both names
    are accepted during recovery, but if a producer writes both they must
    identify the same candidate.
    """
    if not isinstance(payload, Mapping):
        return None
    values: list[tuple[str, int]] = []
    for key in ("candidate_index", "sampling_candidate_index"):
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ProductionRunError(
                f"work item payload {key} must be a nonnegative integer"
            )
        values.append((key, int(value)))
    if not values:
        return None
    if len({value for _key, value in values}) != 1:
        raise ProductionRunError(
            "work item payload candidate_index and sampling_candidate_index disagree"
        )
    return values[0][1]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json_write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _json_write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Process identity: what "this run's worker is still alive" actually means
# ---------------------------------------------------------------------------


def process_identity(pid: int) -> dict[str, Any] | None:
    """Read enough of a live process to recognise it again after a restart.

    A bare pid is not an identity: Linux reuses pids, so a run that adopted a
    recycled pid would report somebody else's process as its own worker. The
    start time plus the command line is what makes the check honest.
    """
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8", errors="replace")
        raw_cmdline = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except (OSError, ValueError):
        return None
    tail = stat.rsplit(")", 1)[-1].split()
    start_ticks = tail[19] if len(tail) > 19 else None
    cmdline = [part for part in raw_cmdline.decode("utf-8", "replace").split("\0") if part]
    return {"pid": int(pid), "start_ticks": start_ticks, "cmdline": cmdline}


def process_matches(record: Mapping[str, Any] | None) -> bool:
    """Is the recorded worker of this run still the process behind that pid?"""
    if not isinstance(record, Mapping) or record.get("pid") is None:
        return False
    current = process_identity(int(record["pid"]))
    if current is None:
        return False
    if record.get("start_ticks") and current["start_ticks"] != record["start_ticks"]:
        return False
    expected_argv = record.get("argv")
    if expected_argv:
        # Saved launch argv is authoritative. Legacy markers were logical IDs
        # containing colons, while the actual task filename used underscores.
        # Such a marker rejected the very process whose full argv matched.
        return current["cmdline"] == [str(value) for value in expected_argv]
    marker = record.get("cmdline_marker")
    if marker and not any(marker in part for part in current["cmdline"]):
        return False
    return True


def _find_live_stage_worker(work_dir: Path) -> dict[str, Any] | None:
    """Rediscover only this task file's exact native worker after parent loss."""
    task_path = str((work_dir / "task.json").resolve())
    result_path = str((work_dir / "stage_result.json").resolve())
    if not Path(task_path).is_file():
        return None
    found = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdecimal():
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            identity = process_identity(int(directory.name))
            argv = identity["cmdline"] if identity else []
            module_index = argv.index("-m")
            task_index = argv.index("--execute-work-item")
            result_index = argv.index("--result")
            if (argv[module_index + 1] != "avengine.dataset.production_runner"
                    or argv[task_index + 1] != task_path
                    or argv[result_index + 1] != result_path):
                continue
        except (OSError, ValueError, IndexError):
            continue
        found.append({
            "pid": int(directory.name), "start_ticks": identity["start_ticks"],
            "argv": argv, "cmdline_marker": task_path,
            "stdout_log": str(work_dir / "stdout.log"),
            "stderr_log": str(work_dir / "stderr.log"),
            "rediscovered_from_exact_task_argv": True,
        })
    if len(found) > 1:
        raise ProductionRunError(f"multiple live workers claim the same task: {task_path}")
    return found[0] if found else None


# ---------------------------------------------------------------------------
# Stage executors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageContext:
    """Everything one unit's executor is given, and nothing it must guess."""

    work_item: dict[str, Any]
    manifest: dict[str, Any]
    manifest_path: Path
    group_id: str | None
    task_family: str | None
    unit_id: str | None
    stage: str
    scope_id: str
    attempt: int
    member_request_ids: tuple[str, ...]
    rows_by_episode_id: Mapping[str, Mapping[str, Any]]
    upstream: Mapping[str, Mapping[str, Any]]
    output_root: Path
    run_root: Path
    repository: Path
    rpc_port: int | None
    graphics_adapter: int | None
    lease: Mapping[str, Any] | None
    # Every result of this scope's valid round, which the recipe's own unit
    # entry points read to check that the finished visuals are one world.
    round_results: tuple[Mapping[str, Any], ...] = ()
    # Configuration the run carries for this recipe: retained visual roots to
    # link instead of rendering, the world id, the QA ids, the split.
    recipe_options: Mapping[str, Any] = field(default_factory=dict)

    @property
    def unit_kind(self) -> str:
        return str(self.work_item.get("payload", {}).get("unit_kind") or self.stage)

    @property
    def request(self) -> dict[str, Any]:
        """The saved legacy request this unit runs, with the leased runtime.

        The lease's device and port are written in here rather than left to a
        default, and `runtime_readback` reports what the callee was actually
        given so a claim about isolation can be checked instead of assumed.
        """
        inputs = self.work_item.get("inputs") or {}
        request = inputs.get("request")
        if not isinstance(request, Mapping):
            for row_id in self.member_request_ids:
                row = self.rows_by_episode_id.get(row_id)
                if isinstance(row, Mapping) and isinstance(row.get("request"), Mapping):
                    request = row["request"]
                    break
        if not isinstance(request, Mapping):
            raise ProductionRunError(
                f"{self.work_item.get('work_item_id')} carries no saved request to run"
            )
        value = deepcopy(dict(request))
        policy_override = (self.recipe_options or {}).get("question_acceptance_policy")
        if policy_override:
            sampling = dict(value.get("qa_sampling") or {})
            sampling["acceptance_policy"] = {
                **dict(sampling.get("acceptance_policy") or {}), **dict(policy_override)}
            value["qa_sampling"] = sampling
        runtime = dict(value.get("runtime") or {})
        runtime_prefix = os.environ.get("AVENGINE_HABITAT_RUNTIME_PREFIX")
        if not runtime.get("runtime_prefix") and runtime_prefix:
            runtime["runtime_prefix"] = runtime_prefix
        if self.graphics_adapter is not None:
            runtime["graphics_adapter"] = int(self.graphics_adapter)
        if self.rpc_port is not None:
            runtime["rpc_port"] = int(self.rpc_port)
        value["runtime"] = runtime
        candidate_index = _sampling_candidate_index_from_payload(
            self.work_item.get("payload") or {}
        )
        if candidate_index is not None:
            value["sampling_candidate_index"] = int(candidate_index)
        return value

    def runtime_readback(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """What the leased values became in the request handed to the callee."""
        runtime = dict((request or {}).get("runtime") or {})
        return {
            "leased_graphics_adapter": self.graphics_adapter,
            "leased_rpc_port": self.rpc_port,
            "request_graphics_adapter": runtime.get("graphics_adapter"),
            "request_rpc_port": runtime.get("rpc_port"),
            "matches": (
                (self.graphics_adapter is None
                 or runtime.get("graphics_adapter") == self.graphics_adapter)
                and (self.rpc_port is None or runtime.get("rpc_port") == self.rpc_port)
            ),
        }

    def upstream_unit(self, unit_id: str) -> Mapping[str, Any]:
        result = self.upstream.get(unit_id)
        if not isinstance(result, Mapping):
            raise ProductionRunError(
                f"{self.work_item.get('work_item_id')} needs the passed result of "
                f"unit {unit_id!r}; it has {sorted(self.upstream)}"
            )
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_item": deepcopy(self.work_item),
            "manifest_path": str(self.manifest_path),
            "group_id": self.group_id,
            "task_family": self.task_family,
            "unit_id": self.unit_id,
            "stage": self.stage,
            "scope_id": self.scope_id,
            "attempt": self.attempt,
            "member_request_ids": list(self.member_request_ids),
            "upstream": {key: deepcopy(dict(value)) for key, value in self.upstream.items()},
            "output_root": str(self.output_root),
            "run_root": str(self.run_root),
            "repository": str(self.repository),
            "rpc_port": self.rpc_port,
            "graphics_adapter": self.graphics_adapter,
            "lease": None if self.lease is None else deepcopy(dict(self.lease)),
            "round_results": [deepcopy(dict(row)) for row in self.round_results],
            "recipe_options": deepcopy(dict(self.recipe_options)),
        }


@dataclass(frozen=True)
class StageOutcome:
    """What a unit produced, before anything checks whether it is real."""

    status: str
    facts: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None
    reason_code: str | None = None
    # Native visual worlds this unit actually started, failed attempts included.
    native_visual_worlds: int = 0
    native_acoustic_contexts: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "facts": deepcopy(self.facts),
            "outputs": deepcopy(self.outputs),
            "reason": self.reason,
            "reason_code": self.reason_code,
            "native_visual_worlds": int(self.native_visual_worlds),
            "native_acoustic_contexts": int(self.native_acoustic_contexts),
            "diagnostics": deepcopy(self.diagnostics),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StageOutcome":
        return cls(
            status=str(value["status"]),
            facts=dict(value.get("facts") or {}),
            outputs=dict(value.get("outputs") or {}),
            reason=None if value.get("reason") is None else str(value["reason"]),
            reason_code=(None if value.get("reason_code") is None
                         else str(value["reason_code"])),
            native_visual_worlds=int(value.get("native_visual_worlds") or 0),
            native_acoustic_contexts=int(value.get("native_acoustic_contexts") or 0),
            diagnostics=dict(value.get("diagnostics") or {}),
        )


StageExecutor = Callable[[StageContext], StageOutcome]


@dataclass(frozen=True)
class StageExecutorEntry:
    task_family: str | None
    unit_kind: str
    call: StageExecutor
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {"task_family": self.task_family, "unit_kind": self.unit_kind,
                "source": self.source}


_EXECUTORS: dict[tuple[str | None, str], StageExecutorEntry] = {}


def register_stage_executor(
    *, task_family: str | None, unit_kind: str, call: StageExecutor, source: str,
    replace_existing: bool = False,
) -> StageExecutorEntry:
    """Bind one recipe unit kind to the real function that produces it."""
    key = (task_family, unit_kind)
    if key in _EXECUTORS and not replace_existing:
        raise ProductionRunError(
            f"an executor for {task_family}/{unit_kind} is already registered by "
            f"{_EXECUTORS[key].source}"
        )
    entry = StageExecutorEntry(task_family=task_family, unit_kind=unit_kind,
                               call=call, source=source)
    _EXECUTORS[key] = entry
    return entry


def resolve_stage_executor(
    task_family: str | None,
    unit_kind: str,
    *,
    unit_id: str | None = None,
) -> StageExecutorEntry:
    """Resolve a unit without letting a core scope become an Episode."""
    candidates = [str(unit_kind)]
    if task_family is not None:
        stage_aliases = {
            "plan": "visual_plan",
            "late_plan": "visual_plan",
            "capture": "visual_capture",
        }
        alias = stage_aliases.get(str(unit_kind))
        if alias is not None:
            candidates.append(alias)
        if unit_id:
            try:
                recipe = recipe_for_task_family(str(task_family))
                candidates.insert(0, recipe.unit(str(unit_id)).unit_kind)
            except (ProductionSpecError, ValueError):
                pass
    for candidate in dict.fromkeys(candidates):
        entry = _EXECUTORS.get((task_family, candidate))
        if entry is not None:
            return entry
        if task_family is None:
            entry = _EXECUTORS.get((None, candidate))
            if entry is not None:
                return entry
    raise StageExecutorMissing(
        f"no stage executor is wired for task family {task_family!r} "
        f"unit kind {unit_kind!r} unit_id {unit_id!r}; registered: "
        f"{sorted((str(family), kind) for family, kind in _EXECUTORS)}",
        task_family=task_family, unit_kind=unit_kind, unit_id=unit_id,
    )


@contextlib.contextmanager
def stage_executor_overrides(
    entries: Mapping[tuple[str | None, str], StageExecutor], *, source: str,
):
    """Swap in executors for one caller, then put the registry back.

    Used to drive the schedule without native work while still going through
    the same task file, verification and journal the real path uses.
    """
    saved = dict(_EXECUTORS)
    try:
        for (task_family, unit_kind), call in entries.items():
            register_stage_executor(task_family=task_family, unit_kind=unit_kind,
                                    call=call, source=source, replace_existing=True)
        yield
    finally:
        _EXECUTORS.clear()
        _EXECUTORS.update(saved)


def registered_stage_executors() -> list[dict[str, Any]]:
    return [entry.to_dict() for _key, entry in sorted(
        _EXECUTORS.items(), key=lambda item: (str(item[0][0]), item[0][1]))]


# ---------------------------------------------------------------------------
# Real per-unit adapters over the existing recipe functions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NativeRecipeBinding:
    """Which module carries one task family's real per-unit stage entry points.

    Adding a recipe is data here, not another executor: every family that has
    a `group_stage_context` / `run_group_stage_work_item` pair is driven the
    same way, and a family without one is reported by name rather than faked.
    """

    task_family: str
    module: str
    context_call: str = "group_stage_context"
    run_call: str = "run_group_stage_work_item"
    load_results_call: str = "load_group_stage_results"

    def resolve(self, name: str) -> Any:
        import importlib

        try:
            module = importlib.import_module(self.module)
        except ImportError as error:
            raise StageExecutorMissing(
                f"{self.module} cannot be imported for the {self.task_family} recipe: "
                f"{error}", module=self.module, task_family=self.task_family) from error
        function = getattr(module, name, None)
        if function is None:
            raise StageExecutorMissing(
                f"{self.module}.{name} does not exist, so the {self.task_family} recipe "
                "cannot be driven one unit at a time yet",
                module=self.module, function=name, task_family=self.task_family)
        return function

    def to_dict(self) -> dict[str, Any]:
        return {"task_family": self.task_family, "module": self.module,
                "context_call": self.context_call, "run_call": self.run_call,
                "load_results_call": self.load_results_call}


NATIVE_RECIPE_BINDINGS = {
    "visible_binding": NativeRecipeBinding(
        task_family="visible_binding",
        module="avengine.dataset.binding_group_native"),
    "visual_conditioned_relation": NativeRecipeBinding(
        task_family="visual_conditioned_relation",
        module="avengine.dataset.binding_group_native",
        context_call="relation_group_stage_context",
        run_call="run_relation_group_stage_work_item"),
    "cross_event_identity": NativeRecipeBinding(
        task_family="cross_event_identity",
        module="avengine.dataset.binding_group_identity",
        context_call="identity_group_stage_context",
        run_call="run_identity_group_stage_work_item"),
    "cross_time_state": NativeRecipeBinding(
        task_family="cross_time_state",
        module="avengine.dataset.binding_group_motion"),
}
# Recipes whose real segmentation is not yet callable per unit, with the exact
# entry point each one still needs. Reporting this is the point: a unit that
# cannot run must say which function is missing, not quietly pass.
PENDING_RECIPE_SEAMS = {}


def recipe_binding(task_family: str | None) -> NativeRecipeBinding:
    binding = NATIVE_RECIPE_BINDINGS.get(str(task_family))
    if binding is None:
        raise StageExecutorMissing(
            f"no per-unit recipe binding for task family {task_family!r}; "
            f"wired: {sorted(NATIVE_RECIPE_BINDINGS)}", task_family=task_family)
    return binding


def _first_member_request(context: StageContext) -> dict[str, Any]:
    for row_id in context.member_request_ids or ():
        row = context.rows_by_episode_id.get(str(row_id))
        if isinstance(row, Mapping) and isinstance(row.get("request"), Mapping):
            return dict(row["request"])
    return {}


def declared_world_id(manifest: Mapping[str, Any], group_id: str | None) -> str | None:
    for entry in (manifest.get("production") or {}).get("core_groups") or ():
        if entry.get("group_id") == group_id and entry.get("world_id"):
            return str(entry["world_id"])
    return None


def native_group_context(context: StageContext) -> dict[str, Any]:
    """Rebuild the recipe's own group context from the saved manifest."""
    binding = recipe_binding(context.task_family)
    build = binding.resolve(binding.context_call)
    options = dict(context.recipe_options or {})
    request = _first_member_request(context)
    qa_ids = options.get("qa_ids") or request.get("qa_ids")
    candidate_index = _sampling_candidate_index_from_payload(
        context.work_item.get("payload") or {}
    )
    manifest = context.manifest
    if candidate_index is not None:
        manifest = deepcopy(context.manifest)
        for row in manifest.get("episodes", []):
            if row.get("group_id") != context.group_id:
                continue
            member_request = row.get("request")
            if isinstance(member_request, Mapping):
                member_request["sampling_candidate_index"] = int(candidate_index)
    value = dict(build(
        manifest, context.group_id,
        world_id=options.get("world_id") or declared_world_id(manifest,
                                                              context.group_id),
        qa_ids=None if qa_ids is None else list(qa_ids),
        retained_visual_roots=options.get("retained_visual_roots") or None,
        split=str(options.get("split") or "pilot"),
    ))
    binding_overrides = options.get("binding_motion_overrides")
    if binding_overrides is not None:
        if not isinstance(binding_overrides, Mapping):
            raise ProductionRunError(
                "binding_motion_overrides must be a mapping"
            )
        for request_value in (value.get("member_requests") or {}).values():
            binding_motion = dict(request_value.get("binding_motion") or {})
            binding_motion.update(deepcopy(dict(binding_overrides)))
            request_value["binding_motion"] = binding_motion
        value["binding_motion_overrides"] = deepcopy(dict(binding_overrides))
    repair = options.get("late_plan_repair")
    if repair is not None:
        if not isinstance(repair, Mapping):
            raise ProductionRunError("late_plan_repair must be a mapping")
        value["late_plan_repair"] = deepcopy(dict(repair))
    return value


def lease_placement(context: StageContext) -> dict[str, Any]:
    """The lease in the vocabulary the recipe's placement helper reads.

    The allocator calls the device `device_index`; a request calls it
    `graphics_adapter`. Translating here is what makes the readback in the
    unit's own `instance_runtime` a check rather than a coincidence.
    """
    if context.lease is None:
        return {}
    lease = dict(context.lease)
    return {
        "lease_id": lease.get("lease_id"),
        "graphics_adapter": lease.get("device_index"),
        "rpc_port": lease.get("rpc_port"),
        "rlr_threads": (context.work_item.get("resource") or {}).get("rlr_threads"),
    }


def retained_root_declared_for(context: StageContext) -> str | None:
    """A retained visual this unit is allowed to link instead of rendering."""
    retained = dict((context.recipe_options or {}).get("retained_visual_roots") or {})
    if not retained or context.unit_id is None:
        return None
    if context.unit_id in retained:
        return str(retained[context.unit_id])
    try:
        recipe = recipe_for_task_family(str(context.task_family))
        unit = recipe.unit(str(context.unit_id))
    except ProductionSpecError:
        return None
    for name in unit.depends_on_units:
        if name in retained:
            return str(retained[name])
    for other in recipe.units:
        if other.unit_kind == "visual_capture" and context.unit_id in other.depends_on_units:
            if other.unit_id in retained:
                return str(retained[other.unit_id])
    return None


def _native_group_unit_executor(context: StageContext) -> StageOutcome:
    """Run one shared unit through the recipe's own per-unit entry point.

    Everything native is the recipe's: the plan, the capture, the audio column
    and the assembly. What this adds is the placement the allocator decided,
    the results of the round so far, and the accounting of how many native
    worlds and acoustic contexts the attempt actually used.
    """
    binding = recipe_binding(context.task_family)
    run_unit = binding.resolve(binding.run_call)
    recipe_context = native_group_context(context)
    result = run_unit(
        context.work_item, recipe_context,
        output_root=context.run_root / "work",
        results=list(context.round_results),
        lease=lease_placement(context) or None,
    )
    outputs = dict(result.get("outputs") or {})
    facts = dict(result.get("facts") or {})
    retained = retained_root_declared_for(context)
    unit_kind = context.unit_kind
    worlds = outputs.get("native_visual_worlds_created")
    if worlds is None:
        # A capture attempt that raised before reporting still started a world
        # unless it was linking a retained one. Failed attempts count.
        worlds = 0 if (unit_kind != "visual_capture" or retained) else 1
    contexts = 0
    if unit_kind == "audio":
        reused = bool((outputs.get("shared_audio_column") or {}).get("reused"))
        contexts = 0 if reused else 1
    return StageOutcome(
        status=str(result.get("status") or "fail"),
        facts=facts,
        outputs=outputs,
        reason=result.get("reason"),
        reason_code=None if result.get("status") == "pass" else "stage_not_passed",
        native_visual_worlds=int(worlds),
        native_acoustic_contexts=int(contexts),
        diagnostics={"recipe_module": binding.module,
                     "recipe_entry": binding.run_call,
                     "retained_visual_root": retained,
                     "world_id": recipe_context.get("world_id"),
                     "world_id_source": recipe_context.get("world_id_source")},
    )



def _ordinary_retained_root(context: StageContext) -> Path | None:
    value = (context.recipe_options or {}).get("retained_episode_root")
    if not isinstance(value, str) or not value.strip():
        return None
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise StageOutputMissing(f"retained ordinary Episode root is missing: {root}")
    return root


def _ordinary_stable_request(request: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "episode_id",
        "room_id",
        "seed",
        "frame_count",
        "frame_rate_hz",
        "sample_rate_hz",
        "camera",
        "audio_layouts",
        "post_assembly_convolution_gain",
        "source_asset_ids",
        "qa_ids",
        "qa_sampling",
        "profile",
        "qa_targets",
        "entity_instances",
        "sampling_candidate_index",
    )
    result = {key: deepcopy(request.get(key)) for key in keys}
    # Which measured answers are accepted does not alter the captured world.
    policy = (result.get("qa_sampling") or {}).get("acceptance_policy")
    if isinstance(policy, dict):
        policy.pop("accept_observed_branches", None)
    return result


def _ordinary_request_with_lease(context: StageContext, request: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(request))
    runtime = dict(value.get("runtime") or {})
    bindings = runtime.get("path_bindings") or {}
    if not runtime.get("mp3d_root") and bindings.get("AVENGINE_MP3D_ROOT"):
        runtime["mp3d_root"] = bindings["AVENGINE_MP3D_ROOT"]
    if context.graphics_adapter is not None:
        runtime["graphics_adapter"] = int(context.graphics_adapter)
    if context.rpc_port is not None:
        runtime["rpc_port"] = int(context.rpc_port)
    value["runtime"] = runtime
    return value


def _ordinary_replay_paths(root: Path) -> dict[str, Path]:
    request = root / "request.json"
    plan = root / "plan" / "episode_plan.json"
    capture = root / "capture"
    if not request.is_file() or not plan.is_file() or not capture.is_dir():
        raise StageOutputMissing(
            f"retained ordinary Episode lacks request/plan/capture under {root}"
        )
    return {"request": request, "plan": plan, "capture": capture}


def _ordinary_audio_report(root: Path) -> Path:
    refs = root / "delivery" / "input_refs.json"
    if refs.is_file():
        value = _read_json(refs)
        if isinstance(value, Mapping):
            path = value.get("audio_report")
            if isinstance(path, str) and Path(path).is_file():
                return Path(path).resolve()
    candidates = (
        root / "delivery" / "research_report.json",
        root / "delivery" / "audio" / "research_report.json",
        root / "delivery" / "audio" / "research_receipt.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise StageOutputMissing(f"retained ordinary Episode has no readable audio report: {root}")


def _ordinary_plan_executor(context: StageContext) -> StageOutcome:
    from avengine.dataset.binding_group_native import plan_visual_variant

    root = context.output_root
    root.mkdir(parents=True, exist_ok=True)
    request = context.request
    retained = _ordinary_retained_root(context)
    preplanned_value = (context.recipe_options or {}).get("preplanned_episode_root")
    if retained is not None and preplanned_value:
        raise ProductionRunError("ordinary Episode cannot select both retained and preplanned roots")
    if preplanned_value:
        plan_root = Path(str(preplanned_value)).expanduser().resolve()
        request_path = plan_root / "request.json"
        plan_path = plan_root / "plan" / "episode_plan.json"
        if not request_path.is_file() or not plan_path.is_file():
            raise StageOutputMissing(f"preplanned ordinary Episode lacks request/plan: {plan_root}")
        if (plan_root / "capture").exists():
            raise ProductionRunError(f"preplanned ordinary Episode already contains capture: {plan_root}")
        planned_request = _read_json(request_path)
        def comparable(value):
            value = deepcopy(dict(value))
            runtime = dict(value.get("runtime") or {})
            for key in ("graphics_adapter", "rpc_port"):
                runtime.pop(key, None)
            value["runtime"] = runtime
            value["sampling_candidate_index"] = int(value.get("sampling_candidate_index") or 0)
            policy = (value.get("qa_sampling") or {}).get("acceptance_policy")
            if isinstance(policy, dict):
                policy.pop("accept_observed_branches", None)
            return value
        current = comparable(request)
        saved = comparable(planned_request)
        mismatches = sorted(key for key in set(current) | set(saved)
                            if current.get(key) != saved.get(key))
        if mismatches:
            raise ProductionRunError(
                f"preplanned ordinary Episode request differs for {context.scope_id}: {mismatches}"
            )
        # Materialize the verified CPU plan inside this stage's fresh root.
        # The executor's ordinary artifact-boundary checks then apply unchanged.
        copied_root = root / "episode"
        shutil.copytree(plan_root, copied_root, symlinks=True)
        plan_root = copied_root
        request_path = copied_root / "request.json"
        plan_path = copied_root / "plan" / "episode_plan.json"
        execution = "reused_cpu_preplanned_episode"
    elif retained is not None:
        paths = _ordinary_replay_paths(retained)
        retained_request = _read_json(paths["request"])
        current_stable = _ordinary_stable_request(request)
        retained_stable = _ordinary_stable_request(retained_request)
        mismatches = {
            key: {"current": current_stable.get(key), "retained": retained_stable.get(key)}
            for key in retained_stable
            if retained_stable.get(key) is not None
            and current_stable.get(key) != retained_stable.get(key)
        }
        if mismatches:
            raise ProductionRunError(
                f"retained ordinary Episode stable request differs for {context.scope_id}: "
                f"{mismatches}"
            )
        request_path = paths["request"]
        plan_path = paths["plan"]
        plan_root = retained
        execution = "reused_retained_plan"
    else:
        request_path = root / "request.json"
        _json_write_new(request_path, request)
        planned = plan_visual_variant(
            request_path, root / "episode", label="ordinary_plan",
            log=root / "ordinary_plan.log",
        )
        plan_path = Path(planned["plan"]).resolve()
        plan_root = Path(planned["output"]).resolve()
        execution = "plan_only"
    plan = _read_json(plan_path)
    resources = plan.get("resources") if isinstance(plan.get("resources"), Mapping) else {}
    package = resources.get("room_package") if isinstance(resources.get("room_package"), Mapping) else {}
    renderer = str(package.get("renderer") or resources.get("renderer") or resources.get("backend") or "")
    return StageOutcome(
        status="pass",
        facts={
            "episode_plan_path": str(plan_path),
            "renderer": renderer,
            "clock": deepcopy(plan.get("clock")),
        },
        outputs={
            "output_root": str(plan_root),
            "plan_root": str(plan_root),
            "episode_plan": str(plan_path),
            "request_path": str(request_path),
            "ordinary_execution": execution,
        },
    )


def _ordinary_capture_executor(context: StageContext) -> StageOutcome:
    from avengine.dataset.binding_group_native import capture_visual_plan

    upstream = context.upstream.get("plan")
    if not isinstance(upstream, Mapping):
        raise ProductionRunError(f"{context.scope_id} capture has no passed plan upstream")
    outputs_up = upstream.get("outputs") or {}
    plan_path = Path(str((upstream.get("facts") or {}).get("episode_plan_path") or
                         outputs_up.get("episode_plan") or ""))
    request_path = Path(str(outputs_up.get("request_path") or ""))
    if not plan_path.is_file() or not request_path.is_file():
        raise StageOutputMissing(f"{context.scope_id} capture lacks plan/request upstream")
    root = context.output_root
    root.mkdir(parents=True, exist_ok=True)
    retained = _ordinary_retained_root(context)
    if retained is not None:
        paths = _ordinary_replay_paths(retained)
        capture = paths["capture"]
        output_root = retained
        visual_video = capture / "ue_visual_only.mp4"
        native_worlds = 0
        reused = str(capture.resolve().parent)
        request_path = paths["request"]
        plan_path = paths["plan"]
        frame_readbacks = capture / (
            "frame_readbacks.json" if (capture / "frame_readbacks.json").is_file()
            else "frame_records.json"
        )
    else:
        request = _ordinary_request_with_lease(
            context, _read_json(request_path)
        )
        # Capture consumes the passed plan's complete execution tree in its own
        # fresh attempt. Never write capture output into the historical plan tree.
        source_root = plan_path.parent.parent
        capture_episode = root / "episode"
        shutil.copytree(source_root, capture_episode, symlinks=True)
        # The M1 link is resolved relative to its case file. Keep this internal
        # reference relocatable when a passed plan moves to a capture attempt.
        copied_case = capture_episode / "plan" / "habitat_execution" / "case_manifest.json"
        copied_m1 = capture_episode / "plan" / "habitat_execution" / "m1_capture_request.json"
        if copied_case.is_file() and copied_m1.is_file():
            case = _read_json(copied_case)
            case["m1_request_path"] = copied_m1.name
            copied_case.write_text(json.dumps(case, ensure_ascii=False, indent=2) + "\n")
        plan_path = capture_episode / "plan" / "episode_plan.json"
        request_path = capture_episode / "request.json"
        request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n")
        captured = capture_visual_plan(
            request, capture_episode, label="ordinary_capture",
            log=root / "ordinary_capture.log",
        )
        capture = Path(captured["capture"]).resolve()
        output_root = Path(captured["output"]).resolve()
        visual_video = Path(captured["visual_video"]).resolve() if captured.get("visual_video") else None
        native_worlds = 1
        reused = None
        frame_readbacks = Path(captured["frame_readbacks"]).resolve()
    if retained is not None:
        from avengine.dataset.binding_group_native import check_requested_visibility
        check_requested_visibility(
            _read_json(plan_path), context.request, capture,
            report_path=root / "native_visibility_acceptance.json")
    count = captured_frame_count(capture)
    return StageOutcome(
        status="pass",
        facts={
            "capture_receipt_path": str((capture / "research_receipt.json").resolve()),
            "captured_frame_count": count,
        },
        outputs={
            "capture": str(capture),
            "capture_root": str(output_root),
            "episode_plan": str(plan_path),
            "request_path": str(request_path),
            "neutral_readback": str((capture / "neutral_readback.json").resolve()),
            "frame_readbacks": str(frame_readbacks),
            "visual_video": None if visual_video is None else str(visual_video),
            "native_visual_worlds_created": native_worlds,
            "reused_retained_visual_root": reused,
            "instance_runtime": context.runtime_readback(_read_json(request_path)),
        },
    )


def _ordinary_audio_layout_readback(
    finalized: Mapping[str, Any], facts: Mapping[str, Any]
) -> dict[str, Any]:
    """Read every declared audio mixture from the actual report and WAV files."""
    import numpy as np
    import soundfile as sf

    problems: list[str] = []
    report_path = Path(str(finalized.get("audio_report") or "")).resolve()
    report = _read_json(report_path) if report_path.is_file() else {}
    audio = report.get("audio") if isinstance(report, Mapping) else None
    audio = audio if isinstance(audio, Mapping) else {}
    clock = report.get("clock") if isinstance(report, Mapping) else None
    clock = clock if isinstance(clock, Mapping) else {}
    facts_audio = facts.get("audio") if isinstance(facts, Mapping) else None
    facts_audio = facts_audio if isinstance(facts_audio, Mapping) else {}
    facts_clock = facts.get("time") if isinstance(facts, Mapping) else None
    facts_clock = facts_clock if isinstance(facts_clock, Mapping) else {}
    layout_delivery = audio.get("layout_delivery")
    layout_delivery = layout_delivery if isinstance(layout_delivery, Mapping) else {}
    declared = finalized.get("declared_audio_delivery")
    declared = declared if isinstance(declared, Mapping) else {}
    layouts = declared.get("layouts")
    layouts = layouts if isinstance(layouts, list) else []
    readbacks: list[dict[str, Any]] = []

    for layout in layouts:
        if not isinstance(layout, Mapping):
            problems.append("declared audio layout is not an object")
            continue
        layout_type = str(layout.get("type") or "")
        entry = layout_delivery.get(layout_type)
        if not isinstance(entry, Mapping):
            problems.append(f"audio report has no layout_delivery[{layout_type!r}]")
            continue
        mixture = entry.get("mixture")
        mixture = mixture if isinstance(mixture, Mapping) else {}
        path_value = mixture.get("path")
        path = Path(path_value).expanduser().resolve() if isinstance(path_value, str) else None
        if path is None or not path.is_file():
            problems.append(f"{layout_type} mixture WAV is missing: {path_value!r}")
            continue
        try:
            pcm, sample_rate = sf.read(path, dtype="float64", always_2d=True)
        except (OSError, RuntimeError, ValueError) as exc:
            problems.append(f"{layout_type} mixture WAV is unreadable: {exc}")
            continue
        expected_channels = int(layout.get("channel_count") or entry.get("channel_count") or 0)
        expected_samples = int(clock.get("sample_count") or 0)
        if list(pcm.shape) != [expected_samples, expected_channels]:
            problems.append(
                f"{layout_type} WAV shape {list(pcm.shape)} != "
                f"[{expected_samples}, {expected_channels}]"
            )
        if int(sample_rate) != int(clock.get("sample_rate_hz") or 0):
            problems.append(
                f"{layout_type} WAV sample rate {sample_rate} != "
                f"{clock.get('sample_rate_hz')}"
            )
        finite = bool(np.isfinite(pcm).all())
        if not finite:
            problems.append(f"{layout_type} WAV contains nonfinite PCM")
        readback = {
            "layout_type": layout_type,
            "path": str(path),
            "channel_count": int(pcm.shape[1]),
            "sample_count": int(pcm.shape[0]),
            "sample_rate_hz": int(sample_rate),
            "finite": finite,
            "channel_labels": entry.get("channel_labels"),
            "peak_dbfs": mixture.get("peak_dbfs"),
        }
        if layout_type == "ambisonics":
            normalization = entry.get("foa_normalization")
            normalization = normalization if isinstance(normalization, Mapping) else {}
            acn_n3d_world = {
                "channel_order": entry.get("channel_order"),
                "coordinate_frame": entry.get("coordinate_frame"),
                "delivered_normalization": normalization.get("delivered_normalization"),
                "native_normalization": normalization.get("native_normalization"),
                "conversion": normalization.get("conversion"),
            }
            readback["acn_n3d_world"] = acn_n3d_world
            if acn_n3d_world["channel_order"] != "ACN":
                problems.append("ambisonics report is not ACN")
            if acn_n3d_world["coordinate_frame"] != "avengine_world":
                problems.append("ambisonics report is not avengine_world")
            if acn_n3d_world["delivered_normalization"] != "N3D":
                problems.append("ambisonics report is not delivered as N3D")
        readbacks.append(readback)

    for key in ("frame_count", "frame_rate_hz", "sample_rate_hz", "sample_count", "time_base_hz"):
        if facts_clock.get(key) != clock.get(key):
            problems.append(f"facts/report clock differs at {key}")
    intervals = facts_audio.get("wet_tail_intervals")
    duration_s = (
        float(clock["sample_count"]) / float(clock["sample_rate_hz"])
        if clock.get("sample_count") and clock.get("sample_rate_hz") else None
    )
    tail_readback = deepcopy(intervals) if isinstance(intervals, list) else []
    if not tail_readback:
        problems.append("audio facts have no wet_tail_intervals")
    else:
        for row in tail_readback:
            if not isinstance(row, Mapping):
                problems.append("wet_tail_intervals contains a non-object")
                continue
            start_s, end_s = row.get("start_s"), row.get("end_s")
            if (
                not isinstance(start_s, (int, float))
                or not isinstance(end_s, (int, float))
                or not (0 <= float(start_s) < float(end_s))
                or (duration_s is not None and float(end_s) > duration_s)
            ):
                problems.append(f"invalid wet tail interval: {row!r}")
    activity = facts_audio.get("source_activity_intervals_samples")
    if isinstance(activity, Mapping) and clock.get("sample_count") is not None:
        limit = int(clock["sample_count"])
        for event_id, rows in activity.items():
            if not isinstance(rows, list):
                problems.append(f"source activity for {event_id} is not a list")
                continue
            for row in rows:
                if (
                    not isinstance(row, Mapping)
                    or not isinstance(row.get("start_sample"), int)
                    or not isinstance(row.get("end_sample_exclusive"), int)
                    or not (0 <= row["start_sample"] < row["end_sample_exclusive"] <= limit)
                ):
                    problems.append(f"invalid source activity interval for {event_id}")
    return {
        "status": "pass" if not problems else "blocked",
        "problems": problems,
        "report_path": str(report_path),
        "receipt_path": str((report_path.parent / "audio" / "research_receipt.json").resolve()),
        "clock": deepcopy(dict(clock)),
        "layouts": readbacks,
        "tail_readback": tail_readback,
        "source_activity_coordinate_space": facts_audio.get("source_activity_coordinate_space"),
        "post_assembly_convolution_gain": audio.get("post_assembly_convolution_gain"),
    }


def _ordinary_audio_executor(context: StageContext) -> StageOutcome:
    from avengine.dataset.binding_group_native import (
        build_audio_assignment_plan,
        finalize_audio_assignment,
        materialize_audio_variant,
    )

    upstream = context.upstream.get("capture")
    if not isinstance(upstream, Mapping):
        raise ProductionRunError(f"{context.scope_id} audio has no passed capture upstream")
    capture_outputs = upstream.get("outputs") or {}
    capture = Path(str(capture_outputs.get("capture") or "")).resolve()
    plan_path = Path(str(capture_outputs.get("episode_plan") or "")).resolve()
    request_path = Path(str(capture_outputs.get("request_path") or "")).resolve()
    if not capture.is_dir() or not plan_path.is_file() or not request_path.is_file():
        raise StageOutputMissing(f"{context.scope_id} audio lacks capture/plan/request upstream")
    plan = _read_json(plan_path)
    request = _ordinary_request_with_lease(context, _read_json(request_path))
    options = context.recipe_options or {}
    context_world_id = context.request.get("world_id")
    if isinstance(context_world_id, str) and context_world_id.strip():
        request["world_id"] = context_world_id
    configured_world_id = options.get("world_id")
    if "world_id" not in request and isinstance(configured_world_id, str) and configured_world_id.strip():
        request["world_id"] = configured_world_id
    audio_overrides = options.get("audio_request_overrides")
    if isinstance(audio_overrides, Mapping):
        for field in ("audio_layouts", "foa_normalization"):
            if field in audio_overrides:
                request[field] = deepcopy(audio_overrides[field])
    # An ordinary Episode preserves each event's declared source. The
    # two-speaker a0/a1 permutations belong only to controlled core recipes.
    event_targets = {
        str(event["event_id"]): str(event["actor_id"])
        for event in plan.get("audio_events") or ()
    }
    assignment_plan, assignment_request = build_audio_assignment_plan(
        plan, request, "a0", assignment_targets={"a0": event_targets},
        require_authoritative_endpoints=False
    )
    retained = _ordinary_retained_root(context)
    render_retained_audio = bool(options.get("render_audio_from_retained_capture"))
    audio_report = (
        None
        if retained is not None and render_retained_audio
        else (_ordinary_audio_report(retained) if retained is not None else None)
    )
    root = context.output_root
    root.mkdir(parents=True, exist_ok=True)
    variant_root = materialize_audio_variant(
        {"capture": str(capture), "visual_video": capture_outputs.get("visual_video")},
        root / "episode",
        assignment_plan,
        assignment_request,
        member_id=context.scope_id,
    )
    finalized = finalize_audio_assignment(
        variant_root,
        assignment_request,
        audio_report=audio_report,
        verify_materialized=True,
    )
    facts_path = Path(finalized["facts"]).resolve()
    facts_value = _read_json(facts_path)
    intervals = ((facts_value.get("audio") or {}).get("wet_tail_intervals")
                 if isinstance(facts_value, Mapping) else None)
    if not isinstance(intervals, list) or not intervals:
        raise StageOutputMissing(f"ordinary audio published no wet_tail_intervals: {facts_path}")
    layout_readback = _ordinary_audio_layout_readback(finalized, facts_value)
    if layout_readback["status"] != "pass":
        raise StageOutputMissing(
            "ordinary audio layout readback failed: "
            + "; ".join(layout_readback["problems"])
        )
    return StageOutcome(
        status="pass",
        facts={
            "facts_path": str(facts_path),
            "audio_report_path": str(Path(finalized["audio_report"]).resolve()),
            "wet_tail_intervals": deepcopy(intervals),
            "declared_audio_delivery": deepcopy(finalized.get("declared_audio_delivery")),
            "delivered_audio_layouts": deepcopy(layout_readback),
            "audio_layout_readback": deepcopy(layout_readback),
            "audio_stage_elapsed_s": finalized.get("elapsed_s"),
        },
        outputs={
            "variant_root": str(variant_root),
            "audio": str(Path(finalized["audio"]).resolve()),
            "questions": str(Path(finalized["questions"]).resolve()),
            "preview": str(Path(finalized["preview"]).resolve()) if finalized.get("preview") else None,
            "visual_video": finalized.get("visual_video") or capture_outputs.get("visual_video"),
            "assignment": "a0",
            "audio_report": str(Path(finalized["audio_report"]).resolve()),
            "declared_audio_delivery": deepcopy(finalized.get("declared_audio_delivery")),
            "delivered_audio_layouts": deepcopy(layout_readback),
            "audio_layout_readback": deepcopy(layout_readback),
            "audio_stage_elapsed_s": finalized.get("elapsed_s"),
            "reused_retained_audio_report": None if audio_report is None else str(audio_report),
            "rendered_from_retained_capture": retained is not None and audio_report is None,
            "native_acoustic_contexts": 0,
        },
    )


def _ordinary_delivery_manifest_entry(
    row: Mapping[str, Any], episode_root: Path
) -> dict[str, Any]:
    """Give the ordinary validator its request-owned asset preallocation.

    Ordinary stage rows do not carry core-group source_assignments.  The
    request still owns the selected visual assets, so delivery validation must
    compare native facts with those IDs instead of treating an empty core
    assignment list as the ordinary allocation.  A written plan is the
    fallback for older rows whose request omitted the explicit list.
    """
    entry = deepcopy(dict(row))
    assignments = entry.get("source_assignments")
    has_assignments = isinstance(assignments, list) and bool(assignments)

    plan: Mapping[str, Any] = {}
    plan_path = Path(episode_root) / "plan" / "episode_plan.json"
    if plan_path.is_file():
        try:
            loaded_plan = _read_json(plan_path)
        except (OSError, TypeError, ValueError):
            loaded_plan = {}
        if isinstance(loaded_plan, Mapping):
            plan = loaded_plan

    # Preallocation uses declared instance IDs; native facts use runtime actor
    # IDs. Join through the plan's explicit association, preserving sound lists.
    if has_assignments:
        visual = plan.get("visual_plan") or {}
        actors = visual.get("actors") or []
        if isinstance(actors, Mapping):
            actors = list(actors.values())
        by_instance = {}
        by_runtime = {}
        for actor in actors:
            if not isinstance(actor, Mapping):
                continue
            runtime_id = actor.get("actor_id")
            instance_id = actor.get("entity_instance_id")
            if runtime_id:
                by_runtime[runtime_id] = actor
            if instance_id:
                if instance_id in by_instance:
                    raise ProductionRunError(f"ambiguous ordinary instance ID: {instance_id}")
                by_instance[instance_id] = actor
        normalized = []
        used_runtime_ids = set()
        for assignment in assignments:
            binding = deepcopy(dict(assignment))
            declared_id = binding.get("instance_id") or binding.get("actor_id")
            actor = by_instance.get(declared_id) or by_runtime.get(binding.get("actor_id"))
            if actor is not None:
                if actor.get("asset_id") != binding.get("asset_id"):
                    raise ProductionRunError(f"ordinary preallocation asset differs for {declared_id}")
                runtime_id = actor["actor_id"]
                if runtime_id in used_runtime_ids:
                    raise ProductionRunError(f"duplicate ordinary preallocation actor: {runtime_id}")
                used_runtime_ids.add(runtime_id)
                binding["declared_instance_id"] = declared_id
                binding["actor_id"] = runtime_id
            normalized.append(binding)
        entry["source_assignments"] = normalized

    asset_ids: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        if isinstance(value, str) and value and value not in seen:
            seen.add(value)
            asset_ids.append(value)

    if not has_assignments:
        request = entry.get("request")
        if isinstance(request, Mapping):
            for value in request.get("source_asset_ids") or ():
                add(value)
        if not asset_ids:
            visual = plan.get("visual_plan")
            actors = visual.get("actors") if isinstance(visual, Mapping) else None
            if isinstance(actors, Mapping):
                actors = list(actors.values())
            for actor in actors or ():
                if isinstance(actor, Mapping):
                    add(actor.get("asset_id"))
        entry["source_assignments"] = [
            {"actor_id": f"source{index}", "asset_id": asset_id}
            for index, asset_id in enumerate(asset_ids, start=1)
        ]

    profile = entry.get("requested_profile")
    planned_profile = plan.get("condition_profile")
    if isinstance(planned_profile, Mapping):
        merged_profile = deepcopy(dict(profile)) if isinstance(profile, Mapping) else {}
        for key, value in planned_profile.items():
            merged_profile.setdefault(key, deepcopy(value))
        entry["requested_profile"] = merged_profile
    return entry


def _ordinary_delivery_executor(context: StageContext) -> StageOutcome:
    from avengine.qa.batch_delivery import finalize_batch_episode

    upstream = context.upstream.get("audio")
    if not isinstance(upstream, Mapping):
        raise ProductionRunError(f"{context.scope_id} delivery has no passed audio upstream")
    variant_root = Path(str((upstream.get("outputs") or {}).get("variant_root") or "")).resolve()
    row = context.rows_by_episode_id.get(context.scope_id)
    if not isinstance(row, Mapping):
        raise ProductionRunError(f"ordinary Episode row is missing for {context.scope_id}")
    validation_row = _ordinary_delivery_manifest_entry(row, variant_root)
    review_root = context.output_root / "review"
    review = finalize_batch_episode(
        variant_root,
        validation_row,
        repository=context.repository,
        review_root=review_root,
    )
    facts_path = variant_root / "delivery" / "facts.json"
    questions_path = variant_root / "delivery" / "questions.json"
    status = "pass" if review.get("status") == "delivered" else "blocked"
    result = deepcopy(dict(review))
    world_id = result.get("world_id")
    if not isinstance(world_id, str) or not world_id.strip():
        request_world_id = context.request.get("world_id")
        world_id = request_world_id if isinstance(request_world_id, str) and request_world_id.strip() else None
    if not isinstance(world_id, str) or not world_id.strip():
        configured_world_id = (context.recipe_options or {}).get("world_id")
        world_id = configured_world_id if isinstance(configured_world_id, str) and configured_world_id.strip() else context.scope_id
    return StageOutcome(
        status=status,
        facts={
            "facts_path": str(facts_path.resolve()),
            "questions_path": str(questions_path.resolve()),
            "validation": {
                "status": "pass" if status == "pass" else "blocked",
                "review_status": review.get("status"),
            },
        },
        outputs={
            "delivered_episode": result,
            "episode_root": str(variant_root),
            "review_root": str(review_root.resolve()),
            "facts_path": str(facts_path.resolve()),
            "questions_path": str(questions_path.resolve()),
            "preview_path": review.get("preview_path"),
            "audio_path": (review.get("audio_level") or {}).get("path"),
            "episode_id": context.scope_id,
            "world_id": world_id,
            "room_family": review.get("room_family"),
            "room_id": review.get("room_id"),
        },
        reason=None if status == "pass" else str(review.get("reason") or review.get("failure_reason") or "ordinary delivery did not pass"),
    )


def _ordinary_stage_executor(context: StageContext) -> StageOutcome:
    runners = {
        "plan": _ordinary_plan_executor,
        "capture": _ordinary_capture_executor,
        "audio": _ordinary_audio_executor,
        "delivery": _ordinary_delivery_executor,
    }
    call = runners.get(context.unit_kind)
    if call is None:
        raise StageExecutorMissing(
            f"ordinary Episode unit kind {context.unit_kind!r} has no executor"
        )
    return call(context)


def _pending_recipe_executor(task_family: str, unit_kind: str) -> StageExecutor:
    seam = PENDING_RECIPE_SEAMS[task_family]

    def call(context: StageContext) -> StageOutcome:
        raise StageExecutorMissing(
            f"{task_family} unit kind {unit_kind!r} has no per-unit entry point yet; "
            f"{seam['owner']} owns {seam['module']} and still needs: "
            + "; ".join(seam["needed"]),
            task_family=task_family, unit_kind=unit_kind, owner=seam["owner"],
            module=seam["module"], needed=list(seam["needed"]),
        )

    return call


def register_default_stage_executors(*, replace_existing: bool = False) -> None:
    """Wire the recipes that really are callable per unit, and name the rest."""
    for family in sorted(NATIVE_RECIPE_BINDINGS):
        source = NATIVE_RECIPE_BINDINGS[family].module
        recipe = recipe_for_task_family(family)
        unit_kinds = {
            kind
            for unit in recipe.units
            for kind in (unit.unit_kind, unit.stage)
        }
        for unit_kind in sorted(unit_kinds):
            register_stage_executor(task_family=family, unit_kind=unit_kind,
                                    call=_native_group_unit_executor,
                                    source=source, replace_existing=replace_existing)
    for family in sorted(PENDING_RECIPE_SEAMS):
        for unit_kind in ("visual_plan", "visual_capture", "audio", "assembly"):
            register_stage_executor(
                task_family=family, unit_kind=unit_kind,
                call=_pending_recipe_executor(family, unit_kind),
                source=PENDING_RECIPE_SEAMS[family]["module"],
                replace_existing=replace_existing)
    for unit_kind in ("plan", "capture", "audio", "delivery"):
        register_stage_executor(
            task_family=None,
            unit_kind=unit_kind,
            call=_ordinary_stage_executor,
            source=f"{__name__}.ordinary",
            replace_existing=replace_existing,
        )


register_default_stage_executors()


# ---------------------------------------------------------------------------
# Reading the artifacts back
# ---------------------------------------------------------------------------


def captured_frame_count(capture_root: Path) -> int:
    """Count the frames a capture actually wrote, from its own readback.

    The recipe counts the camera rows of `neutral_readback.json`; this reads the
    same file, so `captured_frame_count` is checked against the artifact rather
    than taken on trust. A readback that only carries a frame list is accepted
    too, because that is what the older captures wrote.
    """
    neutral = Path(capture_root) / "neutral_readback.json"
    if neutral.is_file():
        value = _read_json(neutral)
        if isinstance(value, Mapping):
            rows = value.get("camera")
            if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) and rows:
                return len(rows)
            for key in ("frame_count", "captured_frame_count"):
                if isinstance(value.get(key), int):
                    return int(value[key])
    for name in ("frame_readbacks.json", "frame_records.json"):
        path = Path(capture_root) / name
        if not path.is_file():
            continue
        value = _read_json(path)
        rows = value.get("frames") if isinstance(value, Mapping) else value
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
            return len(rows)
        if isinstance(value, Mapping) and isinstance(value.get("frame_count"), int):
            return int(value["frame_count"])
    raise StageOutputMissing(
        f"no frame readback under {capture_root} reports a frame count")


def _path_like(key: str, value: Any) -> bool:
    return (isinstance(value, str) and value.strip()
            and any(key.endswith(suffix) for suffix in PATH_FACT_SUFFIXES))


def verify_stage_outputs(
    *,
    work_item: Mapping[str, Any],
    outcome: StageOutcome,
    upstream: Mapping[str, Mapping[str, Any]],
    output_root: Path,
    run_root: Path | None = None,
    retained_roots: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Check the unit's artifacts, not only what its result claims.

    Three things are checked, and each of them has failed in a real run before:
    every published fact that names a file exists and is non-empty; the values
    that carry a measurement agree with the file they were read from; and the
    artifacts descend from the upstream work item this round says is
    authoritative rather than from a superseded attempt.
    """
    stage = str(work_item["stage"])
    checks: list[dict[str, Any]] = []
    problems: list[str] = []

    def check(name: str, ok: bool, detail: Any = None) -> None:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
        if not ok:
            problems.append(f"{name}: {detail}")

    required = STAGE_PUBLISHED_FACTS.get(stage, ())
    missing = [key for key in required
               if key not in outcome.facts or outcome.facts[key] in (None, "", [], {})]
    check("published_facts_present", not missing, {"missing": missing, "required": list(required)})

    for key, value in sorted(outcome.facts.items()):
        if not _path_like(key, value):
            continue
        path = Path(str(value))
        exists = path.exists()
        size = path.stat().st_size if path.is_file() else None
        check(f"fact_artifact_exists:{key}", exists and (size is None or size > 0),
              {"path": str(path), "exists": exists, "size_bytes": size})

    for key, value in sorted(outcome.outputs.items()):
        if not _path_like(key, value):
            continue
        path = Path(str(value))
        check(f"output_artifact_exists:{key}", path.exists(), {"path": str(path)})

    if stage == "capture":
        capture_root = Path(str(outcome.outputs.get("capture")
                                or outcome.outputs.get("capture_root") or ""))
        counted = None
        if capture_root.is_dir():
            try:
                counted = captured_frame_count(capture_root)
            except StageOutputMissing as error:
                counted = None
                check("captured_frame_count_readable", False, str(error))
        declared = outcome.facts.get("captured_frame_count")
        check("captured_frame_count_matches_readback", counted is not None and counted == declared,
              {"declared": declared, "read_back": counted, "capture_root": str(capture_root)})

    if stage == "audio":
        facts_path = Path(str(outcome.facts.get("facts_path") or ""))
        measured = None
        if facts_path.is_file():
            value = _read_json(facts_path)
            if isinstance(value, Mapping):
                measured = (value.get("audio") or {}).get("wet_tail_intervals")
        declared = outcome.facts.get("wet_tail_intervals")
        check("wet_tail_intervals_match_facts_file", bool(measured) and measured == declared,
              {"declared_count": len(declared or []),
               "measured_count": len(measured or []),
               "facts_path": str(facts_path)})
        ends = [row.get("end_s") for row in (declared or []) if isinstance(row, Mapping)]
        check("wet_tail_intervals_carry_end_s",
              bool(ends) and all(isinstance(value, (int, float)) for value in ends),
              {"end_s": ends})

    authoritative = {str(result.get("work_item_id")) for result in upstream.values()}
    declared_depends = {str(value) for value in (work_item.get("depends_on") or [])}
    check("upstream_is_the_authoritative_round",
          declared_depends <= authoritative or not declared_depends,
          {"work_item_depends_on": sorted(declared_depends),
           "authoritative_upstream": sorted(authoritative)})

    # Where a unit's artifacts are allowed to live: its own fresh attempt, the
    # rest of this run (an upstream plan, a shared evidence pack), or a retained
    # root the configuration named. Anywhere else means an artifact wandered
    # outside the run and is not accounted for by it.
    attempt_root = Path(output_root).resolve()
    allowed = [attempt_root]
    if run_root is not None:
        allowed.append(Path(run_root).resolve())
    allowed.extend(Path(str(value)).resolve() for value in retained_roots)

    def located(path: Path) -> str | None:
        for base in allowed:
            try:
                path.relative_to(base)
            except ValueError:
                continue
            return str(base)
        return None

    outside = []
    inside_attempt = 0
    for source, items in (("facts", outcome.facts), ("outputs", outcome.outputs)):
        for key, value in sorted(items.items()):
            if not _path_like(key, value):
                continue
            resolved = Path(str(value)).resolve()
            base = located(resolved)
            if base is None:
                outside.append({"source": source, "key": key, "path": str(value)})
            elif source == "facts" and base == str(attempt_root):
                inside_attempt += 1
    check("artifacts_stay_inside_the_run_or_a_retained_root", not outside,
          {"outside": outside, "allowed_roots": [str(base) for base in allowed]})
    checks.append({"check": "published_facts_inside_this_attempt", "passed": True,
                   "detail": {"count": inside_attempt,
                              "note": ("a unit that linked a retained visual publishes "
                                       "facts under that retained root instead")}})

    return {
        "verified": not problems,
        "stage": stage,
        "work_item_id": work_item.get("work_item_id"),
        "output_root": str(output_root),
        "checks": checks,
        "problems": problems,
    }


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@dataclass
class LeaseHold:
    lease: Lease
    plan: dict[str, Any]

    @property
    def rpc_port(self) -> int | None:
        return self.lease.rpc_port

    @property
    def graphics_adapter(self) -> int | None:
        return self.lease.device_index


RESOURCE_POLICY_SECTIONS = (
    "cpu", "gpu", "ports", "queue", "wait", "backends",
    "backend_by_runtime_profile",
)


def merge_resource_policy(
    base: Mapping[str, Any] | None, override: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Lay a task-local override over the configuration's own policy.

    The merge is one level deep inside each known section, because that is
    what an override is for: raising one floor, or naming one display process
    the machine happens to run, without restating the backend table. A list
    replaces a list rather than growing one, so a task that declares its
    display exceptions declares all of them and cannot silently inherit a
    broader set it never read.
    """

    merged: dict[str, Any] = deepcopy(dict(base or {}))
    for key, value in dict(override or {}).items():
        if (
            key in RESOURCE_POLICY_SECTIONS
            and isinstance(value, Mapping)
            and isinstance(merged.get(key), Mapping)
        ):
            section = dict(merged[key])
            section.update(deepcopy(dict(value)))
            merged[key] = section
        else:
            merged[key] = deepcopy(value)
    return merged


class ResourceBroker:
    """Submit, pump, recheck, bind and release, using the P18 allocator only.

    Nothing here reaches into the allocator's tables. Recovery goes through
    `restore_from_snapshot` with this run's own liveness test, so a lease whose
    worker is gone gives its memory and its port back instead of being adopted.
    """

    def __init__(
        self,
        allocator: ResourceAllocator | None = None,
        *,
        python_executable: str | None = None,
        repository: Path = REPOSITORY,
        python_path: Sequence[str] = DEFAULT_PYTHON_PATH,
        wait_s: float = DEFAULT_LEASE_WAIT_S,
        poll_s: float = DEFAULT_LEASE_POLL_S,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.allocator = allocator or ResourceAllocator(
            ResourcePolicy(backends=backend_profiles(None)))
        self.python_executable = python_executable or sys.executable
        self.repository = Path(repository)
        self.python_path = tuple(str(item) for item in python_path)
        self.wait_s = float(wait_s)
        self.poll_s = float(poll_s)
        self._sleep = sleep or time.sleep
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        # Ids this broker submitted. A grant for one of them that is no longer
        # wanted -- a claim left queued by an expired wait -- is returned here
        # rather than held; a grant for anything else is not this broker's.
        self._outstanding: set[str] = set()
        self._granted: dict[str, Lease] = {}

    def request_for(
        self, work_item: Mapping[str, Any], *, compatibility: WorkerCompatibility,
        estimated_peak_vram_mb: int | None = None,
    ):
        """Build the claim. A start-up floor is never reused as a peak estimate."""
        return lease_request_for_work_item(
            work_item, policy=self.allocator.policy, compatibility=compatibility,
            estimated_peak_vram_mb=estimated_peak_vram_mb,
        )

    def acquire(
        self, work_item: Mapping[str, Any], *, compatibility: WorkerCompatibility,
        entry: Sequence[str], environment: Mapping[str, str] | None = None,
        estimated_peak_vram_mb: int | None = None,
    ) -> tuple[LeaseHold | None, dict[str, Any]]:
        """Queue one claim, pump until it is granted, recheck, then hand it over."""
        request = self.request_for(
            work_item,
            compatibility=compatibility,
            estimated_peak_vram_mb=estimated_peak_vram_mb,
        )
        with self._lock:
            self.allocator.submit(request)
            self._outstanding.add(request.lease_id)
        deadline = self._clock() + self.wait_s
        last: dict[str, Any] = {
            "status": "wait",
            "reason_code": "not_pumped",
            "reason": "",
        }
        while True:
            try:
                with self._lock:
                    mine = self._granted.pop(request.lease_id, None)
                    granted, deferred = self.allocator.pump()
                    for lease in granted:
                        if lease.lease_id == request.lease_id:
                            mine = lease
                        elif lease.lease_id in self._outstanding:
                            # Another acquisition may have been granted by this
                            # pump. Keep it for that caller instead of releasing
                            # a valid lease behind its back.
                            self._granted[lease.lease_id] = lease
                        else:
                            self.allocator.release(lease)
                    for decision in deferred:
                        last = decision.to_dict()
            except Exception as error:
                with self._lock:
                    self._outstanding.discard(request.lease_id)
                return None, {
                    "status": "blocked",
                    "reason_code": "allocator_error",
                    "reason": f"{type(error).__name__}: {error}",
                    "lease_id": request.lease_id,
                    "requirement": request.requirement.to_dict(),
                }
            if mine is not None:
                recheck = self.allocator.recheck_before_launch(mine)
                if not recheck.granted:
                    self.allocator.release(mine)
                    last = recheck.to_dict()
                    if recheck.status == "blocked" or self._clock() >= deadline:
                        with self._lock:
                            self._outstanding.discard(request.lease_id)
                        return None, last
                    with self._lock:
                        self.allocator.submit(request)
                    self._sleep(self.poll_s)
                    continue
                plan = worker_launch_plan(
                    mine,
                    entry=entry,
                    repository_root=self.repository,
                    python_path=self.python_path,
                    environment=environment,
                    rlr_threads=request.rlr_threads,
                )
                with self._lock:
                    self._outstanding.discard(request.lease_id)
                return LeaseHold(lease=mine, plan=plan), recheck.to_dict()
            with self._lock:
                queued = request.lease_id in self.allocator.queued_lease_ids()
            if not queued:
                if (
                    last.get("status") == "wait"
                    and (last.get("detail") or {}).get("terminal")
                ):
                    last = {**last, "status": "blocked"}
                with self._lock:
                    self._outstanding.discard(request.lease_id)
                return None, last
            if self._clock() >= deadline:
                with self._lock:
                    self._outstanding.discard(request.lease_id)
                return None, {
                    **last,
                    "status": "wait",
                    "reason_code": "lease_wait_expired",
                    "reason": f"waited {self.wait_s:g}s for {request.lease_id}",
                }
            self._sleep(self.poll_s)

    def bind(self, lease_id: str, pid: int) -> None:
        self.allocator.bind_worker_pid(lease_id, int(pid), include_descendants=True)

    def release(self, lease: Lease | str) -> None:
        self.allocator.release(lease)

    def snapshot(self) -> dict[str, Any]:
        return self.allocator.snapshot()

    def restore(self, snapshot: Mapping[str, Any],
                is_running: Callable[[int], bool]) -> dict[str, Any]:
        return self.allocator.restore_from_snapshot(snapshot, is_running=is_running)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


@dataclass
class ScopeState:
    """One schedulable scope: an ordinary Episode row, or one core group."""

    scope_kind: str
    scope_key: str
    room_id: str
    task_family: str | None = None
    row: Mapping[str, Any] | None = None
    results: list[dict[str, Any]] = field(default_factory=list)
    blockers: list[dict[str, Any]] = field(default_factory=list)
    candidate_index: int = 0
    interrupted_attempts: dict[str, int] = field(default_factory=dict)
    candidate_failures: list[dict[str, Any]] = field(default_factory=list)
    finished: bool = False
    finished_reason: str | None = None
    # How many times each unit has been explicitly re-opened after its input
    # was repaired, and the failures that were set aside to do it. The
    # offset shifts the next attempt number so the repaired run writes a
    # fresh no-clobber root and the failed attempt stays on disk.
    retry_offsets: dict[str, int] = field(default_factory=dict)
    retired_failures: list[dict[str, Any]] = field(default_factory=list)
    retired_results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope_kind": self.scope_kind, "scope_key": self.scope_key,
            "room_id": self.room_id, "task_family": self.task_family,
            "results": deepcopy(self.results), "blockers": deepcopy(self.blockers),
            "candidate_index": self.candidate_index,
            "interrupted_attempts": dict(self.interrupted_attempts),
            "candidate_failures": deepcopy(self.candidate_failures),
            "finished": self.finished, "finished_reason": self.finished_reason,
            "retry_offsets": dict(self.retry_offsets),
            "retired_failures": deepcopy(self.retired_failures),
            "retired_results": deepcopy(self.retired_results),
        }


class ProductionRunner:
    """Drive one manifest to delivered groups, and pick the run back up.

    The loop is: ask the protocol what may run, lease it, run it in a fresh
    interpreter, read the artifacts back, file the result, persist, repeat.
    """

    def __init__(
        self,
        *,
        manifest: Mapping[str, Any],
        manifest_path: str | Path,
        run_root: str | Path,
        repository: str | Path = REPOSITORY,
        broker: ResourceBroker | None = None,
        resource_policy_override: Mapping[str, Any] | None = None,
        python_executable: str | None = None,
        python_path: Sequence[str] = DEFAULT_PYTHON_PATH,
        group_ids: Sequence[str] | None = None,
        episode_ids: Sequence[str] | None = None,
        native_visual_world_budget: int | None = None,
        candidate_rotation_limit: int = DEFAULT_CANDIDATE_ROTATIONS,
        interrupted_retry_limit: int = DEFAULT_INTERRUPTED_RETRIES,
        recipe_options: Mapping[str, Mapping[str, Any]] | None = None,
        max_waves: int = DEFAULT_MAX_WAVES,
        max_parallel: int = 1,
        launcher: Callable[..., Any] | None = None,
        executor_override: StageExecutor | None = None,
        room_resolver: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
        argv: Sequence[str] | None = None,
    ) -> None:
        self.manifest = deepcopy(dict(manifest))
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.run_root = Path(run_root).expanduser().resolve()
        self.repository = Path(repository).expanduser().resolve()
        self.python_executable = python_executable or sys.executable
        self.python_path = tuple(str(item) for item in python_path)
        manifest_policy = (
            self.manifest.get("resource_policy")
            or (self.manifest.get("production") or {}).get("resource_policy")
        )
        self.resource_policy_override = (
            None if resource_policy_override is None
            else deepcopy(dict(resource_policy_override))
        )
        if self.resource_policy_override is not None and broker is not None:
            raise ResourcePolicyError(
                "a task-local resource policy override and a ready-made broker "
                "are two different sources for the same policy; pass one. The "
                "override builds the broker so the run records what it adopted."
            )
        if broker is None and (
            manifest_policy is not None or self.resource_policy_override is not None
        ):
            if isinstance(manifest_policy, ResourcePolicy):
                if self.resource_policy_override is not None:
                    raise ResourcePolicyError(
                        "the manifest supplies a built ResourcePolicy object, "
                        "which an override cannot be merged into; supply the "
                        "manifest policy as a mapping to override it"
                    )
                policy = manifest_policy
                merged: Mapping[str, Any] | None = None
            else:
                merged = merge_resource_policy(
                    manifest_policy, self.resource_policy_override
                )
                policy = ResourcePolicy.from_mapping(merged)
            broker = ResourceBroker(
                allocator=ResourceAllocator(policy=policy),
                python_executable=self.python_executable,
                repository=self.repository,
                python_path=self.python_path,
            )
            if self.resource_policy_override is None:
                self.resource_policy_source = "manifest"
            elif manifest_policy is None:
                self.resource_policy_source = "task_override"
            else:
                self.resource_policy_source = "manifest+task_override"
        else:
            self.resource_policy_source = (
                "provided_broker" if broker is not None else "default"
            )
        self.broker = broker or ResourceBroker(
            python_executable=self.python_executable, repository=self.repository,
            python_path=self.python_path)
        self.native_visual_world_budget = native_visual_world_budget
        self.candidate_rotation_limit = int(candidate_rotation_limit)
        self.interrupted_retry_limit = int(interrupted_retry_limit)
        self.recipe_options = {str(key): deepcopy(dict(value))
                               for key, value in dict(recipe_options or {}).items()}
        self.max_waves = int(max_waves)
        self.max_parallel = max(1, int(max_parallel))
        self.launcher = launcher or self._launch_worker
        self.executor_override = executor_override
        self.room_resolver = room_resolver or _resolve_room_runtime_report
        self.argv = [str(item) for item in (argv if argv is not None else sys.argv)]
        self.rows_by_episode_id = {
            str(row["episode_id"]): deepcopy(dict(row))
            for row in self.manifest.get("episodes", [])
        }
        self.scopes = self._build_scopes(group_ids, episode_ids)
        self.native_visual_worlds_used = 0
        self.native_acoustic_contexts_used = 0
        self.events_path = self.run_root / "journal.jsonl"
        self.state_path = self.run_root / "state.json"
        self._event_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._result_lock = threading.RLock()
        self._accounting_lock = threading.RLock()
        self._worker_records: dict[str, dict[str, Any]] = {}
        # One durable record per work_item_id. A native attempt is counted
        # before its launcher is called, then finalized in place.
        self._native_accounting: dict[str, dict[str, Any]] = {}
        self._native_accounting_baseline_visual = 0
        self._native_accounting_baseline_acoustic = 0
        self._resumed: dict[str, Any] = {}

    # -- construction ---------------------------------------------------

    def _build_scopes(self, group_ids, episode_ids) -> list[ScopeState]:
        wanted_groups = None if group_ids is None else {str(value) for value in group_ids}
        wanted_episodes = None if episode_ids is None else {str(value) for value in episode_ids}
        scopes: list[ScopeState] = []
        seen_groups: set[str] = set()
        for row in self.manifest.get("episodes", []):
            episode_id = str(row["episode_id"])
            group_id = row.get("group_id")
            if group_id:
                group_id = str(group_id)
                if group_id in seen_groups:
                    continue
                if wanted_groups is not None and group_id not in wanted_groups:
                    continue
                if wanted_groups is None and wanted_episodes is not None:
                    continue
                seen_groups.add(group_id)
                scopes.append(ScopeState(
                    scope_kind="core_group", scope_key=group_id,
                    room_id=str(row.get("room_id") or ""),
                    task_family=str(row.get("task_family") or "")))
                continue
            if wanted_episodes is not None and episode_id not in wanted_episodes:
                continue
            if wanted_episodes is None and wanted_groups is not None:
                continue
            scopes.append(ScopeState(
                scope_kind="episode", scope_key=episode_id,
                room_id=str(row.get("room_id") or ""), row=deepcopy(dict(row))))
        return scopes

    # -- persistence ----------------------------------------------------

    def _append_event(self, event: Mapping[str, Any]) -> None:
        line = json.dumps({"timestamp": _utc_now(), **dict(event)},
                          ensure_ascii=False, sort_keys=True) + "\n"
        with self._event_lock:
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a", encoding="utf-8") as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())

    def state(self) -> dict[str, Any]:
        """The saved run, with only the currently valid round marked as done."""
        native_totals = self._refresh_native_accounting_totals()
        rounds = {}
        for scope in self.scopes:
            try:
                rounds[scope.scope_key] = self._round_state(scope)
            except (ProductionSpecError, ValueError) as error:
                rounds[scope.scope_key] = {"error": f"{type(error).__name__}: {error}"}
        return {
            "schema": SCHEMA,
            "run_root": str(self.run_root),
            "manifest_path": str(self.manifest_path),
            "repository": str(self.repository),
            "batch_id": self.manifest.get("batch_id"),
            "updated_at": _utc_now(),
            # P19 may append bounded config-derived requests after the original
            # scopes finish. Keep that effective manifest in the state so a
            # later resume does not silently fall back to the old input file.
            "manifest_snapshot": deepcopy(self.manifest),
            "scopes": [scope.to_dict() for scope in self.scopes],
            "valid_round": rounds,
            "native_visual_worlds_used": self.native_visual_worlds_used,
            "native_acoustic_contexts_used": self.native_acoustic_contexts_used,
            "native_visual_world_budget": self.native_visual_world_budget,
            "max_parallel": self.max_parallel,
            "native_accounting_totals": native_totals,
            "native_accounting": deepcopy(self._native_accounting),
            "native_accounting_carryover": {
                "native_visual_worlds": self._native_accounting_baseline_visual,
                "native_acoustic_contexts": self._native_accounting_baseline_acoustic,
            },
            "resource_policy_source": self.resource_policy_source,
            "resource_policy": self.broker.allocator.policy.to_dict(),
            "resource_policy_override": deepcopy(self.resource_policy_override),
            "resource_snapshot": self.broker.snapshot(),
            "worker_records": deepcopy(self._worker_records),
            "recipe_options": deepcopy(self.recipe_options),
            "resumed": deepcopy(self._resumed),
            "stage_executors": registered_stage_executors(),
        }

    def _persist(self) -> None:
        with self._state_lock:
            _json_write_atomic(self.state_path, self.state())

    def _native_attempt_descriptor(
        self,
        scope: ScopeState,
        work_item: Mapping[str, Any],
        retained_for_unit: str | None,
    ) -> dict[str, Any]:
        """Describe one native startup without changing budget definitions."""
        payload = work_item.get("payload") or {}
        resource = work_item.get("resource") or {}
        unit_kind = str(payload.get("unit_kind") or work_item.get("stage") or "")
        fresh_visual = int(
            unit_kind in {"visual_capture", "capture"} and retained_for_unit is None
        )
        native_acoustic_launch_attempts = int(
            unit_kind == "audio"
            and str(resource.get("runtime_context") or "") == "rlr_native"
        )
        world_id = None
        group_id = work_item.get("group_id")
        if fresh_visual and group_id:
            world_id = declared_world_id(self.manifest, str(group_id))
        request = (work_item.get("inputs") or {}).get("request")
        if not isinstance(request, Mapping):
            row = scope.row or self.rows_by_episode_id.get(scope.scope_key) or {}
            request = row.get("request") if isinstance(row, Mapping) else None
        if fresh_visual and world_id is None and isinstance(request, Mapping):
            value = request.get("world_id")
            if value is not None and str(value).strip():
                world_id = str(value)
        if not fresh_visual and world_id is None and isinstance(request, Mapping):
            value = request.get("world_id")
            if value is not None and str(value).strip():
                world_id = str(value)
        return {
            "logical_world_id": world_id,
            # The charge unit for logical worlds is not defined by the
            # capture counter. Keep the distinct identity and leave attempts
            # unknown until an explicit world-attempt field is supplied.
            "logical_world_attempts": None,
            "logical_world_attempts_source": "budget_unit_pending",
            "capture_instances": fresh_visual,
            "native_visual_worlds": fresh_visual,
            "native_acoustic_launch_attempts": native_acoustic_launch_attempts,
            "retained_visual_root": retained_for_unit,
        }

    def _native_accounting_totals(self) -> dict[str, Any]:
        totals = {
            "logical_world_attempts": None,
            "logical_world_attempts_known": 0,
            "logical_world_attempts_unknown_capture_records": 0,
            "actual_native_visual_worlds_known": 0,
            "native_visual_attempts_with_unknown_actual": 0,
            "logical_world_identities": 0,
            "capture_instances": 0,
            "native_visual_worlds": 0,
            "native_acoustic_launch_attempts": 0,
            "native_acoustic_contexts_known": 0,
            "native_acoustic_contexts_lower_bound": 0,
            "native_acoustic_contexts_unknown_attempts": 0,
        }
        logical_attempts = 0
        logical_attempts_unknown = 0
        world_ids: set[str] = set()
        for record in self._native_accounting.values():
            logical_value = record.get("logical_world_attempts")
            if isinstance(logical_value, int) and not isinstance(logical_value, bool):
                logical_attempts += int(logical_value)
            elif record.get("capture_instances") or record.get(
                "expected_native_visual_worlds"
            ):
                logical_attempts_unknown += 1
            totals["capture_instances"] += int(
                record.get("effective_capture_instances")
                if record.get("effective_capture_instances") is not None
                else record.get("capture_instances") or 0
            )
            totals["native_visual_worlds"] += int(
                record.get("effective_native_visual_worlds")
                if record.get("effective_native_visual_worlds") is not None
                else record.get("expected_native_visual_worlds") or 0
            )
            launch_attempts = int(
                record.get("native_acoustic_launch_attempts")
                if record.get("native_acoustic_launch_attempts") is not None
                else record.get("expected_native_acoustic_contexts") or 0
            )
            totals["native_acoustic_launch_attempts"] += launch_attempts
            actual = record.get("actual_native_acoustic_contexts")
            if actual is None:
                if launch_attempts:
                    totals["native_acoustic_contexts_unknown_attempts"] += 1
            else:
                totals["native_acoustic_contexts_known"] += int(actual)
                totals["native_acoustic_contexts_lower_bound"] += int(actual)
            world_id = record.get("logical_world_id")
            if world_id:
                world_ids.add(str(world_id))
        actual_visual_known = 0
        actual_visual_unknown = 0
        for record in self._native_accounting.values():
            if not int(record.get("effective_native_visual_worlds") or 0):
                continue
            value = record.get("actual_native_visual_worlds")
            if isinstance(value, int):
                actual_visual_known += value
            else:
                actual_visual_unknown += 1
        totals["actual_native_visual_worlds_known"] = actual_visual_known
        totals["native_visual_attempts_with_unknown_actual"] = actual_visual_unknown
        totals["logical_world_attempts_known"] = logical_attempts
        totals["logical_world_attempts_unknown_capture_records"] = (
            logical_attempts_unknown
        )
        if logical_attempts_unknown == 0:
            totals["logical_world_attempts"] = logical_attempts
        totals["logical_world_identities"] = len(world_ids)
        return totals

    def _refresh_native_accounting_totals(self) -> dict[str, Any]:
        with self._accounting_lock:
            return self._refresh_native_accounting_totals_unlocked()

    def _refresh_native_accounting_totals_unlocked(self) -> dict[str, Any]:
        totals = self._native_accounting_totals()
        self.native_visual_worlds_used = (
            self._native_accounting_baseline_visual
            + int(totals["native_visual_worlds"])
        )
        self.native_acoustic_contexts_used = (
            self._native_accounting_baseline_acoustic
            + int(totals["native_acoustic_contexts_known"])
        )
        return totals

    def _ensure_native_attempt(
        self,
        scope: ScopeState,
        work_item: Mapping[str, Any],
        retained_for_unit: str | None,
    ) -> dict[str, Any] | None:
        # The state file is written after this lock is released, never
        # under it. `_persist` takes the state lock and then, inside
        # `state()`, the accounting lock; finalizing takes them the other way
        # round. Two workers finishing at once each held one and waited for
        # the other, and the whole controller stopped with every thread
        # parked. One order, kept in one place, is the fix.
        with self._accounting_lock:
            record = self._ensure_native_attempt_unlocked(
                scope, work_item, retained_for_unit, persist=False
            )
        self._persist()
        return record

    def _ensure_native_attempt_unlocked(
        self,
        scope: ScopeState,
        work_item: Mapping[str, Any],
        retained_for_unit: str | None,
        *,
        persist: bool = True,
    ) -> dict[str, Any] | None:
        """Persist and count a native startup exactly once for this work item."""
        item_id = str(work_item["work_item_id"])
        existing = self._native_accounting.get(item_id)
        if existing is not None:
            return existing
        descriptor = self._native_attempt_descriptor(
            scope, work_item, retained_for_unit
        )
        if not (
            descriptor["native_visual_worlds"]
            or descriptor["native_acoustic_launch_attempts"]
        ):
            return None
        record = {
            "schema": "avengine_native_attempt_accounting_v1",
            "work_item_id": item_id,
            "stage": str(work_item.get("stage") or ""),
            "unit_kind": str(
                (work_item.get("payload") or {}).get("unit_kind")
                or work_item.get("stage")
                or ""
            ),
            "attempt": int(work_item.get("attempt") or 0),
            "logical_world_id": descriptor["logical_world_id"],
            "logical_world_attempts": descriptor["logical_world_attempts"],
            "logical_world_attempts_source": descriptor[
                "logical_world_attempts_source"
            ],
            "capture_instances": int(descriptor["capture_instances"]),
            "expected_native_visual_worlds": int(
                descriptor["native_visual_worlds"]
            ),
            "native_acoustic_launch_attempts": int(
                descriptor["native_acoustic_launch_attempts"]
            ),
            "effective_capture_instances": int(descriptor["capture_instances"]),
            "effective_native_visual_worlds": int(
                descriptor["native_visual_worlds"]
            ),
            "reported_native_visual_worlds": None,
            "reported_native_acoustic_contexts": None,
            "actual_native_acoustic_contexts": None,
            "actual_native_acoustic_contexts_source": None,
            "actual_native_acoustic_contexts_evidence_path": None,
            "native_acoustic_contexts_lower_bound": 0,
            "worker_logs": {},
            "retained_visual_root": descriptor["retained_visual_root"],
            "status": "started",
            "started_at": _utc_now(),
            "counted": True,
        }
        self._native_accounting[item_id] = record
        self._refresh_native_accounting_totals()
        self._append_event({
            "event": "native_attempt_started",
            "work_item_id": item_id,
            "native_accounting": deepcopy(record),
        })
        if persist:
            self._persist()
        return record

    def _remember_native_worker_logs(
        self,
        work_item_id: str,
        run_record: Mapping[str, Any] | None,
    ) -> None:
        record = self._native_accounting.get(str(work_item_id))
        if record is None or not isinstance(run_record, Mapping):
            return
        logs = dict(record.get("worker_logs") or {})
        changed = False
        for key, output_key in (("stdout", "stdout_log"), ("stderr", "stderr_log")):
            value = run_record.get(output_key)
            if isinstance(value, str) and value.strip() and logs.get(key) != value:
                logs[key] = value
                changed = True
        if changed:
            record["worker_logs"] = logs
            self._persist()

    @staticmethod
    def _actual_native_visual_worlds(
        outcome: StageOutcome | None,
    ) -> tuple[int | None, str | None, str | None]:
        """Did a world actually open, as opposed to being charged for one?

        The charge is decided before the launcher is called, because a
        failure after launch must cost what it cost. Whether anything opened
        is a different question with a different answer, and the two were
        being read off one number: a capture that was refused by a CPU
        precondition looked exactly like a capture that started and crashed.

        The evidence is the producer's own capture receipt. Present and
        proving a native start, the answer is one. An attempt root that
        exists with no receipt in it is a real zero. No attempt root at all
        is unknown, and stays unknown rather than being written down as zero.
        """

        if outcome is None:
            return None, None, None
        roots: list[Path] = []
        for source in (outcome.outputs, outcome.facts, outcome.diagnostics):
            for key in ("attempt_root", "output_root", "capture_root", "capture"):
                value = source.get(key)
                if isinstance(value, str) and value.strip():
                    roots.append(Path(value))
        for key in ("capture_receipt_path",):
            value = outcome.facts.get(key)
            if isinstance(value, str) and value.strip():
                receipt = Path(value)
                if receipt.is_file():
                    return 1, "capture_receipt", str(receipt)
        seen: set[str] = set()
        for root in roots:
            resolved = str(root)
            if resolved in seen:
                continue
            seen.add(resolved)
            for candidate in (root / "capture" / "research_receipt.json",
                              root / "research_receipt.json"):
                if candidate.is_file():
                    return 1, "capture_receipt", str(candidate)
        existing = [root for root in roots if root.is_dir()]
        if existing:
            return 0, "attempt_root_without_capture_receipt", str(existing[0])
        return None, None, None

    @staticmethod
    def _actual_native_acoustic_contexts(
        outcome: StageOutcome | None,
        run_record: Mapping[str, Any] | None = None,
    ) -> tuple[int | None, str | None, str | None]:
        """Read one actual count from this item's own evidence only."""
        sources = ()
        if outcome is not None:
            sources = (("outcome_output", outcome.outputs),
                       ("outcome_diagnostic", outcome.diagnostics))
        for source_name, source in sources:
            for key in (
                "actual_native_acoustic_contexts",
                "native_acoustic_contexts_actual",
                "native_context_count",
                "create_context_count",
                "native_contexts_created",
            ):
                value = source.get(key)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    return int(value), source_name, None

        paths: list[tuple[Path, str]] = []
        if outcome is not None:
            for source in (outcome.outputs, outcome.facts):
                for key in ("output_root", "variant_root"):
                    value = source.get(key)
                    if not isinstance(value, str) or not value.strip():
                        continue
                    candidate = Path(value)
                    # Only the explicitly published stage root and its known
                    # delivery leaf are admissible. Never walk ancestors.
                    paths.append((candidate / "audio.log", "stage_leaf_log"))
                    paths.append((candidate / "delivery" / "audio.log",
                                  "stage_leaf_log"))
        if run_record is not None:
            for key in ("stdout_log", "stderr_log"):
                value = run_record.get(key)
                if isinstance(value, str) and value.strip():
                    paths.append((Path(value), "worker_log"))

        seen: set[str] = set()
        for path, source_name in paths:
            try:
                resolved = str(path.resolve())
            except OSError:
                resolved = str(path)
            if resolved in seen or not path.is_file():
                continue
            seen.add(resolved)
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            count = text.count("CreateContext: Context created")
            if count:
                return count, source_name, resolved
        return None, None, None

    def _finalize_native_attempt(
        self,
        work_item_id: str,
        *,
        status: str,
        outcome: StageOutcome | None = None,
        reason: str | None = None,
        run_record: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        with self._accounting_lock:
            record = self._finalize_native_attempt_unlocked(
                work_item_id,
                persist=False,
                status=status,
                outcome=outcome,
                reason=reason,
                run_record=run_record,
            )
        self._persist()
        return record

    def _finalize_native_attempt_unlocked(
        self,
        work_item_id: str,
        *,
        persist: bool = True,
        status: str,
        outcome: StageOutcome | None = None,
        reason: str | None = None,
        run_record: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Finalize one durable native attempt without a second record."""
        record = self._native_accounting.get(str(work_item_id))
        if record is None:
            return None
        if record.get("finalized_at") is not None:
            self._refresh_native_accounting_totals()
            return record
        expected_visual = int(record.get("expected_native_visual_worlds") or 0)
        reported_visual = (
            None if outcome is None else int(outcome.native_visual_worlds)
        )
        reported_acoustic = (
            None if outcome is None else int(outcome.native_acoustic_contexts)
        )
        evidence_record = run_record
        if evidence_record is None:
            logs = dict(record.get("worker_logs") or {})
            evidence_record = {
                "stdout_log": logs.get("stdout"),
                "stderr_log": logs.get("stderr"),
            }
        actual_acoustic = self._actual_native_acoustic_contexts(
            outcome, evidence_record
        )
        actual_acoustic_count = (
            None if actual_acoustic is None else actual_acoustic[0]
        )
        actual_acoustic_source = (
            None if actual_acoustic is None else actual_acoustic[1]
        )
        actual_acoustic_evidence_path = (
            None if actual_acoustic is None else actual_acoustic[2]
        )
        # A fresh capture that reached its launcher is one capture attempt,
        # even if the worker omitted its optional reported count. Retained
        # captures have expected_visual == 0 and remain free.
        effective_visual = max(expected_visual, reported_visual or 0)
        actual_visual, actual_visual_source, actual_visual_evidence = (
            self._actual_native_visual_worlds(outcome)
        )
        finished_at = _utc_now()
        record.update({
            "status": str(status),
            "finished_at": finished_at,
            "finalized_at": finished_at,
            "reason": reason,
            "reported_native_visual_worlds": reported_visual,
            # What was charged and what actually opened are two columns, on
            # purpose. A charge is never lowered to match the second one.
            "actual_native_visual_worlds": actual_visual,
            "actual_native_visual_worlds_source": actual_visual_source,
            "actual_native_visual_worlds_evidence_path": actual_visual_evidence,
            "reported_native_acoustic_contexts": reported_acoustic,
            "actual_native_acoustic_contexts": actual_acoustic_count,
            "actual_native_acoustic_contexts_source": actual_acoustic_source,
            "actual_native_acoustic_contexts_evidence_path": (
                actual_acoustic_evidence_path
            ),
            "native_acoustic_contexts_lower_bound": (
                0 if actual_acoustic_count is None
                else max(0, int(actual_acoustic_count))
            ),
            "effective_capture_instances": (
                int(record.get("capture_instances") or 0)
                if effective_visual
                else 0
            ),
            "effective_native_visual_worlds": effective_visual,
        })
        self._refresh_native_accounting_totals()
        self._append_event({
            "event": "native_attempt_finalized",
            "work_item_id": str(work_item_id),
            "native_accounting": deepcopy(record),
        })
        if persist:
            self._persist()
        return record

    # -- schedule -------------------------------------------------------

    def _round_state(self, scope: ScopeState) -> dict[str, Any]:
        if scope.scope_kind == "core_group":
            from avengine.dataset.production_spec import group_round_state

            group = core_group_from_manifest(self.manifest, scope.scope_key)
            parsed = [StageResult.from_mapping(value) for value in scope.results]
            return group_round_state(group, parsed).to_dict()
        from avengine.dataset.production_spec import request_round_state
        from avengine.qa.batch_manifest import production_request_from_legacy

        row = scope.row or self.rows_by_episode_id[scope.scope_key]
        request = production_request_from_legacy(
            row["request"], request_id=scope.scope_key, kind="episode")
        parsed = [StageResult.from_mapping(value) for value in scope.results]
        return request_round_state(request, parsed).to_dict()

    def reopen_failed_units(
        self, unit_ids: Sequence[str]
    ) -> dict[str, Any]:
        """Re-offer named units whose *input* was repaired outside this run.

        A stage that failed on a missing input is not a stage that may be
        reseeded: the ordinary retry path refuses it, correctly, because
        running the same thing again produces the same failure. When the
        input is actually repaired that refusal becomes wrong, and the only
        ways out were editing state.json by hand or starting a new run and
        redoing the work that already passed. Neither is acceptable, so the
        re-opening is an explicit, named, bounded operation instead.

        What it does *not* do: it does not touch a passing result, it does
        not delete the failed attempt's directory, and it does not reduce
        anything already charged. The failed result moves to
        ``retired_failures`` where it stays in the state and in the journal,
        and the unit's next attempt number is shifted past it so the repaired
        run writes its own fresh root beside the old one.
        """

        wanted = {str(value) for value in unit_ids if str(value).strip()}
        if not wanted:
            return {"status": "not_run", "reason": "no unit named",
                    "reopened": [], "unmatched": []}
        reopened: list[dict[str, Any]] = []
        matched: set[str] = set()
        for scope in self.scopes:
            keep: list[dict[str, Any]] = []
            for row in scope.results:
                unit = self._unit_key_for_result(row)
                aliases = {unit, f"{scope.scope_key}/{unit}"}
                if scope.scope_kind == "episode":
                    aliases.add(scope.scope_key)
                selected = wanted & aliases
                if not selected or str(row.get("status")) == "pass":
                    keep.append(row)
                    continue
                matched.update(selected)
                retired = deepcopy(dict(row))
                retired["retired_at"] = _utc_now()
                retired["retired_reason"] = (
                    "explicitly re-opened after its input was repaired"
                )
                scope.retired_failures.append(retired)
                scope.retry_offsets[unit] = scope.retry_offsets.get(unit, 0) + 1
                reopened.append({
                    "scope_key": scope.scope_key,
                    "unit_id": unit,
                    "work_item_id": str(row.get("work_item_id") or ""),
                    "previous_status": str(row.get("status")),
                    "previous_reason": row.get("reason"),
                    "retry_offset": scope.retry_offsets[unit],
                })
            if len(keep) != len(scope.results):
                scope.results = keep
                scope.blockers = [
                    blocker for blocker in scope.blockers
                    if str(blocker.get("unit_id") or "") not in wanted
                ]
                scope.finished = False
                scope.finished_reason = None
        record = {
            "status": "reopened" if reopened else "not_run",
            "reopened": reopened,
            "unmatched": sorted(wanted - matched),
            "note": (
                "Failed results are retired, never deleted; passing results "
                "and every attempt directory are untouched."
            ),
        }
        if reopened:
            self._append_event({"event": "failed_units_reopened", **deepcopy(record)})
            self._persist()
        return record

    def replan_units(
        self,
        unit_ids: Sequence[str],
        *,
        reason: str = "explicit replan after a declared input change",
    ) -> dict[str, Any]:
        """Retire a passed plan and its downstream units for a fresh attempt.

        This is the normal programmatic path for a repaired input. It preserves
        every prior result and attempt directory, advances only the affected
        unit offsets, and leaves upstream passed units available to the new
        round. The caller records the new input through recipe_options.
        """
        wanted = {str(value).strip() for value in unit_ids if str(value).strip()}
        if not wanted:
            return {
                "status": "not_run",
                "replanned": [],
                "unmatched": [],
                "reason": "no unit named",
            }
        replanned: list[dict[str, Any]] = []
        matched_roots: set[str] = set()
        for scope in self.scopes:
            if scope.scope_kind != "core_group" or not scope.task_family:
                continue
            recipe = recipe_for_task_family(str(scope.task_family))
            units = {str(unit.unit_id): unit for unit in recipe.units}
            roots = wanted & set(units)
            if not roots:
                continue
            closure = set(roots)
            changed = True
            while changed:
                changed = False
                for unit_id, unit in units.items():
                    if unit_id not in closure and any(
                        dependency in closure for dependency in unit.depends_on_units
                    ):
                        closure.add(unit_id)
                        changed = True
            found_root = False
            keep: list[dict[str, Any]] = []
            for row in scope.results:
                unit_id = self._unit_key_for_result(row)
                if unit_id not in closure:
                    keep.append(row)
                    continue
                if unit_id in roots:
                    found_root = True
                retired = deepcopy(dict(row))
                retired["retired_at"] = _utc_now()
                retired["retired_reason"] = reason
                retired["replan_root_units"] = sorted(roots)
                status = str(row.get("status") or "")
                if status == "pass":
                    scope.retired_results.append(retired)
                else:
                    scope.retired_failures.append(retired)
                scope.retry_offsets[unit_id] = (
                    scope.retry_offsets.get(unit_id, 0) + 1
                )
                replanned.append({
                    "scope_key": scope.scope_key,
                    "unit_id": unit_id,
                    "previous_status": status,
                    "previous_work_item_id": row.get("work_item_id"),
                    "retry_offset": scope.retry_offsets[unit_id],
                    "root_units": sorted(roots),
                })
            if not found_root:
                continue
            matched_roots.update(roots)
            scope.results = [
                row for row in scope.results
                if self._unit_key_for_result(row) not in closure
            ]
            scope.blockers = [
                blocker for blocker in scope.blockers
                if self._unit_key_for_result(blocker) not in closure
            ]
            scope.finished = False
            scope.finished_reason = None
        unmatched = sorted(wanted - matched_roots)
        record = {
            "status": "replanned" if replanned else "not_run",
            "replanned": replanned,
            "unmatched": unmatched,
            "reason": reason,
        }
        if replanned:
            self._append_event({
                "event": "passed_units_replanned",
                **deepcopy(record),
            })
            self._persist()
        return record


    @staticmethod
    def _unit_key_for_result(row: Mapping[str, Any]) -> str:
        """The unit a filed result belongs to, however it was recorded."""
        unit = row.get("unit_id")
        if isinstance(unit, str) and unit.strip():
            return unit.strip()
        scope_id = str(row.get("scope_id") or "")
        if "/" in scope_id:
            return scope_id.rsplit("/", 1)[-1]
        if row.get("stage"):
            return str(row["stage"])
        return str(row.get("work_item_id") or "").split(":", 1)[0].rsplit("/", 1)[-1]

    def next_work_items(self, scope: ScopeState) -> list[dict[str, Any]]:
        if scope.scope_kind == "core_group":
            items = stage_work_items_for_group(
                self.manifest, scope.scope_key, results=scope.results)
        else:
            row = scope.row or self.rows_by_episode_id[scope.scope_key]
            items = stage_work_items_for_row(row, results=scope.results)
        if scope.retry_offsets:
            # A re-opened unit resumes above its retired attempt so the
            # repaired run writes a fresh root and the failed one survives.
            for item in items:
                unit = str(item.get("unit_id") or item.get("stage") or "")
                offset = scope.retry_offsets.get(unit, 0)
                if not offset:
                    continue
                attempt = int(item.get("attempt") or 1) + offset
                item["attempt"] = attempt
                item["work_item_id"] = work_item_id(
                    str(item["request_id"]), str(item["stage"]), attempt
                )
                item["fresh_output_relative"] = fresh_output_relative(
                    str(item["request_id"]), str(item["stage"]), attempt
                )
                item["reopened_retry_offset"] = offset
        if scope.scope_kind == "core_group" and scope.candidate_index:
            attempt = 1 + int(scope.candidate_index)
            for item in items:
                item["attempt"] = attempt
                item["work_item_id"] = work_item_id(
                    str(item["request_id"]), str(item["stage"]), attempt
                )
                item["fresh_output_relative"] = fresh_output_relative(
                    str(item["request_id"]), str(item["stage"]), attempt
                )
                payload = dict(item.get("payload") or {})
                candidate = int(scope.candidate_index)
                # Keep the scheduler's historical name for old task readers,
                # and persist the sampler-facing name in the task itself.
                payload["candidate_index"] = candidate
                payload["sampling_candidate_index"] = candidate
                item["payload"] = payload
        return items

    def blockers(self, scope: ScopeState) -> list[dict[str, Any]]:
        if scope.scope_kind != "core_group":
            return []
        return group_blockers_for_group(
            self.manifest, scope.scope_key, results=scope.results)

    # -- execution ------------------------------------------------------

    def _context_for(self, scope: ScopeState, work_item: Mapping[str, Any],
                     hold: LeaseHold | None) -> StageContext:
        upstream = {}
        depends = {str(value) for value in (work_item.get("depends_on") or [])}
        for result in scope.results:
            if str(result.get("work_item_id")) in depends:
                if scope.scope_kind == "episode":
                    unit = str(result.get("stage") or result.get("scope_id") or "")
                else:
                    unit = str(result.get("scope_id", "")).rsplit("/", 1)[-1]
                upstream[unit] = deepcopy(result)
        relative = str(work_item["fresh_output_relative"])
        member_request_ids = tuple(work_item.get("member_request_ids") or ())
        if scope.scope_kind == "episode" and not member_request_ids:
            member_request_ids = (scope.scope_key,)
        task_family = (
            None if scope.scope_kind == "episode"
            else work_item.get("task_family")
        )
        return StageContext(
            work_item=deepcopy(dict(work_item)),
            manifest=self.manifest,
            manifest_path=self.manifest_path,
            group_id=work_item.get("group_id"),
            task_family=task_family,
            unit_id=work_item.get("unit_id"),
            stage=str(work_item["stage"]),
            scope_id=str(work_item["request_id"]),
            attempt=int(work_item["attempt"]),
            member_request_ids=member_request_ids,
            rows_by_episode_id=self.rows_by_episode_id,
            upstream=upstream,
            output_root=self.run_root / "work" / relative,
            run_root=self.run_root,
            repository=self.repository,
            rpc_port=None if hold is None else hold.rpc_port,
            graphics_adapter=None if hold is None else hold.graphics_adapter,
            lease=None if hold is None else hold.lease.to_dict(),
            round_results=tuple(deepcopy(row) for row in scope.results),
            recipe_options=self.recipe_options_for(scope),
        )

    def _retained_root_for(
        self,
        scope: ScopeState,
        work_item: Mapping[str, Any],
    ) -> str | None:
        context = self._context_for(scope, work_item, None)
        retained = retained_root_declared_for(context)
        if retained is not None:
            return retained
        if scope.scope_kind == "episode":
            value = self.recipe_options_for(scope).get("retained_episode_root")
            if isinstance(value, str) and value.strip():
                return str(Path(value).expanduser().resolve())
        return None

    def recipe_options_for(self, scope: ScopeState) -> dict[str, Any]:
        """Per-group recipe configuration: retained visuals, world id, split.

        A retained visual root is how a run reuses an already produced world
        instead of rendering another one, so it is configuration rather than a
        name this module knows.
        """
        options = deepcopy(self.recipe_options.get(scope.scope_key, {}))
        declared = declared_world_id(self.manifest, scope.scope_key)
        if declared and "world_id" not in options:
            options["world_id"] = declared
        return options

    def _runtime_prefix(self) -> str | None:
        # A runtime prefix is an explicit room/config input. Do not infer one
        # from whichever Python happens to run the controller.
        value = os.environ.get("AVENGINE_HABITAT_RUNTIME_PREFIX")
        if isinstance(value, str) and value.strip() and Path(value).is_dir():
            return str(Path(value).resolve())
        return None

    def _compatibility_for(
        self,
        work_item: Mapping[str, Any],
        scope: ScopeState | None = None,
    ) -> WorkerCompatibility:
        request = (work_item.get("inputs") or {}).get("request")
        if not isinstance(request, Mapping) and scope is not None:
            request = (scope.row or {}).get("request")
        if not isinstance(request, Mapping):
            # Internal units serve no delivered member, but inputs still name their owner.
            row_id = ((work_item.get("inputs") or {}).get("request_id")
                      or (work_item.get("member_request_ids") or [None])[0])
            request = (self.rows_by_episode_id.get(str(row_id), {}) or {}).get("request") or {}
        report = self.room_resolver(request)
        missing = tuple(report.get("missing") or ()) if isinstance(report, Mapping) else ()
        if "runtime_prefix" in missing:
            fallback = self._runtime_prefix()
            if fallback:
                request = deepcopy(dict(request))
                runtime = dict(request.get("runtime") or {})
                runtime["runtime_prefix"] = fallback
                request["runtime"] = runtime
                report = self.room_resolver(request)
        return WorkerCompatibility.from_runtime_report(
            report,
            python_executable=self.python_executable,
            runtime_context=(work_item.get("resource") or {}).get("runtime_context"),
        )

    def _worker_entry(self, task_path: Path, result_path: Path) -> list[str]:
        return ["-m", "avengine.dataset.production_runner",
                "--execute-work-item", str(task_path), "--result", str(result_path)]

    def _launch_worker(self, *, plan: Mapping[str, Any], work_dir: Path,
                       work_item_id_value: str) -> dict[str, Any]:
        """Start the fresh interpreter and wait for it, recording what ran."""
        stdout_path = work_dir / "stdout.log"
        stderr_path = work_dir / "stderr.log"
        # A previous attempt at this work item may have left its logs here.
        # They are kept, under the time they were finished with, rather than
        # being overwritten or being allowed to abort the relaunch: the
        # exclusive create below is there to stop two live workers sharing a
        # log, not to make a retry impossible.
        for path in (stdout_path, stderr_path):
            if path.exists():
                stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                kept = path.with_name(f"{path.stem}.superseded_{stamp}{path.suffix}")
                index = 1
                while kept.exists():
                    index += 1
                    kept = path.with_name(
                        f"{path.stem}.superseded_{stamp}_{index}{path.suffix}")
                path.rename(kept)
                self._append_event({
                    "event": "worker_log_superseded",
                    "work_item_id": work_item_id_value,
                    "kept_as": str(kept),
                })
        environment = dict(os.environ)
        environment.update({str(k): str(v) for k, v in (plan.get("env") or {}).items()})
        started_at = _utc_now()
        with stdout_path.open("x", encoding="utf-8") as stdout, \
                stderr_path.open("x", encoding="utf-8") as stderr:
            process = subprocess.Popen(
                [str(item) for item in plan["argv"]], cwd=str(plan["cwd"]),
                env=environment, stdout=stdout, stderr=stderr,
                stdin=subprocess.DEVNULL, start_new_session=True, text=True)
            identity = process_identity(process.pid) or {"pid": process.pid}
            record = {
                "pid": process.pid,
                "start_ticks": identity.get("start_ticks"),
                "cmdline_marker": work_item_id_value,
                "argv": [str(item) for item in plan["argv"]],
                "cwd": str(plan["cwd"]),
                "pythonpath": (plan.get("env") or {}).get("PYTHONPATH"),
                "started_at": started_at,
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
            }
            self._worker_records[work_item_id_value] = record
            # Persist the worker log paths together with the live worker
            # record. If the parent restarts before the child writes a result,
            # resume can still inspect this work item's own logs.
            self._remember_native_worker_logs(work_item_id_value, record)
            self._persist()
            returncode = process.wait()
        record["returncode"] = returncode
        record["finished_at"] = _utc_now()
        return record

    def _attach_worker(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Wait for a worker of this run that survived the interruption."""
        pid = int(record["pid"])
        while process_matches(record):
            time.sleep(1.0)
        result = dict(record)
        result["returncode"] = None
        result["finished_at"] = _utc_now()
        result["adopted"] = True
        result["adopted_pid"] = pid
        return result

    @staticmethod
    def _recover_existing_stage_result(output_root: Path) -> StageOutcome | None:
        stage_result = output_root / "stage_result.json"
        if not stage_result.is_file():
            return None
        try:
            value = _read_json(stage_result)
            if value.get("status") != "pass":
                return None
            return StageOutcome.from_mapping(value)
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _invoke_recipe_audio_recovery(
        self,
        scope: ScopeState,
        work_item: Mapping[str, Any],
        output_root: Path,
    ) -> StageOutcome | None:
        """Ask a recipe module to CPU-finalize one existing audio attempt."""
        if str((work_item.get("payload") or {}).get("unit_kind") or "") != "audio":
            return None
        if scope.scope_kind != "core_group" or not scope.task_family:
            # Not a recipe-owned unit. Recorded rather than passed over, so a
            # unit that quietly never asked for recovery is distinguishable
            # from one that asked and was refused.
            self._append_event({
                "event": "audio_recovery_not_attempted",
                "work_item_id": str(work_item["work_item_id"]),
                "reason": (
                    "the unit is not a core-group unit with a task family, so "
                    "no recipe owns its audio"
                ),
                "scope_kind": scope.scope_kind,
                "task_family": scope.task_family,
            })
            return None
        try:
            import importlib
            binding = recipe_binding(scope.task_family)
            module = importlib.import_module(binding.module)
            recover = getattr(module, "recover_rendered_audio_attempt", None)
            if not callable(recover):
                self._append_event({
                    "event": "audio_recovery_hook_absent",
                    "work_item_id": str(work_item["work_item_id"]),
                    "task_family": scope.task_family,
                    "module": binding.module,
                    "reason": (
                        "the recipe module exposes no "
                        "recover_rendered_audio_attempt, so a finished render "
                        "here can only be produced again"
                    ),
                })
                return None
            context = self._context_for(scope, work_item, None)
            recipe_context = native_group_context(context)
            request = (work_item.get("inputs") or {}).get("request")
            if not isinstance(request, Mapping):
                request = (scope.row or {}).get("request") if scope.row else {}
            world_ids = []
            for value in (
                (request or {}).get("world_id"),
                declared_world_id(self.manifest, scope.scope_key),
                self.recipe_options_for(scope).get("world_id"),
            ):
                if isinstance(value, str) and value.strip():
                    world_ids.append(value)
            canonical = next(
                (value for value in world_ids if not value.endswith("_foa")),
                None,
            )
            rejected = sorted({value for value in world_ids if value.endswith("_foa")})
            recovery_root = output_root.parent / (
                f"recovery_{uuid4().hex}"
            )
            lineage = {
                "schema": "avengine_audio_recovery_lineage_v1",
                "scope_id": scope.scope_key,
                "group_id": scope.group_id,
                "task_family": scope.task_family,
                "unit_id": work_item.get("unit_id"),
                "member_request_ids": list(
                    work_item.get("member_request_ids") or ()
                ),
                "work_item_id": str(work_item["work_item_id"]),
                "attempt": int(work_item.get("attempt") or 0),
                "candidate_index": (
                    _sampling_candidate_index_from_payload(
                        work_item.get("payload") or {}
                    )
                ),
                "previous_attempt_root": str(output_root.resolve()),
                "previous_stage_result_path": str(
                    (output_root / "stage_result.json").resolve()
                ) if (output_root / "stage_result.json").is_file() else None,
                "world_id": canonical,
                "rejected_world_ids": rejected,
                "native_restart": False,
                "rlr_restart": False,
            }
            self._append_event({
                "event": "audio_recovery_requested",
                "work_item_id": str(work_item["work_item_id"]),
                "lineage": deepcopy(lineage),
                "recovery_root": str(recovery_root),
            })
            recovered = recover(
                dict(work_item),
                recipe_context,
                output_root=recovery_root,
                previous_attempt_root=output_root,
                lineage=lineage,
                results=list(scope.results),
                lease=None,
            )
            if not isinstance(recovered, Mapping):
                self._append_event({
                    "event": "audio_recovery_declined",
                    "work_item_id": str(work_item["work_item_id"]),
                    "reason": (
                        "the recipe returned "
                        f"{type(recovered).__name__}, not a stage result"
                    ),
                })
                return None
            if "outcome" in recovered and isinstance(
                recovered.get("outcome"), Mapping
            ):
                recovered = recovered["outcome"]
            outcome = StageOutcome.from_mapping(recovered)
            if outcome.status != "pass":
                self._append_event({
                    "event": "audio_recovery_declined",
                    "work_item_id": str(work_item["work_item_id"]),
                    "status": outcome.status,
                    "reason": outcome.reason,
                    "reason_code": outcome.reason_code,
                })
                return None
            return outcome
        except Exception as error:
            # Every recipe raises its own error type -- the binding recipe
            # refuses with BindingNativeError, which is not a
            # ProductionRunError -- and naming them here would mean importing
            # each recipe's exceptions into the runner. A recovery that
            # cannot run is not a reason to lose the whole run: the caller
            # falls back to the ordinary interrupted path, which reports the
            # unit honestly and lets it be produced again.
            #
            # Broad is deliberate, silent is not. The error is journalled
            # under its own type and message, so a recovery that failed for a
            # programming reason is visible as that and never reads as "there
            # was nothing here to recover".
            self._append_event({
                "event": "audio_recovery_failed",
                "work_item_id": str(work_item["work_item_id"]),
                "error": f"{type(error).__name__}: {error}",
            })
            return None

    def _recover_completed_audio_attempt(
        self,
        scope: ScopeState,
        work_item: Mapping[str, Any],
        output_root: Path,
    ) -> StageOutcome | None:
        """Persist a completed audio artifact after controller loss.

        This path is readback-only: it requires the finalized facts, real
        research report, assignment request and primary/attached WAVs already
        present in this attempt root. It never calls the native audio runner.
        """
        recipe_outcome = self._invoke_recipe_audio_recovery(
            scope, work_item, output_root
        )
        if recipe_outcome is not None:
            return recipe_outcome
        unit_kind = str((work_item.get("payload") or {}).get("unit_kind") or "")
        if unit_kind != "audio":
            return None
        episode_root = output_root / "episode"
        delivery = episode_root / "delivery"
        facts_path = delivery / "facts.json"
        report_path = delivery / "research_report.json"
        assignment_request_path = output_root / "assignment_request.json"
        if not assignment_request_path.is_file():
            assignment_request_path = episode_root / "assignment_request.json"
        if not (facts_path.is_file() and report_path.is_file()
                and assignment_request_path.is_file()):
            return None
        try:
            facts_value = _read_json(facts_path)
            report = _read_json(report_path)
            assignment_request = _read_json(assignment_request_path)
            if facts_value.get("status") != "pass":
                return None
            from avengine.dataset.binding_group_native import declared_audio_delivery
            declared = declared_audio_delivery(assignment_request)
            finalized = {
                "audio_report": str(report_path.resolve()),
                "declared_audio_delivery": deepcopy(declared),
            }
            layout_readback = _ordinary_audio_layout_readback(
                finalized, facts_value
            )
            if layout_readback.get("status") != "pass":
                return None
            layout_delivery = ((report.get("audio") or {}).get("layout_delivery")
                               if isinstance(report, Mapping) else {})
            layout_delivery = (
                layout_delivery if isinstance(layout_delivery, Mapping) else {}
            )
            ancillary: list[dict[str, Any]] = []
            for layout, entry in layout_delivery.items():
                if str(layout) == "binaural" or not isinstance(entry, Mapping):
                    continue
                mixture = entry.get("mixture") or {}
                mixture_path = mixture.get("path")
                if not isinstance(mixture_path, str) or not Path(mixture_path).is_file():
                    return None
                verification = next(
                    (
                        row for row in layout_readback.get("layouts") or ()
                        if row.get("layout_type") == layout
                    ),
                    None,
                )
                ancillary.append({
                    "layout": str(layout),
                    "layout_type": str(layout),
                    "report_path": str(report_path.resolve()),
                    "mixture_path": str(Path(mixture_path).resolve()),
                    "declaration": deepcopy(dict(entry)),
                    "verification": deepcopy(verification or {}),
                })
            primary = layout_delivery.get("binaural") or {}
            primary_path = (primary.get("mixture") or {}).get("path")
            if not isinstance(primary_path, str) or not Path(primary_path).is_file():
                return None
            upstream_capture = None
            for row in scope.results:
                if str(row.get("work_item_id")) not in {
                    str(value) for value in (work_item.get("depends_on") or ())
                }:
                    continue
                upstream_capture = (row.get("outputs") or {}).get("capture")
                if upstream_capture:
                    break
            outputs = {
                "variant_root": str(episode_root.resolve()),
                "audio": str(Path(primary_path).resolve()),
                "audio_report": str(report_path.resolve()),
                "audio_report_path": str(report_path.resolve()),
                "assignment_request_path": str(assignment_request_path.resolve()),
                "assignment_plan_path": str(
                    (output_root / "assignment_plan.json").resolve()
                ) if (output_root / "assignment_plan.json").is_file() else None,
                "visual_video": str((delivery / "preview.mp4").resolve())
                if (delivery / "preview.mp4").is_file() else None,
                "capture": upstream_capture,
                "declared_audio_delivery": deepcopy(declared),
                "delivered_audio_layouts": deepcopy(layout_readback),
                "ancillary_audio_outputs": ancillary,
                "member_request_id": (
                    (work_item.get("member_request_ids") or [None])[0]
                ),
                "member_request_ids": list(work_item.get("member_request_ids") or ()),
                "recovered_from_completed_attempt": True,
                "recovery_evidence": {
                    "facts_path": str(facts_path.resolve()),
                    "audio_report_path": str(report_path.resolve()),
                    "audio_log_path": str((delivery / "audio.log").resolve())
                    if (delivery / "audio.log").is_file() else None,
                },
            }
            audio_facts = facts_value.get("audio") or {}
            return StageOutcome(
                status="pass",
                facts={
                    "facts_path": str(facts_path.resolve()),
                    "audio_report_path": str(report_path.resolve()),
                    "wet_tail_intervals": deepcopy(
                        audio_facts.get("wet_tail_intervals") or []
                    ),
                    "declared_audio_delivery": deepcopy(declared),
                    "delivered_audio_layouts": deepcopy(layout_readback),
                },
                outputs=outputs,
                diagnostics={
                    "recovery": "completed_render_report_readback",
                    "no_native_restart": True,
                    "ancillary_audio_outputs": ancillary,
                },
                native_acoustic_contexts=0,
            )
        except (OSError, ValueError, KeyError, ProductionRunError):
            return None

    def execute_work_item(self, scope: ScopeState,
                          work_item: Mapping[str, Any]) -> dict[str, Any]:
        """Lease, run, read back and file one unit. Never partly."""
        item_id = str(work_item["work_item_id"])
        stage = str(work_item["stage"])
        unit_kind = str((work_item.get("payload") or {}).get("unit_kind") or stage)
        work_dir = self.run_root / "workers" / item_id.replace("/", "__").replace(":", "_")
        work_dir.mkdir(parents=True, exist_ok=True)
        output_root = self.run_root / "work" / str(work_item["fresh_output_relative"])
        self._append_event({"event": "work_item_ready", "work_item_id": item_id,
                            "stage": stage, "unit_kind": unit_kind,
                            "scope": scope.scope_key,
                            "resource": deepcopy(work_item.get("resource"))})

        retained_for_unit = self._retained_root_for(scope, work_item)
        with self._accounting_lock:
            self._refresh_native_accounting_totals()
            existing_native_attempt = self._native_accounting.get(item_id)
            budget_exhausted = (
                unit_kind in {"visual_capture", "capture"}
                and retained_for_unit is None
                and self.native_visual_world_budget is not None
                and self.native_visual_worlds_used >= self.native_visual_world_budget
                and existing_native_attempt is None
            )
            budget_used = self.native_visual_worlds_used
        if budget_exhausted:
            return self._file_failure(
                scope, work_item,
                reason=(f"this run has used {budget_used} of its "
                        f"{self.native_visual_world_budget} authorised new native "
                        "visual worlds, failed attempts included"),
                reason_code=NativeBudgetExhausted.reason_code, status="blocked")

        adopted = self._worker_records.get(item_id)
        if adopted is None:
            adopted = _find_live_stage_worker(work_dir)
            if adopted is not None:
                self._worker_records[item_id] = adopted
                self._append_event({"event": "live_worker_rediscovered",
                                    "work_item_id": item_id, "pid": adopted["pid"]})
                self._persist()
        dropped_on_resume = item_id in set(
            self._resumed.get("dropped_worker_work_item_ids") or ()
        )
        hold: LeaseHold | None = None
        decision: dict[str, Any] = {}
        result_path = work_dir / "stage_result.json"
        if adopted is not None and process_matches(adopted):
            self._ensure_native_attempt(scope, work_item, retained_for_unit)
            run_record = self._attach_worker(adopted)
        elif result_path.is_file():
            # A controller may have been interrupted after its child wrote a
            # result but before the parent filed it. The resume must consume
            # that known result for any stage, including pure CPU delivery;
            # launching the same work item again would duplicate the stage.
            #
            # This deliberately does not require the run to *remember*
            # launching the worker. A controller that dies without persisting
            # its worker record leaves a complete result sitting in this
            # run's own worker directory, written by the worker for this
            # exact work item, and the old condition threw all of that away
            # and re-rendered: an interruption during a ten-minute RLR render
            # cost a second ten-minute render, and then died on the log file
            # the first one had already created.
            run_record = deepcopy(
                adopted
                or {
                    "recovered_result": True,
                    "work_item_id": item_id,
                    "returncode": None,
                }
            )
            if not dropped_on_resume and adopted is None:
                self._append_event({
                    "event": "worker_result_adopted_from_disk",
                    "work_item_id": item_id,
                    "result_path": str(result_path),
                    "reason": (
                        "this run's worker wrote a result for this work item "
                        "and the controller was lost before filing it; the "
                        "run does not remember launching it"
                    ),
                })
        elif (
            (existing_native_attempt is not None or dropped_on_resume)
            and not result_path.is_file()
        ):
            # An attempt with no filed result is not automatically an attempt
            # with no work done. The render may have finished and only the
            # filing been lost, which is what a controller that dies between
            # the last write and the result file leaves behind. Ask for a CPU
            # finalization of what is on disk *before* calling this
            # interrupted: re-rendering spends a native world and an RLR
            # context on audio that already exists, and the recovery refuses
            # by itself when the artifacts are not complete.
            #
            # Both resume shapes reach here. A worker record that no longer
            # matches a live process is dropped on resume and its id is listed
            # in dropped_worker_work_item_ids, so keying this on the record
            # still being in _worker_records skipped recovery for exactly the
            # interruption it was written for.
            recovered_outcome = self._recover_existing_stage_result(output_root)
            if recovered_outcome is None:
                recovered_outcome = self._recover_completed_audio_attempt(
                    scope, work_item, output_root
                )
            if recovered_outcome is not None:
                # Persist both contracts: the runner worker result and the
                # native stage result consumed by T02's no-clobber dispatcher.
                _json_write_new(result_path, {
                    "status": "ok",
                    "outcome": recovered_outcome.to_dict(),
                    "recovered_completed_attempt": True,
                })
                native_stage_result = output_root / "stage_result.json"
                if not native_stage_result.is_file():
                    _json_write_new(native_stage_result, {
                        "schema": "avengine_native_group_stage_result_v1",
                        **recovered_outcome.to_dict(),
                    })
                run_record = {
                    "recovered_completed_attempt": True,
                    "work_item_id": item_id,
                    "stdout_log": str(
                        (output_root / "episode" / "delivery" / "audio.log").resolve()
                    ),
                    "returncode": 0,
                }
            else:
                # The attempt exists but does not contain a complete, verified
                # audio publication. Preserve the old interruption behavior.
                self._ensure_native_attempt(scope, work_item, retained_for_unit)
                reason = (
                    f"attempt {work_item['attempt']} was interrupted before it "
                    f"produced a result; its partial output is retained at {output_root}"
                )
                accounting = self._finalize_native_attempt(
                    item_id, status="interrupted", reason=reason
                )
                return self._file_failure(
                    scope, work_item, reason=reason,
                    reason_code=INTERRUPTED_REASON_CODE, status="fail",
                    diagnostic={
                        "partial_output_root": str(output_root),
                        "worker_record": deepcopy(dict(adopted or {})),
                        "native_accounting": deepcopy(accounting or {}),
                    })
        elif output_root.exists() and not result_path.is_file():
            # This attempt started and was cut off before it wrote a result. Its
            # partial output is retained where it is; the next attempt gets its
            # own fresh root rather than writing into a half-finished one.
            self._ensure_native_attempt(scope, work_item, retained_for_unit)
            reason = (
                f"attempt {work_item['attempt']} was interrupted before it "
                f"produced a result; its partial output is retained at {output_root}"
            )
            accounting = self._finalize_native_attempt(
                item_id, status="interrupted", reason=reason
            )
            return self._file_failure(
                scope, work_item, reason=reason,
                reason_code=INTERRUPTED_REASON_CODE, status="fail",
                diagnostic={
                    "partial_output_root": str(output_root),
                    "worker_record": deepcopy(dict(adopted or {})),
                    "native_accounting": deepcopy(accounting or {}),
                })
        else:
            try:
                compatibility = self._compatibility_for(work_item, scope)
            except Exception as error:  # a room that will not resolve is a real failure
                return self._file_failure(
                    scope, work_item, reason=f"{type(error).__name__}: {error}",
                    reason_code="room_runtime_unresolved")
            task_path = work_dir / "task.json"
            runtime_prefix = self._runtime_prefix()
            worker_environment = (
                {"AVENGINE_HABITAT_RUNTIME_PREFIX": runtime_prefix}
                if runtime_prefix else None
            )
            hold, decision = self.broker.acquire(
                work_item, compatibility=compatibility,
                entry=self._worker_entry(task_path, result_path),
                environment=worker_environment,
                estimated_peak_vram_mb=_declared_peak_vram_mb(work_item),
            )
            if hold is None:
                return self._file_failure(
                    scope, work_item,
                    reason=str(decision.get("reason") or "no lease was granted"),
                    reason_code=str(decision.get("reason_code") or "lease_unavailable"),
                    status="blocked", diagnostic={"lease_decision": decision})
            try:
                context = self._context_for(scope, work_item, hold)
                if task_path.exists():
                    task_path.unlink()
                _json_write_new(task_path, {
                    "schema": SCHEMA, "context": context.to_dict(),
                    "executor": {"task_family": context.task_family,
                                 "unit_kind": unit_kind},
                })
                with self._accounting_lock:
                    self._refresh_native_accounting_totals()
                    late_existing = self._native_accounting.get(item_id)
                    late_budget_exhausted = (
                        unit_kind in {"visual_capture", "capture"}
                        and retained_for_unit is None
                        and self.native_visual_world_budget is not None
                        and self.native_visual_worlds_used >= self.native_visual_world_budget
                        and late_existing is None
                    )
                    late_budget_used = self.native_visual_worlds_used
                    if not late_budget_exhausted:
                        # Reserve atomically with the budget check, but never
                        # write state while holding the accounting lock.
                        self._ensure_native_attempt_unlocked(
                            scope, work_item, retained_for_unit, persist=False)
                if late_budget_exhausted:
                    return self._file_failure(
                        scope, work_item,
                        reason=(f"this run has used {late_budget_used} of its "
                                f"{self.native_visual_world_budget} authorised new native "
                                "visual worlds, failed attempts included"),
                        reason_code=NativeBudgetExhausted.reason_code,
                        status="blocked")
                self._persist()
                run_record = self.launcher(
                    plan=hold.plan, work_dir=work_dir, work_item_id_value=item_id)
                self._remember_native_worker_logs(item_id, run_record)
                pid = run_record.get("pid")
                if pid is not None:
                    self.broker.bind(hold.lease.lease_id, int(pid))
            finally:
                if hold is not None:
                    self.broker.release(hold.lease)
                # A launcher can raise before returning a run record. The
                # lease has still been released, so persist that fact now;
                # otherwise resume could adopt a stale lease forever.
                self._persist()
        if not result_path.is_file():
            # A worker that fails cleanly writes an error result. One that wrote
            # nothing was cut off -- killed, out of memory, machine restarted --
            # so this is a recoverable interruption of that attempt, bounded by
            # its own limit, not a refusal of the work.
            reason = _worker_error_text(run_record, result_path)
            accounting = self._finalize_native_attempt(
                item_id, status="interrupted", reason=reason,
                run_record=run_record,
            )
            return self._file_failure(
                scope, work_item, reason=reason,
                reason_code=INTERRUPTED_REASON_CODE,
                diagnostic={"worker": deepcopy(run_record),
                            "lease_decision": decision,
                            "partial_output_root": str(output_root),
                            "native_accounting": deepcopy(accounting or {})})
        payload = _read_json(result_path)
        if not isinstance(payload, Mapping) or "outcome" not in payload:
            reason = f"worker result is not a stage outcome: {result_path}"
            accounting = self._finalize_native_attempt(
                item_id, status="worker_result_malformed", reason=reason,
                run_record=run_record,
            )
            return self._file_failure(
                scope, work_item, reason=reason,
                reason_code="worker_result_malformed",
                diagnostic={"worker": deepcopy(run_record),
                            "native_accounting": deepcopy(accounting or {})})
        if payload.get("status") == "error":
            reason = str(payload.get("reason") or "worker raised")
            accounting = self._finalize_native_attempt(
                item_id, status="worker_error", reason=reason,
                run_record=run_record,
            )
            return self._file_failure(
                scope, work_item, reason=reason,
                reason_code=str(payload.get("reason_code") or "worker_failed"),
                diagnostic={"worker": deepcopy(run_record),
                            "traceback_path": payload.get("traceback_path"),
                            "native_accounting": deepcopy(accounting or {})})
        outcome = StageOutcome.from_mapping(payload["outcome"])
        self._finalize_native_attempt(
            item_id,
            status="completed" if outcome.status == "pass" else "failed",
            outcome=outcome,
            reason=outcome.reason,
            run_record=run_record,
        )
        context = self._context_for(scope, work_item, hold)
        verification = verify_stage_outputs(
            work_item=work_item, outcome=outcome, upstream=context.upstream,
            output_root=output_root, run_root=self.run_root,
            retained_roots=sorted(
                {
                    *(
                        str(value)
                        for value in
                        (self.recipe_options_for(scope).get("retained_visual_roots") or {}).values()
                    ),
                    *(
                        [str(self.recipe_options_for(scope)["retained_episode_root"])]
                        if self.recipe_options_for(scope).get("retained_episode_root")
                        else []
                    ),
                    *(
                        [str(outcome.outputs["reused_retained_visual_root"])]
                        if outcome.outputs.get("reused_retained_visual_root")
                        else []
                    ),
                }
            )
        )
        _json_write_atomic(work_dir / "verification.json", verification)
        if outcome.status != "pass":
            return self._file_failure(
                scope, work_item, reason=str(outcome.reason or "the unit did not pass"),
                reason_code=str(outcome.reason_code or "stage_not_passed"),
                status=outcome.status if outcome.status in {"fail", "blocked", "not_run"} else "fail",
                diagnostic={"worker": deepcopy(run_record), "verification": verification})
        if not verification["verified"]:
            return self._file_failure(
                scope, work_item,
                reason=("the unit reported pass but its artifacts do not check out: "
                        + "; ".join(verification["problems"])),
                reason_code=StageOutputMissing.reason_code,
                diagnostic={"worker": deepcopy(run_record), "verification": verification})
        result = {
            "work_item_id": item_id, "stage": stage, "request_id": str(work_item["request_id"]),
            "scope_id": str(work_item["request_id"]), "status": "pass",
            "facts": deepcopy(outcome.facts), "outputs": deepcopy(outcome.outputs),
            "reason": None, "depends_on": list(work_item.get("depends_on") or []),
        }
        StageResult.from_mapping(result)
        with self._result_lock:
            scope.results.append(result)
            self._worker_records.pop(item_id, None)
            self._append_event({"event": "work_item_passed", "work_item_id": item_id,
                                "stage": stage, "scope": scope.scope_key,
                                "verification_path": str(work_dir / "verification.json"),
                                "native_visual_worlds": outcome.native_visual_worlds,
                                "reported_native_acoustic_contexts": outcome.native_acoustic_contexts,
                                "native_accounting": deepcopy(
                                    self._native_accounting.get(item_id) or {})})
            self._persist()
        return result

    def _file_failure(self, scope: ScopeState, work_item: Mapping[str, Any], *,
                      reason: str, reason_code: str, status: str = "fail",
                      diagnostic: Mapping[str, Any] | None = None) -> dict[str, Any]:
        item_id = str(work_item["work_item_id"])
        classified = classify_failure(
            failure_stage=str(work_item["stage"]), reason=reason, reason_code=reason_code)
        result = {
            "work_item_id": item_id, "stage": str(work_item["stage"]),
            "request_id": str(work_item["request_id"]),
            "scope_id": str(work_item["request_id"]), "status": status,
            "facts": {}, "outputs": {}, "reason": reason,
            "depends_on": list(work_item.get("depends_on") or []),
        }
        StageResult.from_mapping(result)
        with self._result_lock:
            scope.results.append(result)
            scope.blockers.append({
                "work_item_id": item_id, "code": reason_code, "reason": reason,
                "gap_state": classified["gap_state"],
                "diagnostic": deepcopy(classified["diagnostic"]),
                **({"detail": deepcopy(dict(diagnostic))} if diagnostic else {}),
            })
            self._worker_records.pop(item_id, None)
            self._append_event({"event": "work_item_failed", "work_item_id": item_id,
                                "status": status, "reason_code": reason_code,
                                "reason": reason, "scope": scope.scope_key,
                                "gap_state": classified["gap_state"]})
            self._persist()
        return result

    # -- retry ----------------------------------------------------------

    def _retry_policy(self, work_item: Mapping[str, Any]) -> RetryPolicy:
        declared = (work_item.get("payload") or {}).get("retry")
        if isinstance(declared, Mapping):
            return RetryPolicy.from_mapping(declared)
        return RetryPolicy()

    def _may_retry(self, scope: ScopeState, work_item: Mapping[str, Any],
                   result: Mapping[str, Any]) -> dict[str, Any]:
        """Decide whether a failure earns another attempt, and say why.

        Three different things are distinguished here, because treating them
        alike is how a defect gets hidden:

        * a **program or interface error** is never repeated with another seed;
        * an **interruption** produced no result at all, so recovering it does
          not spend the stage retry budget, only its own bound;
        * a **legally refused candidate** may be rotated a bounded number of
          times, and the rotation is recorded rather than being a quiet reseed.
        """
        item_id = str(work_item.get("work_item_id") or "")
        blocker = next(
            (
                candidate
                for candidate in reversed(scope.blockers)
                if str(candidate.get("work_item_id") or "") == item_id
            ),
            scope.blockers[-1] if scope.blockers else {},
        )
        reason_code = str(blocker.get("code") or "")
        classified = classify_failure(
            failure_stage=str(work_item["stage"]),
            reason=str(result.get("reason") or ""),
            reason_code=reason_code)
        key = str(work_item.get("unit_id") or work_item["stage"])
        if classified["diagnostic"]["classification_reason"] == "native_visibility_rejection":
            return {"retry": False, "kind": "requires_replan",
                    "reason": "the same captured plan cannot fix a rejected pixel condition",
                    "classification": classified["diagnostic"]}
        if reason_code == INTERRUPTED_REASON_CODE:
            used = scope.interrupted_attempts.get(key, 0)
            if used >= self.interrupted_retry_limit:
                return {"retry": False,
                        "reason": (f"{used} interrupted attempts of {key} are already "
                                   f"recovered, of {self.interrupted_retry_limit}"),
                        "classification": classified["diagnostic"]}
            scope.interrupted_attempts[key] = used + 1
            return {"retry": True, "kind": "interrupted_resume",
                    "reason": "the attempt was interrupted, not refused",
                    "interrupted_attempts": scope.interrupted_attempts[key],
                    "classification": classified["diagnostic"]}
        if classified["gap_state"] == "interface_not_implemented":
            return {"retry": False,
                    "reason": "a code or interface defect is not retried with another seed",
                    "classification": classified["diagnostic"]}
        policy = self._retry_policy(work_item)
        attempt = int(work_item["attempt"])
        allowance = policy.attempts_per_stage + scope.interrupted_attempts.get(key, 0)
        if attempt < allowance:
            return {"retry": True, "kind": "stage_retry",
                    "reason": f"attempt {attempt} of {allowance} under the same conditions",
                    "classification": classified["diagnostic"]}
        candidate_rejection = classified["diagnostic"]["classification_reason"] in {
            "planning_exhaustion", "clip_overflow_rejection"}
        if not candidate_rejection:
            return {"retry": False, "reason": "the stage retry budget is spent",
                    "classification": classified["diagnostic"]}
        retained = self._retained_root_for(scope, work_item)
        if retained is not None:
            return {
                "retry": False,
                "kind": "candidate_rotation_refused_retained_visual",
                "reason": (
                    "candidate rotation cannot resample a retained visual root: "
                    f"{retained}"
                ),
                "classification": classified["diagnostic"],
            }
        if scope.candidate_index >= self.candidate_rotation_limit:
            return {"retry": False,
                    "reason": (f"{scope.candidate_index} candidate rotations are "
                               f"already spent of {self.candidate_rotation_limit}"),
                    "classification": classified["diagnostic"]}
        scope.candidate_index += 1
        return {"retry": True, "kind": "candidate_rotation",
                "reason": (f"a legal candidate was refused; rotation "
                           f"{scope.candidate_index} of {self.candidate_rotation_limit}"),
                "candidate_index": scope.candidate_index,
                "classification": classified["diagnostic"]}

    def _retry_item(self, scope: ScopeState, work_item: Mapping[str, Any],
                    decision: Mapping[str, Any]) -> dict[str, Any] | None:
        """Build the next attempt through the protocol's own retry helper."""
        key = str(work_item.get("unit_id") or work_item["stage"])
        policy = self._retry_policy(work_item)
        # An interruption and a candidate rotation each add one attempt of their
        # own; the budget for repeating a real failure is untouched.
        allowance = (policy.attempts_per_stage
                     + scope.interrupted_attempts.get(key, 0)
                     + scope.candidate_index)
        item = _stage_work_item_from_dict(work_item)
        reason = f"{decision.get('kind')}: {decision.get('reason')}"
        nxt = retry_stage_work_item(
            item, retry=replace(policy, attempts_per_stage=allowance), reason=reason)
        if nxt is None:
            return None
        value = nxt.to_dict()
        if decision.get("kind") == "candidate_rotation":
            candidate = int(decision["candidate_index"])
            value["payload"]["candidate_index"] = candidate
            value["payload"]["sampling_candidate_index"] = candidate
        return value

    def _reset_core_candidate_rotation(
        self,
        scope: ScopeState,
        work_item: Mapping[str, Any],
        result: Mapping[str, Any],
        decision: Mapping[str, Any],
    ) -> None:
        """Start a whole fresh core round after a candidate refusal.

        A core plan is shared by every member. Keeping a passed sibling from
        the old round would combine candidate 0 and candidate 1 into one
        world, so the old results remain only in the durable failure record
        and the active protocol round is cleared atomically before the next
        wave is scheduled.
        """
        candidate = int(decision["candidate_index"])
        superseded = deepcopy(scope.results)
        scope.candidate_failures.append({
            "candidate_index": candidate,
            "failed_work_item_id": str(work_item["work_item_id"]),
            "reason": result.get("reason"),
            "classification": deepcopy(decision.get("classification") or {}),
            "superseded_results": superseded,
            "superseded_result_count": len(superseded),
            "recorded_at": _utc_now(),
        })
        scope.results = []
        scope.blockers = []
        scope.finished = False
        scope.finished_reason = None
        self._append_event({
            "event": "candidate_rotation_reset",
            "scope": scope.scope_key,
            "candidate_index": candidate,
            "failed_work_item_id": str(work_item["work_item_id"]),
            "superseded_result_count": len(superseded),
            "reason": result.get("reason"),
        })
        self._persist()

    def _finish_ready_item(
        self,
        scope: ScopeState,
        work_item: Mapping[str, Any],
        result: Mapping[str, Any],
        *,
        candidate_rotation_scopes: set[str] | None = None,
    ) -> None:
        """Apply bounded retry policy after a ready wave has fully filed."""
        if (
            candidate_rotation_scopes is not None
            and scope.scope_key in candidate_rotation_scopes
        ):
            return
        current = dict(work_item)
        current_result = dict(result)
        while current_result.get("status") != "pass":
            decision = self._may_retry(scope, current, current_result)
            self._append_event({
                "event": "retry_decision",
                "work_item_id": current["work_item_id"],
                **{key: value for key, value in decision.items()},
            })
            if not decision.get("retry"):
                with self._result_lock:
                    scope.finished = True
                    scope.finished_reason = str(decision.get("reason"))
                return
            if (
                decision.get("kind") == "candidate_rotation"
                and scope.scope_kind == "core_group"
            ):
                if candidate_rotation_scopes is not None:
                    candidate_rotation_scopes.add(scope.scope_key)
                self._reset_core_candidate_rotation(
                    scope, current, current_result, decision
                )
                return
            nxt = self._retry_item(scope, current, decision)
            if nxt is None:
                with self._result_lock:
                    scope.finished = True
                    scope.finished_reason = (
                        "the retry budget produced no further attempt"
                    )
                return
            current = nxt
            current_result = self.execute_work_item(scope, current)

    # -- the loop -------------------------------------------------------

    def run(self) -> dict[str, Any]:
        self.run_root.mkdir(parents=True, exist_ok=True)
        run_path = self.run_root / "run.json"
        if not run_path.exists():
            _json_write_new(run_path, self._producer_metadata())
        self._append_event({"event": "run_started",
                            "scopes": [scope.scope_key for scope in self.scopes],
                            "resumed": bool(self._resumed)})
        waves = 0
        while waves < self.max_waves:
            waves += 1
            ready: list[tuple[ScopeState, dict[str, Any]]] = []
            for scope in self.scopes:
                if scope.finished:
                    continue
                try:
                    items = self.next_work_items(scope)
                except (ProductionSpecError, ValueError) as error:
                    scope.finished = True
                    scope.finished_reason = f"{type(error).__name__}: {error}"
                    continue
                if not items:
                    blockers = self.blockers(scope)
                    scope.blockers.extend(blockers)
                    scope.finished = True
                    scope.finished_reason = (
                        "blocked: " + "; ".join(str(row.get("code")) for row in blockers)
                        if blockers else "no further units are ready")
                    continue
                for item in items:
                    ready.append((scope, item))
            if not ready:
                break
            runnable = [
                (scope, item) for scope, item in ready if not scope.finished
            ]
            if self.max_parallel <= 1 or len(runnable) <= 1:
                wave_results = [
                    (scope, item, self.execute_work_item(scope, item))
                    for scope, item in runnable
                ]
            else:
                wave_results = []
                with ThreadPoolExecutor(
                    max_workers=min(self.max_parallel, len(runnable)),
                    thread_name_prefix="avengine-stage",
                ) as executor:
                    futures = [
                        (scope, item, executor.submit(
                            self.execute_work_item, scope, item
                        ))
                        for scope, item in runnable
                    ]
                    for scope, item, future in futures:
                        wave_results.append((scope, item, future.result()))
            # Every item in this ready wave has filed before any retry or next
            # dependency wave is considered. A failure therefore cannot release
            # a dependent unit while an independent sibling is still running.
            candidate_rotation_scopes: set[str] = set()
            for scope, item, result in wave_results:
                if result.get("status") != "pass" and not scope.finished:
                    self._finish_ready_item(
                        scope, item, result,
                        candidate_rotation_scopes=candidate_rotation_scopes,
                    )
        summary = self._summary(waves)
        _json_write_atomic(self.run_root / "run_summary.json", summary)
        self._persist()
        self._append_event({"event": "run_finished", "status": summary["status"],
                            "waves": waves})
        return summary

    def _producer_metadata(self) -> dict[str, Any]:
        def git(args: list[str]) -> str | None:
            try:
                result = subprocess.run(["git", *args], cwd=self.repository,
                                        capture_output=True, text=True, check=False)
            except OSError:
                return None
            return result.stdout.strip() if result.returncode == 0 else None

        status = git(["status", "--porcelain"])
        return {
            "schema": SCHEMA,
            "run_root": str(self.run_root),
            "manifest_path": str(self.manifest_path),
            "repository": str(self.repository),
            "git_commit": git(["rev-parse", "HEAD"]),
            "working_tree_changes_at_launch": status.splitlines() if status else [],
            "python_executable": self.python_executable,
            "python_version": platform.python_version(),
            "pythonpath": os.environ.get("PYTHONPATH", ""),
            "worker_python_path": list(self.python_path),
            "argv": list(self.argv),
            "started_at": _utc_now(),
            "native_visual_world_budget": self.native_visual_world_budget,
            "max_parallel": self.max_parallel,
            "resource_policy_source": self.resource_policy_source,
            "resource_policy": self.broker.allocator.policy.to_dict(),
            "resource_policy_override": deepcopy(self.resource_policy_override),
            "stage_executors": registered_stage_executors(),
            "claim_boundary": (
                "a delivered unit is a produced and read-back artifact, not a "
                "coverage or paper-admission claim"),
        }

    @staticmethod
    def _capture_stage_evidence(scope: ScopeState) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        for row in scope.results:
            if row.get("stage") not in {"capture", "visual_capture"}:
                continue
            if row.get("status") != "pass":
                continue
            facts = row.get("facts") or {}
            outputs = row.get("outputs") or {}
            evidence.append({
                "work_item_id": str(row.get("work_item_id") or ""),
                "group_id": (
                    str(scope.scope_key)
                    if scope.scope_kind == "core_group"
                    else None
                ),
                "episode_id": (
                    str(scope.scope_key)
                    if scope.scope_kind == "episode"
                    else None
                ),
                "world_id": (
                    outputs.get("world_id")
                    or facts.get("world_id")
                    or None
                ),
                "room_id": scope.room_id,
                "task_family": scope.task_family,
                "capture_receipt_path": facts.get("capture_receipt_path"),
                "capture_root": outputs.get("capture") or outputs.get("capture_root"),
                "stage_root": outputs.get("capture_root") or outputs.get("capture"),
                "request_path": outputs.get("request_path"),
                "episode_plan": outputs.get("episode_plan"),
                "native_visual_worlds_created": int(
                    outputs.get("native_visual_worlds_created") or 0
                ),
                "captured_frame_count": facts.get("captured_frame_count"),
            })
        return evidence

    @staticmethod
    def _audio_stage_evidence(scope: ScopeState) -> list[dict[str, Any]]:
        # Assembly may materialize a canonical shared column after proving
        # event-window acoustic equivalence. Export that accepted media, not
        # the earlier per-variant render kept for audit.
        for row in reversed(scope.results):
            if row.get("stage") == "assembly" and row.get("status") == "pass":
                canonical = (row.get("outputs") or {}).get("audio_stage_evidence")
                if isinstance(canonical, list) and canonical:
                    return deepcopy(canonical)
        evidence: list[dict[str, Any]] = []
        for row in scope.results:
            if row.get("stage") != "audio" or row.get("status") != "pass":
                continue
            facts = row.get("facts") or {}
            outputs = row.get("outputs") or {}
            work_item_id = str(row.get("work_item_id") or "")
            prefix = work_item_id.split(":", 1)[0]
            member_id = (
                outputs.get("member_id")
                or outputs.get("member_request_id")
                or (prefix.rsplit("/", 1)[-1] if "/" in prefix else None)
            )
            evidence.append(
                {
                    "work_item_id": work_item_id,
                    "group_id": (
                        str(scope.scope_key)
                        if scope.scope_kind == "core_group"
                        else None
                    ),
                    "member_id": None if member_id is None else str(member_id),
                    "episode_id": (
                        str(scope.scope_key)
                        if scope.scope_kind == "episode" else None
                    ),
                    "audio_path": outputs.get("audio"),
                    "facts_path": (
                        facts.get("facts_path")
                        or outputs.get("facts_path")
                    ),
                    "audio_report_path": (
                        facts.get("audio_report_path")
                        or outputs.get("audio_report")
                    ),
                    "declared_audio_delivery": deepcopy(
                        facts.get("declared_audio_delivery")
                        or outputs.get("declared_audio_delivery")
                        or {}
                    ),
                    "delivered_audio_layouts": deepcopy(
                        facts.get("delivered_audio_layouts")
                        or outputs.get("delivered_audio_layouts")
                        or {}
                    ),
                    "ancillary_audio_outputs": deepcopy(
                        outputs.get("ancillary_audio_outputs")
                        or facts.get("ancillary_audio_outputs")
                        or []
                    ),
                }
            )
        return evidence

    def _summary(self, waves: int) -> dict[str, Any]:
        scopes = []
        delivered_groups = []
        delivered_episodes = []
        for scope in self.scopes:
            passed = [row for row in scope.results if row.get("status") == "pass"]
            audio_stage_evidence = self._audio_stage_evidence(scope)
            capture_stage_evidence = self._capture_stage_evidence(scope)
            if scope.scope_kind == "core_group":
                group_world_id = declared_world_id(self.manifest, scope.scope_key)
                for entry in capture_stage_evidence:
                    entry["world_id"] = entry.get("world_id") or group_world_id
            completion_stage = "assembly" if scope.scope_kind == "core_group" else "delivery"
            completed = [row for row in passed if row.get("stage") == completion_stage]
            scopes.append({
                **scope.to_dict(),
                "passed_units": len(passed),
                "filed_results": len(scope.results),
                "delivered": bool(completed),
            })
            if scope.scope_kind == "core_group" and completed:
                row = completed[-1]
                assembly_outputs = row.get("outputs") or {}
                assembly_facts = row.get("facts") or {}
                audio_delivery_by_member = (
                    assembly_outputs.get("audio_delivery_by_member")
                    or (assembly_facts.get("validation") or {}).get(
                        "audio_delivery_by_member"
                    )
                    or {}
                )
                audio_stage_with_assembly = list(audio_stage_evidence)
                for member_id, mapping in (
                    audio_delivery_by_member.items()
                    if isinstance(audio_delivery_by_member, Mapping)
                    else ()
                ):
                    if not isinstance(mapping, Mapping):
                        continue
                    if any(
                        entry.get("group_id") == scope.scope_key
                        and entry.get("member_id") == str(member_id)
                        for entry in audio_stage_with_assembly
                    ):
                        continue
                    audio_stage_with_assembly.append({
                        "group_id": scope.scope_key,
                        "member_id": str(member_id),
                        "episode_id": None,
                        "work_item_id": row.get("work_item_id"),
                        "audio_path": mapping.get("primary_audio_path"),
                        "facts_path": mapping.get("facts_path"),
                        "audio_report_path": mapping.get("audio_report_path"),
                        "declared_audio_delivery": deepcopy(
                            mapping.get("declared_audio_delivery") or {}
                        ),
                        "delivered_audio_layouts": deepcopy(
                            mapping.get("delivered_audio_layouts") or {}
                        ),
                        "ancillary_audio_outputs": deepcopy(
                            mapping.get("ancillary_audio_outputs") or []
                        ),
                        "source": "assembly.audio_delivery_by_member",
                    })
                delivered_groups.append({
                    "group_id": scope.scope_key,
                    "task_family": scope.task_family,
                    "room_id": scope.room_id,
                    "world_id": (
                        assembly_outputs.get("world_id")
                        or declared_world_id(self.manifest, scope.scope_key)
                    ),
                    "assembled_path": row["facts"].get("assembled_path"),
                    "group_spec_path": row["facts"].get("group_spec_path"),
                    "capture_stage_evidence": deepcopy(capture_stage_evidence),
                    "audio_stage_evidence": deepcopy(audio_stage_with_assembly),
                    "audio_delivery_by_member": deepcopy(audio_delivery_by_member),
                })
            elif scope.scope_kind == "episode" and completed:
                row = completed[-1]
                outputs = row.get("outputs") or {}
                for entry in capture_stage_evidence:
                    entry["episode_root"] = outputs.get("episode_root")
                    entry["world_id"] = (
                        entry.get("world_id")
                        or outputs.get("world_id")
                        or scope.row.get("world_id")
                        if isinstance(scope.row, Mapping)
                        else entry.get("world_id")
                    )
                delivered_episodes.append({
                    "episode_id": scope.scope_key,
                    "facts_path": outputs.get("facts_path") or row["facts"].get("facts_path"),
                    "questions_path": outputs.get("questions_path") or row["facts"].get("questions_path"),
                    "video_path": outputs.get("preview_path") or outputs.get("video_path"),
                    "audio_path": outputs.get("audio_path"),
                    "room_family": outputs.get("room_family"),
                    "room_id": outputs.get("room_id") or scope.room_id,
                    "world_id": outputs.get("world_id") or scope.scope_key,
                    "episode_root": outputs.get("episode_root"),
                    "capture_stage_evidence": deepcopy(capture_stage_evidence),
                    "audio_stage_evidence": deepcopy(audio_stage_evidence),
                })
        blocked = [scope for scope in scopes if not scope["delivered"]]
        return {
            "schema": SCHEMA,
            "status": "complete" if not blocked else "completed_with_diagnostics",
            "run_root": str(self.run_root),
            "manifest_path": str(self.manifest_path),
            "batch_id": self.manifest.get("batch_id"),
            "waves": waves,
            "scopes": scopes,
            "delivered_groups": delivered_groups,
            "delivered_episodes": delivered_episodes,
            "native_visual_worlds_used": self.native_visual_worlds_used,
            "native_acoustic_contexts_used": self.native_acoustic_contexts_used,
            "native_visual_world_budget": self.native_visual_world_budget,
            "native_accounting_totals": self._refresh_native_accounting_totals(),
            "native_accounting": deepcopy(self._native_accounting),
            "state_path": str(self.state_path),
            "journal_path": str(self.events_path),
            "claim_boundary": (
                "delivered groups here are produced and verified units; V1 coverage "
                "and paper admission are separate and are not claimed by this run"),
        }

    # -- recovery -------------------------------------------------------

    @classmethod
    def resume(
        cls, run_root: str | Path, *, manifest: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> "ProductionRunner":
        """Pick up an interrupted run: one valid round, live workers adopted."""
        root = Path(run_root).expanduser().resolve()
        state = _read_json(root / "state.json")
        manifest_path = Path(state["manifest_path"])
        loaded = (
            manifest
            if manifest is not None
            else deepcopy(state.get("manifest_snapshot"))
            if isinstance(state.get("manifest_snapshot"), Mapping)
            else _read_json(manifest_path)
        )
        resume_kwargs = dict(kwargs)
        if "recipe_options" not in resume_kwargs:
            saved_options = state.get("recipe_options")
            if isinstance(saved_options, Mapping):
                resume_kwargs["recipe_options"] = deepcopy(dict(saved_options))
        if "max_parallel" not in resume_kwargs and state.get("max_parallel") is not None:
            resume_kwargs["max_parallel"] = max(1, int(state["max_parallel"]))
        # A resumed run keeps the policy it actually ran under. It is
        # reapplied through the same override argument rather than by reading
        # the effective policy back, so a resume and a fresh start go through
        # one code path and a bad override fails the same way in both.
        # An explicit None counts as "not given". The command line always
        # passes this argument, and passes None when the flag is absent, so a
        # membership test on the keyword silently stopped every resume from
        # picking the policy back up -- the run came back on the manifest's
        # policy and the display exception was gone.
        if (
            resume_kwargs.get("resource_policy_override") is None
            and resume_kwargs.get("broker") is None
            and isinstance(state.get("resource_policy_override"), Mapping)
        ):
            resume_kwargs["resource_policy_override"] = deepcopy(
                dict(state["resource_policy_override"])
            )
        runner = cls(manifest=loaded, manifest_path=manifest_path, run_root=root,
                     **resume_kwargs)
        by_key = {scope.scope_key: scope for scope in runner.scopes}
        restored_rounds: dict[str, Any] = {}
        for saved in state.get("scopes", []):
            scope = by_key.get(str(saved.get("scope_key")))
            if scope is None:
                continue
            scope.results = deepcopy(list(saved.get("results") or []))
            scope.blockers = deepcopy(list(saved.get("blockers") or []))
            scope.candidate_index = int(saved.get("candidate_index") or 0)
            scope.candidate_failures = deepcopy(
                list(saved.get("candidate_failures") or [])
            )
            scope.retry_offsets = {
                str(key): int(value)
                for key, value in (saved.get("retry_offsets") or {}).items()
            }
            scope.retired_failures = deepcopy(
                list(saved.get("retired_failures") or [])
            )
            scope.retired_results = deepcopy(
                list(saved.get("retired_results") or [])
            )
            # finished is recomputed: the protocol decides what may run now.
            scope.finished = False
            scope.finished_reason = None
            restored_rounds[scope.scope_key] = runner._round_state(scope)
        saved_accounting = state.get("native_accounting")
        if isinstance(saved_accounting, Mapping) and saved_accounting:
            runner._native_accounting = {}
            for key, raw in saved_accounting.items():
                if not isinstance(raw, Mapping):
                    continue
                record = deepcopy(dict(raw))
                record.setdefault("work_item_id", str(key))
                record.setdefault("status", "completed")
                record.setdefault(
                    "expected_native_visual_worlds",
                    int(record.get("native_visual_worlds") or 0),
                )
                record.setdefault(
                    "native_acoustic_launch_attempts",
                    int(record.get("expected_native_acoustic_contexts") or 0),
                )
                record.setdefault(
                    "actual_native_acoustic_contexts",
                    None,
                )
                record.setdefault(
                    "native_acoustic_contexts_lower_bound",
                    0 if record.get("actual_native_acoustic_contexts") is None
                    else int(record.get("actual_native_acoustic_contexts") or 0),
                )
                record.setdefault(
                    "effective_native_visual_worlds",
                    int(record.get("expected_native_visual_worlds") or 0),
                )
                record.setdefault(
                    "effective_capture_instances",
                    int(record.get("capture_instances") or 0),
                )
                for column in ("actual_native_visual_worlds",
                               "actual_native_visual_worlds_source",
                               "actual_native_visual_worlds_evidence_path"):
                    record.setdefault(column, None)
                if "logical_world_attempts_source" not in record:
                    record["logical_world_attempts"] = None
                    record["logical_world_attempts_source"] = (
                        "budget_unit_pending"
                    )
                record.setdefault("worker_logs", {})
                runner._native_accounting[str(key)] = record
            carryover = state.get("native_accounting_carryover")
            if isinstance(carryover, Mapping):
                runner._native_accounting_baseline_visual = int(
                    carryover.get("native_visual_worlds") or 0
                )
                runner._native_accounting_baseline_acoustic = int(
                    carryover.get("native_acoustic_contexts") or 0
                )
            else:
                # A partially upgraded state may already contain records but
                # lack the carryover field. Preserve the old aggregate minus
                # the records that are now individually represented.
                represented = runner._native_accounting_totals()
                runner._native_accounting_baseline_visual = max(
                    0,
                    int(state.get("native_visual_worlds_used") or 0)
                    - int(represented["native_visual_worlds"]),
                )
                runner._native_accounting_baseline_acoustic = max(
                    0,
                    int(state.get("native_acoustic_contexts_used") or 0)
                    - int(represented["native_acoustic_contexts_known"]),
                )
        else:
            # Old states only have aggregate counters. Keep them as a
            # baseline while new work items are tracked by id.
            runner._native_accounting_baseline_visual = int(
                state.get("native_visual_worlds_used") or 0
            )
            runner._native_accounting_baseline_acoustic = int(
                state.get("native_acoustic_contexts_used") or 0
            )
        runner._refresh_native_accounting_totals()
        runner._worker_records = {
            key: dict(value) for key, value in (state.get("worker_records") or {}).items()}
        # A legacy argv-marker bug may have filed a false interruption while
        # the exact worker continued rendering. Retire that diagnostic and let
        # the normal executor adopt/validate its result at the same attempt.
        for scope in runner.scopes:
            kept = []
            for row in scope.results:
                item_id = str(row.get("work_item_id") or "")
                reason = str(row.get("reason") or "")
                work_dir = runner.run_root / "workers" / item_id.replace("/", "__").replace(":", "_")
                interrupted = row.get("status") != "pass" and "was interrupted before it" in reason
                live_record = _find_live_stage_worker(work_dir) if interrupted else None
                completed = False
                result_path = work_dir / "stage_result.json"
                if interrupted and result_path.is_file():
                    try:
                        completed = _read_json(result_path).get("status") == "ok"
                    except (OSError, ValueError, TypeError):
                        pass
                if interrupted and (live_record or completed):
                    old_row = deepcopy(row)
                    old_row["retired_reason"] = "interruption disproved by exact live worker or completed worker result"
                    scope.retired_failures.append(old_row)
                    scope.finished = False
                    scope.finished_reason = None
                    scope.blockers = [v for v in scope.blockers if v.get("work_item_id") != item_id]
                    if live_record:
                        runner._worker_records[item_id] = live_record
                    runner._append_event({"event": "false_interruption_recovered",
                                          "work_item_id": item_id,
                                          "worker_still_live": bool(live_record),
                                          "completed_result": completed})
                else:
                    kept.append(row)
            scope.results = kept
        live = {key: value for key, value in runner._worker_records.items()
                if process_matches(value)}
        gone = sorted(set(runner._worker_records) - set(live))
        for key in gone:
            runner._worker_records.pop(key, None)
        snapshot = deepcopy(state.get("resource_snapshot") or {})
        orphaned_lease_ids: list[str] = []
        if isinstance(snapshot, Mapping):
            kept_leases = []
            for raw in snapshot.get("live_leases") or ():
                if not isinstance(raw, Mapping):
                    continue
                lease_id = str(raw.get("lease_id") or "")
                if lease_id in live:
                    kept_leases.append(raw)
                elif lease_id:
                    orphaned_lease_ids.append(lease_id)
            snapshot["live_leases"] = kept_leases
        adoption = runner.broker.restore(
            snapshot, is_running=lambda pid: any(
                int(record.get("pid") or -1) == int(pid) for record in live.values()))
        runner._resumed = {
            "resumed_at": _utc_now(),
            "state_path": str(root / "state.json"),
            "valid_round": restored_rounds,
            "live_worker_work_item_ids": sorted(live),
            "dropped_worker_work_item_ids": gone,
            "orphaned_lease_ids": orphaned_lease_ids,
            "lease_adoption": deepcopy(adoption),
        }
        runner._append_event({"event": "run_resumed", **deepcopy(runner._resumed)})
        return runner


def _stage_work_item_from_dict(value: Mapping[str, Any]) -> StageWorkItem:
    """Rebuild the protocol's work item so its own retry helper can be used."""
    resource = dict(value["resource"])
    return StageWorkItem(
        work_item_id=str(value["work_item_id"]),
        stage=str(value["stage"]),
        request_id=str(value["request_id"]),
        attempt=int(value["attempt"]),
        resource=ResourceRequest.from_mapping(resource, kind=str(resource["kind"])),
        fresh_output_relative=str(value["fresh_output_relative"]),
        depends_on=tuple(str(item) for item in value.get("depends_on") or ()),
        inputs=deepcopy(dict(value.get("inputs") or {})),
        payload=deepcopy(dict(value.get("payload") or {})),
        group_id=value.get("group_id"),
        task_family=value.get("task_family"),
        unit_id=value.get("unit_id"),
        member_request_ids=tuple(str(item) for item in value.get("member_request_ids") or ()),
    )


def _declared_peak_vram_mb(work_item: Mapping[str, Any]) -> int | None:
    """Only a field that means peak becomes a peak estimate.

    `min_free_vram_mb` is a start-up floor. Passing it here would under-reserve
    a shared device by exactly the amount the floor is smaller than the peak.
    """
    resource = work_item.get("resource") or {}
    value = resource.get("estimated_peak_vram_mb")
    return None if value is None else int(value)


def _worker_error_text(record: Mapping[str, Any], result_path: Path) -> str:
    tail = ""
    for key in ("stderr_log", "stdout_log"):
        path = record.get(key)
        if not path:
            continue
        try:
            text = Path(str(path)).read_text(encoding="utf-8", errors="replace")[-4000:]
        except OSError:
            continue
        if text.strip():
            tail = text.strip().splitlines()[-1]
            break
    return (f"the worker wrote no {result_path.name} "
            f"(returncode {record.get('returncode')})"
            + (f": {tail}" if tail else ""))


def _resolve_room_runtime_report(request: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve one room to the runtime report the worker identity comes from."""
    from avengine.rooms.room_providers import load_room_catalog, resolve_catalog_room

    catalog_path = (request or {}).get("room_catalog")
    if not catalog_path:
        catalog_path = REPOSITORY / "examples/rooms/packages/catalog.json"
    catalog = load_room_catalog(catalog_path)
    resolution = resolve_catalog_room(
        catalog, str(request["room_id"]), catalog_path=catalog_path,
        runtime=(request or {}).get("runtime"), request=request)
    if resolution.runtime is None:
        raise ProductionRunError(
            f"room {request.get('room_id')!r} did not resolve a runtime: "
            f"{resolution.reason or resolution.status}")
    return dict(resolution.runtime)


# ---------------------------------------------------------------------------
# Coverage feedback (P19) and delivery export (P20)
# ---------------------------------------------------------------------------

DEFAULT_QA_SAMPLING = {"time_display_precision": 0, "angle_display_precision": 0}

_COVERAGE_PROVIDER: Callable[..., Mapping[str, Any]] | None = None


def set_coverage_feedback_provider(
    provider: Callable[..., Mapping[str, Any]] | None,
) -> None:
    """Let P19 own the deficit numbers once it has them.

    Until then this module computes the same quantities from the delivered
    bundles, so the loop can decide what to run next without a person reading a
    report. A registered provider replaces the computation, not the shape.
    """
    global _COVERAGE_PROVIDER
    _COVERAGE_PROVIDER = provider


def _bundle_paths(delivered_groups: Sequence[Mapping[str, Any]]) -> list[Path]:
    paths = []
    for group in delivered_groups:
        assembled = group.get("assembled_path")
        if not assembled:
            continue
        bundle = Path(str(assembled)) / "binding_groups.json"
        if bundle.is_file():
            paths.append(bundle)
    return paths


def _world_rows(bundle_paths: Sequence[Path], *, origin: str) -> dict[str, dict[str, Any]]:
    """One row per physical world, whichever bundle it came in.

    A world that appears in a retained bundle and again in this run is one
    world. Counting it twice is how an import silently doubles coverage.
    """
    worlds: dict[str, dict[str, Any]] = {}
    for path in bundle_paths:
        bundle = _read_json(path)
        for group in bundle.get("groups", []):
            world_id = str(group.get("world_id") or "")
            if not world_id:
                continue
            row = worlds.setdefault(world_id, {
                "world_id": world_id, "origin": origin, "group_ids": [],
                "task_families": [], "room_ids": [], "qa_ids": [],
                "member_count": 0, "member_keys": [], "main_questions": [],
                "bundle_paths": [],
            })
            if str(path) not in row["bundle_paths"]:
                row["bundle_paths"].append(str(path))
            for key, value in (("group_ids", group.get("group_id")),
                               ("task_families", group.get("task_family")),
                               ("room_ids", group.get("room_id"))):
                if value and str(value) not in row[key]:
                    row[key].append(str(value))
            for member in group.get("members", []):
                qa_id = (member.get("question") or {}).get("qa_id")
                # One member of one group is one main question, however many
                # bundles carry it. A retained bundle re-imported beside the run
                # that produced it must not double the count.
                key = f"{group.get('group_id')}/{member.get('member_id')}"
                if key not in row["member_keys"]:
                    row["member_keys"].append(key)
                    row["member_count"] += 1
                    row["main_questions"].append({"member_key": key,
                                                  "qa_id": None if qa_id is None else str(qa_id)})
                if qa_id and str(qa_id) not in row["qa_ids"]:
                    row["qa_ids"].append(str(qa_id))
    return worlds


def _source_classes_by_group(manifest: Mapping[str, Any]) -> dict[str, list[str]]:
    classes: dict[str, list[str]] = {}
    for row in manifest.get("episodes", []):
        group_id = row.get("group_id")
        if not group_id:
            continue
        declared = row.get("source_classes")
        if isinstance(declared, Sequence) and not isinstance(declared, (str, bytes)):
            classes[str(group_id)] = [str(value) for value in declared]
    return classes


def _legacy_coverage_feedback(
    *,
    manifest: Mapping[str, Any],
    delivered_groups: Sequence[Mapping[str, Any]],
    imported_bundle_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """What the run really delivered against the configured quota, and what is short."""
    if _COVERAGE_PROVIDER is not None:
        return dict(_COVERAGE_PROVIDER(
            manifest=manifest, delivered_groups=delivered_groups,
            imported_bundle_paths=imported_bundle_paths))
    quota = dict((manifest.get("production") or {}).get("coverage_quota") or {})
    fresh = _world_rows(_bundle_paths(delivered_groups), origin="this_run")
    imported = _world_rows([Path(str(value)) for value in imported_bundle_paths],
                           origin="imported")
    worlds = dict(imported)
    for world_id, row in fresh.items():
        if world_id not in worlds:
            worlds[world_id] = row
            continue
        merged = dict(worlds[world_id])
        merged["origin"] = "imported_and_rerun"
        merged["bundle_paths"] = sorted(
            set(merged["bundle_paths"]) | set(row["bundle_paths"]))
        seen = list(merged["member_keys"])
        questions = list(merged["main_questions"])
        for entry in row["main_questions"]:
            if entry["member_key"] not in seen:
                seen.append(entry["member_key"])
                questions.append(entry)
        merged["member_keys"] = seen
        merged["main_questions"] = questions
        merged["member_count"] = len(seen)
        merged["qa_ids"] = sorted(set(merged["qa_ids"]) | set(row["qa_ids"]))
        merged["group_ids"] = sorted(set(merged["group_ids"]) | set(row["group_ids"]))
        worlds[world_id] = merged

    classes_by_group = _source_classes_by_group(manifest)
    main_questions: dict[str, int] = {}
    worlds_by_qa: dict[str, set[str]] = {}
    worlds_by_class: dict[str, set[str]] = {}
    worlds_by_combination: dict[str, set[str]] = {}
    groups_by_family_room: dict[str, set[str]] = {}
    for world_id, row in worlds.items():
        for qa_id in row["qa_ids"]:
            worlds_by_qa.setdefault(qa_id, set()).add(world_id)
        for group_id in row["group_ids"]:
            for family in row["task_families"]:
                for room_id in row["room_ids"]:
                    groups_by_family_room.setdefault(
                        f"{family}|{room_id}", set()).add(group_id)
            for source_class in classes_by_group.get(group_id, []):
                worlds_by_class.setdefault(source_class, set()).add(world_id)
            declared = classes_by_group.get(group_id) or []
            if len(declared) >= 2:
                worlds_by_combination.setdefault(
                    "+".join(sorted(declared)), set()).add(world_id)
    # Counted off the deduplicated world rows, so one member is one question.
    for row in worlds.values():
        for entry in row["main_questions"]:
            if entry["qa_id"]:
                main_questions[entry["qa_id"]] = main_questions.get(entry["qa_id"], 0) + 1

    deficits: list[dict[str, Any]] = []

    def shortfall(key: str, subject: str, required: Any, achieved: int) -> None:
        if not isinstance(required, int) or required <= achieved:
            return
        deficits.append({"quota_key": key, "subject": subject, "required": int(required),
                         "achieved": int(achieved), "deficit": int(required) - int(achieved)})

    for qa_id in sorted(set(worlds_by_qa) | set(main_questions)):
        shortfall("min_main_questions_per_qa_id", qa_id,
                  quota.get("min_main_questions_per_qa_id"), main_questions.get(qa_id, 0))
        shortfall("min_worlds_per_qa_id", qa_id, quota.get("min_worlds_per_qa_id"),
                  len(worlds_by_qa.get(qa_id, ())))
    declared_families = {
        f"{row.get('task_family')}|{row.get('room_id')}"
        for row in manifest.get("episodes", []) if row.get("group_id")
    }
    for key in sorted(declared_families):
        shortfall("min_groups_per_task_family_and_room", key,
                  quota.get("min_groups_per_task_family_and_room"),
                  len(groups_by_family_room.get(key, ())))
    for source_class in sorted(worlds_by_class) or sorted(
            {value for values in classes_by_group.values() for value in values}):
        shortfall("min_worlds_per_source_class", source_class,
                  quota.get("min_worlds_per_source_class"),
                  len(worlds_by_class.get(source_class, ())))
    for combination in sorted(worlds_by_combination):
        shortfall("min_worlds_per_two_entity_combination", combination,
                  quota.get("min_worlds_per_two_entity_combination"),
                  len(worlds_by_combination.get(combination, ())))
    shortfall("min_fresh_core_worlds", "fresh_core_worlds",
              quota.get("min_fresh_core_worlds"), len(fresh))

    return {
        "schema": SCHEMA,
        "provider": "production_runner.coverage_feedback",
        "quota": quota,
        "world_count": len(worlds),
        "fresh_world_count": len(fresh),
        "imported_world_count": len(imported),
        "worlds": [worlds[key] for key in sorted(worlds)],
        "main_questions_by_qa_id": dict(sorted(main_questions.items())),
        "worlds_by_qa_id": {key: sorted(value) for key, value in sorted(worlds_by_qa.items())},
        "groups_by_task_family_and_room": {
            key: sorted(value) for key, value in sorted(groups_by_family_room.items())},
        "worlds_by_source_class": {
            key: sorted(value) for key, value in sorted(worlds_by_class.items())},
        "worlds_by_two_entity_combination": {
            key: sorted(value) for key, value in sorted(worlds_by_combination.items())},
        "deficits": deficits,
        "counting_note": (
            "a world delivered here and also present in an imported bundle is counted "
            "once; fresh_world_count counts only worlds this run produced"),
        "claim_boundary": (
            "these are produced-artifact counts; human answerability and paper "
            "admission are not decided here"),
    }



def _coverage_document(value: Any, *, owner: str) -> Any:
    """Read one explicitly configured P19 document."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    if not isinstance(value, (str, Path)):
        raise ProductionRunError(f"{owner} must be a JSON path or mapping")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY / path
    path = path.resolve()
    if not path.is_file():
        raise ProductionRunError(f"{owner} is missing: {path}")
    return _read_json(path)


def _p19_coverage_inputs(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    values = (
        manifest.get("p19_coverage"),
        manifest.get("coverage_inputs"),
        (manifest.get("production") or {}).get("p19_coverage"),
        (manifest.get("production") or {}).get("coverage_inputs"),
    )
    for value in values:
        if isinstance(value, Mapping) and value.get("enabled", True) is not False:
            return deepcopy(dict(value))
    return None


# What the UE/SPEAR visual producer actually writes beside its receipt. The
# receipt itself carries no `schema` key, so the producer is recognised by the
# shape of its own output rather than by a string it never wrote.
UE_NATIVE_RECEIPT_SIGNATURE = ("backend_role", "scene", "native_pixel", "media", "clock")
UE_NATIVE_SIBLING_READBACKS = (
    "neutral_readback.json",
    "frame_readbacks.json",
    "native_runtime_binding_readback.json",
    "native_pixel_runtime_readbacks.json",
    "pixel_visibility_truth.json",
)
# `comparison_visual` is the MP3D-in-UE diagnostic. It is never production
# output and never admission evidence, so it is refused here by name rather
# than being allowed to pass a structural check it would otherwise satisfy.
UE_PRODUCTION_BACKEND_ROLES = frozenset({"production_visual"})


def _resolve_capture_artifact(capture_root: Path, value: str) -> Path:
    """Resolve one artifact a receipt names, however the producer wrote it.

    The Habitat receipt names its artifacts relative to the capture root; the
    UE receipt writes absolute paths through the external data root. Both are
    resolved, which also follows the repository's `tmp` compatibility symlink
    so a repository-relative record and an external-storage record compare as
    the same file rather than as two.
    """

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = capture_root / path
    try:
        return path.resolve()
    except OSError:
        return path


def _looks_like_ue_native_capture(receipt: Mapping[str, Any]) -> bool:
    """Recognise the UE/SPEAR visual producer by the shape it writes.

    Recognition is not acceptance: it only routes the receipt to the UE
    checks below, which are as strict as the Habitat ones.
    """

    if not all(key in receipt for key in UE_NATIVE_RECEIPT_SIGNATURE):
        return False
    return all(
        isinstance(receipt.get(key), Mapping)
        for key in ("scene", "native_pixel", "media", "clock")
    )


def _p19_qualification_matrix(
    inputs: Mapping[str, Any], *, inventory: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Get the measured asset matrix: read one, or build one from observations.

    Reading a matrix the qualification run already produced is the ordinary
    path. Building one here is for a configuration that names the raw
    observation documents instead, and it goes through the same builder, so
    neither route can invent a verdict: the builder refuses evidence that
    carries a status of its own.
    """

    document = _coverage_document(
        inputs.get("source_type_qualification_matrix")
        or inputs.get("source_type_qualification"),
        owner="P19 source_type_qualification_matrix",
    )
    if document is not None:
        return document
    build = inputs.get("source_type_qualification_build")
    if not isinstance(build, Mapping):
        return None
    from avengine.dataset.source_asset_qualification import (
        build_qualification_matrix,
    )

    def document_for(name: str) -> Any:
        return _coverage_document(
            build.get(name), owner=f"P19 source_type_qualification_build.{name}"
        )

    registry = document_for("registry")
    if registry is None:
        raise ProductionRunError(
            "source_type_qualification_build needs a registry to measure against"
        )
    return build_qualification_matrix(
        registry=registry,
        source_type_inventory=inventory,
        asset_worklist=document_for("asset_worklist") or {},
        config=document_for("config") or {},
        geometry_measurements=document_for("geometry_measurements"),
        support_catalogs=document_for("support_catalogs"),
        placement_plans=document_for("placement_plans"),
        retained_readback=document_for("retained_readback"),
        floor_reference_m=build.get("floor_reference_m"),
    )


def _p19_fresh_world_declarations(
    inputs: Mapping[str, Any],
    *,
    auto_evidence: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Validate fresh capture provenance from the native receipt and stage root."""
    entries: dict[str, dict[str, Any]] = {}
    raw_ids = inputs.get("fresh_world_ids") or ()
    explicit_evidence = inputs.get("fresh_world_evidence") or {}
    for raw in raw_ids:
        world_id = str(raw)
        if world_id.strip():
            entry = (
                explicit_evidence.get(world_id)
                if isinstance(explicit_evidence, Mapping)
                else {}
            )
            entries[world_id] = dict(entry) if isinstance(entry, Mapping) else {
                "capture_receipt_path": entry
            }
    for raw in auto_evidence:
        if not isinstance(raw, Mapping):
            continue
        world_id = raw.get("world_id")
        if not isinstance(world_id, str) or not world_id.strip():
            continue
        # Actual stage capture evidence wins over user-only declarations.
        entries.setdefault(world_id, dict(raw))

    validated: set[str] = set()
    invalid: dict[str, str] = {}
    validation: dict[str, Any] = {}
    canonical_by_receipt: dict[str, str] = {}
    receipt_to_worlds: dict[str, list[str]] = {}
    for world_id, entry in entries.items():
        value = entry.get("capture_receipt_path")
        if isinstance(value, str) and value.strip():
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = REPOSITORY / path
            key = str(path.resolve())
            receipt_to_worlds.setdefault(key, []).append(world_id)
    receipt_conflicts = [
        {"receipt_path": path, "world_ids": sorted(world_ids)}
        for path, world_ids in sorted(receipt_to_worlds.items())
        if len(set(world_ids)) > 1
    ]

    def resolve_path(value: Any, base: Path) -> Path | None:
        if not isinstance(value, str) or not value.strip():
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = base / path
        return path.resolve()

    def fail(world_id: str, reason: str, **detail: Any) -> None:
        invalid[world_id] = reason
        validation[world_id] = {"status": "invalid", "reason": reason, **detail}

    for world_id, entry in sorted(entries.items()):
        receipt_path = resolve_path(entry.get("capture_receipt_path"), REPOSITORY)
        if receipt_path is None or not receipt_path.is_file():
            fail(world_id, "capture receipt path is missing or unreadable")
            continue
        try:
            receipt = _read_json(receipt_path)
        except Exception as error:
            fail(world_id, f"capture receipt is not JSON: {error}")
            continue
        schema = str(receipt.get("schema") or "")
        is_habitat = schema == "avengine_mp3d_multi_actor_native_capture_v1"
        is_m51 = bool(schema) and schema.startswith("avengine_m5_1_")
        is_ue_native = (
            not (is_habitat or is_m51)
            and _looks_like_ue_native_capture(receipt)
        )
        if not (is_habitat or is_m51 or is_ue_native):
            fail(world_id, "capture receipt schema is unsupported for fresh provenance",
                 schema=schema)
            continue
        capture_root = receipt_path.parent
        ue_observed_room: str | None = None
        stage_record = None
        stage_record_path = resolve_path(
            entry.get("stage_run_record_path")
            or entry.get("group_stage_run_record_path"),
            REPOSITORY,
        )
        if stage_record_path is None:
            for parent in list(receipt_path.parents)[:8]:
                candidate = parent.parent / f"{parent.name}_stage_run_001.json"
                if candidate.is_file():
                    stage_record_path = candidate.resolve()
                    break
        if stage_record_path is not None and stage_record_path.is_file():
            try:
                stage_record = _read_json(stage_record_path)
            except Exception:
                stage_record = None
        capture = receipt.get("capture") or {}
        if is_habitat:
            if receipt.get("status") != "research_only" or not receipt.get("research_only"):
                fail(world_id, "capture receipt is not a research-only native capture")
                continue
            if receipt.get("artifact_role") != "observed_native_habitat_capture":
                fail(world_id, "capture receipt artifact_role is not observed native capture")
                continue
            if capture.get("native_habitat_started") is not True:
                fail(world_id, "capture receipt does not prove native Habitat started")
                continue
            required_artifacts = (
                "neutral_readback", "frame_records", "rgb", "depth",
                "semantic", "pixel_masks",
            )
            artifacts = receipt.get("artifacts") or {}
            missing_artifacts = [
                key for key in required_artifacts
                if not isinstance(artifacts.get(key), str)
                or not (capture_root / artifacts[key]).is_file()
            ]
            if missing_artifacts:
                fail(world_id, "native capture artifacts are incomplete",
                     missing_artifacts=missing_artifacts)
                continue
            neutral = receipt.get("neutral_readback") or {}
            if neutral.get("status") != "pass":
                fail(world_id, "neutral_readback is not pass")
                continue
            plan_hint = neutral.get("plan")
        elif is_ue_native:
            # This producer writes status "research_only"; it has never
            # written "pass", which is why the old backend branch could not
            # see a single real UE capture.
            if (
                str(receipt.get("status") or "") not in ("research_only", "pass")
                or receipt.get("research_only") is not True
                or receipt.get("qualification_claim") is not False
            ):
                fail(world_id, "UE capture receipt is not a research-only capture",
                     status=receipt.get("status"))
                continue
            backend_role = str(receipt.get("backend_role") or "")
            if backend_role not in UE_PRODUCTION_BACKEND_ROLES:
                fail(world_id,
                     "UE capture receipt is not production visual output",
                     backend_role=backend_role)
                continue
            scene = receipt.get("scene") or {}
            if str(scene.get("map_path_status") or "") != "launched":
                fail(world_id, "UE capture receipt does not prove the level launched",
                     map_path_status=scene.get("map_path_status"))
                continue
            if not str(scene.get("map_path") or "").strip():
                fail(world_id, "UE capture receipt names no map path")
                continue
            level = receipt.get("native_level_readback") or {}
            if (
                level.get("status") != "pass"
                or not str(level.get("observed") or "").strip()
                or level.get("observed") != level.get("expected")
            ):
                fail(world_id,
                     "the level UE actually loaded is not the level requested",
                     expected=level.get("expected"), observed=level.get("observed"))
                continue
            # The engine's own answer to "which level am I in" is the room
            # authority for a UE capture. The scene block is what the launcher
            # asked for; if the two disagree the receipt is describing two
            # different rooms and neither can be trusted as the room.
            if str(scene.get("room_id") or "") != str(level.get("observed") or ""):
                fail(world_id,
                     "the scene block and the loaded level name different rooms",
                     scene_room_id=scene.get("room_id"),
                     observed_level=level.get("observed"))
                continue
            ue_observed_room = str(level.get("observed"))
            pixel = receipt.get("native_pixel") or {}
            pixel_artifacts = pixel.get("artifacts") or {}
            present_pixel = [
                name for name, value in pixel_artifacts.items()
                if isinstance(value, str)
                and _resolve_capture_artifact(capture_root, value).is_file()
            ]
            if not present_pixel:
                fail(world_id, "UE capture receipt has no readable pixel artifact",
                     declared=sorted(
                         name for name in pixel_artifacts
                         if isinstance(pixel_artifacts.get(name), str)))
                continue
            playable = []
            for name, entry in (receipt.get("media") or {}).items():
                if not isinstance(entry, Mapping):
                    continue
                value = entry.get("path")
                if not isinstance(value, str) or not value.strip():
                    continue
                if _resolve_capture_artifact(capture_root, value).is_file():
                    playable.append((name, entry))
            if not playable:
                fail(world_id, "UE capture receipt names no readable media")
                continue
            missing_siblings = [
                name for name in UE_NATIVE_SIBLING_READBACKS
                if not (capture_root / name).is_file()
            ]
            if missing_siblings:
                fail(world_id,
                     "the UE producer's own execution readbacks are not beside "
                     "the receipt",
                     missing_readbacks=missing_siblings)
                continue
            ue_clock = receipt.get("clock") or {}
            media_mismatch = [
                name for name, entry in playable
                if (
                    entry.get("frame_count") is not None
                    and ue_clock.get("frame_count") is not None
                    and int(entry["frame_count"]) != int(ue_clock["frame_count"])
                ) or (
                    entry.get("frame_rate_hz") is not None
                    and ue_clock.get("frame_rate_hz") is not None
                    and int(entry["frame_rate_hz"]) != int(ue_clock["frame_rate_hz"])
                )
            ]
            if media_mismatch:
                fail(world_id, "UE media clock differs from the capture clock",
                     media=sorted(media_mismatch))
                continue
            plan_hint = receipt.get("episode_plan") or receipt.get("plan_path")
        else:
            if (
                receipt.get("status") != "pass"
                or receipt.get("research_only") is not True
                or receipt.get("qualification_claim") is not False
            ):
                fail(world_id, "backend capture receipt is not a research-only pass")
                continue
            artifacts = receipt.get("artifacts") or receipt.get("capture_artifacts") or {}
            artifact_paths = [
                capture_root / value
                for value in artifacts.values()
                if isinstance(value, str)
            ]
            if not any(path.is_file() for path in artifact_paths):
                fail(world_id, "backend capture receipt has no readable capture artifacts")
                continue
            plan_hint = receipt.get("episode_plan") or receipt.get("plan_path")
        if receipt.get("episode_counted") is True:
            fail(world_id, "capture receipt is already counted as an Episode")
            continue
        clock_source: Mapping[str, Any] = receipt.get("clock") or {} if is_ue_native else capture
        frame_count = clock_source.get("frame_count", receipt.get("frame_count"))
        frame_rate = clock_source.get("frame_rate_hz", receipt.get("frame_rate_hz"))
        sample_count = clock_source.get("sample_count", receipt.get("sample_count"))
        sample_rate = clock_source.get("sample_rate_hz", receipt.get("sample_rate_hz"))
        required_clock = [frame_count, frame_rate]
        # A visual-only UE capture declares its audio "not_requested" and
        # carries no sample clock. Demanding one there rejects a complete
        # capture for not containing something it never claimed to produce;
        # the moment it does claim audio, the sample clock is required again.
        audio_requested = not (
            is_ue_native
            and str((receipt.get("audio") or {}).get("status") or "")
            == "not_requested"
        )
        if audio_requested:
            required_clock.extend([sample_count, sample_rate])
        if not all(
            isinstance(value, (int, float)) and value > 0
            for value in required_clock
        ):
            fail(world_id, "capture clock is incomplete",
                 frame_count=frame_count, frame_rate_hz=frame_rate,
                 sample_count=sample_count, sample_rate_hz=sample_rate,
                 audio_requested=audio_requested)
            continue
        episode_request = resolve_path(
            (
                str(Path(entry["episode_root"]) / "request.json")
                if isinstance(entry.get("episode_root"), str)
                else None
            ),
            REPOSITORY,
        )
        request_path = (
            episode_request
            or resolve_path(entry.get("request_path"), capture_root.parent)
            or (capture_root.parent / "request.json").resolve()
        )
        request: dict[str, Any] = {}
        if request_path.is_file():
            try:
                request = _read_json(request_path)
            except Exception as error:
                fail(world_id, f"stage request is not JSON: {error}")
                continue
        plan_path = resolve_path(
            plan_hint or entry.get("episode_plan"), capture_root
        )
        plan_request: Mapping[str, Any] = {}
        if plan_path is not None and plan_path.is_file():
            try:
                plan = _read_json(plan_path)
                plan_request = plan.get("request") or {}
                plan_clock = plan.get("clock") or {}
                if (
                    plan_clock.get("frame_count") is not None
                    and int(plan_clock["frame_count"]) != int(frame_count)
                ):
                    fail(world_id, "episode plan frame clock differs from capture")
                    continue
            except Exception as error:
                fail(world_id, f"linked episode plan is unreadable: {error}")
                continue
        stage_world_id = (
            stage_record.get("world_id")
            if isinstance(stage_record, Mapping)
            else None
        )
        if isinstance(stage_record, Mapping):
            output_root = stage_record.get("output_root")
            executed_capture = any(
                str(item.get("work_item_id") or "").endswith(":capture:01")
                and item.get("status") == "pass"
                for item in stage_record.get("executed") or ()
                if isinstance(item, Mapping)
            )
            if (
                stage_record.get("schema") == "avengine_native_group_stage_run_v1"
                and isinstance(output_root, str)
                and str(receipt_path).startswith(str(Path(output_root).resolve()))
                and not executed_capture
            ):
                fail(world_id, "stage record does not bind a passing capture birth")
                continue
        request_world = request.get("world_id")
        if request_world is not None and request_world != world_id:
            fail(world_id, "stage request world_id does not match fresh target",
                 request_world_id=request_world)
            continue
        if request_world is None:
            if stage_world_id != world_id:
                fail(
                    world_id,
                    "capture request lacks world_id and original group stage record "
                    "does not bind this world",
                    stage_world_id=stage_world_id,
                )
                continue
        receipt_key = str(receipt_path)
        canonical_world = canonical_by_receipt.get(receipt_key)
        if canonical_world is not None and canonical_world != world_id:
            fail(
                world_id,
                "normalized capture receipt is already bound to another world_id",
                canonical_world_id=canonical_world,
                receipt_path=receipt_key,
            )
            continue
        canonical_by_receipt[receipt_key] = world_id
        expected_room = entry.get("expected_room_id") or entry.get("room_id")
        actual_room = (
            # For a UE capture the room is the level the engine reported
            # loading, not the one the request named: a request can ask for
            # anything, and the capture is of whatever actually loaded.
            ue_observed_room
            or request.get("room_id")
            or plan_request.get("room_id")
            or (
                stage_record.get("room_id")
                if isinstance(stage_record, Mapping)
                else None
            )
        )
        if expected_room is not None and actual_room != expected_room:
            fail(world_id, "stage room_id does not match expected room",
                 expected_room_id=expected_room, actual_room_id=actual_room)
            continue
        expected_task = entry.get("expected_task_family") or entry.get("task_family")
        actual_task = (
            request.get("task_family")
            or plan_request.get("task_family")
            or (
                stage_record.get("task_family")
                if isinstance(stage_record, Mapping)
                else None
            )
        )
        if expected_task is not None and actual_task != expected_task:
            fail(world_id, "stage task_family does not match expected task",
                 expected_task_family=expected_task, actual_task_family=actual_task)
            continue
        if (
            request.get("frame_count") is not None
            and int(request["frame_count"]) != int(frame_count)
        ) or (
            request.get("sample_rate_hz") is not None
            and sample_rate is not None
            and int(request["sample_rate_hz"]) != int(sample_rate)
        ):
            fail(world_id, "stage request clock differs from capture receipt")
            continue
        validated.add(world_id)
        validation[world_id] = {
            "status": "validated",
            "receipt_path": str(receipt_path),
            "request_path": str(request_path),
            "room_id": request.get("room_id"),
            "task_family": request.get("task_family"),
            "clock": {
                "frame_count": int(frame_count),
                "frame_rate_hz": float(frame_rate),
                # None, not 0: a visual-only capture has no sample clock, and
                # writing zero would read as "it produced no samples".
                "sample_count": None if sample_count is None else int(sample_count),
                "sample_rate_hz": None if sample_rate is None else int(sample_rate),
                "audio_requested": audio_requested,
            },
            "capture_backend": (
                "habitat" if is_habitat else "ue_native" if is_ue_native else "m5_1"
            ),
            "native_habitat_started": (
                True if is_habitat else None
            ),
            "ue_level_loaded": (
                (receipt.get("native_level_readback") or {}).get("observed")
                if is_ue_native else None
            ),
            "ue_backend_role": (
                receipt.get("backend_role") if is_ue_native else None
            ),
            "stage_run_record_path": (
                None if stage_record_path is None else str(stage_record_path)
            ),
            "stage_native_visual_worlds_created": int(
                entry.get("native_visual_worlds_created") or 0
            ),
        }
    return {
        "declared_world_ids": sorted(entries),
        "validated_world_ids": sorted(validated),
        "invalid": invalid,
        "validation": validation,
        "receipt_conflicts": receipt_conflicts,
    }


def _p19_world_origins(
    surveys: Sequence[Mapping[str, Any]],
    *,
    declared_fresh_world_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Classify worlds by explicit evidence, never by export directory provenance."""
    retained_worlds = {
        str(row.get("world_id"))
        for row in (surveys[0].get("members") if surveys else ())
        if isinstance(row.get("world_id"), str) and row.get("world_id").strip()
    }
    declared_fresh = {str(value) for value in declared_fresh_world_ids}
    provenance: dict[str, set[str]] = {}
    unknown = 0
    unknown_world_ids: set[str] = set()
    for index, survey in enumerate(surveys):
        source = str(survey.get("source_kind") or "unknown")
        for row in survey.get("members") or ():
            world = row.get("world_id")
            if not isinstance(world, str) or not world.strip():
                unknown += 1
                continue
            world = str(world)
            if world in retained_worlds:
                origin = "retained"
            elif world in declared_fresh:
                origin = "fresh"
            else:
                origin = "unknown"
                unknown_world_ids.add(world)
            provenance.setdefault(world, set()).add(origin)
    for world_id in sorted(declared_fresh - retained_worlds):
        provenance.setdefault(world_id, set()).add("fresh")
    rows = []
    for world, values in sorted(provenance.items()):
        # A repeated retained/fresh export is one identity. It does not create
        # a fresh world unless the identity was explicitly declared with a
        # readable native capture/receipt and is absent from retained survey.
        if "retained" in values:
            origin = "retained"
        elif "fresh" in values:
            origin = "fresh"
        else:
            origin = "unknown"
        rows.append({
            "world_id": world,
            "origin": origin,
            "provenance": sorted(values),
        })
    return {
        "worlds": rows,
        "retained_world_count": sum(row["origin"] == "retained" for row in rows),
        "fresh_world_count": sum(row["origin"] == "fresh" for row in rows),
        "retained_and_fresh_world_count": sum(
            "retained" in row["provenance"] and "fresh" in row["provenance"]
            for row in rows
        ),
        "unknown_world_count": sum(row["origin"] == "unknown" for row in rows),
        "unknown_world_member_rows": unknown,
        "unknown_world_ids": sorted(unknown_world_ids),
        "identity_status": (
            "unknown"
            if unknown or unknown_world_ids
            else "known"
        ),
    }


def _p19_deficits(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for raw in rows:
        row = deepcopy(dict(raw))
        remaining = [
            int(row[key])
            for key in (
                "remaining_valid_main_questions",
                "remaining_distinct_worlds",
                "remaining_group_count",
            )
            if isinstance(row.get(key), int)
        ]
        row["deficit"] = max(remaining or [1])
        row["quota_key"] = str(row.get("kind") or "coverage_request")
        row["subject"] = (
            row.get("qa_id") or row.get("entity_combination")
            or (
                f"{row.get('core_task_family')}|{row.get('room_family')}"
                if row.get("core_task_family") else None
            )
        )
        output.append(row)
    return output


def _p19_registry(inputs: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = inputs.get("registry") or inputs.get("runtime_source_registry")
    value = _coverage_document(value, owner="P19 registry")
    return value if isinstance(value, Mapping) else None


def _p19_combo(instances: Sequence[Mapping[str, Any]],
               registry: Mapping[str, Any] | None) -> str | None:
    # M04 owns the source-family normalization. Raw entity_class names such as
    # articulated_animal are not coverage combination keys.
    if not isinstance(registry, Mapping):
        return None
    from avengine.qa.batch_delivery import source_family_index
    family_by_asset = source_family_index(registry)
    families = []
    for instance in instances:
        family = family_by_asset.get(str(instance.get("asset_id")))
        explicit = instance.get("source_family")
        if explicit in {"human", "animal", "device"}:
            family = str(explicit)
        if family not in {"human", "animal", "device"}:
            return None
        families.append(family)
    if len(families) < 2:
        return None
    from avengine.dataset.source_capabilities import combination_key
    return combination_key(families[0], families[1])
 
 
def _p19_room_family(
    template: Mapping[str, Any],
    room_catalog: Mapping[str, Any] | None,
) -> str | None:
    for key in ("room_family", "family"):
        if template.get(key):
            return str(template[key])
    if isinstance(room_catalog, Mapping):
        for room in room_catalog.get("rooms") or ():
            if str(room.get("room_id")) != str(template.get("room_id")):
                continue
            for key in ("room_family", "family", "backend_family"):
                if room.get(key):
                    return str(room[key])
    return None


#: The room catalog names a renderer; the registry names the backends an
#: asset was actually built for. They are the same fact under two spellings.
#: How many templates one deficit may be probed against before it is
#: reported as uncarryable. The probe compiles conditions, so this is a real
#: cost; the list is ordered so the templates that register the question come
#: first and the bound is generous enough to reach past them.
P19_TEMPLATE_PROBE_LIMIT = 12

P19_RENDERER_BACKENDS = {
    "ue_spear": "spear_unreal",
    "spear_unreal": "spear_unreal",
    "habitat": "habitat",
}


def _p19_room_renderer(
    room_id: Any, room_catalog: Mapping[str, Any] | None
) -> str | None:
    if not isinstance(room_catalog, Mapping) or room_id is None:
        return None
    for room in room_catalog.get("rooms") or ():
        if isinstance(room, Mapping) and str(room.get("room_id")) == str(room_id):
            value = room.get("renderer") or room.get("backend")
            return None if value is None else str(value)
    return None


def _p19_asset_backends(asset: Mapping[str, Any]) -> set[str]:
    declared = asset.get("runtime_backends")
    if isinstance(declared, Mapping):
        return {str(name) for name in declared}
    if isinstance(declared, Sequence) and not isinstance(declared, (str, bytes)):
        return {str(name) for name in declared}
    value = asset.get("runtime_backends_declared")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {str(name) for name in value}
    return set()


def _p19_asset_profiles(
    registry: Mapping[str, Any] | None
) -> dict[str, dict[str, Any]]:
    """What the registry knows about each asset: its family and its backends."""
    if not isinstance(registry, Mapping):
        return {}
    from avengine.dataset.source_capabilities import source_family

    profiles: dict[str, dict[str, Any]] = {}
    for record in registry.get("assets") or ():
        if not isinstance(record, Mapping) or not record.get("asset_id"):
            continue
        profiles[str(record["asset_id"])] = {
            "source_family": source_family(record),
            "entity_class": record.get("entity_class"),
            "backends": _p19_asset_backends(record),
        }
    return profiles


def _p19_bind_asset_into_template(
    template: Mapping[str, Any],
    *,
    asset_id: str,
    profiles: Mapping[str, Mapping[str, Any]],
    room_catalog: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Put one asset into a template slot its own family and backend allow.

    Substitution is like for like: the slot it takes already holds an asset of
    the same source family, so the template's entity combination, its
    qa_targets and the instance ids those targets name all stay valid. A slot
    of another family would silently change what the episode is.
    """

    wanted = profiles.get(asset_id)
    if wanted is None:
        return None, f"{asset_id} is not in the runtime registry"
    if not template.get("qa_targets"):
        # An episode that registers no question measures nothing about the
        # asset standing in it. Rejecting it here keeps the candidate
        # rotation going instead of stopping on a template that could never
        # have helped.
        return None, (
            f"template {template.get('request_id')!r} registers no qa_targets"
        )
    renderer = _p19_room_renderer(template.get("room_id"), room_catalog)
    backend = P19_RENDERER_BACKENDS.get(str(renderer or ""))
    if backend is None:
        return None, (
            f"room {template.get('room_id')!r} declares no renderer this "
            "planner can map to a runtime backend"
        )
    if wanted["backends"] and backend not in wanted["backends"]:
        return None, (
            f"{asset_id} declares backends {sorted(wanted['backends'])} and "
            f"room {template.get('room_id')!r} runs on {backend}"
        )
    bound = deepcopy(dict(template))
    instances = [dict(item) for item in bound.get("instances") or ()]
    if any(str(item.get("asset_id")) == asset_id for item in instances):
        # Another slot already holds it. Binding it again would put the same
        # asset in both slots, which is a different episode from the one the
        # template describes -- and it teaches nothing new about the asset.
        return None, f"{asset_id} already fills a slot in this template"
    for instance in instances:
        current = profiles.get(str(instance.get("asset_id")))
        if current is None or current["source_family"] != wanted["source_family"]:
            continue
        replaced = str(instance.get("asset_id"))
        instance["asset_id"] = asset_id
        if wanted.get("entity_class"):
            instance["source_class"] = str(wanted["entity_class"])
        bound["instances"] = instances
        return (
            {
                "template": bound,
                "bound_asset_id": asset_id,
                "bound_instance_id": instance.get("instance_id"),
                "replaced_asset_id": replaced,
                "source_family": wanted["source_family"],
                "room_id": template.get("room_id"),
                "runtime_backend": backend,
            },
            None,
        )
    return None, (
        f"no instance slot of family {wanted['source_family']!r} in template "
        f"{template.get('request_id')!r}"
    )


def _p19_core_template_legality(
    template: Mapping[str, Any],
    *,
    registry: Mapping[str, Any] | None,
) -> tuple[bool, str | None]:
    """Reject recipe/source combinations the real core entry point cannot use."""
    family = str(template.get("task_family") or "")
    if family != "cross_event_identity":
        return True, None
    if not isinstance(registry, Mapping):
        return False, "identity core template needs the registered source registry"
    assets = {
        str(row.get("asset_id")): row
        for row in registry.get("assets") or ()
        if isinstance(row, Mapping) and row.get("asset_id")
    }
    members = template.get("members") or ()
    if not members:
        return False, "identity core template has no members"
    for member in members:
        instances = member.get("instances") or ()
        ids = [str(item.get("asset_id")) for item in instances]
        if len(ids) != 2 or len(set(ids)) != 2:
            return False, "cross_event_identity requires two distinct source assets per member"
        for asset_id in ids:
            record = assets.get(asset_id)
            if record is None:
                return False, f"identity asset is absent from registry: {asset_id}"
            if record.get("entity_class") != "articulated_human":
                return False, (
                    "cross_event_identity requires articulated_human assets; "
                    f"{asset_id} is {record.get('entity_class')!r}"
                )
    return True, None


def _p19_target_is_inapplicable_here(probe: Mapping[str, Any]) -> bool:
    """Does the probe say this template *cannot* carry the target?

    Only ``not_applicable`` is a definite no: the template's own declaration
    rules the target out, the way an anchor pinned in view rules out an
    out-of-view question. ``evidence_missing`` and ``not_implemented`` mean
    the probe could not decide, and a template is not disqualified for being
    undecidable -- doing that turns every legal candidate the CPU sampler
    cannot yet measure into a refusal, which is how a working configuration
    gets quietly skipped.
    """

    from avengine.dataset.source_capabilities import STATE_NOT_APPLICABLE

    states = probe.get("candidate_states")
    if not states:
        return False
    return all(
        str(row.get("state")) == STATE_NOT_APPLICABLE
        for row in states
        if isinstance(row, Mapping)
    )


def _p19_planning_probe(request: Mapping[str, Any],
                        target: Mapping[str, Any] | None,
                        *, registry: Mapping[str, Any] | None,
                        inputs: Mapping[str, Any]) -> dict[str, Any]:
    """CPU-only current capability probe; retained reports are evidence only."""
    retry_budget = (request.get("profile") or {}).get("retry_budget_within_profile")
    if registry is None:
        return {
            "status": "not_run",
            "reason": "P19 registry was not configured",
            "retry_budget_within_profile": retry_budget,
        }
    try:
        from avengine.qa.generation_conditions import (
            compile_target_candidates, measure_sampler_capabilities,
        )
        from avengine.rooms import conditioned_sampler
        capabilities = measure_sampler_capabilities(
            registry, sampler=conditioned_sampler, request=request
        )
        result = {
            "status": "pass",
            "capability_basis": capabilities,
            "retry_budget_within_profile": retry_budget,
        }
        if target is not None:
            candidates = compile_target_candidates(
                target, instances=request.get("instances") or (),
                registry=registry, capabilities=capabilities.get("declared"),
                backend=inputs.get("backend"),
            )
            result["candidate_states"] = [
                {"qa_id": item.qa_id, "branch": item.branch,
                 "state": item.state, "reason": item.reason}
                for item in candidates
            ]
            result["status"] = (
                "pass" if any(item.state == "available" for item in candidates)
                else "blocked"
            )
        return result
    except Exception as error:
        return {
            "status": "not_run",
            "reason": f"current CPU sampler probe failed: {type(error).__name__}: {error}",
            "retry_budget_within_profile": retry_budget,
        }


def plan_v1_next_requests(
    *,
    manifest: Mapping[str, Any],
    feedback: Mapping[str, Any],
    coverage_inputs: Mapping[str, Any] | None = None,
    max_requests: int | None = None,
) -> dict[str, Any]:
    """Produce fresh CPU request descriptors; never launches native work."""
    inputs = deepcopy(dict(coverage_inputs or _p19_coverage_inputs(manifest) or {}))
    config = _coverage_document(
        inputs.get("config") or inputs.get("production_config")
        or inputs.get("targets"), owner="P19 config/targets"
    )
    if not isinstance(config, Mapping):
        return {"status": "not_run", "reason": "P19 config/targets missing",
                "requests": [], "added_request_count": 0}
    episodes = [dict(row) for row in config.get("episodes") or ()
                if isinstance(row, Mapping)]
    groups = [dict(row) for row in config.get("core_groups") or ()
              if isinstance(row, Mapping)]
    if not episodes and not groups:
        return {"status": "not_run", "reason": "P19 config has no templates",
                "requests": [], "added_request_count": 0}
    registry = _p19_registry(inputs)
    room_catalog = _coverage_document(
        inputs.get("room_catalog"), owner="P19 room_catalog"
    )
    existing = {
        str(row.get("episode_id"))
        for row in manifest.get("episodes") or ()
        if row.get("episode_id") is not None
    }
    existing.update(
        str(value)
        for value in inputs.get("planned_request_ids") or ()
    )
    existing_p19_ordinary = sum(
        1 for row in manifest.get("episodes") or ()
        if not row.get("group_id") and row.get("p19_planned")
    )
    existing_p19_core = sum(
        1 for row in (manifest.get("production") or {}).get("core_groups") or ()
        if row.get("p19_planned")
    )
    seed = inputs.get("seed", config.get("seed", manifest.get("seed", 0)))
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        seed = 0
    outstanding = list(feedback.get("outstanding_requests") or ())
    if not outstanding:
        return {"status": "complete", "requests": [], "added_request_count": 0}
    ordinary_limit = inputs.get("max_new_ordinary")
    core_limit = inputs.get("max_new_core_groups")
    if isinstance(ordinary_limit, int) and ordinary_limit >= 0:
        ordinary_limit = max(0, ordinary_limit - existing_p19_ordinary)
    else:
        ordinary_limit = None
    if isinstance(core_limit, int) and core_limit >= 0:
        core_limit = max(0, core_limit - existing_p19_core)
    else:
        core_limit = None
    limit = max_requests
    if limit is None:
        limit = inputs.get("max_planned_requests")
    if not isinstance(limit, int) or limit <= 0:
        configured_total = [
            value for value in (ordinary_limit, core_limit)
            if isinstance(value, int)
        ]
        limit = sum(configured_total) if configured_total else len(outstanding)
    counter = 0
    ordinary_count = 0
    core_count = 0
    requests: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    def unique_id(prefix: str) -> str:
        nonlocal counter
        while True:
            counter += 1
            value = f"{prefix}_backfill_{counter:03d}"
            if value not in existing:
                existing.add(value)
                return value

    profiles = _p19_asset_profiles(registry)
    # Types the measurement side has already decided need no new native
    # capture. The planner cannot see this for itself: an asset can carry
    # retained native evidence from a tree no configured episode mentions,
    # and "is it in a configured episode" is only a proxy for that. The
    # qualification owner knows, so it says so and this reads it.
    no_new_native = {
        str(value) for value in inputs.get("source_type_no_new_native") or ()
    }

    def registers_target(template: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
        qa_id = row.get("qa_id")
        if not qa_id:
            return True
        branch = row.get("branch")
        return any(
            str(target.get("qa_id")) == str(qa_id)
            and (branch is None or str(target.get("branch")) == str(branch))
            for target in template.get("qa_targets") or ()
        )

    def episode_candidates(row: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Templates that could carry this deficit, best first.

        Registering the question comes first, because a template that does
        not register it cannot close the deficit at all. Beyond that the
        order is the configuration's own.
        """
        wanted = row.get("entity_combination")
        choices = episodes
        if wanted:
            choices = [
                item for item in choices
                if _p19_combo(item.get("instances") or (), registry)
                == str(wanted)
            ]
        carrying = [item for item in choices if registers_target(item, row)]
        rest = [item for item in choices if item not in carrying]
        return [deepcopy(item) for item in (*carrying, *rest)]

    def episode_template(row: Mapping[str, Any]) -> dict[str, Any] | None:
        candidates = episode_candidates(row)
        return candidates[0] if candidates else None

    def source_type_binding(
        row: Mapping[str, Any]
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Choose a candidate asset and a template that can actually host it.

        Taking episodes[0] because the row named no entity combination was
        the whole defect: a request for a missing device went out carrying
        two animals and never mentioned the device. The candidate order is
        the rotation -- assets with no measurement first, then ones a legal
        refusal already knocked back -- and it is bounded by the candidate
        list the inventory declares.
        """

        candidates = [
            *[str(value) for value in row.get("untested_asset_ids") or ()],
            *[str(value) for value in row.get("refused_asset_ids") or ()],
        ]
        seen: set[str] = set()
        ordered = [
            value for value in candidates
            if not (value in seen or seen.add(value))
        ]
        if not ordered:
            ordered = [str(value) for value in row.get("candidate_asset_ids") or ()]
        if not ordered:
            return None, "the deficit names no candidate asset to bind"
        fine_type = str(row.get("fine_type") or "")
        if fine_type in no_new_native:
            return None, (
                f"the qualification owner records {fine_type!r} as needing no "
                "new native capture; its remaining dimensions are measured "
                "from evidence that already exists. Blocking dimensions: "
                + ", ".join(row.get("blocking_dimensions") or ["unknown"])
            )
        if str(row.get("matrix_status") or "") == "fail":
            # The measurement did not merely fail to happen: it happened and
            # the asset did not pass. Sending it to capture again reproduces
            # the same failure and spends a world doing it.
            return None, (
                f"the measured matrix fails {row.get('fine_type')!r} on "
                + ", ".join(row.get("blocking_dimensions") or ["unknown"])
                + "; this type needs replanning before any capture, not "
                "another request"
            )
        already_placed = sorted({
            asset_id for asset_id in ordered
            for template in episodes
            if any(
                str(item.get("asset_id")) == asset_id
                for item in template.get("instances") or ()
            )
        })
        if already_placed and len(already_placed) == len(ordered):
            # Every candidate is already standing in a configured episode.
            # The type is not short of a request; it is short of the
            # observation that would let the matrix judge it. Another episode
            # with the same asset in it would measure nothing new, and this
            # has to be decided before a template search, because some other
            # template will always be willing to take the asset.
            return None, (
                f"every candidate for {row.get('fine_type')!r} is already "
                f"placed in a configured episode ({', '.join(already_placed)}); "
                "this type needs the missing dimensions measured, not another "
                "request. Blocking dimensions: "
                + ", ".join(row.get("blocking_dimensions") or ["unknown"])
            )
        reasons: list[str] = []
        # A template that registers no question would put the asset in a
        # world and ask it nothing, which measures none of the eight
        # dimensions. Those templates are tried last, not first.
        by_usefulness = sorted(
            episodes, key=lambda item: 0 if item.get("qa_targets") else 1)
        for rotation, asset_id in enumerate(ordered):
            for template in by_usefulness:
                bound, reason = _p19_bind_asset_into_template(
                    template, asset_id=asset_id, profiles=profiles,
                    room_catalog=room_catalog,
                )
                if bound is not None:
                    bound["candidate_rotation_index"] = rotation
                    bound["candidate_asset_ids"] = ordered
                    bound["blocking_dimensions"] = list(
                        row.get("blocking_dimensions") or ())
                    return bound, None
                if reason:
                    reasons.append(f"{asset_id} / {template.get('request_id')}: {reason}")
        return None, (
            "no configured template can host any candidate for "
            f"{row.get('fine_type')!r}: " + "; ".join(sorted(set(reasons))[:4])
        )

    def target_for(
        row: Mapping[str, Any], template: Mapping[str, Any]
    ) -> tuple[dict[str, Any] | None, str | None]:
        qa_id = row.get("qa_id")
        if not qa_id:
            return None, None
        branch = row.get("branch")
        candidates = [
            target for target in template.get("qa_targets") or ()
            if str(target.get("qa_id")) == str(qa_id)
            and (branch is None or str(target.get("branch")) == str(branch))
        ]
        if not candidates:
            # A request that asks for nothing in particular cannot close a
            # deficit for a named QA type. Falling back to the config's
            # default coverage produced an episode that never mentioned the
            # question it was appended to answer, and the deficit survived
            # every round.
            return None, (
                f"template {template.get('request_id')!r} has no registered "
                f"target for {qa_id}:{branch}"
            )
        target = deepcopy(dict(candidates[0]))
        ids = {str(item.get("instance_id")) for item in template.get("instances") or ()}
        target_ids = target.get("target_instance_ids") or target.get("target_instances") or ()
        if not target_ids or any(str(value) not in ids for value in target_ids):
            return None, f"template target for {qa_id}:{branch} names unknown instances"
        target["target_instance_ids"] = [str(value) for value in target_ids]
        target.pop("target_instances", None)
        return target, None

    core_rows = [
        row for row in outstanding
        if str(row.get("kind") or "") in {"core_group_cell", "fresh_world"}
    ]
    other_rows = [
        row for row in outstanding
        if str(row.get("kind") or "") not in {"core_group_cell", "fresh_world"}
    ]
    # Take the remaining deficits a kind at a time rather than in list order.
    # A round is capped at a handful of requests, and the asset deficits sit
    # at the end of the list because they are appended last, so straight list
    # order spent every round on QA rows and the thirty-one missing source
    # types were never once asked for. Round-robin keeps the cap but gives
    # each kind a turn; within a kind the original order is preserved.
    by_kind: dict[str, list[Mapping[str, Any]]] = {}
    for row in other_rows:
        by_kind.setdefault(str(row.get("kind") or ""), []).append(row)
    interleaved: list[Mapping[str, Any]] = []
    while any(by_kind.values()):
        for kind in sorted(by_kind):
            queue = by_kind[kind]
            if queue:
                interleaved.append(queue.pop(0))
    for raw in [*core_rows, *interleaved]:
        if len(requests) >= limit:
            break
        row = dict(raw)
        if row.get("kind") in {"core_group_cell", "fresh_world"}:
            if core_limit is not None and core_count >= core_limit:
                continue
            family = str(row.get("core_task_family") or "")
            wanted_room = str(row.get("room_family") or "")
            candidates = [
                item for item in groups
                if str(item.get("task_family")) == family
                and (
                    not wanted_room
                    or _p19_room_family(item, room_catalog) == wanted_room
                )
            ]
            template = None
            legality_reasons = []
            for candidate in candidates:
                legal, reason = _p19_core_template_legality(
                    candidate, registry=registry
                )
                if legal:
                    template = candidate
                    break
                if reason:
                    legality_reasons.append(reason)
            if template is None:
                skipped.append({
                    **row,
                    "reason": (
                        "no legal core template for target"
                        if not legality_reasons
                        else "; ".join(legality_reasons)
                    ),
                })
                continue
            block = deepcopy(template)
            group_id = unique_id(str(template.get("group_id") or family or "core_group"))
            block["group_id"] = group_id
            old_to_new = {}
            for member in block.get("members") or ():
                old_id = str(member.get("request_id") or "")
                new_id = f"{group_id}_{member.get('member_role') or counter}"
                old_to_new[old_id] = new_id
                member["request_id"] = new_id
            for field in ("shared_audio_member_ids", "shared_visual_member_ids"):
                block[field] = [
                    [old_to_new.get(str(value), str(value)) for value in pair]
                    for pair in block.get(field) or ()
                ]
            requests.append({
                "kind": "core_group", "config_block": block,
                "reason": deepcopy(row),
                "planning": {"status": "config_template"},
            })
            core_count += 1
            continue
        if ordinary_limit is not None and ordinary_count >= ordinary_limit:
            continue
        binding = None
        if str(row.get("kind") or "") == "source_fine_type":
            binding, binding_error = source_type_binding(row)
            if binding is None:
                skipped.append({**row, "reason": binding_error})
                continue
            candidates = [binding["template"]]
        else:
            candidates = episode_candidates(row)
            if not candidates:
                skipped.append({**row, "reason": "no ordinary template for target"})
                continue
        # Registering the question is not the same as being able to carry it.
        # A template whose profile pins the anchor in view cannot host an
        # out-of-view target, and attaching that target to it produced a
        # request the sampler would refuse. The probe already knew; it was
        # only ever recorded, never consulted, so the choice is made on it now.
        template = None
        target = None
        planning = None
        rejected: list[dict[str, Any]] = []
        for candidate in candidates[:P19_TEMPLATE_PROBE_LIMIT]:
            candidate_target, target_error = target_for(row, candidate)
            if target_error is not None:
                rejected.append({"request_id": candidate.get("request_id"),
                                 "reason": target_error})
                continue
            probe = _p19_planning_probe(
                candidate, candidate_target, registry=registry, inputs=inputs
            )
            if _p19_target_is_inapplicable_here(probe):
                rejected.append({
                    "request_id": candidate.get("request_id"),
                    "reason": "the sampler cannot satisfy this target in this "
                              "template's profile",
                    "candidate_states": probe.get("candidate_states"),
                })
                continue
            template = candidate
            target = candidate_target
            planning = probe
            break
        if template is None:
            skipped.append({
                **row,
                "reason": (
                    rejected[0]["reason"] if len(rejected) == 1
                    else f"no configured template can carry this target; "
                         f"{len(rejected)} were tried"
                ),
                "templates_rejected": rejected[:6],
            })
            continue
        request = deepcopy(template)
        request["request_id"] = unique_id(
            str(template.get("request_id") or "v1_episode")
        )
        request["seed"] = seed + counter
        if binding is not None:
            # An asset deficit keeps the template's own registered targets:
            # the point is to put this asset in front of real questions, and
            # a request carrying no target would produce nothing to measure
            # the eight dimensions against.
            if not request.get("qa_targets"):
                skipped.append({
                    **row,
                    "reason": (
                        f"template {template.get('request_id')!r} can host "
                        f"{binding['bound_asset_id']} but registers no "
                        "qa_targets, so the bound asset would be asked nothing"
                    ),
                })
                continue
            request.pop("qa_sampling", None)
        else:
            # Omitting these means the normal config defaults cover every QA type.
            request.pop("qa_ids", None)
            request.pop("qa_targets", None)
            request.pop("qa_sampling", None)
        planning = dict(planning or {})
        planning["templates_rejected_before_this_one"] = rejected[:6]
        cost_key = (
            f"{row.get('qa_id')}:{row.get('branch')}"
            if row.get("qa_id") else str(row.get("kind"))
        )
        requests.append({
            "kind": "ordinary_episode", "request": request,
            "reason": deepcopy(row), "target": target,
            "source_type_binding": (
                None if binding is None
                else {key: value for key, value in binding.items()
                      if key != "template"}
            ),
            "planning": planning,
            "planning_cost_basis": deepcopy(
                (inputs.get("planning_costs") or {}).get(cost_key)
            ),
            "config_seed": request["seed"],
        })
        ordinary_count += 1
    return {
        "status": "planned" if requests else "blocked",
        "requests": requests, "skipped": skipped,
        "added_request_count": len(requests),
        "planning_basis": {
            "registry_configured": registry is not None,
            "current_sampler_probe": "attempted_for_ordinary_requests",
            "fixed_seed_semantics": "config seed plus bounded planner ordinal",
            "no_native_or_rlr": True,
        },
    }


def _p19_coverage_feedback(
    *,
    manifest: Mapping[str, Any],
    delivered_groups: Sequence[Mapping[str, Any]],
    delivered_episodes: Sequence[Mapping[str, Any]],
    imported_bundle_paths: Sequence[str | Path],
    delivery_roots: Sequence[str | Path],
) -> dict[str, Any]:
    inputs = _p19_coverage_inputs(manifest) or {}
    targets = _coverage_document(
        inputs.get("targets") or inputs.get("config")
        or inputs.get("production_config"), owner="P19 targets"
    )
    cached = _coverage_document(
        inputs.get("cached_survey") or inputs.get("retained_survey"),
        owner="P19 cached survey"
    )
    if not isinstance(targets, Mapping) or not isinstance(cached, Mapping):
        raise ProductionRunError(
            "P19 coverage_inputs requires targets/config and cached_survey"
        )
    roots: list[Path] = []
    for value in (*inputs.get("delivery_roots", ()), *delivery_roots):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = REPOSITORY / path
        path = path.resolve()
        if path not in roots:
            roots.append(path)
    from avengine.qa.batch_coverage import (
        build_v1_coverage_feedback,
        outstanding_production_requests,
        source_type_accounting,
        source_type_shortfall_requests,
    )
    from avengine.qa.batch_delivery import (
        achieved_coverage_table, merge_achieved_surveys, survey_delivery_export,
    )
    surveys = [cached]
    fresh_surveys = []
    for root in roots:
        if not root.is_dir():
            raise ProductionRunError(f"P19 delivery root is missing: {root}")
        survey = survey_delivery_export(root)
        surveys.append(survey)
        fresh_surveys.append(survey)
    merged = merge_achieved_surveys(*surveys)
    achieved = achieved_coverage_table(merged)
    feedback = build_v1_coverage_feedback(
        targets=targets,
        achieved=achieved,
        planning=_coverage_document(
            inputs.get("planning") or inputs.get("planning_report"),
            owner="P19 planning",
        ),
        applicability=_coverage_document(
            inputs.get("applicability") or inputs.get("source_family_applicability"),
            owner="P19 applicability",
        ),
        sound_inputs=_coverage_document(
            inputs.get("sound_inputs"), owner="P19 sound_inputs"
        ),
    )
    outstanding = outstanding_production_requests(feedback)
    auto_capture_evidence = [
        evidence
        for source in [*delivered_groups, *delivered_episodes]
        for evidence in source.get("capture_stage_evidence") or ()
        if isinstance(evidence, Mapping)
        and evidence.get("capture_receipt_path")
        and evidence.get("world_id")
    ]
    fresh_declarations = _p19_fresh_world_declarations(
        inputs, auto_evidence=auto_capture_evidence
    )
    origins = _p19_world_origins(
        [cached, *fresh_surveys],
        declared_fresh_world_ids=fresh_declarations["validated_world_ids"],
    )
    fresh_world_ids = {
        str(row["world_id"])
        for row in origins["worlds"]
        if row.get("origin") == "fresh"
    }
    fresh_ordinary_rows = [
        row for survey in fresh_surveys
        for row in survey.get("members") or ()
        if not row.get("group_id")
        and str(row.get("world_id")) in fresh_world_ids
    ]
    fresh_ordinary_worlds = {
        str(row.get("world_id"))
        for row in fresh_ordinary_rows
        if isinstance(row.get("world_id"), str) and row.get("world_id").strip()
    }
    fresh_ordinary_worlds.update(
        str(evidence["world_id"])
        for evidence in auto_capture_evidence
        if evidence.get("group_id") in (None, "")
        and str(evidence.get("world_id")) in fresh_world_ids
    )
    ordinary_target_accounting = {
        "status": "not_configured",
        "reason": (
            "fresh ordinary-world/type targets were not declared; the provider "
            "does not infer the fresh4/31 denominator from QA/core/combo rows"
        ),
    }
    ordinary_target = inputs.get("fresh_ordinary_world_target")
    if isinstance(ordinary_target, int) and ordinary_target >= 0:
        remaining = max(0, ordinary_target - len(fresh_ordinary_worlds))
        ordinary_target_accounting = {
            "status": "configured",
            "target": ordinary_target,
            "achieved": len(fresh_ordinary_worlds),
            "remaining": remaining,
            "world_identity_status": (
                "unknown"
                if any(not row.get("world_id") for row in fresh_ordinary_rows)
                else "known"
            ),
        }
        if remaining:
            outstanding.append({
                "kind": "ordinary_world_count",
                "state": "short_of_target",
                "reason": "fresh ordinary-world target is short",
                "remaining_distinct_worlds": remaining,
                "required_distinct_worlds": ordinary_target,
            })
    # The fine source types run through the same provider as everything
    # else: one accounting, one outstanding list, one deficit list, one
    # planner. A separate scheduler for asset coverage would be a second
    # opinion about what to produce next.
    source_type_accounting_result: dict[str, Any] = {
        "status": "not_configured",
        "reason": (
            "no source type inventory was declared; the fine-type denominator "
            "comes from the declaration and is not inferred from deliveries"
        ),
    }
    inventory_document = _coverage_document(
        inputs.get("source_type_inventory"), owner="P19 source_type_inventory"
    )
    if inventory_document is not None:
        matrix = _p19_qualification_matrix(inputs, inventory=inventory_document)
        source_type_accounting_result = source_type_accounting(
            inventory=inventory_document,
            qualification_matrix=matrix,
            candidate_asset_ids_by_class=(
                manifest.get("candidate_asset_ids_by_class")
                or (manifest.get("production") or {}).get(
                    "candidate_asset_ids_by_class")
            ),
        )
        outstanding.extend(
            source_type_shortfall_requests(source_type_accounting_result))
    for target in inputs.get("fresh_world_targets") or ():
        if not isinstance(target, Mapping):
            continue
        world_id = target.get("world_id")
        if not isinstance(world_id, str) or not world_id.strip():
            continue
        if world_id in fresh_world_ids:
            continue
        outstanding.append({
            "kind": "fresh_world",
            "world_id": world_id,
            "core_task_family": target.get("task_family"),
            "room_family": target.get("room_family"),
            "state": "short_of_target",
            "reason": (
                "explicit fresh world target has no validated fresh delivery"
            ),
        })
    result = deepcopy(feedback)
    result.update({
        "provider": "production_runner.p19_default",
        "achieved": achieved,
        "outstanding_requests": outstanding,
        "deficits": _p19_deficits(outstanding),
        "ordinary_target_accounting": ordinary_target_accounting,
        "source_type_accounting": source_type_accounting_result,
        "fresh_world_count": origins["fresh_world_count"],
        "retained_world_count": origins["retained_world_count"],
        "unknown_world_count": origins["unknown_world_count"],
        "unknown_world_member_rows": origins["unknown_world_member_rows"],
        "unknown_world_ids": origins["unknown_world_ids"],
        "fresh_world_declarations": fresh_declarations,
        "world_origins": origins["worlds"],
        "world_identity_status": origins["identity_status"],
        "p19_inputs": {
            "targets": inputs.get("targets") or inputs.get("config")
            or inputs.get("production_config"),
            "cached_survey": inputs.get("cached_survey")
            or inputs.get("retained_survey"),
            "delivery_roots": [str(path) for path in roots],
        },
    })
    result["next_requests"] = plan_v1_next_requests(
        manifest=manifest, feedback=result, coverage_inputs=inputs
    )
    return result


def coverage_feedback(
    *,
    manifest: Mapping[str, Any],
    delivered_groups: Sequence[Mapping[str, Any]] = (),
    delivered_episodes: Sequence[Mapping[str, Any]] = (),
    imported_bundle_paths: Sequence[str | Path] = (),
    delivery_roots: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Use P19 APIs when configured, otherwise preserve legacy accounting."""
    if _COVERAGE_PROVIDER is not None:
        kwargs: dict[str, Any] = {
            "manifest": manifest,
            "delivered_groups": delivered_groups,
            "imported_bundle_paths": imported_bundle_paths,
        }
        if delivered_episodes:
            kwargs["delivered_episodes"] = delivered_episodes
        if delivery_roots:
            kwargs["delivery_roots"] = delivery_roots
        return dict(_COVERAGE_PROVIDER(**kwargs))
    if _p19_coverage_inputs(manifest) is not None:
        return _p19_coverage_feedback(
            manifest=manifest, delivered_groups=delivered_groups,
            delivered_episodes=delivered_episodes,
            imported_bundle_paths=imported_bundle_paths,
            delivery_roots=delivery_roots,
        )
    return _legacy_coverage_feedback(
        manifest=manifest, delivered_groups=delivered_groups,
        imported_bundle_paths=imported_bundle_paths,
    )




def _append_p19_requests_to_runner(
    runner: ProductionRunner,
    plan: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize bounded P19 descriptors into this run's effective manifest."""
    config = _coverage_document(
        inputs.get("config") or inputs.get("production_config")
        or inputs.get("targets"), owner="P19 config/targets"
    )
    if not isinstance(config, Mapping):
        return {
            "status": "not_run",
            "reason": "P19 config/targets missing for manifest append",
            "added_request_count": 0,
            "added_request_ids": [],
        }
    requests = list(plan.get("requests") or ())
    if not requests:
        return {
            "status": "complete",
            "reason": "planner produced no new request descriptors",
            "added_request_count": 0,
            "added_request_ids": [],
            "skipped": list(plan.get("skipped") or ()),
        }
    from avengine.qa.batch_manifest import production_config_slots
    base = {
        "schema": config.get("schema"),
        "batch_id": config.get("batch_id", runner.manifest.get("batch_id")),
        "seed": config.get("seed", runner.manifest.get("seed", 0)),
        "defaults": deepcopy(dict(config.get("defaults") or {})),
        "coverage_quota": deepcopy(dict(config.get("coverage_quota") or {})),
    }
    episodes: list[dict[str, Any]] = []
    new_groups: list[dict[str, Any]] = []
    added_episode_ids: list[str] = []
    added_group_ids: list[str] = []
    skipped = list(plan.get("skipped") or ())
    ordinary_limit = inputs.get("max_new_ordinary")
    core_limit = inputs.get("max_new_core_groups")
    ordinary_count = 0
    core_count = 0
    for descriptor in requests:
        kind = str(descriptor.get("kind") or "")
        if kind == "ordinary_episode":
            if isinstance(ordinary_limit, int) and ordinary_limit >= 0 and ordinary_count >= ordinary_limit:
                skipped.append({
                    **dict(descriptor.get("reason") or {}),
                    "reason": "max_new_ordinary reached",
                })
                continue
            mini = {
                **base,
                "episodes": [deepcopy(dict(descriptor["request"]))],
                "core_groups": [],
            }
            try:
                slots, _summary = production_config_slots(mini)
            except Exception as error:
                skipped.append({
                    **dict(descriptor.get("reason") or {}),
                    "reason": f"ordinary request rejected by production config: {error}",
                })
                continue
            if len(slots) != 1:
                skipped.append({
                    **dict(descriptor.get("reason") or {}),
                    "reason": "ordinary template did not produce exactly one slot",
                })
                continue
            slot = slots[0]
            row = {
                "episode_id": str(slot["episode_id"]),
                "room_id": str(slot["room_id"]),
                "source_classes": deepcopy(slot.get("source_classes") or []),
                "silent_count": slot.get("silent_count", 0),
                "seed": slot.get("seed"),
                "profile": deepcopy(slot.get("profile") or {}),
                "request": deepcopy(slot["request_overrides"]),
                "p19_planned": {
                    "reason": deepcopy(descriptor.get("reason") or {}),
                    "planning": deepcopy(descriptor.get("planning") or {}),
                    "planning_cost_basis": deepcopy(
                        descriptor.get("planning_cost_basis")
                    ),
                },
            }
            episodes.append(row)
            added_episode_ids.append(row["episode_id"])
            ordinary_count += 1
            continue
        if kind == "core_group":
            if isinstance(core_limit, int) and core_limit >= 0 and core_count >= core_limit:
                skipped.append({
                    **dict(descriptor.get("reason") or {}),
                    "reason": "max_new_core_groups reached",
                })
                continue
            block = deepcopy(dict(descriptor.get("config_block") or {}))
            mini = {**base, "episodes": [], "core_groups": [block]}
            try:
                slots, summary = production_config_slots(mini)
            except Exception as error:
                skipped.append({
                    **dict(descriptor.get("reason") or {}),
                    "reason": f"core request rejected by production config: {error}",
                })
                continue
            if not slots or not summary.get("core_groups"):
                skipped.append({
                    **dict(descriptor.get("reason") or {}),
                    "reason": "core template produced no group slots",
                })
                continue
            group_meta = {
                key: deepcopy(summary["core_groups"][0].get(key))
                for key in (
                    "group_id", "task_family", "room_id", "member_request_ids",
                    "shared_audio_member_ids", "shared_visual_member_ids",
                )
                if key in summary["core_groups"][0]
            }
            group_meta["p19_planned"] = {
                "reason": deepcopy(descriptor.get("reason") or {}),
                "planning": deepcopy(descriptor.get("planning") or {}),
            }
            new_groups.append(group_meta)
            added_group_ids.append(str(group_meta["group_id"]))
            for slot in slots:
                row = {
                    "episode_id": str(slot["episode_id"]),
                    "room_id": str(slot["room_id"]),
                    "group_id": str(slot["group_id"]),
                    "task_family": str(slot["task_family"]),
                    "member_index": slot.get("member_index"),
                    "member_role": slot.get("member_role"),
                    "source_classes": deepcopy(slot.get("source_classes") or []),
                    "silent_count": slot.get("silent_count", 0),
                    "seed": slot.get("seed"),
                    "profile": deepcopy(slot.get("profile") or {}),
                    "request": deepcopy(slot["request_overrides"]),
                    "p19_planned": {
                        "reason": deepcopy(descriptor.get("reason") or {}),
                        "planning": deepcopy(descriptor.get("planning") or {}),
                    },
                }
                episodes.append(row)
            core_count += 1
            continue
        skipped.append({
            **dict(descriptor.get("reason") or {}),
            "reason": f"unknown P19 descriptor kind {kind!r}",
        })
    if not episodes and not new_groups:
        return {
            "status": "blocked",
            "reason": "no P19 descriptor survived config validation or limits",
            "added_request_count": 0,
            "added_request_ids": [],
            "skipped": skipped,
        }
    existing_ids = {
        str(row.get("episode_id")) for row in runner.manifest.get("episodes") or ()
    }
    duplicate = sorted(set(added_episode_ids) & existing_ids)
    if duplicate:
        raise ProductionRunError(
            f"P19 planner attempted duplicate manifest request IDs: {duplicate}"
        )
    runner.manifest.setdefault("episodes", []).extend(episodes)
    production = runner.manifest.setdefault("production", {})
    production.setdefault("core_groups", []).extend(new_groups)
    p19_state = runner.manifest.get("coverage_inputs")
    if isinstance(p19_state, dict):
        p19_state["rounds_completed"] = int(p19_state.get("rounds_completed") or 0) + 1
    for row in episodes:
        runner.rows_by_episode_id[str(row["episode_id"])] = deepcopy(row)
    if added_group_ids:
        runner.scopes.extend(runner._build_scopes(added_group_ids, None))
    if added_episode_ids:
        runner.scopes.extend(runner._build_scopes(None, added_episode_ids))
    runner._append_event({
        "event": "p19_requests_appended",
        "ordinary_count": ordinary_count,
        "core_group_count": core_count,
        "request_ids": added_episode_ids,
        "group_ids": added_group_ids,
        "round_limit": inputs.get("round_limit"),
        "native_accounting": deepcopy(
            runner._refresh_native_accounting_totals()
        ),
    })
    runner._persist()
    return {
        "status": "appended",
        "added_request_count": len(added_episode_ids) + sum(
            4 for _ in added_group_ids
        ),
        "added_ordinary_count": ordinary_count,
        "added_core_group_count": core_count,
        "added_request_ids": added_episode_ids,
        "added_group_ids": added_group_ids,
        "skipped": skipped,
        "manifest_snapshot": str(runner.state_path),
    }



def core_bundle_for(bundles: Sequence[Path], *, output: Path) -> Path:
    """The core bundle a delivery joins against: one bundle, or a merged one.

    `export_binding_delivery` joins core members to catalog records by
    (group_id, member_id), so several groups merge safely as long as no group
    id repeats. A run that delivered one group hands its own bundle straight
    through, unchanged.
    """
    if len(bundles) == 1:
        return Path(bundles[0])
    groups: list[dict[str, Any]] = []
    validations: set[str] = set()
    seen: set[str] = set()
    for path in bundles:
        bundle = _read_json(Path(path))
        validations.add(str(bundle.get("validation")))
        for group in bundle.get("groups", []):
            group_id = str(group.get("group_id"))
            if group_id in seen:
                raise ProductionRunError(
                    f"two delivered bundles both carry group {group_id!r}; a core "
                    "delivery joins on (group_id, member_id) and cannot hold it twice",
                    bundles=[str(item) for item in bundles])
            seen.add(group_id)
            groups.append(deepcopy(dict(group)))
    merged = {
        "schema": "avengine_binding_groups_v1",
        "status": "research_candidate",
        "group_count": len(groups),
        "requested_group_count": len(groups),
        "world_count": len({str(group.get("world_id")) for group in groups}),
        "sample_count": sum(len(group.get("members") or ()) for group in groups),
        "groups": groups,
        # The weakest input decides: one structure-only bundle makes the merge
        # structure-only, whatever the others checked.
        "validation": ("media_checked" if validations == {"media_checked"}
                       else "structure_only"),
        "model_evaluation": "not_run",
        "human_answerability": "not_run",
        "merged_from": [str(item) for item in bundles],
    }
    path = Path(output) / "merged_core_bundle.json"
    _json_write_new(path, merged)
    return path


def _attach_exported_audio_layouts(
    destination: Path,
    catalog: Mapping[str, Any],
    delivered_groups: Sequence[Mapping[str, Any]],
    delivered_episodes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Attach declared non-primary layouts from each member's own receipt."""
    from avengine.qa.binding_delivery import attach_audio_layout

    evidence_by_facts: dict[str, Mapping[str, Any]] = {}
    evidence_by_core_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    evidence_by_episode: dict[str, Mapping[str, Any]] = {}

    def add_unique(table, key, evidence, label):
        if key is None:
            return
        previous = table.get(key)
        if previous is not None and previous != evidence:
            raise ProductionRunError(
                f"ambiguous audio evidence for {label}={key!r}; "
                "facts/group/member identity must be unique"
            )
        table[key] = evidence

    for source in [*delivered_groups, *delivered_episodes]:
        for raw in source.get("audio_stage_evidence") or ():
            if not isinstance(raw, Mapping):
                continue
            evidence = deepcopy(dict(raw))
            facts_value = evidence.get("facts_path")
            if isinstance(facts_value, str) and facts_value.strip():
                add_unique(
                    evidence_by_facts,
                    str(Path(facts_value).expanduser().resolve()),
                    evidence,
                    "facts_path",
                )
            group_id = evidence.get("group_id")
            member_id = evidence.get("member_id")
            if isinstance(group_id, str) and group_id.strip() and isinstance(
                member_id, str
            ) and member_id.strip():
                add_unique(
                    evidence_by_core_key,
                    (group_id, member_id),
                    evidence,
                    "(group_id, member_id)",
                )
            elif (
                group_id in (None, "")
                and isinstance(evidence.get("episode_id"), str)
                and evidence.get("episode_id").strip()
            ):
                add_unique(
                    evidence_by_episode,
                    evidence["episode_id"],
                    evidence,
                    "episode_id",
                )

    attachments: list[dict[str, Any]] = []
    for record in catalog.get("records") or ():
        if not isinstance(record, Mapping):
            continue
        facts_source = record.get("facts_path")
        if not isinstance(facts_source, str):
            continue
        facts_path = Path(facts_source).expanduser().resolve()
        if not facts_path.is_file():
            continue
        # Facts identity is authoritative. A core member name such as v0_a0
        # is intentionally not globally unique across groups.
        evidence = evidence_by_facts.get(str(facts_path))
        if evidence is None:
            group_id = record.get("group_id")
            member_id = record.get("member_id")
            if isinstance(group_id, str) and group_id.strip() and isinstance(
                member_id, str
            ) and member_id.strip():
                evidence = evidence_by_core_key.get((group_id, member_id))
            elif (
                group_id in (None, "")
                and isinstance(record.get("episode_id"), str)
                and record.get("episode_id").strip()
            ):
                evidence = evidence_by_episode.get(record["episode_id"])
        facts = _read_json(facts_path)
        source_paths = facts.get("source_paths") if isinstance(facts, Mapping) else {}
        report_value = (
            evidence.get("audio_report_path")
            if isinstance(evidence, Mapping)
            else None
        )
        if not isinstance(report_value, str) or not report_value.strip():
            report_value = (
                source_paths.get("research_report")
                if isinstance(source_paths, Mapping)
                else None
            )
        report_path = (
            Path(report_value).expanduser().resolve()
            if isinstance(report_value, str) and report_value.strip()
            else None
        )
        declared = (
            evidence.get("declared_audio_delivery")
            if isinstance(evidence, Mapping)
            else None
        )
        requested = (
            list(declared.get("attached_view_layouts") or ())
            if isinstance(declared, Mapping)
            else []
        )
        report = (
            _read_json(report_path)
            if report_path is not None and report_path.is_file()
            else {}
        )
        layout_delivery = (
            (report.get("audio") or {}).get("layout_delivery")
            if isinstance(report, Mapping)
            else None
        )
        layout_delivery = (
            layout_delivery if isinstance(layout_delivery, Mapping) else {}
        )
        if not requested:
            requested = [
                str(layout)
                for layout in layout_delivery
                if str(layout) != "binaural"
            ]
        if requested and report_path is None:
            raise ProductionRunError(
                f"sample {record.get('sample_id')} declares attached audio "
                "layouts but has no real research_report path"
            )
        for layout in requested:
            if layout == "binaural":
                continue
            entry = layout_delivery.get(layout)
            mixture = (entry.get("mixture") if isinstance(entry, Mapping) else {})
            mixture_path = mixture.get("path") if isinstance(mixture, Mapping) else None
            if not isinstance(mixture_path, str) or not Path(mixture_path).is_file():
                raise ProductionRunError(
                    f"sample {record.get('sample_id')} declares {layout!r} "
                    "but its real receipt has no readable mixture"
                )
            attachments.append(
                attach_audio_layout(
                    destination,
                    sample_id=str(record["sample_id"]),
                    layout=str(layout),
                    receipt=report_path,
                    mixture=mixture_path,
                )
            )
    return attachments



def _export_with_imported_catalogs(
    *,
    delivered_groups: Sequence[Mapping[str, Any]],
    delivered_episodes: Sequence[Mapping[str, Any]],
    output: str | Path,
    catalog_root: Path,
    core_bundle: str | Path | None,
    imported_catalog_index_paths: Sequence[str | Path],
    imported_core_bundle_paths: Sequence[str | Path],
    imported_delivery_roots: Sequence[str | Path],
    qa_sampling: Mapping[str, Any],
    items_per_type: int,
    seed: str,
    build_index: bool,
) -> dict[str, Any]:
    """Build one self-contained catalog from explicit imported/current sources."""
    from avengine.qa.binding_catalog import (
        derive_binding_catalog,
        derive_episode_catalog,
        merge_binding_catalogs,
    )
    from avengine.qa.binding_delivery import build_dataset_index, export_binding_delivery

    destination = Path(output).expanduser().resolve()
    catalog_root.mkdir(parents=True, exist_ok=True)
    catalog_paths: list[Path] = []
    core_bundles: list[Path] = []

    def add_path(value: str | Path) -> Path:
        path = Path(value).expanduser().resolve()
        if path not in catalog_paths:
            catalog_paths.append(path)
        return path

    for value in imported_catalog_index_paths:
        add_path(value)
    for raw_root in imported_delivery_roots:
        root = Path(raw_root).expanduser().resolve()
        catalog_path = root / "catalog" / "catalog_index.json"
        if catalog_path.is_file():
            add_path(catalog_path)
        for candidate in (
            root / "core" / "merged_core_bundle.json",
            root / "catalog" / "merged_core_bundle.json",
        ):
            if candidate.is_file():
                core_bundles.append(candidate)
                break
    for index, value in enumerate(imported_core_bundle_paths):
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise ProductionRunError(f"imported core bundle is missing: {path}")
        core_bundles.append(path)
        source_root = catalog_root / f"imported_core_source_{index:02d}"
        derive_binding_catalog(
            [str(path)],
            output=source_root,
            qa_sampling=qa_sampling,
            items_per_type=int(items_per_type),
            seed=seed,
        )
        add_path(source_root / "catalog_index.json")
    core_bundles = list(dict.fromkeys(core_bundles))

    current_bundles = _bundle_paths(delivered_groups)
    if current_bundles:
        current_root = catalog_root / "current_core_source"
        catalog = derive_binding_catalog(
            [str(path) for path in current_bundles],
            output=current_root,
            qa_sampling=qa_sampling,
            items_per_type=int(items_per_type),
            seed=seed,
        )
        add_path(current_root / "catalog_index.json")
        core_bundles.extend(current_bundles)
    if delivered_episodes:
        specs = []
        for row in delivered_episodes:
            spec = deepcopy(dict(row))
            media = dict(spec.get("media") or {})
            media.setdefault("video_path", spec.get("video_path"))
            media.setdefault("audio_path", spec.get("audio_path"))
            spec["media"] = media
            specs.append(spec)
        episode_root = catalog_root / "current_episode_source"
        derive_episode_catalog(
            specs,
            output=episode_root,
            qa_sampling=qa_sampling,
            items_per_type=int(items_per_type),
            seed=seed,
        )
        add_path(episode_root / "catalog_index.json")
    core_bundles = list(dict.fromkeys(core_bundles))
    if not catalog_paths:
        return {
            "schema": SCHEMA,
            "status": "no_imported_output",
            "reason": "no imported/current catalog source was available",
            "delivered_group_count": len(delivered_groups),
            "delivered_episode_count": len(delivered_episodes),
        }
    if len(catalog_paths) == 1:
        catalog_index = catalog_paths[0]
        catalog = _read_json(catalog_index)
    else:
        merged_root = catalog_root / "merged"
        catalog = merge_binding_catalogs(
            [str(path) for path in catalog_paths],
            output=merged_root,
            seed=seed,
        )
        catalog_index = merged_root / "catalog_index.json"
    core = None
    if core_bundle is not None:
        core = Path(core_bundle).expanduser().resolve()
    elif core_bundles:
        if len(core_bundles) == 1:
            core = core_bundles[0]
        else:
            join_root = catalog_root / "core_join"
            core = core_bundle_for(core_bundles, output=join_root)
    delivery = export_binding_delivery(
        None if core is None else str(core),
        catalog_index,
        destination,
    )
    result = {
        "schema": SCHEMA,
        "status": "exported",
        "delivery_kind": "imported_mixed_merged_catalog",
        "catalog_root": str(catalog_index.parent),
        "catalog_index": str(catalog_index),
        "core_bundle": None if core is None else str(core),
        "source_catalogs": [str(path) for path in catalog_paths],
        "imported_delivery_roots": [str(Path(path).expanduser().resolve())
                                    for path in imported_delivery_roots],
        "catalog_summary": {
            key: catalog[key]
            for key in (
                "av_sample_count", "group_count", "world_count",
                "catalog_question_count", "core_question_count",
                "generated_by_qa",
            )
            if key in catalog
        },
        "delivered_episode_count": len(delivered_episodes),
        "delivered_group_count": len(delivered_groups),
        "delivery_root": str(destination),
        "delivery": deepcopy(delivery),
    }
    result["audio_attachments"] = _attach_exported_audio_layouts(
        destination, catalog, delivered_groups, delivered_episodes
    )
    if build_index:
        result["dataset_index"] = deepcopy(build_dataset_index(destination))
    return result


def export_run_delivery(
    *,
    delivered_groups: Sequence[Mapping[str, Any]] = (),
    delivered_episodes: Sequence[Mapping[str, Any]] = (),
    output: str | Path,
    catalog_output: str | Path | None = None,
    core_bundle: str | Path | None = None,
    qa_sampling: Mapping[str, Any] | None = None,
    items_per_type: int = 1,
    seed: str = "v1-production-run",
    build_index: bool = True,
    imported_catalog_index_paths: Sequence[str | Path] = (),
    imported_core_bundle_paths: Sequence[str | Path] = (),
    imported_delivery_roots: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Export core, ordinary, or mixed output through the existing catalog path.

    Mixed output derives separate source catalogs, then uses M09's
    merge_binding_catalogs before the existing strict core/ordinary join.
    """
    from avengine.qa.binding_catalog import (
        derive_binding_catalog,
        derive_episode_catalog,
        merge_binding_catalogs,
    )
    from avengine.qa.binding_delivery import build_dataset_index, export_binding_delivery

    groups = list(delivered_groups or ())
    episodes = list(delivered_episodes or ())
    if not groups and not episodes:
        return {
            "schema": SCHEMA,
            "status": "no_delivered_output",
            "reason": "no ordinary Episode reached delivery and no core group reached assembly",
            "delivered_group_count": 0,
            "delivered_episode_count": 0,
        }

    destination = Path(output).expanduser().resolve()
    catalog_root = (
        Path(catalog_output).expanduser().resolve()
        if catalog_output is not None
        else destination.parent / "catalog"
    )
    qa = dict(qa_sampling or DEFAULT_QA_SAMPLING)

    if imported_catalog_index_paths or imported_core_bundle_paths or imported_delivery_roots:
        return _export_with_imported_catalogs(
            delivered_groups=groups,
            delivered_episodes=episodes,
            output=output,
            catalog_root=catalog_root,
            core_bundle=core_bundle,
            imported_catalog_index_paths=imported_catalog_index_paths,
            imported_core_bundle_paths=imported_core_bundle_paths,
            imported_delivery_roots=imported_delivery_roots,
            qa_sampling=qa,
            items_per_type=int(items_per_type),
            seed=seed,
            build_index=build_index,
        )

    if groups and episodes:
        bundles = _bundle_paths(groups)
        if not bundles:
            return {
                "schema": SCHEMA,
                "status": "no_delivered_group",
                "reason": "mixed export has no media-checked core bundle",
                "delivered_group_count": len(groups),
                "delivered_episode_count": len(episodes),
            }
        catalog_root.mkdir(parents=True, exist_ok=True)
        core_source = catalog_root / "core_source"
        episode_source = catalog_root / "episode_source"
        merged_root = catalog_root / "merged"
        core_catalog = derive_binding_catalog(
            [str(path) for path in bundles],
            output=core_source,
            qa_sampling=qa,
            items_per_type=int(items_per_type),
            seed=seed,
        )
        specs = []
        for row in episodes:
            spec = deepcopy(dict(row))
            media = dict(spec.get("media") or {})
            media.setdefault("video_path", spec.get("video_path"))
            media.setdefault("audio_path", spec.get("audio_path"))
            spec["media"] = media
            specs.append(spec)
        episode_catalog = derive_episode_catalog(
            specs,
            output=episode_source,
            qa_sampling=qa,
            items_per_type=int(items_per_type),
            seed=seed,
        )
        merged_catalog = merge_binding_catalogs(
            [
                core_source / "catalog_index.json",
                episode_source / "catalog_index.json",
            ],
            output=merged_root,
            seed=seed,
        )
        catalog = merged_catalog
        core = (
            Path(core_bundle).expanduser().resolve()
            if core_bundle is not None
            else core_bundle_for(bundles, output=catalog_root)
        )
        delivery = export_binding_delivery(
            str(core), merged_root / "catalog_index.json", destination
        )
        result = {
            "schema": SCHEMA,
            "status": "exported",
            "delivery_kind": "mixed_merged_catalog",
            "bundle_paths": [str(path) for path in bundles],
            "source_catalogs": [
                str(core_source / "catalog_index.json"),
                str(episode_source / "catalog_index.json"),
            ],
            "catalog_root": str(merged_root),
            "catalog_index": str(merged_root / "catalog_index.json"),
            "core_bundle": str(core),
            "catalog_summary": {
                key: merged_catalog[key]
                for key in (
                    "av_sample_count", "group_count", "world_count",
                    "catalog_question_count", "core_question_count",
                    "generated_by_qa",
                )
                if key in merged_catalog
            },
            "delivered_episode_count": len(episodes),
            "delivered_group_count": len(groups),
            "delivery_root": str(destination),
            "delivery": deepcopy(delivery),
        }
    elif episodes:
        specs = []
        for row in episodes:
            spec = deepcopy(dict(row))
            media = dict(spec.get("media") or {})
            if not media.get("video_path"):
                media["video_path"] = spec.get("video_path")
            if not media.get("audio_path"):
                media["audio_path"] = spec.get("audio_path")
            spec["media"] = media
            specs.append(spec)
        catalog = derive_episode_catalog(
            specs,
            output=catalog_root,
            qa_sampling=qa,
            items_per_type=int(items_per_type),
            seed=seed,
        )
        delivery = export_binding_delivery(
            None, catalog_root / "catalog_index.json", destination
        )
        result = {
            "schema": SCHEMA,
            "status": "exported",
            "delivery_kind": "episode_catalog",
            "bundle_paths": [],
            "catalog_root": str(catalog_root),
            "catalog_index": str(catalog_root / "catalog_index.json"),
            "core_bundle": None,
            "catalog_summary": {
                key: catalog[key]
                for key in (
                    "av_sample_count",
                    "group_count",
                    "world_count",
                    "catalog_question_count",
                    "core_question_count",
                    "generated_by_qa",
                )
                if key in catalog
            },
            "delivered_episode_count": len(episodes),
            "delivered_group_count": 0,
            "delivery_root": str(destination),
            "delivery": deepcopy(delivery),
        }
    else:
        bundles = _bundle_paths(groups)
        if not bundles:
            return {
                "schema": SCHEMA,
                "status": "no_delivered_group",
                "reason": "no group in this run reached a media-checked assembled bundle",
                "delivered_group_count": 0,
                "delivered_episode_count": 0,
            }
        catalog = derive_binding_catalog(
            [str(path) for path in bundles],
            output=catalog_root,
            qa_sampling=qa,
            items_per_type=int(items_per_type),
            seed=seed,
        )
        core = (
            Path(core_bundle).expanduser().resolve()
            if core_bundle is not None
            else core_bundle_for(bundles, output=catalog_root)
        )
        delivery = export_binding_delivery(
            str(core), catalog_root / "catalog_index.json", destination
        )
        result = {
            "schema": SCHEMA,
            "status": "exported",
            "delivery_kind": "core_groups",
            "bundle_paths": [str(path) for path in bundles],
            "catalog_root": str(catalog_root),
            "catalog_index": str(catalog_root / "catalog_index.json"),
            "core_bundle": str(core),
            "catalog_summary": {
                key: catalog[key]
                for key in (
                    "av_sample_count",
                    "group_count",
                    "world_count",
                    "catalog_question_count",
                    "core_question_count",
                    "generated_by_qa",
                )
                if key in catalog
            },
            "delivered_episode_count": 0,
            "delivered_group_count": len(groups),
            "delivery_root": str(destination),
            "delivery": deepcopy(delivery),
        }

    result["audio_attachments"] = _attach_exported_audio_layouts(
        destination, catalog, groups, episodes
    )
    if build_index:
        result["dataset_index"] = deepcopy(build_dataset_index(destination))
    return result


def run_production(
    *,
    manifest_path: str | Path,
    run_root: str | Path,
    delivery_output: str | Path | None = None,
    imported_bundle_paths: Sequence[str | Path] = (),
    resume: bool = False,
    reopen_failed_units: Sequence[str] = (),
    replan_units: Sequence[str] = (),
    **runner_kwargs: Any,
) -> dict[str, Any]:
    """One call: generate or resume, backfill what is short, and export.

    This is what a configuration entry point drives. Nothing in the sequence
    stops to hand a command back to a caller.
    """
    root = Path(run_root).expanduser().resolve()
    if resume or (root / "state.json").is_file():
        runner = ProductionRunner.resume(root, **runner_kwargs)
    else:
        runner = ProductionRunner(
            manifest=_read_json(Path(manifest_path)), manifest_path=manifest_path,
            run_root=root, **runner_kwargs)
    replanned = None
    if replan_units:
        replanned = runner.replan_units(list(replan_units))
        if replanned["unmatched"]:
            raise ProductionRunError(
                "no current result to replan for "
                f"{replanned['unmatched']}"
            )
    reopened = None
    if reopen_failed_units:
        if not (root / "state.json").is_file():
            raise ProductionRunError(
                "there is nothing to re-open in a run that has no state yet"
            )
        reopened = runner.reopen_failed_units(list(reopen_failed_units))
        if reopened["unmatched"]:
            raise ProductionRunError(
                "no failed result to re-open for "
                f"{reopened['unmatched']}; naming a unit that did not fail "
                "would silently do nothing"
            )
    summary = runner.run()
    feedback = coverage_feedback(
        manifest=runner.manifest,
        delivered_groups=summary["delivered_groups"],
        delivered_episodes=summary.get("delivered_episodes") or (),
        imported_bundle_paths=imported_bundle_paths)
    _json_write_atomic(root / "coverage_feedback.json", feedback)
    # Backfill: a scope that is short and still has ready units gets another
    # pass. A deficit the configuration cannot supply is reported, not invented.
    backfilled = None
    if feedback["deficits"] and any(not scope["delivered"] for scope in summary["scopes"]):
        for scope in runner.scopes:
            scope.finished = False
            scope.finished_reason = None
        backfilled = runner.run()
        summary = backfilled
        feedback = coverage_feedback(
            manifest=runner.manifest,
            delivered_groups=summary["delivered_groups"],
            delivered_episodes=summary.get("delivered_episodes") or (),
            imported_bundle_paths=imported_bundle_paths)
        _json_write_atomic(root / "coverage_feedback.json", feedback)
    p19_append = None
    p19_inputs = _p19_coverage_inputs(runner.manifest)
    if (
        isinstance(p19_inputs, Mapping)
        and p19_inputs.get("append_requests")
        and isinstance(feedback.get("next_requests"), Mapping)
        and all(scope.finished for scope in runner.scopes)
    ):
        round_limit = p19_inputs.get("round_limit", 1)
        rounds_completed = p19_inputs.get("rounds_completed", 0)
        if (
            isinstance(round_limit, int)
            and round_limit > 0
            and isinstance(rounds_completed, int)
            and rounds_completed < round_limit
        ):
            p19_append = _append_p19_requests_to_runner(
                runner, feedback["next_requests"], p19_inputs
            )
            if (
                p19_append.get("status") == "appended"
                and p19_inputs.get("run_appended")
            ):
                summary = runner.run()
                feedback = coverage_feedback(
                    manifest=runner.manifest,
                    delivered_groups=summary.get("delivered_groups") or (),
                    delivered_episodes=summary.get("delivered_episodes") or (),
                    imported_bundle_paths=imported_bundle_paths)
                _json_write_atomic(root / "coverage_feedback.json", feedback)
    export = None
    if delivery_output is not None:
        export = export_run_delivery(
            delivered_groups=summary.get("delivered_groups") or (),
            delivered_episodes=summary.get("delivered_episodes") or (),
            output=delivery_output)
        _json_write_atomic(root / "delivery_export.json", export)
        if export.get("delivery_root"):
            feedback = coverage_feedback(
                manifest=runner.manifest,
                delivered_groups=summary.get("delivered_groups") or (),
                delivered_episodes=summary.get("delivered_episodes") or (),
                imported_bundle_paths=imported_bundle_paths,
                delivery_roots=[export["delivery_root"]],
            )
            _json_write_atomic(root / "coverage_feedback.json", feedback)
    result = {
        "schema": SCHEMA,
        "status": summary["status"],
        "run_summary": summary,
        "coverage_feedback": feedback,
        "backfill_ran": backfilled is not None,
        "p19_append": p19_append,
        "delivery_export": export,
        "run_root": str(root),
        "reopened_failed_units": deepcopy(reopened),
        "replanned_units": deepcopy(replanned),
        "resource_policy_source": runner.resource_policy_source,
        "resource_policy": runner.broker.allocator.policy.to_dict(),
        "resource_policy_override": deepcopy(runner.resource_policy_override),
    }
    _json_write_atomic(root / "production_result.json", result)
    return result


# ---------------------------------------------------------------------------
# Worker entry point
# ---------------------------------------------------------------------------


def _context_from_task(task: Mapping[str, Any]) -> StageContext:
    data = dict(task["context"])
    manifest_path = Path(data["manifest_path"])
    return StageContext(
        work_item=dict(data["work_item"]),
        manifest=_read_json(manifest_path),
        manifest_path=manifest_path,
        group_id=data.get("group_id"),
        task_family=data.get("task_family"),
        unit_id=data.get("unit_id"),
        stage=str(data["stage"]),
        scope_id=str(data["scope_id"]),
        attempt=int(data["attempt"]),
        member_request_ids=tuple(data.get("member_request_ids") or ()),
        rows_by_episode_id={
            str(row["episode_id"]): row
            for row in _read_json(manifest_path).get("episodes", [])},
        upstream={key: dict(value) for key, value in (data.get("upstream") or {}).items()},
        output_root=Path(data["output_root"]),
        run_root=Path(data["run_root"]),
        repository=Path(data["repository"]),
        rpc_port=data.get("rpc_port"),
        graphics_adapter=data.get("graphics_adapter"),
        lease=data.get("lease"),
        round_results=tuple(dict(row) for row in (data.get("round_results") or ())),
        recipe_options=dict(data.get("recipe_options") or {}),
    )


def execute_task_file(task_path: str | Path, result_path: str | Path) -> int:
    """Run one work item in this fresh interpreter and write its outcome."""
    import traceback

    task = _read_json(Path(task_path))
    result_file = Path(result_path)
    try:
        context = _context_from_task(task)
        unit_kind = str(task["executor"]["unit_kind"])
        entry = resolve_stage_executor(
            context.task_family, unit_kind, unit_id=context.unit_id
        )
        outcome = entry.call(context)
        payload = {
            "schema": SCHEMA, "status": "ok",
            "work_item_id": context.work_item["work_item_id"],
            "executor": entry.to_dict(),
            "outcome": outcome.to_dict(),
            "worker": {
                "python_executable": sys.executable,
                "python_version": platform.python_version(),
                "pythonpath": os.environ.get("PYTHONPATH", ""),
                "pid": os.getpid(),
                "cwd": str(Path.cwd()),
            },
        }
        _json_write_atomic(result_file, payload)
        return 0
    except Exception as error:  # the worker reports rather than dying silently
        trace_path = result_file.with_name(result_file.stem + ".traceback.log")
        trace_path.write_text(traceback.format_exc(), encoding="utf-8")
        reason_code = getattr(type(error), "reason_code", None) or "worker_failed"
        _json_write_atomic(result_file, {
            "schema": SCHEMA, "status": "error",
            "reason": f"{type(error).__name__}: {error}",
            "reason_code": str(reason_code),
            "details": deepcopy(getattr(error, "details", {}) or {}),
            "traceback_path": str(trace_path),
            "outcome": {"status": "fail", "reason": f"{type(error).__name__}: {error}",
                        "reason_code": str(reason_code)},
            "worker": {
                "python_executable": sys.executable,
                "pythonpath": os.environ.get("PYTHONPATH", ""),
                "pid": os.getpid(),
            },
        })
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-work-item", type=Path,
                        help="worker mode: run the single unit described by this task file")
    parser.add_argument("--result", type=Path,
                        help="worker mode: where to write the stage outcome")
    parser.add_argument("--describe-executors", action="store_true",
                        help="print which recipe units are actually wired")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.describe_executors:
        print(json.dumps({"schema": SCHEMA, "executors": registered_stage_executors(),
                          "pending_recipe_seams": PENDING_RECIPE_SEAMS},
                         ensure_ascii=False, indent=2))
        return 0
    if args.execute_work_item is None or args.result is None:
        parser.error("--execute-work-item needs --result")
    return execute_task_file(args.execute_work_item, args.result)


if __name__ == "__main__":
    raise SystemExit(main())
