"""Native producers for controlled audiovisual binding groups.

The producer has two layers: execute supplied native visual plans and finalize
supplied audio-event assignments. It reuses audio only after actual native
readback and acoustic-input agreement. Retained captures are never changed.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
import filecmp
import math
import os
from numbers import Integral
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

from avengine.dataset.production_spec import (
    AUDIO_CONTENT_SCOPES,
    PLAN_EQUIVALENCE_MODES,
    QUERY_IDENTITY_POLICIES,
    SOUND_SELECTION_POLICY_FIELDS,
    VISUAL_INTERVENTION_MODES,
    normalize_sound_selection_policy,
)

REPOSITORY = Path(__file__).resolve().parents[3]
ROOM_FAMILIES = frozenset({"apartment", "kujiale", "hm3d", "mp3d"})
TASK_FAMILY = "visible_binding"
RELATION_TASK_FAMILY = "visual_conditioned_relation"


class BindingNativeError(RuntimeError):
    """A supplied native input cannot produce a binding group."""


class RequestedVisibilityError(BindingNativeError):
    """The captured pixels do not support the explicitly requested QA condition."""


def check_requested_visibility(
    plan: Mapping[str, Any], request: Mapping[str, Any], capture: str | Path,
    *, report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Use the existing pixel judge before paying for audio or declaring capture pass."""
    from dataclasses import fields
    from avengine.rooms.conditioned_visibility import (
        VisibilityRequirement, accept_native_visibility, requirements_from_conditions,
    )

    targets = request.get("qa_targets")
    if not isinstance(targets, list) or not targets:
        return {"status": "not_requested", "reason": "no explicit QA targets"}
    target_ids = {str(row.get("qa_id")) for row in targets if isinstance(row, Mapping)}
    match = plan.get("question_condition_match") or {}
    compiled = [row for row in match.get("compiled") or ()
                if isinstance(row, Mapping) and str(row.get("qa_id")) in target_ids]
    screen = (plan.get("camera_condition_sampling") or {}).get("visibility_solver") or {}
    saved = [row for row in screen.get("requirements") or ()
             if isinstance(row, Mapping) and str(row.get("qa_id")) in target_ids]
    windows = {str(row["subject"]): row.get("observation_windows") or ()
               for row in saved if row.get("subject")}
    requirements = []
    if saved:
        # Keep the exact per-requirement windows solved by the planner.
        allowed = {field.name for field in fields(VisibilityRequirement)}
        requirements = [VisibilityRequirement(**{k: v for k, v in row.items() if k in allowed})
                        for row in saved]
    else:
        for item in compiled:
            requirements.extend(requirements_from_conditions(
                item.get("conditions") or (), observation_windows_by_subject=windows,
                public_time_precision=0, qa_id=item.get("qa_id")))
    if not requirements:
        optical_kinds = {"visibility_state", "entry_transition", "occlusion_transition",
                         "occluder_identity"}
        untranslated = [condition.get("key")
                        for item in compiled for condition in item.get("conditions") or ()
                        if condition.get("kind") in optical_kinds]
        if not compiled or untranslated:
            report = {"status": "incomplete",
                      "reason": "requested planning conditions lack pixel requirements",
                      "untranslated_conditions": untranslated}
        else:
            return {"status": "not_requested",
                    "reason": "the explicit targets carry no pixel-state requirement"}
    else:
        capture = Path(capture)
        pixel_path = capture / "pixel_visibility_truth.json"
        camera = (plan.get("visual_plan") or {}).get("camera") or {}
        clock = plan.get("clock") or {}
        actors = (plan.get("visual_plan") or {}).get("actors") or []
        actor_mapping = {str(actor.get("entity_instance_id") or actor["actor_id"]):
                         str(actor["actor_id"]) for actor in actors}
        if not pixel_path.is_file() or not camera.get("resolution_hw"):
            report = {"status": "incomplete",
                      "reason": "pixel truth or planned camera resolution is missing"}
        else:
            occluders = capture / "actor_occluders.json"
            occluder_evidence = _load(occluders) if occluders.is_file() else None
            occluder_registry = None
            visual_root = None
            if (occluder_evidence is None
                    and (capture / "native_pixel_masks_depth_authority_v1.npz").is_file()
                    and any(r.kind == "registered_occluder_visible" for r in requirements)):
                from avengine.rooms.qa_evidence import acquire_shared_visual_evidence
                from avengine.rooms.qa_delivery import _asset_registry, _reviewed_occluder_registry
                visual_root = (Path(report_path).parent / "native_visibility_visual_evidence"
                               if report_path is not None else None)
                visual = acquire_shared_visual_evidence(
                    capture, plan, _load(pixel_path), shared_root=visual_root,
                    asset_registry=_asset_registry(REPOSITORY, request.get("source_registry")),
                    frame_stride=1)
                occluder_evidence = visual["actor_occluders"]
                occluder_registry = _reviewed_occluder_registry(
                    visual["appearance_review"], {str(a["actor_id"]): a for a in actors})
            report = accept_native_visibility(
                requirements, pixel_truth=_load(pixel_path),
                frame_rate_hz=float(clock["frame_rate_hz"]),
                frame_count=int(clock["frame_count"]),
                resolution_hw=camera["resolution_hw"],
                actor_by_instance=actor_mapping,
                occluder_evidence=occluder_evidence,
                occluder_registry=occluder_registry,
                public_time_precision=0,
                acceptance_policy=(request.get("qa_sampling") or {}).get("acceptance_policy"),
                screen=screen if screen.get("record") == "screen_series" else None)
            if not report.get("pixel_truth_authority_registered"):
                report["status"] = "fail"
                report["reason"] = "pixel truth authority is not registered"
            report["pixel_truth_path"] = str(pixel_path.resolve())
            if visual_root is not None:
                report["shared_visual_root"] = str(visual_root.resolve())
    keep_scene = False
    reason = None
    if report["status"] != "pass":
        reasons = [str(row.get("reason")) for row in report.get("requirements") or ()
                   if row.get("status") != "pass"]
        reason = "; ".join(reasons) or str(report.get("reason") or report["status"])
        policy = (request.get("qa_sampling") or {}).get("acceptance_policy") or {}
        keep_scene = isinstance(policy, dict) and bool(policy.get("keep_scene_when_target_unmet"))
        if keep_scene:
            # An ordinary scene that missed its requested pixel condition is
            # still a legal scene for every other question type. Record the
            # miss and let audio and delivery continue; the requested target
            # stays unmet in the accounting and no question is invented for it.
            report["requested_target_unmet"] = True
            report["requested_target_unmet_reason"] = reason
            report["salvage"] = "scene_kept_for_other_question_types"
    if report_path is not None:
        _write(Path(report_path), report)
    if report["status"] != "pass" and not keep_scene:
        raise RequestedVisibilityError(
            "requested visibility conditions not satisfied; replan before audio: " + reason)
    return report


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BindingNativeError(f"cannot read JSON input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BindingNativeError(f"JSON input must be an object: {path}")
    return value


def _write(path: Path, value: Any) -> Path:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise BindingNativeError(f"refusing to replace output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _file(value: Any, *, base: Path, owner: str) -> Path:
    if isinstance(value, Path):
        value = str(value)
    if not isinstance(value, str) or not value.strip():
        raise BindingNativeError(f"{owner} must be a nonempty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise BindingNativeError(f"{owner} is unavailable: {path}")
    return path


def _env(repository: Path) -> dict[str, str]:
    value = dict(os.environ)
    paths = [str((repository / "src").resolve())]
    addons = repository / "tmp/native_python_addons_v1"
    if addons.is_dir():
        paths.append(str(addons.resolve()))
    if value.get("PYTHONPATH"):
        paths.append(value["PYTHONPATH"])
    value["PYTHONPATH"] = os.pathsep.join(paths)
    value["PYTHONDONTWRITEBYTECODE"] = "1"
    return value


def _run(command: Sequence[str], *, log: Path, label: str, cwd: Path | None = None) -> dict[str, Any]:
    log = log.expanduser().resolve()
    if log.exists() or log.is_symlink():
        raise BindingNativeError(f"refusing to replace {label} log: {log}")
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    environment = _env(REPOSITORY)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8") as stream:
        completed = subprocess.run(
            [str(item) for item in command],
            cwd=(cwd or REPOSITORY).expanduser().resolve(), env=environment,
            stdout=stream, stderr=subprocess.STDOUT, check=False,
        )
    result = {
        "label": label, "command": [str(item) for item in command],
        "returncode": int(completed.returncode),
        "status": "pass" if completed.returncode == 0 else "fail",
        "elapsed_s": time.monotonic() - started, "started_at": started_at,
        "cwd": str((cwd or REPOSITORY).expanduser().resolve()), "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "PYTHONPATH": environment.get("PYTHONPATH", ""), "log": str(log),
    }
    if completed.returncode:
        raise BindingNativeError(
            f"{label} failed with returncode {completed.returncode}; see {log}"
        )
    return result


# ---------------------------------------------------------------------------
# Which registered assets may take part in a controlled visual swap
#
# The visual intervention of a binding group swaps which registered asset holds
# each generic source slot. That is only a controlled intervention when the
# swap changes the appearance and nothing else: two assets that render
# different bodies would also move the silhouette, the pose and the emitter,
# and the group would no longer isolate appearance.
#
# Which assets qualify is therefore read from the registry rather than from an
# asset id spelling or a hand-kept list. These are the record fields that drive
# a transform, a pose, an animation phase or an emitter position; if any of
# them differs the two assets are different bodies.
# ---------------------------------------------------------------------------

#: Registry paths compared to decide that two records render one body.
CONTROLLED_SWAP_BODY_FIELDS = (
    ("timeline", "template_id"),
    ("timeline", "body_plan_id"),
    ("timeline", "walk_phase_period_frames"),
    ("timeline", "idle_action_id"),
    ("timeline", "walking_action_id"),
    ("timeline", "local_anatomical_forward_axis"),
    ("geometry", "rig_authority"),
    ("emitter_anchors",),
    ("default_emitter_anchor_id",),
    ("runtime_backends", "spear_unreal", "actor_scale"),
    ("runtime_backends", "habitat", "asset_kind"),
)

#: Registry paths that carry a registered appearance value, in the order the
#: delivery's appearance review reads them. Kept next to the swap rule so a
#: newly registered appearance field enters both at once.
CONTROLLED_SWAP_APPEARANCE_FIELDS = (
    "finish", "surface_finish", "body_color", "top_color",
)


def _registry_at(record: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = record
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def controlled_swap_body_key(record: Mapping[str, Any]) -> str:
    """A stable key for the body a registered asset renders.

    Two records with one key may swap slots without moving anything: every
    field that places, poses or animates the asset, and the emitter anchors
    that decide where its sound leaves it, are equal. The asset id is not part
    of the key, so two differently named registrations of one body match and a
    renamed asset does not stop matching.
    """
    if not isinstance(record, Mapping):
        raise BindingNativeError("a registry record must be a mapping")
    parts = [
        json.dumps(_registry_at(record, path), ensure_ascii=False, sort_keys=True)
        for path in CONTROLLED_SWAP_BODY_FIELDS
    ]
    return "|".join(parts)


def registered_appearance(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """The appearance field and value a registry record declares, if any."""
    attributes = record.get("realized_attributes")
    if not isinstance(attributes, Mapping):
        return None
    for field in CONTROLLED_SWAP_APPEARANCE_FIELDS:
        value = attributes.get(field)
        if isinstance(value, str) and value.strip():
            return {"field": field, "value": value.strip()}
    return None


def appearance_family_of(record: Mapping[str, Any]) -> str | None:
    """The colour family the registered appearance belongs to.

    Two registered values inside one family are one colour on screen, so a
    binding group made of them could not be answered by looking. The family is
    computed by the shared appearance rule, not by comparing spellings.
    """
    from avengine.rooms.appearance_color import appearance_distinction_family

    appearance = registered_appearance(record)
    if appearance is None:
        return None
    identity = record.get("identity") if isinstance(record.get("identity"), Mapping) else {}
    species = identity.get("species_id") or record.get("entity_class")
    family = appearance_distinction_family(appearance["value"], species)
    return str(family or appearance["value"])


def select_controlled_swap_assets(
    registry: Mapping[str, Any], *, entity_class: str = "articulated_human",
    count: int = 2, prefer: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Choose registered assets that differ only in a distinguishable appearance.

    Every candidate comes from the supplied registry: the assets are grouped by
    body key, each group is reduced to one asset per appearance family, and the
    body that offers the most distinguishable appearances is used. prefer names
    assets the caller already decided on; they are still checked against the
    same two rules and refused with a reason rather than silently accepted.
    """
    if not isinstance(count, Integral) or isinstance(count, bool) or int(count) < 2:
        raise BindingNativeError("a controlled swap needs at least two assets")
    count = int(count)
    assets = registry.get("assets") if isinstance(registry, Mapping) else None
    if not isinstance(assets, list) or not assets:
        raise BindingNativeError("the source registry declares no assets")
    records = {
        str(row["asset_id"]): row for row in assets
        if isinstance(row, Mapping) and isinstance(row.get("asset_id"), str)
        and row["asset_id"].strip()
        and (entity_class is None or row.get("entity_class") == entity_class)
    }
    if prefer:
        chosen = [str(value) for value in prefer]
        missing = [value for value in chosen if value not in records]
        if missing:
            raise BindingNativeError(
                f"requested swap assets are not registered as {entity_class}: {missing}"
            )
        keys = {controlled_swap_body_key(records[value]) for value in chosen}
        if len(keys) != 1:
            raise BindingNativeError(
                "requested swap assets do not render one body; a visual swap between "
                "them would move geometry as well as appearance"
            )
        families = [appearance_family_of(records[value]) for value in chosen]
        if any(family is None for family in families):
            raise BindingNativeError(
                "every asset in a controlled swap needs a registered appearance value"
            )
        if len(set(families)) != len(families):
            raise BindingNativeError(
                f"requested swap assets share a colour family: {families}"
            )
        selected = chosen
    else:
        by_body: dict[str, dict[str, str]] = {}
        for asset_id, record in sorted(records.items()):
            family = appearance_family_of(record)
            if family is None:
                continue
            by_body.setdefault(controlled_swap_body_key(record), {}).setdefault(
                family, asset_id)
        usable = {key: value for key, value in by_body.items() if len(value) >= count}
        if not usable:
            raise BindingNativeError(
                f"no registered {entity_class} body carries {count} appearances in "
                "different colour families"
            )
        body = max(sorted(usable), key=lambda key: len(usable[key]))
        selected = [usable[body][family] for family in sorted(usable[body])][:count]
    return {
        "asset_ids": list(selected),
        "body_key": controlled_swap_body_key(records[selected[0]]),
        "appearances": {
            asset_id: registered_appearance(records[asset_id]) for asset_id in selected
        },
        "appearance_families": {
            asset_id: appearance_family_of(records[asset_id]) for asset_id in selected
        },
        "authority": (
            "one registry body key across every selected asset and one colour family "
            "per asset; both computed from the registry record, not from asset names"
        ),
    }


def _keep_declared(
    target: dict[str, Any], key: str, required: Any, *, supplied: dict[str, Any],
    owner: str | None = None,
) -> None:
    """Fill in a recipe-required value, or refuse to overwrite a different one."""
    name = owner or key
    current = target.get(key)
    if current is None:
        target[key] = required
        supplied[name] = required
        return
    if current != required:
        raise BindingNativeError(
            f"the request declares {name}={current!r} but this recipe requires {required!r}; "
            "an explicit request is not silently replaced by a route default"
        )


def _declared_instance_rows(request: Mapping[str, Any]) -> list[tuple[str, int]]:
    """Every place a request pins one entity instance, in declaration order."""
    rows: list[tuple[str, int]] = []
    for owner in ("entity_instances", "entities"):
        value = request.get(owner)
        instances = (
            value.get("instances") if owner == "entities" and isinstance(value, Mapping)
            else value
        )
        if not isinstance(instances, list):
            continue
        for index, row in enumerate(instances):
            if isinstance(row, Mapping):
                rows.append((owner, index))
    return rows


def _instance_list(request: Mapping[str, Any], owner: str) -> list[Any] | None:
    value = request.get(owner)
    if owner == "entities":
        value = value.get("instances") if isinstance(value, Mapping) else None
    return value if isinstance(value, list) else None


def _base_selected_assets(
    base_request: Mapping[str, Any], count: int
) -> list[str] | None:
    """The asset order the base request selected, however it stated it."""
    declared = base_request.get("source_asset_ids")
    if (
        isinstance(declared, Sequence) and not isinstance(declared, (str, bytes))
        and len(declared) == count
        and all(isinstance(value, str) and value.strip() for value in declared)
    ):
        return [str(value) for value in declared]
    for owner in ("entity_instances", "entities"):
        instances = _instance_list(base_request, owner)
        if instances is None:
            continue
        pinned = [str(row["asset_id"]) for row in instances
                  if isinstance(row, Mapping) and isinstance(row.get("asset_id"), str)
                  and row["asset_id"].strip()]
        if len(pinned) == count:
            return pinned
    return None


def permute_declared_instance_assets(
    request: dict[str, Any], *, base_order: Sequence[str], new_order: Sequence[str],
) -> dict[str, Any]:
    """Move the declared per-instance assets with the selected asset order.

    A request may pin which registered asset each entity instance holds as well
    as listing the selected assets. The visual intervention reorders the
    selection, and an instance list left behind would then contradict it: the
    planner takes the actor bindings from the selection while the compiled
    condition profile keeps repeating the stale pin, and the two variants can no
    longer be compared. Both statements are rewritten with one permutation, so
    they keep saying the same thing.

    The permutation is taken from the two selections rather than from instance
    order, so it also holds for a request whose instances are declared in a
    different order or which pins only some of them.
    """
    if len(base_order) != len(new_order) or set(base_order) != set(new_order):
        raise BindingNativeError(
            "an asset-order intervention must be a permutation of the same assets"
        )
    mapping = {str(old): str(new) for old, new in zip(base_order, new_order, strict=True)}
    moved: list[dict[str, Any]] = []
    for owner in ("entity_instances", "entities"):
        instances = _instance_list(request, owner)
        if instances is None:
            continue
        for index, row in enumerate(instances):
            if not isinstance(row, Mapping):
                continue
            current = row.get("asset_id")
            if not isinstance(current, str) or current not in mapping:
                continue
            replacement = mapping[current]
            if replacement == current:
                continue
            updated = dict(row)
            updated["asset_id"] = replacement
            instances[index] = updated
            moved.append({"owner": owner, "instance_id": row.get("instance_id"),
                          "from": current, "to": replacement})
    return {
        "status": "applied" if moved else "no_declared_instance_assets",
        "moved": moved,
        "authority": ("declared instance assets follow the selected asset order, so the "
                      "compiled condition profile and the planned actors cannot disagree"),
    }


def build_variant_request(
    base_request: Mapping[str, Any], *, episode_id: str,
    source_asset_ids: Sequence[str], rpc_port: int | None = None,
    graphics_adapter: int | None = None,
    qa_ids: Sequence[str] | None = None, seed: int | None = None,
) -> dict[str, Any]:
    """Copy a request while changing the selected visual asset order.

    The low-level request, plan, and capture layers accept two or more source
    assets. Task recipes apply stricter arity at their own boundary, so this
    path can support a three-candidate relation.

    Only the declared intervention changes. A camera, clock, sampling policy or
    question selection that the request already states is never silently
    replaced by a recipe default: a conflicting value is an error, and an
    absent one is filled in and recorded under
    request["binding_variant"]["recipe_supplied"]. rpc_port and
    graphics_adapter are the per-instance values a resource lease selected;
    when they are omitted the request keeps whatever it already declared.
    """
    if (
        isinstance(source_asset_ids, (str, bytes))
        or not isinstance(source_asset_ids, Sequence)
        or len(source_asset_ids) < 2
        or len(set(source_asset_ids)) != len(source_asset_ids)
        or any(not isinstance(value, str) or not value.strip() for value in source_asset_ids)
    ):
        raise BindingNativeError("at least two distinct source assets are required")
    if not episode_id.strip():
        raise BindingNativeError("invalid episode_id")
    if rpc_port is not None and (isinstance(rpc_port, bool) or not isinstance(rpc_port, int) or rpc_port <= 0):
        raise BindingNativeError("rpc_port must be a positive integer when supplied")
    if graphics_adapter is not None and (
        isinstance(graphics_adapter, bool)
        or not isinstance(graphics_adapter, int)
        or graphics_adapter < 0
    ):
        raise BindingNativeError("graphics_adapter must be a nonnegative integer when supplied")
    request = deepcopy(dict(base_request))
    supplied: dict[str, Any] = {}
    request["episode_id"] = episode_id
    base_order = _base_selected_assets(base_request, len(source_asset_ids))
    request["source_asset_ids"] = list(source_asset_ids)
    instance_permutation = (
        permute_declared_instance_assets(
            request, base_order=base_order, new_order=[str(v) for v in source_asset_ids])
        if base_order is not None
        else {"status": "base_request_declares_no_selection",
              "moved": [],
              "authority": "nothing to move: the base request pins no selected asset order"}
    )
    declared_qa = request.get("qa_ids")
    if qa_ids is None:
        if (
            isinstance(declared_qa, (str, bytes))
            or not isinstance(declared_qa, Sequence)
            or not [value for value in declared_qa if isinstance(value, str) and value.strip()]
        ):
            raise BindingNativeError(
                "the request must declare qa_ids, or the caller must select them explicitly; "
                "a route default must not stand in for a configured question selection"
            )
        selected_qa = [str(value) for value in declared_qa]
        qa_source = "request"
    else:
        selected_qa = [str(value) for value in qa_ids]
        if not selected_qa:
            raise BindingNativeError("an explicit qa_ids selection must not be empty")
        qa_source = "caller"
    request["qa_ids"] = selected_qa
    _keep_declared(request, "sampling_policy", "conditioned_static_v2", supplied=supplied)
    entities = dict(request.get("entities") or {})
    entities.update(total_count=len(source_asset_ids), silent_count=0)
    request["entities"] = entities
    camera = dict(request.get("camera") or {})
    _keep_declared(camera, "motion", "static", supplied=supplied, owner="camera.motion")
    request["camera"] = camera
    runtime = dict(request.get("runtime") or {})
    if rpc_port is not None:
        runtime["rpc_port"] = int(rpc_port)
    if graphics_adapter is not None:
        runtime["graphics_adapter"] = int(graphics_adapter)
    request["runtime"] = runtime
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise BindingNativeError("seed must be an integer when supplied")
        request["seed"] = int(seed)
    if not isinstance(request.get("room_id"), str) or not request["room_id"]:
        raise BindingNativeError("base request must select one room_id")
    request["binding_variant"] = {
        "selected_source_asset_ids": list(request["source_asset_ids"]),
        "qa_ids_source": qa_source,
        "declared_qa_ids": (
            [str(value) for value in declared_qa]
            if isinstance(declared_qa, Sequence) and not isinstance(declared_qa, (str, bytes))
            else None
        ),
        "recipe_supplied": dict(sorted(supplied.items())),
        "instance_runtime_source": {
            "rpc_port": "lease" if rpc_port is not None else "request",
            "graphics_adapter": "lease" if graphics_adapter is not None else "request",
        },
        "entity_counts_follow": "selected_source_asset_ids",
        "declared_instance_assets": instance_permutation,
    }
    return request


def _qa_module() -> Any:
    path = REPOSITORY / "tools/studio/run_qa_episode.py"
    spec = importlib.util.spec_from_file_location("binding_run_qa_episode", path)
    if spec is None or spec.loader is None:
        raise BindingNativeError(f"cannot load planner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plan_visual_variant(
    request_path: str | Path, output: str | Path, *, label: str,
    log: str | Path | None = None,
) -> dict[str, Any]:
    """Plan one visual variant through the existing QA planner.

    log places the planner log explicitly; a staged run keeps it inside the
    attempt directory so a retry does not collide with the previous attempt.
    """
    output = Path(output).expanduser().resolve()
    command = [
        sys.executable, str(REPOSITORY / "tools/studio/run_qa_episode.py"),
        "--request", str(Path(request_path).expanduser().resolve()),
        "--output", str(output), "--plan-only",
    ]
    log_path = Path(log) if log is not None else output.parent / f"{label}.plan.log"
    process = _run(command, log=log_path, label=f"{label}_plan")
    plan = output / "plan/episode_plan.json"
    if not plan.is_file():
        raise BindingNativeError(f"planner did not write {plan}")
    return {"output": str(output), "plan": str(plan), "process": process}


#: Habitat executor flags whose value the request already states once, under
#: the path bindings every stage of the chain resolves its room through. The
#: ordinary single-episode finisher bridges these before it builds the capture
#: command; a group capture goes through the same command builder, so it makes
#: the same bridge instead of asking every caller to restate the path.
CAPTURE_RUNTIME_FROM_PATH_BINDINGS = {
    "mp3d_root": "AVENGINE_MP3D_ROOT",
}


def capture_runtime_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of the request with the capture flags its path bindings imply."""
    value = deepcopy(dict(request))
    runtime = dict(value.get("runtime") or {})
    bindings = runtime.get("path_bindings")
    bindings = dict(bindings) if isinstance(bindings, Mapping) else {}
    for key, binding in CAPTURE_RUNTIME_FROM_PATH_BINDINGS.items():
        if not runtime.get(key) and bindings.get(binding):
            runtime[key] = str(bindings[binding])
    value["runtime"] = runtime
    return value


def capture_visual_plan(
    request: Mapping[str, Any], output: str | Path, *, label: str,
    log: str | Path | None = None,
) -> dict[str, Any]:
    """Execute a planned visual episode through the existing room executor.

    log places the capture log explicitly; a staged run keeps it inside the
    attempt directory so a retry does not collide with the previous attempt.
    """
    root = Path(output).expanduser().resolve()
    plan = root / "plan/episode_plan.json"
    if not plan.is_file():
        raise BindingNativeError(f"visual plan is missing: {plan}")
    effective = capture_runtime_request(request)
    command = _qa_module().capture_command(effective, root)
    log_path = Path(log) if log is not None else root.parent / f"{label}.capture.log"
    process = _run(command, log=log_path, label=f"{label}_capture", cwd=root)
    capture = root / "capture"
    plan_value = _load(plan)
    resources = plan_value.get("resources") if isinstance(plan_value.get("resources"), Mapping) else {}
    package = resources.get("room_package") if isinstance(resources.get("room_package"), Mapping) else {}
    renderer = str(package.get("renderer") or resources.get("renderer") or "").strip().lower()
    if renderer == "habitat" or resources.get("backend") == "habitat":
        # The Habitat executor leaves the renderer-neutral readback to its
        # caller, exactly as the ordinary single-episode run does. Writing it
        # here keeps the group capture on the same evidence as every other
        # episode instead of failing on a file the executor never promised.
        from avengine.capture.habitat_neutral_readback import write_habitat_neutral_readback
        from avengine.capture.neutral_readback import validate_neutral_readback

        neutral_path = capture / "neutral_readback.json"
        if neutral_path.is_file():
            validate_neutral_readback(_load(neutral_path), plan=plan_value)
        else:
            write_habitat_neutral_readback(capture, plan_value)
        required = (
            "frame_records.json", "rgb.npy", "depth.npy", "semantic.npy",
            "neutral_readback.json", "pixel_visibility_truth.json",
            "native_pixel_masks_depth_authority_v1.npz", "research_receipt.json",
        )
    else:
        required = (
            "ue_visual_only.mp4", "frame_readbacks.json", "neutral_readback.json",
            "pixel_visibility_truth.json", "native_pixel_masks_depth_authority_v1.npz",
            "research_receipt.json",
        )
    missing = [name for name in required if not (capture / name).is_file()]
    if missing:
        raise BindingNativeError(f"native capture lacks {missing}: {capture}")
    check_requested_visibility(plan_value, request, capture,
                               report_path=root / "native_visibility_acceptance.json")
    visual_video = capture / "ue_visual_only.mp4"
    return {
        "output": str(root), "plan": str(plan), "capture": str(capture),
        "visual_video": str(visual_video.resolve()) if visual_video.is_file() else None,
        "frame_readbacks": str(
            (capture / "frame_readbacks.json").resolve()
            if (capture / "frame_readbacks.json").is_file()
            else (capture / "frame_records.json").resolve()
        ),
        "neutral_readback": str((capture / "neutral_readback.json").resolve()),
        "process": process,
    }


def _delta(left: Any, right: Any) -> float:
    if isinstance(left, bool) or isinstance(right, bool):
        return 0.0 if left == right else math.inf
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        try:
            a, b = float(left), float(right)
        except (TypeError, ValueError, OverflowError):
            return math.inf
        return abs(a - b) if math.isfinite(a) and math.isfinite(b) else math.inf
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return math.inf
        return max((_delta(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)) and isinstance(right, Sequence) and not isinstance(right, (str, bytes)):
        if len(left) != len(right):
            return math.inf
        return max((_delta(a, b) for a, b in zip(left, right)), default=0.0)
    return 0.0 if left == right else math.inf


def _native_geometry(readback: Mapping[str, Any]) -> dict[str, Any]:
    camera, entities = readback.get("camera"), readback.get("entities")
    if not isinstance(camera, list) or not isinstance(entities, Mapping):
        raise BindingNativeError("neutral readback must contain camera and entities")
    selected_camera = [
        {key: deepcopy(row[key]) for key in ("frame_index", "position_m", "basis", "pts_ticks") if key in row}
        for row in camera if isinstance(row, Mapping)
    ]
    selected_entities = {}
    for actor_id, rows in entities.items():
        if not isinstance(rows, list):
            raise BindingNativeError(f"neutral entity readback is invalid: {actor_id}")
        selected_entities[str(actor_id)] = [
            {key: deepcopy(row[key]) for key in ("frame_index", "root", "emitter", "moving") if key in row}
            for row in rows if isinstance(row, Mapping)
        ]
    return {"clock": deepcopy(readback.get("clock")), "camera": selected_camera, "entities": selected_entities}


def compare_native_visuals(left: Mapping[str, Any], right: Mapping[str, Any], *, tolerance: float = 1.0e-5) -> dict[str, Any]:
    """Permit audio reuse only when actual camera/clock/emitter readbacks agree."""
    left_value, right_value = _native_geometry(_load(Path(left["neutral_readback"]))), _native_geometry(_load(Path(right["neutral_readback"])))
    if left_value["clock"] != right_value["clock"]:
        raise BindingNativeError("native visual clocks differ")
    maximum = _delta(left_value, right_value)
    if maximum > tolerance:
        raise BindingNativeError(f"native camera/clock/emitter readbacks differ: {maximum:.9g}")
    return {
        "status": "pass", "max_numeric_delta": maximum, "tolerance": tolerance,
        "authority": "actual_native_neutral_readback_camera_clock_entities",
        "left": str(Path(left["neutral_readback"]).resolve()),
        "right": str(Path(right["neutral_readback"]).resolve()),
    }


def compare_group_native_visuals(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    left_unit_id: str | None = None,
    right_unit_id: str | None = None,
    tolerance: float = 1.0e-5,
) -> dict[str, Any]:
    """Compare group readbacks under the recipe's declared path intervention.

    compare_native_visuals remains the strict comparator used by audio reuse.
    A group comparator may ignore actor root/emitter trajectories only for an
    explicit path intervention, while it always checks the native camera,
    clock, coordinate frame and each actor's declared asset binding.
    """
    visual_units = contract.get("visual_units")
    if not isinstance(visual_units, Mapping) or len(visual_units) < 2:
        raise BindingNativeError(
            "group native comparison needs at least two visual units"
        )
    unit_ids = sorted(str(value) for value in visual_units)
    left_id = str(left_unit_id or unit_ids[0])
    right_id = str(right_unit_id or unit_ids[1])
    if left_id not in visual_units or right_id not in visual_units:
        raise BindingNativeError(
            f"group native comparison names unknown visual units: {left_id}, {right_id}"
        )

    left_readback = _load(Path(left["neutral_readback"]))
    right_readback = _load(Path(right["neutral_readback"]))
    left_value = _native_geometry(left_readback)
    right_value = _native_geometry(right_readback)
    if left_value["clock"] != right_value["clock"]:
        raise BindingNativeError("native visual clocks differ")
    left_frame = left_readback.get("coordinate_frame")
    right_frame = right_readback.get("coordinate_frame")
    if left_frame != right_frame:
        raise BindingNativeError("native visual coordinate frames differ")
    camera_delta = _delta(left_value["camera"], right_value["camera"])
    if camera_delta > tolerance:
        raise BindingNativeError(
            f"group native camera readbacks differ: {camera_delta:.9g}"
        )

    def expected_assets(unit_id: str) -> list[str]:
        entry = visual_units[unit_id]
        values = entry.get("source_asset_ids") if isinstance(entry, Mapping) else None
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or not values
            or any(not isinstance(value, str) or not value.strip() for value in values)
        ):
            raise BindingNativeError(
                f"visual unit {unit_id} has no declared native asset order"
            )
        return [str(value) for value in values]

    left_assets = expected_assets(left_id)
    right_assets = expected_assets(right_id)

    def asset_bindings(
        readback: Mapping[str, Any], expected: Sequence[str], label: str
    ) -> tuple[dict[str, str], str]:
        entities = readback.get("entities")
        if not isinstance(entities, Mapping):
            raise BindingNativeError(f"{label} native readback has no entities")
        actor_ids = [f"source{index + 1}" for index in range(len(expected))]
        if set(str(key) for key in entities) != set(actor_ids):
            raise BindingNativeError(
                f"{label} native entity slots differ from its declared binding"
            )
        declared = readback.get("entity_identities")
        if declared is None:
            # Legacy neutral readbacks do not expose native identity rows. The
            # plan supplies the expected source-slot order, but that is not an
            # actual entity-binding measurement.
            return dict(zip(actor_ids, expected, strict=True)), "plan_fallback"
        if not isinstance(declared, Mapping):
            raise BindingNativeError(
                f"{label} native entity identities are invalid"
            )
        declared_by_slot = {str(key): value for key, value in declared.items()}
        if set(declared_by_slot) != set(actor_ids):
            raise BindingNativeError(
                f"{label} native entity identity rows differ from its declared binding"
            )
        bindings: dict[str, str] = {}
        for actor_id, expected_asset in zip(actor_ids, expected, strict=True):
            row = declared_by_slot.get(actor_id)
            if not isinstance(row, Mapping) or row.get("asset_id") != expected_asset:
                raise BindingNativeError(
                    f"{label} native entity binding for {actor_id} differs from "
                    f"{expected_asset!r}"
                )
            bindings[actor_id] = expected_asset
        return bindings, "native_entity_identities"

    left_bindings, left_binding_source = asset_bindings(
        left_readback, left_assets, "left"
    )
    right_bindings, right_binding_source = asset_bindings(
        right_readback, right_assets, "right"
    )
    # Preserve multiplicity: two physical instances may intentionally use the
    # same asset, so a set comparison would hide a missing or extra instance.
    if sorted(left_bindings.values()) != sorted(right_bindings.values()):
        raise BindingNativeError(
            "native visual entity populations differ between group variants"
        )

    intervention = str(contract.get("visual_intervention") or "")
    allow_declared_path = (
        str(contract.get("plan_equivalence") or "") == "world"
        and intervention in {"identity_path_topology", "after_wet_tail_motion"}
    )
    if not allow_declared_path:
        # The source slot is the physical instance correspondence. Comparing
        # through asset-id dictionaries can reorder a visible slot swap and
        # overwrites duplicate asset instances.
        entity_delta = _delta(left_value["entities"], right_value["entities"])
        if entity_delta > tolerance:
            raise BindingNativeError(
                f"native group geometry readbacks differ: {entity_delta:.9g}"
            )
    else:
        # Path recipes permit trajectory changes, but preserve the source-slot
        # frame domain for the same physical instance.
        left_indices = {
            actor_id: [row.get("frame_index") for row in rows]
            for actor_id, rows in left_value["entities"].items()
        }
        right_indices = {
            actor_id: [row.get("frame_index") for row in rows]
            for actor_id, rows in right_value["entities"].items()
        }
        if left_indices != right_indices:
            raise BindingNativeError(
                "native group frame indices differ under the declared path intervention"
            )

    native_entity_binding_verified = (
        left_binding_source == "native_entity_identities"
        and right_binding_source == "native_entity_identities"
    )
    binding_authority = (
        "actual_native_entity_binding"
        if native_entity_binding_verified
        else "plan_fallback_for_missing_entity_identities"
    )
    geometry_authority = (
        "declared_path_intervention_by_source_slot"
        if allow_declared_path
        else "strict_entity_geometry_by_source_slot"
    )
    return {
        "status": "pass",
        "max_camera_delta": camera_delta,
        "tolerance": tolerance,
        "authority": (
            "native_camera_clock_coordinate_frame_and_"
            f"{binding_authority}; {geometry_authority}"
        ),
        "entity_binding_source": {
            "left": left_binding_source,
            "right": right_binding_source,
        },
        "native_entity_binding_verified": native_entity_binding_verified,
        "left": str(Path(left["neutral_readback"]).resolve()),
        "right": str(Path(right["neutral_readback"]).resolve()),
        "path_intervention_allowed": allow_declared_path,
    }


_AUDIO_CONTENT_KEYS = frozenset({
    "audio_events",
    "audio_schedule",
    "audio_event_schedule",
    "voice_bindings",
    "audio_program",
    "audio_assignment_targets",
    "audio_assignment_variant",
})


def _without_audio_content(value: Any) -> Any:
    """Remove member-scoped audio payloads while retaining visual conditions."""
    if isinstance(value, Mapping):
        return {
            key: _without_audio_content(item)
            for key, item in value.items()
            if key not in _AUDIO_CONTENT_KEYS
        }
    if isinstance(value, list):
        return [_without_audio_content(item) for item in value]
    return deepcopy(value)


def _plan_sampling_candidate_index(plan: Mapping[str, Any]) -> int:
    """Read the candidate stream from the saved request/provenance."""
    request = plan.get("request")
    if isinstance(request, Mapping) and "sampling_candidate_index" in request:
        return _sampling_candidate_index_value(
            request.get("sampling_candidate_index"),
            owner="plan.request.sampling_candidate_index",
        )
    provenance = plan.get("sampling_provenance")
    if isinstance(provenance, Mapping) and "sampling_candidate_index" in provenance:
        return _sampling_candidate_index_value(
            provenance.get("sampling_candidate_index"),
            owner="plan.sampling_provenance.sampling_candidate_index",
        )
    if isinstance(request, Mapping):
        return _world_field(request, "sampling_candidate_index")
    return 0


def _visual_signature(
    plan: Mapping[str, Any], *, include_audio: bool = True
) -> dict[str, Any]:
    visual = plan.get("visual_plan")
    if (
        not isinstance(visual, Mapping)
        or not isinstance(visual.get("actors"), list)
        or not isinstance(visual.get("frames"), list)
    ):
        raise BindingNativeError("plan lacks visual actors/frames")
    activity_plan = deepcopy(plan.get("activity_plan"))
    camera_sampling = deepcopy(plan.get("camera_condition_sampling"))
    if not include_audio:
        activity_plan = _without_audio_content(activity_plan)
        camera_sampling = _without_audio_content(camera_sampling)
    signature = {
        "clock": deepcopy(plan.get("clock")),
        "scene": deepcopy(plan.get("scene")),
        "condition_profile": deepcopy(plan.get("condition_profile")),
        "sampling_candidate_index": _plan_sampling_candidate_index(plan),
        "activity_plan": activity_plan,
        "camera_condition_sampling": camera_sampling,
        "camera": deepcopy(visual.get("camera")),
        "actor_slots": [
            str(row.get("actor_id"))
            for row in visual["actors"]
            if isinstance(row, Mapping)
        ],
        "frames": [
            {
                "frame_index": row.get("frame_index"),
                "pts_ticks": row.get("pts_ticks"),
                "camera_state": deepcopy(row.get("camera_state")),
                "actor_states": deepcopy(row.get("actor_states")),
            }
            for row in visual["frames"]
            if isinstance(row, Mapping)
        ],
    }
    if include_audio:
        signature["audio_events"] = [
            {
                key: deepcopy(row.get(key))
                for key in (
                    "event_id",
                    "start_sample",
                    "end_sample",
                    "start_tick",
                    "end_tick",
                    "sound_asset_id",
                    "path",
                )
                if key in row
            }
            for row in plan.get("audio_events", [])
            if isinstance(row, Mapping)
        ]
    return signature


def compare_visual_plans(left_path: str | Path, right_path: str | Path) -> dict[str, Any]:
    """Require two plans to be identical down to the bound asset identities.

    Use this when the two plans are meant to name the same entities as well as
    the same world. Two variants of one controlled group deliberately swap
    which registered asset occupies each generic slot, so they are compared
    with compare_controlled_visual_plans instead.
    """
    if _visual_signature(_load(Path(left_path))) != _visual_signature(_load(Path(right_path))):
        raise BindingNativeError("visual plans differ in transform-driving fields")
    return {
        "status": "pass", "authority": "identical_planned_camera_routes_clock",
        "left": str(Path(left_path).resolve()), "right": str(Path(right_path).resolve()),
    }


def plan_slot_identities(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Which registered asset and entity instance each generic slot holds."""
    visual = plan.get("visual_plan") if isinstance(plan.get("visual_plan"), Mapping) else {}
    rows: dict[str, dict[str, Any]] = {}
    for actor in visual.get("actors", []):
        if not isinstance(actor, Mapping) or not actor.get("actor_id"):
            continue
        rows[str(actor["actor_id"])] = {
            "asset_id": actor.get("asset_id"),
            "entity_instance_id": actor.get("entity_instance_id"),
        }
    frames = visual.get("frames")
    if isinstance(frames, list) and frames and isinstance(frames[0], Mapping):
        states = frames[0].get("actor_states")
        if isinstance(states, list):
            for state in states:
                if not isinstance(state, Mapping) or not state.get("actor_id"):
                    continue
                row = rows.setdefault(str(state["actor_id"]), {})
                if row.get("entity_instance_id") is None:
                    row["entity_instance_id"] = state.get("entity_instance_id")
                if row.get("asset_id") is None:
                    row["asset_id"] = state.get("asset_id")
    return rows


def _slot_identity_tokens(plan: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Replacements that turn a bound identity into the slot that holds it."""
    rows: list[tuple[str, str]] = []
    for actor_id, row in plan_slot_identities(plan).items():
        for field in ("asset_id",):
            value = row.get(field)
            if isinstance(value, str) and value.strip():
                rows.append((value, f"<{actor_id}.{field}>"))
    rows.sort(key=lambda entry: len(entry[0]), reverse=True)
    return rows


def _with_slot_tokens(value: Any, replacements: Sequence[tuple[str, str]]) -> Any:
    if isinstance(value, str):
        # Replace only original text: a generated token must not be replaced again.
        import re
        mapping = dict(replacements)
        if not mapping:
            return value
        pattern = "|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True))
        return re.sub(pattern, lambda match: mapping[match.group(0)], value)
    if isinstance(value, Mapping):
        return {_with_slot_tokens(key, replacements): _with_slot_tokens(item, replacements)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_with_slot_tokens(item, replacements) for item in value]
    return value


def _controlled_visual_signature(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize instance bindings by physical slot, preserving their consistency."""
    identities = plan_slot_identities(plan)
    signature = _visual_signature(plan, include_audio=False)
    sampling = _normalize_planned_query_identities(signature.get("camera_condition_sampling"), plan)
    if isinstance(sampling, dict):
        sampling.pop("legal_event_start_ranges_samples", None)
        query = sampling.get("planned_query_window")
        if isinstance(query, dict):
            # Audio timing differs by member; the queried slot and event stay checked.
            timing_fields = {"window_s", "available_s", "public_query_window_s",
                             "blocked_by_other_event", "ends_at_next_event",
                             "publishable_event_ids"}
            def without_query_timing(value):
                if isinstance(value, Mapping):
                    return {k: without_query_timing(v) for k, v in value.items()
                            if k not in timing_fields}
                if isinstance(value, list):
                    return [without_query_timing(v) for v in value]
                return value
            sampling["planned_query_window"] = without_query_timing(query)
    signature["camera_condition_sampling"] = sampling
    for frame in signature["frames"]:
        for state in frame.get("actor_states") or []:
            if not isinstance(state, dict) or "entity_instance_id" not in state:
                continue
            slot = str(state.get("actor_id"))
            declared = identities.get(slot, {}).get("entity_instance_id")
            if declared is not None and state["entity_instance_id"] != declared:
                raise BindingNativeError(f"frame instance binding disagrees with slot {slot}")
            state["entity_instance_id"] = f"<slot:{slot}>"
    profile = signature.get("condition_profile")
    if isinstance(profile, dict) and isinstance(profile.get("instances"), list):
        instance_slots = {str(row.get("entity_instance_id")): slot
                          for slot, row in identities.items()
                          if row.get("entity_instance_id") is not None}
        for row in profile["instances"]:
            if not isinstance(row, dict):
                continue
            instance = row.get("entity_instance_id")
            slot = row.get("source_slot_id") or instance_slots.get(str(instance))
            if slot is None:
                raise BindingNativeError("condition profile instance has no declared source slot")
            declared = identities.get(str(slot), {})
            if instance is not None and declared.get("entity_instance_id") not in (None, instance):
                raise BindingNativeError(f"profile instance binding disagrees with slot {slot}")
            if row.get("asset_id") is not None and declared.get("asset_id") not in (None, row["asset_id"]):
                raise BindingNativeError(f"profile asset binding disagrees with slot {slot}")
            if "entity_instance_id" in row:
                row["entity_instance_id"] = f"<slot:{slot}>"
        profile["instances"].sort(key=lambda row: str(row.get("source_slot_id") or row.get("entity_instance_id")) if isinstance(row, Mapping) else str(row))
    return _with_slot_tokens(signature, _slot_identity_tokens(plan))


def compare_controlled_visual_plans(
    left_path: str | Path, right_path: str | Path, *,
    expected_slot_assets: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Require two variants of one controlled group to plan the same world.

    The declared intervention is which registered asset occupies each generic
    source slot, so every bound identity is rewritten to the slot that holds it
    before the visual plans are compared. The sampled camera route, clock,
    actor transforms, actions and movement flags still have to match exactly;
    member-scoped audio content is checked separately for the shared audio
    columns from their assignment plans. When expected_slot_assets is given,
    each plan must also hold the assets that side declared.
    """
    left, right = _load(Path(left_path)), _load(Path(right_path))
    left_identities = plan_slot_identities(left)
    right_identities = plan_slot_identities(right)
    if sorted(left_identities) != sorted(right_identities):
        raise BindingNativeError(
            f"visual plans expose different source slots: {sorted(left_identities)} "
            f"vs {sorted(right_identities)}"
        )
    for label, plan, identities in (("left", left, left_identities),
                                    ("right", right, right_identities)):
        if expected_slot_assets is None:
            continue
        declared = [str(value) for value in expected_slot_assets.get(label, ())]
        held = [str(identities[actor_id].get("asset_id"))
                for actor_id in sorted(identities)
                if identities[actor_id].get("asset_id") is not None]
        if declared and sorted(held) != sorted(declared):
            raise BindingNativeError(
                f"the {label} plan holds {sorted(held)} but its group declared "
                f"{sorted(declared)}"
            )
    canonical_left = _controlled_visual_signature(left)
    canonical_right = _controlled_visual_signature(right)
    if canonical_left != canonical_right:
        differing = sorted(key for key in canonical_left
                           if canonical_left[key] != canonical_right[key])
        raise BindingNativeError(
            "visual variants did not plan one controlled world; differing fields "
            f"beyond the declared slot identities: {differing}"
        )
    return {
        "status": "pass",
        "authority": ("identical_planned_world_under_declared_slot_identities: camera "
                      "route, clock, every actor transform/action/movement flag and "
                      "query identity compared with each bound identity rewritten to "
                      "its slot; shared audio columns are checked from assignment "
                      "plans"),
        "left": str(Path(left_path).resolve()), "right": str(Path(right_path).resolve()),
        "slot_identities": {"left": left_identities, "right": right_identities},
    }


def room_family_from_plan(plan: Mapping[str, Any]) -> str:
    """Read the family from validated RoomPackage metadata only."""
    resources = plan.get("resources") if isinstance(plan.get("resources"), Mapping) else {}
    candidates = [resources.get("room_package"), resources]
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        value = candidate.get("family") or candidate.get("room_family")
        if not isinstance(value, str) or not value.strip():
            continue
        family = value.strip().lower()
        if family not in ROOM_FAMILIES:
            raise BindingNativeError(f"unsupported validated room family: {family}")
        return family
    raise BindingNativeError(
        "validated RoomPackage metadata must declare family; room_id naming is not a family authority"
    )


def acoustic_identity(request: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    runtime = request.get("runtime") if isinstance(request.get("runtime"), Mapping) else {}
    resources = plan.get("resources") if isinstance(plan.get("resources"), Mapping) else {}
    room_package = resources.get("room_package") if isinstance(resources.get("room_package"), Mapping) else {}
    package_value = (
        resources.get("acoustic_package")
        or room_package.get("acoustic_package")
        or request.get("acoustic_package")
    )
    package = _file(package_value, base=REPOSITORY, owner="acoustic package")
    hrtf = _file(runtime.get("hrtf"), base=REPOSITORY, owner="HRTF")
    manifest = _load(package)
    room_package = resources.get("room_package") if isinstance(resources.get("room_package"), Mapping) else {}
    provenance = room_package.get("provenance") if isinstance(room_package.get("provenance"), Mapping) else {}
    source_context_policy = request.get("source_context_policy", "joint")
    if source_context_policy not in {"joint", "independent_states"}:
        raise BindingNativeError(
            "source_context_policy must be joint or independent_states"
        )
    return {
        "room_family": room_family_from_plan(plan),
        "source_context_policy": source_context_policy,
        "acoustic_package": {
            "path": str(package),
            "package_id": manifest.get("package_id"),
            "schema": manifest.get("schema"),
            "registered_source_revision": provenance.get("source_revision"),
        },
        "hrtf": {"path": str(hrtf), "id": runtime.get("hrtf_id") or hrtf.name},
        "rir_stride": int(request.get("rir_stride", 3)),
        "post_assembly_convolution_gain": request.get("post_assembly_convolution_gain"),
        "diffraction": request.get("diffraction"),
        "max_diffraction_order": request.get("max_diffraction_order"),
        "audio_view": _audio_view_fields(request),
        "clock": deepcopy(plan.get("clock")),
    }


def _plan_actor_assets(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for values in (
        plan.get("actors"),
        (plan.get("visual_plan") or {}).get("actors")
        if isinstance(plan.get("visual_plan"), Mapping) else None,
    ):
        if not isinstance(values, list):
            continue
        for actor in values:
            if isinstance(actor, Mapping) and isinstance(actor.get("actor_id"), str):
                result.setdefault(str(actor["actor_id"]), actor)
    return result


def _check_sound_target_compatibility(
    plan: Mapping[str, Any],
    event: Mapping[str, Any],
    binding: Mapping[str, Any],
    target_actor_id: str,
) -> dict[str, Any]:
    actors = _plan_actor_assets(plan)
    actor = actors.get(target_actor_id)
    if not isinstance(actor, Mapping):
        raise BindingNativeError(f"audio target actor is absent from visual plan: {target_actor_id}")
    target_asset = actor.get("asset_id")
    if not isinstance(target_asset, str) or not target_asset:
        raise BindingNativeError(f"audio target actor lacks a registered asset: {target_actor_id}")
    compatible_assets = event.get("compatible_asset_ids")
    if not isinstance(compatible_assets, list):
        compatible_assets = binding.get("compatible_asset_ids")
    compatible_assets = [
        str(value) for value in (compatible_assets or ())
        if isinstance(value, str) and value
    ]
    compatible_categories = event.get("compatible_object_categories")
    if not isinstance(compatible_categories, list):
        compatible_categories = binding.get("compatible_object_categories")
    compatible_categories = [
        str(value) for value in (compatible_categories or ())
        if isinstance(value, str) and value
    ]
    identity = actor.get("identity") if isinstance(actor.get("identity"), Mapping) else {}
    target_category = (
        identity.get("category")
        or actor.get("object_category")
        or actor.get("entity_kind")
        or actor.get("entity_class")
    )
    actor_for_match = {
        "entity_class": actor.get("entity_class"),
        "asset_id": target_asset,
        "identity": deepcopy(dict(identity)),
        "realized_attributes": deepcopy(
            actor.get("realized_attributes")
            if isinstance(actor.get("realized_attributes"), Mapping)
            else {}
        ),
    }
    try:
        from avengine.rooms.conditioned_sampler import sound_matches
        biology_and_class_ok = bool(sound_matches(actor_for_match, binding))
    except (ImportError, KeyError, TypeError, ValueError):
        biology_and_class_ok = False
    asset_ok = not compatible_assets or target_asset in compatible_assets
    category_ok = (
        not compatible_categories
        or target_category in compatible_categories
        or target_asset in compatible_assets
    )
    if not biology_and_class_ok or not asset_ok or not category_ok:
        raise BindingNativeError(
            f"sound {event.get('sound_asset_id')} is incompatible with target "
            f"{target_actor_id} asset/category ({target_asset!r}, {target_category!r})"
        )
    return {
        "event_id": str(event.get("event_id") or ""),
        "sound_asset_id": str(event.get("sound_asset_id") or ""),
        "target_actor_id": target_actor_id,
        "target_asset_id": target_asset,
        "target_object_category": target_category,
        "sound_matches": biology_and_class_ok,
        "compatible_asset_match": target_asset in compatible_assets if compatible_assets else None,
        "compatible_object_category_match": (
            target_category in compatible_categories if compatible_categories else None
        ),
    }


def build_audio_assignment_plan(
    visual_plan: Mapping[str, Any],
    request: Mapping[str, Any],
    assignment: str,
    *,
    assignment_targets: Mapping[str, Sequence[str]] | None = None,
    expected_event_count: int | None = None,
    endpoint_by_actor: Mapping[str, str] | None = None,
    require_authoritative_endpoints: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebind retained events to generic source slots.

    Assignment targets provide the target actor order for events sorted by
    their actual start samples. The visible-binding recipe retains its
    two-slot behavior; relation recipes may provide a three-slot permutation.
    """
    if assignment not in {"a0", "a1"}:
        raise BindingNativeError(f"unsupported audio assignment {assignment}")
    plan = deepcopy(dict(visual_plan))
    events = sorted(
        [deepcopy(row) for row in plan.get("audio_events", []) if isinstance(row, Mapping)],
        key=lambda row: (int(row.get("start_sample", 0)), str(row.get("event_id", ""))),
    )
    targets_by_assignment = assignment_targets or {
        "a0": ("source1", "source2"),
        "a1": ("source2", "source1"),
    }
    targets_value = targets_by_assignment.get(assignment)
    event_id_values = [str(event.get("event_id") or "") for event in events]
    if any(not event_id for event_id in event_id_values):
        raise BindingNativeError("audio events must declare non-empty event_id values")
    if len(set(event_id_values)) != len(event_id_values):
        raise BindingNativeError("audio events must have unique event_id values")
    event_ids = set(event_id_values)
    if isinstance(targets_value, Mapping):
        target_by_event = {
            str(event_id): str(target)
            for event_id, target in targets_value.items()
            if isinstance(event_id, str) and event_id
            and isinstance(target, str) and target.strip()
        }
        if set(target_by_event) != event_ids:
            raise BindingNativeError("audio assignment must target every event exactly once")
        # Multiple events may intentionally target one physical source. Keep
        # event IDs unique while resolving one endpoint per target actor.
    elif (
        isinstance(targets_value, Sequence)
        and not isinstance(targets_value, (str, bytes))
        and len(targets_value) == len(events)
        and all(isinstance(value, str) and value.strip() for value in targets_value)
    ):
        target_by_event = {
            str(event["event_id"]): str(target)
            for event, target in zip(events, targets_value, strict=True)
        }
    else:
        raise BindingNativeError("audio assignment target count does not match event count")
    if expected_event_count is not None and len(events) != expected_event_count:
        raise BindingNativeError(
            f"audio assignment requires exactly {expected_event_count} sound events"
        )
    raw_bindings = plan.get("voice_bindings")
    if not isinstance(raw_bindings, list):
        raise BindingNativeError("voice_bindings must be a list")
    bindings_by_sound: dict[str, list[dict[str, Any]]] = {}
    for row in raw_bindings:
        if not isinstance(row, Mapping) or not row.get("sound_asset_id"):
            raise BindingNativeError("voice_bindings must declare sound_asset_id")
        sound_id = str(row["sound_asset_id"])
        bindings_by_sound.setdefault(sound_id, []).append(deepcopy(dict(row)))
    # A sound binding is reusable across multiple events. Reject only a
    # conflicting sound-level declaration for the same sound ID. Event-level
    # fields (event_id, actor, endpoint, timing, gain) may differ per use.
    for sound_id, rows in bindings_by_sound.items():
        for key in ("path", "prepared", "prepared_audio_id", "sample_count"):
            values = {row[key] for row in rows if key in row}
            if len(values) > 1:
                raise BindingNativeError(
                    f"voice_bindings duplicate sound_asset_id has conflicting {key}: {sound_id}"
                )
    targets = tuple(target_by_event[str(event["event_id"])] for event in events)
    selected = {target: [] for target in set(targets)}
    declared_endpoints: dict[str, str] = {
        str(key): str(value)
        for key, value in (endpoint_by_actor or {}).items()
        if isinstance(key, str) and key and isinstance(value, str) and value
    }
    for values in (plan.get("actors"), (plan.get("visual_plan") or {}).get("actors") if isinstance(plan.get("visual_plan"), Mapping) else None):
        if isinstance(values, list):
            for actor in values:
                if not isinstance(actor, Mapping) or not actor.get("actor_id"):
                    continue
                actor_id = str(actor["actor_id"])
                endpoint = actor.get("source_endpoint_id")
                if not isinstance(endpoint, str) or not endpoint:
                    binding = actor.get("emitter_binding")
                    endpoint = binding.get("source_endpoint_id") if isinstance(binding, Mapping) else None
                if isinstance(endpoint, str) and endpoint:
                    declared_endpoints.setdefault(actor_id, endpoint)
    new_events, new_bindings = [], []
    for event in events:
        target = target_by_event[str(event["event_id"])]
        sound_id = str(event.get("sound_asset_id") or "")
        candidates = bindings_by_sound.get(sound_id)
        if not candidates:
            raise BindingNativeError(f"event sound binding is absent: {sound_id}")
        event_id = str(event["event_id"])
        # Prefer an explicitly event-matched declaration, then an
        # actor-matched legacy declaration, and finally the deterministic
        # first sound-level template. This keeps same-PCM repeated events
        # legal while preserving any event-specific metadata when present.
        exact = [row for row in candidates if str(row.get("event_id") or "") == event_id]
        if exact:
            binding = deepcopy(exact[0])
        else:
            actor_matches = [
                row for row in candidates
                if str(row.get("actor_id") or "") == str(event.get("actor_id") or "")
            ]
            binding = deepcopy(actor_matches[0] if actor_matches else candidates[0])
        endpoint = declared_endpoints.get(target)
        if require_authoritative_endpoints and not endpoint:
            raise BindingNativeError(
                f"audio target {target} lacks an authoritative native source endpoint"
            )
        endpoint = endpoint or f"{target}_mouth"
        compatibility = _check_sound_target_compatibility(plan, event, binding, target)
        event.update(
            actor_id=target,
            source_endpoint_id=endpoint,
            voice_binding_actor_id=target,
            assignment_variant=assignment,
            target_sound_compatibility=compatibility,
        )
        selected[target].append(sound_id)
        binding.update(
            event_id=event_id,
            actor_id=target,
            source_endpoint_id=endpoint,
            target_sound_compatibility=compatibility,
        )
        new_events.append(event)
        new_bindings.append(binding)
    for key in ("actors",):
        values = plan.get(key)
        if isinstance(values, list):
            for actor in values:
                if isinstance(actor, Mapping) and actor.get("actor_id"):
                    actor["source_endpoint_id"] = declared_endpoints.get(str(actor["actor_id"]), f"{actor['actor_id']}_mouth")
    visual = plan.get("visual_plan")
    if isinstance(visual, Mapping) and isinstance(visual.get("actors"), list):
        for actor in visual["actors"]:
            if isinstance(actor, Mapping) and actor.get("actor_id"):
                actor["source_endpoint_id"] = declared_endpoints.get(str(actor["actor_id"]), f"{actor['actor_id']}_mouth")
    new_bindings.sort(key=lambda row: (str(row.get("actor_id") or ""), str(row.get("event_id") or "")))
    plan["audio_events"], plan["voice_bindings"] = new_events, new_bindings
    plan["audio_assignment_variant"] = assignment
    plan["audio_assignment_targets"] = deepcopy(dict(target_by_event))
    plan["audio_target_compatibility"] = [
        deepcopy(event.get("target_sound_compatibility"))
        for event in new_events
        if isinstance(event.get("target_sound_compatibility"), Mapping)
    ]
    rebound_request = deepcopy(dict(request))
    selection = dict(rebound_request.get("sound_selection") or {})
    selection["selected_sound_asset_ids_by_actor"] = selected
    rebound_request["sound_selection"] = selection
    rebound_request["audio_assignment_variant"] = assignment
    rebound_request["audio_assignment_targets"] = deepcopy(dict(target_by_event))
    plan["request"] = rebound_request
    return plan, rebound_request


def materialize_audio_variant(visual_capture: Mapping[str, Any], output: str | Path, plan: Mapping[str, Any], request: Mapping[str, Any], *, member_id: str) -> Path:
    """Create a variant root whose capture is linked read-only to a visual root."""
    target = Path(output).expanduser().resolve()
    if target.exists() or target.is_symlink():
        raise BindingNativeError(f"refusing existing variant root: {target}")
    target.mkdir(parents=True)
    (target / "plan").mkdir()
    capture = Path(visual_capture["capture"]).resolve()
    if not capture.is_dir():
        raise BindingNativeError(f"visual capture is unavailable: {capture}")
    (target / "capture").symlink_to(capture, target_is_directory=True)
    for name in (
        "room_package.json",
        "path_bindings.json",
        "room_layout.json",
        "navigation.npz",
        "habitat_room_manifest.json",
        "habitat_execution/m1_capture_request.json",
        "habitat_execution/case_manifest.json",
        "habitat_execution/research_receipt.json",
    ):
        source = capture.parent / "plan" / name
        destination = target / "plan" / name
        if source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    _write(target / "request.json", request)
    _write(target / "plan/episode_plan.json", plan)
    _write(target / "plan/voice_bindings.json", plan["voice_bindings"])
    _write(target / "plan/audio_events.json", plan["audio_events"])
    _write(target / "native_linkage.json", {
        "member_id": member_id, "capture_source": str(capture),
        "plan_source": str((capture.parent / "plan/episode_plan.json").resolve()),
        "native_capture_reused": True,
        "claim_boundary": "capture is reused only after native readback equivalence; audio finalization is separate",
    })
    return target


def _install_float_wav_info_compat() -> None:
    """Provide only the info() surface needed by episode export when absent.

    The installed Habitat Python prefix used for native RLR does not carry
    python-soundfile, while the RLR writer emits a valid IEEE FLOAT WAV. Keep
    the actual float file untouched and provide a stdlib header reader for the
    same-process export check.
    """
    try:
        import soundfile  # noqa: F401
        return
    except ImportError:
        pass
    import struct
    import types

    def info(path: str | Path) -> Any:
        value = Path(path).expanduser().resolve()
        with value.open("rb") as stream:
            if stream.read(4) != b"RIFF":
                raise OSError(f"not a RIFF WAV: {value}")
            stream.seek(8)
            if stream.read(4) != b"WAVE":
                raise OSError(f"not a WAVE file: {value}")
            fmt = None
            data_size = None
            while True:
                header = stream.read(8)
                if len(header) != 8:
                    break
                chunk, size = struct.unpack("<4sI", header)
                payload = stream.read(size)
                if size & 1:
                    stream.read(1)
                if chunk == b"fmt " and len(payload) >= 16:
                    fmt = struct.unpack("<HHIIHH", payload[:16])
                elif chunk == b"data":
                    data_size = int(size)
                if fmt is not None and data_size is not None:
                    break
        if fmt is None or data_size is None:
            raise OSError(f"WAV lacks fmt/data chunks: {value}")
        audio_format, channels, sample_rate, _, block_align, bits = fmt
        if channels < 1 or sample_rate < 1 or block_align < 1:
            raise OSError(f"WAV header is invalid: {value}")
        frames = data_size // block_align
        subtype = "FLOAT" if audio_format == 3 and bits == 32 else f"PCM_{bits}"
        return types.SimpleNamespace(
            channels=int(channels),
            samplerate=int(sample_rate),
            frames=int(frames),
            subtype=subtype,
        )

    module = types.ModuleType("soundfile")
    module.info = info
    sys.modules.setdefault("soundfile", module)


AUDIO_ROOT_REQUIRED_FILES = (
    "request.json", "plan/episode_plan.json", "plan/voice_bindings.json",
    "plan/audio_events.json", "native_linkage.json",
)


def verify_materialized_audio_root(
    root: str | Path,
    *,
    request: Mapping[str, Any] | None = None,
    plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Check that an audio root really carries the plan it is asked to render.

    An audio worker consumes a root that materialize_audio_variant already
    produced: the request, the episode plan, the voice bindings, the audio
    events and the read-only link to the visual capture. Creating the directory
    is not enough, so this refuses a bare or half-written root at the boundary
    and names the missing member instead of failing deep inside delivery.
    """
    target = Path(root).expanduser().resolve()
    if target.is_symlink() or not target.is_dir():
        raise BindingNativeError(
            f"audio root is not a materialized directory: {target}; "
            "call materialize_audio_variant before finalizing an audio assignment"
        )
    missing = [name for name in AUDIO_ROOT_REQUIRED_FILES if not (target / name).is_file()]
    if missing:
        raise BindingNativeError(
            f"audio root {target} is not materialized: missing {missing}; "
            "call materialize_audio_variant so the root carries its plan and request"
        )
    capture = target / "capture"
    if not capture.is_dir():
        raise BindingNativeError(
            f"audio root {target} has no readable visual capture at {capture}"
        )
    saved_request = _load(target / "request.json")
    saved_plan = _load(target / "plan/episode_plan.json")
    if request is not None and dict(request) != saved_request:
        raise BindingNativeError(
            f"audio root {target} carries request {saved_request.get('episode_id')!r}, "
            f"which differs from the request this work item renders "
            f"({dict(request).get('episode_id')!r})"
        )
    if plan is not None and dict(plan) != saved_plan:
        raise BindingNativeError(
            f"audio root {target} carries an episode plan that differs from the "
            "assignment plan this work item renders"
        )
    saved_events = json.loads(
        (target / "plan/audio_events.json").read_text(encoding="utf-8")
    )
    plan_events = saved_plan.get("audio_events")
    if saved_events != plan_events:
        raise BindingNativeError(
            f"audio root {target} audio_events.json disagrees with its episode plan"
        )
    linkage = _load(target / "native_linkage.json")
    return {
        "status": "pass",
        "root": str(target),
        "member_id": linkage.get("member_id"),
        "episode_id": saved_request.get("episode_id"),
        "capture_source": linkage.get("capture_source"),
        "capture_is_symlink": (target / "capture").is_symlink(),
        "event_count": len(plan_events) if isinstance(plan_events, list) else None,
        "checked": list(AUDIO_ROOT_REQUIRED_FILES) + ["capture"],
    }


def declared_audio_delivery(request: Mapping[str, Any]) -> dict[str, Any]:
    """The layouts, ambisonic normalization and source-context policy requested.

    These are the P15 vocabulary: audio_layouts names one primary layout
    plus any attached views, foa_normalization selects native N3D or SN3D,
    and source_context_policy selects joint or independent native contexts.
    Nothing here renders; it reads what the configuration actually asked for so
    the delivered audio can be compared against it.
    """
    layouts = request.get("audio_layouts")
    rows: list[dict[str, Any]] = []
    if layouts is None:
        rows.append({"type": "binaural", "channel_count": 2, "role": "primary",
                     "source": "default_when_undeclared"})
    else:
        if isinstance(layouts, (str, bytes)) or not isinstance(layouts, Sequence):
            raise BindingNativeError("audio_layouts must be a list of layout objects")
        for index, item in enumerate(layouts):
            if not isinstance(item, Mapping):
                raise BindingNativeError(f"audio_layouts[{index}] must be an object")
            layout_type = item.get("type", item.get("layout_type"))
            if layout_type not in {"mono", "binaural", "ambisonics"}:
                raise BindingNativeError(
                    f"audio_layouts[{index}].type must be mono, binaural or ambisonics"
                )
            rows.append({"type": str(layout_type),
                         "channel_count": item.get("channel_count"),
                         "role": str(item.get("role", "primary")),
                         "ambisonic_order": item.get("ambisonic_order"),
                         "indirect_sh_order": item.get("indirect_sh_order"),
                         "source": "request"})
    primary = [row for row in rows if row["role"] == "primary"]
    if len(primary) != 1:
        raise BindingNativeError("audio_layouts must declare exactly one primary layout")
    normalization = request.get("foa_normalization", "native_n3d")
    if normalization not in {"native_n3d", "sn3d"}:
        raise BindingNativeError("foa_normalization must be native_n3d or sn3d")
    policy = request.get("source_context_policy") or "joint"
    if policy not in {"joint", "independent_states"}:
        raise BindingNativeError("source_context_policy must be joint or independent_states")
    return {
        "layouts": rows,
        "primary_layout": primary[0]["type"],
        "attached_view_layouts": [row["type"] for row in rows if row["role"] == "attached_view"],
        "foa_normalization": normalization,
        "source_context_policy": policy,
    }


def verify_delivered_audio_layouts(
    declared: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare the requested layouts against the files delivery actually made.

    The canonical two-channel mix is the primary layout. An attached view such
    as first-order ambisonics only counts when delivery published a real
    ancillary file for it; a declared attached view with no delivered file is
    reported as a blocker rather than quietly dropped.
    """
    ancillary = result.get("ancillary_audio_outputs")
    delivered = []
    if isinstance(ancillary, Sequence) and not isinstance(ancillary, (str, bytes)):
        for row in ancillary:
            if isinstance(row, Mapping) and isinstance(row.get("path"), str):
                delivered.append({"role": str(row.get("role") or ""), "path": row["path"]})
    missing = []
    for layout in declared.get("attached_view_layouts", ()):
        names = {"ambisonics": ("ambisonic", "foa", "ambisonics")}.get(layout, (layout,))
        if not any(any(name in row["role"].lower() or name in Path(row["path"]).name.lower()
                       for name in names)
                   for row in delivered):
            missing.append(layout)
    return {
        "status": "pass" if not missing else "blocked",
        "declared_primary_layout": declared.get("primary_layout"),
        "declared_attached_view_layouts": list(declared.get("attached_view_layouts", ())),
        "foa_normalization": declared.get("foa_normalization"),
        "source_context_policy": declared.get("source_context_policy"),
        "delivered_ancillary_outputs": delivered,
        "undelivered_attached_view_layouts": missing,
        "reason": None if not missing else (
            "delivery produced no file for declared attached-view layouts "
            f"{missing}; avengine.rooms.qa_delivery build_audio_command and the "
            "MP3D dynamic-audio command builder do not pass --layouts / "
            "--foa-normalization, so only the primary layout can be rendered"
        ),
    }


def finalize_audio_assignment(
    root: str | Path, request: Mapping[str, Any], *,
    audio_report: str | Path | None = None,
    shared_visual_root: str | Path | None = None,
    verify_materialized: bool = True,
) -> dict[str, Any]:
    """Finalize one actual audio assignment through qa_delivery.

    shared_visual_root pins the group-scoped shared visual evidence pack so
    two members that reuse one capture compute the appearance review, occluders
    and video master once. verify_materialized checks at the boundary that
    the root really carries the plan and request it is about to render.
    """
    from avengine.rooms.qa_delivery import finalize_qa_episode
    target = Path(root).expanduser().resolve()
    materialized = (
        verify_materialized_audio_root(target, request=request)
        if verify_materialized else None
    )
    declared_delivery = declared_audio_delivery(request)
    started = time.monotonic()
    previous_pythonpath = os.environ.get("PYTHONPATH")
    previous_no_bytecode = os.environ.get("PYTHONDONTWRITEBYTECODE")
    _install_float_wav_info_compat()
    # qa_delivery launches the audio CLI through subprocess.run; carry the
    # authority repository path into that child so it cannot resolve an older
    # installed avengine.cli from the host environment.
    os.environ["PYTHONPATH"] = _env(REPOSITORY)["PYTHONPATH"]
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        finalize_kwargs: dict[str, Any] = {}
        if shared_visual_root is not None:
            finalize_kwargs["shared_visual_root"] = Path(
                shared_visual_root
            ).expanduser().resolve()
        result = finalize_qa_episode(
            target, target / "delivery", repository=REPOSITORY, request=request,
            audio_report=Path(audio_report).resolve() if audio_report is not None else None,
            **finalize_kwargs,
        )
    finally:
        if previous_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = previous_pythonpath
        if previous_no_bytecode is None:
            os.environ.pop("PYTHONDONTWRITEBYTECODE", None)
        else:
            os.environ["PYTHONDONTWRITEBYTECODE"] = previous_no_bytecode
    delivery = target / "delivery"
    return {
        "result": result, "elapsed_s": time.monotonic() - started,
        "materialized_root": materialized,
        "declared_audio_delivery": declared_delivery,
        "delivered_audio_layouts": verify_delivered_audio_layouts(declared_delivery, result),
        "shared_visual_root": (
            str(Path(shared_visual_root).expanduser().resolve())
            if shared_visual_root is not None else None
        ),
        "facts": str(Path(result["facts"]).resolve()),
        "questions": str(Path(result["questions_path"]).resolve()),
        "audio": str(Path(result["lossless_stereo_wav"]).resolve()),
        "audio_report": str((delivery / "research_report.json").resolve()),
        "visual_video": (
            str(Path(result["visual_video"]).resolve())
            if result.get("visual_video") else None
        ),
        "preview": str(Path(result["preview"]).resolve()) if result.get("preview") else None,
    }




def recover_rendered_audio_attempt(
    item: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    output_root: str | Path,
    previous_attempt_root: str | Path,
    lineage: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]] = (),
    lease: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recover one interrupted rendered audio attempt without launching RLR.

    The old attempt is read-only. A fresh episode copy is finalized through the
    existing CPU delivery contract using its raw native audio receipt and
    rendered WAVs. The returned StageResult carries the old lineage plus the
    complete primary/ancillary delivery declaration for runner/export.
    """
    del results, lease
    if not isinstance(lineage, Mapping):
        raise BindingNativeError("audio recovery lineage must be an object")
    scope_id = str(item.get("scope_id") or item.get("request_id") or "")
    if lineage.get("scope_id") is not None and str(lineage["scope_id"]) != scope_id:
        raise BindingNativeError(
            f"audio recovery scope mismatch: {lineage.get('scope_id')!r} != {scope_id!r}"
        )
    previous = Path(previous_attempt_root).expanduser().resolve()
    source_episode = previous / "episode" if (previous / "episode").is_dir() else previous
    if not source_episode.is_dir():
        raise BindingNativeError(f"previous audio attempt has no episode root: {previous}")
    materialized = verify_materialized_audio_root(source_episode)
    saved_request = _load(source_episode / "request.json")
    saved_plan = _load(source_episode / "plan/episode_plan.json")
    capture = (source_episode / "capture").resolve()
    neutral = capture / "neutral_readback.json"
    if not neutral.is_file():
        raise BindingNativeError(
            f"previous audio attempt lacks its actual capture neutral readback: {neutral}"
        )
    canonical_world = lineage.get("world_id") or context.get("world_id")
    if lineage.get("world_id") is not None and context.get("world_id") is not None:
        if str(lineage["world_id"]) != str(context["world_id"]):
            raise BindingNativeError(
                f"audio recovery world mismatch: {lineage['world_id']!r} != "
                f"{context['world_id']!r}"
            )
    previous_world = saved_request.get("world_id") or saved_plan.get("world_id")
    if previous_world is not None and canonical_world is not None and str(previous_world) != str(canonical_world):
        raise BindingNativeError(
            f"previous audio attempt was produced for world {previous_world!r}, "
            f"expected {canonical_world!r}"
        )
    contract = context.get("contract")
    world_check = {"status": "not_run", "reason": "no group contract supplied"}
    if isinstance(contract, Mapping) and contract.get("shared_world") is not None:
        world_check = verify_retained_visual_request(
            saved_request, contract, label=str(item.get("unit_id") or scope_id)
        )
    raw_report_candidates = (
        source_episode / "delivery/audio/research_receipt.json",
        source_episode / "delivery/audio/research_report.json",
        source_episode / "delivery/research_report.json",
    )
    raw_report = next((path for path in raw_report_candidates if path.is_file()), None)
    if raw_report is None:
        raise BindingNativeError(
            f"previous audio attempt has no raw native audio receipt/report: {source_episode}"
        )
    raw_facts = source_episode / "delivery/facts.json"
    raw_audio = None
    if raw_facts.is_file():
        raw_value = _load(raw_facts).get("audio")
        if isinstance(raw_value, Mapping) and isinstance(raw_value.get("path"), str):
            raw_audio = Path(raw_value["path"]).expanduser().resolve()
    if raw_audio is None:
        candidate = source_episode / "delivery/audio/audio/binaural/mixture.wav"
        raw_audio = candidate.resolve()
    if not raw_audio.is_file():
        raise BindingNativeError(f"previous audio attempt has no readable primary WAV: {raw_audio}")
    raw_program = source_episode / "plan/audio_events.json"
    if not raw_program.is_file():
        raise BindingNativeError(f"previous audio attempt has no readable audio program: {raw_program}")

    relative = item.get("fresh_output_relative")
    if not isinstance(relative, str) or not relative.strip():
        relative = f"{item.get('group_id', 'group')}/{item.get('unit_id', 'audio')}/audio/recovery_01"
    fresh_episode = Path(output_root).expanduser().resolve() / relative / "episode"
    if fresh_episode.exists() or fresh_episode.is_symlink():
        raise BindingNativeError(f"audio recovery output already exists: {fresh_episode}")
    fresh_episode.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source_episode,
        fresh_episode,
        symlinks=True,
        ignore=shutil.ignore_patterns("delivery"),
    )
    fresh_request = _load(fresh_episode / "request.json")
    finalized = finalize_audio_assignment(
        fresh_episode,
        fresh_request,
        audio_report=raw_report,
        shared_visual_root=Path(output_root).expanduser().resolve()
        / str(context.get("group_id") or item.get("group_id") or "group")
        / "shared_visual_evidence",
    )
    delivered = finalized["delivered_audio_layouts"]
    if delivered.get("status") != "pass":
        raise BindingNativeError(
            f"audio recovery delivery layouts are not valid: {delivered}"
        )
    fresh_facts_path = Path(finalized["facts"]).resolve()
    fresh_report_path = Path(finalized["audio_report"]).resolve()
    fresh_facts = _load(fresh_facts_path)
    fresh_report = _load(fresh_report_path)
    intervals = (fresh_facts.get("audio") or {}).get("wet_tail_intervals")
    if not isinstance(intervals, list) or not intervals:
        raise BindingNativeError("audio recovery produced no measured wet_tail_intervals")
    audio = fresh_facts.get("audio") if isinstance(fresh_facts.get("audio"), Mapping) else {}
    report_clock = fresh_report.get("clock") if isinstance(fresh_report.get("clock"), Mapping) else {}
    time_block = fresh_facts.get("time") if isinstance(fresh_facts.get("time"), Mapping) else {}
    for key in ("sample_rate_hz", "sample_count"):
        if time_block.get(key) is not None and report_clock.get(key) is not None:
            if int(time_block[key]) != int(report_clock[key]):
                raise BindingNativeError(f"audio recovery clock mismatch: {key}")
    if int(audio.get("sample_rate_hz", 0)) <= 0 or int(audio.get("sample_count", 0)) <= 0:
        raise BindingNativeError("audio recovery facts lack a valid primary audio clock")
    primary = Path(finalized["audio"]).resolve()
    if not primary.is_file():
        raise BindingNativeError(f"audio recovery finalized primary WAV is unavailable: {primary}")
    if not filecmp.cmp(primary, raw_audio, shallow=False):
        raise BindingNativeError("audio recovery primary PCM differs from the interrupted rendered WAV")
    ancillary = deepcopy((finalized.get("result") or {}).get("ancillary_audio_outputs"))
    for row in ancillary or ():
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise BindingNativeError("audio recovery ancillary declaration is incomplete")
        path = Path(row["path"]).expanduser().resolve()
        if not path.is_file():
            raise BindingNativeError(f"audio recovery ancillary media is unavailable: {path}")
        if str(row.get("layout_type") or "").lower() == "ambisonics":
            if row.get("channel_count") != 4 or row.get("channel_order") != "ACN":
                raise BindingNativeError("audio recovery FOA ancillary is not ACN 4-channel")
            if row.get("normalization") != "N3D":
                raise BindingNativeError("audio recovery FOA ancillary is not N3D")
    audio_delivery = {
        "audio_report_path": str(fresh_report_path),
        "assignment_request_path": str((fresh_episode / "request.json").resolve()),
        "primary_audio_path": str(primary),
        "declared_audio_delivery": deepcopy(finalized["declared_audio_delivery"]),
        "delivered_audio_layouts": deepcopy(delivered),
        "ancillary_audio_outputs": ancillary,
        "world_check": world_check,
        "capture": str(capture),
        "neutral_readback": str(neutral),
        "clock": {
            "facts": {key: audio.get(key) for key in ("sample_rate_hz", "sample_count", "channel_count")},
            "report": {key: report_clock.get(key) for key in ("sample_rate_hz", "sample_count", "frame_rate_hz", "frame_count")},
        },
        "wet_tail_intervals": deepcopy(intervals),
        "raw_native_audio_receipt": str(raw_report),
    }
    member_id = str(item.get("unit_id") or item.get("member_request_id") or scope_id)
    return _stage_result(
        item,
        status="pass",
        facts={
            "facts_path": str(fresh_facts_path),
            "audio_report_path": str(fresh_report_path),
            "wet_tail_intervals": deepcopy(intervals),
            "recovery": {
                "kind": "cpu_only_finalize_existing_materialized_audio",
                "previous_attempt_root": str(previous),
                "fresh_episode_root": str(fresh_episode.resolve()),
                "previous_stage_result_path": lineage.get("previous_stage_result_path"),
                "previous_work_item_id": lineage.get("previous_work_item_id"),
                "canonical_world_id": canonical_world,
                "native_acoustic_contexts_created": 0,
            },
            "audio_delivery_by_member": {member_id: deepcopy(audio_delivery)},
        },
        outputs={
            "variant_root": str(fresh_episode.resolve()),
            "audio": str(primary),
            "audio_report": str(fresh_report_path),
            "questions": finalized.get("questions"),
            "visual_video": finalized.get("visual_video"),
            "capture": str(capture),
            "neutral_readback": str(neutral),
            "episode_plan": str((fresh_episode / "plan/episode_plan.json").resolve()),
            "request_path": str((fresh_episode / "request.json").resolve()),
            "assignment_plan_path": str((fresh_episode / "plan/episode_plan.json").resolve()),
            "assignment_request_path": str((fresh_episode / "request.json").resolve()),
            "visual_unit_id": item.get("visual_unit_id"),
            "member_request_id": item.get("member_request_id"),
            "audio_delivery_by_member": {member_id: deepcopy(audio_delivery)},
            "ancillary_audio_outputs": ancillary,
            "delivered_audio_layouts": deepcopy(delivered),
            "native_visual_worlds_created": 0,
            "native_acoustic_contexts_created": 0,
            "cpu_recovery": True,
            "recovery_of_work_item_id": lineage.get("previous_work_item_id"),
            "canonical_world_id": canonical_world,
            "lineage": deepcopy(dict(lineage)),
        },
    )


def _parallel_audio_worker(task_path: str | Path, result_path: str | Path) -> int:
    """Finalize one audio root from a JSON task in a child process."""
    task_file = Path(task_path).expanduser().resolve()
    result_file = Path(result_path).expanduser().resolve()
    task: Mapping[str, Any] = {}
    try:
        task = _load(task_file)
        member_id = task.get("member_id")
        root = task.get("root")
        request = task.get("request")
        audio_report = task.get("audio_report")
        shared_visual_root = task.get("shared_visual_root")
        if (
            not isinstance(member_id, str)
            or not member_id.strip()
            or not isinstance(root, str)
            or not root.strip()
            or not isinstance(request, Mapping)
            or (audio_report is not None and not isinstance(audio_report, str))
            or (shared_visual_root is not None and not isinstance(shared_visual_root, str))
        ):
            raise BindingNativeError("parallel audio worker task has invalid root/request")
        result = finalize_audio_assignment(
            Path(root),
            request,
            audio_report=Path(audio_report) if audio_report else None,
            shared_visual_root=Path(shared_visual_root) if shared_visual_root else None,
        )
        _write(
            result_file,
            {
                "schema": "avengine_native_parallel_audio_result_v1",
                "status": "pass",
                "member_id": member_id,
                "result": result,
            },
        )
        return 0
    except Exception as exc:
        try:
            _write(
                result_file,
                {
                    "schema": "avengine_native_parallel_audio_result_v1",
                    "status": "fail",
                    "member_id": task.get("member_id"),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        except Exception:
            pass
        return 1


def finalize_audio_assignments(
    tasks: Mapping[str, Mapping[str, Any]],
    *,
    max_workers: int = 2,
    shared_visual_root: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Finalize independent audio roots in bounded child processes.

    Each task is keyed by a member ID and contains ``root`` and ``request``;
    ``audio_report`` and ``shared_visual_root`` are optional. Every root must
    already be materialized by ``materialize_audio_variant`` and carry the very
    request this task renders -- an empty directory is refused here, with the
    missing member named, rather than failing inside delivery. The parent
    serializes that input into a
    private task directory under the already materialized root, launches the
    worker through the authority Python environment, and accepts a result only
    when both the process exit code and result JSON pass. At most
    ``max_workers`` children run at once, and no worker shares the parent's
    environment, GIL, or random state.
    """
    if not isinstance(tasks, Mapping) or not tasks:
        raise BindingNativeError("parallel audio tasks must be a nonempty mapping")
    if (
        isinstance(max_workers, bool)
        or not isinstance(max_workers, int)
        or max_workers < 1
        or max_workers > 4
    ):
        raise BindingNativeError("parallel audio max_workers must be an integer in [1, 4]")
    normalised: dict[str, dict[str, Any]] = {}
    materialized: dict[str, dict[str, Any]] = {}
    group_visual_root = (
        str(Path(shared_visual_root).expanduser().resolve())
        if shared_visual_root is not None else None
    )
    for member_id, task in tasks.items():
        if not isinstance(member_id, str) or not member_id.strip():
            raise BindingNativeError("parallel audio member IDs must be nonempty strings")
        if not isinstance(task, Mapping):
            raise BindingNativeError(f"parallel audio task must be a mapping: {member_id}")
        root_value = task.get("root")
        if isinstance(root_value, Path):
            root_value = str(root_value)
        if not isinstance(root_value, str) or not root_value.strip():
            raise BindingNativeError(f"parallel audio task lacks root: {member_id}")
        root = Path(root_value).expanduser().resolve()
        request = task.get("request")
        if isinstance(request, (str, Path)):
            request_path = Path(request).expanduser().resolve()
            if not request_path.is_file():
                raise BindingNativeError(f"parallel audio request is unavailable: {request_path}")
            request = _load(request_path)
        if not isinstance(request, Mapping):
            raise BindingNativeError(f"parallel audio task lacks request: {member_id}")
        try:
            materialized[member_id] = verify_materialized_audio_root(root, request=request)
        except BindingNativeError as exc:
            raise BindingNativeError(
                f"parallel audio task {member_id} was handed an unusable root: {exc}"
            ) from exc
        audio_report = task.get("audio_report")
        if audio_report is not None:
            if isinstance(audio_report, Path):
                audio_report = str(audio_report)
            if not isinstance(audio_report, str) or not audio_report.strip():
                raise BindingNativeError(f"parallel audio task has invalid audio_report: {member_id}")
            audio_report = str(Path(audio_report).expanduser().resolve())
        private = root / "parallel_worker"
        if private.exists() or private.is_symlink():
            raise BindingNativeError(f"parallel audio worker root already exists: {private}")
        private.mkdir()
        task_path = private / "task.json"
        result_path = private / "result.json"
        payload = {
            "schema": "avengine_native_parallel_audio_task_v1",
            "member_id": member_id,
            "root": str(root),
            "request": deepcopy(dict(request)),
            "audio_report": audio_report,
            "shared_visual_root": (
                str(Path(task["shared_visual_root"]).expanduser().resolve())
                if task.get("shared_visual_root") else group_visual_root
            ),
        }
        _write(task_path, payload)
        normalised[member_id] = {
            "root": root,
            "task_path": task_path,
            "result_path": result_path,
            "log_path": private / "worker.log",
        }

    pending = list(normalised)
    active: dict[str, tuple[Any, Any]] = {}
    completed: dict[str, dict[str, Any]] = {}

    def launch(member_id: str) -> None:
        item = normalised[member_id]
        log = item["log_path"].open("x", encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "avengine.dataset.binding_group_native",
            "--parallel-audio-worker",
            str(item["task_path"]),
            str(item["result_path"]),
        ]
        try:
            process = subprocess.Popen(
                command,
                cwd=REPOSITORY,
                env=_env(REPOSITORY),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception:
            log.close()
            raise
        active[member_id] = (process, log)

    def stop_active() -> None:
        for process, _log in active.values():
            if process.poll() is None:
                process.terminate()
        deadline = time.monotonic() + 10.0
        for process, log in active.values():
            remaining = max(0.0, deadline - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
            finally:
                log.close()

    try:
        while pending or active:
            while pending and len(active) < min(max_workers, len(normalised)):
                launch(pending.pop(0))
            observed = False
            for member_id, (process, log) in list(active.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                observed = True
                log.close()
                del active[member_id]
                result_path = normalised[member_id]["result_path"]
                if not result_path.is_file():
                    raise BindingNativeError(
                        f"parallel audio worker exited {returncode} without result: {member_id}"
                    )
                result_doc = _load(result_path)
                if returncode != 0 or result_doc.get("status") != "pass":
                    raise BindingNativeError(
                        f"parallel audio worker failed for {member_id} with returncode {returncode}: "
                        f"{result_doc.get('error', 'missing passing result')}"
                    )
                if (
                    result_doc.get("member_id") != member_id
                    or not isinstance(result_doc.get("result"), Mapping)
                ):
                    raise BindingNativeError(
                        f"parallel audio worker returned an invalid result for {member_id}"
                    )
                completed[member_id] = {
                    **dict(result_doc["result"]),
                    "materialized_root_precheck": materialized.get(member_id),
                }
            if active and not observed:
                time.sleep(0.2)
    except Exception:
        stop_active()
        raise
    return {member_id: completed[member_id] for member_id in normalised}


def schedule_relation_audio_plan(
    visual_plan: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    start_times_s: Sequence[float] | None = None,
    reserve_tail_s: float | None = None,
    query_window_s: Sequence[int] = (4, 6),
    rng_seed: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Apply an explicit three-source schedule to retained prepared clips.

    Source clips and their local prepared activity intervals remain untouched.
    Only the episode placement, corresponding ticks, and derived schedule
    metadata change. The effective tail is derived from the actual scheduled
    clip ends, so the original planner reserve is never reported as current.
    """
    plan = deepcopy(dict(visual_plan))
    request_value = deepcopy(dict(request))
    raw_events = plan.get("audio_events")
    if not isinstance(raw_events, list):
        raise BindingNativeError("relation schedule requires audio_events")
    events = sorted(
        [deepcopy(row) for row in raw_events if isinstance(row, Mapping)],
        key=lambda row: str(row.get("actor_id") or row.get("event_id") or ""),
    )
    if len(events) != 3:
        raise BindingNativeError("visual conditioned relation requires exactly three sound events")
    actor_ids = [str(row.get("actor_id") or "") for row in events]
    if any(not value for value in actor_ids) or len(set(actor_ids)) != len(actor_ids):
        raise BindingNativeError("relation schedule requires one event per source actor")
    if start_times_s is not None and (
        isinstance(start_times_s, (str, bytes))
        or len(start_times_s) != len(events)
    ):
        raise BindingNativeError("relation schedule requires one start time per source actor")
    clock = plan.get("clock")
    if not isinstance(clock, Mapping):
        raise BindingNativeError("relation schedule requires an episode clock")
    try:
        sample_rate = int(clock["sample_rate_hz"])
        sample_count = int(clock["sample_count"])
        time_base = int(clock["time_base_hz"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BindingNativeError("relation schedule clock is invalid") from exc
    if sample_rate <= 0 or sample_count <= 0 or time_base <= 0:
        raise BindingNativeError("relation schedule clock is invalid")
    starts_by_actor: dict[str, float] = {}
    selected_times = start_times_s
    if selected_times is None:
        if (
            isinstance(query_window_s, (str, bytes))
            or len(query_window_s) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in query_window_s)
            or int(query_window_s[1]) <= int(query_window_s[0])
        ):
            raise BindingNativeError("relation schedule query window must contain two increasing integer seconds")
        window_start_sample = int(query_window_s[0])
        window_end_sample = int(query_window_s[1])
        import random
        rng = random.Random(int(rng_seed if rng_seed is not None else plan.get("seed", 0)))
        profile_for_tail = (
            request_value.get("profile")
            if isinstance(request_value.get("profile"), Mapping) else {}
        )
        requested_tail = profile_for_tail.get("reserve_tail_s", 0.0)
        if isinstance(requested_tail, bool) or not isinstance(requested_tail, (int, float)):
            raise BindingNativeError("relation schedule reserve_tail_s must be finite")
        max_starts: dict[str, int] = {}
        for event in events:
            duration = event.get("sample_count")
            if isinstance(duration, bool) or not isinstance(duration, (int, float)):
                duration = int(round(float(event.get("source_crop_duration_s", 0.0)) * sample_rate))
            max_start = sample_count - int(round(float(requested_tail) * sample_rate)) - int(duration)
            max_starts[str(event["actor_id"])] = max_start
        if any(value < 0 for value in max_starts.values()):
            raise BindingNativeError("no legal relation schedule satisfies the declared reserve tail")
        def overlap_for(left: Mapping[str, Any], left_start_s: int, right: Mapping[str, Any], right_start_s: int) -> bool:
            left_intervals = left.get("source_activity_intervals_samples") or []
            right_intervals = right.get("source_activity_intervals_samples") or []
            lo, hi = int(query_window_s[0]) * sample_rate, int(query_window_s[1]) * sample_rate
            for a in left_intervals:
                if not isinstance(a, Sequence) or len(a) != 2:
                    continue
                la, lb = left_start_s + int(a[0]), left_start_s + int(a[1])
                for b in right_intervals:
                    if not isinstance(b, Sequence) or len(b) != 2:
                        continue
                    ra, rb = right_start_s + int(b[0]), right_start_s + int(b[1])
                    if max(lo, la, ra) < min(hi, lb, rb):
                        return True
            return False
        found = None
        for _ in range(20000):
            candidate = {
                actor_id: rng.randrange(max_starts[actor_id] + 1) / sample_rate
                for actor_id in sorted(actor_ids)
            }
            first, second, third = events
            if overlap_for(first, int(round(candidate[first["actor_id"]] * sample_rate)), second, int(round(candidate[second["actor_id"]] * sample_rate))) and not overlap_for(
                first, int(round(candidate[first["actor_id"]] * sample_rate)), third, int(round(candidate[third["actor_id"]] * sample_rate))
            ):
                found = candidate
                break
        if found is None:
            raise BindingNativeError(
                "no legal random relation schedule satisfies the requested query window and reserve tail"
            )
        selected_times = tuple(found[actor_id] for actor_id in sorted(actor_ids))
    for actor_id, value in zip(sorted(actor_ids), selected_times, strict=True):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise BindingNativeError("relation start times must be finite numbers")
        start = float(value)
        if not math.isfinite(start) or start < 0.0:
            raise BindingNativeError("relation start times must be finite and nonnegative")
        starts_by_actor[actor_id] = start
    scheduled: list[dict[str, Any]] = []
    selected_schedule: list[dict[str, Any]] = []
    latest_end = 0
    for event in events:
        actor_id = str(event["actor_id"])
        duration = event.get("sample_count")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            duration_s = event.get("source_crop_duration_s")
            if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)):
                raise BindingNativeError(f"event {event.get('event_id')} has no prepared duration")
            duration = int(round(float(duration_s) * sample_rate))
        duration = int(duration)
        if duration <= 0:
            raise BindingNativeError(f"event {event.get('event_id')} has no positive prepared duration")
        start = int(round(starts_by_actor[actor_id] * sample_rate))
        end = start + duration
        if end > sample_count:
            raise BindingNativeError(
                f"event {event.get('event_id')} exceeds the episode clock at its requested start"
            )
        start_tick = int(round(start * time_base / sample_rate))
        end_tick = int(round(end * time_base / sample_rate))
        local_audible_start = int(event.get("audible_start_sample", 0))
        local_audible_end = int(event.get("audible_end_sample_exclusive", duration))
        if local_audible_start < 0 or local_audible_end <= local_audible_start:
            raise BindingNativeError(f"event {event.get('event_id')} has invalid local activity bounds")
        if local_audible_end > duration:
            local_audible_end = duration
        updated = deepcopy(event)
        updated.update(
            start_sample=start,
            end_sample=end,
            end_sample_exclusive=end,
            start_tick=start_tick,
            end_tick=end_tick,
            end_tick_exclusive=end_tick,
            source_start_sample=int(event.get("source_start_sample") or 0),
            source_end_sample_exclusive=int(event.get("source_start_sample") or 0) + duration,
            planned_audible_interval_samples=[
                start + local_audible_start,
                start + local_audible_end,
            ],
            activity_coordinate="prepared_clip_samples",
            source_activity_schedule_status="scheduled_from_prepared_clip_metadata",
        )
        scheduled.append(updated)
        latest_end = max(latest_end, end)
        selected_schedule.append({
            "event_id": str(event.get("event_id") or ""),
            "actor_id": actor_id,
            "start_sample": start,
            "end_sample_exclusive": end,
            "start_tick": start_tick,
            "end_tick_exclusive": end_tick,
            "prepared_sample_count": duration,
            "prepared_duration_s": duration / sample_rate,
            "planned_audible_interval_samples": list(updated["planned_audible_interval_samples"]),
        })
    scheduled.sort(key=lambda row: (int(row["start_sample"]), str(row["event_id"])))
    available_tail = (sample_count - latest_end) / sample_rate
    if available_tail < 0.0:
        raise BindingNativeError("relation schedule has a negative available tail")
    original_profile = (
        request_value.get("profile")
        if isinstance(request_value.get("profile"), Mapping)
        else {}
    )
    original_tail = original_profile.get("reserve_tail_s")
    if reserve_tail_s is None:
        effective_tail = float(original_tail or 0.0)
        if available_tail + 1.0e-9 < effective_tail:
            raise BindingNativeError(
                f"scheduled clips leave only {available_tail:.6g}s, below declared reserve tail {effective_tail:.6g}s"
            )
    else:
        if isinstance(reserve_tail_s, bool) or not isinstance(reserve_tail_s, (int, float)):
            raise BindingNativeError("reserve_tail_s must be finite and nonnegative")
        effective_tail = float(reserve_tail_s)
        if not math.isfinite(effective_tail) or effective_tail < 0.0 or effective_tail > available_tail:
            raise BindingNativeError("reserve_tail_s exceeds the available scheduled tail")
    schedule = {
        "schema": "avengine_relation_audio_schedule_v1",
        "status": "explicit_prepared_clip_schedule",
        "event_relation": "scheduled_source_events",
        "start_times_s_by_actor": {
            actor_id: starts_by_actor[actor_id] for actor_id in sorted(starts_by_actor)
        },
        "events": selected_schedule,
        "source_profile_reserve_tail_s": original_tail,
        "effective_reserve_tail_s": effective_tail,
        "available_tail_s": available_tail,
        "schedule_clock": {
            "sample_rate_hz": sample_rate,
            "sample_count": sample_count,
            "time_base_hz": time_base,
        },
        "activity_authority": "prepared_clip_metadata_then_actual_native_audio_readback",
        "selection_mode": (
            "explicit" if start_times_s is not None else "seeded_random_legal"
        ),
    }
    profile = dict(original_profile)
    profile["reserve_tail_s"] = effective_tail
    request_value["profile"] = profile
    request_value["audio_schedule"] = deepcopy(schedule)
    manifests = {
        str(event.get("source_metadata_manifest")).strip()
        for event in events
        if isinstance(event.get("source_metadata_manifest"), str)
        and str(event.get("source_metadata_manifest")).strip()
    }
    if len(manifests) == 1 and Path(next(iter(manifests))).is_file():
        request_value["prepared_manifest"] = next(iter(manifests))
        plan["prepared_manifest"] = next(iter(manifests))
    plan["audio_events"] = scheduled
    plan_profile = dict(request_value.get("profile") or {})
    plan_profile["reserve_tail_s"] = effective_tail
    plan_request = deepcopy(request_value)
    plan_request["profile"] = plan_profile
    plan_request["audio_schedule"] = deepcopy(schedule)
    plan["request"] = plan_request
    planned_conditions = deepcopy(plan.get("planned_conditions") or {})
    if isinstance(planned_conditions, Mapping):
        planned_conditions = dict(planned_conditions)
    else:
        planned_conditions = {}
    if "legal_event_start_ranges_samples" in planned_conditions:
        planned_conditions.setdefault(
            "original_legal_event_start_ranges_samples",
            deepcopy(planned_conditions["legal_event_start_ranges_samples"]),
        )
    planned_conditions["selected_event_schedule"] = deepcopy(selected_schedule)
    planned_conditions["effective_reserve_tail_s"] = effective_tail
    plan["planned_conditions"] = planned_conditions
    activity_plan = deepcopy(plan.get("activity_plan") or {})
    if not isinstance(activity_plan, Mapping):
        activity_plan = {}
    activity_plan = dict(activity_plan)
    activity_plan["audio_schedule"] = deepcopy(schedule)
    plan["activity_plan"] = activity_plan
    return plan, request_value, schedule


_RELATION_ASSIGNMENT_TARGETS = {
    "a0": {
        "event_001": "source1",
        "event_002": "source2",
        "event_003": "source3",
    },
    "a1": {
        "event_001": "source1",
        "event_002": "source3",
        "event_003": "source2",
    },
}


def _neutral_endpoint_bindings(
    path: str | Path,
    *,
    plan: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Resolve endpoints from native identity or an explicit capture/plan map."""
    neutral_path = Path(path).expanduser().resolve()
    value = _load(neutral_path)
    identities = value.get("entity_identities")
    result: dict[str, str] = {}
    if isinstance(identities, Mapping):
        for actor_id, record in identities.items():
            if not isinstance(record, Mapping):
                continue
            endpoint = record.get("source_endpoint_id")
            if isinstance(endpoint, str) and endpoint.strip():
                result[str(actor_id)] = endpoint.strip()
                actor = record.get("actor_id")
                if isinstance(actor, str) and actor.strip():
                    result.setdefault(actor.strip(), endpoint.strip())
    if result:
        return result
    if not isinstance(plan, Mapping):
        raise BindingNativeError(
            "native neutral readback lacks entity source endpoint identities and explicit plan anchors"
        )
    # UE neutral readbacks carry the actual actor entity series but no
    # entity_identities table. Verify the producer's actual frame source first.
    producer = value.get("producer")
    source_readbacks = producer.get("source_readbacks") if isinstance(producer, Mapping) else None
    frame_source = None
    if isinstance(source_readbacks, list):
        for candidate in source_readbacks:
            if isinstance(candidate, str) and Path(candidate).expanduser().is_file():
                frame_source = Path(candidate).expanduser().resolve()
                break
    if frame_source is None:
        candidate = neutral_path.parent / "frame_readbacks.json"
        if candidate.is_file():
            frame_source = candidate.resolve()
    if frame_source is None:
        raise BindingNativeError(
            "native neutral readback lacks an actual frame readback source"
        )
    frame_value = _load(frame_source)
    emitters = frame_value.get("emitters")
    if not isinstance(emitters, Mapping):
        raise BindingNativeError(
            "actual frame readbacks lack an emitters mapping for endpoint resolution"
        )
    entities = value.get("entities")
    if not isinstance(entities, Mapping):
        raise BindingNativeError("native neutral readback lacks actual entity series")
    visual = plan.get("visual_plan")
    actors = visual.get("actors") if isinstance(visual, Mapping) else None
    if not isinstance(actors, list):
        raise BindingNativeError("endpoint resolution plan lacks visual actor declarations")
    for actor in actors:
        if not isinstance(actor, Mapping):
            continue
        actor_id = actor.get("actor_id")
        if not isinstance(actor_id, str) or not actor_id.strip():
            continue
        actor_id = actor_id.strip()
        if actor_id not in entities or actor_id not in emitters:
            raise BindingNativeError(
                f"actual capture lacks emitter readback for plan actor {actor_id}"
            )
        binding = actor.get("emitter_binding")
        if not isinstance(binding, Mapping):
            raise BindingNativeError(f"plan actor lacks explicit emitter binding: {actor_id}")
        endpoint = actor.get("source_endpoint_id") or binding.get("source_endpoint_id")
        if not isinstance(endpoint, str) or not endpoint.strip():
            slot = binding.get("source_slot_id") or actor_id
            anchor = binding.get("semantic_anchor_id")
            if not isinstance(slot, str) or not slot.strip() or not isinstance(anchor, str) or not anchor.strip():
                raise BindingNativeError(
                    f"plan actor lacks explicit source slot/semantic anchor: {actor_id}"
                )
            endpoint = f"{slot.strip()}_{anchor.strip()}"
        result[actor_id] = endpoint.strip()
    if not result:
        raise BindingNativeError("actual UE capture has no endpoint-bearing actors")
    return result


def _relation_group_spec(
    group_id: str,
    world_id: str,
    room_family: str,
    room_id: str,
    query: Mapping[str, Any],
    variants: Mapping[str, Mapping[str, Any]],
    *,
    request: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    members = []
    for member_id in ("v0_a0", "v0_a1", "v1_a0", "v1_a1"):
        visual_id, assignment = member_id.split("_")
        member = variants[member_id]
        visual_root = member.get("visual_capture_root")
        members.append({
            "member_id": member_id,
            "facts_path": str(Path(member["facts"]).resolve()),
            "video_path": str(Path(member["visual_video"]).resolve()),
            "audio_path": str(Path(member["audio"]).resolve()),
            "interventions": {
                "visual_variant": visual_id,
                "audio_assignment": assignment,
                "native_capture_root": str(Path(visual_root).resolve()) if visual_root else None,
            },
        })
    return {
        "schema": "avengine_binding_group_spec_v1",
        "request": deepcopy(dict(request)),
        "groups": [{
            "group_id": group_id,
            "world_id": world_id,
            "task_family": RELATION_TASK_FAMILY,
            "room_family": room_family,
            "room_id": room_id,
            "split": "pilot",
            "query": dict(query),
            "request": deepcopy(dict(request)),
            "profile": deepcopy(dict(profile)),
            "angle_tolerance_deg": 10,
            "members": members,
            "comparisons": [
                {"members": ["v0_a0", "v0_a1"], "shared_modality": "video", "answer_relation": "different", "kind": "necessity"},
                {"members": ["v1_a0", "v1_a1"], "shared_modality": "video", "answer_relation": "different", "kind": "necessity"},
                {"members": ["v0_a0", "v1_a0"], "shared_modality": "audio", "answer_relation": "different", "kind": "necessity"},
                {"members": ["v0_a1", "v1_a1"], "shared_modality": "audio", "answer_relation": "different", "kind": "necessity"},
            ],
        }],
    }


def prepare_visual_conditioned_relation_group(
    *,
    base_request_path: str | Path,
    first_visual_capture_root: str | Path,
    output_root: str | Path,
    second_visual_capture_root: str | Path | None = None,
    sound_pool: str | Path | None = None,
    source_asset_ids: Sequence[str] | None = None,
    room_id: str | None = None,
    rpc_port: int | None = None,
    graphics_adapter: int | None = None,
    group_id: str = "visual_conditioned_relation_mp3d_group_v1",
    world_id: str = "world_mp3d_relation_0001",
    qa_ids: Sequence[str] | None = None,
    seed: int | None = None,
    reference_time_s: int = 0,
    appearance_values: Sequence[str] = (),
    window_s: Sequence[int] = (4, 6),
    start_times_s: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Build one four-member visual-conditioned relation group.

    The supplied first capture is a read-only native v0 visual. Only the
    swapped v1 visual is planned and captured here; each audio assignment is
    rendered for v0 and then reused for v1 after native equivalence checks.
    """
    base_path = Path(base_request_path).expanduser().resolve()
    first_root = Path(first_visual_capture_root).expanduser().resolve()
    if not base_path.is_file():
        raise BindingNativeError(f"base request is unavailable: {base_path}")
    first_plan_path = first_root / "plan/episode_plan.json"
    first_capture = first_root / "capture"
    if not first_plan_path.is_file() or not first_capture.is_dir():
        raise BindingNativeError("first visual capture must contain plan/episode_plan.json and capture/")
    base = _load(base_path)
    if sound_pool is not None:
        pool_path = _file(sound_pool, base=REPOSITORY, owner="sound pool")
        base["sound_pool"] = str(pool_path)
    first_plan = _load(first_plan_path)
    capture_required = ("neutral_readback.json", "pixel_visibility_truth.json", "native_pixel_masks_depth_authority_v1.npz", "research_receipt.json")
    missing = [name for name in capture_required if not (first_capture / name).is_file()]
    if missing:
        raise BindingNativeError(f"first visual capture lacks {missing}: {first_capture}")
    selected_raw = source_asset_ids if source_asset_ids is not None else base.get("source_asset_ids")
    if (
        isinstance(selected_raw, (str, bytes))
        or not isinstance(selected_raw, Sequence)
        or len(selected_raw) != 3
        or len(set(selected_raw)) != 3
        or any(not isinstance(value, str) or not value.strip() for value in selected_raw)
    ):
        raise BindingNativeError("visual conditioned relation requires three distinct source assets")
    selected = tuple(str(value) for value in selected_raw)
    first_assets = tuple(
        str(actor.get("asset_id"))
        for actor in first_plan.get("visual_plan", {}).get("actors", [])
        if isinstance(actor, Mapping) and actor.get("actor_id") in {"source1", "source2", "source3"}
    )
    if first_assets != selected:
        raise BindingNativeError("supplied v0 capture asset order differs from selected source assets")
    if room_id is not None:
        base["room_id"] = room_id
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise BindingNativeError("seed must be an integer when supplied")
        base["seed"] = int(seed)
    if not isinstance(reference_time_s, int) or isinstance(reference_time_s, bool):
        raise BindingNativeError("reference_time_s must be an integer")
    if (
        isinstance(appearance_values, (str, bytes))
        or len(appearance_values) != 2
        or len(set(str(value) for value in appearance_values)) != 2
        or any(not isinstance(value, str) or not value.strip() for value in appearance_values)
    ):
        raise BindingNativeError("relation query requires two distinct appearance values")
    if (
        isinstance(window_s, (str, bytes))
        or len(window_s) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in window_s)
        or int(window_s[1]) <= int(window_s[0])
    ):
        raise BindingNativeError("relation query window must contain two increasing integer seconds")
    query = {
        "reference_time_s": int(reference_time_s),
        "appearance_values": [str(value) for value in appearance_values],
        "window_s": [int(window_s[0]), int(window_s[1])],
    }
    output = Path(output_root).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise BindingNativeError(f"refusing existing output root: {output}")
    output.mkdir(parents=True)
    for name in ("requests", "visual", "variants"):
        (output / name).mkdir()
    base_seed = base.get("seed")
    request0 = build_variant_request(
        base,
        episode_id=str(first_plan.get("episode_id") or f"{group_id}_v0"),
        source_asset_ids=selected,
        rpc_port=rpc_port,
        graphics_adapter=graphics_adapter,
        qa_ids=qa_ids,
        seed=base_seed,
    )
    request1 = build_variant_request(
        base,
        episode_id=f"{group_id}_v1",
        source_asset_ids=(selected[0], selected[2], selected[1]),
        rpc_port=rpc_port,
        graphics_adapter=graphics_adapter,
        qa_ids=qa_ids,
        seed=base_seed,
    )
    provenance = {
        "schema": "avengine_binding_group_native_provenance_v1",
        "status": "running",
        "repository": str(REPOSITORY.resolve()),
        "base_request": str(base_path),
        "first_visual_capture_root": str(first_root),
        "second_visual_capture_root": (
            str(Path(second_visual_capture_root).expanduser().resolve())
            if second_visual_capture_root is not None else None
        ),
        "base_request_episode_id": base.get("episode_id"),
        "base_request_schema": base.get("schema"),
        "group_id": group_id,
        "world_id": world_id,
        "task_family": RELATION_TASK_FAMILY,
        "room_id": base.get("room_id"),
        "source_asset_ids": list(selected),
        "sound_pool": base.get("sound_pool"),
        "relation_visual_swap": {"source2": selected[1], "source3": selected[2]},
        "runtime": {
            "host": platform.node(),
            "cwd": str(REPOSITORY.resolve()),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "PYTHONPATH": _env(REPOSITORY).get("PYTHONPATH", ""),
            "graphics_adapter": graphics_adapter,
            "rpc_port": rpc_port,
            "seed": base_seed,
        },
        "query": query,
        "start_times_s": list(start_times_s) if start_times_s is not None else None,
    }
    _write(output / "provenance_started.json", provenance)
    try:
        _write(output / "requests/v0_request.json", request0)
        _write(output / "requests/v1_request.json", request1)
        scheduled0, scheduled_request0, schedule = schedule_relation_audio_plan(
            first_plan,
            request0,
            start_times_s=start_times_s,
            query_window_s=window_s,
            rng_seed=base_seed,
        )
        _write(output / "requests/v0_scheduled_request.json", scheduled_request0)
        visual0_root = output / "visual/v0"
        (visual0_root / "plan").mkdir(parents=True)
        scheduled0_path = _write(visual0_root / "plan/episode_plan.json", scheduled0)
        visual0 = {
            "output": str(visual0_root),
            "plan": str(scheduled0_path),
            "capture": str(first_capture),
            "neutral_readback": str((first_capture / "neutral_readback.json").resolve()),
            "frame_readbacks": str((first_capture / "frame_readbacks.json").resolve() if (first_capture / "frame_readbacks.json").is_file() else (first_capture / "frame_records.json").resolve()),
            "visual_video": None,
            "reused": True,
        }
        if second_visual_capture_root is None:
            planned1 = plan_visual_variant(
                output / "requests/v1_request.json",
                output / "visual/v1",
                label="v1",
            )
            raw1 = _load(Path(planned1["plan"]))
        else:
            second_root = Path(second_visual_capture_root).expanduser().resolve()
            second_plan_path = second_root / "plan/episode_plan.json"
            second_capture = second_root / "capture"
            if not second_plan_path.is_file() or not second_capture.is_dir():
                raise BindingNativeError(
                    "second visual capture must contain plan/episode_plan.json and capture/"
                )
            missing = [
                name for name in capture_required
                if not (second_capture / name).is_file()
            ]
            if missing:
                raise BindingNativeError(f"second visual capture lacks {missing}: {second_capture}")
            raw1 = _load(second_plan_path)
            visual1_root = output / "visual/v1"
            (visual1_root / "plan").mkdir(parents=True)
            copied_plan = _write(visual1_root / "plan/episode_plan.json", raw1)
            planned1 = {
                "output": str(visual1_root),
                "plan": str(copied_plan),
                "capture": str(second_capture),
                "reused": True,
                "source_capture_root": str(second_root),
            }
        expected_second_assets = (selected[0], selected[2], selected[1])
        second_assets = tuple(
            str(actor.get("asset_id"))
            for actor in raw1.get("visual_plan", {}).get("actors", [])
            if isinstance(actor, Mapping)
            and actor.get("actor_id") in {"source1", "source2", "source3"}
        )
        if second_assets != expected_second_assets:
            raise BindingNativeError(
                "supplied v1 capture/plan asset order does not match the requested source2/source3 swap"
            )
        # The visual plan must be compared after applying the same explicit
        # schedule to both variants, while source captures remain read-only.
        selected_start_times = (
            None
            if start_times_s is None
            else tuple(
                schedule["start_times_s_by_actor"][actor_id]
                for actor_id in sorted(schedule["start_times_s_by_actor"])
            )
        )
        scheduled1, scheduled_request1, _ = schedule_relation_audio_plan(
            raw1,
            request1,
            start_times_s=selected_start_times,
            reserve_tail_s=(
                None
                if start_times_s is None
                else schedule["effective_reserve_tail_s"]
            ),
            query_window_s=window_s,
            rng_seed=base_seed,
        )
        _write(output / "requests/v1_scheduled_request.json", scheduled_request1)
        scheduled1_path = _write(
            Path(planned1["plan"]).with_name("relation_episode_plan.json"), scheduled1
        )
        plan_equivalence = compare_controlled_visual_plans(
            scheduled0_path, scheduled1_path)
        room_families = {
            "v0": room_family_from_plan(scheduled0),
            "v1": room_family_from_plan(scheduled1),
        }
        if room_families["v0"] != room_families["v1"]:
            raise BindingNativeError("visual variants resolve to different validated room families")
        if second_visual_capture_root is None:
            captured1 = capture_visual_plan(
                request1, planned1["output"], label="v1",
            )
        else:
            second_root = Path(second_visual_capture_root).expanduser().resolve()
            second_capture = second_root / "capture"
            captured1 = {
                "output": str(second_root),
                "plan": str(second_root / "plan/episode_plan.json"),
                "capture": str(second_capture),
                "neutral_readback": str((second_capture / "neutral_readback.json").resolve()),
                "frame_readbacks": str(
                    (second_capture / "frame_readbacks.json").resolve()
                    if (second_capture / "frame_readbacks.json").is_file()
                    else (second_capture / "frame_records.json").resolve()
                ),
                "visual_video": (
                    str((second_capture / "ue_visual_only.mp4").resolve())
                    if (second_capture / "ue_visual_only.mp4").is_file() else None
                ),
                "reused": True,
            }
        captured = {"v0": visual0, "v1": captured1}
        readback_equivalence = compare_native_visuals(captured["v0"], captured["v1"])
        endpoint_by_actor = _neutral_endpoint_bindings(
            captured["v0"]["neutral_readback"],
            plan=scheduled0,
        )
        endpoint_v1 = _neutral_endpoint_bindings(
            captured["v1"]["neutral_readback"],
            plan=scheduled1,
        )
        if endpoint_by_actor != endpoint_v1:
            raise BindingNativeError("visual variants expose different native source endpoint identities")
        acoustics = {
            "v0": acoustic_identity(scheduled_request0, scheduled0),
            "v1": acoustic_identity(scheduled_request1, scheduled1),
        }
        if acoustics["v0"] != acoustics["v1"]:
            raise BindingNativeError("acoustic input/configuration identity differs")
        variants, reports = {}, {}
        for assignment in ("a0", "a1"):
            plan0, req0 = build_audio_assignment_plan(
                scheduled0, scheduled_request0, assignment,
                assignment_targets=_RELATION_ASSIGNMENT_TARGETS,
                expected_event_count=3,
                endpoint_by_actor=endpoint_by_actor,
                require_authoritative_endpoints=True,
            )
            root0 = materialize_audio_variant(
                visual0, output / "variants" / f"v0_{assignment}",
                plan0, req0, member_id=f"v0_{assignment}",
            )
            variants[f"v0_{assignment}"] = finalize_audio_assignment(root0, req0)
            reports[assignment] = Path(variants[f"v0_{assignment}"]["audio_report"]).resolve()
            plan1, req1 = build_audio_assignment_plan(
                scheduled1, scheduled_request1, assignment,
                assignment_targets=_RELATION_ASSIGNMENT_TARGETS,
                expected_event_count=3,
                endpoint_by_actor=endpoint_by_actor,
                require_authoritative_endpoints=True,
            )
            root1 = materialize_audio_variant(
                captured1, output / "variants" / f"v1_{assignment}",
                plan1, req1, member_id=f"v1_{assignment}",
            )
            variants[f"v1_{assignment}"] = finalize_audio_assignment(
                root1, req1, audio_report=reports[assignment],
            )
            variants[f"v0_{assignment}"]["visual_capture_root"] = str(first_capture)
            variants[f"v1_{assignment}"]["visual_capture_root"] = str(Path(captured1["capture"]).resolve())
        relation_profile = {
            "task_family": RELATION_TASK_FAMILY,
            "source_count": 3,
            "camera": deepcopy(scheduled0.get("visual_plan", {}).get("camera") or {}),
            "audio_schedule": deepcopy(schedule),
            "reserve_tail_s": schedule["effective_reserve_tail_s"],
            "source_profile_reserve_tail_s": schedule["source_profile_reserve_tail_s"],
            "schedule_selection": schedule.get("selection_mode"),
            "query_window_s": list(query["window_s"]),
        }
        spec_path = _write(output / "group_spec.json", _relation_group_spec(
            group_id,
            world_id,
            room_families["v0"],
            str(scheduled_request0["room_id"]),
            query,
            variants,
            request=request0,
            profile=relation_profile,
        ))
        summary = {
            **provenance,
            "status": "pass",
            "room_family": room_families["v0"],
            "seed": base_seed,
            "query": query,
            "schedule": schedule,
            "plan_equivalence": plan_equivalence,
            "native_readback_equivalence": readback_equivalence,
            "acoustic_configuration": acoustics["v0"],
            "acoustic_equivalence": {"status": "pass", "same": True},
            "planned": {"v0": visual0, "v1": planned1},
            "captured": captured,
            "variants": variants,
            "group_spec": str(spec_path),
            "shared_audio_by_column": {
                assignment: {
                    "source_member": f"v0_{assignment}",
                    "reused_members": [f"v1_{assignment}"],
                    "audio_path": variants[f"v0_{assignment}"]["audio"],
                    "audio_report": str(reports[assignment]),
                }
                for assignment in ("a0", "a1")
            },
            "claim_boundary": "research_only native media and visual conditioned relation; no human/model/formal admission claim",
        }
        _write(output / "summary.json", summary)
        return summary
    except Exception as exc:
        try:
            _write(output / "failure.json", {
                **provenance,
                "status": "fail",
                "error": f"{type(exc).__name__}: {exc}",
            })
        except BindingNativeError:
            pass
        raise

def _retained_visual_entry(
    root: str | Path,
    *,
    expected_assets: Sequence[str],
    label: str,
) -> dict[str, Any]:
    """Validate and reference one retained static native visual capture."""
    retained_root = Path(root).expanduser().resolve()
    plan_path = retained_root / "plan/episode_plan.json"
    capture = retained_root / "capture"
    if not plan_path.is_file() or not capture.is_dir():
        raise BindingNativeError(
            f"{label} retained capture must contain plan/episode_plan.json and capture/"
        )
    plan = _load(plan_path)
    visual = plan.get("visual_plan") if isinstance(plan.get("visual_plan"), Mapping) else {}
    camera = visual.get("camera") if isinstance(visual.get("camera"), Mapping) else {}
    if str(camera.get("motion") or "").lower() != "static":
        raise BindingNativeError(f"{label} retained capture camera is not static")
    actors = visual.get("actors") if isinstance(visual.get("actors"), list) else []
    expected_actor_ids = {
        f"source{index + 1}" for index in range(len(expected_assets))
    }
    assets = tuple(
        str(actor.get("asset_id"))
        for actor in actors
        if isinstance(actor, Mapping)
        and actor.get("actor_id") in expected_actor_ids
    )
    if assets != tuple(str(value) for value in expected_assets):
        raise BindingNativeError(f"{label} retained capture asset order differs from the selected source assets")
    neutral = capture / "neutral_readback.json"
    if not neutral.is_file():
        raise BindingNativeError(f"{label} retained capture lacks neutral_readback.json")
    from avengine.capture.neutral_readback import validate_neutral_readback
    try:
        validate_neutral_readback(_load(neutral), plan=plan)
    except (TypeError, ValueError, KeyError) as exc:
        raise BindingNativeError(f"{label} retained native neutral readback is invalid: {exc}") from exc
    resources = plan.get("resources") if isinstance(plan.get("resources"), Mapping) else {}
    package = resources.get("room_package") if isinstance(resources.get("room_package"), Mapping) else {}
    renderer = str(package.get("renderer") or resources.get("renderer") or resources.get("backend") or "").lower()
    required = [
        "neutral_readback.json",
        "pixel_visibility_truth.json",
        "native_pixel_masks_depth_authority_v1.npz",
        "research_receipt.json",
    ]
    if renderer == "habitat" or resources.get("backend") == "habitat":
        required.extend(["frame_records.json", "rgb.npy"])
    else:
        required.extend(["frame_readbacks.json", "ue_visual_only.mp4"])
    missing = [name for name in required if not (capture / name).is_file()]
    if missing:
        raise BindingNativeError(f"{label} retained capture lacks {missing}: {capture}")
    frame_readbacks = capture / "frame_readbacks.json"
    if not frame_readbacks.is_file():
        frame_readbacks = capture / "frame_records.json"
    visual_video = capture / "ue_visual_only.mp4"
    return {
        "output": str(retained_root),
        "plan": str(plan_path),
        "capture": str(capture),
        "neutral_readback": str(neutral.resolve()),
        "frame_readbacks": str(frame_readbacks.resolve()),
        "visual_video": str(visual_video.resolve()) if visual_video.is_file() else None,
        "reused": True,
    }


# ---------------------------------------------------------------------------
# What the four members are measured to share
#
# The group's claim is that the picture cannot say which member you are
# watching. Two separate statements have to hold for that: the two visual
# variants have to move identically, and swapping which slot speaks first must
# not change anything in the picture either. Both are measured here from the
# members' own saved plans rather than argued from how the code is written.
# ---------------------------------------------------------------------------

#: Substrings that would mark a per-frame channel able to show who is speaking.
#: A plan carrying one of these could leak the audio intervention through the
#: picture, so the measurement looks for them by name instead of assuming the
#: renderer has no mouth animation.
SPEECH_ANIMATION_TOKENS = (
    "viseme", "lip", "mouth", "jaw", "phoneme", "speak", "talk",
    "utter", "voice_pose", "blendshape", "blend_shape", "morph",
)

#: Per-frame fields that carry the visible motion of one actor. Named so the
#: report can say which fields were compared rather than "the frames matched".
ACTOR_MOTION_FIELDS = (
    "root_transform", "emitter_transform", "planned_emitter_m",
    "action_id", "action_phase", "action_time_ticks", "moving", "support_identity",
)


def _speech_animation_channels(value: Any, *, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = f"{path}.{key}" if path else str(key)
            lowered = str(key).lower()
            if any(token in lowered for token in SPEECH_ANIMATION_TOKENS):
                # An emitter anchor names the mouth as a position, not as a pose
                # channel; it moves with the body and says nothing about speech.
                if not lowered.startswith("emitter") and "emitter" not in lowered:
                    found.append(name)
            found.extend(_speech_animation_channels(item, path=name))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_speech_animation_channels(item, path=f"{path}[{index}]"))
    return found


def measure_group_visual_invariance(
    plan_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Measure that every member of one group plans the same picture.

    plan_paths maps member id to that member's own saved episode plan. Each plan
    is reduced to the slot-normalized visual signature the controlled comparison
    uses, so a swapped asset identity is not counted as a difference while every
    transform, action, animation phase and movement flag still is.

    The returned row says how many frames and actor states were compared, which
    fields were compared, which differed, and whether any per-frame channel in
    the plans could show who is speaking.
    """
    rows = {str(member): Path(path).expanduser().resolve()
            for member, path in dict(plan_paths).items()}
    if len(rows) < 2:
        raise BindingNativeError("visual invariance needs at least two member plans")
    signatures, plans = {}, {}
    for member, path in sorted(rows.items()):
        plan = _load(path)
        plans[member] = plan
        signatures[member] = _controlled_visual_signature(plan)
    reference_id = sorted(signatures)[0]
    reference = signatures[reference_id]["frames"]
    frame_count = len(reference)
    actor_state_count = sum(len(frame.get("actor_states") or ()) for frame in reference)
    compared_fields: set[str] = set()
    for frame in reference:
        for state in frame.get("actor_states") or ():
            if isinstance(state, Mapping):
                compared_fields.update(str(key) for key in state)
    differences: list[dict[str, Any]] = []
    for member, signature in sorted(signatures.items()):
        if member == reference_id:
            continue
        frames = signature["frames"]
        if len(frames) != frame_count:
            differences.append({"member": member, "field": "frame_count",
                                "reference": frame_count, "member_value": len(frames)})
            continue
        for index, (left, right) in enumerate(zip(reference, frames, strict=True)):
            if left != right:
                keys = sorted({key for key in set(left) | set(right)
                               if left.get(key) != right.get(key)})
                differences.append({"member": member, "frame_index": index,
                                    "differing_keys": keys})
        for key in ("camera", "clock", "actor_slots"):
            if signatures[reference_id].get(key) != signature.get(key):
                differences.append({"member": member, "field": key})
    speech_channels = sorted({
        channel for plan in plans.values()
        for channel in _speech_animation_channels(
            (plan.get("visual_plan") or {}).get("frames"))
    })
    motion_flags_present = sorted(set(ACTOR_MOTION_FIELDS) & compared_fields)
    return {
        "status": "pass" if not differences and not speech_channels else "fail",
        "members": sorted(rows),
        "frames_compared_per_member": frame_count,
        "actor_states_compared_per_member": actor_state_count,
        "actor_state_fields_compared": sorted(compared_fields),
        "motion_fields_present": motion_flags_present,
        "differing_entries": differences,
        "speech_animation_channels_found": speech_channels,
        "plan_paths": {member: str(path) for member, path in sorted(rows.items())},
        "authority": (
            "each member's own saved episode plan, reduced to the slot-normalized "
            "visual signature; a swapped asset identity is not a difference, every "
            "transform, action, animation phase and movement flag is"
        ),
    }


def _group_comparisons(
    selected: Sequence[tuple[str, str, str]]
) -> list[dict[str, Any]]:
    """Pair the members that share exactly one modality.

    Two members of one shared visual differ only in their audio column, and two
    members of one audio column differ only in their video. Those are the
    necessity comparisons; they are read off the member/unit table rather than
    written out by name.
    """
    rows: list[dict[str, Any]] = []
    for modality, key in (("video", 1), ("audio", 2)):
        buckets: dict[str, list[str]] = {}
        for member in selected:
            name = (member[1] if key == 1 else member[2].rsplit("_", 1)[-1])
            buckets.setdefault(str(name), []).append(member[0])
        for name, member_ids in sorted(buckets.items()):
            if len(member_ids) != 2:
                raise BindingNativeError(
                    f"{modality} group {name!r} must pair exactly two members, "
                    f"got {member_ids}"
                )
            rows.append({"members": sorted(member_ids), "shared_modality": modality,
                         "answer_relation": "different", "kind": "necessity"})
    return rows


def _member_plan_path(row: Mapping[str, Any]) -> Path | None:
    """The saved plan of the member a finalized variant row describes."""
    for key in ("plan", "episode_plan"):
        value = row.get(key)
        if isinstance(value, str) and Path(value).is_file():
            return Path(value).resolve()
    facts = row.get("facts")
    if not isinstance(facts, str):
        return None
    current = Path(facts).resolve().parent
    for _ in range(4):
        candidate = current / "plan/episode_plan.json"
        if candidate.is_file():
            return candidate
        current = current.parent
    return None


def member_world_bindings(
    plan: Mapping[str, Any], *, registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """What one member's world binds: appearance per slot, slot per event.

    Read from the member's own saved plan. The appearance comes from the
    registry record of the asset each slot holds, so the group states the same
    value the delivery's appearance review checks against, and a plan whose
    asset carries no registered appearance says so instead of guessing.
    """
    visual = plan.get("visual_plan") if isinstance(plan.get("visual_plan"), Mapping) else {}
    records = {}
    if isinstance(registry, Mapping) and isinstance(registry.get("assets"), list):
        records = {str(row["asset_id"]): row for row in registry["assets"]
                   if isinstance(row, Mapping) and isinstance(row.get("asset_id"), str)}
    slot_appearances: dict[str, str] = {}
    slot_assets: dict[str, str] = {}
    for actor in visual.get("actors") or ():
        if not isinstance(actor, Mapping) or not actor.get("actor_id"):
            continue
        slot = str(actor["actor_id"])
        asset_id = actor.get("asset_id")
        if not isinstance(asset_id, str) or not asset_id:
            continue
        slot_assets[slot] = asset_id
        appearance = registered_appearance(records.get(asset_id, {}))
        if appearance is not None:
            slot_appearances[slot] = str(appearance["value"])
    events = sorted(
        [row for row in plan.get("audio_events") or () if isinstance(row, Mapping)],
        key=lambda row: (int(row.get("start_sample", 0)), str(row.get("event_id", ""))),
    )
    order = [str(row["event_id"]) for row in events if row.get("event_id")]
    declared = plan.get("audio_assignment_targets")
    event_slots = ({str(key): str(value) for key, value in declared.items()}
                   if isinstance(declared, Mapping) else
                   {str(row["event_id"]): str(row.get("actor_id") or "") for row in events
                    if row.get("event_id")})
    return {
        "slot_appearances": slot_appearances,
        "slot_assets": slot_assets,
        "event_order": order,
        "event_slots": event_slots,
        "authority": "the member's own saved episode plan and the request's source registry",
    }


def _group_spec(
    group_id: str,
    world_id: str,
    room_family: str,
    room_id: str,
    visual: Mapping[str, Mapping[str, Any]],
    variants: Mapping[str, Mapping[str, Any]],
    *,
    request: Mapping[str, Any] | None = None,
    profile: Mapping[str, Any] | None = None,
    member_units: Sequence[tuple[str, str, str]] | None = None,
    split: str = "pilot",
    task_family: str | None = None,
    qa_id: str | None = None,
    source_registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe one assembled group.

    member_units names (member_id, visual_unit_id, variant_key) triples
    so the recipe decides which shared visual and which audio column each
    member is made of. Without it the historical two-visual naming is used.
    """
    selected = list(member_units) if member_units is not None else [
        (member_id, member_id.split("_")[0], member_id)
        for member_id in ("v0_a0", "v0_a1", "v1_a0", "v1_a1")
    ]
    if len(selected) != 4 or len({row[0] for row in selected}) != 4:
        raise BindingNativeError(
            f"a binding group delivers four distinct members, got {[row[0] for row in selected]}"
        )
    members = []
    for member_id, visual_id, variant_key in selected:
        assignment = variant_key.rsplit("_", 1)[-1]
        video_value = visual[visual_id].get("visual_video")
        if not video_value:
            video_value = variants[variant_key].get("visual_video")
        if not video_value:
            raise BindingNativeError(f"visible binding member lacks a finalized visual video: {member_id}")
        member = {
            "member_id": member_id,
            "facts_path": str(Path(variants[variant_key]["facts"]).resolve()),
            "video_path": str(Path(video_value).resolve()),
            "audio_path": str(Path(variants[variant_key]["audio"]).resolve()),
            "interventions": {
                "visual_variant": visual_id,
                "audio_assignment": assignment,
                "native_capture_root": visual[visual_id]["capture"],
            },
        }
        audio_delivery = variants[variant_key].get("audio_delivery")
        if isinstance(audio_delivery, Mapping):
            member["audio_delivery"] = deepcopy(dict(audio_delivery))
        member["_factor_levels"] = {
            "visual_appearance_slots": visual_id,
            "audio_event_slot_assignment": assignment,
        }
        member["_plan_path"] = _member_plan_path(variants[variant_key])
        members.append(member)
    from avengine.qa.binding_conditions import (
        BindingConditionError, derive_group_comparisons, group_question_recipe,
        member_intervention_record,
    )

    recipe = group_question_recipe(qa_id) if qa_id else None
    query = dict(recipe["default_query"]) if recipe else {"event_number": 1}
    comparisons = None
    if recipe is not None and all(member.get("_plan_path") for member in members):
        rows = []
        for member in members:
            bindings = member_world_bindings(
                _load(Path(member["_plan_path"])), registry=source_registry)
            member["world_bindings"] = deepcopy(bindings)
            rows.append({"member_id": member["member_id"],
                         "factor_levels": member["_factor_levels"],
                         "bindings": bindings})
        comparisons = derive_group_comparisons(rows, qa_id=recipe["qa_id"], query=query)
        planned = {}
        for row in comparisons:
            planned.update(row["predicted_answers"])
        for member in members:
            member["planned_answer"] = {
                "value": planned.get(member["member_id"]),
                "qa_id": recipe["qa_id"],
                "answer_variable": recipe["answer_variable"],
                "source": "declared world bindings before rendering; the delivered "
                          "answer is recomputed from this member's own facts",
            }
    for member in members:
        levels = member.pop("_factor_levels")
        member.pop("_plan_path", None)
        if recipe is not None:
            member["interventions"] = member_intervention_record(
                levels, qa_id=recipe["qa_id"], extra=member["interventions"])
    group = {
        "group_id": group_id, "world_id": world_id,
        "task_family": task_family or TASK_FAMILY, "room_family": room_family,
        "room_id": room_id, "split": split, "query": query,
        "angle_tolerance_deg": 10,
        "members": members,
        "comparisons": comparisons if comparisons is not None
                       else _group_comparisons(selected),
    }
    if recipe is not None:
        group["question_recipe"] = deepcopy(recipe)
    if request is not None:
        group["request"] = deepcopy(dict(request))
    if profile is not None:
        group["profile"] = deepcopy(dict(profile))
    return {
        "schema": "avengine_binding_group_spec_v1",
        **({"request": deepcopy(dict(request))} if request is not None else {}),
        "groups": [group],
    }


def _requested_group_question(request: Mapping[str, Any]) -> str | None:
    """The catalog question a base request asks its group to be built around."""
    targets = request.get("qa_targets")
    named = [str(row["qa_id"]) for row in targets or ()
             if isinstance(row, Mapping) and isinstance(row.get("qa_id"), str)
             and row["qa_id"].strip()]
    quota = [str(key) for key in (request.get("quota_by_qa") or {})]
    unique = sorted(set(named) or set(quota))
    if len(unique) > 1:
        raise BindingNativeError(
            f"a core group is built around one question; the request names {unique}")
    return unique[0] if unique else None


def _registry_document(request: Mapping[str, Any]) -> dict[str, Any] | None:
    """The source registry a request selected, loaded once for appearance reads."""
    path = request.get("source_registry")
    if not isinstance(path, str) or not path.strip():
        return None
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (REPOSITORY / resolved).resolve()
    return _load(resolved) if resolved.is_file() else None


def prepare_visible_binding_group(
    *,
    base_request_path: str | Path,
    output_root: str | Path,
    first_visual_capture_root: str | Path | None = None,
    second_visual_capture_root: str | Path | None = None,
    sound_pool: str | Path | None = None,
    prepared_manifest: str | Path | None = None,
    source_asset_ids: Sequence[str] | None = None,
    room_id: str | None = None,
    rpc_port: int | None = None,
    graphics_adapter: int | None = None,
    group_id: str = "visible_binding_group_v0",
    world_id: str = "world_0001",
    qa_ids: Sequence[str] | None = None,
    seed: int | None = None,
    group_question: str | None = None,
) -> dict[str, Any]:
    """Build one four-member visible-binding group.

    Retained visual roots are validated and linked read-only. When a retained
    root is absent, only that visual variant is planned and captured; every
    audio assignment is still finalized from an actual native audio run.

    ``group_question`` names the catalog question the group is built around. It
    defaults to whatever the base request asks for, so the question type is a
    property of the request rather than a constant in this function; a question
    without a group recipe or without a builder is refused by name.
    """
    if second_visual_capture_root is not None and first_visual_capture_root is None:
        raise BindingNativeError("second visual capture requires a retained first visual capture")
    base_path = Path(base_request_path).expanduser().resolve()
    base = _load(base_path)
    if sound_pool is not None:
        base["sound_pool"] = str(_file(sound_pool, base=REPOSITORY, owner="sound pool"))
    if prepared_manifest is not None:
        base["prepared_manifest"] = str(
            _file(prepared_manifest, base=REPOSITORY, owner="prepared manifest")
        )
    if room_id is not None:
        base["room_id"] = room_id
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise BindingNativeError("seed must be an integer when supplied")
        base["seed"] = int(seed)
    selected_raw = source_asset_ids
    retained_first_plan = None
    if selected_raw is None and first_visual_capture_root is not None:
        retained_first_plan = _load(
            Path(first_visual_capture_root).expanduser().resolve()
            / "plan/episode_plan.json"
        )
        selected_raw = [
            actor.get("asset_id")
            for actor in retained_first_plan.get("visual_plan", {}).get("actors", [])
            if isinstance(actor, Mapping)
            and actor.get("actor_id") in {"source1", "source2"}
        ]
    if selected_raw is None:
        selected_raw = base.get("source_asset_ids")
    if (
        isinstance(selected_raw, (str, bytes))
        or not isinstance(selected_raw, Sequence)
        or len(selected_raw) != 2
        or len(set(selected_raw)) != 2
        or any(not isinstance(value, str) or not value.strip() for value in selected_raw)
    ):
        raise BindingNativeError(
            "visible binding requires two distinct selected source assets"
        )
    selected = tuple(str(value) for value in selected_raw)
    from avengine.qa.binding_conditions import (
        TASK_QA_IDS, implemented_group_question_recipe,
    )

    requested_question = group_question or _requested_group_question(base)
    question_recipe = implemented_group_question_recipe(
        requested_question or TASK_QA_IDS[TASK_FAMILY])
    if question_recipe["task_family"] != TASK_FAMILY:
        raise BindingNativeError(
            f"{question_recipe['qa_id']} is built by the "
            f"{question_recipe['task_family']!r} recipe, not by visible binding")
    if first_visual_capture_root is not None:
        first_plan = retained_first_plan or _load(
            Path(first_visual_capture_root).expanduser().resolve()
            / "plan/episode_plan.json"
        )
        first_assets = tuple(
            str(actor.get("asset_id"))
            for actor in first_plan.get("visual_plan", {}).get("actors", [])
            if isinstance(actor, Mapping)
            and actor.get("actor_id") in {"source1", "source2"}
        )
        if first_assets != selected:
            raise BindingNativeError(
                "supplied v0 capture asset order differs from selected source assets"
            )
    output = Path(output_root).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise BindingNativeError(f"refusing existing output root: {output}")
    output.mkdir(parents=True)
    for name in ("requests", "visual", "variants"):
        (output / name).mkdir()
    prepared_manifests = {
        str(event.get("source_metadata_manifest")).strip()
        for event in (
            (retained_first_plan or {}).get("audio_events", [])
            if retained_first_plan is not None else ()
        )
        if isinstance(event, Mapping)
        and isinstance(event.get("source_metadata_manifest"), str)
        and str(event.get("source_metadata_manifest")).strip()
    }
    if len(prepared_manifests) == 1 and Path(next(iter(prepared_manifests))).is_file():
        base["prepared_manifest"] = next(iter(prepared_manifests))
    requests = {}
    provenance = {
        "schema": "avengine_binding_group_native_provenance_v1",
        "status": "running",
        "repository": str(REPOSITORY.resolve()),
        "base_request": str(base_path),
        "first_visual_capture_root": (
            str(Path(first_visual_capture_root).expanduser().resolve())
            if first_visual_capture_root is not None else None
        ),
        "second_visual_capture_root": (
            str(Path(second_visual_capture_root).expanduser().resolve())
            if second_visual_capture_root is not None else None
        ),
        "base_request_episode_id": base.get("episode_id"),
        "base_request_schema": base.get("schema"),
        "group_id": group_id,
        "world_id": world_id,
        "task_family": TASK_FAMILY,
        "room_id": base.get("room_id"),
        "source_asset_ids": list(selected),
        "sound_pool": base.get("sound_pool"),
        "prepared_manifest": base.get("prepared_manifest"),
        "runtime": {
            "host": platform.node(),
            "cwd": str(REPOSITORY.resolve()),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "PYTHONPATH": _env(REPOSITORY).get("PYTHONPATH", ""),
            "graphics_adapter": graphics_adapter,
            "rpc_port": rpc_port,
            "seed": base.get("seed"),
        },
    }
    _write(output / "provenance_started.json", provenance)
    try:
        requests["v0"] = build_variant_request(
            base,
            episode_id=f"{group_id}_v0",
            source_asset_ids=selected,
            rpc_port=rpc_port,
            graphics_adapter=graphics_adapter,
            qa_ids=qa_ids,
            seed=base.get("seed"),
        )
        requests["v1"] = build_variant_request(
            base,
            episode_id=f"{group_id}_v1",
            source_asset_ids=(selected[1], selected[0]),
            rpc_port=rpc_port,
            graphics_adapter=graphics_adapter,
            qa_ids=qa_ids,
            seed=base.get("seed"),
        )
        _write(output / "requests/v0_request.json", requests["v0"])
        _write(output / "requests/v1_request.json", requests["v1"])
        retained_roots = {
            "v0": first_visual_capture_root,
            "v1": second_visual_capture_root,
        }
        planned = {}
        for visual_id, assets in (("v0", selected), ("v1", (selected[1], selected[0]))):
            retained = retained_roots[visual_id]
            if retained is not None:
                planned[visual_id] = _retained_visual_entry(
                    retained,
                    expected_assets=assets,
                    label=visual_id,
                )
            else:
                planned[visual_id] = plan_visual_variant(
                    output / "requests" / f"{visual_id}_request.json",
                    output / "visual" / visual_id,
                    label=visual_id,
                )
        plan_equivalence = compare_controlled_visual_plans(
            Path(planned["v0"]["plan"]), Path(planned["v1"]["plan"]),
            expected_slot_assets={"left": selected, "right": (selected[1], selected[0])},
        )
        room_families = {
            visual_id: room_family_from_plan(_load(Path(planned[visual_id]["plan"])))
            for visual_id in ("v0", "v1")
        }
        if room_families["v0"] != room_families["v1"]:
            raise BindingNativeError(
                "visual variants resolve to different validated room families"
            )
        room_family = room_families["v0"]
        captured = {}
        for visual_id in ("v0", "v1"):
            if planned[visual_id].get("reused"):
                captured[visual_id] = planned[visual_id]
            else:
                captured[visual_id] = capture_visual_plan(
                    requests[visual_id], planned[visual_id]["output"], label=visual_id
                )
        readback_equivalence = compare_native_visuals(
            captured["v0"], captured["v1"]
        )
        endpoint_by_actor = _neutral_endpoint_bindings(
            captured["v0"]["neutral_readback"],
            plan=_load(Path(planned["v0"]["plan"])),
        )
        endpoint_v1 = _neutral_endpoint_bindings(
            captured["v1"]["neutral_readback"],
            plan=_load(Path(planned["v1"]["plan"])),
        )
        if endpoint_by_actor != endpoint_v1:
            raise BindingNativeError(
                "visual variants expose different native source endpoint identities"
            )
        acoustics = {
            visual_id: acoustic_identity(
                requests[visual_id], _load(Path(planned[visual_id]["plan"]))
            )
            for visual_id in ("v0", "v1")
        }
        if acoustics["v0"] != acoustics["v1"]:
            raise BindingNativeError("acoustic input/configuration identity differs")
        variants, reports = {}, {}
        for assignment in ("a0", "a1"):
            plan0, req0 = build_audio_assignment_plan(
                _load(Path(planned["v0"]["plan"])),
                requests["v0"],
                assignment,
                endpoint_by_actor=endpoint_by_actor,
                require_authoritative_endpoints=True,
            )
            root0 = materialize_audio_variant(
                captured["v0"],
                output / "variants" / f"v0_{assignment}",
                plan0,
                req0,
                member_id=f"v0_{assignment}",
            )
            variants[f"v0_{assignment}"] = finalize_audio_assignment(root0, req0)
            reports[assignment] = Path(
                variants[f"v0_{assignment}"]["audio_report"]
            ).resolve()
            plan1, req1 = build_audio_assignment_plan(
                _load(Path(planned["v1"]["plan"])),
                requests["v1"],
                assignment,
                endpoint_by_actor=endpoint_by_actor,
                require_authoritative_endpoints=True,
            )
            root1 = materialize_audio_variant(
                captured["v1"],
                output / "variants" / f"v1_{assignment}",
                plan1,
                req1,
                member_id=f"v1_{assignment}",
            )
            variants[f"v1_{assignment}"] = finalize_audio_assignment(
                root1, req1, audio_report=reports[assignment]
            )
            variants[f"v0_{assignment}"]["visual_capture_root"] = str(
                Path(captured["v0"]["capture"]).resolve()
            )
            variants[f"v1_{assignment}"]["visual_capture_root"] = str(
                Path(captured["v1"]["capture"]).resolve()
            )
        visual_profile = {
            "task_family": TASK_FAMILY,
            "source_count": 2,
            "camera": deepcopy(
                _load(Path(planned["v0"]["plan"]))
                .get("visual_plan", {})
                .get("camera", {})
            ),
            "camera_motion": "static",
            "audio": {
                "rir_stride": requests["v0"].get("rir_stride"),
                "post_assembly_convolution_gain": requests["v0"].get(
                    "post_assembly_convolution_gain"
                ),
            },
            "reserve_tail_s": (
                requests["v0"].get("profile", {}).get("reserve_tail_s")
            ),
            "sound_pool": requests["v0"].get("sound_pool"),
        }
        spec_path = _write(
            output / "group_spec.json",
            _group_spec(
                group_id,
                world_id,
                room_family,
                str(requests["v0"]["room_id"]),
                captured,
                variants,
                request=requests["v0"],
                profile=visual_profile,
                qa_id=question_recipe["qa_id"],
                source_registry=_registry_document(requests["v0"]),
            ),
        )
        summary = {
            **provenance,
            "status": "pass",
            "room_family": room_family,
            "seed": base.get("seed"),
            "group_question": deepcopy(question_recipe),
            "visual_invariance": measure_group_visual_invariance({
                f"{visual_id}_{assignment}": Path(
                    output / "variants" / f"{visual_id}_{assignment}" / "plan/episode_plan.json")
                for visual_id in ("v0", "v1") for assignment in ("a0", "a1")
            }),
            "plan_equivalence": plan_equivalence,
            "native_readback_equivalence": readback_equivalence,
            "acoustic_configuration": acoustics["v0"],
            "acoustic_equivalence": {"status": "pass", "same": True},
            "planned": planned,
            "captured": captured,
            "variants": variants,
            "group_spec": str(spec_path),
            "shared_audio_by_column": {
                assignment: {
                    "source_member": f"v0_{assignment}",
                    "reused_members": [f"v1_{assignment}"],
                    "audio_path": variants[f"v0_{assignment}"]["audio"],
                    "audio_report": str(reports[assignment]),
                }
                for assignment in ("a0", "a1")
            },
            "claim_boundary": "research_only native media and binding relations; no human/model/formal admission claim",
        }
        _write(output / "summary.json", summary)
        return summary
    except Exception as exc:
        try:
            _write(
                output / "failure.json",
                {**provenance, "status": "fail", "error": f"{type(exc).__name__}: {exc}"},
            )
        except BindingNativeError:
            pass
        raise




# ---------------------------------------------------------------------------
# Shared native stages for a controlled four-member core group
#
# The four members of a core group are crossed conditions of ONE controlled
# world, not four independent worlds. They share the room, the clock, the
# camera rig and the sampled geometry; the only differences are the ones the
# recipe declares. These stages are what P01's GROUP_RECIPES /
# initial_group_work_items / next_group_work_items describe: each unit is one
# resumable step an ordinary program worker can run, save and pick up again.
# ---------------------------------------------------------------------------

GROUP_STAGE_SCHEMA = "avengine_native_group_stage_result_v1"
STAGE_RESULT_FILENAME = "stage_result.json"

# Fields every member of one controlled world must agree on. A difference here
# means the members are not crossed conditions of a single world.  Sound
# selection policy is checked separately; selected dry content belongs to the
# declared shared audio columns.
CONTROLLED_WORLD_REQUEST_FIELDS = (
    "room_id", "room_catalog", "source_registry",
    "frame_count", "frame_rate_hz", "sample_rate_hz", "clock",
    "camera", "sampling_policy", "seed", "sampling_candidate_index",
    "sound_pool", "prepared_manifest",
    "rir_stride", "post_assembly_convolution_gain",
    "diffraction", "max_diffraction_order",
    "source_context_policy", "foa_normalization", "audio_layouts",
    "motion_timing", "simulation", "profile",
    "allow_research_candidate_assets",
)
# Runtime keys that describe the shared acoustic/visual runtime rather than the
# per-instance placement a resource lease chooses.
CONTROLLED_WORLD_RUNTIME_KEYS = (
    "hrtf", "hrtf_id", "runtime_prefix", "rlr_sdk_root", "magnum_python_site",
    "magnum_site", "mp3d_root", "uproject", "unreal_editor", "spear_ext_dir",
    "streaming_warmup_frames",
)
# Per-instance runtime values a lease selects; they never define the world.
INSTANCE_RUNTIME_KEYS = ("graphics_adapter", "rpc_port", "rlr_threads")


# What an undeclared world field actually means, so a request that omits it and
# one that states the default are not read as two different worlds.
WORLD_FIELD_DEFAULTS: dict[str, Any] = {
    "motion_timing": "none",
    "source_context_policy": "joint",
    "foa_normalization": "native_n3d",
    "audio_layouts": [{"type": "binaural", "channel_count": 2, "role": "primary"}],
    "diffraction": False,
    "sound_selection": {},
    "simulation": {},
    "allow_research_candidate_assets": False,
    # T03's candidate rotation changes the sampled geometry stream.  An
    # omitted legacy request is the original candidate-0 stream.
    "sampling_candidate_index": 0,
}
# Fields a retained visual root is allowed to state differently: they name the
# episode and the declared intervention, not the world.
RETAINED_REQUEST_INTERVENTION_FIELDS = frozenset({
    "episode_id", "request_id", "source_asset_ids", "entity_instances",
    "qa_ids", "qa_targets", "qa_sampling", "quota_by_qa",
    "task_family", "group_id", "member_role", "condition_group",
    "binding_variant", "production",
})
# These outputs are rendered after a visual capture and do not change its RGB,
# pose, camera or geometry. Retained visual validation records them separately;
# the group contract still requires all members to agree on them.
RETAINED_VISUAL_AUDIO_VIEW_FIELDS = frozenset({
    "audio_layouts", "foa_normalization",
})


def _sampling_candidate_index_value(value: Any, *, owner: str) -> int:
    """Validate the geometry candidate stream selected by a request."""
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise BindingNativeError(
            f"{owner} must be a nonnegative integer"
        )
    return int(value)


def _world_field(request: Mapping[str, Any], field: str) -> Any:
    """One world field, with an undeclared value read as its actual default."""
    value = request.get(field)
    if value is None and field in WORLD_FIELD_DEFAULTS:
        value = deepcopy(WORLD_FIELD_DEFAULTS[field])
    if field == "sampling_candidate_index":
        return _sampling_candidate_index_value(
            value, owner="sampling_candidate_index"
        )
    return value


def _audio_view_fields(request: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: _world_field(request, field)
        for field in sorted(RETAINED_VISUAL_AUDIO_VIEW_FIELDS)
    }


def _apply_member_audio_view_fields(
    capture_request: Mapping[str, Any],
    member_request: Mapping[str, Any],
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Overlay only the current member's post-capture audio view settings."""
    if not isinstance(capture_request, Mapping):
        raise BindingNativeError(f"{label} capture request is not an object")
    if not isinstance(member_request, Mapping):
        raise BindingNativeError(f"{label} member request is not an object")
    retained = _audio_view_fields(capture_request)
    requested = _audio_view_fields(member_request)
    applied = deepcopy(dict(capture_request))
    for field, value in requested.items():
        applied[field] = deepcopy(value)
    record = {
        "label": label,
        "requested": deepcopy(requested),
        "retained": deepcopy(retained),
        "applied": deepcopy(requested),
        "fields": {
            field: {
                "requested": deepcopy(requested[field]),
                "retained": deepcopy(retained[field]),
                "same": requested[field] == retained[field],
            }
            for field in sorted(RETAINED_VISUAL_AUDIO_VIEW_FIELDS)
        },
    }
    return applied, record


def _member_request_for_audio_unit(
    context: Mapping[str, Any],
    item: Mapping[str, Any],
    unit_spec: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    requests = context.get("member_requests")
    if not isinstance(requests, Mapping):
        raise BindingNativeError(
            f"{item.get('unit_id')} audio unit has no current member_requests"
        )
    member_ids = list(item.get("member_request_ids") or ())
    if not member_ids:
        member_ids = list(unit_spec.get("member_request_ids") or ())
    if len(member_ids) != 1:
        raise BindingNativeError(
            f"{item.get('unit_id')} audio unit must identify exactly one current "
            f"member request, got {member_ids}"
        )
    member_id = str(member_ids[0])
    request = requests.get(member_id)
    if not isinstance(request, Mapping):
        raise BindingNativeError(
            f"{item.get('unit_id')} audio unit has no request for member {member_id}"
        )
    return member_id, deepcopy(dict(request))


def verify_retained_visual_request(
    retained_request: Mapping[str, Any], contract: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    """Reuse a retained visual only when it is a valid input for this group.

    A retained capture carries its own request, and adopting its plan adopts
    that world. So the group's declared world has to be the same world: the
    room, the clock, the camera rig, the sampling seed, the sound inputs and the
    acoustic configuration must agree. The file being on disk is not the
    evidence, and a group config that names a different seed than the world it
    reuses is refused instead of quietly inheriting the retained one.
    """
    shared = contract["shared_world"]
    conflicts = []
    requested_audio_view = _audio_view_fields(shared)
    retained_audio_view = _audio_view_fields(retained_request)
    audio_view_record = {
        "requested": deepcopy(requested_audio_view),
        "retained": deepcopy(retained_audio_view),
        "fields": {
            field: {
                "requested": deepcopy(requested_audio_view[field]),
                "retained": deepcopy(retained_audio_view[field]),
                "same": requested_audio_view[field] == retained_audio_view[field],
            }
            for field in sorted(RETAINED_VISUAL_AUDIO_VIEW_FIELDS)
        },
    }
    for field in CONTROLLED_WORLD_REQUEST_FIELDS:
        if (
            field in RETAINED_REQUEST_INTERVENTION_FIELDS
            or field in RETAINED_VISUAL_AUDIO_VIEW_FIELDS
        ):
            continue
        expected = _world_field(shared, field)
        actual = _world_field(retained_request, field)
        if expected != actual:
            conflicts.append({"field": field, "group_declares": _short(expected),
                              "retained_root_declares": _short(actual)})
    shared_audio_policy = contract.get("shared_audio_policy")
    if shared_audio_policy is not None:
        actual_audio_policy = normalize_sound_selection_policy(
            retained_request.get("sound_selection")
        )
        if actual_audio_policy != shared_audio_policy:
            conflicts.append({
                "field": "sound_selection_policy",
                "group_declares": _short(shared_audio_policy),
                "retained_root_declares": _short(actual_audio_policy),
            })
    retained_runtime = dict(retained_request.get("runtime") or {})
    for key in CONTROLLED_WORLD_RUNTIME_KEYS:
        expected = (contract["shared_runtime"] or {}).get(key)
        actual = retained_runtime.get(key)
        if expected != actual:
            conflicts.append({"field": f"runtime.{key}", "group_declares": _short(expected),
                              "retained_root_declares": _short(actual)})
    if conflicts:
        raise BindingNativeError(
            f"the retained visual root for {label} was produced for a different world; "
            f"conflicting declarations: {conflicts}"
        )
    return {
        "status": "pass",
        "label": label,
        "retained_episode_id": retained_request.get("episode_id"),
        "checked_fields": [
            field for field in CONTROLLED_WORLD_REQUEST_FIELDS
            if (
                field not in RETAINED_REQUEST_INTERVENTION_FIELDS
                and field not in RETAINED_VISUAL_AUDIO_VIEW_FIELDS
            )
        ],
        "separately_recorded_audio_view_fields": sorted(
            RETAINED_VISUAL_AUDIO_VIEW_FIELDS
        ),
        "audio_view_fields": audio_view_record,
        "checked_audio_policy_fields": list(SOUND_SELECTION_POLICY_FIELDS),
        "checked_runtime_keys": list(CONTROLLED_WORLD_RUNTIME_KEYS),
        "authority": (
            "the retained root's own saved request; visual world fields compared "
            "field by field; audio view fields recorded separately for downstream delivery"
        ),
    }


def _short(value: Any, limit: int = 200) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _recipe_options(group_spec: Mapping[str, Any]) -> dict[str, str]:
    """Resolve recipe-owned intervention and equivalence semantics."""
    recipe = group_spec.get("recipe")
    recipe = recipe if isinstance(recipe, Mapping) else {}
    family = str(group_spec.get("task_family") or recipe.get("task_family") or "")
    fallback_plan = (
        "world" if family in {"cross_event_identity", "cross_time_state"}
        else "controlled_slots"
    )
    fallback_intervention = {
        "cross_event_identity": "identity_path_topology",
        "cross_time_state": "after_wet_tail_motion",
    }.get(family, "source_slot_permutation")
    fallback_query = "slot" if family in {
        "visible_binding", "visual_conditioned_relation"
    } else "exact"
    plan_equivalence = str(recipe.get("plan_equivalence") or fallback_plan)
    visual_intervention = str(
        recipe.get("visual_intervention") or fallback_intervention
    )
    query_identity_policy = str(
        recipe.get("query_identity_policy") or fallback_query
    )
    audio_content_scope = str(
        recipe.get("audio_content_scope") or "shared_audio_pairs"
    )
    if plan_equivalence not in PLAN_EQUIVALENCE_MODES:
        raise BindingNativeError(
            f"{family} declares unknown plan equivalence {plan_equivalence!r}"
        )
    if visual_intervention not in VISUAL_INTERVENTION_MODES:
        raise BindingNativeError(
            f"{family} declares unknown visual intervention {visual_intervention!r}"
        )
    if query_identity_policy not in QUERY_IDENTITY_POLICIES:
        raise BindingNativeError(
            f"{family} declares unknown query identity policy {query_identity_policy!r}"
        )
    if audio_content_scope not in AUDIO_CONTENT_SCOPES:
        raise BindingNativeError(
            f"{family} declares unknown audio content scope {audio_content_scope!r}"
        )
    return {
        "plan_equivalence": plan_equivalence,
        "visual_intervention": visual_intervention,
        "query_identity_policy": query_identity_policy,
        "audio_content_scope": audio_content_scope,
    }


def _declared_audio_content(request: Mapping[str, Any]) -> dict[str, Any]:
    """Keep selected dry content while dropping policy and candidate allowlists."""
    selection = request.get("sound_selection")
    if not isinstance(selection, Mapping):
        return {}
    content = {
        str(key): deepcopy(value)
        for key, value in selection.items()
        if key not in SOUND_SELECTION_POLICY_FIELDS
        and key != "preallocated_sound_asset_ids_by_actor"
    }
    selected = content.pop("selected_sound_asset_ids_by_actor", None)
    if isinstance(selected, Mapping):
        pools = []
        for values in selected.values():
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                pools.append(sorted(str(value) for value in values))
            else:
                pools.append([str(values)])
        content["selected_sound_asset_ids"] = sorted(pools)
    return content


def _audio_candidate_by_asset(
    request: Mapping[str, Any],
) -> dict[str, list[str]] | None:
    """Read a declared actor-keyed candidate pool by physical asset."""
    selection = request.get("sound_selection")
    if not isinstance(selection, Mapping):
        return None
    declared = selection.get("preallocated_sound_asset_ids_by_actor")
    if declared is None:
        return None
    if not isinstance(declared, Mapping):
        raise BindingNativeError(
            "explicit sound preallocation must be an actor-to-sound-ID mapping"
        )
    instances = request.get("entity_instances")
    asset_by_actor: dict[str, str] = {}
    if isinstance(instances, Sequence) and not isinstance(instances, (str, bytes)):
        for instance in instances:
            if not isinstance(instance, Mapping):
                continue
            actor_id = instance.get("instance_id")
            asset_id = instance.get("asset_id")
            if isinstance(actor_id, str) and asset_id is not None:
                asset_by_actor[actor_id] = str(asset_id)
    if not asset_by_actor:
        assets = request.get("source_asset_ids")
        if isinstance(assets, Sequence) and not isinstance(assets, (str, bytes)):
            asset_by_actor = {
                f"source{index + 1}": str(asset_id)
                for index, asset_id in enumerate(assets)
                if asset_id is not None
            }
    result: dict[str, list[str]] = {}
    for actor_id, values in declared.items():
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) or not value for value in values)
        ):
            raise BindingNativeError(
                "explicit sound preallocation must map actors to nonempty sound-ID lists"
            )
        asset_id = asset_by_actor.get(str(actor_id), f"actor:{actor_id}")
        copied = [str(value) for value in values]
        if asset_id in result and result[asset_id] != copied:
            raise BindingNativeError(
                f"one physical asset has conflicting sound preallocation: {asset_id}"
            )
        result[asset_id] = copied
    return result


def _shared_audio_pairs(
    group_spec: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
    columns: Mapping[str, Sequence[str]],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Resolve declared member pairs to the audio units they actually share."""
    member_ids = [str(value) for value in group_spec.get("member_request_ids") or ()]
    audio_unit_by_member: dict[str, str] = {}
    row_by_unit = {}
    for row in units:
        if (
            row.get("unit_kind") != "audio"
            or row.get("internal_only")
        ):
            continue
        unit_id = str(row.get("unit_id"))
        row_by_unit[unit_id] = row
        served = [str(value) for value in row.get("member_request_ids") or ()]
        if len(served) == 1:
            audio_unit_by_member[served[0]] = unit_id

    declared = group_spec.get("shared_audio_member_ids") or ()
    member_pairs: list[tuple[str, str]] = []
    unit_pairs: list[tuple[str, str]] = []
    if declared:
        for raw_pair in declared:
            if (
                not isinstance(raw_pair, Sequence)
                or isinstance(raw_pair, (str, bytes))
                or len(raw_pair) != 2
            ):
                raise BindingNativeError(
                    f"shared_audio_member_ids must contain two-member pairs: {raw_pair!r}"
                )
            pair = tuple(str(value) for value in raw_pair)
            if len(set(pair)) != 2 or any(value not in member_ids for value in pair):
                raise BindingNativeError(
                    f"shared audio pair names unknown or duplicate members: {pair}"
                )
            resolved = tuple(audio_unit_by_member.get(value) for value in pair)
            if any(value is None for value in resolved):
                raise BindingNativeError(
                    f"shared audio pair has no audio unit: {pair}"
                )
            if not any(set(resolved) == set(str(value) for value in ids)
                       for ids in columns.values()):
                raise BindingNativeError(
                    f"shared audio pair is not one assignment column: {pair}"
                )
            member_pairs.append(pair)
            unit_pairs.append((str(resolved[0]), str(resolved[1])))
    else:
        for ids in columns.values():
            resolved_units = [str(value) for value in ids]
            if len(resolved_units) != 2:
                continue
            resolved_members = []
            for unit_id in resolved_units:
                served = [
                    str(value)
                    for value in row_by_unit[unit_id].get("member_request_ids") or ()
                ]
                if len(served) != 1:
                    break
                resolved_members.append(served[0])
            if len(resolved_members) == 2:
                member_pairs.append((resolved_members[0], resolved_members[1]))
                unit_pairs.append((resolved_units[0], resolved_units[1]))
    return member_pairs, unit_pairs


def controlled_world_contract(
    group_spec: Mapping[str, Any], member_requests: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Read one group's declared interventions and refuse four separate worlds.

    group_spec is CoreGroupRequest.to_dict() -- its stage_units rows
    say which members each shared unit actually delivers, so this never guesses
    from a unit name. Every member must agree on
    CONTROLLED_WORLD_REQUEST_FIELDS; the visual units must populate the
    world with the same entities; and two visual units that neither swap their
    source slots nor carry a declared motion timing are two arbitrary samples
    rather than one controlled world.
    """
    member_ids = list(group_spec.get("member_request_ids") or ())
    if len(member_ids) != 4:
        raise BindingNativeError(
            f"a controlled core group has four members, got {len(member_ids)}"
        )
    missing = [value for value in member_ids if value not in member_requests]
    if missing:
        raise BindingNativeError(f"member requests are missing for {missing}")
    recipe = dict(group_spec.get("recipe") or {})
    options = _recipe_options(group_spec)
    motion_timing = str(recipe.get("motion_timing") or "none")
    units = list(group_spec.get("stage_units") or ())
    if not units:
        raise BindingNativeError("group_spec carries no stage_units")

    reference_id = member_ids[0]
    reference = member_requests[reference_id]
    differences: list[dict[str, Any]] = []
    shared: dict[str, Any] = {}
    for field_name in CONTROLLED_WORLD_REQUEST_FIELDS:
        shared[field_name] = _world_field(reference, field_name)
        for member_id in member_ids[1:]:
            other = _world_field(member_requests[member_id], field_name)
            if other != shared[field_name]:
                differences.append({
                    "field": field_name, "members": [reference_id, member_id],
                    "values": [_short(shared[field_name]), _short(other)],
                })
    reference_runtime = dict(reference.get("runtime") or {})
    shared_runtime = {key: deepcopy(reference_runtime.get(key))
                      for key in CONTROLLED_WORLD_RUNTIME_KEYS}
    for member_id in member_ids[1:]:
        other_runtime = dict(member_requests[member_id].get("runtime") or {})
        for key in CONTROLLED_WORLD_RUNTIME_KEYS:
            if other_runtime.get(key) != reference_runtime.get(key):
                differences.append({
                    "field": f"runtime.{key}", "members": [reference_id, member_id],
                    "values": [_short(reference_runtime.get(key)), _short(other_runtime.get(key))],
                })
    reference_entities = dict(reference.get("entities") or {})
    for member_id in member_ids[1:]:
        other_entities = dict(member_requests[member_id].get("entities") or {})
        for key in ("total_count", "silent_count"):
            if other_entities.get(key) != reference_entities.get(key):
                differences.append({
                    "field": f"entities.{key}", "members": [reference_id, member_id],
                    "values": [_short(reference_entities.get(key)), _short(other_entities.get(key))],
                })
    shared_audio_policy = normalize_sound_selection_policy(
        reference.get("sound_selection")
    )
    for member_id in member_ids[1:]:
        other_audio_policy = normalize_sound_selection_policy(
            member_requests[member_id].get("sound_selection")
        )
        if other_audio_policy != shared_audio_policy:
            differences.append({
                "field": "sound_selection_policy",
                "members": [reference_id, member_id],
                "values": [_short(shared_audio_policy), _short(other_audio_policy)],
            })
    if differences:
        raise BindingNativeError(
            "core group members do not describe one controlled world; "
            f"differing declarations: {differences}"
        )

    visual_units: dict[str, dict[str, Any]] = {}
    for row in units:
        if (
            row.get("unit_kind") != "visual_capture"
            or row.get("internal_only")
        ):
            continue
        served = list(row.get("member_request_ids") or ())
        if not served:
            raise BindingNativeError(
                f"visual unit {row.get('unit_id')} delivers no member"
            )
        orders = {tuple(member_requests[value].get("source_asset_ids") or ())
                  for value in served}
        if len(orders) != 1:
            raise BindingNativeError(
                f"visual unit {row.get('unit_id')} serves members with different source "
                f"asset orders {sorted(orders)}; one capture is one video"
            )
        order = list(next(iter(orders)))
        if not order or any(not isinstance(value, str) or not value.strip() for value in order):
            raise BindingNativeError(
                f"visual unit {row.get('unit_id')} members declare no source_asset_ids"
            )
        plan_units = [str(name) for name in row.get("depends_on_units") or ()]
        visual_units[str(row["unit_id"])] = {
            "unit_id": str(row["unit_id"]),
            "plan_unit_ids": plan_units,
            "member_request_ids": served,
            "source_asset_ids": order,
        }
    if not visual_units:
        raise BindingNativeError("group_spec declares no visual capture unit")
    populations = {tuple(sorted(entry["source_asset_ids"])) for entry in visual_units.values()}
    if len(populations) != 1:
        raise BindingNativeError(
            "visual units populate the world with different entities "
            f"{sorted(populations)}; that is more than one world, not one intervention"
        )
    orders = {entry["unit_id"]: tuple(entry["source_asset_ids"])
              for entry in visual_units.values()}
    distinct_orders = set(orders.values())
    slot_permutation = len(distinct_orders) > 1
    if (
        len(visual_units) > 1
        and not slot_permutation
        and motion_timing == "none"
        and options["visual_intervention"] == "source_slot_permutation"
    ):
        raise BindingNativeError(
            f"{group_spec.get('group_id')} has {len(visual_units)} visual units with the "
            "same source slot order and no declared source-slot intervention; two samples "
            "of the same declaration are not a controlled intervention"
        )

    columns = _audio_columns(units, visual_units)
    shared_audio_member_ids, shared_audio_unit_pairs = _shared_audio_pairs(
        group_spec, units, columns
    )
    if options["audio_content_scope"] == "shared_audio_pairs":
        audio_differences = []
        for member_left, member_right in shared_audio_member_ids:
            left_request = member_requests[member_left]
            right_request = member_requests[member_right]
            left_content = _declared_audio_content(left_request)
            right_content = _declared_audio_content(right_request)
            if left_content != right_content:
                audio_differences.append({
                    "field": "shared_audio_content",
                    "members": [member_left, member_right],
                    "values": [_short(left_content), _short(right_content)],
                })
            left_candidates = _audio_candidate_by_asset(left_request)
            right_candidates = _audio_candidate_by_asset(right_request)
            if left_candidates is not None and right_candidates is not None:
                for asset_id in sorted(
                    set(left_candidates) | set(right_candidates)
                ):
                    common = set(left_candidates.get(asset_id, ())) & set(
                        right_candidates.get(asset_id, ())
                    )
                    if not common:
                        audio_differences.append({
                            "field": "shared_audio_candidate_intersection",
                            "members": [member_left, member_right],
                            "asset_id": asset_id,
                            "values": [
                                _short(left_candidates.get(asset_id, [])),
                                _short(right_candidates.get(asset_id, [])),
                            ],
                        })
        if audio_differences:
            raise BindingNativeError(
                "declared dry sound content differs within a shared audio column; "
                f"differing declarations: {audio_differences}"
            )

    return {
        "schema": "avengine_native_controlled_world_contract_v1",
        "status": "pass",
        "group_id": group_spec.get("group_id"),
        "task_family": group_spec.get("task_family"),
        "room_id": group_spec.get("room_id"),
        "member_request_ids": member_ids,
        "motion_timing": motion_timing,
        "sampling_candidate_index": shared["sampling_candidate_index"],
        "plan_equivalence": options["plan_equivalence"],
        "visual_intervention": options["visual_intervention"],
        "query_identity_policy": options["query_identity_policy"],
        "audio_content_scope": options["audio_content_scope"],
        "shared_world": shared,
        "shared_runtime": shared_runtime,
        "shared_audio_policy": shared_audio_policy,
        "shared_audio_member_ids": [list(pair) for pair in shared_audio_member_ids],
        "shared_audio_unit_pairs": [list(pair) for pair in shared_audio_unit_pairs],
        "shared_entities": {key: reference_entities.get(key)
                             for key in ("total_count", "silent_count")},
        "visual_units": visual_units,
        "audio_columns": columns,
        "declared_interventions": {
            "visual_source_slot_order": {unit_id: list(value)
                                          for unit_id, value in sorted(orders.items())},
            "visual_slot_permutation": slot_permutation,
            "visual_intervention": options["visual_intervention"],
            "audio_assignment_column": columns,
            "audio_content_scope": options["audio_content_scope"],
            "motion_timing": motion_timing,
        },
        "world_population": list(next(iter(populations))),
        "plan_equivalence_rule": (
            "identical_planned_world_under_declared_slot_identities"
            if options["plan_equivalence"] == "controlled_slots"
            else "identical_planned_scene_clock_and_camera_route_only"
        ),
    }


def _audio_columns(
    units: Sequence[Mapping[str, Any]], visual_units: Mapping[str, Mapping[str, Any]]
) -> dict[str, list[str]]:
    """Group the audio units of every visual unit into shared assignment columns.

    The column of an audio unit is its position among the audio units of its own
    visual unit, ordered by the member it delivers. That is structural, so a
    renamed unit cannot silently re-pair two columns; when a unit id still
    carries an explicit _a<N> suffix it has to agree with that position.
    """
    by_visual: dict[str, list[Mapping[str, Any]]] = {}
    for row in units:
        if (
            row.get("unit_kind") != "audio"
            or row.get("internal_only")
        ):
            continue
        visual_unit_id = row.get("visual_unit_id")
        if visual_unit_id not in visual_units:
            raise BindingNativeError(
                f"audio unit {row.get('unit_id')} consumes unknown visual unit "
                f"{visual_unit_id!r}"
            )
        by_visual.setdefault(str(visual_unit_id), []).append(row)
    if not by_visual:
        raise BindingNativeError("group_spec declares no audio unit")
    sizes = {len(rows) for rows in by_visual.values()}
    if len(sizes) != 1:
        raise BindingNativeError(
            "every visual unit of a controlled group carries the same number of audio "
            f"columns, got {sorted(sizes)}"
        )
    columns: dict[str, list[str]] = {}
    for visual_unit_id, rows in sorted(by_visual.items()):
        ordered = sorted(rows, key=lambda row: row.get("member_index") or 0)
        for position, row in enumerate(ordered):
            unit_id = str(row["unit_id"])
            suffix = unit_id.rsplit("_a", 1)[-1] if "_a" in unit_id else None
            if suffix is not None and suffix.isdigit() and int(suffix) != position:
                raise BindingNativeError(
                    f"audio unit {unit_id} sits at assignment column {position} of "
                    f"{visual_unit_id} but its name declares column {suffix}"
                )
            columns.setdefault(f"a{position}", []).append(unit_id)
    return {name: sorted(value) for name, value in sorted(columns.items())}


def audio_column_of_unit(contract: Mapping[str, Any], unit_id: str) -> str:
    for column, unit_ids in (contract.get("audio_columns") or {}).items():
        if unit_id in unit_ids:
            return str(column)
    raise BindingNativeError(f"{unit_id} is not an audio unit of this group")


def group_stage_context(
    manifest: Mapping[str, Any] | None = None,
    group_id: str | None = None,
    *,
    group: Any = None,
    member_requests: Mapping[str, Mapping[str, Any]] | None = None,
    group_spec: Mapping[str, Any] | None = None,
    retained_visual_roots: Mapping[str, str | Path] | None = None,
    world_id: str | None = None,
    qa_ids: Sequence[str] | None = None,
    sound_pool: str | Path | None = None,
    prepared_manifest: str | Path | None = None,
    split: str = "pilot",
) -> dict[str, Any]:
    """Everything a stage worker needs for one group, read from real inputs.

    Pass the saved batch manifest and a group_id and this rebuilds the group
    through avengine.qa.batch_manifest.core_group_from_manifest, so the unit
    graph, the member order and the per-stage resources are P01's, not a second
    copy. A CoreGroupRequest can also be handed in directly. Nothing here
    touches the filesystem except to check that a declared retained visual root
    exists.
    """
    if group is None and manifest is not None:
        if not group_id:
            raise BindingNativeError("group_stage_context needs a group_id with a manifest")
        from avengine.qa.batch_manifest import core_group_from_manifest
        group = core_group_from_manifest(manifest, group_id)
    if group is not None:
        group_spec = group.to_dict()
        member_requests = {member.request_id: member.to_legacy_request()
                           for member in group.members}
    if group_spec is None or member_requests is None:
        raise BindingNativeError(
            "group_stage_context needs a manifest+group_id, a CoreGroupRequest, "
            "or an explicit group_spec plus member_requests"
        )
    requests = {str(key): deepcopy(dict(value)) for key, value in member_requests.items()}
    if sound_pool is not None:
        resolved_pool = str(_file(sound_pool, base=REPOSITORY, owner="sound pool"))
        for value in requests.values():
            value["sound_pool"] = resolved_pool
    if prepared_manifest is not None:
        resolved_manifest = str(
            _file(prepared_manifest, base=REPOSITORY, owner="prepared manifest")
        )
        for value in requests.values():
            value["prepared_manifest"] = resolved_manifest
    contract = controlled_world_contract(group_spec, requests)
    retained: dict[str, str] = {}
    for unit_id, root in (retained_visual_roots or {}).items():
        resolved = Path(root).expanduser().resolve()
        if not (resolved / "plan/episode_plan.json").is_file():
            raise BindingNativeError(
                f"retained visual root for {unit_id} has no plan/episode_plan.json: {resolved}"
            )
        retained[str(unit_id)] = str(resolved)
    unknown = [unit_id for unit_id in retained if unit_id not in contract["visual_units"]
               and unit_id not in {name for entry in contract["visual_units"].values()
                                   for name in entry["plan_unit_ids"]}]
    if unknown:
        raise BindingNativeError(
            f"retained visual roots name units that are not visual units of this group: {unknown}"
        )
    return {
        "schema": "avengine_native_group_stage_context_v1",
        "group_id": str(group_spec["group_id"]),
        "task_family": str(group_spec["task_family"]),
        "room_id": str(group_spec["room_id"]),
        "world_id": world_id or f"world_{group_spec['group_id']}",
        "world_id_source": "caller" if world_id else "derived_from_group_id",
        "split": split,
        "group_spec": deepcopy(dict(group_spec)),
        "member_requests": requests,
        "contract": contract,
        "retained_visual_roots": retained,
        "qa_ids": None if qa_ids is None else [str(value) for value in qa_ids],
        "repository": str(REPOSITORY.resolve()),
    }


def _query_identity_tokens(plan: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(row["entity_instance_id"]): f"<{actor_id}.entity_instance_id>"
        for actor_id, row in plan_slot_identities(plan).items()
        if isinstance(row.get("entity_instance_id"), str)
        and row.get("entity_instance_id")
    }


def _replace_query_entity_ids(value: Any, tokens: Mapping[str, str]) -> Any:
    if isinstance(value, Mapping):
        return {
            key: (
                tokens.get(item, item)
                if key == "entity_instance_id" and isinstance(item, str)
                else _replace_query_entity_ids(item, tokens)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_query_entity_ids(item, tokens) for item in value]
    return deepcopy(value)


def _normalize_planned_query_identities(
    sampling: Any, plan: Mapping[str, Any]
) -> Any:
    tokens = _query_identity_tokens(plan)
    if not tokens:
        return deepcopy(sampling)

    def visit(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: (
                    _replace_query_entity_ids(item, tokens)
                    if key == "planned_query_window"
                    else visit(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [visit(item) for item in value]
        return deepcopy(value)

    return visit(sampling)


def _world_signature(
    plan: Mapping[str, Any], *, query_identity_policy: str = "exact"
) -> dict[str, Any]:
    """The world facts that remain after the recipe declares its intervention."""
    if query_identity_policy not in QUERY_IDENTITY_POLICIES:
        raise BindingNativeError(
            f"unknown query identity policy: {query_identity_policy!r}"
        )
    visual = plan.get("visual_plan")
    if not isinstance(visual, Mapping) or not isinstance(visual.get("frames"), list):
        raise BindingNativeError("plan lacks visual frames")
    resources = plan.get("resources")
    resources = resources if isinstance(resources, Mapping) else {}
    room_package = resources.get("room_package")
    sampling = _without_audio_content(plan.get("camera_condition_sampling"))
    if query_identity_policy == "slot":
        sampling = _normalize_planned_query_identities(sampling, plan)
    geometry = plan.get("geometry")
    if geometry is None:
        geometry = plan.get("geometry_signature")
    static_geometry = plan.get("static_geometry")
    if static_geometry is None and isinstance(room_package, Mapping):
        static_geometry = room_package.get("static_geometry")
    return {
        "clock": deepcopy(plan.get("clock")),
        "scene": deepcopy(plan.get("scene")),
        "resources_room_package": deepcopy(room_package),
        "condition_profile": deepcopy(plan.get("condition_profile")),
        "sampling_candidate_index": _plan_sampling_candidate_index(plan),
        "coordinate_frame": deepcopy(plan.get("coordinate_frame")),
        "geometry": deepcopy(geometry),
        "static_geometry": deepcopy(static_geometry),
        "camera_condition_sampling": sampling,
        "camera": deepcopy(visual.get("camera")),
        "actor_slots": [
            str(row.get("actor_id"))
            for row in visual.get("actors", [])
            if isinstance(row, Mapping)
        ],
        "camera_route": [
            {
                "frame_index": row.get("frame_index"),
                "pts_ticks": row.get("pts_ticks"),
                "camera_state": deepcopy(row.get("camera_state")),
            }
            for row in visual["frames"]
            if isinstance(row, Mapping)
        ],
    }


def compare_visual_world(
    left_path: str | Path,
    right_path: str | Path,
    *,
    query_identity_policy: str = "exact",
) -> dict[str, Any]:
    """Require two plans to describe one world under a declared query policy."""
    left = _world_signature(
        _load(Path(left_path)), query_identity_policy=query_identity_policy
    )
    right = _world_signature(
        _load(Path(right_path)), query_identity_policy=query_identity_policy
    )
    if left != right:
        differing = sorted(key for key in left if left[key] != right[key])
        raise BindingNativeError(
            f"visual plans describe different worlds; differing fields: {differing}"
        )
    authority = "identical_planned_scene_clock_and_camera_route"
    if query_identity_policy == "slot":
        authority += "_with_declared_query_identity_normalization"
    return {
        "status": "pass",
        "authority": authority,
        "left": str(Path(left_path).resolve()),
        "right": str(Path(right_path).resolve()),
    }


def compare_group_visual_plans(
    left_path: str | Path, right_path: str | Path, *, contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the plan-equivalence and query rule owned by this recipe."""
    if not isinstance(contract, Mapping):
        raise BindingNativeError("group visual comparison needs a recipe contract")
    mode = contract.get("plan_equivalence")
    if mode is None:
        mode = (
            "world"
            if str(contract.get("motion_timing") or "none") != "none"
            else "controlled_slots"
        )
    if mode == "controlled_slots":
        result = compare_controlled_visual_plans(left_path, right_path)
    elif mode == "world":
        result = compare_visual_world(
            left_path,
            right_path,
            query_identity_policy=str(
                contract.get("query_identity_policy") or "exact"
            ),
        )
    else:
        raise BindingNativeError(f"unknown group plan equivalence: {mode!r}")
    return {**result, "rule": contract.get("plan_equivalence_rule")}


def _result_rows(results: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index passing stage results by unit id, newest attempt wins."""
    latest: dict[str, dict[str, Any]] = {}
    for row in results:
        scope = str(row.get("scope_id") or row.get("request_id") or "")
        if "/" not in scope:
            continue
        unit_id = scope.rsplit("/", 1)[-1]
        attempt = int(str(row.get("work_item_id") or "::0").rsplit(":", 1)[-1] or 0)
        current = latest.get(unit_id)
        if current is None or attempt >= int(current["_attempt"]):
            latest[unit_id] = {**dict(row), "_attempt": attempt}
    return {unit_id: row for unit_id, row in latest.items() if row.get("status") == "pass"}


def group_world_equivalence(
    context: Mapping[str, Any], results: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Check the completed visual units really are one controlled world.

    Cheap: it reads the plan JSON and the native neutral readbacks the finished
    units published. It is meant to run before an expensive stage, so a group
    whose two visuals drifted apart is stopped before its captures and audio
    columns are paid for.
    """
    contract = context["contract"]
    done = _result_rows(results)
    visual_units = contract["visual_units"]
    plan_by_unit: dict[str, str] = {}
    readback_by_unit: dict[str, str] = {}
    for unit_id, entry in visual_units.items():
        for plan_unit in entry["plan_unit_ids"]:
            row = done.get(plan_unit)
            if row is not None:
                path = (row.get("facts") or {}).get("episode_plan_path")
                if isinstance(path, str) and path:
                    plan_by_unit[unit_id] = path
        row = done.get(unit_id)
        if row is not None:
            outputs = row.get("outputs") or {}
            if isinstance(outputs.get("neutral_readback"), str):
                readback_by_unit[unit_id] = outputs["neutral_readback"]
            if isinstance(outputs.get("episode_plan"), str):
                plan_by_unit.setdefault(unit_id, outputs["episode_plan"])
    checks: list[dict[str, Any]] = []
    unit_ids = sorted(visual_units)
    anchor = unit_ids[0]
    for unit_id in unit_ids[1:]:
        if anchor in plan_by_unit and unit_id in plan_by_unit:
            try:
                checks.append({
                    "check": "planned_world", "units": [anchor, unit_id],
                    **compare_group_visual_plans(
                        plan_by_unit[anchor], plan_by_unit[unit_id], contract=contract),
                })
            except BindingNativeError as exc:
                checks.append({"check": "planned_world", "units": [anchor, unit_id],
                               "status": "fail", "reason": str(exc)})
        if anchor in readback_by_unit and unit_id in readback_by_unit:
            try:
                checks.append({
                    "check": "native_readback", "units": [anchor, unit_id],
                    **compare_group_native_visuals(
                        {"neutral_readback": readback_by_unit[anchor]},
                        {"neutral_readback": readback_by_unit[unit_id]},
                        contract=contract,
                        left_unit_id=anchor,
                        right_unit_id=unit_id),
                })
            except BindingNativeError as exc:
                checks.append({"check": "native_readback", "units": [anchor, unit_id],
                               "status": "fail", "reason": str(exc)})
    failed = [row for row in checks if row.get("status") != "pass"]
    return {
        "status": "fail" if failed else ("pass" if checks else "not_run"),
        "checks": checks,
        "compared_plan_units": sorted(plan_by_unit),
        "compared_readback_units": sorted(readback_by_unit),
        "rule": contract.get("plan_equivalence_rule"),
        "reason": None if not failed else "; ".join(
            str(row.get("reason")) for row in failed),
    }


def resolve_instance_runtime(
    item: Mapping[str, Any], *, lease: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Per-instance placement for one work item: lease first, then the item.

    A resource lease is the allocator's decision about which device and which
    RPC port this attempt may use, so it wins over the work item's declared
    values; when no lease is handed in, the work item's own resource fields are
    used, and when neither says anything the request keeps what it declared.
    """
    resource = dict(item.get("resource") or {})
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for key in INSTANCE_RUNTIME_KEYS:
        if lease is not None and lease.get(key) is not None:
            values[key] = lease[key]
            sources[key] = "lease"
        elif resource.get(key) is not None:
            values[key] = resource[key]
            sources[key] = "work_item_resource"
        else:
            values[key] = None
            sources[key] = "request"
    return {
        "graphics_adapter": values["graphics_adapter"],
        "rpc_port": values["rpc_port"],
        "rlr_threads": values["rlr_threads"],
        "sources": sources,
        "lease_id": None if lease is None else lease.get("lease_id"),
        "execution": resource.get("execution"),
        "runtime_context": resource.get("runtime_context"),
        "resource_kind": resource.get("kind"),
        "rlr_threads_note": (
            "RLR reads its thread count from the simulation configuration, not an "
            "environment variable; the acoustic entry point places this value"
        ),
    }


def _stage_result(
    item: Mapping[str, Any], *, status: str, facts: Mapping[str, Any] | None = None,
    outputs: Mapping[str, Any] | None = None, reason: str | None = None,
) -> dict[str, Any]:
    """One StageResult mapping, with the identity fields P01 checks."""
    return {
        "schema": GROUP_STAGE_SCHEMA,
        "work_item_id": item["work_item_id"],
        "stage": item["stage"],
        "request_id": item.get("scope_id", item.get("request_id")),
        "scope_id": item.get("scope_id", item.get("request_id")),
        "status": status,
        "facts": deepcopy(dict(facts or {})),
        "outputs": deepcopy(dict(outputs or {})),
        "reason": reason,
        "depends_on": list(item.get("depends_on") or ()),
    }


def _upstream(item: Mapping[str, Any], unit_id: str) -> dict[str, Any]:
    entry = (item.get("inputs") or {}).get(unit_id)
    if not isinstance(entry, Mapping):
        raise BindingNativeError(
            f"{item['work_item_id']} has no input for upstream unit {unit_id!r}"
        )
    return dict(entry)


def _unit_row(context: Mapping[str, Any], unit_id: str) -> dict[str, Any]:
    for row in context["group_spec"].get("stage_units") or ():
        if str(row.get("unit_id")) == unit_id:
            return dict(row)
    raise BindingNativeError(f"{unit_id} is not a unit of {context['group_id']}")


def shared_visual_evidence_root(output_root: str | Path, group_id: str) -> Path:
    """One shared visual evidence pack per group, beside its units."""
    return Path(output_root).expanduser().resolve() / group_id / "shared_visual_evidence"


def load_group_stage_results(
    output_root: str | Path, group_id: str
) -> list[dict[str, Any]]:
    """Read every saved stage result of one group back off disk, in order.

    This is the restore half of save/restore: an interrupted run hands these
    straight back to next_group_work_items and continues from the units
    that actually finished.
    """
    root = Path(output_root).expanduser().resolve() / group_id
    rows: list[tuple[str, int, dict[str, Any]]] = []
    if not root.is_dir():
        return []
    for path in sorted(root.glob(f"*/*/attempt_*/{STAGE_RESULT_FILENAME}")):
        value = _load(path)
        work_item_id = str(value.get("work_item_id") or "")
        try:
            attempt = int(work_item_id.rsplit(":", 1)[-1])
        except ValueError:
            attempt = 0
        rows.append((work_item_id, attempt, value))
    rows.sort(key=lambda entry: (entry[0].rsplit(":", 2)[0], entry[1]))
    return [
        {key: value for key, value in row[2].items() if key != "schema"}
        for row in rows
    ]


def _retained_root_for(context: Mapping[str, Any], unit_id: str) -> Path | None:
    retained = context.get("retained_visual_roots") or {}
    if unit_id in retained:
        return Path(retained[unit_id])
    for capture_unit_id, entry in (context["contract"]["visual_units"]).items():
        plan_units = list(entry["plan_unit_ids"])
        if unit_id == capture_unit_id:
            for plan_unit in plan_units:
                if plan_unit in retained:
                    return Path(retained[plan_unit])
        elif unit_id in plan_units and capture_unit_id in retained:
            return Path(retained[capture_unit_id])
    return None


def _visual_unit_for(context: Mapping[str, Any], unit_id: str) -> dict[str, Any]:
    """The capture unit entry a plan or capture unit belongs to."""
    for capture_unit_id, entry in context["contract"]["visual_units"].items():
        if unit_id == capture_unit_id or unit_id in entry["plan_unit_ids"]:
            return dict(entry)
    raise BindingNativeError(f"{unit_id} is not part of a visual unit of this group")


def _plan_renderer(plan: Mapping[str, Any]) -> str:
    resources = plan.get("resources") if isinstance(plan.get("resources"), Mapping) else {}
    package = (resources.get("room_package")
               if isinstance(resources.get("room_package"), Mapping) else {})
    value = package.get("renderer") or resources.get("renderer") or resources.get("backend")
    if not isinstance(value, str) or not value.strip():
        raise BindingNativeError("episode plan declares no renderer")
    return value.strip().lower()


def _plan_actor_order(plan: Mapping[str, Any]) -> list[str]:
    visual = plan.get("visual_plan") if isinstance(plan.get("visual_plan"), Mapping) else {}
    return [str(actor.get("asset_id")) for actor in visual.get("actors", [])
            if isinstance(actor, Mapping) and actor.get("actor_id") in {"source1", "source2", "source3"}]


def _captured_frame_count(capture: Path, plan: Mapping[str, Any]) -> int:
    readback = _load(capture / "neutral_readback.json")
    rows = readback.get("camera")
    if not isinstance(rows, list) or not rows:
        raise BindingNativeError(f"native capture readback has no camera frames: {capture}")
    declared = (plan.get("clock") or {}).get("frame_count")
    if isinstance(declared, int) and len(rows) != declared:
        raise BindingNativeError(
            f"native capture read back {len(rows)} camera frames but the plan clock "
            f"declares {declared}"
        )
    return len(rows)


def run_visual_plan_unit(
    item: Mapping[str, Any], context: Mapping[str, Any], unit_root: Path,
    *, output_root: str | Path, results: Sequence[Mapping[str, Any]] = (),
    lease: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Plan one shared visual through the real QA planner, or adopt a retained plan."""
    unit_id = str(item["unit_id"])
    entry = _visual_unit_for(context, unit_id)
    runtime = resolve_instance_runtime(item, lease=lease)
    served = list(item.get("member_request_ids") or entry["member_request_ids"])
    base = deepcopy(context["member_requests"][served[0]])
    retained = _retained_root_for(context, unit_id)
    unit_root.mkdir(parents=True)
    if retained is not None:
        adopted = _retained_visual_entry(
            retained, expected_assets=entry["source_asset_ids"], label=unit_id)
        plan_path = Path(adopted["plan"])
        plan = _load(plan_path)
        request_path = retained / "request.json"
        if not request_path.is_file():
            raise BindingNativeError(
                f"the retained visual root for {unit_id} has no request.json, so the world "
                f"it was produced for cannot be checked: {retained}"
            )
        retained_request = _load(request_path)
        retained_check = verify_retained_visual_request(
            retained_request, context["contract"], label=unit_id)
        outputs = {
            "plan_root": str(retained), "episode_plan": str(plan_path),
            "request_path": str(request_path),
            "source_asset_ids": list(entry["source_asset_ids"]),
            "reused_retained_visual_root": str(retained),
            "retained_request_check": retained_check,
            "instance_runtime": runtime,
        }
    else:
        request = build_variant_request(
            base,
            episode_id=f"{context['group_id']}_{unit_id}",
            source_asset_ids=entry["source_asset_ids"],
            rpc_port=runtime["rpc_port"],
            graphics_adapter=runtime["graphics_adapter"],
            qa_ids=context.get("qa_ids"),
            seed=base.get("seed"),
        )
        request_path = _write(unit_root / "request.json", request)
        planned = plan_visual_variant(
            request_path, unit_root / "episode", label=unit_id,
            log=unit_root / f"{unit_id}.plan.log")
        plan_path = Path(planned["plan"])
        plan = _load(plan_path)
        planner_sources = _planner_sources()
        outputs = {
            "plan_root": planned["output"], "episode_plan": str(plan_path),
            "request_path": str(request_path),
            "source_asset_ids": list(entry["source_asset_ids"]),
            "reused_retained_visual_root": None,
            "instance_runtime": runtime,
            "planner": planner_sources,
            "process": planned["process"],
            "binding_variant": request.get("binding_variant"),
        }
    planned_order = _plan_actor_order(plan)
    if planned_order != list(entry["source_asset_ids"]):
        raise BindingNativeError(
            f"{unit_id} planned actor asset order {planned_order} differs from the "
            f"declared intervention {entry['source_asset_ids']}"
        )
    facts = {
        "episode_plan_path": str(plan_path),
        "renderer": _plan_renderer(plan),
        "clock": deepcopy(plan.get("clock")),
    }
    equivalence = group_world_equivalence(
        context, list(results) + [_stage_result(item, status="pass", facts=dict(facts),
                                                outputs=outputs)])
    facts["world_equivalence"] = equivalence
    if equivalence["status"] == "fail":
        raise BindingNativeError(
            f"{unit_id} does not plan the same controlled world as its siblings: "
            f"{equivalence['reason']}"
        )
    return _stage_result(item, status="pass", facts=facts, outputs=outputs)


def _planner_sources() -> dict[str, Any]:
    """Where the planning code actually came from, read back from the modules."""
    sources: dict[str, Any] = {
        "planner_entrypoint": str(REPOSITORY / "tools/studio/run_qa_episode.py"),
    }
    for name in ("avengine.rooms.conditioned_sampler", "avengine.rooms.qa_episode",
                 "avengine.rooms.native_qa_room", "avengine.rooms.room_providers"):
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
        sources[name] = None if spec is None else spec.origin
    return sources


def run_visual_capture_unit(
    item: Mapping[str, Any], context: Mapping[str, Any], unit_root: Path,
    *, output_root: str | Path, results: Sequence[Mapping[str, Any]] = (),
    lease: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture one shared visual once, or link a validated retained capture."""
    unit_id = str(item["unit_id"])
    unit_spec = _unit_row(context, unit_id)
    entry = _visual_unit_for(context, unit_id)
    runtime = resolve_instance_runtime(item, lease=lease)
    plan_units = [str(name) for name in unit_spec.get("depends_on_units") or ()]
    if len(plan_units) != 1:
        raise BindingNativeError(
            f"{unit_id} must depend on exactly one visual plan unit, got {plan_units}"
        )
    upstream = _upstream(item, plan_units[0])
    plan_path = Path((upstream.get("facts") or {})["episode_plan_path"])
    if not plan_path.is_file():
        raise BindingNativeError(f"planned episode is unavailable: {plan_path}")
    equivalence = group_world_equivalence(context, results)
    if equivalence["status"] == "fail":
        raise BindingNativeError(
            f"refusing to capture {unit_id}: the finished visual units are not one "
            f"controlled world ({equivalence['reason']})"
        )
    retained = _retained_root_for(context, unit_id)
    if unit_root.exists():
        # A recipe may have prepared this attempt's upstream plan before
        # delegating capture. Accept only that referenced preparation subtree;
        # an old capture or unrelated contents must still refuse no-clobber.
        try:
            prepared = plan_path.resolve().relative_to(unit_root.resolve())
        except ValueError:
            raise FileExistsError(f"refusing existing capture attempt: {unit_root}")
        if (
            not unit_root.is_dir()
            or len(prepared.parts) < 2
            or {child.name for child in unit_root.iterdir()} != {prepared.parts[0]}
        ):
            raise FileExistsError(f"refusing existing capture attempt: {unit_root}")
    unit_root.mkdir(parents=True, exist_ok=True)
    if retained is not None:
        adopted = _retained_visual_entry(
            retained, expected_assets=entry["source_asset_ids"], label=unit_id)
        verify_retained_visual_request(
            _load(retained / "request.json"), context["contract"], label=unit_id)
        retained_plan = Path(adopted["plan"]).resolve()
        if retained_plan != plan_path.resolve():
            compare_group_visual_plans(
                plan_path, retained_plan, contract=context["contract"])
        captured = {
            "output": adopted["output"], "plan": adopted["plan"],
            "capture": adopted["capture"], "visual_video": adopted["visual_video"],
            "frame_readbacks": adopted["frame_readbacks"],
            "neutral_readback": adopted["neutral_readback"],
        }
        native_worlds = 0
        reuse = {"reused_retained_visual_root": str(retained),
                 "validated": ["static_camera", "declared_asset_order",
                               "neutral_readback_contract", "required_capture_files",
                               "plan_equivalence_against_this_unit_plan"]}
    else:
        request_value = (upstream.get("outputs") or {}).get("request_path")
        if not isinstance(request_value, str) or not Path(request_value).is_file():
            raise BindingNativeError(
                f"{unit_id} cannot capture without the planned request its plan unit saved"
            )
        request = _load(Path(request_value))
        if runtime["graphics_adapter"] is not None:
            request.setdefault("runtime", {})
            request["runtime"] = {**dict(request.get("runtime") or {}),
                                  "graphics_adapter": int(runtime["graphics_adapter"])}
        if runtime["rpc_port"] is not None:
            request["runtime"] = {**dict(request.get("runtime") or {}),
                                  "rpc_port": int(runtime["rpc_port"])}
        (unit_root / "plan").symlink_to(plan_path.parent, target_is_directory=True)
        _write(unit_root / "capture_request.json", request)
        captured = capture_visual_plan(
            request, unit_root, label=unit_id,
            log=unit_root / f"{unit_id}.capture.log")
        native_worlds = 1
        reuse = {"reused_retained_visual_root": None, "validated": ["required_capture_files"]}
    capture_dir = Path(captured["capture"])
    plan = _load(plan_path)
    frame_count = _captured_frame_count(capture_dir, plan)
    facts = {
        "capture_receipt_path": str((capture_dir / "research_receipt.json").resolve()),
        "captured_frame_count": frame_count,
    }
    outputs = {
        "capture": str(capture_dir.resolve()),
        "capture_root": str(Path(captured["output"]).resolve()),
        "episode_plan": str(plan_path.resolve()),
        "request_path": (upstream.get("outputs") or {}).get("request_path"),
        "neutral_readback": str(Path(captured["neutral_readback"]).resolve()),
        "frame_readbacks": str(Path(captured["frame_readbacks"]).resolve()),
        "visual_video": captured.get("visual_video"),
        "source_asset_ids": list(entry["source_asset_ids"]),
        "member_request_ids": list(item.get("member_request_ids") or ()),
        "native_visual_worlds_created": native_worlds,
        "instance_runtime": runtime,
        "reuse": reuse,
    }
    after = group_world_equivalence(
        context, list(results) + [_stage_result(item, status="pass", facts=dict(facts),
                                                outputs=outputs)])
    facts["world_equivalence"] = after
    if after["status"] == "fail":
        raise BindingNativeError(
            f"{unit_id} native readback does not match its sibling visual: {after['reason']}"
        )
    return _stage_result(item, status="pass", facts=facts, outputs=outputs)


# Rebound rows carry one record that names the visual asset a sound was checked
# against. That is the declared visual intervention, so it differs between two
# members on purpose and is not an acoustic input. Everything else is compared,
# so a field that does affect the render blocks reuse even if it is new here.
AUDIO_REUSE_VISUAL_ONLY_FIELDS = frozenset({"target_sound_compatibility"})


def audio_render_inputs(assignment_plan: Mapping[str, Any]) -> dict[str, Any]:
    """The rebound events and bindings that actually drive the acoustic render."""
    def strip(rows: Any) -> list[dict[str, Any]]:
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise BindingNativeError("an assignment plan needs event and binding lists")
        return [
            {key: deepcopy(value) for key, value in row.items()
             if key not in AUDIO_REUSE_VISUAL_ONLY_FIELDS}
            for row in rows if isinstance(row, Mapping)
        ]

    return {
        "audio_events": strip(assignment_plan.get("audio_events")),
        "voice_bindings": strip(assignment_plan.get("voice_bindings")),
    }


_SHARED_AUDIO_INTERVENTION_FIELDS = frozenset({
    "actor_id",
    "source_endpoint_id",
    "voice_binding_actor_id",
    "assignment_variant",
    "target_sound_compatibility",
    "entity_instance_id",
    "instance_event_id",
})


def audio_shared_content_signature(
    assignment_plan: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Fields consumed by the audio renderer for one shared assignment column."""
    rendered = audio_render_inputs(assignment_plan)

    def canonical(rows: Any) -> list[dict[str, Any]]:
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise BindingNativeError(
                "an assignment plan needs event and binding lists for shared audio checks"
            )
        result = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            result.append({
                str(key): deepcopy(value)
                for key, value in row.items()
                if key not in _SHARED_AUDIO_INTERVENTION_FIELDS
            })
        return sorted(
            result,
            key=lambda row: json.dumps(
                row, ensure_ascii=False, sort_keys=True, default=str
            ),
        )

    return {
        "audio_events": canonical(rendered["audio_events"]),
        "voice_bindings": canonical(rendered["voice_bindings"]),
    }


def _audio_reuse_candidate(
    context: Mapping[str, Any], item: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]], *, column: str,
    assignment_plan: Mapping[str, Any], acoustics: Mapping[str, Any],
    neutral_readback: str,
) -> dict[str, Any]:
    """Reuse a finished audio column only after re-checking this run's inputs.

    Two members of a controlled group differ in their video, never in their
    acoustic input, so one rendered column legitimately serves both. The file
    existing is not the evidence: the acoustic package, HRTF, stride, gain,
    clock and policy must match, the native camera/clock/emitter readbacks must
    still agree, and the rebound events and bindings must be identical.
    """
    unit_id = str(item["unit_id"])
    siblings = [name for name in (context["contract"]["audio_columns"].get(column) or ())
                if name != unit_id]
    done = _result_rows(results)
    rejected: list[dict[str, Any]] = []
    for sibling in siblings:
        row = done.get(sibling)
        if row is None:
            rejected.append({"unit_id": sibling, "reason": "not finished in this round"})
            continue
        outputs = row.get("outputs") or {}
        report = (row.get("facts") or {}).get("audio_report_path")
        if not isinstance(report, str) or not Path(report).is_file():
            rejected.append({"unit_id": sibling, "reason": "no readable audio report"})
            continue
        if outputs.get("acoustic_identity") != dict(acoustics):
            rejected.append({"unit_id": sibling,
                             "reason": "acoustic input or configuration identity differs"})
            continue
        sibling_readback = outputs.get("neutral_readback")
        if not isinstance(sibling_readback, str) or not Path(sibling_readback).is_file():
            rejected.append({"unit_id": sibling, "reason": "no readable native readback"})
            continue
        try:
            readback_equivalence = compare_native_visuals(
                {"neutral_readback": sibling_readback},
                {"neutral_readback": neutral_readback},
            )
        except BindingNativeError as exc:
            rejected.append({"unit_id": sibling, "reason": str(exc)})
            continue
        sibling_plan_path = outputs.get("assignment_plan_path")
        if not isinstance(sibling_plan_path, str) or not Path(sibling_plan_path).is_file():
            rejected.append({"unit_id": sibling, "reason": "no readable assignment plan"})
            continue
        sibling_plan = _load(Path(sibling_plan_path))
        mine = audio_render_inputs(assignment_plan)
        theirs = audio_render_inputs(sibling_plan)
        if mine != theirs:
            rejected.append({"unit_id": sibling,
                             "reason": "rebound audio events or voice bindings differ"})
            continue
        return {
            "reused": True, "source_unit_id": sibling, "audio_report": report,
            "native_readback_equivalence": readback_equivalence,
            "checked": ["acoustic_identity", "native_camera_clock_emitter_readback",
                        "rebound_events_and_bindings", "report_file_present"],
            "compared_ignoring_fields": sorted(AUDIO_REUSE_VISUAL_ONLY_FIELDS),
            "rejected": rejected,
        }
    return {"reused": False, "source_unit_id": None, "audio_report": None,
            "rejected": rejected}


def run_audio_unit(
    item: Mapping[str, Any], context: Mapping[str, Any], unit_root: Path,
    *, output_root: str | Path, results: Sequence[Mapping[str, Any]] = (),
    lease: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Render and finalize one audio column of one shared visual."""
    unit_id = str(item["unit_id"])
    unit_spec = _unit_row(context, unit_id)
    runtime = resolve_instance_runtime(item, lease=lease)
    capture_unit_id = str(unit_spec.get("visual_unit_id") or "")
    if not capture_unit_id:
        raise BindingNativeError(f"{unit_id} declares no visual_unit_id")
    upstream = _upstream(item, capture_unit_id)
    outputs_up = upstream.get("outputs") or {}
    capture_dir = Path(str(outputs_up["capture"]))
    plan_path = Path(str(outputs_up["episode_plan"]))
    request_value = outputs_up.get("request_path")
    if not isinstance(request_value, str) or not Path(request_value).is_file():
        raise BindingNativeError(
            f"{unit_id} needs the planned request its visual unit saved; got {request_value!r}"
        )
    capture_request = _load(Path(request_value))
    member_request_id, member_request = _member_request_for_audio_unit(
        context, item, unit_spec
    )
    request, audio_view_fields = _apply_member_audio_view_fields(
        capture_request, member_request, label=unit_id
    )
    plan = _load(plan_path)
    column = audio_column_of_unit(context["contract"], unit_id)
    endpoint_by_actor = _neutral_endpoint_bindings(
        str(outputs_up["neutral_readback"]), plan=plan)
    assignment_plan, assignment_request = build_audio_assignment_plan(
        plan, request, column,
        endpoint_by_actor=endpoint_by_actor,
        require_authoritative_endpoints=True,
    )
    acoustics = acoustic_identity(assignment_request, plan)
    declared_delivery = declared_audio_delivery(assignment_request)
    reuse = _audio_reuse_candidate(
        context, item, results, column=column, assignment_plan=assignment_plan,
        acoustics=acoustics, neutral_readback=str(outputs_up["neutral_readback"]),
    )
    unit_root.mkdir(parents=True)
    assignment_plan_path = _write(unit_root / "assignment_plan.json", assignment_plan)
    assignment_request_path = _write(
        unit_root / "assignment_request.json", assignment_request
    )
    variant_root = materialize_audio_variant(
        {"capture": str(capture_dir), "visual_video": outputs_up.get("visual_video")},
        unit_root / "episode", assignment_plan, assignment_request, member_id=unit_id,
    )
    shared_visual = shared_visual_evidence_root(output_root, context["group_id"])
    finalized = finalize_audio_assignment(
        variant_root, assignment_request,
        audio_report=reuse["audio_report"],
        shared_visual_root=shared_visual,
    )
    facts_path = Path(finalized["facts"])
    delivered_facts = _load(facts_path)
    audio_facts = delivered_facts.get("audio")
    intervals = (audio_facts or {}).get("wet_tail_intervals")
    if not isinstance(intervals, list) or not intervals:
        raise BindingNativeError(
            f"{unit_id} delivery published no measured wet_tail_intervals: {facts_path}"
        )
    facts = {
        "facts_path": str(facts_path),
        "audio_report_path": finalized["audio_report"],
        "wet_tail_intervals": deepcopy(intervals),
    }
    outputs = {
        "variant_root": str(variant_root),
        "audio": finalized["audio"],
        "questions": finalized["questions"],
        "visual_video": finalized["visual_video"] or outputs_up.get("visual_video"),
        "preview": finalized["preview"],
        "assignment_plan_path": str(assignment_plan_path),
        "assignment_request_path": str(assignment_request_path),
        "assignment_column": column,
        "visual_unit_id": capture_unit_id,
        "member_request_id": member_request_id,
        "audio_view_fields": audio_view_fields,
        "capture_request_path": request_value,
        "capture": str(capture_dir),
        "neutral_readback": str(outputs_up["neutral_readback"]),
        "acoustic_identity": acoustics,
        "declared_audio_delivery": declared_delivery,
        "delivered_audio_layouts": finalized["delivered_audio_layouts"],
        "ancillary_audio_outputs": deepcopy(
            (finalized.get("result") or {}).get("ancillary_audio_outputs")
        ),
        "shared_audio_column": reuse,
        "shared_visual_root": str(shared_visual),
        "shared_visual_evidence": (finalized["result"] or {}).get("shared_visual_evidence"),
        "stage_timings": (finalized["result"] or {}).get("stage_timings"),
        "member_request_ids": list(item.get("member_request_ids") or ()),
        "instance_runtime": runtime,
        "elapsed_s": finalized["elapsed_s"],
    }
    layouts = finalized["delivered_audio_layouts"]
    if layouts.get("status") != "pass":
        return _stage_result(item, status="blocked", facts=facts, outputs=outputs,
                             reason=layouts.get("reason"))
    return _stage_result(item, status="pass", facts=facts, outputs=outputs)


def _relation_query_value(value: Any, *, owner: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BindingNativeError(
            f"{owner} must declare reference_time_s, appearance_values and window_s"
        )
    reference = value.get("reference_time_s")
    if isinstance(reference, bool) or not isinstance(reference, int):
        raise BindingNativeError(f"{owner}.reference_time_s must be an integer")
    appearances = value.get("appearance_values")
    if (
        isinstance(appearances, (str, bytes))
        or not isinstance(appearances, Sequence)
        or len(appearances) != 2
        or any(not isinstance(item, str) or not item.strip() for item in appearances)
        or len(set(str(item) for item in appearances)) != 2
    ):
        raise BindingNativeError(
            f"{owner}.appearance_values must contain two distinct nonempty strings"
        )
    window = value.get("window_s", value.get("query_window_s"))
    if (
        isinstance(window, (str, bytes))
        or not isinstance(window, Sequence)
        or len(window) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in window)
        or int(window[1]) <= int(window[0])
    ):
        raise BindingNativeError(
            f"{owner}.window_s must contain two increasing integer seconds"
        )
    return {
        "reference_time_s": int(reference),
        "appearance_values": [str(item) for item in appearances],
        "window_s": [int(window[0]), int(window[1])],
    }


def _relation_query_from_context(
    context: Mapping[str, Any], item: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    candidates: list[tuple[str, Any]] = []
    for key in ("relation_query", "query"):
        if key in context:
            candidates.append((f"context.{key}", context.get(key)))
    contract = context.get("contract")
    if isinstance(contract, Mapping):
        for key in ("relation_query", "query"):
            if key in contract:
                candidates.append((f"contract.{key}", contract.get(key)))
    group_spec = context.get("group_spec")
    if isinstance(group_spec, Mapping):
        for key in ("relation_query", "query"):
            if key in group_spec:
                candidates.append((f"group_spec.{key}", group_spec.get(key)))
    if item is not None:
        payload = item.get("payload")
        if isinstance(payload, Mapping):
            for key in ("relation_query", "query"):
                if key in payload:
                    candidates.append((f"item.payload.{key}", payload.get(key)))
    requests = context.get("member_requests")
    if isinstance(requests, Mapping):
        for member_id, request in requests.items():
            if not isinstance(request, Mapping):
                continue
            for key in ("relation_query", "query"):
                if key in request:
                    candidates.append(
                        (f"member_requests.{member_id}.{key}", request.get(key))
                    )
            targets = request.get("qa_targets")
            if isinstance(targets, Sequence) and not isinstance(targets, (str, bytes)):
                for target in targets:
                    if not isinstance(target, Mapping):
                        continue
                    event = target.get("event")
                    if not isinstance(event, Mapping) or "window_s" not in event:
                        continue
                    derived = {
                        "reference_time_s": request.get("reference_time_s", 0),
                        "appearance_values": request.get("appearance_values"),
                        "window_s": event.get("window_s"),
                    }
                    if derived["appearance_values"] is not None:
                        candidates.append(
                            (f"member_requests.{member_id}.qa_targets", derived)
                        )
    if not candidates:
        raise BindingNativeError(
            "visual_conditioned_relation needs an explicit relation query with "
            "reference_time_s, appearance_values and window_s"
        )
    for owner, candidate in candidates:
        return _relation_query_value(candidate, owner=owner)
    raise BindingNativeError(
        "visual_conditioned_relation relation query is present but invalid"
    )


def relation_group_stage_context(
    *args: Any,
    relation_query: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build the native group context and attach relation query semantics."""
    context = group_stage_context(*args, **kwargs)
    context["relation_query"] = _relation_query_value(
        relation_query, owner="relation_query"
    ) if relation_query is not None else _relation_query_from_context(context)
    return context


def _relation_source_order(
    context: Mapping[str, Any], visual_unit_id: str
) -> tuple[str, ...]:
    entry = (context.get("contract") or {}).get("visual_units", {}).get(visual_unit_id)
    values = entry.get("source_asset_ids") if isinstance(entry, Mapping) else None
    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, Sequence)
        or len(values) != 3
        or len(set(values)) != 3
        or any(not isinstance(value, str) or not value.strip() for value in values)
    ):
        raise BindingNativeError(
            "visual_conditioned_relation stage requires exactly three distinct source assets"
        )
    return tuple(str(value) for value in values)


def run_relation_visual_plan_unit(
    item: Mapping[str, Any], context: Mapping[str, Any], unit_root: Path,
    *, output_root: str | Path, results: Sequence[Mapping[str, Any]] = (),
    lease: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    unit_id = str(item.get("unit_id") or "")
    capture_id = unit_id if unit_id.endswith("_capture") else f"{unit_id}_capture"
    _relation_source_order(context, capture_id)
    _relation_query_from_context(context, item)
    return run_visual_plan_unit(
        item, context, unit_root, output_root=output_root,
        results=results, lease=lease,
    )


def run_relation_visual_capture_unit(
    item: Mapping[str, Any], context: Mapping[str, Any], unit_root: Path,
    *, output_root: str | Path, results: Sequence[Mapping[str, Any]] = (),
    lease: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    unit_id = str(item.get("unit_id") or "")
    _relation_source_order(context, unit_id)
    _relation_query_from_context(context, item)
    return run_visual_capture_unit(
        item, context, unit_root, output_root=output_root,
        results=results, lease=lease,
    )


def _relation_schedule_seed(
    plan: Mapping[str, Any], request: Mapping[str, Any]
) -> int:
    value = request.get("seed", plan.get("seed", 0))
    if isinstance(value, bool) or not isinstance(value, int):
        raise BindingNativeError(
            "visual_conditioned_relation schedule requires an integer seed"
        )
    return int(value)


def _relation_schedule_from_mapping(value: Any) -> tuple[tuple[float, ...], float | None] | None:
    if not isinstance(value, Mapping):
        return None
    starts = value.get("start_times_s_by_actor")
    if not isinstance(starts, Mapping):
        return None
    actors = sorted(str(key) for key in starts)
    if actors != ["source1", "source2", "source3"]:
        raise BindingNativeError(
            "visual_conditioned_relation schedule must name source1, source2 and source3"
        )
    try:
        values = tuple(float(starts[actor]) for actor in actors)
    except (TypeError, ValueError) as exc:
        raise BindingNativeError(
            "visual_conditioned_relation schedule start times are invalid"
        ) from exc
    reserve = value.get("effective_reserve_tail_s")
    if reserve is not None:
        if isinstance(reserve, bool) or not isinstance(reserve, (int, float)):
            raise BindingNativeError(
                "visual_conditioned_relation schedule reserve tail is invalid"
            )
        reserve = float(reserve)
    return values, reserve


def _relation_schedule_inputs(
    results: Sequence[Mapping[str, Any]],
    assignment: str,
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
) -> tuple[tuple[float, ...] | None, float | None]:
    done = _result_rows(results)
    for row in done.values():
        outputs = row.get("outputs") or {}
        if outputs.get("assignment_column") != assignment:
            continue
        schedule = _relation_schedule_from_mapping(
            outputs.get("relation_schedule")
        )
        if schedule is not None:
            return schedule
    for value in (
        plan.get("audio_schedule"),
        request.get("audio_schedule"),
    ):
        schedule = _relation_schedule_from_mapping(value)
        if schedule is not None:
            return schedule
    return None, None


def run_relation_audio_unit(
    item: Mapping[str, Any], context: Mapping[str, Any], unit_root: Path,
    *, output_root: str | Path, results: Sequence[Mapping[str, Any]] = (),
    lease: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Schedule and finalize one relation assignment through the native path."""
    unit_id = str(item["unit_id"])
    unit_spec = _unit_row(context, unit_id)
    runtime = resolve_instance_runtime(item, lease=lease)
    capture_unit_id = str(unit_spec.get("visual_unit_id") or "")
    if not capture_unit_id:
        raise BindingNativeError(f"{unit_id} declares no visual_unit_id")
    _relation_source_order(context, capture_unit_id)
    query = _relation_query_from_context(context, item)
    upstream = _upstream(item, capture_unit_id)
    outputs_up = upstream.get("outputs") or {}
    capture_dir = Path(str(outputs_up["capture"]))
    plan_path = Path(str(outputs_up["episode_plan"]))
    request_value = outputs_up.get("request_path")
    if not isinstance(request_value, str) or not Path(request_value).is_file():
        raise BindingNativeError(
            f"{unit_id} needs the planned request its visual unit saved; got {request_value!r}"
        )
    capture_request = _load(Path(request_value))
    member_request_id, member_request = _member_request_for_audio_unit(
        context, item, unit_spec
    )
    request, audio_view_fields = _apply_member_audio_view_fields(
        capture_request, member_request, label=unit_id
    )
    plan = _load(plan_path)
    assignment = unit_id.rsplit("_", 1)[-1]
    if assignment not in {"a0", "a1"}:
        raise BindingNativeError(f"unsupported relation audio assignment {assignment!r}")
    start_times, reserve_tail = _relation_schedule_inputs(
        results, assignment, plan, request
    )
    scheduled_plan, scheduled_request, schedule = schedule_relation_audio_plan(
        plan,
        request,
        start_times_s=start_times,
        reserve_tail_s=reserve_tail,
        query_window_s=query["window_s"],
        rng_seed=_relation_schedule_seed(plan, request),
    )
    scheduled_plan["relation_query"] = deepcopy(query)
    scheduled_request["relation_query"] = deepcopy(query)
    endpoint_by_actor = _neutral_endpoint_bindings(
        str(outputs_up["neutral_readback"]), plan=scheduled_plan
    )
    assignment_plan, assignment_request = build_audio_assignment_plan(
        scheduled_plan,
        scheduled_request,
        assignment,
        assignment_targets=_RELATION_ASSIGNMENT_TARGETS,
        expected_event_count=3,
        endpoint_by_actor=endpoint_by_actor,
        require_authoritative_endpoints=True,
    )
    acoustics = acoustic_identity(assignment_request, assignment_plan)
    declared_delivery = declared_audio_delivery(assignment_request)
    reuse = _audio_reuse_candidate(
        context,
        item,
        results,
        column=audio_column_of_unit(context["contract"], unit_id),
        assignment_plan=assignment_plan,
        acoustics=acoustics,
        neutral_readback=str(outputs_up["neutral_readback"]),
    )
    unit_root.mkdir(parents=True)
    scheduled_plan_path = _write(
        unit_root / "scheduled_episode_plan.json", scheduled_plan
    )
    scheduled_request_path = _write(
        unit_root / "scheduled_request.json", scheduled_request
    )
    assignment_plan_path = _write(
        unit_root / "assignment_plan.json", assignment_plan
    )
    assignment_request_path = _write(
        unit_root / "assignment_request.json", assignment_request
    )
    variant_root = materialize_audio_variant(
        {"capture": str(capture_dir), "visual_video": outputs_up.get("visual_video")},
        unit_root / "episode",
        assignment_plan,
        assignment_request,
        member_id=unit_id,
    )
    shared_visual = shared_visual_evidence_root(output_root, context["group_id"])
    finalized = finalize_audio_assignment(
        variant_root,
        assignment_request,
        audio_report=reuse["audio_report"],
        shared_visual_root=shared_visual,
    )
    visual_video = finalized.get("visual_video") or outputs_up.get("visual_video")
    if (
        not isinstance(visual_video, str)
        or not Path(visual_video).expanduser().resolve().is_file()
    ):
        raise BindingNativeError(
            f"{unit_id} audio finalization published no readable visual_video"
        )
    facts_path = Path(finalized["facts"]).expanduser().resolve()
    delivered_facts = _load(facts_path)
    audio_facts = delivered_facts.get("audio")
    intervals = (audio_facts or {}).get("wet_tail_intervals")
    if not isinstance(intervals, list) or not intervals:
        raise BindingNativeError(
            f"{unit_id} delivery published no measured wet_tail_intervals: {facts_path}"
        )
    facts = {
        "facts_path": str(facts_path),
        "audio_report_path": str(Path(finalized["audio_report"]).resolve()),
        "wet_tail_intervals": deepcopy(intervals),
        "relation_query": deepcopy(query),
        "relation_schedule": deepcopy(schedule),
    }
    outputs = {
        "variant_root": str(variant_root),
        "audio": finalized["audio"],
        "questions": finalized["questions"],
        "visual_video": visual_video,
        "preview": finalized["preview"],
        "capture": str(capture_dir),
        "neutral_readback": str(outputs_up["neutral_readback"]),
        "episode_plan": str(plan_path),
        "request_path": str(scheduled_request_path),
        "capture_request_path": request_value,
        "scheduled_episode_plan_path": str(scheduled_plan_path),
        "scheduled_request_path": str(scheduled_request_path),
        "assignment_plan_path": str(assignment_plan_path),
        "assignment_request_path": str(assignment_request_path),
        "assignment_column": assignment,
        "visual_unit_id": capture_unit_id,
        "member_request_id": member_request_id,
        "member_request_ids": list(item.get("member_request_ids") or ()),
        "audio_view_fields": audio_view_fields,
        "relation_query": deepcopy(query),
        "relation_schedule": deepcopy(schedule),
        "acoustic_identity": acoustics,
        "declared_audio_delivery": declared_delivery,
        "delivered_audio_layouts": finalized["delivered_audio_layouts"],
        "ancillary_audio_outputs": deepcopy(
            (finalized.get("result") or {}).get("ancillary_audio_outputs")
        ),
        "shared_audio_column": reuse,
        "shared_visual_root": str(shared_visual),
        "shared_visual_evidence": (finalized["result"] or {}).get("shared_visual_evidence"),
        "stage_timings": (finalized["result"] or {}).get("stage_timings"),
        "source_asset_ids": list(_relation_source_order(context, capture_unit_id)),
        "instance_runtime": runtime,
        "elapsed_s": finalized["elapsed_s"],
        "native_visual_worlds_created": 0,
    }
    layouts = finalized["delivered_audio_layouts"]
    if layouts.get("status") != "pass":
        return _stage_result(
            item, status="blocked", facts=facts, outputs=outputs,
            reason=layouts.get("reason"),
        )
    return _stage_result(item, status="pass", facts=facts, outputs=outputs)


def _relation_file_shared(
    left: Any, right: Any, *, label: str
) -> dict[str, Any]:
    if not isinstance(left, str) or not isinstance(right, str):
        raise BindingNativeError(f"{label} paths must be strings")
    left_path = Path(left).expanduser().resolve()
    right_path = Path(right).expanduser().resolve()
    if not left_path.is_file() or not right_path.is_file():
        raise BindingNativeError(
            f"{label} files are unavailable: {left_path}, {right_path}"
        )
    try:
        same = os.path.samefile(left_path, right_path)
    except OSError:
        same = False
    if not same:
        try:
            same = filecmp.cmp(left_path, right_path, shallow=False)
        except OSError:
            same = False
    if not same:
        raise BindingNativeError(
            f"{label} PCM/RGB files differ: {left_path}, {right_path}"
        )
    return {
        "status": "pass",
        "left": str(left_path),
        "right": str(right_path),
        "samefile_or_bytewise_equal": True,
    }


def _relation_audio_rows(
    context: Mapping[str, Any],
    item: Mapping[str, Any],
    done: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[tuple[str, str, str]]]:
    variants: dict[str, dict[str, Any]] = {}
    assignment_plans: dict[str, dict[str, Any]] = {}
    member_units: list[tuple[str, str, str]] = []
    for column, unit_ids in sorted(context["contract"]["audio_columns"].items()):
        for audio_unit_id in unit_ids:
            row = done.get(str(audio_unit_id))
            if row is None:
                raise BindingNativeError(
                    f"{audio_unit_id} has no passing relation audio stage result"
                )
            outputs = row.get("outputs") or {}
            facts = row.get("facts") or {}
            ancillary = outputs.get("ancillary_audio_outputs")
            if ancillary:
                raise BindingNativeError(
                    f"{audio_unit_id} published ancillary_audio_outputs, but "
                    "_relation_group_spec/assemble_binding_dataset has no "
                    "attached-view mapping; R01 must export this declaration "
                    "before relation assembly"
                )
            plan_path = outputs.get("assignment_plan_path")
            if (
                not isinstance(plan_path, str)
                or not Path(plan_path).expanduser().resolve().is_file()
            ):
                raise BindingNativeError(
                    f"{audio_unit_id} has no readable assignment plan"
                )
            assignment_plans[str(audio_unit_id)] = _load(Path(plan_path))
            for key in ("facts_path", "audio_report_path"):
                if (
                    not isinstance(facts.get(key), str)
                    or not Path(facts[key]).expanduser().resolve().is_file()
                ):
                    raise BindingNativeError(
                        f"{audio_unit_id} facts lack readable {key}"
                    )
            for key in ("audio", "visual_video", "capture"):
                if (
                    not isinstance(outputs.get(key), str)
                    or not Path(outputs[key]).expanduser().resolve().exists()
                ):
                    raise BindingNativeError(
                        f"{audio_unit_id} outputs lack readable {key}"
                    )
            visual_unit_id = str(outputs.get("visual_unit_id") or "")
            if not visual_unit_id:
                raise BindingNativeError(
                    f"{audio_unit_id} relation audio output has no visual unit"
                )
            variants[str(audio_unit_id)] = {
                "facts": str(Path(facts["facts_path"]).resolve()),
                "audio": str(Path(outputs["audio"]).resolve()),
                "visual_video": str(Path(outputs["visual_video"]).resolve()),
                "audio_report": str(Path(facts["audio_report_path"]).resolve()),
                "visual_capture_root": str(Path(outputs["capture"]).resolve()),
            }
            member_units.append(
                (str(audio_unit_id), visual_unit_id, str(audio_unit_id))
            )
    return variants, assignment_plans, member_units


def run_relation_assembly_unit(
    item: Mapping[str, Any], context: Mapping[str, Any], unit_root: Path,
    *, output_root: str | Path, results: Sequence[Mapping[str, Any]] = (),
    lease: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble relation media while preserving shared PCM/RGB and query meaning."""
    del output_root, lease
    contract = context["contract"]
    _relation_query_from_context(context, item)
    _relation_source_order(context, sorted(contract["visual_units"])[0])
    visual_ids = sorted(str(value) for value in contract["visual_units"])
    if len(visual_ids) != 2:
        raise BindingNativeError(
            f"visual_conditioned_relation assembly needs two visual units, got {visual_ids}"
        )
    done = _result_rows(results)
    visual: dict[str, dict[str, Any]] = {}
    plans: dict[str, str] = {}
    readbacks: dict[str, str] = {}
    visual_requests: dict[str, dict[str, Any]] = {}
    for visual_id in visual_ids:
        row = done.get(visual_id)
        if row is None:
            raise BindingNativeError(
                f"{visual_id} has no passing relation visual capture result"
            )
        outputs = row.get("outputs") or {}
        for key in ("capture", "episode_plan", "neutral_readback", "request_path"):
            if (
                not isinstance(outputs.get(key), str)
                or not Path(outputs[key]).expanduser().resolve().exists()
            ):
                raise BindingNativeError(
                    f"{visual_id} output lacks readable {key}"
                )
        visual[visual_id] = {
            "capture": str(Path(outputs["capture"]).resolve()),
            "visual_video": outputs.get("visual_video"),
            "neutral_readback": str(Path(outputs["neutral_readback"]).resolve()),
        }
        plans[visual_id] = str(Path(outputs["episode_plan"]).resolve())
        readbacks[visual_id] = str(Path(outputs["neutral_readback"]).resolve())
        visual_requests[visual_id] = _load(Path(outputs["request_path"]))
    variants, assignment_plans, member_units = _relation_audio_rows(
        context, item, done
    )
    query = _relation_query_from_context(context, item)
    checks: list[dict[str, Any]] = []
    for left_id, right_id in [(visual_ids[0], visual_ids[1])]:
        checks.append({
            "check": "planned_world",
            "units": [left_id, right_id],
            **compare_group_visual_plans(
                plans[left_id], plans[right_id], contract=contract
            ),
        })
        checks.append({
            "check": "native_readback",
            "units": [left_id, right_id],
            **compare_group_native_visuals(
                {"neutral_readback": readbacks[left_id]},
                {"neutral_readback": readbacks[right_id]},
                contract=contract,
                left_unit_id=left_id,
                right_unit_id=right_id,
            ),
        })
        left_plan = _load(Path(plans[left_id]))
        right_plan = _load(Path(plans[right_id]))
        left_endpoints = _neutral_endpoint_bindings(
            readbacks[left_id], plan=left_plan
        )
        right_endpoints = _neutral_endpoint_bindings(
            readbacks[right_id], plan=right_plan
        )
        if left_endpoints != right_endpoints:
            raise BindingNativeError(
                "relation visual variants expose different native source endpoints"
            )
        checks.append({
            "check": "native_source_endpoints",
            "units": [left_id, right_id],
            "status": "pass",
            "same": True,
        })
        left_audio_request = None
        right_audio_request = None
        for audio_unit_id, visual_unit_id, _variant in member_units:
            if visual_unit_id == left_id and left_audio_request is None:
                row = done[audio_unit_id]
                path = (row.get("outputs") or {}).get("scheduled_request_path")
                left_audio_request = (
                    _load(Path(path)) if isinstance(path, str) else None
                )
            if visual_unit_id == right_id and right_audio_request is None:
                row = done[audio_unit_id]
                path = (row.get("outputs") or {}).get("scheduled_request_path")
                right_audio_request = (
                    _load(Path(path)) if isinstance(path, str) else None
                )
        if not isinstance(left_audio_request, Mapping) or not isinstance(
            right_audio_request, Mapping
        ):
            raise BindingNativeError(
                "relation audio stages did not publish scheduled requests"
            )
        left_identity = acoustic_identity(left_audio_request, left_plan)
        right_identity = acoustic_identity(right_audio_request, right_plan)
        if left_identity != right_identity:
            raise BindingNativeError(
                "relation acoustic input/configuration identity differs between visuals"
            )
        checks.append({
            "check": "acoustic_identity",
            "units": [left_id, right_id],
            "status": "pass",
            "same": True,
        })
    shared_audio_checks: list[dict[str, Any]] = []
    shared_pairs = contract.get("shared_audio_unit_pairs") or ()
    if not shared_pairs:
        shared_pairs = [
            tuple(str(value) for value in ids)
            for ids in contract.get("audio_columns", {}).values()
            if len(ids) == 2
        ]
    for pair in shared_pairs:
        if len(pair) != 2:
            raise BindingNativeError(
                f"relation shared audio unit pair is invalid: {pair}"
            )
        left_id, right_id = (str(pair[0]), str(pair[1]))
        left_signature = audio_shared_content_signature(
            assignment_plans[left_id]
        )
        right_signature = audio_shared_content_signature(
            assignment_plans[right_id]
        )
        if left_signature != right_signature:
            raise BindingNativeError(
                f"relation shared audio content differs between {left_id} and {right_id}"
            )
        shared_audio_checks.append({
            "units": [left_id, right_id],
            "content": {"status": "pass", "signature": left_signature},
            "pcm": _relation_file_shared(
                variants[left_id]["audio"],
                variants[right_id]["audio"],
                label=f"relation shared audio {left_id}/{right_id}",
            ),
        })
    shared_rgb_checks: list[dict[str, Any]] = []
    for visual_id in visual_ids:
        members = [
            member_id for member_id, visual_unit_id, _variant in member_units
            if visual_unit_id == visual_id
        ]
        if len(members) != 2:
            raise BindingNativeError(
                f"relation visual {visual_id} needs two audio members, got {members}"
            )
        roots = {variants[member_id]["visual_capture_root"] for member_id in members}
        if len(roots) != 1:
            raise BindingNativeError(
                f"relation visual {visual_id} audio members do not share one capture root"
            )
        shared_rgb_checks.append({
            "visual_unit": visual_id,
            "capture_root": next(iter(roots)),
            "video": _relation_file_shared(
                variants[members[0]]["visual_video"],
                variants[members[1]]["visual_video"],
                label=f"relation shared RGB {visual_id}",
            ),
        })
    anchor_visual = visual_ids[0]
    anchor_audio = next(
        member_id for member_id, visual_unit_id, _variant in member_units
        if visual_unit_id == anchor_visual
    )
    anchor_request = visual_requests[anchor_visual]
    anchor_plan = _load(Path(plans[anchor_visual]))
    schedule = next(
        (
            (done[member_id].get("outputs") or {}).get("relation_schedule")
            for member_id, _visual_unit_id, _variant in member_units
            if isinstance(
                (done[member_id].get("outputs") or {}).get("relation_schedule"),
                Mapping,
            )
        ),
        None,
    )
    if not isinstance(schedule, Mapping):
        raise BindingNativeError("relation audio stages published no schedule")
    profile = {
        "task_family": RELATION_TASK_FAMILY,
        "source_count": 3,
        "camera": deepcopy(
            (anchor_plan.get("visual_plan") or {}).get("camera") or {}
        ),
        "audio_schedule": deepcopy(dict(schedule)),
        "reserve_tail_s": schedule.get("effective_reserve_tail_s"),
        "source_profile_reserve_tail_s": schedule.get(
            "source_profile_reserve_tail_s"
        ),
        "schedule_selection": schedule.get("selection_mode"),
        "query_window_s": list(query["window_s"]),
    }
    spec = _relation_group_spec(
        context["group_id"],
        context["world_id"],
        room_family_from_plan(anchor_plan),
        context["room_id"],
        query,
        variants,
        request=anchor_request,
        profile=profile,
    )
    unit_root.mkdir(parents=True)
    spec_path = _write(unit_root / "group_spec.json", spec)
    from avengine.qa.binding_groups import assemble_binding_dataset
    assembled = assemble_binding_dataset(
        spec,
        input_base=REPOSITORY,
        output=unit_root / "assembled",
        seed=f"{context['group_id']}-relation-assembly",
        verify_media=True,
    )
    group_validation = (
        assembled.get("groups", [{}])[0].get("validation")
        if assembled.get("groups")
        else None
    )
    comparisons = (
        group_validation.get("comparisons")
        if isinstance(group_validation, Mapping)
        else None
    )
    if (
        not isinstance(comparisons, list)
        or len(comparisons) != 4
        or any(
            not isinstance(row, Mapping)
            or row.get("answer_relation") != "different"
            or row.get("media_check") != "pass"
            for row in comparisons
        )
    ):
        raise BindingNativeError(
            "relation assembly did not preserve four different-answer media checks"
        )
    answer_checks = [
        {
            "members": list(row.get("members") or ()),
            "shared_modality": row.get("shared_modality"),
            "answer_relation": row.get("answer_relation"),
            "media_check": row.get("media_check"),
        }
        for row in comparisons
    ]
    facts = {
        "group_spec_path": str(spec_path),
        "assembled_path": str((unit_root / "assembled").resolve()),
        "relation_query": deepcopy(query),
        "validation": {
            "status": "pass",
            "planned_and_native_checks": checks,
            "shared_audio_checks": shared_audio_checks,
            "shared_rgb_checks": shared_rgb_checks,
            "answer_relation_checks": answer_checks,
            "group_count": assembled.get("group_count"),
            "world_count": assembled.get("world_count"),
            "sample_count": assembled.get("sample_count"),
            "media_validation": assembled.get("validation"),
            "group_validation": deepcopy(group_validation),
            "public_payload_check": deepcopy(
                assembled.get("public_payload_check")
            ),
        },
    }
    outputs = {
        "assembled_root": str((unit_root / "assembled").resolve()),
        "group_spec": str(spec_path),
        "relation_query": deepcopy(query),
        "shared_audio_checks": shared_audio_checks,
        "shared_rgb_checks": shared_rgb_checks,
        "answer_relation_checks": answer_checks,
        "assembled_summary": {
            key: assembled[key]
            for key in sorted(assembled)
            if not isinstance(assembled[key], (list, dict))
        },
        "member_sample_ids": {
            str(row.get("member_id")): row.get("sample_id")
            for group in assembled.get("groups", [])
            for row in group.get("members", [])
        },
        "member_units": [list(row) for row in member_units],
        "world_id": context["world_id"],
        "world_id_source": context["world_id_source"],
        "native_visual_worlds_created": sum(
            int(
                (done[visual_id].get("outputs") or {}).get(
                    "native_visual_worlds_created"
                )
                or 0
            )
            for visual_id in visual_ids
        ),
    }
    return _stage_result(item, status="pass", facts=facts, outputs=outputs)


RELATION_STAGE_RUNNERS = {
    "visual_plan": run_relation_visual_plan_unit,
    "visual_capture": run_relation_visual_capture_unit,
    "audio": run_relation_audio_unit,
    "assembly": run_relation_assembly_unit,
}


def run_relation_group_stage_work_item(
    item: Mapping[str, Any], context: Mapping[str, Any], *,
    output_root: str | Path,
    results: Sequence[Mapping[str, Any]] = (),
    lease: Mapping[str, Any] | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Run relation units through native's fresh/resume/save dispatcher."""
    return run_group_stage_work_item(
        item,
        context,
        output_root=output_root,
        results=results,
        lease=lease,
        resume=resume,
        stage_runners=RELATION_STAGE_RUNNERS,
    )


def run_assembly_unit(
    item: Mapping[str, Any], context: Mapping[str, Any], unit_root: Path,
    *, output_root: str | Path, results: Sequence[Mapping[str, Any]] = (),
    lease: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the finished units and verify the group on its actual media."""
    from avengine.qa.binding_groups import assemble_binding_dataset
    contract = context["contract"]
    visual_units = contract["visual_units"]
    visual: dict[str, dict[str, Any]] = {}
    plans: dict[str, str] = {}
    readbacks: dict[str, str] = {}
    requests: dict[str, dict[str, Any]] = {}
    for unit_id in sorted(visual_units):
        outputs = _upstream(item, unit_id).get("outputs") or {}
        visual[unit_id] = {
            "capture": str(outputs["capture"]),
            "visual_video": outputs.get("visual_video"),
            "neutral_readback": str(outputs["neutral_readback"]),
        }
        plans[unit_id] = str(outputs["episode_plan"])
        readbacks[unit_id] = str(outputs["neutral_readback"])
        requests[unit_id] = _load(Path(str(outputs["request_path"])))
    member_units: list[tuple[str, str, str]] = []
    variants: dict[str, dict[str, Any]] = {}
    audio_delivery_by_member: dict[str, dict[str, Any]] = {}
    assignment_plans: dict[str, dict[str, Any]] = {}
    for column, unit_ids in sorted(contract["audio_columns"].items()):
        for audio_unit_id in unit_ids:
            outputs = _upstream(item, audio_unit_id).get("outputs") or {}
            facts = _upstream(item, audio_unit_id).get("facts") or {}
            assignment_plan_path = outputs.get("assignment_plan_path")
            if (
                not isinstance(assignment_plan_path, str)
                or not Path(assignment_plan_path).is_file()
            ):
                raise BindingNativeError(
                    f"{audio_unit_id} has no readable assignment plan for shared-audio "
                    "content verification"
                )
            assignment_plans[audio_unit_id] = _load(Path(assignment_plan_path))
            visual_unit_id = str(outputs["visual_unit_id"])
            audio_request_path = (
                outputs.get("assignment_request_path")
                or outputs.get("request_path")
            )
            if (
                not isinstance(audio_request_path, str)
                or not Path(audio_request_path).expanduser().resolve().is_file()
            ):
                raise BindingNativeError(
                    f"{audio_unit_id} has no readable assignment request for "
                    "audio layout validation"
                )
            audio_request = _load(Path(audio_request_path).expanduser().resolve())
            declared_delivery = declared_audio_delivery(audio_request)
            ancillary = deepcopy(outputs.get("ancillary_audio_outputs"))
            delivered_layouts = outputs.get("delivered_audio_layouts")
            validated_layouts = (
                deepcopy(dict(delivered_layouts))
                if isinstance(delivered_layouts, Mapping)
                and delivered_layouts.get("status") == "pass"
                else verify_delivered_audio_layouts(
                    declared_delivery,
                    {"ancillary_audio_outputs": ancillary},
                )
            )
            if validated_layouts.get("status") != "pass":
                raise BindingNativeError(
                    f"{audio_unit_id} audio layout delivery is not valid: "
                    f"{validated_layouts.get('reason') or validated_layouts}"
                )
            audio_report_path = (
                facts.get("audio_report_path") or outputs.get("audio_report")
            )
            if (
                not isinstance(audio_report_path, str)
                or not Path(audio_report_path).expanduser().resolve().is_file()
            ):
                raise BindingNativeError(
                    f"{audio_unit_id} has no readable finalized audio report"
                )
            audio_delivery_by_member[audio_unit_id] = {
                "audio_report_path": str(
                    Path(audio_report_path).expanduser().resolve()
                ),
                "assignment_request_path": str(
                    Path(audio_request_path).expanduser().resolve()
                ),
                "primary_audio_path": str(
                    Path(outputs["audio"]).expanduser().resolve()
                ),
                "declared_audio_delivery": deepcopy(declared_delivery),
                "delivered_audio_layouts": deepcopy(validated_layouts),
                "ancillary_audio_outputs": deepcopy(ancillary),
            }
            variants[audio_unit_id] = {
                "facts": str(facts["facts_path"]),
                "audio": str(outputs["audio"]),
                "visual_video": outputs.get("visual_video"),
                "audio_report": str(audio_report_path),
                "visual_capture_root": str(outputs["capture"]),
                "audio_delivery": audio_delivery_by_member[audio_unit_id],
            }
            member_units.append((audio_unit_id, visual_unit_id, audio_unit_id))
    checks: list[dict[str, Any]] = []
    if contract.get("audio_content_scope") == "shared_audio_pairs":
        for left_unit_id, right_unit_id in contract.get(
            "shared_audio_unit_pairs", ()
        ):
            left_signature = audio_shared_content_signature(
                assignment_plans[str(left_unit_id)]
            )
            right_signature = audio_shared_content_signature(
                assignment_plans[str(right_unit_id)]
            )
            if left_signature != right_signature:
                raise BindingNativeError(
                    "shared audio column dry content or timing differs between "
                    f"{left_unit_id} and {right_unit_id}"
                )
            checks.append({
                "check": "shared_audio_content",
                "units": [str(left_unit_id), str(right_unit_id)],
                "status": "pass",
                "compared_fields": [
                    "event_id",
                    "sound_asset_id",
                    "path",
                    "sample_count",
                    "start_sample",
                    "end_sample_exclusive",
                    "start_tick",
                    "end_tick",
                    "source_start_sample",
                    "source_end_sample_exclusive",
                ],
                "ignored_declared_intervention_fields": sorted(
                    _SHARED_AUDIO_INTERVENTION_FIELDS
                ),
            })
    anchor = sorted(visual_units)[0]
    for unit_id in sorted(visual_units)[1:]:
        checks.append({"check": "planned_world", "units": [anchor, unit_id],
                       **compare_group_visual_plans(plans[anchor], plans[unit_id],
                                                    contract=contract)})
        checks.append({"check": "native_readback", "units": [anchor, unit_id],
                       **compare_group_native_visuals(
                           {"neutral_readback": readbacks[anchor]},
                           {"neutral_readback": readbacks[unit_id]},
                           contract=contract,
                           left_unit_id=anchor,
                           right_unit_id=unit_id)})
        left = acoustic_identity(requests[anchor], _load(Path(plans[anchor])))
        right = acoustic_identity(requests[unit_id], _load(Path(plans[unit_id])))
        if left != right:
            raise BindingNativeError(
                f"acoustic input/configuration identity differs between {anchor} and {unit_id}"
            )
        checks.append({"check": "acoustic_identity", "units": [anchor, unit_id],
                       "status": "pass", "same": True})
        endpoints_left = _neutral_endpoint_bindings(readbacks[anchor],
                                                    plan=_load(Path(plans[anchor])))
        endpoints_right = _neutral_endpoint_bindings(readbacks[unit_id],
                                                     plan=_load(Path(plans[unit_id])))
        if endpoints_left != endpoints_right:
            raise BindingNativeError(
                f"native source endpoint identities differ between {anchor} and {unit_id}"
            )
        checks.append({"check": "native_source_endpoints", "units": [anchor, unit_id],
                       "status": "pass", "same": True})
    room_family = room_family_from_plan(_load(Path(plans[anchor])))
    unit_root.mkdir(parents=True)
    anchor_request = requests[anchor]
    anchor_plan = _load(Path(plans[anchor]))
    profile = {
        "task_family": context["task_family"],
        "source_count": len(contract["world_population"]),
        "camera": deepcopy((anchor_plan.get("visual_plan") or {}).get("camera", {})),
        "camera_motion": "static",
        "audio": {
            "rir_stride": anchor_request.get("rir_stride"),
            "post_assembly_convolution_gain": anchor_request.get(
                "post_assembly_convolution_gain"),
            "declared_delivery": declared_audio_delivery(anchor_request),
        },
        "reserve_tail_s": (anchor_request.get("profile") or {}).get("reserve_tail_s"),
        "sound_pool": anchor_request.get("sound_pool"),
        "motion_timing": contract["motion_timing"],
    }
    spec = _group_spec(
        context["group_id"], context["world_id"], room_family, context["room_id"],
        visual, variants, request=anchor_request, profile=profile,
        member_units=member_units, split=context.get("split", "pilot"),
        task_family=context["task_family"],
    )
    spec_path = _write(unit_root / "group_spec.json", spec)
    assembled = assemble_binding_dataset(
        spec, input_base=REPOSITORY, output=unit_root / "assembled",
        seed=f"{context['group_id']}-assembly", verify_media=True,
    )
    facts = {
        "group_spec_path": str(spec_path),
        "assembled_path": str((unit_root / "assembled").resolve()),
        "validation": {
            "status": "pass",
            "controlled_world_checks": checks,
            "group_count": assembled.get("group_count"),
            "world_count": assembled.get("world_count"),
            "sample_count": assembled.get("sample_count"),
            "media_validation": assembled.get("validation"),
            "group_validation": [deepcopy(row.get("validation"))
                                  for row in assembled.get("groups", [])],
            "audio_delivery_by_member": deepcopy(audio_delivery_by_member),
            "public_payload_check": deepcopy(assembled.get("public_payload_check")),
        },
    }
    outputs = {
        "assembled_root": str((unit_root / "assembled").resolve()),
        "group_spec": str(spec_path),
        "assembled_summary": {key: assembled[key] for key in sorted(assembled)
                               if not isinstance(assembled[key], (list, dict))},
        "member_sample_ids": {str(row.get("member_id")): row.get("sample_id")
                               for group in assembled.get("groups", [])
                               for row in group.get("members", [])},
        "member_units": [list(row) for row in member_units],
        "audio_delivery_by_member": deepcopy(audio_delivery_by_member),
        "world_id": context["world_id"],
        "world_id_source": context["world_id_source"],
        "native_visual_worlds_created": sum(
            int((_upstream(item, unit_id).get("outputs") or {}).get(
                "native_visual_worlds_created") or 0)
            for unit_id in sorted(visual_units)
        ),
        "instance_runtime": resolve_instance_runtime(item, lease=lease),
    }
    return _stage_result(item, status="pass", facts=facts, outputs=outputs)


STAGE_RUNNERS = {
    "visual_plan": run_visual_plan_unit,
    "visual_capture": run_visual_capture_unit,
    "audio": run_audio_unit,
    "assembly": run_assembly_unit,
}


def run_group_stage_work_item(
    item: Mapping[str, Any], context: Mapping[str, Any], *,
    output_root: str | Path,
    results: Sequence[Mapping[str, Any]] = (),
    lease: Mapping[str, Any] | None = None,
    resume: bool = True,
    stage_runners: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one shared unit of a controlled group and save its stage result.

    This is what an ordinary program worker calls for each item that
    next_group_work_items hands out. It writes the result next to the unit's
    fresh output so an interrupted run can restore it with
    load_group_stage_results and continue; it never replaces an existing
    attempt, so a half-written attempt is reported and retried under a new
    attempt number instead of being silently adopted.
    """
    item = dict(item)
    if str(item.get("group_id")) != context["group_id"]:
        raise BindingNativeError(
            f"{item.get('work_item_id')} belongs to group {item.get('group_id')!r}, "
            f"not {context['group_id']!r}"
        )
    unit_id = item.get("unit_id")
    if not isinstance(unit_id, str) or not unit_id:
        raise BindingNativeError(
            f"{item.get('work_item_id')} is not a shared group unit; an Episode row has no "
            "group schedule"
        )
    unit_spec = _unit_row(context, unit_id)
    kind = str(unit_spec.get("unit_kind"))
    runners = STAGE_RUNNERS if stage_runners is None else stage_runners
    runner = runners.get(kind)
    if runner is None:
        raise BindingNativeError(f"no stage runner for unit kind {kind!r}")
    if unit_spec.get("stage") != item.get("stage"):
        raise BindingNativeError(
            f"{item['work_item_id']} is a {item.get('stage')} item filed against the "
            f"{unit_spec.get('stage')} unit {unit_id}"
        )
    root = Path(output_root).expanduser().resolve()
    unit_root = root / str(item["fresh_output_relative"])
    saved = unit_root / STAGE_RESULT_FILENAME
    if saved.is_file():
        if not resume:
            raise BindingNativeError(f"refusing to rerun a saved attempt: {saved}")
        return {key: value for key, value in _load(saved).items() if key != "schema"}
    if unit_root.exists() or unit_root.is_symlink():
        return _stage_result(
            item, status="fail",
            reason=(f"attempt output {unit_root} already exists without a saved stage "
                    "result; an interrupted attempt is kept for diagnosis and this unit "
                    "needs a new attempt"),
            outputs={"partial_attempt_root": str(unit_root)},
        )
    try:
        result = runner(item, context, unit_root, output_root=root,
                        results=results, lease=lease)
    except Exception as exc:
        result = _stage_result(
            item, status="fail", reason=f"{type(exc).__name__}: {exc}",
            outputs={"attempt_root": str(unit_root),
                     "traceback": traceback.format_exc()},
        )
    _write(saved, {"schema": GROUP_STAGE_SCHEMA, **result})
    return result


__all__ = [
    "BindingNativeError", "ROOM_FAMILIES", "TASK_FAMILY",
    "RELATION_TASK_FAMILY",
    "build_variant_request", "plan_visual_variant", "capture_visual_plan",
    "compare_visual_plans", "compare_controlled_visual_plans",
    "plan_slot_identities", "compare_native_visuals",
    "compare_group_native_visuals", "room_family_from_plan",
    "acoustic_identity",
    "build_audio_assignment_plan", "materialize_audio_variant",
    "finalize_audio_assignment", "finalize_audio_assignments",
    "recover_rendered_audio_attempt", "schedule_relation_audio_plan",
    "prepare_visible_binding_group", "prepare_visual_conditioned_relation_group",
    "verify_materialized_audio_root", "AUDIO_ROOT_REQUIRED_FILES",
    "audio_render_inputs", "audio_shared_content_signature",
    "AUDIO_REUSE_VISUAL_ONLY_FIELDS",
    "verify_retained_visual_request", "WORLD_FIELD_DEFAULTS",
    "RETAINED_REQUEST_INTERVENTION_FIELDS",
    "RETAINED_VISUAL_AUDIO_VIEW_FIELDS",
    "declared_audio_delivery", "verify_delivered_audio_layouts",
    "GROUP_STAGE_SCHEMA", "STAGE_RESULT_FILENAME",
    "CONTROLLED_WORLD_REQUEST_FIELDS", "CONTROLLED_WORLD_RUNTIME_KEYS",
    "INSTANCE_RUNTIME_KEYS",
    "controlled_world_contract", "audio_column_of_unit", "group_stage_context",
    "compare_visual_world", "compare_group_visual_plans", "group_world_equivalence",
    "resolve_instance_runtime", "shared_visual_evidence_root",
    "load_group_stage_results", "run_group_stage_work_item", "STAGE_RUNNERS",
    "run_visual_plan_unit", "run_visual_capture_unit", "run_audio_unit",
    "run_assembly_unit", "relation_group_stage_context",
    "RELATION_STAGE_RUNNERS", "run_relation_group_stage_work_item",
    "run_relation_visual_plan_unit", "run_relation_visual_capture_unit",
    "run_relation_audio_unit", "run_relation_assembly_unit",
]




if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "--parallel-audio-worker":
    raise SystemExit(_parallel_audio_worker(sys.argv[2], sys.argv[3]))
