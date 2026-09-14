#!/usr/bin/env python3
"""Prepare the first native four-member visible-binding group.

Two modes. ``whole-group`` runs the original single-call preparation.
``stage-run`` drives the same work as the separate shared units that P01's
``GROUP_RECIPES`` declares: an ordinary loop asks
``next_group_work_items`` what may run, executes each unit through
``run_group_stage_work_item``, saves its stage result and continues. It can be
interrupted and restarted -- the saved results are read back with
``load_group_stage_results`` and the finished units are not redone. No agent
and no per-sample human step is on that path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.dataset.binding_group_native import (
    BindingNativeError,
    group_stage_context,
    load_group_stage_results,
    prepare_visible_binding_group,
    run_group_stage_work_item,
)
from avengine.dataset.production_spec import (
    CoreGroupRequest,
    ProductionSpecError,
    group_blockers,
    initial_group_work_items,
    next_group_work_items,
    production_request_from_legacy,
)


def _key_value(values, *, owner):
    result = {}
    for entry in values or ():
        if "=" not in entry:
            raise SystemExit(f"{owner} must be given as key=value, got {entry!r}")
        key, value = entry.split("=", 1)
        if not key.strip() or not value.strip():
            raise SystemExit(f"{owner} must be given as key=value, got {entry!r}")
        result[key.strip()] = value.strip()
    return result


def _group_from_config(path: Path) -> CoreGroupRequest:
    """Build one core group from a small declarative config.

    The config names the base request, the four members and each member's
    selected source assets. Nothing about the room, the assets or the question
    types is hard-coded here; every value is read from the config or the base
    request it points at.
    """
    config = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    base_path = Path(config["base_request"]).expanduser()
    if not base_path.is_absolute():
        base_path = (Path(path).expanduser().resolve().parent / base_path).resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    group_id = str(config["group_id"])
    task_family = str(config["task_family"])
    members = []
    for row in config["members"]:
        value = dict(base)
        value.update({key: row[key] for key in row if key != "request_id"})
        value["episode_id"] = str(row["request_id"])
        value["task_family"] = task_family
        value["group_id"] = group_id
        for key, extra in (config.get("request_overrides") or {}).items():
            value[key] = extra
        members.append(production_request_from_legacy(
            value, request_id=str(row["request_id"]), kind="core_group_member"))
    return CoreGroupRequest(
        group_id=group_id, task_family=task_family,
        room_id=str(config.get("room_id") or base["room_id"]),
        members=tuple(members),
    )


def _stage_run(args) -> int:
    lease = {}
    if args.graphics_adapter is not None:
        lease["graphics_adapter"] = args.graphics_adapter
    if args.rpc_port is not None:
        lease["rpc_port"] = args.rpc_port
    if args.rlr_threads is not None:
        lease["rlr_threads"] = args.rlr_threads
    if args.lease_id:
        lease["lease_id"] = args.lease_id
    if args.lease:
        lease.update(json.loads(Path(args.lease).expanduser().read_text(encoding="utf-8")))
    manifest = None
    group = None
    if args.manifest is not None:
        manifest = json.loads(Path(args.manifest).expanduser().read_text(encoding="utf-8"))
        from avengine.qa.batch_manifest import core_group_from_manifest
        group = core_group_from_manifest(manifest, args.group_id)
    else:
        group = _group_from_config(Path(args.group_config))
    context = group_stage_context(
        group=group,
        retained_visual_roots=_key_value(args.retained_visual_root,
                                         owner="--retained-visual-root"),
        world_id=args.world_id,
        qa_ids=args.qa_ids,
        sound_pool=args.sound_pool,
        prepared_manifest=args.prepared_manifest,
        split=args.split,
    )
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results = load_group_stage_results(output_root, context["group_id"])
    restored = [row["work_item_id"] for row in results]
    executed, rounds = [], 0
    while rounds < args.max_rounds:
        rounds += 1
        items = (initial_group_work_items(group) if not results
                 else next_group_work_items(group, [
                     _stage_result_obj(row) for row in results]))
        if not items:
            break
        stop = False
        for item in items:
            result = run_group_stage_work_item(
                item.to_dict(), context, output_root=output_root,
                results=results, lease=lease or None,
            )
            results.append(result)
            executed.append({"work_item_id": result["work_item_id"],
                             "status": result["status"], "reason": result["reason"]})
            if result["status"] != "pass":
                stop = True
        if stop:
            break
    blockers = group_blockers(group, [_stage_result_obj(row) for row in results])
    summary = {
        "schema": "avengine_native_group_stage_run_v1",
        "group_id": context["group_id"],
        "task_family": context["task_family"],
        "room_id": context["room_id"],
        "world_id": context["world_id"],
        "output_root": str(output_root),
        "restored_work_item_ids": restored,
        "executed": executed,
        "rounds": rounds,
        "declared_interventions": context["contract"]["declared_interventions"],
        "native_visual_worlds_created": sum(
            int((row.get("outputs") or {}).get("native_visual_worlds_created") or 0)
            for row in results if row.get("stage") == "capture"),
        "blockers": blockers,
        "status": ("pass" if all(row["status"] == "pass" for row in executed)
                   and not blockers and executed else
                   "pass_restored" if not executed and not blockers else "fail"),
    }
    for index in range(1, 1000):
        record = output_root / f"{context['group_id']}_stage_run_{index:03d}.json"
        if not record.exists():
            break
    else:
        raise SystemExit(f"too many saved stage-run records under {output_root}")
    summary["record"] = str(record)
    record.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] != "fail" else 1


def _stage_result_obj(row):
    from avengine.dataset.production_spec import StageResult
    return StageResult.from_mapping(row, owner="stage_result")


def _whole_group(args, parser) -> int:
    try:
        summary = prepare_visible_binding_group(
            base_request_path=args.base_request,
            output_root=args.output_root,
            first_visual_capture_root=args.first_visual_capture_root,
            second_visual_capture_root=args.second_visual_capture_root,
            sound_pool=args.sound_pool,
            prepared_manifest=args.prepared_manifest,
            source_asset_ids=(
                None
                if args.source1_asset_id is None and args.source2_asset_id is None
                else (args.source1_asset_id, args.source2_asset_id)
            ),
            room_id=args.room_id,
            rpc_port=args.rpc_port,
            graphics_adapter=args.graphics_adapter,
            group_id=args.group_id,
            world_id=args.world_id,
            qa_ids=args.qa_ids,
            seed=args.seed,
            group_question=args.group_question,
        )
    except (BindingNativeError, ProductionSpecError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "status": summary["status"],
        "group_question": summary.get("group_question", {}).get("qa_id"),
        "visual_invariance": {
            key: value for key, value in (summary.get("visual_invariance") or {}).items()
            if key in ("status", "frames_compared_per_member",
                       "actor_states_compared_per_member",
                       "speech_animation_channels_found")
        },
        "group_spec": summary["group_spec"],
        "plan_equivalence": summary["plan_equivalence"],
        "native_readback_equivalence": summary["native_readback_equivalence"],
        "shared_audio_by_column": summary["shared_audio_by_column"],
    }, ensure_ascii=False, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("whole-group", "stage-run"),
                        default="whole-group")
    parser.add_argument("--base-request", type=Path)
    parser.add_argument("--first-visual-capture-root", type=Path)
    parser.add_argument("--second-visual-capture-root", type=Path)
    parser.add_argument("--sound-pool", type=Path)
    parser.add_argument("--prepared-manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source1-asset-id")
    parser.add_argument("--source2-asset-id")
    parser.add_argument("--room-id")
    parser.add_argument("--rpc-port", type=int)
    parser.add_argument("--graphics-adapter", type=int)
    parser.add_argument("--rlr-threads", type=int)
    parser.add_argument("--lease", type=Path,
                        help="JSON file with the resource lease this attempt may use")
    parser.add_argument("--lease-id")
    parser.add_argument("--group-id", default="visible_binding_group_v0")
    parser.add_argument("--world-id")
    parser.add_argument("--split", default="pilot")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--qa-id", action="append", dest="qa_ids",
                        help="which catalog questions the delivered episodes emit")
    parser.add_argument("--group-question",
                        help="the catalog question the group itself is built around; "
                             "defaults to the one the base request asks for")
    parser.add_argument("--manifest", type=Path,
                        help="saved batch manifest; stage-run reads the group from it")
    parser.add_argument("--group-config", type=Path,
                        help="declarative four-member group config for stage-run")
    parser.add_argument("--retained-visual-root", action="append",
                        help="unit_id=path for a validated retained visual capture")
    parser.add_argument("--max-rounds", type=int, default=64)
    args = parser.parse_args(argv)
    if args.mode == "stage-run":
        if (args.manifest is None) == (args.group_config is None):
            parser.error("stage-run needs exactly one of --manifest or --group-config")
        if args.manifest is not None and not args.group_id:
            parser.error("stage-run with --manifest needs --group-id")
        try:
            return _stage_run(args)
        except (BindingNativeError, ProductionSpecError, OSError, RuntimeError,
                ValueError, KeyError) as exc:
            parser.error(f"{type(exc).__name__}: {exc}")
    if args.base_request is None:
        parser.error("whole-group needs --base-request")
    if args.rpc_port is None:
        args.rpc_port = 39782
    if args.graphics_adapter is None:
        args.graphics_adapter = 0
    if args.world_id is None:
        args.world_id = "world_0001"
    return _whole_group(args, parser)


if __name__ == "__main__":
    raise SystemExit(main())
