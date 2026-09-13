#!/usr/bin/env python3
"""Bounded QA-driven scene backfill using the existing production runner."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from avengine.dataset.binding_group_native import plan_visual_variant
from avengine.dataset.production_runner import run_production, _find_live_stage_worker
from avengine.dataset.production_spec import production_request_from_legacy, initial_stage_work_items


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp_{os.getpid()}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp, path)


def now():
    return datetime.now(timezone.utc).isoformat()


# Two-way answer branches a template may alternate between when the config
# says ``alternate_branches``; the observed branch is accepted at capture time.
BRANCH_PAIRS = {"QA-06": ("moving", "still"), "QA-07": ("left", "right"),
                "QA-09": ("yes", "no"), "QA-15": ("nearer", "farther"), "QA-17": ("yes", "no")}


def screen_verdict(plan):
    """The visibility screen's verdict on the camera the plan selected, or None."""
    solver = (plan.get("camera_condition_sampling") or {}).get("visibility_solver") or {}
    camera = ((plan.get("visual_plan") or {}).get("camera") or {}).get("candidate_id")
    for candidate in solver.get("candidates") or []:
        if candidate.get("candidate_id") == camera:
            return candidate.get("verdict")
    return None


def apply_candidate_variation(request, number, config):
    """Alternate the answer branch and rotate rooms across candidate numbers."""
    if config.get("alternate_branches"):
        for target in request.get("qa_targets") or []:
            branches = BRANCH_PAIRS.get(target.get("qa_id"))
            if branches and target.get("branch") in branches:
                target["branch"] = branches[number % len(branches)]
    rooms = config.get("room_ids")
    if rooms:
        request["room_id"] = rooms[number % len(rooms)]
    return request


def scene_key(plan):
    visual = plan["visual_plan"]
    camera = {k: v for k, v in visual["camera"].items() if k != "candidate_id"}
    tracks = [[(a["actor_id"], a["root_transform"], a.get("action_id"),
                a.get("action_phase")) for a in f["actor_states"]]
              for f in visual["frames"]]
    value = [plan["scene"], plan["clock"], camera,
             [(a["actor_id"], a["asset_id"], a.get("asset_revision")) for a in visual["actors"]],
             tracks]
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def totals(root):
    value = {"visual": 0, "audio": 0, "rlr_reserved": 0}
    for path in root.glob("waves/*/run/state.json"):
        counts = read(path)["native_accounting_totals"]
        value["visual"] += counts["native_visual_worlds"]
        value["audio"] += counts["native_acoustic_launch_attempts"]
        value["rlr_reserved"] += max(counts["native_acoustic_contexts_known"],
                                   4 * counts["native_acoustic_launch_attempts"])
    return value


def question_types(source):
    data = read(source["questions_path"])
    return sorted({q["qa_id"] for q in data["items"] if q.get("status") == "pass"})


def source_row(delivery):
    facts = read(delivery["facts_path"])
    return {
        "episode_id": delivery["episode_id"], "world_id": delivery["world_id"],
        "room_family": delivery["room_family"], "room_id": facts.get("source_paths", {}).get("room_id"),
        "facts_path": delivery["facts_path"], "questions_path": delivery["questions_path"],
        "video_path": delivery.get("video_path") or delivery.get("preview_path"),
        "audio_path": delivery.get("audio_path") or facts["audio"]["path"],
        "asset_ids": sorted({a["asset_id"] for a in facts["actors"].values()}),
        "sound_asset_ids": sorted({e["sound_asset_id"] for e in facts["events"]}),
        "source_tag": "ordinary_automatic_qa_backfill", "task_family": "ordinary_qa",
    }


def counts(state, config):
    # One contribution per QA per distinct physical scene; extra query variants
    # and MCQ/Open forms do not inflate the scene backfill stopping condition.
    result = {q: int(config["baseline_counts"][q]) for q in config["minimum_by_qa"]}
    for source in state["sources"].values():
        for q in source["generated_qa_ids"]:
            if q in result and source.get("new_physical_scene"):
                result[q] += 1
    return result


def deficits(state, config):
    observed = counts(state, config)
    return {q: max(0, target - observed[q]) for q, target in config["minimum_by_qa"].items()}


def normalize_candidate(template, number, config):
    row = deepcopy(template)
    episode = f'{config["batch_id"]}_{number:04d}'
    req = deepcopy(row["request"])
    req.update(episode_id=episode, world_id=episode,
               world_identity_source="declared_before_planning",
               seed=int(config["seed"]) + number, sampling_candidate_index=0)
    req = apply_candidate_variation(req, number, config)
    # Convert through the same typed interface used by the real stage runner.
    spec = production_request_from_legacy(req)
    req = spec.to_legacy_request()
    row.update(episode_id=episode, world_id=episode, request=req,
               execution_status="not_run", achieved_conditions=None,
               stage_scope={"kind": "episode", "scope_id": episode},
               stage_work_items=[item.to_dict() for item in initial_stage_work_items(spec)])
    row["qa_targets"] = req["qa_targets"]
    return row



def plan_candidate(job):
    """Independent CPU planning job; the controller owns all ledger mutations."""
    number, row, candidate, template_id = job
    candidate.mkdir(parents=True, exist_ok=False)
    request_path = candidate / "request.json"
    write(request_path, row["request"])
    result = {"number": number, "template_id": template_id}
    try:
        planned = plan_visual_variant(request_path, candidate / "episode",
                                      label="automatic_cpu_plan", log=candidate / "plan.log")
        plan_path = Path(planned["plan"]).resolve()
        plan_value = read(plan_path)
        row["request_path"] = str(request_path)
        row["screen_verdict"] = screen_verdict(plan_value)
        result.update(outcome="cpu_plan_accepted", candidate={
            "row": row, "plan_root": str(plan_path.parent.parent),
            "scene_key": scene_key(plan_value), "template_id": template_id,
            "screen_verdict": row["screen_verdict"]})
    except Exception as exc:
        result["outcome"] = f"planning_failed: {type(exc).__name__}: {exc}"
    write(candidate / "cpu_result.json", result)
    return result


def execution_options(root):
    path = root / "execution_options.json"
    value = read(path) if path.exists() else {}
    result = {"planning_workers": int(value.get("planning_workers", 1)),
              "wave_size": int(value.get("wave_size", 2)),
              "max_parallel": int(value.get("max_parallel", 2))}
    if not 1 <= result["planning_workers"] <= 4:
        raise ValueError("planning_workers must be in [1,4]")
    if not 1 <= result["wave_size"] <= 6 or not 1 <= result["max_parallel"] <= 6:
        raise ValueError("wave_size and max_parallel must be in [1,6]")
    if value.get("accept_observed_branches"):
        result["accept_observed_branches"] = value["accept_observed_branches"]
    return result


def observed_branch_recovery(wave_root, options, execution):
    saved = wave_root / "run/state.json"
    if not saved.exists() or not execution.get("accept_observed_branches"):
        return []
    reopened = []
    for scope in read(saved)["scopes"]:
        for result in scope["results"]:
            reason = str(result.get("reason") or "")
            if result.get("stage") != "capture" or result.get("status") == "pass":
                continue
            if "measured reappearance answer is " not in reason or "this branch needs" not in reason:
                continue
            episode = scope["scope_key"]
            if episode not in options:
                continue
            attempt = int(result["work_item_id"].rsplit(":", 1)[1])
            retained = wave_root / "run/work" / episode / "capture" / f"attempt_{attempt:02d}" / "episode"
            if not (retained / "capture/research_receipt.json").exists():
                continue
            options[episode].pop("preplanned_episode_root", None)
            options[episode]["retained_episode_root"] = str(retained)
            # The retained root holds a capture and no audio report yet, so the
            # audio stage has to render from it instead of looking one up.
            options[episode]["render_audio_from_retained_capture"] = True
            reopened.append(episode + "/capture")
    return reopened

def publish_sources(root, state, config):
    base = read(config["base_sources_manifest"])
    base["sources"] = base["sources"] + [
        {k: v for k, v in row.items() if k not in ["scene_key", "generated_qa_ids", "new_physical_scene"]}
        for row in state["sources"].values()
    ]
    base["status"] = "native_sources_for_ordinary_question_bank"
    base["counts"] = {"source_count": len(base["sources"])}
    write(root / "source_manifest.json", base)


def run(config, root, resume=False, plan_only=False):
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = root / "state.json"
    if state_path.exists():
        if not resume:
            raise RuntimeError("existing backfill state requires --resume")
        state = read(state_path)
        if state["config"] != config:
            raise RuntimeError("resume configuration differs from saved run")
    else:
        state = {"config": config, "next_candidate": 1, "sources": {}, "seen_scenes": [],
                 "pending_wave": None, "completed_waves": [], "failed_templates": {},
                 "candidate_results": [], "status": "starting"}
        # Exclude already existing geometry before spending new native work.
        for source in read(config["base_sources_manifest"])["sources"]:
            try:
                facts = read(source["facts_path"])
                key = scene_key(read(facts["source_paths"]["plan"]))
                if key not in state["seen_scenes"]:
                    state["seen_scenes"].append(key)
            except (KeyError, OSError, ValueError, TypeError):
                continue
        pilot = read(config["pilot_result"])["run_summary"]
        for delivered in pilot["delivered_episodes"]:
            row = source_row(delivered)
            facts = read(row["facts_path"])
            key = scene_key(read(facts["source_paths"]["plan"]))
            row.update(scene_key=key, generated_qa_ids=question_types(row),
                       new_physical_scene=key not in state["seen_scenes"])
            state["sources"][row["episode_id"]] = row
            if key not in state["seen_scenes"]:
                state["seen_scenes"].append(key)
        write(state_path, state)
    stat = Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()
    write(root / "controller.json", {"pid": os.getpid(), "start_ticks": stat[19],
          "argv": sys.argv, "started_at": now(), "cwd": str(Path.cwd())})
    # Recover late results from workers that survived a controller switch.
    # This goes through the normal runner and its artifact checks; it neither
    # changes attempt numbers nor launches replacement native work.
    for wave_id in state["completed_waves"]:
        wave_root = root / "waves" / wave_id
        saved_path = wave_root / "run/state.json"
        if not saved_path.exists():
            continue
        saved_state = read(saved_path)
        recovered_options = deepcopy(saved_state.get("recipe_options") or {})
        execution = execution_options(root)
        for row in read(wave_root / "manifest.json")["episodes"]:
            recovered_options.setdefault(row["episode_id"], {})["question_acceptance_policy"] = {
                "accept_observed_branches": execution.get("accept_observed_branches", {})}
        reopened = observed_branch_recovery(wave_root, recovered_options, execution)
        recoverable = bool(reopened)
        for scope in saved_state["scopes"]:
            for result in scope["results"]:
                if result.get("status") == "pass" or "was interrupted before it" not in str(result.get("reason")):
                    continue
                work_id = result["work_item_id"]
                work_dir = wave_root / "run/workers" / work_id.replace("/", "__").replace(":", "_")
                if (work_dir / "stage_result.json").exists() or _find_live_stage_worker(work_dir):
                    recoverable = True
        if recoverable:
            policy = read(config["resource_policy_override"])
            execution = execution_options(root)
            policy.setdefault("cpu", {}).update(max_workers=execution["max_parallel"],
                                                max_threads=max(16, execution["max_parallel"]*2))
            recovered = run_production(manifest_path=wave_root/"manifest.json", run_root=wave_root/"run",
                delivery_output=None, resume=True, repository=REPO, resource_policy_override=policy,
                recipe_options=recovered_options, reopen_failed_units=reopened,
                max_parallel=execution["max_parallel"], max_waves=32,
                candidate_rotation_limit=0, interrupted_retry_limit=0)
            before=wave_root/"result.before_live_recovery.json"
            if not before.exists() and (wave_root/"result.json").exists():
                write(before, read(wave_root/"result.json"))
            write(wave_root/"result.json", recovered)
            for delivery in recovered["run_summary"]["delivered_episodes"]:
                row=source_row(delivery)
                facts=read(row["facts_path"])
                row.update(scene_key=scene_key(read(facts["source_paths"]["plan"])),
                           generated_qa_ids=question_types(row), new_physical_scene=True)
                state["sources"][row["episode_id"]]=row
            write(state_path,state)
    native_streaks = {}
    for wave_id in state["completed_waves"]:
        wave_root = root / "waves" / wave_id
        if not (wave_root / "result.json").exists():
            continue
        old_result = read(wave_root / "result.json")["run_summary"]
        old_manifest = read(wave_root / "manifest.json")
        for row in old_manifest["episodes"]:
            episode = row["episode_id"]
            original = next((x["template_id"] for x in state["candidate_results"]
                             if x["number"] == int(episode.rsplit("_", 1)[1])), None)
            if original is None:
                continue
            charged = any(a.get("logical_world_id") == episode and a.get("stage") == "capture"
                          for a in old_result.get("native_accounting", {}).values())
            if not charged:
                continue
            passed = episode in state["sources"] and row["request"]["qa_targets"][0]["qa_id"] in state["sources"][episode]["generated_qa_ids"]
            if passed:
                native_streaks[original] = 0
            elif row.get("screen_verdict") in (None, "consistent"):
                # A capture the screen could not vouch for says nothing about
                # the template; only a consistent screen that still failed counts.
                native_streaks[original] = native_streaks.get(original, 0) + 1
    state["failed_templates"] = native_streaks
    manifest_base = read(config["pilot_manifest"])
    templates = {row["episode_id"]: row for row in manifest_base["episodes"]}
    allowed = config["templates"]
    while True:
        remaining = deficits(state, config)
        budget = totals(root)
        state.update(status="running", updated_at=now(), deficits=remaining, accounting=budget)
        write(state_path, state)
        publish_sources(root, state, config)
        if not any(remaining.values()):
            state["status"] = "minimum_scenes_met"
            break
        execution = execution_options(root)
        state["execution_options"] = execution
        slots = min(execution["wave_size"], config["visual_cap"] - budget["visual"],
                    config["audio_cap"] - budget["audio"],
                    (config["rlr_cap"] - budget["rlr_reserved"]) // 4)
        if not state["pending_wave"] and slots <= 0:
            state["status"] = "completed_with_budget_gaps"
            break
        if not state["pending_wave"]:
            ready = state.setdefault("ready_candidates", [])
            selected = ready[:slots]
            state["ready_candidates"] = ready[slots:]
            selected_by_qa = {}
            for candidate in selected:
                qa = candidate["row"]["request"]["qa_targets"][0]["qa_id"]
                selected_by_qa[qa] = selected_by_qa.get(qa, 0) + 1
            with ThreadPoolExecutor(max_workers=execution["planning_workers"]) as pool:
                while len(selected) < slots and state["next_candidate"] <= config["cpu_candidate_cap"]:
                    # CPU search misses do not prove a template impossible. The
                    # global CPU cap bounds search; native failures have their own cap.
                    choices = [t for t in allowed if remaining.get(t["qa_id"], 0) > 0
                               and state["failed_templates"].get(t["episode_id"], 0) < 3]
                    if not choices:
                        break
                    jobs = []
                    assigned = dict(selected_by_qa)
                    count = min(execution["planning_workers"], slots - len(selected),
                                config["cpu_candidate_cap"] - state["next_candidate"] + 1)
                    for _ in range(count):
                        choices.sort(key=lambda t: (assigned.get(t["qa_id"], 0),
                                                   -remaining[t["qa_id"]], t["qa_id"]))
                        target = choices[0]
                        assigned[target["qa_id"]] = assigned.get(target["qa_id"], 0) + 1
                        number = state["next_candidate"]
                        state["next_candidate"] += 1
                        row = normalize_candidate(templates[target["episode_id"]], number, config)
                        jobs.append((number, row, root / "candidates" / f"{number:04d}",
                                     target["episode_id"]))
                    # Reserve identifiers before workers start; restart never
                    # overwrites a half-written or completed candidate directory.
                    write(state_path, state)
                    futures = [pool.submit(plan_candidate, job) for job in jobs]
                    for future in as_completed(futures):
                        item = future.result()
                        candidate = item.pop("candidate", None)
                        if candidate:
                            if candidate["scene_key"] in state["seen_scenes"]:
                                item["outcome"] = "duplicate_physical_scene"
                            else:
                                state["seen_scenes"].append(candidate["scene_key"])
                                selected.append(candidate)
                                qa = candidate["row"]["request"]["qa_targets"][0]["qa_id"]
                                selected_by_qa[qa] = selected_by_qa.get(qa, 0) + 1
                        state["candidate_results"].append(item)
                        write(state_path, state)
            if not selected:
                state["status"] = "completed_with_planning_gaps"
                break
            wave_id = f'{len(state["completed_waves"]) + 1:04d}'
            wave_root = root / "waves" / wave_id
            wave_root.mkdir(parents=True, exist_ok=False)
            manifest = deepcopy(manifest_base)
            manifest.update(batch_id=f'{config["batch_id"]}_wave_{wave_id}',
                            episodes=[v["row"] for v in selected], production=None,
                            requested_episode_count=len(selected), executed_episode_count=0)
            write(wave_root / "manifest.json", manifest)
            state["pending_wave"] = {"id": wave_id, "selected": selected}
            write(state_path, state)
        if plan_only:
            state["status"] = "cpu_ready_no_native_started"
            break
        pending = state["pending_wave"]
        wave_root = root / "waves" / pending["id"]
        # Plan the following candidates while native audio is running.
        # Only the main thread updates IDs, counts and state; workers write
        # their own independent cpu_result.json and never launch native work.
        prefetch_pool = ThreadPoolExecutor(max_workers=execution["planning_workers"])
        prefetch_futures = []
        reserved = state.setdefault("prefetch_jobs", [])
        if not reserved and state["next_candidate"] <= config["cpu_candidate_cap"]:
            choices = [t for t in allowed if remaining.get(t["qa_id"], 0) > 0
                       and state["failed_templates"].get(t["episode_id"], 0) < 3]
            assigned = {}
            for _ in range(min(execution["wave_size"], config["cpu_candidate_cap"] - state["next_candidate"] + 1)):
                if not choices:
                    break
                choices.sort(key=lambda t: (assigned.get(t["qa_id"], 0), -remaining[t["qa_id"]], t["qa_id"]))
                target = choices[0]
                assigned[target["qa_id"]] = assigned.get(target["qa_id"], 0) + 1
                number = state["next_candidate"]
                state["next_candidate"] += 1
                reserved.append({"number": number, "template_id": target["episode_id"],
                    "row": normalize_candidate(templates[target["episode_id"]], number, config)})
            write(state_path, state)
        for job in reserved:
            candidate_dir = root / "candidates" / f'{job["number"]:04d}'
            if (candidate_dir / "cpu_result.json").exists():
                prefetch_futures.append(prefetch_pool.submit(read, candidate_dir / "cpu_result.json"))
            elif not candidate_dir.exists():
                prefetch_futures.append(prefetch_pool.submit(plan_candidate,
                    (job["number"], job["row"], candidate_dir, job["template_id"])))
            else:
                state["candidate_results"].append({"number": job["number"],
                    "template_id": job["template_id"], "outcome": "interrupted_cpu_candidate_preserved"})
        policy = read(config["resource_policy_override"])
        policy.setdefault("cpu", {}).update(max_workers=execution["max_parallel"],
                                            max_threads=max(16, execution["max_parallel"] * 2))
        recipe_options = {v["row"]["episode_id"]: {
            "preplanned_episode_root": v["plan_root"],
            "question_acceptance_policy": {
                "accept_observed_branches": execution.get("accept_observed_branches", {})}}
            for v in pending["selected"]}
        reopened = observed_branch_recovery(wave_root, recipe_options, execution)
        result = run_production(
            manifest_path=wave_root / "manifest.json", run_root=wave_root / "run",
            delivery_output=None, resume=(wave_root / "run/state.json").exists(),
            repository=REPO, native_visual_world_budget=len(pending["selected"]),
            recipe_options=recipe_options, reopen_failed_units=reopened,
            resource_policy_override=policy,
            candidate_rotation_limit=0, interrupted_retry_limit=0, max_parallel=execution["max_parallel"], max_waves=32)
        write(wave_root / "result.json", result)
        try:
            for future in as_completed(prefetch_futures):
                item = future.result()
                candidate = item.pop("candidate", None)
                if candidate:
                    if candidate["scene_key"] in state["seen_scenes"]:
                        item["outcome"] = "duplicate_physical_scene"
                    else:
                        state["seen_scenes"].append(candidate["scene_key"])
                        state.setdefault("ready_candidates", []).append(candidate)
                state["candidate_results"].append(item)
                write(state_path, state)
        finally:
            prefetch_pool.shutdown(wait=True)
        state["prefetch_jobs"] = []
        delivered = {d["episode_id"]: d for d in result["run_summary"]["delivered_episodes"]}
        for candidate in pending["selected"]:
            episode = candidate["row"]["episode_id"]
            template_id = candidate["template_id"]
            target = candidate["row"]["request"]["qa_targets"][0]["qa_id"]
            successful = False
            if episode in delivered:
                row = source_row(delivered[episode])
                row.update(scene_key=candidate["scene_key"], generated_qa_ids=question_types(row),
                           new_physical_scene=True)
                state["sources"][episode] = row
                successful = target in row["generated_qa_ids"]
            capture_charged = any(
                a.get("logical_world_id") == episode and a.get("stage") == "capture"
                for a in result["run_summary"].get("native_accounting", {}).values())
            if successful:
                state["failed_templates"][template_id] = 0
            elif capture_charged and candidate.get("screen_verdict") in (None, "consistent"):
                state["failed_templates"][template_id] = state["failed_templates"].get(template_id, 0) + 1
        state["completed_waves"].append(pending["id"])
        state["pending_wave"] = None
        write(state_path, state)
    state.update(updated_at=now(), deficits=deficits(state, config), accounting=totals(root))
    write(state_path, state)
    publish_sources(root, state, config)
    if not plan_only and config.get("export_bank", True):
        bank_config = read(config["base_bank_config"])
        bank_config["sources_manifest"] = str(root / "source_manifest.json")
        write(root / "bank_config.json", bank_config)
        command = [sys.executable, str(REPO / "tools/dataset/generate_retained_qa_bank.py"),
                   "--config", str(root / "bank_config.json"), "--output", str(root / "bank")]
        if (root / "bank").exists():
            command.append("--resume")
        with (root / "bank.log").open("a") as log:
            subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        state["bank_report"] = str(root / "bank/report.json")
        state["bank_minimum_met"] = read(root / "bank/report.json")["minimum_per_type_met"]
        write(state_path, state)
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    os.chdir(REPO)
    try:
        result = run(read(args.config), args.output.resolve(), args.resume, args.plan_only)
        print(json.dumps({"status": result["status"], "deficits": result["deficits"]}))
    except Exception as exc:
        write(args.output / "controller_failure.json", {"at": now(), "reason": str(exc),
              "traceback": traceback.format_exc(), "action": "repair input/code then ordinary --resume"})
        raise


if __name__ == "__main__":
    main()
