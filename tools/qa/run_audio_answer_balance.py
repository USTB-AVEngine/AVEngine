#!/usr/bin/env python3
"""Plan and render answer-balanced audio over retained two-actor captures; resume safely."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from avengine.qa.audio_variants import plan_audio_jobs, prepare_audio_variant
from avengine.qa.unified_catalog import generate_unified_questions
from avengine.qa.choice_support import apply_choice_support


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp_{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    temporary.replace(path)


def target_questions(facts, job, requested):
    """Request branches, then compare with questions derived from realized facts."""
    if len(facts["events"]) != job["event_count"]:
        raise ValueError("rendered event count missed the declared target")
    sounding = sorted({e["actor_id"] for e in facts["events"]})
    actor = requested["silent_actors"][0] if job["qa01_answer"] == "no" else sounding[job["seed"] % len(sounding)]
    targets = [("QA-01", job["qa01_answer"], actor, 1), ("QA-21", None, None, 32),
               ("QA-22", None, None, 1), ("QA-23", None, None, 1), ("QA-18", None, None, 1)]
    if job["qa05_branch"]:
        targets.append(("QA-05", job["qa05_branch"], None, 1))
    items, results = [], []
    for qa, branch, actor, quota in targets:
        scoped = deepcopy(facts)
        target = {"qa_id": qa, "items": 1}
        if branch is not None:
            target["branch"] = branch
        if actor is not None:
            target["target_actor_ids"] = [actor]
        scoped.setdefault("sampling", {})["qa_targets"] = [target]
        scoped["sampling"]["time_display_precision"] = 0
        scoped["sampling"].setdefault("qa_sampling", {})["time_display_precision"] = 0
        questions = generate_unified_questions(scoped, qa_ids=[qa], items_per_type=quota,
                                               include_angle_followups=False, seed=str(job["seed"]))
        chosen = [apply_choice_support(item) for item in questions["items"]]
        if qa != "QA-18" and not chosen:
            raise ValueError(f"target has no valid question: {job['id']} {qa}: {questions['deferred']}")
        if branch is not None and any(r["status"] != "met" for r in questions["qa_target_results"]):
            raise ValueError(f"observed answer missed target: {job['id']} {qa}: {questions['qa_target_results']}")
        if qa == "QA-23" and chosen[0]["truth"]["value"] != [job["event_count"]]:
            raise ValueError("QA-23 disagrees with the observed native event count")
        items.extend(chosen)
        results.extend(questions["qa_target_results"])
    return {"items": items, "requested_targets": job, "qa_target_results": results,
            "scoring_policy": {"form_denominator": "offered_forms"}}


def run(config, output, *, resume=False, repository=ROOT, max_new_jobs=None):
    # Native rendering requires soundfile's float-WAV reader. Fail before work.
    import soundfile
    from avengine.rooms.qa_delivery import finalize_qa_episode
    from avengine.dataset.binding_group_native import check_requested_visibility

    output = Path(output).resolve()
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
        write(output / "run_config.json", config)
    elif load(output / "run_config.json") != config:
        raise ValueError("resume config differs; use a fresh output")
    import fcntl
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        planned_path = output / "planned_jobs.json"
        if not planned_path.exists():
            write(planned_path, plan_audio_jobs(config, repository=repository))
        planned = load(planned_path)
        records, hist, appearance_hist = [], defaultdict(Counter), defaultdict(Counter)
        max_attempts = int(config.get("max_attempts_per_job", 2))
        if max_attempts < 1:
            raise ValueError("max_attempts_per_job must be positive")
        if max_new_jobs is not None and max_new_jobs < 1:
            raise ValueError("max_new_jobs must be positive")
        new_jobs = 0
        for job in planned["jobs"]:
            job_root = output / job["id"]
            job_root.mkdir(exist_ok=True)
            completed = job_root / "complete.json"
            if completed.exists():
                record = load(completed)
                out = Path(record["output"])
                questions = load(out / "targeted_questions.json")
                facts = load(out / "delivery/facts.json")
            else:
                if max_new_jobs is not None and new_jobs >= max_new_jobs:
                    break
                attempts = sorted(job_root.glob("attempt_*"))
                if len(attempts) >= max_attempts:
                    raise RuntimeError(f"attempt budget exhausted: {job_root}; inspect failure before starting a fresh run")
                out = job_root / f"attempt_{len(attempts) + 1:02d}"
                try:
                    requested = prepare_audio_variant(job["source"], out, config["pool"], job["sounds"],
                        repository=repository, profile=job["profile"], seed=job["seed"],
                        human_nonverbal_classes=config.get("human_nonverbal_classes", []))
                    request, plan = load(out / "request.json"), load(out / "plan/episode_plan.json")
                    check_requested_visibility(plan, request, out / "capture")
                    source = Path(job["source"])
                    review = source / "delivery/appearance_review.json"
                    if not review.is_file():
                        refs = load(source / "delivery/input_refs.json")
                        review = Path(refs["appearance_review"]) if refs.get("appearance_review") else None
                    audio_report = None
                    for previous in reversed(attempts):
                        previous_events = previous / "plan/audio_events.json"
                        if previous_events.is_file() and load(previous_events) == load(out/"plan/audio_events.json"):
                            for name in ("research_report.json", "research_receipt.json"):
                                candidate = previous/"delivery/audio"/name
                                if candidate.is_file():
                                    audio_report = candidate
                                    break
                        if audio_report is not None:
                            break
                    result = finalize_qa_episode(out, out / "delivery", repository=repository,
                                                 request=request, appearance_review=review, audio_report=audio_report)
                    write(out / "result.json", result)
                    facts = load(out / "delivery/facts.json")
                    questions = target_questions(facts, job, requested)
                    questions["facts_path"] = str(out / "delivery/facts.json")
                    write(out / "targeted_questions.json", questions)
                    record = {"job": job, "target_met": True, "events": len(facts["events"]),
                              "questions": dict(Counter(q["qa_id"] for q in questions["items"])),
                              "output": str(out)}
                    record["reused_completed_audio_report"] = str(audio_report) if audio_report else None
                    write(completed, record)
                    new_jobs += 1
                except Exception as error:
                    write(job_root / "failure.json", {"attempt": str(out), "error": f"{type(error).__name__}: {error}",
                        "recovery": "Preserved failed attempt. Diagnose/fix first; --resume starts a fresh bounded attempt."})
                    raise
                print(json.dumps({"job": job["id"], "source": job["source"], "target_met": True}), flush=True)
            actors = {a["actor_id"]: a for a in load(out / "plan/episode_plan.json")["visual_plan"]["actors"]}
            for item in questions["items"]:
                hist[item["qa_id"]][json.dumps(item["truth"]["value"], ensure_ascii=False)] += 1
            for actor, cls in {(e["actor_id"], e["sound_class"]) for e in facts["events"]}:
                appearance_hist[actors[actor]["asset_id"]][cls] += 1
            records.append(record)
            write(output / "progress.json", {"completed": len(records), "planned": len(planned["jobs"]),
                                            "answer_counts": {k: dict(v) for k, v in hist.items()}})
        summary = {"status": "partial_budget" if len(records) < len(planned["jobs"]) else "completed_with_deficits" if planned["deficits"] else "completed",
                   "planned_jobs": len(planned["jobs"]), "completed_jobs": len(records),
                   "jobs": records, "answer_counts": {k: dict(v) for k, v in hist.items()},
                   "sound_class_by_appearance": {k: dict(v) for k, v in appearance_hist.items()},
                   "retained_visual_worlds": len({r["job"]["world_id"] for r in records}),
                   "new_visual_renders": 0, "new_audio_renders": len(records),
                   "deficits": planned["deficits"], "rejected_sources": planned["rejected_sources"],
                   "claim_boundary": "Verified audio-answer variants of retained worlds, not additional independent worlds or human/model admission."}
        write(output / "summary.json", summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-jobs", type=int, help="Bound native canary work; --resume later continues the same plan")
    args = parser.parse_args()
    run(load(args.config), args.output, resume=args.resume, max_new_jobs=args.max_new_jobs)


if __name__ == "__main__":
    main()
