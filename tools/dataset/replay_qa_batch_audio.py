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


def prepare_attempt(row, entry, output, attempt, gain, catalog_path, producer, *, reuse_audio=False):
    from avengine.rooms.room_package import write_room_package_plan_snapshot
    source = source_episode(row)
    retained_report = None
    if reuse_audio:
        refs = read_json(source / "delivery/input_refs.json")
        retained_report = Path(refs["audio_report"]).resolve()
        report = read_json(retained_report)
        recorded_gain = report.get("gain_application", {}).get("post_assembly_convolution_gain")
        if recorded_gain != gain:
            raise ValueError(f"retained audio gain {recorded_gain!r} differs from declared gain {gain!r}")
    target = Path(output) / "episodes" / row["episode_id"] / attempt / "episode"
    target.mkdir(parents=True, exist_ok=False)
    # Finalization only reads capture. Keep the actual pixel/readback identity intact.
    (target / "capture").symlink_to((source / "capture").resolve(), target_is_directory=True)
    shutil.copytree(source / "plan", target / "plan", symlinks=False)
    request_path = source / "request.json"
    request = read_json(request_path) if request_path.is_file() else deepcopy(entry["request"])
    catalog = read_json(catalog_path)
    request["room_catalog"] = str(Path(catalog_path).resolve())
    request["source_registry"] = str(REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json")
    request["post_assembly_convolution_gain"] = gain
    if row.get("rir_cache"):
        cache = Path(row["rir_cache"]).resolve()
        if not cache.is_dir():
            raise FileNotFoundError(f"selected RIR cache is unavailable: {cache}")
        request["rir_cache"] = str(cache)
    runtime = request.setdefault("runtime", {})
    runtime["path_bindings"] = {**catalog.get("path_bindings", {}), **runtime.get("path_bindings", {})}
    write_json(target / "request.json", request)
    package = read_json(target / "plan/room_package.json")
    write_room_package_plan_snapshot(target / "plan", package,
        path_bindings=runtime["path_bindings"], catalog_path=Path(catalog_path))
    write_json(target / "plan/audio_replay.json", {
        "source_episode_root": str(source), "captured_plan": str(source / "plan/episode_plan.json"),
        "post_assembly_convolution_gain": gain, "capture_reused": True,
        "rir_cache": request.get("rir_cache"),
        "audio_reused": reuse_audio,
        "retained_audio_report": str(retained_report) if retained_report else None,
        "capture_producer": str(source / "producer_version.json") if (source / "producer_version.json").is_file() else None,
        "capture_receipt": str((source / "capture/research_receipt.json").resolve()), "producer": producer,
    })
    write_json(target / "producer_version.json", producer)
    prepared = deepcopy(entry)
    prepared["request"] = request
    prepared["request_path"] = str(target / "request.json")
    prepared["controller_entrypoint"] = str(REPOSITORY / "tools/studio/run_qa_episode.py")
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
            args.attempt, args.convolution_gain, args.catalog.resolve(), producer, reuse_audio=args.reuse_audio)
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
            request = deepcopy(entry["request"])
            request["room_catalog"] = str(args.catalog.resolve())
            request["source_registry"] = str(REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json")
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
    write_json(output / "manifest.json", final_manifest)
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
