"""Run CPU test receipts; optional file groups isolate existing daemon workers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label")
    parser.add_argument("output", type=Path)
    parser.add_argument("--test-path", action="append")
    parser.add_argument("--ignore", action="append", default=[])
    options = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise RuntimeError("CPU-only nice 10 test process required")
    # These controls affect this process only.
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except ImportError:
        pass
    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass
    try:
        import onnxruntime as ort
        old_session = ort.InferenceSession
        def bounded_session(*args, **kwargs):
            session_options = kwargs.get("sess_options")
            if session_options is None and len(args) < 2:
                session_options = ort.SessionOptions()
                kwargs["sess_options"] = session_options
            elif len(args) >= 2:
                session_options = args[1]
            if session_options is not None:
                session_options.intra_op_num_threads = 1
                session_options.inter_op_num_threads = 1
            return old_session(*args, **kwargs)
        ort.InferenceSession = bounded_session
    except ImportError:
        pass
    import pytest
    label, out = options.label, options.output
    reports, collected = [], []
    current = {"nodeid": None}
    peak = [0]
    stop = threading.Event()
    def monitor():
        while not stop.wait(.1):
            count = len(list(Path("/proc/self/task").iterdir()))
            peak[0] = max(peak[0], count)
            if count > 32:
                receipt = {"pid": os.getpid(), "threads": count, "timestamp": time.time(),
                           "current_test": current["nodeid"], "partial_reports": reports,
                           "python_threads": [t.name for t in threading.enumerate()]}
                (out / (label + "_thread_violation.json")).write_text(json.dumps(receipt))
                os._exit(125)
    observer = threading.Thread(target=monitor, daemon=True)
    observer.start()
    class Receipts:
        def pytest_collection_finish(self, session):
            collected.extend(item.nodeid for item in session.items)
        def pytest_runtest_setup(self, item):
            current["nodeid"] = item.nodeid
        def pytest_runtest_logreport(self, report):
            reports.append({"nodeid": report.nodeid, "when": report.when, "outcome": report.outcome})
        def pytest_collectreport(self, report):
            if report.failed:
                reports.append({"nodeid": report.nodeid, "when": "collect", "outcome": "failed"})
    tick = time.monotonic()
    args = list(options.test_path or ["tests"]) + [
        "-q", "-m", "not integration and not canary and not native_habitat and not rlr_audio and not blender_assets and not media_readback and not release_canary",
        "--basetemp", str(out / (label + "_tmp")), "-o", "cache_dir=" + str(out / (label + "_cache"))]
    for path in options.ignore:
        args += ["--ignore", path]
    try:
        code = int(pytest.main(args, plugins=[Receipts()]))
    finally:
        stop.set()
        observer.join()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    summary = {"label": label, "commit": os.environ["TEST_COMMIT"], "command": sys.argv,
               "pytest_args": args, "scope": "all CPU tests; matching native/integration/media/canary exclusions",
               "exit_code": code,
               "failure_ids": sorted({r["nodeid"] for r in reports if r["when"] == "call" and r["outcome"] == "failed"}),
               "error_ids": sorted({r["nodeid"] for r in reports if r["when"] != "call" and r["outcome"] == "failed"}),
               "passed": sum(r["when"] == "call" and r["outcome"] == "passed" for r in reports),
               "skipped": sum(r["outcome"] == "skipped" for r in reports),
               "wall_seconds": time.monotonic() - tick, "cpu_user_seconds": usage.ru_utime,
               "cpu_system_seconds": usage.ru_stime, "os_threads_peak": peak[0],
               "private_runtime_caps": ["torch intra/inter 1", "cv2 1", "onnxruntime intra/inter 1", "BLAS/OMP/numba 1"],
               "collected_ids": collected, "reports": reports}
    (out / (label + ".json")).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print("JSON_SUMMARY", json.dumps({k: v for k, v in summary.items() if k not in ("reports", "collected_ids")}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
