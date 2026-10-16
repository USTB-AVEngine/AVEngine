#!/usr/bin/env python3
"""Low-priority one-worker CPU finalization queue for strict room batches."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _initialize_worker(environment: Mapping[str, str], nice_increment: int) -> None:
    for key, value in environment.items():
        os.environ[str(key)] = str(value)
    if nice_increment:
        os.nice(int(nice_increment))


def _run_finalizer(argv: list[str], cwd: str, expected_receipt: str) -> Path:
    completed = subprocess.run(
        argv,
        cwd=cwd,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "CPU finalizer failed: "
            + " ".join(argv)
            + "\n"
            + completed.stdout
            + completed.stderr
        )
    path = Path(expected_receipt)
    _require(path.is_file(), f"CPU finalizer did not publish FINAL_READY: {path}")
    return path


class ProcessFinalizeQueue:
    """Submit one Episode finalizer process with fixed resource constraints.

    The room-batch controller owns queue-depth backpressure.  This class owns
    exactly one long-lived worker process; each finalizer command is still an
    isolated subprocess so a codec/library failure cannot corrupt the worker's
    Python state.
    """

    def __init__(
        self,
        *,
        policy: Mapping[str, Any],
        finalizer_python: Path,
        finalizer_script: Path,
        repo_root: Path,
    ) -> None:
        _require(policy.get("worker_count") == 1, "CPU worker count must be one")
        _require(policy.get("queue_depth") == 2, "CPU queue depth must be two")
        environment = policy.get("environment")
        _require(isinstance(environment, Mapping), "CPU worker environment missing")
        expected_caps = {
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        }
        _require(set(environment) == expected_caps, "CPU thread-cap variables drift")
        _require(
            all(value == "2" for value in environment.values()), "thread cap drift"
        )
        _require(finalizer_python.is_file(), "CPU finalizer Python is missing")
        _require(finalizer_script.is_file(), "CPU finalizer script is missing")
        _require(repo_root.is_dir(), "CPU finalizer repository root is missing")
        self.finalizer_python = finalizer_python.resolve()
        self.finalizer_script = finalizer_script.resolve()
        self.repo_root = repo_root.resolve()
        self.executor = ProcessPoolExecutor(
            max_workers=1,
            initializer=_initialize_worker,
            initargs=(dict(environment), int(policy.get("nice_increment", 0))),
        )
        self.closed = False

    def submit(
        self,
        *,
        batch: Any,
        episode: Any,
        raw_ready: Path,
        attempt_root: Path,
    ) -> Future[Path]:
        _require(not self.closed, "CPU finalization queue is closed")
        final_receipt = episode.output_root / "FINAL_READY.json"
        _require(not final_receipt.exists(), "FINAL_READY must be new")
        argv = [
            str(self.finalizer_python),
            str(self.finalizer_script),
            "--batch-request",
            str(batch.request_path),
            "--batch-request-sha256",
            str(batch.request_sha256),
            "--episode-id",
            str(episode.episode_id),
            "--input-binding-sha256",
            str(episode.bindings["binding_sha256"]),
            "--raw-ready",
            str(raw_ready.resolve()),
            "--attempt-root",
            str(attempt_root.resolve()),
            "--output",
            str(episode.output_root.resolve()),
        ]
        return self.executor.submit(
            _run_finalizer,
            argv,
            str(self.repo_root),
            str(final_receipt),
        )

    def close(self) -> None:
        if self.closed:
            return
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.closed = True
