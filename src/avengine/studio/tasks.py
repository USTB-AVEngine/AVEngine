"""Serial subprocess render queue with on-disk task state.

One daemon worker executes tasks strictly in submission order (the box has
one GPU render lane). Each task owns a fresh directory under the tasks root
holding task.json, task.log (combined stdout/stderr, nohup style), and an
output/ subtree the wrapped CLI verb writes into. Task state survives server
restarts; tasks that were queued or running when the server died are marked
interrupted rather than silently resumed.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

TERMINAL_STATUSES = frozenset({"pass", "fail", "interrupted"})


class StudioTaskError(ValueError):
    """Raised for invalid task operations (unknown id, bad state)."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class StudioTaskQueue:
    def __init__(self, tasks_root: str | Path) -> None:
        self._tasks_root = Path(tasks_root).resolve()
        self._tasks_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._pending: "queue.Queue[str]" = queue.Queue()
        self._records: dict[str, dict] = {}
        self._recover_existing()
        self._worker = threading.Thread(
            target=self._worker_loop, name="studio-task-worker", daemon=True
        )
        self._worker.start()

    @property
    def tasks_root(self) -> Path:
        return self._tasks_root

    def _recover_existing(self) -> None:
        for task_json in sorted(self._tasks_root.glob("*/task.json")):
            try:
                record = json.loads(task_json.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            task_id = record.get("task_id")
            if not task_id:
                continue
            if record.get("status") in {"queued", "running"}:
                record["status"] = "interrupted"
                record["error"] = "server restarted while the task was queued or running"
                record["finished_at"] = _utc_now_iso()
                self._persist(record)
            self._records[str(task_id)] = record

    def _claim_task_dir(self, template: str) -> tuple[str, Path]:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        for attempt in range(1000):
            suffix = f"-{attempt:02d}" if attempt else ""
            task_id = f"{stamp}{suffix}-{template}"
            task_dir = self._tasks_root / task_id
            try:
                task_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                continue
            return task_id, task_dir
        raise StudioTaskError(f"could not claim a fresh task directory under {self._tasks_root}")

    def submit(
        self,
        *,
        template: str,
        argv_builder: Callable[[Path], list[str]],
        cwd: str | Path,
        metadata: dict | None = None,
    ) -> str:
        with self._lock:
            task_id, task_dir = self._claim_task_dir(template)
        output_root = task_dir / "output"
        output_root.mkdir()
        output_path = output_root / "render"
        try:
            argv = [str(item) for item in argv_builder(output_path)]
        except Exception:
            shutil.rmtree(task_dir, ignore_errors=True)
            raise
        record = {
            "task_id": task_id,
            "template": template,
            "status": "queued",
            "argv": argv,
            "cwd": str(Path(cwd).resolve()),
            "created_at": _utc_now_iso(),
            "started_at": None,
            "finished_at": None,
            "returncode": None,
            "error": None,
            "task_dir": str(task_dir),
            "log_path": str(task_dir / "task.log"),
            "output_dir": str(output_path),
            "metadata": metadata or {},
        }
        with self._lock:
            self._records[task_id] = record
            self._persist(record)
        self._pending.put(task_id)
        return task_id

    def _worker_loop(self) -> None:
        while True:
            task_id = self._pending.get()
            try:
                self._execute(task_id)
            except Exception as exc:  # keep the worker alive on any failure
                with self._lock:
                    record = self._records.get(task_id)
                    if record is not None:
                        record["status"] = "fail"
                        record["error"] = f"{type(exc).__name__}: {exc}"
                        record["finished_at"] = _utc_now_iso()
                        self._persist(record)
            finally:
                self._pending.task_done()

    def _execute(self, task_id: str) -> None:
        with self._lock:
            record = self._records[task_id]
            record["status"] = "running"
            record["started_at"] = _utc_now_iso()
            self._persist(record)
            argv = list(record["argv"])
            cwd = record["cwd"]
            log_path = Path(record["log_path"])
        returncode: int | None
        error: str | None
        try:
            with log_path.open("ab") as log_file:
                completed = subprocess.run(
                    argv,
                    cwd=cwd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            returncode = completed.returncode
            error = None
        except OSError as exc:
            returncode = None
            error = f"{type(exc).__name__}: {exc}"
        with self._lock:
            record["returncode"] = returncode
            record["error"] = error
            record["status"] = "pass" if returncode == 0 else "fail"
            record["finished_at"] = _utc_now_iso()
            self._persist(record)

    def _persist(self, record: dict) -> None:
        task_dir = Path(record["task_dir"])
        target = task_dir / "task.json"
        staging = task_dir / "task.json.tmp"
        staging.write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        os.replace(staging, target)

    def get(self, task_id: str) -> dict:
        with self._lock:
            record = self._records.get(task_id)
            if record is None:
                raise StudioTaskError(f"unknown task: {task_id}")
            return json.loads(json.dumps(record))

    def list_tasks(self) -> list[dict]:
        with self._lock:
            records = [json.loads(json.dumps(r)) for r in self._records.values()]
        records.sort(key=lambda r: (r.get("created_at") or "", r["task_id"]), reverse=True)
        return records

    def read_log_tail(self, task_id: str, *, max_lines: int = 100) -> str:
        record = self.get(task_id)
        log_path = Path(record["log_path"])
        if not log_path.is_file():
            return ""
        text = log_path.read_text(encoding="utf-8", errors="replace")
        return "\n".join(text.splitlines()[-max(1, max_lines):])

    def wait(self, task_id: str, *, timeout_s: float = 60.0, poll_s: float = 0.2) -> str:
        deadline = time.monotonic() + timeout_s
        while True:
            status = self.get(task_id)["status"]
            if status in TERMINAL_STATUSES:
                return status
            if time.monotonic() >= deadline:
                raise TimeoutError(f"task {task_id} still {status!r} after {timeout_s}s")
            time.sleep(poll_s)
