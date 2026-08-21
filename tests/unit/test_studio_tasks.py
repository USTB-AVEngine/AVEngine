from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from avengine.studio.tasks import StudioTaskError, StudioTaskQueue


def _echo_builder(text: str):
    def build(output_path: Path) -> list[str]:
        return [sys.executable, "-c", f"print({text!r})"]

    return build


def test_task_runs_to_pass_and_persists(tmp_path: Path) -> None:
    queue = StudioTaskQueue(tmp_path / "tasks")
    task_id = queue.submit(
        template="echo",
        argv_builder=_echo_builder("hello studio"),
        cwd=tmp_path,
        metadata={"submitted_via": "test"},
    )
    assert queue.wait(task_id, timeout_s=30.0) == "pass"
    record = queue.get(task_id)
    assert record["returncode"] == 0
    assert "hello studio" in queue.read_log_tail(task_id)
    persisted = json.loads(
        (Path(record["task_dir"]) / "task.json").read_text(encoding="utf-8")
    )
    assert persisted["status"] == "pass"
    assert persisted["metadata"] == {"submitted_via": "test"}


def test_failing_command_marks_fail(tmp_path: Path) -> None:
    queue = StudioTaskQueue(tmp_path / "tasks")
    task_id = queue.submit(
        template="boom",
        argv_builder=lambda output: [sys.executable, "-c", "raise SystemExit(3)"],
        cwd=tmp_path,
    )
    assert queue.wait(task_id, timeout_s=30.0) == "fail"
    assert queue.get(task_id)["returncode"] == 3


def test_builder_failure_leaves_no_task_dir(tmp_path: Path) -> None:
    tasks_root = tmp_path / "tasks"
    queue = StudioTaskQueue(tasks_root)

    def bad_builder(output: Path) -> list[str]:
        raise ValueError("bad overrides")

    with pytest.raises(ValueError, match="bad overrides"):
        queue.submit(template="echo", argv_builder=bad_builder, cwd=tmp_path)
    assert list(tasks_root.iterdir()) == []
    assert queue.list_tasks() == []


def test_unknown_task_raises(tmp_path: Path) -> None:
    queue = StudioTaskQueue(tmp_path / "tasks")
    with pytest.raises(StudioTaskError):
        queue.get("nope")


def test_restart_recovers_history_and_marks_interrupted(tmp_path: Path) -> None:
    tasks_root = tmp_path / "tasks"
    first = StudioTaskQueue(tasks_root)
    done_id = first.submit(
        template="echo", argv_builder=_echo_builder("done"), cwd=tmp_path
    )
    assert first.wait(done_id, timeout_s=30.0) == "pass"

    # simulate a task that was still running when the server died
    stale_dir = tasks_root / "19990101T000000Z-stale"
    stale_dir.mkdir()
    (stale_dir / "task.json").write_text(
        json.dumps(
            {
                "task_id": "19990101T000000Z-stale",
                "template": "echo",
                "status": "running",
                "task_dir": str(stale_dir),
                "log_path": str(stale_dir / "task.log"),
                "output_dir": str(stale_dir / "output" / "render"),
                "created_at": "1999-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    second = StudioTaskQueue(tasks_root)
    by_id = {record["task_id"]: record for record in second.list_tasks()}
    assert by_id[done_id]["status"] == "pass"
    assert by_id["19990101T000000Z-stale"]["status"] == "interrupted"


def test_task_ids_do_not_collide_within_one_second(tmp_path: Path) -> None:
    queue = StudioTaskQueue(tmp_path / "tasks")
    ids = [
        queue.submit(template="echo", argv_builder=_echo_builder(str(i)), cwd=tmp_path)
        for i in range(3)
    ]
    assert len(set(ids)) == 3
    for task_id in ids:
        assert queue.wait(task_id, timeout_s=30.0) == "pass"
