from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from avengine.studio import tasks as studio_tasks
from avengine.studio.tasks import StudioTaskError, StudioTaskQueue


def _echo_builder(text: str):
    def build(output_path: Path) -> list[str]:
        return [sys.executable, "-c", f"print({text!r})"]

    return build


def _queue_with_log(
    tmp_path: Path, data: bytes | None
) -> tuple[StudioTaskQueue, str, Path]:
    tasks_root = tmp_path / "tasks"
    task_id = "20000101T000000Z-tail"
    task_dir = tasks_root / task_id
    task_dir.mkdir(parents=True)
    log_path = task_dir / "task.log"
    if data is not None:
        log_path.write_bytes(data)
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "template": "tail",
                "status": "pass",
                "task_dir": str(task_dir),
                "log_path": str(log_path),
                "output_dir": str(task_dir / "output" / "render"),
                "created_at": "2000-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    return StudioTaskQueue(tasks_root), task_id, log_path


def _legacy_log_tail(data: bytes, max_lines: int) -> str:
    text = data.decode("utf-8", errors="replace")
    return "\n".join(text.splitlines()[-max(1, max_lines):])


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


@pytest.mark.parametrize(
    ("data", "max_lines"),
    [
        (b"", 5),
        (b"one", 5),
        (b"one\n", 5),
        (b"one\n\ntwo\n", 3),
        (b"one\r\ntwo\rthree\n", 2),
        (
            "a\rb\nc\r\nd\ve\ff\x1cg\x1dh\x1ei\x85j\u2028k\u2029l".encode(),
            8,
        ),
        (b"zero\none\ntwo\nthree", 2),
        (b"zero\none\ntwo\nthree\n", 2),
        (b"zero\none", 0),
        (b"zero\none", -5),
    ],
)
def test_read_log_tail_matches_previous_text_semantics(
    tmp_path: Path, data: bytes, max_lines: int
) -> None:
    queue, task_id, _ = _queue_with_log(tmp_path, data)
    assert queue.read_log_tail(task_id, max_lines=max_lines) == _legacy_log_tail(
        data, max_lines
    )


def test_read_log_tail_default_is_last_100_lines(tmp_path: Path) -> None:
    data = ("\n".join(f"line-{index}" for index in range(105)) + "\n").encode()
    queue, task_id, _ = _queue_with_log(tmp_path, data)
    assert queue.read_log_tail(task_id) == _legacy_log_tail(data, 100)


def test_read_log_tail_handles_utf8_at_reverse_block_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(studio_tasks, "_LOG_TAIL_BLOCK_BYTES", 3)
    data = "🙂\nold\nkeep猫\nlast犬".encode()
    queue, task_id, _ = _queue_with_log(tmp_path, data)
    assert queue.read_log_tail(task_id, max_lines=2) == "keep猫\nlast犬"


def test_read_log_tail_handles_splitline_breaks_across_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(studio_tasks, "_LOG_TAIL_BLOCK_BYTES", 1)
    data = "discard\u2028old\x85keep\r\nlast\u2029".encode()
    queue, task_id, _ = _queue_with_log(tmp_path, data)
    assert queue.read_log_tail(task_id, max_lines=2) == _legacy_log_tail(data, 2)


def test_read_log_tail_replaces_invalid_utf8_and_keeps_long_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(studio_tasks, "_LOG_TAIL_BLOCK_BYTES", 7)
    data = b"discard\ninvalid-\xff\n" + ("猫🙂" * 1000).encode() + b"\xf0\x9f"
    queue, task_id, _ = _queue_with_log(tmp_path, data)
    assert queue.read_log_tail(task_id, max_lines=2) == _legacy_log_tail(data, 2)


def test_read_log_tail_preserves_lookup_and_missing_file_behavior(
    tmp_path: Path,
) -> None:
    queue, task_id, log_path = _queue_with_log(tmp_path, None)
    with pytest.raises(StudioTaskError, match="unknown task: nope"):
        queue.read_log_tail("nope")
    assert queue.read_log_tail(task_id, max_lines="invalid") == ""  # type: ignore[arg-type]
    log_path.mkdir()
    assert queue.read_log_tail(task_id) == ""


def test_read_log_tail_propagates_open_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue, task_id, log_path = _queue_with_log(tmp_path, b"line\n")
    original_open = Path.open

    def failing_open(path: Path, *args, **kwargs):
        if path == log_path:
            raise OSError("tail read failed")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(OSError, match="tail read failed"):
        queue.read_log_tail(task_id)


def test_read_log_tail_preserves_read_error_before_invalid_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue, task_id, log_path = _queue_with_log(tmp_path, b"line\n")
    original_open = Path.open

    def failing_open(path: Path, *args, **kwargs):
        if path == log_path:
            raise OSError("tail read failed first")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(OSError, match="tail read failed first"):
        queue.read_log_tail(task_id, max_lines="invalid")  # type: ignore[arg-type]


def test_read_log_tail_reads_only_sparse_file_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    block_bytes = 4096
    monkeypatch.setattr(studio_tasks, "_LOG_TAIL_BLOCK_BYTES", block_bytes)
    queue, task_id, log_path = _queue_with_log(tmp_path, None)
    separator = "\u2028".encode()
    with log_path.open("wb") as log_file:
        log_file.seek((1 << 30) - 1)
        log_file.write(
            separator
            + separator.join([b"keep-1", b"keep-2"])
        )

    original_open = Path.open
    reads: list[int] = []

    class TrackingReader:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def __enter__(self):
            self._wrapped.__enter__()
            return self

        def __exit__(self, *args):
            return self._wrapped.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

        def read(self, size: int = -1) -> bytes:
            assert 0 <= size <= block_bytes
            data = self._wrapped.read(size)
            reads.append(len(data))
            return data

    def tracked_open(path: Path, *args, **kwargs):
        opened = original_open(path, *args, **kwargs)
        if path == log_path and args and args[0] == "rb":
            return TrackingReader(opened)
        return opened

    monkeypatch.setattr(Path, "open", tracked_open)
    assert queue.read_log_tail(task_id, max_lines=2) == "keep-1\nkeep-2"
    assert reads == [block_bytes]


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
