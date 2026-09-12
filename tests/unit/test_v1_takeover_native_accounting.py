"""T04 software-boundary tests for resource diagnostics and native accounting."""
from __future__ import annotations

import json
import os
from pathlib import Path

from avengine.dataset import production_resources as resources
from avengine.dataset import production_runner as runner


def _device(index: int = 0, *, free: int = 48000, total: int = 49140) -> resources.GpuDevice:
    return resources.GpuDevice(
        index=index,
        uuid=f"GPU-test-{index}",
        name="test",
        total_memory_mb=total,
        free_memory_mb=free,
    )


def _policy(**overrides) -> resources.ResourcePolicy:
    data = {
        "cpu": {"max_workers": 3},
        "gpu": {
            "max_workers": 2,
            "min_free_vram_mb": 1024,
            "headroom_mb": 1024,
            "max_workers_per_device": 2,
        },
        "ports": {"start": 41000, "count": 4},
        "queue": {"max_backlog": 4},
        "wait": {"poll_interval_s": 0.01, "max_wait_s": 0.05},
    }
    for key, value in overrides.items():
        if isinstance(value, dict):
            data[key] = {**data.get(key, {}), **value}
        else:
            data[key] = value
    return resources.ResourcePolicy.from_mapping(data)


def _allocator(
    inventory,
    *,
    policy: resources.ResourcePolicy | None = None,
    sleeps: list[float] | None = None,
) -> resources.ResourceAllocator:
    return resources.ResourceAllocator(
        policy or _policy(),
        inventory_reader=lambda: tuple(inventory),
        process_reader=lambda: (),
        port_probe=lambda port, host: True,
        clock=lambda: 0.0,
        sleep=(sleeps if sleeps is not None else []).append,
    )


def _worker() -> resources.WorkerCompatibility:
    return resources.WorkerCompatibility.from_runtime(
        {
            "runtime_prefix": "/fake/habitat-prefix",
            "magnum_site": "/fake/magnum-site",
            "rlr_sdk_root": "/fake/rlr",
        },
        renderer="habitat",
        python_executable="/fake/python",
    )


def _capture_request(
    allocator: resources.ResourceAllocator,
    lease_id: str = "capture",
    *,
    pinned_device_index: int | None = None,
) -> resources.LeaseRequest:
    return resources.LeaseRequest(
        lease_id=lease_id,
        compatibility=_worker(),
        requirement=allocator.policy.requirement("habitat_native_capture"),
        output_relative=f"tmp/t04/{lease_id}",
        pinned_device_index=pinned_device_index,
    )


def test_empty_inventory_is_a_terminal_diagnostic_without_max_empty():
    sleeps: list[float] = []
    allocator = _allocator([], sleeps=sleeps)
    decision = allocator.try_acquire(_capture_request(allocator))
    assert decision.status == "blocked"
    assert decision.reason_code == "gpu_inventory_empty"
    assert "GPU inventory is empty" in decision.reason
    assert "max" not in decision.reason
    assert decision.detail["reported"] == []

    second = allocator.acquire(
        _capture_request(allocator, "second"),
        max_wait_s=600.0,
        poll_interval_s=60.0,
    )
    assert second.status == "blocked"
    assert second.reason_code == "gpu_inventory_empty"
    assert sleeps == []


def test_configured_device_mismatch_is_blocked_with_reported_indices():
    allocator = _allocator([_device()], policy=_policy(gpu={"devices": [7]}))
    decision = allocator.try_acquire(_capture_request(allocator))
    assert decision.status == "blocked"
    assert decision.reason_code == "configured_device_not_present"
    assert decision.detail["configured"] == [7]
    assert decision.detail["reported"] == [0]


def test_pinned_mismatch_is_terminal_to_acquire_and_removed_from_pump():
    sleeps: list[float] = []
    allocator = _allocator([_device()], sleeps=sleeps)
    request = _capture_request(allocator, pinned_device_index=7)
    one_shot = allocator.try_acquire(request)
    assert one_shot.status == "wait"
    assert one_shot.reason_code == "pinned_device_not_available"
    assert one_shot.detail["terminal"] is True

    decision = allocator.acquire(
        _capture_request(allocator, "acquire", pinned_device_index=7),
        max_wait_s=600.0,
        poll_interval_s=60.0,
    )
    assert decision.status == "blocked"
    assert decision.reason_code == "pinned_device_not_available"
    assert sleeps == []

    allocator.submit(_capture_request(allocator, "queued", pinned_device_index=7))
    _, deferred = allocator.pump()
    assert deferred[0].detail["terminal"] is True
    assert allocator.queued_lease_ids() == ()


def _manifest(tmp_path: Path) -> tuple[dict, Path]:
    manifest = {
        "schema": "avengine_qa_batch_manifest_v1",
        "batch_id": "t04",
        "episodes": [],
        "production": {
            "core_groups": [{"group_id": "group_t04", "world_id": "world_t04"}],
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest, path


def _room_report(_request):
    return {
        "renderer": "habitat",
        "effective": {
            "runtime_prefix": "/fake/habitat-prefix",
            "magnum_site": "/fake/magnum-site",
            "rlr_sdk_root": "/fake/rlr",
        },
        "isolation_keys": ("runtime_prefix",),
        "requires_fresh_interpreter": True,
    }


def _runner(tmp_path: Path) -> runner.ProductionRunner:
    manifest, manifest_path = _manifest(tmp_path)
    allocator = _allocator([_device()])
    broker = runner.ResourceBroker(
        allocator,
        python_executable="/fake/python",
        repository=tmp_path,
        wait_s=0.0,
        sleep=lambda seconds: None,
    )
    return runner.ProductionRunner(
        manifest=manifest,
        manifest_path=manifest_path,
        run_root=tmp_path / "run",
        repository=tmp_path,
        broker=broker,
        launcher=lambda **kwargs: {
            "pid": os.getpid(),
            "returncode": 0,
        },
        room_resolver=_room_report,
    )


def _scope() -> runner.ScopeState:
    return runner.ScopeState(
        scope_kind="core_group",
        scope_key="group_t04",
        room_id="room_t04",
        task_family="visible_binding",
    )


def _item(
    item_id: str,
    *,
    unit_kind: str,
    stage: str,
    runtime_context: str,
    resource_kind: str,
) -> dict:
    return {
        "work_item_id": item_id,
        "stage": stage,
        "request_id": item_id.split(":", 1)[0],
        "attempt": 1,
        "fresh_output_relative": f"tmp/t04/{item_id.replace('/', '_')}",
        "depends_on": [],
        "inputs": {"request": {}},
        "payload": {"unit_kind": unit_kind},
        "resource": {
            "kind": resource_kind,
            "execution": "gpu" if resource_kind == "gpu_native_visual" else "cpu",
            "runtime_context": runtime_context,
        },
        "group_id": "group_t04",
        "task_family": "visible_binding",
        "unit_id": unit_id_from_item(item_id),
        "member_request_ids": [],
    }


def unit_id_from_item(item_id: str) -> str:
    return item_id.rsplit("/", 1)[-1].split(":", 1)[0]


def test_failed_capture_is_counted_before_result_and_persisted(tmp_path):
    instance = _runner(tmp_path)
    scope = _scope()
    item = _item(
        "group_t04/v0_capture:capture:01",
        unit_kind="visual_capture",
        stage="capture",
        runtime_context="renderer_native",
        resource_kind="gpu_native_visual",
    )

    def failed_launcher(**_kwargs):
        return {"pid": os.getpid(), "returncode": -9}

    instance.launcher = failed_launcher
    result = instance.execute_work_item(scope, item)
    assert result["status"] == "fail"
    assert instance.native_visual_worlds_used == 1

    state = json.loads((instance.run_root / "state.json").read_text(encoding="utf-8"))
    record = state["native_accounting"][item["work_item_id"]]
    assert record["status"] == "interrupted"
    assert record["effective_native_visual_worlds"] == 1
    assert record["effective_capture_instances"] == 1
    assert record["native_acoustic_launch_attempts"] == 0
    events = [
        json.loads(line)
        for line in instance.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert sum(event["event"] == "native_attempt_started" for event in events) == 1
    assert sum(event["event"] == "native_attempt_finalized" for event in events) == 1


def test_resume_and_repeated_finalize_do_not_double_count(tmp_path):
    instance = _runner(tmp_path)
    scope = _scope()
    item = _item(
        "group_t04/v0_capture:capture:01",
        unit_kind="visual_capture",
        stage="capture",
        runtime_context="renderer_native",
        resource_kind="gpu_native_visual",
    )
    started = instance._ensure_native_attempt(scope, item, None)
    assert started is not None
    assert instance.native_visual_worlds_used == 1

    resumed = runner.ProductionRunner.resume(
        instance.run_root,
        broker=runner.ResourceBroker(
            _allocator([_device()]),
            python_executable="/fake/python",
            repository=tmp_path,
            wait_s=0.0,
            sleep=lambda seconds: None,
        ),
        launcher=lambda **kwargs: {"pid": os.getpid(), "returncode": 0},
        room_resolver=_room_report,
    )
    assert resumed.native_visual_worlds_used == 1
    resumed._finalize_native_attempt(
        item["work_item_id"],
        status="interrupted",
        reason="test interruption",
    )
    resumed._finalize_native_attempt(
        item["work_item_id"],
        status="interrupted",
        reason="duplicate finalize",
    )
    assert resumed.native_visual_worlds_used == 1
    totals = resumed._native_accounting_totals()
    assert totals["capture_instances"] == 1
    assert totals["logical_world_identities"] == 1


def test_old_state_aggregate_counters_remain_a_baseline(tmp_path):
    instance = _runner(tmp_path)
    scope = _scope()
    first = _item(
        "group_t04/v0_capture:capture:01",
        unit_kind="visual_capture",
        stage="capture",
        runtime_context="renderer_native",
        resource_kind="gpu_native_visual",
    )
    instance._ensure_native_attempt(scope, first, None)
    state_path = instance.run_root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.pop("native_accounting", None)
    state.pop("native_accounting_totals", None)
    state_path.write_text(json.dumps(state), encoding="utf-8")

    resumed = runner.ProductionRunner.resume(
        instance.run_root,
        broker=runner.ResourceBroker(
            _allocator([_device()]),
            python_executable="/fake/python",
            repository=tmp_path,
            wait_s=0.0,
            sleep=lambda seconds: None,
        ),
        launcher=lambda **kwargs: {"pid": os.getpid(), "returncode": 0},
        room_resolver=_room_report,
    )
    assert resumed.native_visual_worlds_used == 1


def test_audio_launch_attempt_is_separate_from_real_context_count(tmp_path):
    instance = _runner(tmp_path)
    scope = _scope()
    item = _item(
        "group_t04/v0_a0:audio:01",
        unit_kind="audio",
        stage="audio",
        runtime_context="rlr_native",
        resource_kind="cpu_native_acoustic",
    )
    instance._ensure_native_attempt(scope, item, None)
    instance._finalize_native_attempt(
        item["work_item_id"],
        status="completed",
        outcome=runner.StageOutcome(
            status="pass",
            native_acoustic_contexts=1,
        ),
    )
    record = instance._native_accounting[item["work_item_id"]]
    totals = instance._native_accounting_totals()
    assert record["native_acoustic_launch_attempts"] == 1
    assert record["reported_native_acoustic_contexts"] == 1
    assert record["actual_native_acoustic_contexts"] is None
    assert totals["native_acoustic_launch_attempts"] == 1
    assert totals["native_acoustic_contexts_known"] == 0
    assert totals["native_acoustic_contexts_lower_bound"] == 0
    assert totals["native_acoustic_contexts_unknown_attempts"] == 1
    assert instance.native_acoustic_contexts_used == 0


def test_explicit_real_context_count_is_used_when_present(tmp_path):
    instance = _runner(tmp_path)
    scope = _scope()
    item = _item(
        "group_t04/v0_a0:audio:01",
        unit_kind="audio",
        stage="audio",
        runtime_context="rlr_native",
        resource_kind="cpu_native_acoustic",
    )
    instance._ensure_native_attempt(scope, item, None)
    instance._finalize_native_attempt(
        item["work_item_id"],
        status="completed",
        outcome=runner.StageOutcome(
            status="pass",
            native_acoustic_contexts=1,
            outputs={"native_context_count": 2},
        ),
    )
    totals = instance._native_accounting_totals()
    assert instance._native_accounting[item["work_item_id"]][
        "actual_native_acoustic_contexts"
    ] == 2
    assert totals["native_acoustic_contexts_known"] == 2
    assert totals["native_acoustic_contexts_unknown_attempts"] == 0


def test_logical_world_and_capture_instance_columns_stay_separate(tmp_path):
    instance = _runner(tmp_path)
    scope = _scope()
    for unit in ("v0_capture", "v1_capture"):
        instance._ensure_native_attempt(
            scope,
            _item(
                f"group_t04/{unit}:capture:01",
                unit_kind="visual_capture",
                stage="capture",
                runtime_context="renderer_native",
                resource_kind="gpu_native_visual",
            ),
            None,
        )
    totals = instance._native_accounting_totals()
    assert totals["logical_world_attempts"] is None
    assert totals["logical_world_attempts_known"] == 0
    assert totals["logical_world_attempts_unknown_capture_records"] == 2
    assert totals["logical_world_identities"] == 1
    assert totals["capture_instances"] == 2
    assert totals["native_visual_worlds"] == 2


def test_leaf_audio_log_count_is_kept_separate_from_stage_report(tmp_path):
    instance = _runner(tmp_path)
    scope = _scope()
    item = _item(
        "group_t04/v0_a0:audio:01",
        unit_kind="audio",
        stage="audio",
        runtime_context="rlr_native",
        resource_kind="cpu_native_acoustic",
    )
    stage_root = tmp_path / "stage"
    leaf_root = stage_root / "episode" / "delivery"
    leaf_root.mkdir(parents=True)
    (stage_root / "audio.log").write_text(
        "CreateContext: Context created\n" * 9,
        encoding="utf-8",
    )
    leaf_log = leaf_root / "audio.log"
    leaf_log.write_text(
        "CreateContext: Context created\nCreateContext: Context created\n",
        encoding="utf-8",
    )
    instance._ensure_native_attempt(scope, item, None)
    instance._finalize_native_attempt(
        item["work_item_id"],
        status="completed",
        outcome=runner.StageOutcome(
            status="pass",
            native_acoustic_contexts=1,
            outputs={"variant_root": str(stage_root / "episode")},
        ),
    )
    record = instance._native_accounting[item["work_item_id"]]
    totals = instance._native_accounting_totals()
    assert record["native_acoustic_launch_attempts"] == 1
    assert record["reported_native_acoustic_contexts"] == 1
    assert record["actual_native_acoustic_contexts"] == 2
    assert record["actual_native_acoustic_contexts_source"] == "stage_leaf_log"
    assert record["actual_native_acoustic_contexts_evidence_path"] == str(leaf_log.resolve())
    assert totals["native_acoustic_launch_attempts"] == 1
    assert totals["native_acoustic_contexts_known"] == 2
    assert totals["native_acoustic_contexts_unknown_attempts"] == 0


def test_outcome_none_reads_the_same_work_item_worker_log(tmp_path):
    instance = _runner(tmp_path)
    scope = _scope()
    item = _item(
        "group_t04/v0_a0:audio:01",
        unit_kind="audio",
        stage="audio",
        runtime_context="rlr_native",
        resource_kind="cpu_native_acoustic",
    )
    worker_log = tmp_path / "worker.log"
    worker_log.write_text(
        "CreateContext: Context created\nCreateContext: Context created\n",
        encoding="utf-8",
    )
    instance._ensure_native_attempt(scope, item, None)
    instance._finalize_native_attempt(
        item["work_item_id"],
        status="interrupted",
        reason="no stage result",
        run_record={"stdout_log": str(worker_log)},
    )
    record = instance._native_accounting[item["work_item_id"]]
    assert record["native_acoustic_launch_attempts"] == 1
    assert record["actual_native_acoustic_contexts"] == 2
    assert record["actual_native_acoustic_contexts_source"] == "worker_log"
    assert record["actual_native_acoustic_contexts_evidence_path"] == str(
        worker_log.resolve()
    )


# --------------------------------------------------------------------------
# C02: a shared ancestor log belongs to no single unit
#
# Several units of one group write under a common stage root, and the group's
# own log there records every context any of them opened. Reading it for one
# unit would charge that unit with its siblings' work, and charge the same
# contexts again for each sibling. A unit with no log of its own has an
# unknown count, which is a different thing from zero and from nine.
# --------------------------------------------------------------------------


def test_a_shared_group_log_is_not_charged_to_one_unit(tmp_path):
    instance = _runner(tmp_path)
    scope = _scope()
    item = _item(
        "group_t04/v0_a0:audio:01",
        unit_kind="audio",
        stage="audio",
        runtime_context="rlr_native",
        resource_kind="cpu_native_acoustic",
    )
    group_root = tmp_path / "stage"
    unit_root = group_root / "episode"
    (unit_root / "delivery").mkdir(parents=True)
    # The whole group's contexts, one directory above this unit's own root.
    (group_root / "audio.log").write_text(
        "CreateContext: Context created\n" * 9, encoding="utf-8")
    instance._ensure_native_attempt(scope, item, None)
    instance._finalize_native_attempt(
        item["work_item_id"],
        status="completed",
        outcome=runner.StageOutcome(
            status="pass",
            native_acoustic_contexts=1,
            outputs={"variant_root": str(unit_root)},
        ),
    )
    record = instance._native_accounting[item["work_item_id"]]
    assert record["actual_native_acoustic_contexts"] is None
    assert record["actual_native_acoustic_contexts_source"] is None
    assert record["actual_native_acoustic_contexts_evidence_path"] is None
    totals = instance._native_accounting_totals()
    assert totals["native_acoustic_contexts_known"] == 0
    assert totals["native_acoustic_contexts_unknown_attempts"] == 1


def test_two_units_under_one_stage_root_do_not_share_a_count(tmp_path):
    """Each unit reports its own leaf log; the group total is their sum."""
    instance = _runner(tmp_path)
    scope = _scope()
    group_root = tmp_path / "stage"
    for unit, contexts in (("v0_a0", 2), ("v0_a1", 3)):
        item = _item(
            f"group_t04/{unit}:audio:01",
            unit_kind="audio",
            stage="audio",
            runtime_context="rlr_native",
            resource_kind="cpu_native_acoustic",
        )
        leaf = group_root / unit / "episode" / "delivery"
        leaf.mkdir(parents=True)
        (leaf / "audio.log").write_text(
            "CreateContext: Context created\n" * contexts, encoding="utf-8")
        instance._ensure_native_attempt(scope, item, None)
        instance._finalize_native_attempt(
            item["work_item_id"],
            status="completed",
            outcome=runner.StageOutcome(
                status="pass",
                native_acoustic_contexts=1,
                outputs={"variant_root": str(group_root / unit / "episode")},
            ),
        )
    counts = {
        key: value["actual_native_acoustic_contexts"]
        for key, value in instance._native_accounting.items()
    }
    assert counts == {
        "group_t04/v0_a0:audio:01": 2,
        "group_t04/v0_a1:audio:01": 3,
    }
    assert instance._native_accounting_totals()[
        "native_acoustic_contexts_known"] == 5


# --------------------------------------------------------------------------
# C02-R2: what was charged and what actually opened
#
# The charge is decided before the launcher runs, so a failure after launch
# costs what it cost. Whether a world actually opened is a separate fact, and
# reading both off one number made a capture refused by a CPU precondition
# indistinguishable from one that started and crashed.
# --------------------------------------------------------------------------


def test_a_capture_refused_before_launch_is_charged_but_opened_nothing(tmp_path):
    instance = _runner(tmp_path)
    scope = _scope()
    item = _item("group_t04/v0_capture:capture:01", unit_kind="visual_capture",
                 stage="capture", runtime_context="habitat_native",
                 resource_kind="gpu_native_visual")
    attempt_root = tmp_path / "work" / "v0_capture" / "attempt_01"
    attempt_root.mkdir(parents=True)
    instance._ensure_native_attempt(scope, item, None)
    instance._finalize_native_attempt(
        item["work_item_id"], status="failed",
        reason="QAPlanningError: the materialized inputs are not there",
        outcome=runner.StageOutcome(
            status="fail", native_visual_worlds=1,
            outputs={"attempt_root": str(attempt_root)}),
    )
    record = instance._native_accounting[item["work_item_id"]]
    # Charged, because the attempt was made.
    assert record["effective_native_visual_worlds"] == 1
    # And nothing opened, on the evidence of the attempt root itself.
    assert record["actual_native_visual_worlds"] == 0
    assert record["actual_native_visual_worlds_source"] == (
        "attempt_root_without_capture_receipt")
    totals = instance._native_accounting_totals()
    assert totals["native_visual_worlds"] == 1
    assert totals["actual_native_visual_worlds_known"] == 0
    assert totals["native_visual_attempts_with_unknown_actual"] == 0


def test_a_capture_that_wrote_its_receipt_counts_as_one_that_opened(tmp_path):
    instance = _runner(tmp_path)
    scope = _scope()
    item = _item("group_t04/v1_capture:capture:01", unit_kind="visual_capture",
                 stage="capture", runtime_context="habitat_native",
                 resource_kind="gpu_native_visual")
    attempt_root = tmp_path / "work" / "v1_capture" / "attempt_01"
    (attempt_root / "capture").mkdir(parents=True)
    receipt = attempt_root / "capture" / "research_receipt.json"
    receipt.write_text(json.dumps({"status": "research_only"}), encoding="utf-8")
    instance._ensure_native_attempt(scope, item, None)
    instance._finalize_native_attempt(
        item["work_item_id"], status="completed",
        outcome=runner.StageOutcome(
            status="pass", native_visual_worlds=1,
            outputs={"attempt_root": str(attempt_root)}),
    )
    record = instance._native_accounting[item["work_item_id"]]
    assert record["effective_native_visual_worlds"] == 1
    assert record["actual_native_visual_worlds"] == 1
    assert record["actual_native_visual_worlds_evidence_path"] == str(receipt)
    assert instance._native_accounting_totals()[
        "actual_native_visual_worlds_known"] == 1


def test_an_attempt_with_no_root_on_disk_is_unknown_not_zero(tmp_path):
    instance = _runner(tmp_path)
    scope = _scope()
    item = _item("group_t04/v0_capture:capture:01", unit_kind="visual_capture",
                 stage="capture", runtime_context="habitat_native",
                 resource_kind="gpu_native_visual")
    instance._ensure_native_attempt(scope, item, None)
    instance._finalize_native_attempt(
        item["work_item_id"], status="interrupted",
        reason="the controller died", outcome=None)
    record = instance._native_accounting[item["work_item_id"]]
    assert record["effective_native_visual_worlds"] == 1
    assert record["actual_native_visual_worlds"] is None
    totals = instance._native_accounting_totals()
    assert totals["native_visual_attempts_with_unknown_actual"] == 1
    assert totals["actual_native_visual_worlds_known"] == 0


# --------------------------------------------------------------------------
# C02-R2: one lock order
#
# `_persist` takes the state lock and then, inside `state()`, the accounting
# lock. Finalizing a native attempt took the accounting lock and then called
# `_persist`. Two workers finishing at the same moment each held one and
# waited for the other: a real M23 run stopped dead with all sixty-six
# threads parked and a zombie child, after a ten-minute audio render had
# already finished. The state file is written after the accounting lock is
# released, never under it.
# --------------------------------------------------------------------------


def _accounting_item(instance, unit="v0_a0"):
    return _item(
        f"group_t04/{unit}:audio:01",
        unit_kind="audio",
        stage="audio",
        runtime_context="rlr_native",
        resource_kind="cpu_native_acoustic",
    )


def test_the_state_is_never_written_while_the_accounting_lock_is_held(tmp_path):
    instance = _runner(tmp_path)
    scope = _scope()
    seen = []
    original = instance._persist

    def watched_persist():
        # An RLock can tell us whether this thread already owns it. Owning it
        # here is exactly the inversion that deadlocked the run.
        seen.append(instance._accounting_lock._is_owned())
        return original()

    instance._persist = watched_persist
    item = _accounting_item(instance)
    instance._ensure_native_attempt(scope, item, None)
    instance._finalize_native_attempt(
        item["work_item_id"], status="completed",
        outcome=runner.StageOutcome(status="pass", native_acoustic_contexts=1))
    assert seen, "the run never persisted, so this proves nothing"
    assert not any(seen), (
        "the accounting lock was held while writing the state: "
        f"{seen}"
    )


def test_two_units_finalizing_at_once_do_not_wedge_the_runner(tmp_path):
    """The shape that stopped M23: two workers finishing together."""
    import threading

    instance = _runner(tmp_path)
    scope = _scope()
    items = [_accounting_item(instance, unit) for unit in ("v0_a0", "v0_a1")]
    for item in items:
        instance._ensure_native_attempt(scope, item, None)

    start = threading.Barrier(len(items))
    errors: list[BaseException] = []

    def finalize(item):
        try:
            start.wait(timeout=10)
            for _ in range(20):
                instance._finalize_native_attempt(
                    item["work_item_id"], status="completed",
                    outcome=runner.StageOutcome(
                        status="pass", native_acoustic_contexts=1))
                instance._persist()
        except BaseException as error:  # noqa: BLE001 - reported, not hidden
            errors.append(error)

    threads = [threading.Thread(target=finalize, args=(item,), daemon=True)
               for item in items]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    alive = [thread for thread in threads if thread.is_alive()]
    assert not alive, (
        "a finalizing thread is still running after 60s; the accounting and "
        "state locks are being taken in two different orders again"
    )
    assert not errors, errors
    totals = instance._native_accounting_totals()
    assert totals["native_acoustic_launch_attempts"] == len(items)
