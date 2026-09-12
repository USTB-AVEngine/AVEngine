"""The production runner: shared units, real artifacts, resources, recovery.

These exercise the same task file, verification and journal the native path
uses. What they replace is the native work itself, so a group's schedule,
readback, retry and recovery can be checked without starting a world.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from avengine.dataset import production_runner as runner
from avengine.dataset.production_resources import (
    GpuDevice,
    ResourceAllocator,
    ResourcePolicy,
    backend_profiles,
)
from avengine.dataset.production_spec import recipe_for_task_family

GROUP_ID = "test_visible_binding_g01"
ROOM_ID = "test_room"
MEMBER_IDS = [f"{GROUP_ID}_{unit}" for unit in ("v0_a0", "v0_a1", "v1_a0", "v1_a1")]
CLOCK = {"frame_count": 150, "frame_rate_hz": 15, "sample_rate_hz": 16000}


# ---------------------------------------------------------------------------
# A real manifest shape, without a real room
# ---------------------------------------------------------------------------


def _legacy_request(episode_id: str) -> dict:
    return {
        "schema": "avengine_native_qa_room_request_v1",
        "episode_id": episode_id,
        "room_id": ROOM_ID,
        "seed": 7,
        "frame_count": CLOCK["frame_count"],
        "frame_rate_hz": CLOCK["frame_rate_hz"],
        "sample_rate_hz": CLOCK["sample_rate_hz"],
        "camera": {"motion": "static", "resolution_hw": [720, 1280], "fov_deg": 85.0},
        "entities": {"total_count": 2, "silent_count": 0,
                     "source_classes": ["articulated_human", "articulated_animal"]},
        "source_asset_ids": ["asset_a", "asset_b"],
        "profile": {"separation_bin_deg": [30, 60], "anchor_count": 1,
                    "reserve_tail_s": 3.0},
        "qa_ids": ["QA-20"],
        "post_assembly_convolution_gain": 0.5,
        "task_family": "visible_binding",
        "group_id": GROUP_ID,
        "runtime": {"graphics_adapter": 0, "rpc_port": 43001},
        "production": {"retry": {"attempts_per_stage": 1},
                       "stage_resources": {"capture": {"kind": "gpu_native_visual",
                                                       "execution": "gpu"}}},
    }


def _manifest() -> dict:
    rows = []
    for index, episode_id in enumerate(MEMBER_IDS):
        rows.append({
            "episode_id": episode_id,
            "room_id": ROOM_ID,
            "group_id": GROUP_ID,
            "task_family": "visible_binding",
            "member_index": index,
            "source_classes": ["articulated_human", "articulated_animal"],
            "source_assignments": [{"actor_id": "source1", "asset_id": "asset_a"},
                                   {"actor_id": "source2", "asset_id": "asset_b"}],
            "requested_profile": {"separation_bin_deg": [30, 60]},
            "preallocation_gaps": [],
            "request": _legacy_request(episode_id),
        })
    return {
        "schema": "avengine_qa_batch_manifest_v1",
        "batch_id": "test_production_batch",
        "episodes": rows,
        "production": {
            "core_groups": [{
                "group_id": GROUP_ID,
                "task_family": "visible_binding",
                "room_id": ROOM_ID,
                "world_id": "world_test_0001",
                "member_request_ids": list(MEMBER_IDS),
            }],
            "coverage_quota": {"min_main_questions_per_qa_id": 8,
                               "min_worlds_per_qa_id": 2,
                               "min_fresh_core_worlds": 1},
        },
    }


@pytest.fixture
def manifest_path(tmp_path: Path) -> Path:
    path = tmp_path / "batch_manifest.json"
    path.write_text(json.dumps(_manifest(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _fake_room_runtime(_request):
    return {"renderer": "habitat",
            "effective": {"runtime_prefix": "/fake/habitat-prefix",
                          "mp3d_root": "/fake/mp3d"},
            "isolation_keys": ("runtime_prefix",),
            "requires_fresh_interpreter": True}


def _allocator() -> ResourceAllocator:
    devices = (GpuDevice(index=0, uuid="GPU-test-0", name="test",
                         total_memory_mb=49140, free_memory_mb=48000),)
    return ResourceAllocator(
        ResourcePolicy(backends=backend_profiles(None)),
        inventory_reader=lambda: devices,
        process_reader=lambda: (),
        port_probe=lambda port, host: True,
        descendants_reader=lambda pid: (),
    )


# ---------------------------------------------------------------------------
# Executors that write real files, without native work
# ---------------------------------------------------------------------------


def _write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


CALLS: list[str] = []


def fake_plan(context: runner.StageContext) -> runner.StageOutcome:
    CALLS.append(f"plan:{context.unit_id}")
    root = context.output_root / "episode"
    plan_path = _write(root / "plan/episode_plan.json", {
        "episode_id": f"{context.group_id}_{context.unit_id}",
        "clock": dict(CLOCK),
        "request": context.request,
        "resources": {"renderer": "habitat", "room_package": {"renderer": "habitat"}},
        "audio_events": [{"event_id": "e1", "start_sample": 0}],
        "voice_bindings": [],
    })
    return runner.StageOutcome(
        status="pass",
        facts={"episode_plan_path": str(plan_path), "renderer": "habitat",
               "clock": dict(CLOCK)},
        outputs={"episode_root": str(root),
                 "runtime_readback": context.runtime_readback(context.request)},
    )


def fake_capture(context: runner.StageContext) -> runner.StageOutcome:
    CALLS.append(f"capture:{context.unit_id}")
    recipe = recipe_for_task_family(str(context.task_family))
    plan_unit = recipe.unit(str(context.unit_id)).depends_on_units[0]
    upstream = context.upstream_unit(plan_unit)
    plan_path = Path(upstream["facts"]["episode_plan_path"])
    capture_root = context.output_root / "episode/capture"
    _write(capture_root / "frame_readbacks.json",
           {"frames": [{"index": index} for index in range(CLOCK["frame_count"])]})
    receipt = _write(capture_root / "research_receipt.json", {"status": "pass"})
    (capture_root / "ue_visual_only.mp4").write_bytes(b"video")
    return runner.StageOutcome(
        status="pass",
        facts={"capture_receipt_path": str(receipt),
               "captured_frame_count": CLOCK["frame_count"]},
        outputs={"capture_root": str(capture_root),
                 "visual_video": str(capture_root / "ue_visual_only.mp4"),
                 "plan_source": str(plan_path),
                 "runtime_readback": context.runtime_readback(context.request)},
        native_visual_worlds=1,
    )


def fake_audio(context: runner.StageContext) -> runner.StageOutcome:
    CALLS.append(f"audio:{context.unit_id}")
    recipe = recipe_for_task_family(str(context.task_family))
    unit = recipe.unit(str(context.unit_id))
    capture = context.upstream_unit(str(unit.visual_unit_id))
    intervals = [{"start_s": 2.0, "end_s": 3.4}]
    variant = context.output_root / "variant"
    facts = _write(variant / "delivery/facts.json",
                   {"episode_id": str(context.unit_id),
                    "audio": {"wet_tail_intervals": intervals}})
    report = _write(variant / "delivery/research_report.json", {"status": "pass"})
    (variant / "delivery/audio.wav").write_bytes(b"pcm")
    return runner.StageOutcome(
        status="pass",
        facts={"facts_path": str(facts), "audio_report_path": str(report),
               "wet_tail_intervals": intervals},
        outputs={"variant_root": str(variant),
                 "audio": str(variant / "delivery/audio.wav"),
                 "capture_source": (capture.get("outputs") or {}).get("capture_root"),
                 "member_id": str(context.unit_id)},
        native_acoustic_contexts=1,
    )


def fake_assembly(context: runner.StageContext) -> runner.StageOutcome:
    CALLS.append(f"assembly:{context.unit_id}")
    recipe = recipe_for_task_family(str(context.task_family))
    members = []
    for unit_id in recipe.member_unit_ids:
        result = context.upstream_unit(unit_id)
        members.append({"member_id": unit_id,
                        "question": {"qa_id": "QA-20"},
                        "facts_path": result["facts"]["facts_path"]})
    spec = _write(context.output_root / "group_spec.json",
                  {"schema": "avengine_binding_group_spec_v1", "groups": [{
                      "group_id": str(context.group_id), "world_id": "world_test_0001"}]})
    assembled = context.output_root / "assembled"
    _write(assembled / "binding_groups.json", {
        "schema": "avengine_binding_groups_v1", "group_count": 1,
        "validation": "media_checked",
        "groups": [{"group_id": str(context.group_id), "world_id": "world_test_0001",
                    "task_family": str(context.task_family), "room_id": ROOM_ID,
                    "members": members}]})
    return runner.StageOutcome(
        status="pass",
        facts={"group_spec_path": str(spec), "assembled_path": str(assembled),
               "validation": {"status": "media_checked"}},
        outputs={"assembled_root": str(assembled)},
    )


FAKE_EXECUTORS = {
    ("visible_binding", "visual_plan"): fake_plan,
    ("visible_binding", "visual_capture"): fake_capture,
    ("visible_binding", "audio"): fake_audio,
    ("visible_binding", "assembly"): fake_assembly,
}


def _inprocess_launcher(*, plan, work_dir, work_item_id_value):
    """Run the worker's own task file here instead of in a subprocess.

    The launch plan, the task file and the result file are the real ones, so
    what is skipped is the process boundary, not the contract across it.
    """
    stdout = Path(work_dir) / "stdout.log"
    stderr = Path(work_dir) / "stderr.log"
    stdout.write_text("in-process worker\n", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    code = runner.execute_task_file(Path(work_dir) / "task.json",
                                    Path(work_dir) / "stage_result.json")
    return {"pid": os.getpid(), "returncode": code, "start_ticks": None,
            "cmdline_marker": work_item_id_value, "argv": list(plan["argv"]),
            "cwd": str(plan["cwd"]), "stdout_log": str(stdout), "stderr_log": str(stderr),
            "pythonpath": (plan.get("env") or {}).get("PYTHONPATH")}


def _runner(manifest_path: Path, run_root: Path, **kwargs) -> runner.ProductionRunner:
    broker = runner.ResourceBroker(_allocator(), sleep=lambda seconds: None)
    return runner.ProductionRunner(
        manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
        manifest_path=manifest_path, run_root=run_root, broker=broker,
        launcher=_inprocess_launcher, room_resolver=_fake_room_runtime, **kwargs)


@pytest.fixture(autouse=True)
def _reset_calls():
    CALLS.clear()
    yield
    CALLS.clear()


# ---------------------------------------------------------------------------
# Group recognition and shared-unit consumption
# ---------------------------------------------------------------------------


def test_runner_consumes_shared_units_and_captures_one_visual_twice(manifest_path, tmp_path):
    with runner.stage_executor_overrides(FAKE_EXECUTORS, source="test"):
        summary = _runner(manifest_path, tmp_path / "run").run()
    assert summary["status"] == "complete", summary
    recipe = recipe_for_task_family("visible_binding")
    assert len(CALLS) == len(recipe.units) == 9
    assert sorted(CALLS) == sorted([
        "plan:v0", "plan:v1", "capture:v0_capture", "capture:v1_capture",
        "audio:v0_a0", "audio:v0_a1", "audio:v1_a0", "audio:v1_a1",
        "assembly:group"])
    # Four members, two captures. A per-row schedule would have made four.
    assert sum(1 for call in CALLS if call.startswith("capture:")) == 2
    assert summary["native_visual_worlds_used"] == 2
    assert [row["group_id"] for row in summary["delivered_groups"]] == [GROUP_ID]


def test_units_run_in_dependency_order(manifest_path, tmp_path):
    with runner.stage_executor_overrides(FAKE_EXECUTORS, source="test"):
        _runner(manifest_path, tmp_path / "run").run()
    order = {name: index for index, name in enumerate(CALLS)}
    assert order["plan:v0"] < order["capture:v0_capture"] < order["audio:v0_a0"]
    assert order["plan:v1"] < order["capture:v1_capture"] < order["audio:v1_a0"]
    assert order["assembly:group"] == len(CALLS) - 1


def test_each_unit_output_root_is_its_own_attempt(manifest_path, tmp_path):
    run_root = tmp_path / "run"
    with runner.stage_executor_overrides(FAKE_EXECUTORS, source="test"):
        _runner(manifest_path, run_root).run()
    roots = {path.name for path in (run_root / "work" / GROUP_ID).iterdir()}
    assert roots == {"v0", "v1", "v0_capture", "v1_capture", "v0_a0", "v0_a1",
                     "v1_a0", "v1_a1", "group"}
    assert (run_root / "work" / GROUP_ID / "v0/plan/attempt_01").is_dir()
    assert (run_root / "work" / GROUP_ID / "v0_capture/capture/attempt_01").is_dir()


# ---------------------------------------------------------------------------
# A pass is a claim; the artifacts are the check
# ---------------------------------------------------------------------------


def test_a_pass_without_its_artifact_is_recorded_as_a_failure(manifest_path, tmp_path):
    def lying_plan(context):
        return runner.StageOutcome(
            status="pass",
            facts={"episode_plan_path": str(context.output_root / "absent.json"),
                   "renderer": "habitat", "clock": dict(CLOCK)},
            outputs={})

    overrides = {**FAKE_EXECUTORS, ("visible_binding", "visual_plan"): lying_plan}
    with runner.stage_executor_overrides(overrides, source="test"):
        instance = _runner(manifest_path, tmp_path / "run")
        summary = instance.run()
    assert summary["status"] == "completed_with_diagnostics"
    codes = {row["code"] for row in instance.scopes[0].blockers}
    assert runner.StageOutputMissing.reason_code in codes
    assert not instance.scopes[0].results[-1]["facts"]


def test_declared_frame_count_must_match_the_readback(manifest_path, tmp_path):
    def wrong_count(context):
        outcome = fake_capture(context)
        return runner.StageOutcome(
            status="pass",
            facts={**outcome.facts, "captured_frame_count": 4},
            outputs=outcome.outputs, native_visual_worlds=1)

    overrides = {**FAKE_EXECUTORS, ("visible_binding", "visual_capture"): wrong_count}
    with runner.stage_executor_overrides(overrides, source="test"):
        instance = _runner(manifest_path, tmp_path / "run")
        instance.run()
    problems = [row for row in instance.scopes[0].blockers
                if row["code"] == runner.StageOutputMissing.reason_code]
    assert problems, instance.scopes[0].blockers
    assert any("captured_frame_count_matches_readback" in problem
               for problem in problems[0]["reason"].split(";"))


def test_declared_wet_tail_must_match_the_facts_file(manifest_path, tmp_path):
    def wrong_tail(context):
        outcome = fake_audio(context)
        return runner.StageOutcome(
            status="pass",
            facts={**outcome.facts, "wet_tail_intervals": [{"start_s": 0.0, "end_s": 9.9}]},
            outputs=outcome.outputs)

    overrides = {**FAKE_EXECUTORS, ("visible_binding", "audio"): wrong_tail}
    with runner.stage_executor_overrides(overrides, source="test"):
        instance = _runner(manifest_path, tmp_path / "run")
        instance.run()
    assert any(row["code"] == runner.StageOutputMissing.reason_code
               for row in instance.scopes[0].blockers)


def test_verification_names_the_authoritative_upstream_round():
    work_item = {"work_item_id": "g/v0_capture:capture:01", "stage": "capture",
                 "depends_on": ["g/v0:plan:01"]}
    outcome = runner.StageOutcome(status="pass",
                                  facts={"capture_receipt_path": "", "captured_frame_count": 0})
    report = runner.verify_stage_outputs(
        work_item=work_item, outcome=outcome,
        upstream={"v0": {"work_item_id": "g/v0:plan:02"}}, output_root=Path("/tmp"))
    failed = {row["check"] for row in report["checks"] if not row["passed"]}
    assert "upstream_is_the_authoritative_round" in failed


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


def test_leased_device_and_port_reach_the_request(manifest_path, tmp_path):
    seen = []

    def recording_capture(context):
        seen.append(context.runtime_readback(context.request))
        return fake_capture(context)

    overrides = {**FAKE_EXECUTORS, ("visible_binding", "visual_capture"): recording_capture}
    with runner.stage_executor_overrides(overrides, source="test"):
        _runner(manifest_path, tmp_path / "run").run()
    assert seen and all(row["matches"] for row in seen), seen
    assert all(row["request_graphics_adapter"] == row["leased_graphics_adapter"]
               for row in seen)


def test_cpu_audio_unit_does_not_hold_a_graphics_slot(manifest_path, tmp_path):
    holds = {}

    def watching_audio(context):
        holds[str(context.unit_id)] = dict(context.lease or {})
        return fake_audio(context)

    overrides = {**FAKE_EXECUTORS, ("visible_binding", "audio"): watching_audio}
    with runner.stage_executor_overrides(overrides, source="test"):
        _runner(manifest_path, tmp_path / "run").run()
    assert holds
    for unit_id, lease in holds.items():
        assert lease["device_index"] is None, (unit_id, lease)
        assert lease["reserved_vram_mb"] is None, (unit_id, lease)
        assert lease["requirement"]["lane"] == "cpu"


def test_min_free_vram_is_a_floor_and_never_the_peak_estimate(manifest_path, tmp_path):
    broker = runner.ResourceBroker(_allocator(), sleep=lambda seconds: None)
    from avengine.dataset.production_resources import WorkerCompatibility

    compatibility = WorkerCompatibility.from_runtime_report(
        _fake_room_runtime({}), python_executable="/usr/bin/python3",
        runtime_context="renderer_native")
    work_item = {
        "work_item_id": "g/v0_capture:capture:01", "stage": "capture",
        "request_id": "g/v0_capture", "attempt": 1,
        "fresh_output_relative": "g/v0_capture/capture/attempt_01",
        "resource": {"kind": "gpu_native_visual", "execution": "gpu",
                     "runtime_context": "renderer_native", "min_free_vram_mb": 1024},
    }
    request = broker.request_for(work_item, compatibility=compatibility)
    assert request.min_free_vram_mb == 1024
    assert request.requirement.estimated_peak_vram_mb == 6144
    assert request.effective_min_free_vram_mb == 1024
    with_peak = broker.request_for(work_item, compatibility=compatibility,
                                   estimated_peak_vram_mb=9000)
    assert with_peak.requirement.estimated_peak_vram_mb == 9000
    assert with_peak.min_free_vram_mb == 1024


def test_a_device_too_small_for_the_peak_stops_the_capture_with_its_reason(
        manifest_path, tmp_path):
    tiny = ResourceAllocator(
        ResourcePolicy(backends=backend_profiles(None)),
        inventory_reader=lambda: (GpuDevice(index=0, uuid="GPU-tiny", name="tiny",
                                            total_memory_mb=512, free_memory_mb=512),),
        process_reader=lambda: (),
        port_probe=lambda port, host: True)
    broker = runner.ResourceBroker(tiny, sleep=lambda seconds: None, wait_s=0.0)
    instance = runner.ProductionRunner(
        manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
        manifest_path=manifest_path, run_root=tmp_path / "run", broker=broker,
        launcher=_inprocess_launcher, room_resolver=_fake_room_runtime)
    with runner.stage_executor_overrides(FAKE_EXECUTORS, source="test"):
        summary = instance.run()
    assert summary["status"] == "completed_with_diagnostics"
    assert summary["native_visual_worlds_used"] == 0
    blockers = instance.scopes[0].blockers
    assert any(row["code"] == "no_configured_device_can_ever_fit" for row in blockers), blockers
    # The CPU planning units still ran; only the graphics unit was refused.
    assert "plan:v0" in CALLS and "capture:v0_capture" not in CALLS


def test_an_allocator_that_raises_is_reported_and_does_not_kill_the_run(
        manifest_path, tmp_path):
    """An empty inventory must be an explicit terminal resource diagnostic."""
    empty = ResourceAllocator(
        ResourcePolicy(backends=backend_profiles(None)),
        inventory_reader=lambda: (),
        process_reader=lambda: (),
        port_probe=lambda port, host: True)
    broker = runner.ResourceBroker(empty, sleep=lambda seconds: None, wait_s=0.0)
    instance = runner.ProductionRunner(
        manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
        manifest_path=manifest_path, run_root=tmp_path / "run", broker=broker,
        launcher=_inprocess_launcher, room_resolver=_fake_room_runtime)
    with runner.stage_executor_overrides(FAKE_EXECUTORS, source="test"):
        summary = instance.run()
    assert summary["status"] == "completed_with_diagnostics"
    blockers = instance.scopes[0].blockers
    assert any(row["code"] == "gpu_inventory_empty" for row in blockers), blockers
    assert any("GPU inventory is empty" in row["reason"] for row in blockers)


# ---------------------------------------------------------------------------
# Interruption and recovery
# ---------------------------------------------------------------------------


class _RunKilled(Exception):
    """Stands in for the process going away mid-unit."""


def test_resume_continues_the_group_without_repeating_passed_units(manifest_path, tmp_path):
    run_root = tmp_path / "run"

    def dying_launcher(*, plan, work_dir, work_item_id_value):
        if "v1_capture" in work_item_id_value:
            raise _RunKilled(work_item_id_value)
        return _inprocess_launcher(plan=plan, work_dir=work_dir,
                                   work_item_id_value=work_item_id_value)

    with runner.stage_executor_overrides(FAKE_EXECUTORS, source="test"):
        first = runner.ProductionRunner(
            manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
            manifest_path=manifest_path, run_root=run_root,
            broker=runner.ResourceBroker(_allocator(), sleep=lambda s: None),
            launcher=dying_launcher, room_resolver=_fake_room_runtime)
        with pytest.raises(_RunKilled):
            first.run()
    assert (run_root / "state.json").is_file()
    passed_first = sorted(row["work_item_id"] for row in first.scopes[0].results
                          if row["status"] == "pass")
    assert "test_visible_binding_g01/v0:plan:01" in passed_first
    assert "test_visible_binding_g01/v0_capture:capture:01" in passed_first
    CALLS.clear()
    with runner.stage_executor_overrides(FAKE_EXECUTORS, source="test"):
        second = runner.ProductionRunner.resume(
            run_root, broker=runner.ResourceBroker(_allocator(), sleep=lambda s: None),
            launcher=_inprocess_launcher, room_resolver=_fake_room_runtime)
        summary = second.run()
    assert summary["status"] == "complete", summary
    # The units that already passed are not produced a second time.
    assert "plan:v0" not in CALLS and "plan:v1" not in CALLS
    assert [row["group_id"] for row in summary["delivered_groups"]] == [GROUP_ID]


def test_resume_uses_only_the_valid_round(manifest_path, tmp_path):
    run_root = tmp_path / "run"
    with runner.stage_executor_overrides(FAKE_EXECUTORS, source="test"):
        first = _runner(manifest_path, run_root)
        first.run()
    scope = first.scopes[0]
    # A newer capture attempt supersedes the audio produced from the older one.
    superseded = deepcopy(next(row for row in scope.results
                               if row["work_item_id"].endswith("v0_capture:capture:01")))
    superseded["work_item_id"] = superseded["work_item_id"].replace(":01", ":02")
    scope.results.append(superseded)
    first._persist()
    state = json.loads((run_root / "state.json").read_text(encoding="utf-8"))
    valid = state["valid_round"][GROUP_ID]
    assert valid["done"]["v0_capture"].endswith(":02")
    assert set(valid["stale"]) >= {"v0_a0", "v0_a1"}
    assert "v0_a0" not in valid["done"]


def test_an_interrupted_attempt_is_recovered_without_spending_the_failure_budget(
        manifest_path, tmp_path):
    run_root = tmp_path / "run"
    killed = {"done": False}

    def flaky_launcher(*, plan, work_dir, work_item_id_value):
        if work_item_id_value.endswith("/v0:plan:01") and not killed["done"]:
            # A killed worker leaves a partial root and writes no result at all.
            killed["done"] = True
            task = json.loads((Path(work_dir) / "task.json").read_text(encoding="utf-8"))
            Path(task["context"]["output_root"], "episode/plan").mkdir(parents=True)
            stderr = Path(work_dir) / "stderr.log"
            stderr.write_text("Killed\n", encoding="utf-8")
            return {"pid": os.getpid(), "returncode": -9, "start_ticks": None,
                    "cmdline_marker": work_item_id_value,
                    "stdout_log": str(Path(work_dir) / "stdout.log"),
                    "stderr_log": str(stderr)}
        return _inprocess_launcher(plan=plan, work_dir=work_dir,
                                   work_item_id_value=work_item_id_value)

    with runner.stage_executor_overrides(FAKE_EXECUTORS, source="test"):
        instance = runner.ProductionRunner(
            manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
            manifest_path=manifest_path, run_root=run_root,
            broker=runner.ResourceBroker(_allocator(), sleep=lambda s: None),
            launcher=flaky_launcher, room_resolver=_fake_room_runtime)
        summary = instance.run()
    codes = [row["code"] for row in instance.scopes[0].blockers]
    assert runner.INTERRUPTED_REASON_CODE in codes, codes
    assert summary["status"] == "complete", (summary["status"], codes)
    assert instance.scopes[0].interrupted_attempts.get("v0", 0) == 1
    # The interrupted attempt is retained; the recovery got its own fresh root.
    assert (run_root / "work" / GROUP_ID / "v0/plan/attempt_01/episode/plan").is_dir()
    assert (run_root / "work" / GROUP_ID / "v0/plan/attempt_02").is_dir()


def test_resume_keeps_legacy_aggregate_after_adding_accounting_record(
        manifest_path, tmp_path):
    run_root = tmp_path / "run"
    first = _runner(manifest_path, run_root)
    first._persist()
    state_path = run_root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["native_visual_worlds_used"] = 2
    state["native_acoustic_contexts_used"] = 3
    state.pop("native_accounting", None)
    state.pop("native_accounting_totals", None)
    state.pop("native_accounting_carryover", None)
    state_path.write_text(json.dumps(state), encoding="utf-8")

    def resume_runner():
        return runner.ProductionRunner.resume(
            run_root,
            broker=runner.ResourceBroker(_allocator(), sleep=lambda s: None),
            launcher=_inprocess_launcher,
            room_resolver=_fake_room_runtime,
        )

    first = resume_runner()
    assert first.native_visual_worlds_used == 2
    assert first.native_acoustic_contexts_used == 3
    item = {
        "work_item_id": f"{GROUP_ID}/v0_capture:capture:01",
        "stage": "capture",
        "request_id": f"{GROUP_ID}/v0_capture",
        "attempt": 1,
        "fresh_output_relative": f"{GROUP_ID}/v0_capture/capture/attempt_01",
        "depends_on": [],
        "inputs": {"request": {}},
        "payload": {"unit_kind": "visual_capture"},
        "resource": {
            "kind": "gpu_native_visual",
            "execution": "gpu",
            "runtime_context": "renderer_native",
        },
        "group_id": GROUP_ID,
        "task_family": "visible_binding",
        "unit_id": "v0_capture",
        "member_request_ids": [],
    }
    first._ensure_native_attempt(first.scopes[0], item, None)
    first._persist()
    assert first.native_visual_worlds_used == 3
    assert first.native_acoustic_contexts_used == 3

    second = resume_runner()
    assert second.native_visual_worlds_used == 3
    assert second.native_acoustic_contexts_used == 3
    second._persist()
    third = resume_runner()
    assert third.native_visual_worlds_used == 3
    assert third.native_acoustic_contexts_used == 3


def test_a_dead_worker_is_dropped_and_a_live_one_is_adopted(manifest_path, tmp_path):
    run_root = tmp_path / "run"
    with runner.stage_executor_overrides(FAKE_EXECUTORS, source="test"):
        first = _runner(manifest_path, run_root)
        first.run()
    state = json.loads((run_root / "state.json").read_text(encoding="utf-8"))
    state["worker_records"] = {
        "ghost": {"pid": 2, "start_ticks": "1", "cmdline_marker": "ghost"},
        "live": {"pid": os.getpid(), "start_ticks": None, "cmdline_marker": ""},
    }
    (run_root / "state.json").write_text(json.dumps(state), encoding="utf-8")
    with runner.stage_executor_overrides(FAKE_EXECUTORS, source="test"):
        second = runner.ProductionRunner.resume(
            run_root, broker=runner.ResourceBroker(_allocator(), sleep=lambda s: None),
            launcher=_inprocess_launcher, room_resolver=_fake_room_runtime)
    assert second._resumed["dropped_worker_work_item_ids"] == ["ghost"]
    assert second._resumed["live_worker_work_item_ids"] == ["live"]
    assert "lease_adoption" in second._resumed


def test_process_identity_rejects_a_recycled_pid():
    assert runner.process_matches(None) is False
    assert runner.process_matches({"pid": os.getpid(), "start_ticks": "999999999"}) is False
    assert runner.process_matches({"pid": os.getpid(), "cmdline_marker": "no-such-marker"}) is False
    assert runner.process_matches({"pid": os.getpid()}) is True


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------


def test_a_program_error_is_not_retried_with_another_seed(manifest_path, tmp_path):
    seen = {"count": 0}

    def broken_plan(context):
        seen["count"] += 1
        raise AttributeError("this recipe has a defect")

    overrides = {**FAKE_EXECUTORS, ("visible_binding", "visual_plan"): broken_plan}
    with runner.stage_executor_overrides(overrides, source="test"):
        instance = _runner(manifest_path, tmp_path / "run")
        summary = instance.run()
    assert summary["status"] == "completed_with_diagnostics"
    # Both independent core plans in the ready wave are observed; the
    # interface error earns no candidate rotation or repeat.
    assert seen["count"] == 2
    assert instance.scopes[0].candidate_index == 0
    assert "not retried" in str(instance.scopes[0].finished_reason)


def test_a_refused_candidate_rotates_within_a_bound(manifest_path, tmp_path):
    seen = {"count": 0}

    def exhausted_plan(context):
        seen["count"] += 1
        raise RuntimeError(
            "ConditionedPlanningFailure: fixed condition profile exhausted "
            '{"separation": 12}')

    overrides = {**FAKE_EXECUTORS, ("visible_binding", "visual_plan"): exhausted_plan}
    with runner.stage_executor_overrides(overrides, source="test"):
        instance = _runner(manifest_path, tmp_path / "run", candidate_rotation_limit=2)
        instance.run()
    # Each candidate is a whole core round: two visual plans per candidate,
    # with no old sibling plan retained in the next round.
    assert seen["count"] == 6, seen
    assert instance.scopes[0].candidate_index == 2
    assert len(instance.scopes[0].candidate_failures) == 2
    rotations = [json.loads(line) for line in
                 (tmp_path / "run/journal.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(row.get("kind") == "candidate_rotation" for row in rotations)


def test_a_rotation_reaches_each_real_sampler_request_and_keeps_core_members_equal(
        manifest_path, tmp_path, monkeypatch):
    payloads = []
    requests = []

    def recording_plan(context):
        payloads.append(dict(context.work_item.get("payload") or {}))
        requests.append(dict(context.request))
        raise RuntimeError("ConditionedPlanningFailure: fixed condition profile exhausted {}")

    overrides = {**FAKE_EXECUTORS, ("visible_binding", "visual_plan"): recording_plan}
    with runner.stage_executor_overrides(overrides, source="test"):
        instance = _runner(
            manifest_path, tmp_path / "run", candidate_rotation_limit=1
        )
        instance.run()
    assert len(payloads) == len(requests) == 4
    assert all("candidate_index" not in payloads[i] for i in (0, 1))
    assert [requests[i].get("sampling_candidate_index") for i in (0, 1)] == [None, None]
    assert [requests[i]["sampling_candidate_index"] for i in (2, 3)] == [1, 1]
    assert [requests[i]["seed"] for i in range(4)] == [7, 7, 7, 7]
    assert all(payloads[i]["candidate_index"] == 1 for i in (2, 3))
    assert all(payloads[i]["sampling_candidate_index"] == 1 for i in (2, 3))
    journal = [
        json.loads(line)
        for line in (tmp_path / "run/journal.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(row.get("event") == "candidate_rotation_reset" for row in journal)

    # The native/state path receives the same sampler-facing field in all
    # member requests, rather than only on the owner work item.
    probe = _runner(manifest_path, tmp_path / "probe")
    probe.scopes[0].candidate_index = 1
    item = probe.next_work_items(probe.scopes[0])[0]
    context = probe._context_for(probe.scopes[0], item, None)
    seen_manifest_requests = {}

    class _Binding:
        context_call = "group_stage_context"

        def resolve(self, _name):
            def capture(manifest, _group_id, **kwargs):
                seen_manifest_requests.update({
                    row["episode_id"]: row["request"]
                    for row in manifest["episodes"]
                    if row.get("group_id") == GROUP_ID
                })
                return {
                    "member_requests": {
                        row["episode_id"]: row["request"]
                        for row in manifest["episodes"]
                        if row.get("group_id") == GROUP_ID
                    }
                }
            return capture

    monkeypatch.setattr(runner, "recipe_binding", lambda _family: _Binding())
    group_context = runner.native_group_context(context)
    assert {
        request["sampling_candidate_index"]
        for request in group_context["member_requests"].values()
    } == {1}
    assert {
        request["seed"] for request in seen_manifest_requests.values()
    } == {7}


# ---------------------------------------------------------------------------
# Recipes that are not wired yet say so
# ---------------------------------------------------------------------------


def test_state_recipe_uses_the_motion_module_entry_point():
    entry = runner.resolve_stage_executor("cross_time_state", "audio")
    assert entry.task_family == "cross_time_state"
    assert entry.source == "avengine.dataset.binding_group_motion"


def test_every_recipe_unit_kind_resolves_to_an_executor():
    for family in ("visible_binding", "visual_conditioned_relation",
                   "cross_event_identity", "cross_time_state"):
        recipe = recipe_for_task_family(family)
        for unit in recipe.units:
            entry = runner.resolve_stage_executor(family, unit.unit_kind)
            assert entry.task_family == family


# ---------------------------------------------------------------------------
# Coverage feedback and the one configuration command
# ---------------------------------------------------------------------------


def test_coverage_feedback_counts_an_imported_world_once(manifest_path, tmp_path):
    run_root = tmp_path / "run"
    with runner.stage_executor_overrides(FAKE_EXECUTORS, source="test"):
        summary = _runner(manifest_path, run_root).run()
    delivered = summary["delivered_groups"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    alone = runner.coverage_feedback(manifest=manifest, delivered_groups=delivered)
    assert alone["fresh_world_count"] == 1
    assert alone["world_count"] == 1
    same_world = Path(delivered[0]["assembled_path"]) / "binding_groups.json"
    twice = runner.coverage_feedback(manifest=manifest, delivered_groups=delivered,
                                     imported_bundle_paths=[same_world])
    assert twice["world_count"] == 1, twice["worlds"]
    other = tmp_path / "imported/binding_groups.json"
    payload = json.loads(same_world.read_text(encoding="utf-8"))
    payload["groups"][0]["world_id"] = "world_imported_0002"
    payload["groups"][0]["group_id"] = "imported_group"
    _write(other, payload)
    with_import = runner.coverage_feedback(manifest=manifest, delivered_groups=delivered,
                                           imported_bundle_paths=[other])
    assert with_import["world_count"] == 2
    assert with_import["fresh_world_count"] == 1


def test_coverage_feedback_reports_the_quota_shortfall(manifest_path, tmp_path):
    with runner.stage_executor_overrides(FAKE_EXECUTORS, source="test"):
        summary = _runner(manifest_path, tmp_path / "run").run()
    feedback = runner.coverage_feedback(
        manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
        delivered_groups=summary["delivered_groups"])
    shortfall = {row["quota_key"]: row for row in feedback["deficits"]}
    assert shortfall["min_main_questions_per_qa_id"]["achieved"] == 4
    assert shortfall["min_main_questions_per_qa_id"]["deficit"] == 4
    assert shortfall["min_worlds_per_qa_id"]["achieved"] == 1
    assert "min_fresh_core_worlds" not in shortfall


def test_a_registered_provider_replaces_the_deficit_computation(manifest_path):
    marker = {"schema": "x", "deficits": [], "world_count": 0, "fresh_world_count": 0,
              "provider": "p19"}
    runner.set_coverage_feedback_provider(lambda **kwargs: marker)
    try:
        assert runner.coverage_feedback(manifest={}, delivered_groups=[])["provider"] == "p19"
    finally:
        runner.set_coverage_feedback_provider(None)


def test_one_call_runs_generation_then_coverage_then_export(manifest_path, tmp_path):
    run_root = tmp_path / "run"
    exported = {}

    def fake_export(**kwargs):
        exported.update(kwargs)
        return {"schema": "x", "status": "exported",
                "delivery_root": str(kwargs["output"])}

    original = runner.export_run_delivery
    runner.export_run_delivery = fake_export
    try:
        with runner.stage_executor_overrides(FAKE_EXECUTORS, source="test"):
            result = runner.run_production(
                manifest_path=manifest_path, run_root=run_root,
                delivery_output=tmp_path / "delivery",
                broker=runner.ResourceBroker(_allocator(), sleep=lambda s: None),
                launcher=_inprocess_launcher, room_resolver=_fake_room_runtime)
    finally:
        runner.export_run_delivery = original
    assert result["status"] == "complete"
    assert result["coverage_feedback"]["fresh_world_count"] == 1
    assert result["delivery_export"]["status"] == "exported"
    assert (run_root / "coverage_feedback.json").is_file()
    assert (run_root / "production_result.json").is_file()
    assert exported["delivered_groups"][0]["group_id"] == GROUP_ID


def test_the_run_records_its_producer_and_worker_python_path(manifest_path, tmp_path):
    run_root = tmp_path / "run"
    with runner.stage_executor_overrides(FAKE_EXECUTORS, source="test"):
        _runner(manifest_path, run_root).run()
    producer = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    assert producer["worker_python_path"] == ["src", "tmp/native_python_addons_v1"]
    assert producer["manifest_path"] == str(manifest_path)
    assert "claim_boundary" in producer


def test_the_worker_launch_plan_carries_both_python_path_entries(manifest_path, tmp_path):
    seen = {}

    def capturing_launcher(*, plan, work_dir, work_item_id_value):
        seen.setdefault("env", dict(plan["env"]))
        seen.setdefault("argv", list(plan["argv"]))
        return _inprocess_launcher(plan=plan, work_dir=work_dir,
                                   work_item_id_value=work_item_id_value)

    broker = runner.ResourceBroker(_allocator(), sleep=lambda s: None)
    instance = runner.ProductionRunner(
        manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
        manifest_path=manifest_path, run_root=tmp_path / "run", broker=broker,
        launcher=capturing_launcher, room_resolver=_fake_room_runtime)
    with runner.stage_executor_overrides(FAKE_EXECUTORS, source="test"):
        instance.run()
    assert seen["env"]["PYTHONPATH"] == "src" + os.pathsep + "tmp/native_python_addons_v1"
    assert seen["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert seen["env"]["AVENGINE_HABITAT_RUNTIME_PREFIX"] == "/fake/habitat-prefix"
    assert seen["argv"][1] == "-B"
    assert "avengine.dataset.production_runner" in seen["argv"]


def test_native_visual_world_budget_blocks_a_further_capture(manifest_path, tmp_path):
    with runner.stage_executor_overrides(FAKE_EXECUTORS, source="test"):
        instance = _runner(manifest_path, tmp_path / "run", native_visual_world_budget=1)
        summary = instance.run()
    assert summary["native_visual_worlds_used"] == 1
    assert sum(1 for call in CALLS if call.startswith("capture:")) == 1
    assert any(row["code"] == runner.NativeBudgetExhausted.reason_code
               for row in instance.scopes[0].blockers)


# ---------------------------------------------------------------------------
# The default executor is the recipe's own per-unit entry point
# ---------------------------------------------------------------------------


def test_the_default_executor_calls_the_recipe_per_unit_entry_point(
        monkeypatch, manifest_path, tmp_path):
    from avengine.dataset import binding_group_native as native

    seen = []

    def fake_context(manifest, group_id, **kwargs):
        return {"group_id": group_id, "task_family": "visible_binding",
                "world_id": kwargs.get("world_id"), "world_id_source": "caller",
                "qa_ids": kwargs.get("qa_ids"),
                "retained_visual_roots": dict(kwargs.get("retained_visual_roots") or {}),
                "group_spec": {"group_id": group_id}, "contract": {}}

    def fake_run(item, context, *, output_root, results=(), lease=None, resume=True):
        seen.append({"work_item_id": item["work_item_id"], "unit_id": item["unit_id"],
                     "output_root": str(output_root), "result_count": len(results),
                     "lease": None if lease is None else dict(lease)})
        stage = item["stage"]
        root = Path(output_root) / item["fresh_output_relative"]
        if stage == "plan":
            plan = _write(root / "plan/episode_plan.json", {"clock": dict(CLOCK)})
            facts = {"episode_plan_path": str(plan), "renderer": "habitat",
                     "clock": dict(CLOCK)}
            outputs = {"request_path": str(_write(root / "request.json", {}))}
        else:
            return {"work_item_id": item["work_item_id"], "stage": stage,
                    "request_id": item["request_id"], "scope_id": item["request_id"],
                    "status": "blocked", "facts": {}, "outputs": {},
                    "reason": "only the plan unit is exercised here",
                    "depends_on": list(item.get("depends_on") or ())}
        return {"work_item_id": item["work_item_id"], "stage": stage,
                "request_id": item["request_id"], "scope_id": item["request_id"],
                "status": "pass", "facts": facts, "outputs": outputs, "reason": None,
                "depends_on": list(item.get("depends_on") or ())}

    monkeypatch.setattr(native, "group_stage_context", fake_context)
    monkeypatch.setattr(native, "run_group_stage_work_item", fake_run)
    instance = _runner(manifest_path, tmp_path / "run")
    instance.run()
    plans = [row for row in seen if row["unit_id"] in {"v0", "v1"}]
    assert len(plans) == 2, seen
    assert plans[0]["output_root"] == str((tmp_path / "run/work").resolve())
    assert plans[0]["lease"]["lease_id"] == "test_visible_binding_g01/v0:plan:01"
    # A planning unit takes no device and no port, so the placement is empty.
    assert plans[0]["lease"]["graphics_adapter"] is None
    assert plans[0]["lease"]["rpc_port"] is None


def test_a_retained_visual_does_not_spend_the_native_world_budget(manifest_path, tmp_path):
    retained = tmp_path / "retained/v0"
    _write(retained / "plan/episode_plan.json", {"clock": dict(CLOCK)})

    def retained_capture(context):
        assert runner.retained_root_declared_for(context) == str(retained)
        outcome = fake_capture(context)
        return runner.StageOutcome(
            status="pass", facts=outcome.facts,
            outputs={**outcome.outputs, "native_visual_worlds_created": 0},
            native_visual_worlds=0)

    overrides = {**FAKE_EXECUTORS, ("visible_binding", "visual_capture"): retained_capture}
    with runner.stage_executor_overrides(overrides, source="test"):
        instance = _runner(
            manifest_path, tmp_path / "run", native_visual_world_budget=0,
            recipe_options={GROUP_ID: {"retained_visual_roots": {"v0": str(retained),
                                                                 "v1": str(retained)}}})
        summary = instance.run()
    assert summary["native_visual_worlds_used"] == 0
    assert sum(1 for call in CALLS if call.startswith("capture:")) == 2
    assert summary["status"] == "complete", summary


def test_the_declared_world_id_reaches_the_recipe_options(manifest_path, tmp_path):
    instance = _runner(manifest_path, tmp_path / "run")
    options = instance.recipe_options_for(instance.scopes[0])
    assert options["world_id"] == "world_test_0001"


def test_an_artifact_outside_the_run_and_every_retained_root_is_flagged(tmp_path):
    outside = tmp_path / "elsewhere/thing.json"
    _write(outside, {})
    report = runner.verify_stage_outputs(
        work_item={"work_item_id": "g/v0:plan:01", "stage": "plan", "depends_on": []},
        outcome=runner.StageOutcome(
            status="pass",
            facts={"episode_plan_path": str(outside), "renderer": "habitat",
                   "clock": dict(CLOCK)},
            outputs={}),
        upstream={}, output_root=tmp_path / "run/work/g/v0/plan/attempt_01",
        run_root=tmp_path / "run")
    failed = {row["check"] for row in report["checks"] if not row["passed"]}
    assert "artifacts_stay_inside_the_run_or_a_retained_root" in failed
    report_allowed = runner.verify_stage_outputs(
        work_item={"work_item_id": "g/v0:plan:01", "stage": "plan", "depends_on": []},
        outcome=runner.StageOutcome(
            status="pass",
            facts={"episode_plan_path": str(outside), "renderer": "habitat",
                   "clock": dict(CLOCK)},
            outputs={}),
        upstream={}, output_root=tmp_path / "run/work/g/v0/plan/attempt_01",
        run_root=tmp_path / "run", retained_roots=[tmp_path / "elsewhere"])
    assert report_allowed["verified"], report_allowed["problems"]
def test_core_missing_plan_executor_never_falls_back_to_ordinary():
    keys = [
        ("cross_time_state", "plan"),
        ("cross_time_state", "late_plan"),
        ("cross_time_state", "visual_plan"),
    ]
    saved = {key: runner._EXECUTORS.pop(key)
             for key in keys if key in runner._EXECUTORS}
    try:
        for stage, unit_id in (("plan", "v0"), ("late_plan", "v1")):
            with pytest.raises(runner.StageExecutorMissing):
                runner.resolve_stage_executor(
                    "cross_time_state", stage, unit_id=unit_id
                )
    finally:
        runner._EXECUTORS.update(saved)



# --------------------------------------------------------------------------
# C02: a task-local resource policy
#
# One qualification run needed an exclusive device on a machine whose display
# server sits on every device. The fix is not to edit the shared manifest or
# to hand-write the run's state: the run takes an override of its own, records
# what it actually adopted, and a resume of the same root reapplies it.
# --------------------------------------------------------------------------

XORG_RULE = {
    "executable": "/usr/lib/xorg/Xorg",
    "uid": 128,
    "effective_uid": 0,
    "process_type": "G",
    "max_memory_mb": 16,
}

MANIFEST_POLICY = {
    "cpu": {"max_workers": 4},
    "gpu": {"max_workers": 2, "min_free_vram_mb": 12000, "headroom_mb": 2048,
            "max_workers_per_device": 1, "allow_shared_device": False},
    "ports": {"start": 41000, "count": 8},
}


def _manifest_with_policy(tmp_path: Path) -> Path:
    document = _manifest()
    document["resource_policy"] = deepcopy(MANIFEST_POLICY)
    written = tmp_path / "manifest_with_policy.json"
    written.write_text(json.dumps(document), encoding="utf-8")
    return written


def test_an_override_section_merges_and_a_list_replaces():
    merged = runner.merge_resource_policy(
        {"gpu": {"min_free_vram_mb": 12000, "headroom_mb": 2048,
                 "nonblocking_display_processes": [{"executable": "/old"}]},
         "cpu": {"max_workers": 4}},
        {"gpu": {"nonblocking_display_processes": [XORG_RULE]}},
    )
    # The untouched keys in the same section survive.
    assert merged["gpu"]["min_free_vram_mb"] == 12000
    assert merged["cpu"] == {"max_workers": 4}
    # The list is replaced, not extended: a task declares all of its
    # exceptions rather than inheriting ones it never read.
    assert merged["gpu"]["nonblocking_display_processes"] == [XORG_RULE]


def test_the_override_does_not_write_back_into_what_it_read():
    base = {"gpu": {"headroom_mb": 2048}}
    override = {"gpu": {"headroom_mb": 4096}}
    merged = runner.merge_resource_policy(base, override)
    merged["gpu"]["headroom_mb"] = 1
    assert base["gpu"]["headroom_mb"] == 2048
    assert override["gpu"]["headroom_mb"] == 4096


def test_the_run_records_the_policy_it_adopted_not_the_flag_it_was_given(tmp_path):
    manifest_path = _manifest_with_policy(tmp_path)
    run_root = tmp_path / "run"
    override = {"gpu": {"nonblocking_display_processes": [XORG_RULE]}}
    instance = runner.ProductionRunner(
        manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
        manifest_path=manifest_path, run_root=run_root,
        resource_policy_override=deepcopy(override),
        launcher=_inprocess_launcher, room_resolver=_fake_room_runtime)
    assert instance.resource_policy_source == "manifest+task_override"
    adopted = instance.broker.allocator.policy
    # The override's own value is in force ...
    assert [rule.to_dict()["executable"]
            for rule in adopted.gpu.nonblocking_display_processes] == [
        "/usr/lib/xorg/Xorg"]
    assert adopted.gpu.nonblocking_display_processes[0].effective_uid == 0
    # ... and the manifest's untouched values are still the ones in force.
    assert adopted.gpu.min_free_vram_mb == 12000
    assert adopted.gpu.allow_shared_device is False
    assert adopted.ports.start == 41000

    instance._persist()
    state = json.loads((run_root / "state.json").read_text(encoding="utf-8"))
    assert state["resource_policy_source"] == "manifest+task_override"
    assert state["resource_policy_override"] == override
    assert state["resource_policy"]["gpu"]["nonblocking_display_processes"] == [
        XORG_RULE]
    assert state["resource_policy"]["gpu"]["min_free_vram_mb"] == 12000


def test_a_resume_reapplies_the_override_without_repeating_it(tmp_path):
    manifest_path = _manifest_with_policy(tmp_path)
    run_root = tmp_path / "run"
    override = {"gpu": {"nonblocking_display_processes": [XORG_RULE],
                        "headroom_mb": 4096}}
    first = runner.ProductionRunner(
        manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
        manifest_path=manifest_path, run_root=run_root,
        resource_policy_override=deepcopy(override),
        launcher=_inprocess_launcher, room_resolver=_fake_room_runtime)
    first._persist()

    second = runner.ProductionRunner.resume(
        run_root, launcher=_inprocess_launcher, room_resolver=_fake_room_runtime)
    assert second.resource_policy_override == override
    assert second.resource_policy_source == "manifest+task_override"
    assert (second.broker.allocator.policy.to_dict()
            == first.broker.allocator.policy.to_dict())
    assert second.broker.allocator.policy.gpu.headroom_mb == 4096


def test_a_resume_that_passes_the_argument_as_none_still_reapplies_it(tmp_path):
    """The command line always passes this argument; absent means None."""
    manifest_path = _manifest_with_policy(tmp_path)
    run_root = tmp_path / "run"
    override = {"gpu": {"nonblocking_display_processes": [XORG_RULE]}}
    first = runner.ProductionRunner(
        manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
        manifest_path=manifest_path, run_root=run_root,
        resource_policy_override=deepcopy(override),
        launcher=_inprocess_launcher, room_resolver=_fake_room_runtime)
    first._persist()

    second = runner.ProductionRunner.resume(
        run_root, resource_policy_override=None,
        launcher=_inprocess_launcher, room_resolver=_fake_room_runtime)
    assert second.resource_policy_source == "manifest+task_override"
    assert second.resource_policy_override == override
    assert len(
        second.broker.allocator.policy.gpu.nonblocking_display_processes) == 1


def test_an_override_with_no_manifest_policy_is_named_as_such(tmp_path):
    manifest_path = tmp_path / "plain.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    instance = runner.ProductionRunner(
        manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
        manifest_path=manifest_path, run_root=tmp_path / "run",
        resource_policy_override={"gpu": {"allow_shared_device": False}},
        launcher=_inprocess_launcher, room_resolver=_fake_room_runtime)
    assert instance.resource_policy_source == "task_override"
    assert instance.broker.allocator.policy.gpu.allow_shared_device is False


def test_an_override_and_a_ready_made_broker_are_two_sources_for_one_policy(tmp_path):
    manifest_path = _manifest_with_policy(tmp_path)
    with pytest.raises(runner.ResourcePolicyError):
        runner.ProductionRunner(
            manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
            manifest_path=manifest_path, run_root=tmp_path / "run",
            broker=runner.ResourceBroker(_allocator(), sleep=lambda s: None),
            resource_policy_override={"gpu": {"allow_shared_device": False}},
            launcher=_inprocess_launcher, room_resolver=_fake_room_runtime)


def test_an_unrunnable_override_fails_at_the_start_not_as_a_missing_result(tmp_path):
    """A policy that cannot be executed is a program error, never a skip."""
    manifest_path = _manifest_with_policy(tmp_path)
    with pytest.raises(runner.ResourcePolicyError) as raised:
        runner.ProductionRunner(
            manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
            manifest_path=manifest_path, run_root=tmp_path / "run",
            resource_policy_override={"gpu": {"max_workers": 9}},
            launcher=_inprocess_launcher, room_resolver=_fake_room_runtime)
    assert "cpu.max_workers" in str(raised.value)


def test_a_display_rule_the_machine_cannot_mean_is_refused(tmp_path):
    manifest_path = _manifest_with_policy(tmp_path)
    with pytest.raises(runner.ResourcePolicyError):
        runner.ProductionRunner(
            manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
            manifest_path=manifest_path, run_root=tmp_path / "run",
            resource_policy_override={"gpu": {"nonblocking_display_processes": [
                {**XORG_RULE, "process_type": "C+G"}]}},
            launcher=_inprocess_launcher, room_resolver=_fake_room_runtime)


# --------------------------------------------------------------------------
# C02: a finished render whose filing was lost
#
# A controller can die between the last artifact write and the result file.
# The work is on disk; only the paperwork is missing. Re-running the unit
# would spend a native world and an RLR context reproducing audio that
# already exists, so the resume reads it back. What it must never do is turn
# a missing result into a pass: the recovered outcome goes through the same
# artifact verification as a freshly produced one.
# --------------------------------------------------------------------------


def _lose_the_filing_launcher(written: dict, *, finished: bool):
    """Do the unit's real work, then die before the result file is filed."""

    def launcher(*, plan, work_dir, work_item_id_value):
        if "v0_a1" not in work_item_id_value:
            return _inprocess_launcher(plan=plan, work_dir=work_dir,
                                       work_item_id_value=work_item_id_value)
        work_dir = Path(work_dir)
        task = json.loads((work_dir / "task.json").read_text(encoding="utf-8"))
        output_root = Path(task["context"]["output_root"])
        output_root.mkdir(parents=True, exist_ok=True)
        if finished:
            # The unit really ran and really wrote its artifacts, and the
            # native dispatcher's own stage result landed beside them.
            _inprocess_launcher(plan=plan, work_dir=work_dir,
                                work_item_id_value=work_item_id_value)
            filed = json.loads(
                (work_dir / "stage_result.json").read_text(encoding="utf-8"))
            (output_root / "stage_result.json").write_text(
                json.dumps({"schema": "avengine_native_group_stage_result_v1",
                            **filed["outcome"]}), encoding="utf-8")
            # ... and this is the write that was lost.
            (work_dir / "stage_result.json").unlink()
        else:
            (output_root / "partial_render.log").write_text(
                "cut off mid render\n", encoding="utf-8")
        written["output_root"] = str(output_root)
        written["work_item_id"] = work_item_id_value
        raise _RunKilled(work_item_id_value)

    return launcher


def _run_until_the_filing_is_lost(manifest_path, run_root, *, finished):
    written: dict = {}
    with runner.stage_executor_overrides(FAKE_EXECUTORS, source="test"):
        first = runner.ProductionRunner(
            manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
            manifest_path=manifest_path, run_root=run_root,
            broker=runner.ResourceBroker(_allocator(), sleep=lambda s: None),
            launcher=_lose_the_filing_launcher(written, finished=finished),
            room_resolver=_fake_room_runtime)
        with pytest.raises(_RunKilled):
            first.run()
    assert written, "the launcher never reached the audio unit"
    assert not (run_root / "workers"
                / written["work_item_id"].replace("/", "__").replace(":", "_")
                / "stage_result.json").exists()
    # What a killed controller leaves behind: a worker record naming a
    # process that is not running any more. The resume drops that record and
    # lists the work item as dropped, which is the shape this covers.
    state = json.loads((run_root / "state.json").read_text(encoding="utf-8"))
    assert written["work_item_id"] in (state.get("native_accounting") or {})
    records = dict(state.get("worker_records") or {})
    records[written["work_item_id"]] = {
        "pid": 2 ** 22 + 7,
        "start_ticks": 1,
        "cmdline_marker": written["work_item_id"],
        "argv": ["python", "-m", "avengine.dataset.production_runner"],
        "cwd": str(run_root),
    }
    state["worker_records"] = records
    (run_root / "state.json").write_text(json.dumps(state), encoding="utf-8")
    return written


def _resume_watching_relaunches(run_root, item_id):
    relaunched: list[str] = []

    def watching_launcher(*, plan, work_dir, work_item_id_value):
        if work_item_id_value == item_id:
            relaunched.append(work_item_id_value)
        return _inprocess_launcher(plan=plan, work_dir=work_dir,
                                   work_item_id_value=work_item_id_value)

    with runner.stage_executor_overrides(FAKE_EXECUTORS, source="test"):
        second = runner.ProductionRunner.resume(
            run_root, broker=runner.ResourceBroker(_allocator(), sleep=lambda s: None),
            launcher=watching_launcher, room_resolver=_fake_room_runtime)
        assert item_id in set(
            second._resumed["dropped_worker_work_item_ids"]), second._resumed
        summary = second.run()
    return second, summary, relaunched


def test_a_resume_reads_back_a_render_whose_result_file_was_lost(
        manifest_path, tmp_path):
    run_root = tmp_path / "run"
    written = _run_until_the_filing_is_lost(manifest_path, run_root, finished=True)
    item_id = written["work_item_id"]
    second, summary, relaunched = _resume_watching_relaunches(run_root, item_id)

    # The finished render was read back, not produced a second time.
    assert relaunched == [], relaunched
    rows = {row["work_item_id"]: row for row in second.scopes[0].results}
    assert rows[item_id]["status"] == "pass", rows[item_id]
    filed = json.loads(
        (run_root / "workers" / item_id.replace("/", "__").replace(":", "_")
         / "stage_result.json").read_text(encoding="utf-8"))
    assert filed["recovered_completed_attempt"] is True
    assert summary["status"] == "complete", summary


def test_the_readback_costs_no_new_native_work(manifest_path, tmp_path):
    """Recovery is the point: the world and the acoustic context are not paid twice."""
    run_root = tmp_path / "run"
    written = _run_until_the_filing_is_lost(manifest_path, run_root, finished=True)
    before = json.loads((run_root / "state.json").read_text(encoding="utf-8"))
    second, _, _ = _resume_watching_relaunches(run_root, written["work_item_id"])
    after = json.loads((run_root / "state.json").read_text(encoding="utf-8"))
    audio = written["work_item_id"]
    assert (after["native_accounting"][audio]["attempt"]
            == before["native_accounting"][audio]["attempt"])
    assert (after["native_acoustic_contexts_used"]
            == before["native_acoustic_contexts_used"])


def test_a_half_finished_render_is_still_reported_as_interrupted(
        manifest_path, tmp_path):
    """Recovery is a readback, never a way to pass an incomplete attempt."""
    run_root = tmp_path / "run"
    written = _run_until_the_filing_is_lost(manifest_path, run_root, finished=False)
    item_id = written["work_item_id"]
    second, _, _ = _resume_watching_relaunches(run_root, item_id)
    rows = [row for row in second.scopes[0].results
            if row["work_item_id"] == item_id]
    assert rows, "the interrupted unit left no record at all"
    assert rows[0]["status"] == "fail"
    assert "interrupted before it produced a result" in rows[0]["reason"]


def test_a_recovery_that_raises_is_journalled_and_not_a_lost_run(
        manifest_path, tmp_path, monkeypatch):
    """A recipe's own error ends that unit, not the whole run."""
    run_root = tmp_path / "run"
    written = _run_until_the_filing_is_lost(manifest_path, run_root, finished=False)
    item_id = written["work_item_id"]

    def exploding(context):
        raise RuntimeError("the recipe module disagreed")

    # Raised from inside the recovery, the way a recipe's own error type is.
    monkeypatch.setattr(runner, "native_group_context", exploding)
    second, _, _ = _resume_watching_relaunches(run_root, item_id)
    rows = [row for row in second.scopes[0].results
            if row["work_item_id"] == item_id]
    assert rows and rows[0]["status"] == "fail"
    assert "interrupted before it produced a result" in rows[0]["reason"]
    # The error is on the record under its own name, not swallowed.
    events = [json.loads(line) for line in
              (run_root / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
    failures = [e for e in events if e.get("event") == "audio_recovery_failed"]
    assert failures, [e.get("event") for e in events]
    assert "RuntimeError: the recipe module disagreed" == failures[-1]["error"]


# --------------------------------------------------------------------------
# C02-R2: re-opening a unit whose input was repaired
#
# A stage that failed because an input was missing is refused by the ordinary
# retry path, and rightly: reseeding it produces the same failure. Once the
# input is genuinely repaired that refusal is wrong, and the only ways
# forward were editing state.json by hand or starting over and redoing the
# work that already passed. Re-opening is the named, bounded third option.
# --------------------------------------------------------------------------

MISSING_INPUT = "the materialized inputs this unit needs are not there"


def _repairable_audio(repaired: dict):
    """An audio unit that fails until somebody fixes its input."""

    def executor(context):
        if not repaired.get("done"):
            raise runner.ProductionRunError(MISSING_INPUT)
        return fake_audio(context)

    return executor


def _run_to_the_failure(manifest_path, run_root, repaired):
    executors = {**FAKE_EXECUTORS,
                 ("visible_binding", "audio"): _repairable_audio(repaired)}
    with runner.stage_executor_overrides(executors, source="test"):
        first = runner.ProductionRunner(
            manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
            manifest_path=manifest_path, run_root=run_root,
            broker=runner.ResourceBroker(_allocator(), sleep=lambda s: None),
            launcher=_inprocess_launcher, room_resolver=_fake_room_runtime)
        first.run()
    return first


def test_a_repaired_unit_is_re_offered_without_redoing_what_passed(
        manifest_path, tmp_path):
    run_root = tmp_path / "run"
    repaired = {"done": False}
    first = _run_to_the_failure(manifest_path, run_root, repaired)
    failed = [row for row in first.scopes[0].results if row["status"] != "pass"]
    assert failed, [row["work_item_id"] for row in first.scopes[0].results]
    units = sorted({runner.ProductionRunner._unit_key_for_result(row)
                    for row in failed})
    passed_before = {row["work_item_id"] for row in first.scopes[0].results
                     if row["status"] == "pass"}
    assert passed_before

    repaired["done"] = True
    CALLS.clear()
    executors = {**FAKE_EXECUTORS,
                 ("visible_binding", "audio"): _repairable_audio(repaired)}
    with runner.stage_executor_overrides(executors, source="test"):
        second = runner.ProductionRunner.resume(
            run_root, broker=runner.ResourceBroker(_allocator(), sleep=lambda s: None),
            launcher=_inprocess_launcher, room_resolver=_fake_room_runtime)
        record = second.reopen_failed_units(units)
        second.run()

    assert record["status"] == "reopened"
    assert sorted(row["unit_id"] for row in record["reopened"]) == units
    assert record["unmatched"] == []
    # Nothing that already passed was produced again.
    assert not any(call.startswith("plan:") for call in CALLS), CALLS
    # The failures are retired, not deleted, and they keep their reasons.
    retired = second.scopes[0].retired_failures
    assert len(retired) == len(units)
    assert all(MISSING_INPUT in str(row["reason"]) for row in retired)
    assert all(row["retired_reason"] for row in retired)
    # The repaired unit ran again, under a fresh attempt number.
    rows = {row["work_item_id"]: row for row in second.scopes[0].results}
    reopened_ids = [key for key in rows if key.endswith(":audio:02")]
    assert reopened_ids, sorted(rows)
    assert rows[reopened_ids[0]]["status"] == "pass"
    assert passed_before <= set(rows)


def test_the_failed_attempt_directory_is_left_where_it_is(manifest_path, tmp_path):
    run_root = tmp_path / "run"
    repaired = {"done": False}
    first = _run_to_the_failure(manifest_path, run_root, repaired)
    units = sorted({runner.ProductionRunner._unit_key_for_result(row)
                    for row in first.scopes[0].results if row["status"] != "pass"})
    before = sorted(
        path.name for path in (run_root / "work").rglob("attempt_*") if path.is_dir())

    repaired["done"] = True
    executors = {**FAKE_EXECUTORS,
                 ("visible_binding", "audio"): _repairable_audio(repaired)}
    with runner.stage_executor_overrides(executors, source="test"):
        second = runner.ProductionRunner.resume(
            run_root, broker=runner.ResourceBroker(_allocator(), sleep=lambda s: None),
            launcher=_inprocess_launcher, room_resolver=_fake_room_runtime)
        second.reopen_failed_units(units)
        second.run()
    after = sorted(
        path.name for path in (run_root / "work").rglob("attempt_*") if path.is_dir())
    assert set(before) <= set(after)
    assert "attempt_02" in after


def test_re_opening_does_not_refund_what_was_already_charged(
        manifest_path, tmp_path):
    """A failed native attempt stays charged; the retry adds to it."""
    run_root = tmp_path / "run"
    repaired = {"done": False}
    first = _run_to_the_failure(manifest_path, run_root, repaired)
    units = sorted({runner.ProductionRunner._unit_key_for_result(row)
                    for row in first.scopes[0].results if row["status"] != "pass"})
    charged_before = dict(first._native_accounting)
    assert charged_before

    repaired["done"] = True
    executors = {**FAKE_EXECUTORS,
                 ("visible_binding", "audio"): _repairable_audio(repaired)}
    with runner.stage_executor_overrides(executors, source="test"):
        second = runner.ProductionRunner.resume(
            run_root, broker=runner.ResourceBroker(_allocator(), sleep=lambda s: None),
            launcher=_inprocess_launcher, room_resolver=_fake_room_runtime)
        second.reopen_failed_units(units)
        second.run()
    charged_after = dict(second._native_accounting)
    # Every attempt that was charged before is still charged, under its own id.
    assert set(charged_before) <= set(charged_after)
    assert len(charged_after) > len(charged_before)


def test_the_offset_survives_an_ordinary_resume(manifest_path, tmp_path):
    run_root = tmp_path / "run"
    repaired = {"done": False}
    first = _run_to_the_failure(manifest_path, run_root, repaired)
    failed = [row for row in first.scopes[0].results if row["status"] != "pass"][0]
    unit = runner.ProductionRunner._unit_key_for_result(failed)
    with runner.stage_executor_overrides(FAKE_EXECUTORS, source="test"):
        second = runner.ProductionRunner.resume(
            run_root, broker=runner.ResourceBroker(_allocator(), sleep=lambda s: None),
            launcher=_inprocess_launcher, room_resolver=_fake_room_runtime)
        second.reopen_failed_units([unit])
    saved = json.loads((run_root / "state.json").read_text(encoding="utf-8"))
    assert saved["scopes"][0]["retry_offsets"][unit] == 1
    assert len(saved["scopes"][0]["retired_failures"]) == 1

    third = runner.ProductionRunner.resume(
        run_root, broker=runner.ResourceBroker(_allocator(), sleep=lambda s: None),
        launcher=_inprocess_launcher, room_resolver=_fake_room_runtime)
    assert third.scopes[0].retry_offsets[unit] == 1
    assert len(third.scopes[0].retired_failures) == 1


def test_naming_a_unit_that_did_not_fail_is_an_error_not_a_no_op(
        manifest_path, tmp_path):
    run_root = tmp_path / "run"
    repaired = {"done": True}
    _run_to_the_failure(manifest_path, run_root, repaired)
    with runner.stage_executor_overrides(FAKE_EXECUTORS, source="test"):
        with pytest.raises(runner.ProductionRunError) as raised:
            runner.run_production(
                manifest_path=manifest_path, run_root=run_root, resume=True,
                reopen_failed_units=["v0"],
                broker=runner.ResourceBroker(_allocator(), sleep=lambda s: None),
                launcher=_inprocess_launcher, room_resolver=_fake_room_runtime)
    assert "no failed result to re-open" in str(raised.value)


def test_a_passing_result_is_never_retired(manifest_path, tmp_path):
    run_root = tmp_path / "run"
    repaired = {"done": False}
    first = _run_to_the_failure(manifest_path, run_root, repaired)
    passing = [row for row in first.scopes[0].results if row["status"] == "pass"]
    assert passing
    unit = runner.ProductionRunner._unit_key_for_result(passing[0])
    record = first.reopen_failed_units([unit])
    assert record["status"] == "not_run"
    assert record["unmatched"] == [unit]
    assert first.scopes[0].retired_failures == []
    assert all(row in first.scopes[0].results for row in passing)


def test_re_opening_only_some_of_the_failures_leaves_the_group_blocked(
        manifest_path, tmp_path):
    """A group with a failure still in it has nothing ready to schedule."""
    run_root = tmp_path / "run"
    repaired = {"done": False}
    first = _run_to_the_failure(manifest_path, run_root, repaired)
    failed = [row for row in first.scopes[0].results if row["status"] != "pass"]
    assert len(failed) > 1
    one = runner.ProductionRunner._unit_key_for_result(failed[0])
    record = first.reopen_failed_units([one])
    assert record["status"] == "reopened"
    # The unit is re-opened, and the group still offers nothing, because its
    # other units are still failed. Re-opening is not a way to half-start.
    assert first.next_work_items(first.scopes[0]) == []
