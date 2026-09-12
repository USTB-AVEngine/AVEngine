"""Contracts for the V1 CPU/GPU resource allocator.

These are software tests. They prove the allocation arithmetic, the process
compatibility rule, the release path and the bounded failure reporting. They
do not prove that a Habitat or SPEAR world actually starts; that is the small
native trial recorded in this task's attempt directory, not something a mock
can stand in for.
"""
from __future__ import annotations

from dataclasses import replace
import json
import sys

import pytest

from avengine.dataset.production_resources import (
    DEFAULT_BACKEND_PROFILES,
    PER_INSTANCE_RUNTIME_KEYS,
    PROCESS_IDENTITY_RUNTIME_KEYS,
    BackendRequirement,
    GpuProcess,
    GpuDevice,
    Lease,
    LeaseRequest,
    ResourceAllocator,
    ResourcePolicy,
    ResourcePolicyError,
    ResourceUnavailable,
    WorkerCompatibility,
    assert_clean_worker_parent,
    backend_profiles,
    classified_runtime_keys,
    group_by_worker,
    lease_request_for_work_item,
    native_runtime_modules_loaded,
    worker_launch_plan,
)
from avengine.rooms.room_package import RENDERER_RUNTIME_KEYS

MP3D_PREFIX = "/data/avengine_external/runtime-prefixes/avengine-habitat-pbr-ibl-c78db29-20260821T0111Z"
HM3D_PREFIX = "/data/avengine_external/runtime-prefixes/avengine-habitat-object-id-732f264-20260824T1041Z"
MAGNUM_SITE = "/data/avengine_external/runtime-prefixes/magnum-python-cp312-45811bb-20260820T1845Z/lib/python3.12/site-packages"


def device(index, free_mb, total_mb=49140, uuid=None):
    return GpuDevice(
        index=index,
        uuid=uuid or f"GPU-fake-{index}",
        name="test device",
        total_memory_mb=total_mb,
        free_memory_mb=free_mb,
    )


class FakeMachine:
    """A driver whose free memory the test moves on purpose."""

    def __init__(self, devices, apps=()):
        self.devices = list(devices)
        self.apps = list(apps)
        self.inventory_reads = 0

    def inventory(self):
        self.inventory_reads += 1
        return tuple(self.devices)

    def processes(self):
        return tuple(self.apps)

    def set_free(self, index, free_mb):
        self.devices = [
            device(item.index, free_mb, item.total_memory_mb, item.uuid)
            if item.index == index else item
            for item in self.devices
        ]


def policy(**overrides):
    base = {
        "cpu": {"max_workers": 3},
        "gpu": {"max_workers": 2, "min_free_vram_mb": 1024, "headroom_mb": 1024,
                "max_workers_per_device": 2},
        "ports": {"start": 41000, "count": 4},
        "queue": {"max_backlog": 4},
        "wait": {"poll_interval_s": 0.01, "max_wait_s": 0.05},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return ResourcePolicy.from_mapping(base)


def allocator(machine=None, resource_policy=None, ports_free=True):
    machine = machine or FakeMachine([device(0, 40000), device(1, 40000)])
    return ResourceAllocator(
        resource_policy or policy(),
        inventory_reader=machine.inventory,
        process_reader=machine.processes,
        port_probe=lambda port, host: bool(ports_free),
        clock=lambda: 0.0,
        sleep=lambda seconds: None,
    ), machine


def habitat_worker(prefix=MP3D_PREFIX):
    return WorkerCompatibility.from_runtime(
        {"runtime_prefix": prefix, "magnum_site": MAGNUM_SITE,
         "rlr_sdk_root": "/data/avengine_external/rlr-sdk/RLRAudioPropagationPkg"},
        renderer="habitat",
        python_executable="/opt/env/bin/python",
    )


def claim(alloc, lease_id, backend, *, worker=None, output=None):
    return LeaseRequest(
        lease_id=lease_id,
        compatibility=worker or habitat_worker(),
        requirement=alloc.policy.requirement(backend),
        output_relative=output or f"tmp/p18_test/{lease_id}",
    )


# --------------------------------------------------------------------------
# What decides the device
# --------------------------------------------------------------------------


def test_a_cpu_acoustic_backend_holds_no_graphics_quota():
    """A native context is not a graphics-device requirement."""
    alloc, _ = allocator()
    requirement = alloc.policy.requirement("rlr_audio_cpu")
    assert requirement.native_context is True
    assert requirement.holds_gpu is False
    assert requirement.visual_worlds == 0

    lease = alloc.try_acquire(claim(alloc, "audio", "rlr_audio_cpu")).lease
    assert lease is not None
    assert lease.device_index is None
    assert lease.reserved_vram_mb is None
    assert alloc.snapshot()["gpu_workers_used"] == 0


def test_a_cpu_backend_may_not_declare_video_memory():
    with pytest.raises(ResourcePolicyError, match="not by itself a graphics-device"):
        BackendRequirement(backend_id="wrong", lane="cpu", estimated_peak_vram_mb=512)


def test_a_graphics_backend_must_state_its_expected_peak():
    with pytest.raises(ResourcePolicyError, match="estimated_peak_vram_mb"):
        BackendRequirement(backend_id="wrong", lane="gpu")


def test_render_jobs_worlds_and_native_contexts_are_counted_apart():
    profiles = backend_profiles(None)
    audio = profiles["rlr_audio_cpu"]
    capture = profiles["habitat_native_capture"]
    assert (audio.native_context, audio.visual_worlds, audio.render_jobs) == (True, 0, 0)
    assert (capture.native_context, capture.visual_worlds, capture.render_jobs) == (True, 1, 1)
    with pytest.raises(ResourcePolicyError, match="visual world to render"):
        BackendRequirement(backend_id="wrong", lane="gpu",
                           estimated_peak_vram_mb=1024, render_jobs=3)


def test_backend_profiles_come_from_configuration():
    """A run may state its measured peak instead of the starting estimate."""
    resolved = backend_profiles({"habitat_native_capture": {"estimated_peak_vram_mb": 2048},
                                 "my_new_backend": {"lane": "cpu", "cpu_threads": 8}})
    assert resolved["habitat_native_capture"].estimated_peak_vram_mb == 2048
    assert resolved["habitat_native_capture"].visual_worlds == 1
    assert resolved["my_new_backend"].cpu_threads == 8
    assert DEFAULT_BACKEND_PROFILES["habitat_native_capture"]["estimated_peak_vram_mb"] == 6144


# --------------------------------------------------------------------------
# Which process can run it
# --------------------------------------------------------------------------


def test_two_habitat_prefixes_are_different_workers():
    mp3d = habitat_worker(MP3D_PREFIX)
    hm3d = habitat_worker(HM3D_PREFIX)
    assert not mp3d.compatible_with(hm3d)
    assert mp3d.compatible_with(habitat_worker(MP3D_PREFIX))


def test_room_family_does_not_split_or_join_workers():
    """Family is routing data. Two families on one prefix share a worker."""
    alloc, _ = allocator()
    mp3d_family = claim(alloc, "a", "habitat_native_capture",
                        worker=habitat_worker(MP3D_PREFIX))
    other_family_same_prefix = LeaseRequest(
        lease_id="b",
        compatibility=habitat_worker(MP3D_PREFIX),
        requirement=alloc.policy.requirement("habitat_native_capture"),
        output_relative="tmp/p18_test/b",
        routing={"family": "hm3d"},
    )
    same_family_other_prefix = LeaseRequest(
        lease_id="c",
        compatibility=habitat_worker(HM3D_PREFIX),
        requirement=alloc.policy.requirement("habitat_native_capture"),
        output_relative="tmp/p18_test/c",
        routing={"family": "mp3d"},
    )
    buckets = group_by_worker([mp3d_family, other_family_same_prefix, same_family_other_prefix])
    assert len(buckets) == 2
    assert sorted(next(v for v in buckets.values() if len(v) == 2)) == ["a", "b"]


def test_an_alias_spelling_resolves_to_the_same_worker():
    """`magnum_site` and `magnum_python_site` are the same parameter."""
    aliased = WorkerCompatibility.from_runtime(
        {"runtime_prefix": MP3D_PREFIX, "magnum_site": MAGNUM_SITE},
        renderer="habitat", python_executable="/opt/env/bin/python")
    canonical = WorkerCompatibility.from_runtime(
        {"runtime_prefix": MP3D_PREFIX, "magnum_python_site": MAGNUM_SITE},
        renderer="habitat", python_executable="/opt/env/bin/python")
    assert aliased.compatible_with(canonical)


def test_a_different_interpreter_is_a_different_worker():
    first = WorkerCompatibility.from_runtime(
        {"runtime_prefix": MP3D_PREFIX}, renderer="habitat",
        python_executable="/opt/a/bin/python")
    second = WorkerCompatibility.from_runtime(
        {"runtime_prefix": MP3D_PREFIX}, renderer="habitat",
        python_executable="/opt/b/bin/python")
    assert not first.compatible_with(second)


def test_per_instance_parameters_do_not_split_workers():
    with_adapter = WorkerCompatibility.from_runtime(
        {"runtime_prefix": MP3D_PREFIX, "graphics_adapter": 2, "mp3d_root": "/data/a"},
        renderer="habitat", python_executable="/opt/env/bin/python")
    without = WorkerCompatibility.from_runtime(
        {"runtime_prefix": MP3D_PREFIX, "graphics_adapter": 3, "mp3d_root": "/data/b"},
        renderer="habitat", python_executable="/opt/env/bin/python")
    assert with_adapter.compatible_with(without)


def test_every_declared_runtime_key_is_classified():
    """A new runtime key must be declared identity or per-instance, not guessed."""
    for renderer in RENDERER_RUNTIME_KEYS:
        report = classified_runtime_keys(renderer)
        assert report["unclassified"] == (), report
        overlap = set(report["process_identity"]) & set(report["per_instance"])
        assert overlap == set(), overlap
    assert set(PROCESS_IDENTITY_RUNTIME_KEYS) == set(RENDERER_RUNTIME_KEYS)
    assert set(PER_INSTANCE_RUNTIME_KEYS) == set(RENDERER_RUNTIME_KEYS)


def test_a_native_loaded_parent_may_not_fork_a_worker():
    assert native_runtime_modules_loaded({"json": None}) == ()
    assert assert_clean_worker_parent({"json": None}) is None
    with pytest.raises(ResourceUnavailable) as error:
        assert_clean_worker_parent({"habitat_sim": None, "habitat_sim.utils": None})
    assert error.value.code == "native_runtime_already_loaded"
    assert "fresh interpreter" in str(error.value)


def test_the_worker_launch_plan_is_a_fresh_interpreter_not_a_fork():
    alloc, _ = allocator()
    lease = alloc.try_acquire(claim(alloc, "cap", "habitat_native_capture")).lease
    plan = worker_launch_plan(lease, entry=("tools/x.py", "--room", "r"),
                              repository_root="/repo", python_path=("src", "tmp/addons"))
    assert plan["start_method"] == "fresh_interpreter_subprocess"
    assert plan["argv"][:2] == ["/opt/env/bin/python", "-B"]
    assert plan["env"]["AVENGINE_HABITAT_RUNTIME_PREFIX"] == MP3D_PREFIX
    assert plan["env"]["PYTHONPATH"] == "src:tmp/addons"
    # Renumbering the visible devices would break the adapter index the
    # backends are handed, so the plan passes the index instead.
    assert "CUDA_VISIBLE_DEVICES" not in plan["env"]
    assert plan["graphics_device_index"] == lease.device_index


# --------------------------------------------------------------------------
# Competition, release and the two layers
# --------------------------------------------------------------------------


def test_two_pending_items_cannot_spend_the_same_free_memory():
    """The whole point of a run-wide ledger: free VRAM is counted once."""
    machine = FakeMachine([device(0, 14000)])
    alloc, _ = allocator(machine, policy(gpu={"headroom_mb": 1024, "max_workers_per_device": 2},
                                        backends={"big": {"lane": "gpu",
                                                          "estimated_peak_vram_mb": 8000}}))
    first = alloc.try_acquire(claim(alloc, "first", "big"))
    assert first.granted
    # The driver still reports 14000 MiB free because the first worker has not
    # started. Without the ledger the second item would see room that is gone.
    second = alloc.try_acquire(claim(alloc, "second", "big"))
    assert second.status == "wait"
    assert second.reason_code == "insufficient_free_vram"
    assert "outstanding" in second.reason
    assert alloc.effective_free_mb(machine.devices[0]) == 6000


def test_an_observed_worker_stops_being_counted_twice():
    """Once real usage shows up in the driver, only the remainder is held back."""
    machine = FakeMachine([device(0, 20000)])
    alloc, _ = allocator(machine, policy(gpu={"headroom_mb": 0, "max_workers_per_device": 2},
                                        backends={"big": {"lane": "gpu",
                                                          "estimated_peak_vram_mb": 8000}}))
    lease = alloc.try_acquire(claim(alloc, "first", "big")).lease
    assert alloc.effective_free_mb(machine.devices[0]) == 12000
    # The worker starts and the driver now shows its 3000 MiB.
    machine.set_free(0, 17000)
    machine.apps = [GpuProcess(pid=4242, gpu_uuid=machine.devices[0].uuid,
                                  used_memory_mb=3000)]
    alloc.bind_worker_pid(lease.lease_id, 4242)
    assert alloc.refresh_observations() == {"first": 3000}
    # 17000 free minus the 5000 MiB of the estimate that has not landed yet.
    assert alloc.effective_free_mb(machine.devices[0]) == 12000
    assert alloc.observed_peak_vram_mb("first") == 3000


def test_a_released_capture_returns_its_memory_to_the_next_item():
    machine = FakeMachine([device(0, 14000)])
    alloc, _ = allocator(machine, policy(gpu={"headroom_mb": 1024, "max_workers_per_device": 2},
                                        backends={"big": {"lane": "gpu",
                                                          "estimated_peak_vram_mb": 8000}}))
    first = alloc.try_acquire(claim(alloc, "first", "big")).lease
    assert alloc.try_acquire(claim(alloc, "second", "big")).status == "wait"
    alloc.release(first)
    assert alloc.try_acquire(claim(alloc, "second", "big")).granted


def test_the_cpu_stage_after_capture_does_not_keep_the_graphics_slot():
    """Delivery and CPU audio run beside the next capture, not instead of it."""
    alloc, _ = allocator(FakeMachine([device(0, 40000)]),
                         policy(gpu={"max_workers": 1, "max_workers_per_device": 1}))
    capture = alloc.try_acquire(claim(alloc, "cap1", "habitat_native_capture")).lease
    assert alloc.try_acquire(claim(alloc, "cap2", "habitat_native_capture")).reason_code == "gpu_slots_busy"
    alloc.release(capture)
    audio = alloc.try_acquire(claim(alloc, "audio1", "rlr_audio_cpu"))
    delivery = alloc.try_acquire(claim(alloc, "deliver1", "cpu_only"))
    assert audio.granted and delivery.granted
    # Both CPU stages hold no graphics quota, so the next capture starts now.
    second_capture = alloc.try_acquire(claim(alloc, "cap2", "habitat_native_capture"))
    assert second_capture.granted
    snapshot = alloc.snapshot()
    assert snapshot["gpu_workers_used"] == 1
    assert snapshot["worker_slots_used"] == 3


def test_the_two_layers_add_up_instead_of_multiplying():
    alloc, _ = allocator(FakeMachine([device(0, 40000), device(1, 40000)]),
                         policy(cpu={"max_workers": 2},
                                gpu={"max_workers": 2, "max_workers_per_device": 1}))
    assert alloc.try_acquire(claim(alloc, "g1", "habitat_native_capture")).granted
    assert alloc.try_acquire(claim(alloc, "g2", "habitat_native_capture")).granted
    # Two graphics workers already used both process slots. A third CPU item
    # waits rather than making 2 x 2 processes.
    third = alloc.try_acquire(claim(alloc, "c1", "cpu_only"))
    assert third.status == "wait"
    assert third.reason_code == "worker_slots_busy"


def test_more_graphics_workers_than_processes_is_refused_up_front():
    with pytest.raises(ResourcePolicyError, match="add up rather than multiply"):
        ResourcePolicy.from_mapping({"cpu": {"max_workers": 2}, "gpu": {"max_workers": 4}})


def test_cpu_thread_memory_and_io_budgets_are_bounded():
    alloc, _ = allocator(FakeMachine([device(0, 40000)]),
                         policy(cpu={"max_workers": 4, "max_threads": 4,
                                     "max_memory_mb": 8192, "max_io_weight": 2.0},
                                backends={"heavy": {"lane": "cpu", "cpu_threads": 3,
                                                    "memory_mb": 4096, "io_weight": 1.0}}))
    assert alloc.try_acquire(claim(alloc, "h1", "heavy")).granted
    second = alloc.try_acquire(claim(alloc, "h2", "heavy"))
    assert second.status == "wait"
    assert second.reason_code == "cpu_threads_busy"


def test_rlr_keeps_one_internal_thread_while_outer_concurrency_is_configurable():
    """Raising RLR's own thread count changes the waveform; it is not a speed knob."""
    default = ResourcePolicy.from_mapping({"cpu": {"max_workers": 8}, "gpu": {"max_workers": 2}})
    assert default.cpu.rlr_threads == 1
    assert default.cpu.max_workers == 8
    alloc, _ = allocator(FakeMachine([device(0, 40000)]))
    lease = alloc.try_acquire(claim(alloc, "audio", "rlr_audio_cpu")).lease
    plan = worker_launch_plan(lease, entry=("tools/audio.py",),
                              rlr_threads=default.cpu.rlr_threads)
    assert plan["acoustic_thread_count"] == 1
    explicit = ResourcePolicy.from_mapping({"cpu": {"rlr_threads": 4}})
    assert explicit.cpu.rlr_threads == 4


# --------------------------------------------------------------------------
# Sharing a device with other people
# --------------------------------------------------------------------------


def test_a_busy_device_with_room_is_a_valid_co_tenant():
    """Other users are read, never required to leave and never signalled."""
    machine = FakeMachine(
        [device(0, 20000, uuid="GPU-shared")],
        apps=[GpuProcess(pid=999, gpu_uuid="GPU-shared", used_memory_mb=29000)],
    )
    alloc, _ = allocator(machine, policy(gpu={"min_free_vram_mb": 8192, "headroom_mb": 1024}))
    granted = alloc.try_acquire(claim(alloc, "cap", "habitat_native_capture"))
    assert granted.granted
    assert granted.lease.device_index == 0
    assert machine.apps[0].pid == 999


def test_a_device_below_the_configured_floor_is_left_alone():
    machine = FakeMachine([device(0, 4000), device(1, 30000)])
    alloc, _ = allocator(machine, policy(gpu={"min_free_vram_mb": 8192}))
    granted = alloc.try_acquire(claim(alloc, "cap", "habitat_native_capture"))
    assert granted.lease.device_index == 1


def test_exclusive_mode_is_available_but_not_the_default():
    machine = FakeMachine(
        [device(0, 40000, uuid="GPU-shared")],
        apps=[GpuProcess(pid=999, gpu_uuid="GPU-shared", used_memory_mb=5000)],
    )
    assert ResourcePolicy.from_mapping({}).gpu.allow_shared_device is True
    alloc, _ = allocator(machine, policy(gpu={"allow_shared_device": False}))
    refused = alloc.try_acquire(claim(alloc, "cap", "habitat_native_capture"))
    assert refused.status == "wait"
    assert "allow_shared_device is false" in refused.reason


def test_the_device_is_chosen_from_configuration_not_from_code():
    machine = FakeMachine([device(0, 48000), device(1, 30000), device(2, 20000)])
    alloc, _ = allocator(machine, policy(gpu={"devices": [1, 2]}))
    granted = alloc.try_acquire(claim(alloc, "cap", "habitat_native_capture"))
    assert granted.lease.device_index == 1
    first_fit, _ = allocator(machine, policy(gpu={"devices": [2, 1], "selection": "first_fit"}))
    assert first_fit.try_acquire(claim(first_fit, "cap", "habitat_native_capture")).lease.device_index == 1


def test_a_configured_device_the_machine_does_not_report_is_named():
    machine = FakeMachine([device(0, 40000)])
    alloc, _ = allocator(machine, policy(gpu={"devices": [7]}))
    with pytest.raises(ResourceUnavailable) as error:
        alloc.candidate_devices(machine.inventory())
    assert error.value.code == "configured_device_not_present"


def test_a_request_larger_than_any_device_is_blocked_not_waited_on():
    machine = FakeMachine([device(0, 40000, total_mb=49140)])
    alloc, _ = allocator(machine, policy(backends={"huge": {"lane": "gpu",
                                                           "estimated_peak_vram_mb": 60000}}))
    decision = alloc.try_acquire(claim(alloc, "huge", "huge"))
    assert decision.status == "blocked"
    assert decision.reason_code == "no_configured_device_can_ever_fit"


def test_the_driver_is_read_again_between_the_grant_and_the_launch():
    """Someone else may take the memory in between; that is a wait, not a crash."""
    machine = FakeMachine([device(0, 20000)])
    alloc, _ = allocator(machine, policy(gpu={"headroom_mb": 1024},
                                         backends={"big": {"lane": "gpu",
                                                           "estimated_peak_vram_mb": 8000}}))
    lease = alloc.try_acquire(claim(alloc, "cap", "big")).lease
    assert alloc.recheck_before_launch(lease).granted
    machine.set_free(0, 3000)
    again = alloc.recheck_before_launch(lease)
    assert again.status == "wait"
    assert again.reason_code == "insufficient_free_vram"
    assert again.detail["available_mb"] == 3000


def test_a_missing_driver_is_reported_not_swallowed():
    def broken():
        raise ResourceUnavailable("gpu_inventory_unavailable", "cannot run nvidia-smi")

    alloc = ResourceAllocator(policy(), inventory_reader=broken,
                              process_reader=lambda: (),
                              port_probe=lambda port, host: True)
    decision = alloc.try_acquire(claim(alloc, "cap", "habitat_native_capture"))
    assert decision.status == "blocked"
    assert decision.reason_code == "gpu_inventory_unavailable"
    # A CPU item is unaffected by a graphics driver problem.
    assert alloc.try_acquire(claim(alloc, "audio", "rlr_audio_cpu")).granted


# --------------------------------------------------------------------------
# Ports and output isolation
# --------------------------------------------------------------------------


def test_two_spear_instances_never_share_a_port_or_its_files():
    alloc, _ = allocator(FakeMachine([device(0, 48000), device(1, 48000)]),
                         policy(gpu={"max_workers": 2, "max_workers_per_device": 1},
                                backends={"ue": {"lane": "gpu", "estimated_peak_vram_mb": 4096,
                                                 "needs_rpc_port": True, "visual_worlds": 1}}))
    spear = WorkerCompatibility.from_runtime(
        {"uproject": "/data/ue/SpearSim.uproject", "unreal_editor": "/opt/UE/Editor",
         "spear_ext": "/data/ue/ext"},
        renderer="ue_spear", python_executable="/opt/env/bin/python")
    first = alloc.try_acquire(claim(alloc, "ue1", "ue", worker=spear)).lease
    second = alloc.try_acquire(claim(alloc, "ue2", "ue", worker=spear)).lease
    assert first.rpc_port != second.rpc_port
    plans = [worker_launch_plan(lease, entry=("tools/ue.py",)) for lease in (first, second)]
    settings = [plan["spear_instance"] for plan in plans]
    assert settings[0]["temp_dir"] != settings[1]["temp_dir"]
    assert settings[0]["log"] != settings[1]["log"]
    assert (settings[0]["shared_memory_initial_unique_id"]
            != settings[1]["shared_memory_initial_unique_id"])
    assert settings[0]["graphics_adapter"] == first.device_index


def test_a_shared_uproject_and_cache_are_reported_as_unproven():
    """Different ports do not isolate a shared uproject or derived-data cache."""
    alloc, _ = allocator(FakeMachine([device(0, 48000)]),
                         policy(backends={"ue": {"lane": "gpu", "estimated_peak_vram_mb": 4096,
                                                 "needs_rpc_port": True, "visual_worlds": 1}}))
    spear = WorkerCompatibility.from_runtime(
        {"uproject": "/data/ue/SpearSim.uproject", "unreal_editor": "/opt/UE/Editor",
         "spear_ext": "/data/ue/ext"},
        renderer="ue_spear", python_executable="/opt/env/bin/python")
    lease = alloc.try_acquire(claim(alloc, "ue1", "ue", worker=spear)).lease
    plan = worker_launch_plan(lease, entry=("tools/ue.py",),
                              environment={"ddc_directory": "/data/ue/ddc"})
    isolation = plan["spear_isolation"]
    assert isolation["shared_between_instances"]["uproject"] == "/data/ue/SpearSim.uproject"
    assert isolation["shared_between_instances"]["ddc_directory"] == "/data/ue/ddc"
    assert isolation["shared_input_concurrency"]["status"] == "not_run"


def test_an_exhausted_port_pool_waits_with_its_reason():
    alloc, _ = allocator(FakeMachine([device(0, 48000)]),
                         policy(ports={"start": 41000, "count": 1},
                                gpu={"max_workers": 2, "max_workers_per_device": 2},
                                backends={"ue": {"lane": "gpu", "estimated_peak_vram_mb": 4096,
                                                 "needs_rpc_port": True, "visual_worlds": 1}}))
    spear = WorkerCompatibility.from_runtime(
        {"uproject": "/p", "unreal_editor": "/e", "spear_ext_dir": "/x"},
        renderer="ue_spear", python_executable="/opt/env/bin/python")
    assert alloc.try_acquire(claim(alloc, "ue1", "ue", worker=spear)).granted
    second = alloc.try_acquire(claim(alloc, "ue2", "ue", worker=spear))
    assert second.status == "wait"
    assert second.reason_code == "no_free_rpc_port"


def test_a_port_another_process_holds_is_skipped():
    taken = {41000}
    machine = FakeMachine([device(0, 48000)])
    alloc = ResourceAllocator(
        policy(ports={"start": 41000, "count": 2},
               backends={"ue": {"lane": "gpu", "estimated_peak_vram_mb": 4096,
                                "needs_rpc_port": True, "visual_worlds": 1}}),
        inventory_reader=machine.inventory,
        process_reader=machine.processes,
        port_probe=lambda port, host: port not in taken,
    )
    spear = WorkerCompatibility.from_runtime(
        {"uproject": "/p", "unreal_editor": "/e", "spear_ext_dir": "/x"},
        renderer="ue_spear", python_executable="/opt/env/bin/python")
    assert alloc.try_acquire(claim(alloc, "ue1", "ue", worker=spear)).lease.rpc_port == 41001


def test_two_live_workers_may_not_write_the_same_output_root():
    alloc, _ = allocator()
    assert alloc.try_acquire(claim(alloc, "a", "cpu_only", output="tmp/run/x")).granted
    collided = alloc.try_acquire(claim(alloc, "b", "cpu_only", output="tmp/run/x"))
    assert collided.status == "blocked"
    assert collided.reason_code == "output_collision"


def test_an_absolute_output_path_is_refused():
    alloc, _ = allocator()
    with pytest.raises(ResourcePolicyError, match="repository-relative"):
        claim(alloc, "a", "cpu_only", output="/data/elsewhere")


# --------------------------------------------------------------------------
# Queue, waiting and bounded exit
# --------------------------------------------------------------------------


def test_a_full_backlog_is_refused_with_its_reason():
    alloc, _ = allocator(FakeMachine([device(0, 40000)]), policy(queue={"max_backlog": 2}))
    alloc.submit(claim(alloc, "a", "cpu_only"))
    alloc.submit(claim(alloc, "b", "cpu_only"))
    with pytest.raises(ResourceUnavailable) as error:
        alloc.submit(claim(alloc, "c", "cpu_only"))
    assert error.value.code == "queue_backlog_full"
    assert error.value.details["lease_id"] == "c"
    assert alloc.backlog == 2


def test_the_pump_runs_the_other_lane_while_one_lane_waits():
    alloc, _ = allocator(FakeMachine([device(0, 40000)]),
                         policy(cpu={"max_workers": 4},
                                gpu={"max_workers": 1, "max_workers_per_device": 1}))
    held = alloc.try_acquire(claim(alloc, "held", "habitat_native_capture")).lease
    alloc.submit(claim(alloc, "gpu_next", "habitat_native_capture"))
    alloc.submit(claim(alloc, "gpu_after", "habitat_native_capture"))
    alloc.submit(claim(alloc, "cpu_audio", "rlr_audio_cpu"))
    granted, deferred = alloc.pump()
    assert [lease.lease_id for lease in granted] == ["cpu_audio"]
    assert alloc.queued_lease_ids() == ("gpu_next", "gpu_after")
    # The oldest graphics item is the one that reports, and the younger one is
    # not tried ahead of it.
    assert [d.reason_code for d in deferred] == ["gpu_slots_busy"]
    alloc.release(held)
    granted, _ = alloc.pump()
    assert [lease.lease_id for lease in granted] == ["gpu_next"]


def test_a_permanently_impossible_item_leaves_the_queue():
    alloc, _ = allocator(FakeMachine([device(0, 40000, total_mb=49140)]),
                         policy(backends={"huge": {"lane": "gpu",
                                                   "estimated_peak_vram_mb": 60000}}))
    alloc.submit(claim(alloc, "huge", "huge"))
    granted, deferred = alloc.pump()
    assert granted == []
    assert deferred[0].reason_code == "no_configured_device_can_ever_fit"
    assert alloc.queued_lease_ids() == ()


def test_waiting_is_bounded_and_ends_with_the_real_shortage():
    ticks = iter([0.0, 0.0, 0.02, 0.04, 0.06, 0.08, 0.10])
    machine = FakeMachine([device(0, 40000)])
    alloc = ResourceAllocator(
        policy(gpu={"max_workers": 1, "max_workers_per_device": 1},
               wait={"poll_interval_s": 0.02, "max_wait_s": 0.05}),
        inventory_reader=machine.inventory,
        process_reader=machine.processes,
        port_probe=lambda port, host: True,
        clock=lambda: next(ticks),
        sleep=lambda seconds: None,
    )
    alloc.try_acquire(claim(alloc, "held", "habitat_native_capture"))
    decision = alloc.acquire(claim(alloc, "next", "habitat_native_capture"))
    assert decision.status == "blocked"
    assert decision.reason_code == "wait_deadline_exceeded"
    assert decision.detail["last_reason_code"] == "gpu_slots_busy"
    assert decision.detail["max_wait_s"] == 0.05


def test_a_permanent_mismatch_does_not_wait_at_all():
    slept = []
    machine = FakeMachine([device(0, 40000, total_mb=49140)])
    alloc = ResourceAllocator(
        policy(backends={"huge": {"lane": "gpu", "estimated_peak_vram_mb": 60000}},
               wait={"poll_interval_s": 60.0, "max_wait_s": 600.0}),
        inventory_reader=machine.inventory,
        process_reader=machine.processes,
        port_probe=lambda port, host: True,
        clock=lambda: 0.0,
        sleep=slept.append,
    )
    decision = alloc.acquire(claim(alloc, "huge", "huge"))
    assert decision.reason_code == "no_configured_device_can_ever_fit"
    assert slept == []


def test_a_waiting_item_starts_once_the_holder_releases():
    machine = FakeMachine([device(0, 40000)])
    released = []

    def sleep(seconds):
        released.append(seconds)
        if len(released) == 2:
            alloc.release("held")

    ticks = iter([0.0] + [0.01 * index for index in range(1, 20)])
    alloc = ResourceAllocator(
        policy(gpu={"max_workers": 1, "max_workers_per_device": 1},
               wait={"poll_interval_s": 0.01, "max_wait_s": 10.0}),
        inventory_reader=machine.inventory,
        process_reader=machine.processes,
        port_probe=lambda port, host: True,
        clock=lambda: next(ticks),
        sleep=sleep,
    )
    alloc.try_acquire(claim(alloc, "held", "habitat_native_capture"))
    decision = alloc.acquire(claim(alloc, "next", "habitat_native_capture"))
    assert decision.granted
    assert len(released) == 2


def test_the_snapshot_is_writable_as_a_run_record():
    alloc, _ = allocator(FakeMachine([device(0, 40000)]))
    lease = alloc.try_acquire(claim(alloc, "cap", "habitat_native_capture")).lease
    alloc.bind_worker_pid("cap", 1234)
    alloc.report_observed_vram_mb("cap", 5120)
    snapshot = alloc.snapshot()
    assert json.loads(json.dumps(snapshot))["observed_peak_vram_mb"]["cap"] == 5120
    assert snapshot["per_device"]["0"]["workers"] == 1
    assert snapshot["per_device"]["0"]["outstanding_vram_mb"] == 6144 - 5120
    assert snapshot["worker_pids"] == {"cap": 1234}
    alloc.release(lease)
    assert alloc.snapshot()["per_device"] == {}


# --------------------------------------------------------------------------
# Consuming a stage work item
# --------------------------------------------------------------------------


class FakeWorkItem:
    """The shape P01's stage protocol hands a runner, without its resource enum."""

    def __init__(self, stage, backend_id=None, attempt=1):
        self.work_item_id = f"req_a:{stage}:{attempt:02d}"
        self.stage = stage
        self.request_id = "req_a"
        self.attempt = attempt
        self.group_id = "group_1"
        self.task_family = "cross_time_state"
        self.fresh_output_relative = f"req_a/{stage}/attempt_{attempt:02d}"
        self.payload = {} if backend_id is None else {"backend_id": backend_id}
        self.inputs = {}


def test_a_work_item_is_scheduled_by_its_backend_not_by_its_stage():
    alloc, _ = allocator(FakeMachine([device(0, 40000)]))
    capture = lease_request_for_work_item(
        FakeWorkItem("capture", "habitat_native_capture"),
        policy=alloc.policy, compatibility=habitat_worker())
    audio = lease_request_for_work_item(
        FakeWorkItem("audio", "rlr_audio_cpu"),
        policy=alloc.policy, compatibility=habitat_worker())
    assert capture.requirement.holds_gpu is True
    assert audio.requirement.holds_gpu is False
    assert audio.routing["stage"] == "audio"
    assert audio.output_relative == "req_a/audio/attempt_01"


def test_the_same_stage_may_run_on_a_cpu_or_a_gpu_backend():
    """Which acoustic backend a run picks is the fact; the stage name is not."""
    alloc, _ = allocator(FakeMachine([device(0, 40000)]))
    on_cpu = lease_request_for_work_item(
        FakeWorkItem("audio"), policy=alloc.policy,
        compatibility=habitat_worker(), backend_id="rlr_audio_cpu")
    on_gpu = lease_request_for_work_item(
        FakeWorkItem("audio"), policy=alloc.policy,
        compatibility=habitat_worker(), backend_id="rlr_audio_gpu")
    assert (on_cpu.requirement.lane, on_gpu.requirement.lane) == ("cpu", "gpu")


def test_a_work_item_without_a_backend_says_so():
    alloc, _ = allocator()
    with pytest.raises(ResourcePolicyError, match="stage name is not a device"):
        lease_request_for_work_item(FakeWorkItem("capture"), policy=alloc.policy,
                                    compatibility=habitat_worker())


def test_a_mapping_shaped_work_item_is_accepted_too():
    alloc, _ = allocator()
    item = {"work_item_id": "req_b:capture:01", "stage": "capture",
            "fresh_output_relative": "req_b/capture/attempt_01",
            "payload": {"backend_id": "cpu_only"}}
    request = lease_request_for_work_item(item, policy=alloc.policy,
                                          compatibility=habitat_worker())
    assert request.lease_id == "req_b:capture:01"


def test_an_unknown_backend_names_the_declared_ones():
    alloc, _ = allocator()
    with pytest.raises(ResourcePolicyError, match="declared backends are"):
        alloc.policy.requirement("not_a_backend")


def test_a_configured_device_that_vanished_blocks_the_item_not_the_pump():
    machine = FakeMachine([device(0, 40000)])
    alloc, _ = allocator(machine, policy(gpu={"devices": [7]}))
    alloc.submit(claim(alloc, "cap", "habitat_native_capture"))
    alloc.submit(claim(alloc, "audio", "rlr_audio_cpu"))
    granted, deferred = alloc.pump()
    assert [lease.lease_id for lease in granted] == ["audio"]
    assert deferred[0].reason_code == "configured_device_not_present"


def test_concurrent_callers_do_not_oversubscribe_the_slots():
    """Two threads asking at once still get one lease each, never three."""
    import threading

    machine = FakeMachine([device(0, 40000)])
    alloc, _ = allocator(machine, policy(cpu={"max_workers": 2}))
    outcomes: list[str] = []
    lock = threading.Lock()

    def ask(index):
        decision = alloc.try_acquire(claim(alloc, f"w{index}", "cpu_only"))
        with lock:
            outcomes.append(decision.status)

    threads = [threading.Thread(target=ask, args=(index,)) for index in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert outcomes.count("granted") == 2
    assert outcomes.count("wait") == 4
    assert len(alloc.live_leases()) == 2


def test_a_graphics_only_worker_is_still_attributed():
    """A headless Habitat worker is a graphics process, not a compute one.

    A compute-only driver query returns nothing for it, so its reservation
    would be held back forever and its peak would read as zero.
    """
    machine = FakeMachine([device(0, 20000, uuid="GPU-shared")])
    alloc, _ = allocator(machine, policy(gpu={"headroom_mb": 0},
                                         backends={"big": {"lane": "gpu",
                                                           "estimated_peak_vram_mb": 8000}}))
    lease = alloc.try_acquire(claim(alloc, "cap", "big")).lease
    alloc.bind_worker_pid("cap", 5150)
    machine.apps = [GpuProcess(pid=5150, gpu_uuid="GPU-shared",
                               used_memory_mb=2600, process_type="G")]
    assert alloc.refresh_observations() == {"cap": 2600}
    assert alloc.observed_peak_vram_mb("cap") == 2600


def test_one_process_holding_two_contexts_is_summed_once():
    """The driver lists a C+G worker once per device it touches."""
    machine = FakeMachine([device(0, 20000, uuid="GPU-a"), device(1, 20000, uuid="GPU-b")])
    alloc, _ = allocator(machine, policy(gpu={"headroom_mb": 0},
                                         backends={"big": {"lane": "gpu",
                                                           "estimated_peak_vram_mb": 8000}}))
    alloc.try_acquire(claim(alloc, "cap", "big"))
    alloc.bind_worker_pid("cap", 6100)
    machine.apps = [
        GpuProcess(pid=6100, gpu_uuid="GPU-a", used_memory_mb=1500, process_type="C+G"),
        GpuProcess(pid=6100, gpu_uuid="GPU-b", used_memory_mb=900, process_type="G"),
    ]
    assert alloc.refresh_observations() == {"cap": 2400}


def test_a_peak_is_kept_once_it_has_been_seen():
    machine = FakeMachine([device(0, 20000, uuid="GPU-a")])
    alloc, _ = allocator(machine, policy(gpu={"headroom_mb": 0},
                                         backends={"big": {"lane": "gpu",
                                                           "estimated_peak_vram_mb": 8000}}))
    alloc.try_acquire(claim(alloc, "cap", "big"))
    alloc.bind_worker_pid("cap", 7000)
    machine.apps = [GpuProcess(pid=7000, gpu_uuid="GPU-a", used_memory_mb=5000)]
    alloc.refresh_observations()
    machine.apps = [GpuProcess(pid=7000, gpu_uuid="GPU-a", used_memory_mb=1200)]
    alloc.refresh_observations()
    assert alloc.observed_peak_vram_mb("cap") == 5000


# --------------------------------------------------------------------------
# Reading the stage protocol's own two axes
# --------------------------------------------------------------------------


class FakeResourceRequest:
    """The shape the stage protocol's resource request exposes."""

    def __init__(self, kind, execution, runtime_context, **fields):
        self.kind = kind
        self.execution = execution
        self.runtime_context = runtime_context
        self.min_free_vram_mb = fields.get("min_free_vram_mb")
        # Not a field the stage protocol declares today. It is read by name so
        # that when a measured peak becomes available it has somewhere honest
        # to go, instead of being smuggled in through the floor.
        self.estimated_peak_vram_mb = fields.get("estimated_peak_vram_mb")
        self.graphics_adapter = fields.get("graphics_adapter")
        self.rpc_port = fields.get("rpc_port")
        self.rlr_threads = fields.get("rlr_threads")
        self.max_parallel = fields.get("max_parallel")


def routed_item(stage, kind, execution, runtime_context, **fields):
    item = FakeWorkItem(stage)
    item.resource = FakeResourceRequest(kind, execution, runtime_context, **fields)
    return item


def test_the_declared_execution_slot_and_runtime_context_pick_the_backend():
    alloc, _ = allocator()
    on_cpu = lease_request_for_work_item(
        routed_item("audio", "cpu_native_acoustic", "cpu", "rlr_native"),
        policy=alloc.policy, compatibility=habitat_worker())
    on_gpu = lease_request_for_work_item(
        routed_item("audio", "gpu_native_acoustic", "gpu", "rlr_native"),
        policy=alloc.policy, compatibility=habitat_worker())
    assert on_cpu.requirement.backend_id == "rlr_audio_cpu"
    assert on_cpu.requirement.holds_gpu is False
    assert on_cpu.requirement.native_context is True
    assert on_gpu.requirement.backend_id == "rlr_audio_gpu"
    assert on_gpu.requirement.holds_gpu is True
    assert on_cpu.routing["resource_kind"] == "cpu_native_acoustic"


def test_a_habitat_geometry_stage_holds_a_native_context_and_no_device():
    alloc, _ = allocator()
    claim = lease_request_for_work_item(
        routed_item("late_plan", "cpu_native_geometry", "cpu", "habitat_native"),
        policy=alloc.policy, compatibility=habitat_worker())
    assert claim.requirement.backend_id == "habitat_native_cpu"
    assert (claim.requirement.native_context, claim.requirement.visual_worlds) == (True, 0)
    assert claim.requirement.holds_gpu is False


def test_the_renderer_separates_two_capture_backends():
    alloc, _ = allocator()
    spear = WorkerCompatibility.from_runtime(
        {"uproject": "/p", "unreal_editor": "/e", "spear_ext_dir": "/x"},
        renderer="ue_spear", python_executable="/opt/env/bin/python")
    item = routed_item("capture", "gpu_native_visual", "gpu", "renderer_native")
    on_habitat = lease_request_for_work_item(item, policy=alloc.policy,
                                             compatibility=habitat_worker())
    on_spear = lease_request_for_work_item(item, policy=alloc.policy,
                                           compatibility=spear)
    assert on_habitat.requirement.backend_id == "habitat_native_capture"
    assert on_spear.requirement.backend_id == "ue_spear_capture"
    assert on_spear.requirement.needs_rpc_port is True


def test_an_unmapped_runtime_profile_is_named_not_guessed():
    alloc, _ = allocator()
    with pytest.raises(ResourcePolicyError, match="backend_by_runtime_profile"):
        lease_request_for_work_item(
            routed_item("capture", "gpu_native_quantum", "gpu", "quantum_native"),
            policy=alloc.policy, compatibility=habitat_worker())


def test_a_run_may_retarget_one_runtime_profile_from_configuration():
    retargeted = ResourcePolicy.from_mapping(
        {"backend_by_runtime_profile": {"cpu/rlr_native": "rlr_audio_gpu"}})
    assert retargeted.requirement_for_runtime_profile("cpu", "rlr_native").backend_id == "rlr_audio_gpu"
    assert ResourcePolicy.from_mapping({}).requirement_for_runtime_profile(
        "cpu", "rlr_native").backend_id == "rlr_audio_cpu"


def test_an_acoustic_worker_and_a_geometry_worker_are_two_processes():
    """RLR must be imported before Habitat, so the context is part of the key."""
    runtime = {"runtime_prefix": MP3D_PREFIX, "magnum_site": MAGNUM_SITE}
    acoustic = WorkerCompatibility.from_runtime(
        runtime, renderer="habitat", python_executable="/opt/env/bin/python",
        runtime_context="rlr_native")
    geometry = WorkerCompatibility.from_runtime(
        runtime, renderer="habitat", python_executable="/opt/env/bin/python",
        runtime_context="habitat_native")
    assert not acoustic.compatible_with(geometry)
    assert acoustic.to_dict()["runtime_context"] == "rlr_native"


def test_a_declared_start_up_floor_is_not_a_peak_estimate():
    """A worker allowed to start on a nearly full device is not a small worker.

    Reading a floor as a peak under-reserves: the run would hold back the
    floor instead of what the worker actually uses, and a co-tenant would be
    handed memory this worker is about to take.
    """
    alloc, _ = allocator()
    claim = lease_request_for_work_item(
        routed_item("capture", "gpu_native_visual", "gpu", "renderer_native",
                    min_free_vram_mb=1024),
        policy=alloc.policy, compatibility=habitat_worker())
    assert claim.requirement.estimated_peak_vram_mb == 6144
    assert claim.min_free_vram_mb == 1024
    assert alloc.policy.requirement("habitat_native_capture").estimated_peak_vram_mb == 6144


def test_a_better_peak_estimate_is_passed_under_its_own_name():
    alloc, _ = allocator()
    explicit = lease_request_for_work_item(
        routed_item("capture", "gpu_native_visual", "gpu", "renderer_native"),
        policy=alloc.policy, compatibility=habitat_worker(),
        estimated_peak_vram_mb=20000)
    assert explicit.requirement.estimated_peak_vram_mb == 20000
    declared = lease_request_for_work_item(
        routed_item("capture", "gpu_native_visual", "gpu", "renderer_native",
                    estimated_peak_vram_mb=17000),
        policy=alloc.policy, compatibility=habitat_worker())
    assert declared.requirement.estimated_peak_vram_mb == 17000
    # The starting estimate in the profile is untouched by either route.
    assert alloc.policy.requirement("habitat_native_capture").estimated_peak_vram_mb == 6144


def test_a_cpu_item_may_not_declare_a_peak_at_all():
    alloc, _ = allocator()
    with pytest.raises(ResourcePolicyError, match="runs on the CPU"):
        lease_request_for_work_item(
            routed_item("audio", "cpu_native_acoustic", "cpu", "rlr_native"),
            policy=alloc.policy, compatibility=habitat_worker(),
            estimated_peak_vram_mb=4096)


def test_a_saved_request_may_pin_its_device_and_port():
    machine = FakeMachine([device(0, 48000), device(1, 48000)])
    alloc, _ = allocator(machine, policy(gpu={"max_workers": 2, "max_workers_per_device": 2}))
    spear = WorkerCompatibility.from_runtime(
        {"uproject": "/p", "unreal_editor": "/e", "spear_ext_dir": "/x"},
        renderer="ue_spear", python_executable="/opt/env/bin/python")
    claim = lease_request_for_work_item(
        routed_item("capture", "gpu_native_visual", "gpu", "renderer_native",
                    graphics_adapter=1, rpc_port=41002),
        policy=alloc.policy, compatibility=spear)
    lease = alloc.try_acquire(claim).lease
    assert (lease.device_index, lease.rpc_port) == (1, 41002)


def test_a_pinned_device_that_is_not_configured_says_so():
    machine = FakeMachine([device(0, 48000), device(1, 48000)])
    alloc, _ = allocator(machine, policy(gpu={"devices": [0]}))
    claim = lease_request_for_work_item(
        routed_item("capture", "gpu_native_visual", "gpu", "renderer_native",
                    graphics_adapter=1),
        policy=alloc.policy, compatibility=habitat_worker())
    decision = alloc.try_acquire(claim)
    assert decision.status == "wait"
    assert decision.reason_code == "pinned_device_not_available"


def test_a_per_backend_cap_applies_under_the_layer_caps():
    alloc, _ = allocator(FakeMachine([device(0, 48000)]),
                         policy(cpu={"max_workers": 4}))
    first = lease_request_for_work_item(
        routed_item("audio", "cpu_native_acoustic", "cpu", "rlr_native",
                    max_parallel=1),
        policy=alloc.policy, compatibility=habitat_worker())
    assert alloc.try_acquire(first).granted
    second = LeaseRequest(
        lease_id="second_audio", compatibility=habitat_worker(),
        requirement=alloc.policy.requirement("rlr_audio_cpu"),
        output_relative="tmp/p18_test/second_audio", max_parallel=1)
    decision = alloc.try_acquire(second)
    assert decision.status == "wait"
    assert decision.reason_code == "backend_max_parallel_reached"
    # A different backend is unaffected by that cap.
    assert alloc.try_acquire(claim(alloc, "deliver", "cpu_only")).granted


def test_every_resource_kind_the_stage_protocol_declares_resolves_to_a_backend():
    """If the protocol grows a kind, this says so instead of guessing a device."""
    from avengine.dataset import production_spec

    resource_policy = ResourcePolicy.from_mapping({})
    for kind, (execution, runtime_context) in production_spec.RESOURCE_KIND_PROFILE.items():
        for renderer in ("habitat", "ue_spear"):
            requirement = resource_policy.requirement_for_runtime_profile(
                execution, runtime_context, renderer=renderer)
            assert requirement.lane == execution, (kind, renderer, requirement)
            if runtime_context == "pure_python":
                assert requirement.native_context is False, kind
            else:
                assert requirement.native_context is True, kind


# --------------------------------------------------------------------------
# One table, shared with the room route
# --------------------------------------------------------------------------


def test_the_worker_key_matches_the_room_route_isolation_key():
    """Two owners must not disagree about what makes two rooms incompatible."""
    from avengine.rooms.room_package import (
        RUNTIME_ISOLATION_KEYS,
        runtime_isolation_key,
    )

    assert PROCESS_IDENTITY_RUNTIME_KEYS is RUNTIME_ISOLATION_KEYS
    report = {
        "renderer": "habitat",
        "effective": {"runtime_prefix": MP3D_PREFIX, "magnum_python_site": MAGNUM_SITE,
                      "rlr_sdk_root": "/data/rlr", "mp3d_root": "/data/mp3d"},
        "isolation_keys": RUNTIME_ISOLATION_KEYS["habitat"],
    }
    other = {**report, "effective": {**report["effective"],
                                     "runtime_prefix": HM3D_PREFIX}}
    same_room_different_dataset = {
        **report, "effective": {**report["effective"], "mp3d_root": "/data/other"}}

    worker = WorkerCompatibility.from_runtime_report(
        report, python_executable="/opt/env/bin/python")
    assert worker.renderer == "habitat"
    assert worker.requires_fresh_interpreter is True

    def agree(first, second):
        route_says = runtime_isolation_key(first) == runtime_isolation_key(second)
        p18_says = WorkerCompatibility.from_runtime_report(
            first, python_executable="/opt/env/bin/python").compatible_with(
            WorkerCompatibility.from_runtime_report(
                second, python_executable="/opt/env/bin/python"))
        assert route_says == p18_says, (first, second)
        return p18_says

    assert agree(report, other) is False
    assert agree(report, same_room_different_dataset) is True


def test_a_spear_worker_does_not_claim_a_habitat_style_fork_ban():
    spear = WorkerCompatibility.from_runtime(
        {"uproject": "/p", "unreal_editor": "/e", "spear_ext_dir": "/x"},
        renderer="ue_spear", python_executable="/opt/env/bin/python")
    habitat = habitat_worker()
    assert habitat.requires_fresh_interpreter is True
    assert spear.requires_fresh_interpreter is False


# --------------------------------------------------------------------------
# A start-up floor and an expected peak are two different numbers
# --------------------------------------------------------------------------


def gpu_policy(**gpu):
    base = {"max_workers": 2, "min_free_vram_mb": 1024, "headroom_mb": 0,
            "max_workers_per_device": 2}
    base.update(gpu)
    return policy(gpu=base, backends={"big": {"lane": "gpu",
                                              "estimated_peak_vram_mb": 8000}})


def test_a_low_floor_does_not_shrink_the_reservation():
    """The bug this replaces: a 1 GiB floor turned an 8 GiB worker into 1 GiB.

    The device has room for one such worker, not for several, and the ledger
    has to say so however permissive the start-up floor is.
    """
    machine = FakeMachine([device(0, 12000)])
    alloc, _ = allocator(machine, gpu_policy(min_free_vram_mb=8192))
    request = LeaseRequest(
        lease_id="one", compatibility=habitat_worker(),
        requirement=alloc.policy.requirement("big"),
        output_relative="tmp/p18_test/one", min_free_vram_mb=1024)
    granted = alloc.try_acquire(request)
    assert granted.granted
    assert granted.lease.reserved_vram_mb == 8000
    assert alloc.effective_free_mb(machine.devices[0]) == 4000
    second = alloc.try_acquire(LeaseRequest(
        lease_id="two", compatibility=habitat_worker(),
        requirement=alloc.policy.requirement("big"),
        output_relative="tmp/p18_test/two", min_free_vram_mb=1024))
    assert second.status == "wait"
    assert second.reason_code == "insufficient_free_vram"


def test_a_low_floor_still_lets_a_nearly_full_device_be_used():
    """A permissive floor is permissive; it just is not a smaller worker."""
    machine = FakeMachine([device(0, 9000)])
    strict, _ = allocator(machine, gpu_policy(min_free_vram_mb=9500))
    assert strict.try_acquire(claim(strict, "cap", "big")).reason_code == "insufficient_free_vram"
    permissive, _ = allocator(machine, gpu_policy(min_free_vram_mb=1024))
    granted = permissive.try_acquire(claim(permissive, "cap", "big"))
    assert granted.granted
    assert granted.lease.reserved_vram_mb == 8000


def test_a_high_floor_still_takes_effect():
    machine = FakeMachine([device(0, 20000)])
    alloc, _ = allocator(machine, gpu_policy(min_free_vram_mb=1024))
    request = LeaseRequest(
        lease_id="picky", compatibility=habitat_worker(),
        requirement=alloc.policy.requirement("big"),
        output_relative="tmp/p18_test/picky", min_free_vram_mb=24000)
    decision = alloc.try_acquire(request)
    assert decision.status == "wait"
    assert decision.reason_code == "insufficient_free_vram"
    assert "floor of 24000 MiB" in decision.reason
    assert alloc.effective_min_free_vram_mb(request) == 24000


def test_the_strictest_of_policy_backend_and_item_floors_wins():
    alloc, _ = allocator(FakeMachine([device(0, 40000)]),
                         gpu_policy(min_free_vram_mb=4000))
    from_backend = replace(alloc.policy.requirement("big"), min_free_vram_mb=9000)
    request = LeaseRequest(
        lease_id="mix", compatibility=habitat_worker(), requirement=from_backend,
        output_relative="tmp/p18_test/mix", min_free_vram_mb=6000)
    assert request.effective_min_free_vram_mb == 9000
    assert alloc.effective_min_free_vram_mb(request) == 9000
    policy_wins = LeaseRequest(
        lease_id="mix2", compatibility=habitat_worker(),
        requirement=alloc.policy.requirement("big"),
        output_relative="tmp/p18_test/mix2", min_free_vram_mb=100)
    assert alloc.effective_min_free_vram_mb(policy_wins) == 4000


def test_a_backend_floor_comes_from_configuration():
    resolved = backend_profiles({"habitat_native_capture": {"min_free_vram_mb": 16000}})
    assert resolved["habitat_native_capture"].min_free_vram_mb == 16000
    assert resolved["habitat_native_capture"].estimated_peak_vram_mb == 6144


def test_a_cpu_backend_may_not_declare_a_floor_either():
    with pytest.raises(ResourcePolicyError, match="min_free_vram_mb"):
        BackendRequirement(backend_id="wrong", lane="cpu", min_free_vram_mb=1024)


def test_three_pending_workers_do_not_spend_the_same_remaining_memory():
    """Nothing has started yet, so the driver number has not moved at all."""
    # 20000 free, each worker needs 8000 plus 1000 of headroom.
    machine = FakeMachine([device(0, 20000)])
    alloc, _ = allocator(machine, policy(
        cpu={"max_workers": 4},
        gpu={"max_workers": 4, "min_free_vram_mb": 1024, "headroom_mb": 1000,
             "max_workers_per_device": 4},
        backends={"big": {"lane": "gpu", "estimated_peak_vram_mb": 8000}}))
    outcomes = [alloc.try_acquire(claim(alloc, f"w{index}", "big")) for index in range(4)]
    assert [item.status for item in outcomes] == ["granted", "granted", "wait", "wait"]
    # The driver number never moved: nothing has started yet. Only the run's
    # own ledger stopped the third and fourth.
    assert machine.devices[0].free_memory_mb == 20000
    assert alloc.effective_free_mb(machine.devices[0]) == 20000 - 2 * 8000
    assert outcomes[2].reason_code == "insufficient_free_vram"
    assert "outstanding" in outcomes[2].reason


# --------------------------------------------------------------------------
# The pid the driver charges is not always the pid the launcher got
# --------------------------------------------------------------------------


def tree_allocator(machine, tree, resource_policy=None):
    return ResourceAllocator(
        resource_policy or gpu_policy(),
        inventory_reader=machine.inventory,
        process_reader=machine.processes,
        port_probe=lambda port, host: True,
        descendants_reader=lambda pid: tuple(tree.get(pid, ())),
        clock=lambda: 0.0,
        sleep=lambda seconds: None,
    )


def test_a_child_render_process_is_charged_to_its_worker():
    """SPEAR starts the packaged game as a child; the parent shows nothing.

    Attributing only the parent would read 0 MiB for a worker holding 12 GiB,
    keep the whole reservation held back and report a peak of nothing.
    """
    machine = FakeMachine([device(0, 40000, uuid="GPU-a")])
    alloc = tree_allocator(machine, {8100: (8140,)})
    lease = alloc.try_acquire(claim(alloc, "ue", "big")).lease
    alloc.bind_worker_pid("ue", 8100)
    machine.apps = [
        GpuProcess(pid=8140, gpu_uuid="GPU-a", used_memory_mb=12000, process_type="G"),
    ]
    assert alloc.attributed_pids("ue") == (8100, 8140)
    assert alloc.refresh_observations() == {"ue": 12000}
    assert alloc.observed_peak_vram_mb("ue") == 12000
    # The estimate is already covered by the real usage, so nothing extra is
    # held back from a co-tenant.
    assert alloc.effective_free_mb(machine.devices[0]) == 40000


def test_a_worker_that_renders_in_its_own_process_is_unaffected():
    machine = FakeMachine([device(0, 40000, uuid="GPU-a")])
    alloc = tree_allocator(machine, {})
    alloc.try_acquire(claim(alloc, "habitat", "big"))
    alloc.bind_worker_pid("habitat", 9000)
    machine.apps = [GpuProcess(pid=9000, gpu_uuid="GPU-a", used_memory_mb=900,
                               process_type="G")]
    assert alloc.attributed_pids("habitat") == (9000,)
    assert alloc.refresh_observations() == {"habitat": 900}


def test_a_worker_and_its_child_are_summed_not_counted_twice():
    machine = FakeMachine([device(0, 40000, uuid="GPU-a")])
    alloc = tree_allocator(machine, {8100: (8140, 8150)})
    alloc.try_acquire(claim(alloc, "ue", "big"))
    alloc.bind_worker_pid("ue", 8100)
    machine.apps = [
        GpuProcess(pid=8100, gpu_uuid="GPU-a", used_memory_mb=300, process_type="C"),
        GpuProcess(pid=8140, gpu_uuid="GPU-a", used_memory_mb=11000, process_type="G"),
        GpuProcess(pid=8150, gpu_uuid="GPU-a", used_memory_mb=200, process_type="G"),
    ]
    assert alloc.refresh_observations() == {"ue": 11500}
    # A second sweep over the same processes must not add them again.
    assert alloc.refresh_observations() == {"ue": 11500}


def test_a_launcher_may_name_a_process_the_tree_does_not_show():
    machine = FakeMachine([device(0, 40000, uuid="GPU-a")])
    alloc = tree_allocator(machine, {})
    alloc.try_acquire(claim(alloc, "ue", "big"))
    alloc.bind_worker_pid("ue", 8100)
    machine.apps = [GpuProcess(pid=7777, gpu_uuid="GPU-a", used_memory_mb=9000,
                               process_type="G")]
    assert alloc.refresh_observations() == {}
    alloc.register_worker_pid("ue", 7777)
    assert alloc.refresh_observations() == {"ue": 9000}


def test_following_descendants_can_be_turned_off():
    machine = FakeMachine([device(0, 40000, uuid="GPU-a")])
    alloc = tree_allocator(machine, {8100: (8140,)})
    alloc.try_acquire(claim(alloc, "ue", "big"))
    alloc.bind_worker_pid("ue", 8100, include_descendants=False)
    machine.apps = [GpuProcess(pid=8140, gpu_uuid="GPU-a", used_memory_mb=12000)]
    assert alloc.attributed_pids("ue") == (8100,)
    assert alloc.refresh_observations() == {}


def test_one_process_is_charged_to_one_lease_only():
    """A shared helper must not inflate two peaks at once."""
    machine = FakeMachine([device(0, 40000, uuid="GPU-a")])
    alloc = tree_allocator(machine, {}, gpu_policy(max_workers=2))
    alloc.try_acquire(claim(alloc, "first", "big"))
    alloc.try_acquire(claim(alloc, "second", "big"))
    alloc.bind_worker_pid("first", 5000)
    alloc.bind_worker_pid("second", 6000)
    alloc.register_worker_pid("second", 5000)
    machine.apps = [GpuProcess(pid=5000, gpu_uuid="GPU-a", used_memory_mb=4000),
                    GpuProcess(pid=6000, gpu_uuid="GPU-a", used_memory_mb=1000)]
    observed = alloc.refresh_observations()
    assert observed == {"first": 4000, "second": 1000}


def test_a_bound_pid_stops_the_estimate_being_subtracted_twice():
    """Driver free already includes what the worker took; only the rest is held."""
    machine = FakeMachine([device(0, 40000, uuid="GPU-a")])
    alloc = tree_allocator(machine, {8100: (8140,)})
    lease = alloc.try_acquire(claim(alloc, "ue", "big")).lease
    assert alloc.effective_free_mb(machine.devices[0]) == 40000 - 8000
    alloc.bind_worker_pid("ue", 8100)
    machine.set_free(0, 37000)
    machine.apps = [GpuProcess(pid=8140, gpu_uuid="GPU-a", used_memory_mb=3000,
                               process_type="G")]
    alloc.refresh_observations()
    # 37000 free, 5000 of the 8000 estimate still unspent.
    assert alloc.effective_free_mb(machine.devices[0]) == 37000 - 5000
    alloc.release(lease)
    assert alloc.effective_free_mb(machine.devices[0]) == 37000


def test_descendant_pids_finds_a_real_child_process():
    """The default reader against real processes, no GPU work involved."""
    import subprocess

    parent = subprocess.Popen(
        [sys.executable, "-c",
         "import subprocess, sys, time;"
         "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']);"
         "print(child.pid, flush=True); time.sleep(30)"],
        stdout=subprocess.PIPE, text=True)
    try:
        child_pid = int(parent.stdout.readline().strip())
        from avengine.dataset.production_resources import descendant_pids
        found = descendant_pids(parent.pid)
        assert child_pid in found, (parent.pid, child_pid, found)
        assert descendant_pids(child_pid) == ()
    finally:
        parent.kill()
        parent.wait(timeout=30)


# --------------------------------------------------------------------------
# Recovery: taking running workers back onto the books
# --------------------------------------------------------------------------


def test_a_snapshot_round_trips_into_a_fresh_allocator():
    machine = FakeMachine([device(0, 40000, uuid="GPU-a")])
    before = tree_allocator(machine, {8100: (8140,)}, gpu_policy(max_workers=2))
    lease = before.try_acquire(claim(before, "cap", "big")).lease
    before.bind_worker_pid("cap", 8100)
    before.register_worker_pid("cap", 7777)
    before.report_observed_vram_mb("cap", 3000)
    saved = json.loads(json.dumps(before.snapshot()))

    after = tree_allocator(machine, {8100: (8140,)}, gpu_policy(max_workers=2))
    result = after.restore_from_snapshot(saved, is_running=lambda pid: True)
    assert result["adopted"] == ["cap"]
    restored = after.live_leases()[0]
    assert (restored.device_index, restored.reserved_vram_mb) == (
        lease.device_index, lease.reserved_vram_mb)
    assert restored.requirement.estimated_peak_vram_mb == 8000
    assert after.observed_peak_vram_mb("cap") == 3000
    assert after.attributed_pids("cap") == (7777, 8100, 8140)
    # The reservation is on the books again, so the next grant sees it.
    assert after.effective_free_mb(machine.devices[0]) == 40000 - (8000 - 3000)


def test_a_recovered_worker_still_holds_its_port_and_output():
    machine = FakeMachine([device(0, 48000, uuid="GPU-a")])
    spear_policy = policy(
        gpu={"max_workers": 2, "min_free_vram_mb": 1024, "headroom_mb": 0,
             "max_workers_per_device": 2},
        backends={"ue": {"lane": "gpu", "estimated_peak_vram_mb": 4096,
                         "needs_rpc_port": True, "visual_worlds": 1}})
    spear = WorkerCompatibility.from_runtime(
        {"uproject": "/p", "unreal_editor": "/e", "spear_ext_dir": "/x"},
        renderer="ue_spear", python_executable="/opt/env/bin/python")
    before = tree_allocator(machine, {}, spear_policy)
    lease = before.try_acquire(LeaseRequest(
        lease_id="ue1", compatibility=spear,
        requirement=spear_policy.requirement("ue"),
        output_relative="tmp/run/ue1")).lease
    saved = json.loads(json.dumps(before.snapshot()))

    after = tree_allocator(machine, {}, spear_policy)
    after.restore_from_snapshot(saved)
    other = after.try_acquire(LeaseRequest(
        lease_id="ue2", compatibility=spear,
        requirement=spear_policy.requirement("ue"),
        output_relative="tmp/run/ue2"))
    assert other.granted
    assert other.lease.rpc_port != lease.rpc_port
    collision = after.try_acquire(LeaseRequest(
        lease_id="ue3", compatibility=spear,
        requirement=spear_policy.requirement("ue"),
        output_relative="tmp/run/ue1"))
    assert collision.reason_code == "output_collision"


def test_a_worker_that_died_while_we_were_gone_is_dropped_not_adopted():
    machine = FakeMachine([device(0, 40000, uuid="GPU-a")])
    before = tree_allocator(machine, {}, gpu_policy(max_workers=2))
    before.try_acquire(claim(before, "alive", "big"))
    before.try_acquire(claim(before, "dead", "big"))
    before.bind_worker_pid("alive", 111)
    before.bind_worker_pid("dead", 222)
    saved = json.loads(json.dumps(before.snapshot()))

    after = tree_allocator(machine, {}, gpu_policy(max_workers=2))
    result = after.restore_from_snapshot(saved, is_running=lambda pid: pid == 111)
    assert result["adopted"] == ["alive"]
    assert [item["lease_id"] for item in result["dropped"]] == ["dead"]
    # The dead worker's memory and slot came back into circulation.
    assert after.effective_free_mb(machine.devices[0]) == 40000 - 8000
    assert after.snapshot()["gpu_workers_used"] == 1


def test_adopting_the_same_lease_twice_is_refused():
    machine = FakeMachine([device(0, 40000, uuid="GPU-a")])
    alloc = tree_allocator(machine, {})
    lease = alloc.try_acquire(claim(alloc, "cap", "big")).lease
    with pytest.raises(ResourcePolicyError, match="count its slot and its memory twice"):
        alloc.adopt_lease(lease.to_dict())


def test_adoption_does_not_re_test_capacity_but_does_account_for_it():
    """The process exists whether or not the budget likes it; the books must say so."""
    machine = FakeMachine([device(0, 12000, uuid="GPU-a")])
    alloc = tree_allocator(machine, {},
                           gpu_policy(max_workers=1, max_workers_per_device=1))
    running = Lease(
        lease_id="already_running",
        compatibility=habitat_worker(),
        requirement=alloc.policy.requirement("big"),
        output_relative="tmp/run/already",
        worker_slot=1, device_index=0, device_uuid="GPU-a",
        reserved_vram_mb=8000)
    alloc.adopt_lease(running, pid=4242)
    assert alloc.snapshot()["gpu_workers_used"] == 1
    blocked = alloc.try_acquire(claim(alloc, "next", "big"))
    assert blocked.reason_code == "gpu_slots_busy"


def test_the_launch_plan_says_what_the_runner_must_bind():
    alloc, _ = allocator(FakeMachine([device(0, 48000)]),
                         policy(backends={"ue": {"lane": "gpu", "estimated_peak_vram_mb": 4096,
                                                 "needs_rpc_port": True, "visual_worlds": 1}}))
    spear = WorkerCompatibility.from_runtime(
        {"uproject": "/p", "unreal_editor": "/e", "spear_ext_dir": "/x"},
        renderer="ue_spear", python_executable="/opt/env/bin/python")
    lease = alloc.try_acquire(claim(alloc, "ue1", "ue", worker=spear)).lease
    plan = worker_launch_plan(lease, entry=("tools/ue.py",))
    assert plan["pid_binding"]["include_descendants"] is True
    # The existing launcher already takes these two by name; nothing has to be
    # renamed for the allocator's choice to reach the game.
    assert plan["spear_launch_arguments"] == {
        "rpc_port": lease.rpc_port, "graphics_adapter": lease.device_index}


def test_the_launch_arguments_match_what_the_existing_launcher_expects():
    import inspect

    from avengine.backends.spear_ue.launch import launch_arguments_for_lease
    from avengine.backends.spear_ue.research_runtime import (
        launch_external_game_instance,
    )

    alloc, _ = allocator(FakeMachine([device(0, 48000)]),
                         policy(backends={"ue": {"lane": "gpu", "estimated_peak_vram_mb": 4096,
                                                 "needs_rpc_port": True, "visual_worlds": 1}}))
    spear = WorkerCompatibility.from_runtime(
        {"uproject": "/p", "unreal_editor": "/e", "spear_ext_dir": "/x"},
        renderer="ue_spear", python_executable="/opt/env/bin/python")
    lease = alloc.try_acquire(claim(alloc, "ue1", "ue", worker=spear)).lease
    arguments = launch_arguments_for_lease(lease)
    accepted = inspect.signature(launch_external_game_instance).parameters
    assert set(arguments) <= set(accepted), (set(arguments), sorted(accepted))


# --------------------------------------------------------------------------
# C02: the exact, opt-in display exception
#
# The machine these run on keeps a display server on every graphics device.
# It holds four megabytes and renders nothing for anybody, but under
# allow_shared_device=false it made every device look occupied and stopped a
# qualification run before it launched anything. The exception below is the
# narrowest thing that unblocks it: one executable, both uids, the driver's
# own process type, and a memory ceiling. Its memory is still spent.
# --------------------------------------------------------------------------

XORG_RULE = {
    "executable": "/usr/lib/xorg/Xorg",
    "uid": 128,
    "effective_uid": 0,
    "process_type": "G",
    "max_memory_mb": 16,
}


def display_process(uuid, **overrides):
    """The display server as this host's /proc actually reports it."""
    fields = {
        "pid": 2412,
        "gpu_uuid": uuid,
        "used_memory_mb": 4,
        "process_type": "G",
        "executable": "/usr/lib/xorg/Xorg",
        "uid": 128,
        "effective_uid": 0,
        "identity_source": "cmdline",
    }
    fields.update(overrides)
    return GpuProcess(**fields)


def exclusive(rules=(XORG_RULE,), **gpu):
    settings = {
        "allow_shared_device": False,
        "nonblocking_display_processes": list(rules),
    }
    settings.update(gpu)
    return policy(gpu=settings)


def test_the_display_server_alone_no_longer_reserves_the_whole_machine():
    machine = FakeMachine(
        [device(0, 40000, uuid="GPU-a"), device(1, 40000, uuid="GPU-b")],
        apps=[display_process("GPU-a"), display_process("GPU-b")],
    )
    refused, _ = allocator(machine, exclusive(rules=()))
    assert refused.try_acquire(
        claim(refused, "cap", "habitat_native_capture")
    ).status == "wait"

    alloc, _ = allocator(machine, exclusive())
    granted = alloc.try_acquire(claim(alloc, "cap", "habitat_native_capture"))
    assert granted.granted
    assert granted.lease.device_index in (0, 1)


def test_the_exception_is_off_unless_it_is_configured():
    assert ResourcePolicy.from_mapping({}).gpu.nonblocking_display_processes == ()


def test_real_work_on_the_device_is_still_refused_beside_the_display():
    """The display server is let past; somebody's training job is not."""
    machine = FakeMachine(
        [device(0, 40000, uuid="GPU-a")],
        apps=[
            display_process("GPU-a"),
            GpuProcess(pid=942915, gpu_uuid="GPU-a", used_memory_mb=8334,
                       process_type="C+G", executable="/data/other/bin/python",
                       uid=1014, effective_uid=1014, identity_source="exe"),
        ],
    )
    alloc, _ = allocator(machine, exclusive())
    refused = alloc.try_acquire(claim(alloc, "cap", "habitat_native_capture"))
    assert refused.status == "wait"
    assert "[942915]" in refused.reason
    assert "2412" not in refused.reason


@pytest.mark.parametrize(
    "difference",
    [
        {"executable": "/usr/bin/Xorg"},
        {"executable": None},
        {"uid": 0},
        {"uid": None},
        {"effective_uid": 128},
        {"effective_uid": None},
        {"process_type": "C"},
        {"process_type": "C+G"},
        {"used_memory_mb": 17},
    ],
)
def test_one_field_off_is_not_the_display_server(difference):
    """Unknown or merely similar identity never buys a pass."""
    machine = FakeMachine(
        [device(0, 40000, uuid="GPU-a")],
        apps=[display_process("GPU-a", **difference)],
    )
    alloc, _ = allocator(machine, exclusive())
    refused = alloc.try_acquire(claim(alloc, "cap", "habitat_native_capture"))
    assert refused.status == "wait"
    assert "allow_shared_device is false" in refused.reason
    assert alloc.display_exclusions() == {}


def test_a_rule_may_not_name_a_compute_process_type():
    with pytest.raises(ResourcePolicyError):
        ResourcePolicy.from_mapping(
            {"gpu": {"nonblocking_display_processes": [
                {**XORG_RULE, "process_type": "C+G"}]}}
        )


def test_excluded_display_memory_is_still_spent_not_forgiven():
    """The floor and the headroom are arithmetic, not a permission check."""
    machine = FakeMachine(
        [device(0, 6000, uuid="GPU-a")],
        apps=[display_process("GPU-a")],
    )
    alloc, _ = allocator(
        machine, exclusive(min_free_vram_mb=8192, headroom_mb=1024))
    refused = alloc.try_acquire(claim(alloc, "cap", "habitat_native_capture"))
    assert refused.reason_code == "insufficient_free_vram"
    assert "below the min_free_vram_mb floor" in refused.reason


def test_the_snapshot_says_which_display_process_was_let_past():
    machine = FakeMachine(
        [device(0, 40000, uuid="GPU-a")], apps=[display_process("GPU-a")])
    alloc, _ = allocator(machine, exclusive())
    granted = alloc.try_acquire(claim(alloc, "cap", "habitat_native_capture"))
    assert granted.granted
    recorded = alloc.snapshot()["gpu_display_exclusions"]["0"]
    assert [item["process"]["pid"] for item in recorded] == [2412]
    assert recorded[0]["rule"]["executable"] == "/usr/lib/xorg/Xorg"
    assert recorded[0]["process"]["identity_source"] == "cmdline"
    # It was let past, not adopted: it is nobody's worker.
    assert 2412 not in {
        pid
        for pids in alloc.snapshot()["attributed_pids"].values()
        for pid in pids
    }
    assert alloc.snapshot()["worker_pids"] == {}


def test_a_job_that_lands_between_the_grant_and_the_launch_is_caught():
    machine = FakeMachine(
        [device(0, 40000, uuid="GPU-a")], apps=[display_process("GPU-a")])
    alloc, _ = allocator(machine, exclusive())
    granted = alloc.try_acquire(claim(alloc, "cap", "habitat_native_capture"))
    assert granted.granted
    assert alloc.recheck_before_launch(granted.lease).granted

    machine.apps.append(
        GpuProcess(pid=777, gpu_uuid="GPU-a", used_memory_mb=20000,
                   process_type="C", executable="/data/other/bin/python",
                   uid=1014, effective_uid=1014, identity_source="exe"))
    late = alloc.recheck_before_launch(granted.lease)
    assert late.status == "wait"
    assert late.reason_code == "foreign_workload_appeared"
    assert late.detail["foreign_pids"] == [777]
    assert [item["process"]["pid"]
            for item in late.detail["excluded_display_processes"]] == [2412]


def test_this_runs_own_child_is_not_a_foreign_tenant():
    """A SPEAR worker charges its memory to a child it started itself."""
    machine = FakeMachine(
        [device(0, 40000, uuid="GPU-a")], apps=[display_process("GPU-a")])
    alloc = ResourceAllocator(
        exclusive(),
        inventory_reader=machine.inventory,
        process_reader=machine.processes,
        port_probe=lambda port, host: True,
        descendants_reader=lambda pid: (5001,) if pid == 5000 else (),
        clock=lambda: 0.0,
        sleep=lambda seconds: None,
    )
    granted = alloc.try_acquire(claim(alloc, "cap", "habitat_native_capture"))
    assert granted.granted
    alloc.bind_worker_pid("cap", 5000)
    machine.apps.append(
        GpuProcess(pid=5001, gpu_uuid="GPU-a", used_memory_mb=6000,
                   process_type="G", executable="/opt/spear/Game.sh",
                   uid=1000, effective_uid=1000, identity_source="exe"))
    assert alloc.recheck_before_launch(granted.lease).granted
