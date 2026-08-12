from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _module():
    root = Path(__file__).resolve().parents[2]
    path = root / "tools/qa/probe_packaged_imported_glb_room.py"
    spec = importlib.util.spec_from_file_location("packaged_imported_room_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PROBE = _module()


def _adapter() -> dict[str, object]:
    return {
        "schema": "avengine_spear_imported_glb_room_adapter_v1",
        "scene_id": "17DRP5sb8fy",
        "entry_map": PROBE.ENTRY_MAP,
        "expected_static_mesh_count": 71,
        "static_mesh_object_paths": [
            f"/Game/MP3D/mesh_{index:03d}.mesh_{index:03d}" for index in range(71)
        ],
    }


def _readback() -> dict[str, object]:
    adapter = _adapter()
    paths = adapter["static_mesh_object_paths"]
    assert isinstance(paths, list)
    return {
        "schema": PROBE.READBACK_SCHEMA,
        "status": "pass",
        "scene_id": "17DRP5sb8fy",
        "entry_map": PROBE.ENTRY_MAP,
        "expected_static_mesh_count": 71,
        "spawned_static_mesh_count": 71,
        "all_expected_handles_match_components": True,
        "unique_loaded_object_handle_count": 71,
        "unique_component_mesh_handle_count": 71,
        "meshes": [
            {
                "mesh_index": index,
                "object_path": path,
                "stable_actor_name": f"AVEngine/ImportedGLB/room/mesh_{index:03d}",
                "expected_object_handle": 10_000 + index,
                "observed_component_mesh_handle": 10_000 + index,
                "readback_method": "UStaticMeshComponent.StaticMesh_property",
                "status": "pass",
            }
            for index, path in enumerate(paths)
        ],
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }


def _result() -> dict[str, object]:
    return {
        "schema": PROBE.RESULT_SCHEMA,
        "status": "pass",
        "readiness_status": "packaged_71_mesh_readback_pass_gpu_f15_pending",
        "scene_id": "17DRP5sb8fy",
        "entry_map": PROBE.ENTRY_MAP,
        "nullrhi": True,
        "rendering_or_capture_called": False,
        "room_live_readback": _readback(),
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }


def _exit() -> dict[str, object]:
    return {
        "schema": PROBE.EXIT_SCHEMA,
        "status": "pass",
        "worker_exit_code": 0,
        "timed_out": False,
        "result_status": "pass",
        "exact_packaged_processes_before": [],
        "exact_packaged_processes_after": [],
        "exact_packaged_process_exit_closed": True,
        "nullrhi": True,
        "rendering_or_capture_called": False,
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }


def test_complete_71_mesh_result_is_f15_bindable_but_stays_pending() -> None:
    evidence = PROBE.validate_result_and_exit(
        adapter=_adapter(), result=_result(), exit_receipt=_exit()
    )
    assert evidence["status"] == "pass"
    assert evidence["mesh_count"] == 71
    assert evidence["gpu_f15_status"] == "pending"
    assert evidence["qualification_claim"] is False
    assert evidence["formal_dataset_count"] == 0


def test_runner_source_hides_cuda_from_worker() -> None:
    source = Path(PROBE.__file__).read_text(encoding="utf-8")
    assert '"CUDA_VISIBLE_DEVICES": ""' in source


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(status="pending"), "71-mesh boundary"),
        (
            lambda value: value["meshes"][9].update(
                observed_component_mesh_handle=900_009
            ),
            "readback 9",
        ),
        (
            lambda value: value["meshes"][9].update(
                expected_object_handle=value["meshes"][8]["expected_object_handle"],
                observed_component_mesh_handle=value["meshes"][8][
                    "observed_component_mesh_handle"
                ],
            ),
            "not unique",
        ),
        (
            lambda value: value["meshes"].reverse(),
            "mesh readback 0",
        ),
        (
            lambda value: value["meshes"][4].update(expected_object_handle=0),
            "positive handle",
        ),
        (
            lambda value: value["meshes"][4].update(
                stable_actor_name=value["meshes"][3]["stable_actor_name"]
            ),
            "not unique",
        ),
    ],
)
def test_live_readback_fails_closed(mutation, message: str) -> None:
    readback = _readback()
    mutation(readback)
    with pytest.raises(RuntimeError, match=message):
        PROBE.validate_live_readback(_adapter(), readback)


def test_adapter_requires_exact_71_unique_ordered_paths() -> None:
    adapter = _adapter()
    paths = adapter["static_mesh_object_paths"]
    assert isinstance(paths, list)
    paths.pop()
    with pytest.raises(RuntimeError, match="exactly 71 unique"):
        PROBE.validate_live_readback(adapter, _readback())

    adapter = _adapter()
    paths = adapter["static_mesh_object_paths"]
    assert isinstance(paths, list)
    paths[-1] = paths[0]
    with pytest.raises(RuntimeError, match="exactly 71 unique"):
        PROBE.validate_live_readback(adapter, _readback())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(status="pending"),
        lambda value: value.update(timed_out=True),
        lambda value: value.update(worker_exit_code=1),
        lambda value: value.update(exact_packaged_processes_after=[{"pid": 12}]),
        lambda value: value.update(rendering_or_capture_called=True),
        lambda value: value.update(formal_dataset_count=1),
    ],
)
def test_exit_receipt_never_promotes_pending_or_unclosed_execution(mutation) -> None:
    exit_receipt = _exit()
    mutation(exit_receipt)
    with pytest.raises(RuntimeError, match="process boundary"):
        PROBE.validate_result_and_exit(
            adapter=_adapter(), result=_result(), exit_receipt=exit_receipt
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(status="pending"),
        lambda value: value.update(nullrhi=False),
        lambda value: value.update(rendering_or_capture_called=True),
        lambda value: value.update(qualification_claim=True),
        lambda value: value.update(formal_dataset_count=1),
        lambda value: value.update(readiness_status="pass"),
    ],
)
def test_result_claim_boundary_is_fail_closed(mutation) -> None:
    result = _result()
    mutation(result)
    with pytest.raises(RuntimeError, match="claim boundary"):
        PROBE.validate_result_and_exit(
            adapter=_adapter(), result=result, exit_receipt=_exit()
        )
