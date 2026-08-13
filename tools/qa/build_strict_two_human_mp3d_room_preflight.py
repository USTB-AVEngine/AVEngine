#!/usr/bin/env python3
"""Build a CPU-only MP3D strict-two-human room/capture/RIR preflight."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from avengine.actor_framing import build_actor_framing_frames
from avengine.camera_framing import solve_static_camera_candidates
from avengine.m5_1.camera_candidate_gate import (
    HabitatRuntimeCameraProvider,
    evaluate_camera_candidates,
)
from avengine.m6x.room_feasibility import (
    TrajectoryBank,
    TrajectoryEpisode,
    build_rir_job_plan,
)

from spear_imported_glb_room_adapter import (
    ENTRY_MAP,
    build_room_adapter_record,
    load_json_object,
    validate_room_adapter,
)

REQUEST_SCHEMA = "avengine_native_strict_two_human_mp3d_room_atom_request_v1"
REQUEST_SCHEMA_V2 = "avengine_native_strict_two_human_mp3d_room_atom_request_v2"
SEMANTIC_RIR_EXECUTION_MODE = "semantic_no_file_evidence"
PREFLIGHT_SCHEMA = "avengine_native_strict_two_human_mp3d_room_preflight_v1"
SUITE_SCHEMA = "avengine_optional_spear_imported_glb_suite_v1"
SCENARIO_SCHEMA = "avengine_optional_spear_imported_glb_scenario_v1"
FRAME_COUNT = 75
FPS = 15
TICKS_PER_FRAME = 3200
REMOTE_REPOSITORY = Path("/data/jzy/code/AVEngine-lead-a")
SPEAR_CAPTURE_PYTHON = Path("/data/jzy/miniconda3/envs/spear-env/bin/python")
HABITAT_RUNTIME_ROOT = "/data/jzy/code/habitat-sim-AVEngine"
SOUNDSPACES_ROOT = "/data/jzy/code/sound-spaces"
HABITAT_PYTHON = Path("/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python")
HABITAT_PATH = (
    "/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)
HABITAT_EDITABLE_BUILD = (
    "/data/jzy/code/habitat-sim-AVEngine/build/cp312-cp312-linux_x86_64"
)
MP3D_SCENE = Path(
    "/data/jzy/code/habitat-sim-AVEngine/data/versioned_data/"
    "mp3d_example_scene_1.1/17DRP5sb8fy/17DRP5sb8fy.glb"
)
MP3D_DATASET = Path(
    "/data/jzy/code/habitat-sim-AVEngine/data/versioned_data/"
    "mp3d_example_scene_1.1/mp3d.scene_dataset_config.json"
)


def validate_rir_execution_environment(environment: Mapping[str, Any]) -> None:
    """Reject plans that cannot resolve the selected MP3D acoustic profile."""

    required = {
        "AVENGINE_HABITAT_RUNTIME_ROOT": HABITAT_RUNTIME_ROOT,
        "AVENGINE_SOUNDSPACES_ROOT": SOUNDSPACES_ROOT,
        "AVENGINE_MP3D_SOUNDSPACES2_PACKAGE_ROOT": None,
        "PATH": HABITAT_PATH,
        "PYTHONPATH": str(REMOTE_REPOSITORY / "src"),
        "SKBUILD_EDITABLE_SKIP": HABITAT_EDITABLE_BUILD,
        "NUMBA_DISABLE_JIT": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    for name, expected in required.items():
        value = environment.get(name)
        _require(isinstance(value, str), f"RIR execution environment lacks {name}")
        if expected is None:
            _require(
                value.startswith("/"),
                f"RIR execution environment is missing absolute {name}",
            )
        else:
            _require(value == expected, f"RIR execution environment drifted {name}")


def validate_rir_runtime_binding(
    python_executable: str | Path,
    environment: Mapping[str, Any],
) -> None:
    """Bind native RLR to the reviewed Habitat interpreter and environment."""

    _require(
        Path(python_executable) == HABITAT_PYTHON,
        "RIR runtime interpreter differs from the authoritative Habitat Python",
    )
    validate_rir_execution_environment(environment)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _authoritative_rir_acoustic_state_sha256(
    source_position_m: Sequence[float],
    listener_position_m: Sequence[float],
    listener_orientation_wxyz: Sequence[float],
) -> str:
    """Use A's runtime authority; keep staging self-contained for local QA."""

    try:
        from avengine.m6x.room_feasibility import (
            rir_acoustic_state_sha256,
        )
    except ModuleNotFoundError:
        return _canonical_sha256(
            {
                "schema": "avengine_rir_acoustic_pair_state_v1",
                "source_position_m": list(source_position_m),
                "listener_position_m": list(listener_position_m),
                "listener_orientation_wxyz": list(listener_orientation_wxyz),
            }
        )
    return rir_acoustic_state_sha256(
        source_position_m,
        listener_position_m,
        listener_orientation_wxyz,
    )


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {"path": str(resolved)}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def probe_rir_runtime(output: Path) -> Path:
    """Import the exact native CPU runtime and emit a fail-closed receipt."""

    validate_rir_runtime_binding(sys.executable, os.environ)
    _require(not output.exists(), f"refusing to replace runtime probe: {output}")
    numpy = importlib.import_module("numpy")
    quaternion = importlib.import_module("quaternion")
    habitat_sim = importlib.import_module("habitat_sim")
    avengine = importlib.import_module("avengine")
    driver = importlib.import_module("numba.cuda.cudadrv.driver")

    avengine_path = Path(avengine.__file__).resolve()
    expected_avengine_path = (REMOTE_REPOSITORY / "src/avengine/__init__.py").resolve()
    _require(
        avengine_path == expected_avengine_path,
        "runtime probe imported avengine from an unreviewed checkout",
    )
    cuda_initialized = bool(driver.driver.is_initialized)
    _require(not cuda_initialized, "runtime probe initialized CUDA unexpectedly")
    receipt = {
        "schema": "avengine_mp3d_rir_runtime_probe_v1",
        "status": "pass",
        "python_executable": str(Path(sys.executable).resolve()),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": numpy.__version__,
            "quaternion": getattr(quaternion, "__version__", None),
            "habitat_sim": getattr(habitat_sim, "__version__", None),
        },
        "avengine_source": str(avengine_path),
        "import_order": ["numpy", "quaternion", "habitat_sim", "avengine"],
        "environment": {
            name: os.environ[name]
            for name in (
                "PATH",
                "PYTHONPATH",
                "SKBUILD_EDITABLE_SKIP",
                "NUMBA_DISABLE_JIT",
                "CUDA_VISIBLE_DEVICES",
                "AVENGINE_HABITAT_RUNTIME_ROOT",
                "AVENGINE_SOUNDSPACES_ROOT",
                "AVENGINE_MP3D_SOUNDSPACES2_PACKAGE_ROOT",
            )
        },
        "compute_device": "CPU",
        "gpu_required": False,
        "cuda_initialized": cuda_initialized,
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }
    _write_json(output, receipt)
    print(
        f"MP3D_RIR_RUNTIME_PROBE_OK output={output} cuda_initialized=false",
        flush=True,
    )
    return output


def _vector3(value: Any, *, owner: str) -> list[float]:
    _require(
        isinstance(value, list)
        and len(value) == 3
        and all(
            not isinstance(item, bool)
            and isinstance(item, (int, float))
            and math.isfinite(float(item))
            for item in value
        ),
        f"{owner} must be three finite numbers",
    )
    return [float(item) for item in value]


def _habitat_to_ue_cm(position: Sequence[float]) -> list[float]:
    return [100.0 * position[0], 100.0 * position[2], 100.0 * position[1]]


def _validate_request(request: Mapping[str, Any]) -> None:
    _require(
        request.get("schema") in {REQUEST_SCHEMA, REQUEST_SCHEMA_V2},
        "request schema drift",
    )
    _require(request.get("qualification_claim") is False, "formal claim forbidden")
    _require(request.get("formal_dataset_count") == 0, "formal count must remain zero")
    room = request.get("room")
    _require(
        isinstance(room, Mapping)
        and room.get("room_id") == "habitat_mp3d_example_17DRP5sb8fy"
        and room.get("room_revision") == "raw_v1_plus_declared_proxy_v2_research"
        and room.get("scene_id") == "17DRP5sb8fy"
        and room.get("entry_map") == ENTRY_MAP,
        "MP3D room identity drift",
    )
    capture = request.get("capture")
    _require(
        isinstance(capture, Mapping)
        and capture.get("full_episode_frame_count") == FRAME_COUNT
        and capture.get("frame_rate_hz") == FPS
        and capture.get("sparse_frame_indices") == [15],
        "capture must declare one f15 probe and one 75-frame/15-Hz suite",
    )
    acoustics = request.get("acoustics")
    _require(isinstance(acoustics, Mapping), "request acoustics must be an object")
    rir_execution_mode = acoustics.get("rir_execution_mode", "legacy_registry")
    _require(
        rir_execution_mode in {"legacy_registry", SEMANTIC_RIR_EXECUTION_MODE},
        "RIR execution mode is invalid",
    )
    if rir_execution_mode == SEMANTIC_RIR_EXECUTION_MODE:
        _require(
            request.get("schema") == REQUEST_SCHEMA_V2,
            "semantic RIR execution requires the v2 request shape",
        )
        for field_name in ("simulation_request", "hrtf"):
            raw = Path(str(acoustics.get(field_name, "")))
            _require(
                raw.is_absolute()
                and not any(candidate.is_symlink() for candidate in (raw, *raw.parents))
                and raw.is_file()
                and raw.resolve(strict=True) == raw,
                f"semantic RIR {field_name} must be an absolute regular file",
            )


@contextmanager
def _open_mp3d_camera_runtime(request: Mapping[str, Any]):
    """Open one sensorless physics-enabled CPU runtime for all camera gates."""

    runtime = request.get("camera_runtime")
    _require(isinstance(runtime, Mapping), "v2 request lacks camera_runtime")
    scene = Path(str(runtime.get("scene_path"))).resolve()
    dataset = Path(str(runtime.get("dataset_config_path"))).resolve()
    navmesh = Path(str(request["room"]["navmesh_path"])).resolve()
    physics = Path(str(runtime.get("physics_config_path"))).resolve()
    _require(scene == MP3D_SCENE and scene.is_file(), "MP3D runtime scene drift")
    _require(
        dataset == MP3D_DATASET and dataset.is_file(), "MP3D runtime dataset drift"
    )
    _require(navmesh.is_file(), "MP3D runtime navmesh is missing")
    _require(physics.is_file(), "MP3D physics config is missing")
    import quaternion  # noqa: F401

    import habitat_sim
    import magnum as mn

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(scene)
    sim_cfg.scene_dataset_config_file = str(dataset)
    sim_cfg.physics_config_file = str(physics)
    sim_cfg.enable_physics = True
    if hasattr(sim_cfg, "create_renderer"):
        sim_cfg.create_renderer = False
    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = []
    agent_cfg.action_space = {}
    agent_cfg.height = float(runtime.get("agent_height_m"))
    agent_cfg.radius = float(runtime.get("agent_radius_m"))
    configuration = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    with habitat_sim.Simulator(configuration) as simulator:
        _require(
            simulator.pathfinder.load_nav_mesh(str(navmesh))
            and simulator.pathfinder.is_loaded,
            "Habitat failed to load the exact MP3D navmesh",
        )
        provider = HabitatRuntimeCameraProvider(
            simulator,
            habitat_sim,
            mn,
            {
                "configured_scene_id": str(scene),
                "loaded_scene_id": str(runtime.get("loaded_scene_id")),
                "active_dataset": str(dataset),
                "stage_surface": str(scene),
            },
            provider_id=f"{request['episode_id']}__sensorless-physics",
        )
        yield provider


def _validate_navigation(
    evidence: Mapping[str, Any],
    fresh_probe: Mapping[str, Any],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        evidence.get("schema") == "avengine_mp3d_ue_visual_comparison_v1"
        and evidence.get("status") == "passed",
        "retained MP3D runtime evidence did not pass",
    )
    runtime = evidence.get("runtime")
    _require(
        isinstance(runtime, Mapping) and runtime.get("map") == ENTRY_MAP,
        "retained MP3D evidence did not execute the Entry map",
    )
    nav = evidence.get("navigation_authority")
    _require(
        isinstance(nav, Mapping)
        and nav.get("source_status") == "pass"
        and nav.get("semantics") == "actor_root_center_only"
        and nav.get("full_body_or_obb_clearance_claim") is False,
        "retained MP3D navigation authority is invalid",
    )
    routes = nav.get("routes")
    _require(isinstance(routes, Mapping), "navigation routes are missing")
    placement = request["placement"]
    route_bindings = {
        "source1": ("human0", placement["source1_root_habitat_m"]),
        "source2": ("dog0", placement["source2_root_habitat_m"]),
    }
    selected: dict[str, Any] = {}
    island_ids = set()
    _require(
        fresh_probe.get("schema") == "avengine_mp3d_strict_two_human_navmesh_probe_v1"
        and fresh_probe.get("status") == "pass"
        and fresh_probe.get("navmesh_path") == request["room"]["navmesh_path"],
        "fresh MP3D navmesh probe is invalid",
    )
    fresh_positions = fresh_probe.get("positions")
    _require(isinstance(fresh_positions, Mapping), "fresh navmesh positions missing")
    for source_slot, (route_id, requested_position) in route_bindings.items():
        route = routes.get(route_id)
        _require(isinstance(route, Mapping), f"missing nav route {route_id}")
        position = _vector3(requested_position, owner=f"{source_slot} root")
        _require(
            route.get("all_frames_navigable") is True
            and route.get("navigable_frame_count") == route.get("frame_count") == 270
            and route.get("island_id") == 1,
            f"{source_slot} retained room/island context is invalid",
        )
        island_ids.add(route.get("island_id"))
        fresh = fresh_positions.get(source_slot)
        _require(
            isinstance(fresh, Mapping)
            and fresh.get("requested_m") == position
            and fresh.get("is_navigable") is True
            and fresh.get("island_id") == route.get("island_id") == 1
            and float(fresh.get("snap_error_m", math.inf)) <= 1.0e-5
            and float(fresh.get("clearance_m", -math.inf)) >= 0.30,
            f"{source_slot} failed fresh navmesh island/clearance replay",
        )
        selected[source_slot] = {
            "retained_route_id": route_id,
            "habitat_root_m": position,
            "ue_root_cm": _habitat_to_ue_cm(position),
            "island_id": route["island_id"],
            "maximum_snap_error_m": route["maximum_snap_error_m"],
            "fresh_snap_error_m": fresh["snap_error_m"],
            "fresh_clearance_m": fresh["clearance_m"],
            "all_frames_navigable": True,
            "reuse_scope": (
                "retained route proves room/island context only; exact selected "
                "root comes from the fresh PathFinder search"
            ),
        }
    _require(island_ids == {1}, "selected source roots are not on one retained island")
    first = selected["source1"]["habitat_root_m"]
    second = selected["source2"]["habitat_root_m"]
    separation = math.hypot(first[0] - second[0], first[2] - second[2])
    basic_root_gate = separation >= 0.8
    _require(basic_root_gate, "selected source roots are too close even for a probe")
    adult_clearance_gate = all(
        item["fresh_clearance_m"] >= 0.50 for item in selected.values()
    )
    adult_separation_gate = separation >= 1.30
    adult_pair_gate = adult_clearance_gate and adult_separation_gate
    return {
        "status": "pass" if adult_pair_gate else "blocked",
        "authority": (
            "fresh real Habitat PathFinder pair search with retained room/island context"
        ),
        "semantics": "actor_root_center_only",
        "selected_positions": selected,
        "shared_island_id": 1,
        "horizontal_source_separation_m": separation,
        "adult_static_pair_gate": {
            "status": "pass" if adult_pair_gate else "blocked",
            "minimum_each_root_clearance_m": 0.50,
            "minimum_horizontal_separation_m": 1.30,
            "clearance_gate_passed": adult_clearance_gate,
            "separation_gate_passed": adult_separation_gate,
            "claim_boundary": (
                "navmesh root-distance heuristic only; articulated-body and arm "
                "collision still require sparse visual review"
            ),
        },
        "female_rebind_scope": (
            "source2 is a fresh >=0.5m-clearance root on island 1; dog0 is retained "
            "only as room/island context and the runtime entity is a distinct female"
        ),
        "fresh_pathfinder_replay_status": "pass",
        "fresh_probe_date": fresh_probe["executed_at_date"],
        "minimum_root_probe_clearance_m": 0.30,
        "recommended_adult_clearance_m": 0.50,
        "qualification_claim": False,
    }


def _project_mouth_proxies(request: Mapping[str, Any]) -> dict[str, Any]:
    """Project declared root-plus-mouth proxies into the planned BP camera."""

    camera = request["camera_listener"]
    camera_ue = _vector3(camera["ue_position_cm"], owner="camera UE position")
    yaw = math.radians(float(camera["ue_yaw_deg"]))
    forward = [math.cos(yaw), math.sin(yaw), 0.0]
    right = [-math.sin(yaw), math.cos(yaw), 0.0]
    width, height = 1280, 720
    hfov = float(camera["horizontal_fov_deg"])
    focal = 0.5 * width / math.tan(math.radians(hfov) / 2.0)
    principal = [(width - 1) / 2.0, (height - 1) / 2.0]
    placement = request["placement"]
    projected: dict[str, Any] = {}
    for slot in ("source1", "source2"):
        mouth_h = _add3(
            placement[f"{slot}_root_habitat_m"],
            placement[f"{slot}_emitter_offset_m"],
        )
        mouth_ue = _habitat_to_ue_cm(mouth_h)
        relative = [mouth_ue[i] - camera_ue[i] for i in range(3)]
        depth = sum(relative[i] * forward[i] for i in range(3))
        horizontal = sum(relative[i] * right[i] for i in range(3))
        _require(depth > 0.0, f"{slot} mouth proxy is behind the camera")
        pixel = [
            principal[0] + focal * horizontal / depth,
            principal[1] - focal * relative[2] / depth,
        ]
        radius = 48.0
        envelope = [
            pixel[0] - radius,
            pixel[1] - radius,
            pixel[0] + radius,
            pixel[1] + radius,
        ]
        _require(
            envelope[0] >= 0.0
            and envelope[1] >= 0.0
            and envelope[2] < width
            and envelope[3] < height,
            f"{slot} planned mouth audit envelope escapes the frame",
        )
        projected[slot] = {
            "mouth_proxy_habitat_m": mouth_h,
            "mouth_proxy_ue_cm": mouth_ue,
            "camera_depth_cm": depth,
            "pixel_uv": pixel,
            "audit_envelope_xyxy_px": envelope,
            "audit_envelope_semantics": (
                "planned 48px-radius review window around root-plus-declared-mouth-offset; "
                "not a live mouth bone/socket or body bbox"
            ),
            "inside_frame": True,
        }
    separation = (
        projected["source2"]["pixel_uv"][0] - projected["source1"]["pixel_uv"][0]
    )
    minimum_separation = 256.0
    if request.get("schema") == REQUEST_SCHEMA_V2:
        camera_framing = request.get("camera_framing")
        _require(
            isinstance(camera_framing, Mapping),
            "v2 request lacks camera_framing",
        )
        minimum_separation = float(
            camera_framing.get("minimum_mouth_proxy_separation_px")
        )
        _require(
            math.isfinite(minimum_separation) and minimum_separation >= 0.0,
            "v2 mouth proxy separation must be finite and nonnegative",
        )
    _require(
        separation >= minimum_separation,
        "planned mouths lack declared left/right separation",
    )
    return {
        "status": "proxy_projection_pass_live_bbox_pending",
        "coordinate_chain": [
            "Matterport raw GLB Z-up/+Y-front",
            "Habitat H=(S.x,S.z,-S.y), right-handed Y-up/-Z-forward",
            "UE U_cm=(100*H.x,100*H.z,100*H.y)",
            "BP_CameraSensor yaw-only pinhole projection",
        ],
        "resolution_hw": [height, width],
        "horizontal_fov_deg": hfov,
        "focal_length_px": focal,
        "principal_point_uv": principal,
        "source_order": ["source1_left", "source2_right"],
        "horizontal_mouth_separation_px": separation,
        "minimum_horizontal_mouth_separation_px": minimum_separation,
        "per_source": projected,
        "live_human_bbox_status": "pending_sparse_f15",
        "live_mouth_bone_or_socket_status": "pending_not_declared",
        "qualification_claim": False,
    }


def _validate_acoustic_registration(
    manifest: Mapping[str, Any],
    room_registry: Mapping[str, Any],
    profile_registry: Mapping[str, Any],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    room = request["room"]
    _require(
        manifest.get("schema") == "avengine_acoustic_scene_package_v1"
        and manifest.get("package_id")
        == "habitat_mp3d_example_17DRP5sb8fy_soundspaces2_public_v1"
        and manifest.get("package_mode") == "research_candidate",
        "MP3D SoundSpaces package identity drift",
    )
    source_room = manifest.get("source_room")
    materials = manifest.get("materials")
    geometry = manifest.get("geometry")
    _require(
        isinstance(source_room, Mapping)
        and source_room.get("room_id") == room["room_id"]
        and isinstance(materials, Mapping)
        and materials.get("material_semantics") == "research_placeholder"
        and materials.get("qualification_claim") == "unqualified_research_placeholder"
        and isinstance(geometry, Mapping)
        and geometry.get("vertex_count") == 1_570_132
        and geometry.get("triangle_count") == 3_016_249,
        "MP3D acoustic package closure drift",
    )
    records = room_registry.get("records")
    _require(isinstance(records, list), "room registry lacks records")
    rooms = [
        item
        for item in records
        if item.get("room_id") == room["room_id"]
        and item.get("revision") == room["room_revision"]
    ]
    _require(len(rooms) == 1, "MP3D room registry selection is not unique")
    representations = rooms[0].get("acoustic_representations")
    _require(
        isinstance(representations, list)
        and any(
            item.get("representation_id") == "mp3d_17DRP5sb8fy_soundspaces2_acoustic_v1"
            and item.get("resource_id") == "mp3d_soundspaces2_acoustic_package_v1"
            for item in representations
        ),
        "MP3D SoundSpaces representation is not room-registered",
    )
    bindings = profile_registry.get("bindings")
    profiles = profile_registry.get("profiles")
    _require(
        isinstance(bindings, list) and isinstance(profiles, list),
        "profile registry invalid",
    )
    selected_bindings = [
        item
        for item in bindings
        if item.get("binding_id") == "mp3d_17DRP5sb8fy_soundspaces2_v1"
        and item.get("room_ref", {}).get("room_id") == room["room_id"]
        and item.get("room_ref", {}).get("revision") == room["room_revision"]
    ]
    _require(len(selected_bindings) == 1, "MP3D acoustic binding is not unique")
    binding = selected_bindings[0]
    profile_ref = binding.get("profile_ref")
    _require(isinstance(profile_ref, Mapping), "MP3D profile_ref is missing")
    selected_profiles = [
        item
        for item in profiles
        if item.get("profile_id") == profile_ref.get("profile_id")
        and item.get("revision") == profile_ref.get("revision")
    ]
    _require(len(selected_profiles) == 1, "MP3D acoustic profile is not unique")
    return {
        "status": "pass",
        "selection_mode": "room_and_profile_registry",
        "binding_id": binding["binding_id"],
        "room_ref": dict(binding["room_ref"]),
        "representation_id": binding["acoustic_representation_id"],
        "profile_ref": dict(profile_ref),
        "retained_reference_package": {
            "package_id": manifest["package_id"],
            "vertex_count": geometry["vertex_count"],
            "triangle_count": geometry["triangle_count"],
            "material_semantics": materials["material_semantics"],
        },
        "fresh_compile_required": True,
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }


def _actor_frame_state(
    declaration: Mapping[str, Any],
    *,
    root: Sequence[float],
    camera: Sequence[float],
    frame_index: int,
) -> dict[str, Any]:
    delta_x = camera[0] - root[0]
    delta_z = camera[2] - root[2]
    norm = math.hypot(delta_x, delta_z)
    _require(norm > 0.0, "actor and camera positions coincide")
    forward_h = [delta_x / norm, 0.0, delta_z / norm]
    habitat_yaw = math.degrees(math.atan2(forward_h[0], forward_h[2]))
    ue_forward = [forward_h[0], forward_h[2], 0.0]
    desired_ue_yaw = math.degrees(math.atan2(ue_forward[1], ue_forward[0]))
    actor_yaw_ue = desired_ue_yaw - float(declaration["ue_anatomical_forward_yaw_deg"])
    half = math.radians(habitat_yaw) / 2.0
    return {
        "frame_index": frame_index,
        "actor_id": declaration["actor_id"],
        "asset_id": declaration["asset_id"],
        "blueprint_class_path": declaration["blueprint_class_path"],
        "translation_m": list(root),
        "translation_ue_cm": _habitat_to_ue_cm(root),
        "rotation_xyzw": [0.0, math.sin(half), 0.0, math.cos(half)],
        "actor_yaw_ue_deg": actor_yaw_ue,
        "anatomical_forward_habitat_world": forward_h,
        "anatomical_forward_ue_world": ue_forward,
        "action_id": "idle",
        "action_phase": 0.0,
        "action_time_ticks": frame_index * TICKS_PER_FRAME,
        "ue_animation": declaration["idle_animation"],
    }


def _build_suite(
    request: Mapping[str, Any],
    template_suite: Mapping[str, Any],
    room_adapter: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    scenarios = template_suite.get("scenarios")
    _require(
        isinstance(scenarios, list) and len(scenarios) == 1,
        "strict template closure drift",
    )
    template_scenario = scenarios[0]
    actors = deepcopy(template_scenario["plan"]["actors"])
    _require(
        [item.get("actor_id") for item in actors] == ["source1_actor", "source2_actor"]
        and actors[0].get("body_plan_id")
        == actors[1].get("body_plan_id")
        == "biped_human"
        and "male_adult_01" in actors[0].get("template_id", "")
        and "female_adult_01" in actors[1].get("template_id", ""),
        "template must bind one male and one distinct female human",
    )
    placement = request["placement"]
    actors[0]["emitter_anchor_id"] = "source1_declared_mouth_proxy"
    actors[0]["emitter_offset_m"] = _vector3(
        placement["source1_emitter_offset_m"], owner="source1 emitter offset"
    )
    actors[1]["emitter_anchor_id"] = "source2_declared_mouth_proxy"
    actors[1]["emitter_offset_m"] = _vector3(
        placement["source2_emitter_offset_m"], owner="source2 emitter offset"
    )
    source_roots = {
        "source1_actor": _vector3(
            placement["source1_root_habitat_m"], owner="source1 root"
        ),
        "source2_actor": _vector3(
            placement["source2_root_habitat_m"], owner="source2 root"
        ),
    }
    camera = request["camera_listener"]
    camera_h = _vector3(camera["habitat_position_m"], owner="camera position")
    camera_ue = _vector3(camera["ue_position_cm"], owner="camera UE position")
    _require(
        max(abs(a - b) for a, b in zip(camera_ue, _habitat_to_ue_cm(camera_h)))
        <= 1.0e-6,
        "camera Habitat-to-UE projection drift",
    )
    yaw_h = float(camera["habitat_yaw_deg"])
    yaw_ue = float(camera["ue_yaw_deg"])
    _require(abs(yaw_ue - (-90.0 - yaw_h)) <= 1.0e-9, "camera yaw conversion drift")
    half = math.radians(yaw_h) / 2.0
    world_from_rig = {
        "translation_m": camera_h,
        "rotation_xyzw": [0.0, math.sin(half), 0.0, math.cos(half)],
    }
    pose_hash = _canonical_sha256(world_from_rig)
    frames = []
    rig_frames = []
    for frame_index in range(FRAME_COUNT):
        pts = frame_index * TICKS_PER_FRAME
        camera_state = {
            "frame_index": frame_index,
            "pts_ticks": pts,
            "pose_hash": pose_hash,
            "habitat_position_m": camera_h,
            "habitat_yaw_deg": yaw_h,
            "ue_position_cm": camera_ue,
            "ue_yaw_deg": yaw_ue,
            "world_from_rig": world_from_rig,
        }
        frames.append(
            {
                "frame_index": frame_index,
                "pts_ticks": pts,
                "camera_state": camera_state,
                "actor_states": [
                    _actor_frame_state(
                        declaration,
                        root=source_roots[declaration["actor_id"]],
                        camera=camera_h,
                        frame_index=frame_index,
                    )
                    for declaration in actors
                ],
            }
        )
        rig_frames.append(
            {
                "frame_index": frame_index,
                "pts_ticks": pts,
                "pose_hash": pose_hash,
                "world_from_rig": world_from_rig,
            }
        )
    episode_id = request["episode_id"]
    plan = deepcopy(template_scenario["plan"])
    plan["actors"] = actors
    plan["frames"] = frames
    plan["camera"] = {
        "dynamic": False,
        "habitat_position_m": camera_h,
        "habitat_yaw_deg": yaw_h,
        "ue_position_cm": camera_ue,
        "ue_yaw_deg": yaw_ue,
        "horizontal_fov_deg": float(camera["horizontal_fov_deg"]),
        "listener_id": "listener0",
        "sensor_rig_trajectory_id": f"{episode_id}__sensor_rig",
    }
    plan["coordinate_contract"] = {
        "habitat_to_ue_position": "U_cm=(100*H_x,100*H_z,100*H_y)",
        "camera_yaw": "UE_yaw_deg=-90-Habitat_yaw_deg",
        "actor_yaw": (
            "yaw(UE(world_from_actor*Habitat_local_anatomical_forward))"
            "-UE_asset_local_forward_yaw"
        ),
    }
    plan["room"] = {
        "room_id": request["room"]["room_id"],
        "room_revision": request["room"]["room_revision"],
        "scene_id": request["room"]["scene_id"],
        "runtime_map": ENTRY_MAP,
        "room_adapter": room_adapter,
    }
    plan["qualification"] = {
        "status": "research_preflight_only",
        "source_root_navmesh_gate_status": (
            "fresh_adult_clearance_and_separation_pass_body_bbox_pending"
        ),
        "full_body_clearance_claim": False,
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }
    plan["source_logic"]["scenario_id"] = episode_id
    scenario = deepcopy(template_scenario)
    scenario.update(
        {
            "schema": SCENARIO_SCHEMA,
            "scenario_id": episode_id,
            "scenario_directory": episode_id,
            "variant_id": "mp3d_strict_two_human_static_v1",
            "plan": plan,
            "native_scene": {
                "layout": "spawn_reload_verified_imported_glb_mesh_closure",
                "map": ENTRY_MAP,
                "room_adapter": room_adapter,
                "lighting": "retained_mp3d_entry_map_review_profile",
            },
            "render": {
                "frame_count": FRAME_COUNT,
                "frame_rate_hz": FPS,
                "width": 1280,
                "height": 720,
                "horizontal_fov_deg": float(camera["horizontal_fov_deg"]),
                "streaming_warmup_frames": 120,
                "camera_warmup_frames": 40,
            },
            "authoritative_capture_request": {
                "request_id": f"{episode_id}__native_capture",
                "episode_id": episode_id,
                "scenario_type": "strict_two_human_static_mp3d_research_probe",
                "target_source_slot_id": "source1",
                "fact_path": "PENDING_NATIVE_CAPTURE",
                "fact_selector": "/capture_fact",
            },
            "reuse_contract": {
                "actors": "distinct retained male/female Rocketbox runtime bindings",
                "room": "reload-verified 71-mesh MP3D import spawned in Entry map",
                "placement": "retained starts plus fresh PathFinder replay; actor-root only",
                "audio": "new exact two-source MP3D RIR cache required",
            },
        }
    )
    suite = {
        "schema": SUITE_SCHEMA,
        "backend_role": "comparison_visual",
        "native_map": ENTRY_MAP,
        "room_adapter": room_adapter,
        "authority": {
            "habitat_native": [
                "actor-root navmesh placement",
                "Timeline_v2",
                "source-center acoustic positions",
                "binaural audio",
            ],
            "spear_unreal": [
                "fresh cooked mesh load/readback",
                "normal RGB and metric depth",
                "source1/source2 target-only metric depth",
            ],
        },
        "scenarios": [scenario],
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }
    rig = {
        "schema": "avengine_sensor_rig_trajectory_v1",
        "trajectory_id": f"{episode_id}__sensor_rig",
        "formal_view_id": "view0",
        "camera_listener_coupling": "rigid_colocated_cooriented",
        "coordinate_frame": "avengine_world_right_handed_y_up_m",
        "frame_count": FRAME_COUNT,
        "frame_rate_hz": FPS,
        "duration_ticks": FRAME_COUNT * TICKS_PER_FRAME,
        "frames": rig_frames,
    }
    return suite, rig


def _solve_full75_actor_framing(
    request: Mapping[str, Any], suite: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind sampled actor envelopes to explicit planning-only camera candidates."""

    actor_contract = request.get("actor_framing")
    camera_contract = request.get("camera_framing")
    _require(
        isinstance(actor_contract, Mapping) and isinstance(camera_contract, Mapping),
        "request must declare actor_framing and camera_framing CPU planning contracts",
    )
    candidates = camera_contract.get("candidates")
    _require(
        isinstance(candidates, list) and candidates,
        "camera_framing.candidates must be a non-empty explicit list",
    )
    for candidate in candidates:
        _require(isinstance(candidate, Mapping), "camera candidate must be an object")
        room_gate = candidate.get("room_gate")
        _require(
            isinstance(room_gate, Mapping)
            and room_gate.get("provenance") == "declared_cpu_planning"
            and room_gate.get("native_habitat_validation_status") == "pending"
            and room_gate.get("line_of_sight_validation_status") == "pending"
            and room_gate.get("full_body_clearance_status") == "pending",
            (
                "camera room_gate must be declared_cpu_planning with native Habitat, "
                "line-of-sight, and full-body clearance explicitly pending"
            ),
        )
    plan_frames = suite["scenarios"][0]["plan"]["frames"]
    actor_inputs = build_actor_framing_frames(
        actor_bindings=actor_contract.get("actor_bindings"),
        frame_states=[
            {
                "frame_index": frame["frame_index"],
                "actor_states": frame["actor_states"],
            }
            for frame in plan_frames
        ],
        sample_rate_hz=actor_contract.get("sample_rate_hz", 120.0),
        padding_m=actor_contract.get("padding_m", 0.02),
        expected_frame_count=FRAME_COUNT,
    )
    actor_inputs["actor_orientation_policy"] = (
        "frozen_suite_actor_states_not_retargeted_to_selected_camera"
    )
    solution = solve_static_camera_candidates(
        frames=actor_inputs["frames"],
        candidates=candidates,
        calibration=camera_contract.get("calibration"),
        trajectory_id=f"{request['episode_id']}__sensor_rig",
        ordered_actor_ids=camera_contract.get("ordered_actor_ids"),
        minimum_order_gap_px=camera_contract.get("minimum_order_gap_px"),
    )
    _require(
        solution["selected_candidate_id"] is not None,
        "no explicit CPU planning camera candidate contains both actor envelopes",
    )
    return actor_inputs, solution


def _runtime_gate_and_solve_full75_actor_framing(
    request: Mapping[str, Any],
    suite: Mapping[str, Any],
    *,
    runtime_provider: Any,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Admit candidates in one runtime before solving sampled actor framing."""

    actor_contract = request.get("actor_framing")
    camera_contract = request.get("camera_framing")
    _require(
        isinstance(actor_contract, Mapping) and isinstance(camera_contract, Mapping),
        "fresh request must declare actor_framing and camera_framing contracts",
    )
    scenario = suite["scenarios"][0]
    plan = scenario["plan"]
    actor_declarations = {actor["actor_id"]: actor for actor in plan["actors"]}
    _require(
        len(actor_declarations) >= 2,
        "runtime camera gate requires at least two declared actors",
    )
    bindings = actor_contract.get("actor_bindings")
    _require(isinstance(bindings, list), "actor_framing.actor_bindings must be a list")
    bindings_by_id = {
        binding.get("actor_id"): binding
        for binding in bindings
        if isinstance(binding, Mapping)
    }
    _require(
        len(bindings_by_id) == len(bindings) == len(actor_declarations)
        and set(bindings_by_id) == set(actor_declarations),
        "actor framing bindings must exactly close suite actors",
    )
    for actor_id, declaration in actor_declarations.items():
        binding = bindings_by_id[actor_id]
        _require(
            binding.get("asset_id") == declaration.get("asset_id")
            and binding.get("asset_revision") == declaration.get("asset_revision"),
            f"{actor_id} framing asset identity differs from suite declaration",
        )
    plan_frames = plan["frames"]
    actor_inputs = build_actor_framing_frames(
        actor_bindings=actor_contract.get("actor_bindings"),
        frame_states=[
            {
                "frame_index": frame["frame_index"],
                "actor_states": frame["actor_states"],
            }
            for frame in plan_frames
        ],
        sample_rate_hz=actor_contract.get("sample_rate_hz", 120.0),
        padding_m=actor_contract.get("padding_m", 0.02),
        expected_frame_count=FRAME_COUNT,
    )
    actor_inputs["actor_orientation_policy"] = (
        "frozen_suite_actor_states_not_retargeted_to_selected_camera"
    )
    floor_paths: dict[str, list[list[float]]] = {
        actor_id: [] for actor_id in actor_declarations
    }
    visibility: dict[str, dict[str, list[list[float]]]] = {
        actor_id: {"torso_envelope_center": [], "declared_emitter_proxy": []}
        for actor_id in actor_declarations
    }
    for frame, framing_frame in zip(plan_frames, actor_inputs["frames"], strict=True):
        states = {state["actor_id"]: state for state in frame["actor_states"]}
        _require(
            set(states) == set(actor_declarations),
            "suite actor states differ from actor declarations",
        )
        for actor_id, declaration in actor_declarations.items():
            state = states[actor_id]
            _require(
                state.get("asset_id") == declaration.get("asset_id"),
                f"{actor_id} state asset differs from declaration",
            )
            root = _vector3(state.get("translation_m"), owner=f"{actor_id} root")
            emitter = _add3(
                root,
                _vector3(
                    declaration.get("emitter_offset_m"),
                    owner=f"{actor_id} declared emitter offset",
                ),
            )
            floor_paths[actor_id].append(root)
            bounds = framing_frame["actor_aabbs"][actor_id]
            minimum = bounds["minimum_m"]
            maximum = bounds["maximum_m"]
            torso = [
                (float(minimum[0]) + float(maximum[0])) / 2.0,
                float(minimum[1]) + 0.55 * (float(maximum[1]) - float(minimum[1])),
                (float(minimum[2]) + float(maximum[2])) / 2.0,
            ]
            visibility[actor_id]["torso_envelope_center"].append(torso)
            visibility[actor_id]["declared_emitter_proxy"].append(emitter)

    generation = camera_contract.get("candidate_generation")
    _require(isinstance(generation, Mapping), "camera candidate_generation is missing")
    offsets = generation.get("offsets_xz_m")
    _require(isinstance(offsets, list) and offsets, "candidate offsets are missing")
    floor_height = float(camera_contract.get("floor_height_m"))
    eye_height = float(generation.get("eye_height_m"))
    first_roots = [floor_paths[actor_id][0] for actor_id in sorted(floor_paths)]
    midpoint = [
        sum(point[axis] for point in first_roots) / len(first_roots)
        for axis in range(3)
    ]
    declared_candidates = []
    for index, offset in enumerate(offsets):
        values = _vector3(offset, owner=f"candidate offset {index}")
        position = [
            midpoint[0] + values[0],
            floor_height + eye_height,
            midpoint[2] + values[2],
        ]
        delta_x = midpoint[0] - position[0]
        delta_z = midpoint[2] - position[2]
        _require(math.hypot(delta_x, delta_z) > 0.0, "camera candidate hits midpoint")
        declared_candidates.append(
            {
                "candidate_id": f"midpoint_grid_{index:03d}",
                "position_m": position,
                "yaw_deg": math.degrees(math.atan2(-delta_x, -delta_z)),
                "priority": index,
            }
        )
    gate_results = evaluate_camera_candidates(
        runtime_provider=runtime_provider,
        candidates=declared_candidates,
        actor_floor_paths_m=floor_paths,
        actor_visibility_anchors_m=visibility,
        floor_height_m=floor_height,
        evaluation_id=f"{request['episode_id']}__camera-runtime-gate",
        maximum_y_delta_m=camera_contract.get("maximum_y_delta_m", 0.25),
        maximum_snap_error_m=camera_contract.get("maximum_snap_error_m", 0.05),
        minimum_clearance_m=camera_contract.get("minimum_clearance_m", 0.25),
        maximum_clearance_query_m=camera_contract.get(
            "maximum_clearance_query_m", 10.0
        ),
        line_of_sight_tolerance_m=camera_contract.get(
            "line_of_sight_tolerance_m", 0.03
        ),
    )
    declared_by_id = {
        candidate["candidate_id"]: candidate for candidate in declared_candidates
    }
    _require(
        len(declared_by_id) == len(declared_candidates),
        "generated camera candidate IDs must be unique",
    )
    gate_ids = [result.get("candidate_id") for result in gate_results]
    _require(
        len(gate_ids) == len(set(gate_ids)) and set(gate_ids) == set(declared_by_id),
        "runtime gate results must exactly cover generated candidates",
    )
    passing_candidates = []
    for result in gate_results:
        if result.get("status") != "pass":
            continue
        room_gate = result.get("room_gate")
        _require(isinstance(room_gate, Mapping), "passing runtime gate lacks room_gate")
        candidate = deepcopy(declared_by_id[result["candidate_id"]])
        candidate["room_gate"] = deepcopy(room_gate)
        passing_candidates.append(candidate)
    _require(passing_candidates, "no camera candidate passed the Habitat runtime gate")

    solution = solve_static_camera_candidates(
        frames=actor_inputs["frames"],
        candidates=passing_candidates,
        calibration=camera_contract.get("calibration"),
        trajectory_id=f"{request['episode_id']}__sensor_rig",
        ordered_actor_ids=camera_contract.get("ordered_actor_ids"),
        minimum_order_gap_px=camera_contract.get("minimum_order_gap_px"),
    )
    _require(
        solution.get("selected_candidate_id") is not None,
        "no runtime-admitted camera candidate contains both actor envelopes",
    )
    return actor_inputs, solution, gate_results


def _bind_v2_actor_revisions(
    suite: Mapping[str, Any],
    request: Mapping[str, Any],
    runtime_profiles: Mapping[str, Any],
) -> dict[str, Any]:
    result = deepcopy(suite)
    declarations = {
        actor["actor_id"]: actor for actor in result["scenarios"][0]["plan"]["actors"]
    }
    bindings = request["actor_framing"]["actor_bindings"]
    _require(
        isinstance(bindings, list)
        and len(bindings) == len(declarations)
        and all(isinstance(binding, Mapping) for binding in bindings),
        "v2 actor bindings differ from suite actors",
    )
    profiles = runtime_profiles.get("assets")
    _require(isinstance(profiles, list), "runtime profile registry lacks assets")
    for binding in bindings:
        actor_id = binding["actor_id"]
        _require(actor_id in declarations, "v2 actor binding contains unknown actor")
        declaration = declarations[actor_id]
        revision = str(binding.get("asset_revision", ""))
        _require(
            binding.get("asset_id") == declaration.get("asset_id") and revision,
            f"{actor_id} v2 asset identity drift",
        )
        selected = [
            profile
            for profile in profiles
            if isinstance(profile, Mapping)
            and profile.get("asset_id") == binding.get("asset_id")
            and profile.get("revision") == revision
        ]
        _require(len(selected) == 1, f"{actor_id} runtime profile is not unique")
        profile = selected[0]
        timeline = profile["timeline"]
        unreal = profile["runtime_backends"]["spear_unreal"]
        anchors = [
            item
            for item in profile["emitter_anchors"]
            if item["anchor_id"] == profile["default_emitter_anchor_id"]
        ]
        _require(len(anchors) == 1, f"{actor_id} default emitter anchor is not unique")
        anchor = anchors[0]
        _require(
            declaration.get("template_id") == timeline["template_id"]
            and declaration.get("body_plan_id") == timeline["body_plan_id"]
            and declaration.get("habitat_local_anatomical_forward_axis")
            == timeline["local_anatomical_forward_axis"]
            and declaration.get("blueprint_class_path")
            == unreal["blueprint_class_path"]
            and declaration.get("idle_animation") == unreal["idle_animation"]
            and declaration.get("walking_animation") == unreal["walking_animation"]
            and declaration.get("emitter_offset_m") == anchor["offset_m"]
            and binding.get("action_name_by_action_id")
            == {"idle": "Standing_Idle", "walk": "Walking"},
            f"{actor_id} differs from selected runtime profile",
        )
        declaration["asset_revision"] = revision
    return result


def _add3(first: Sequence[float], second: Sequence[float]) -> list[float]:
    _require(len(first) == len(second) == 3, "3D vector length drift")
    return [float(a + b) for a, b in zip(first, second)]


def _apply_selected_sensor_rig(
    suite: Mapping[str, Any], solution: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Atomically bind the selected canonical HOLD rig into all suite locations."""

    binding = solution.get("sensor_rig_binding")
    _require(isinstance(binding, Mapping), "framing solution lacks sensor rig binding")
    rig = deepcopy(binding.get("trajectory"))
    _require(
        isinstance(rig, Mapping)
        and isinstance(rig.get("frames"), list)
        and len(rig["frames"]) == FRAME_COUNT,
        "selected sensor rig must contain 75 frames",
    )
    result = deepcopy(suite)
    plan = result["scenarios"][0]["plan"]
    plan_frames = plan["frames"]
    _require(len(plan_frames) == FRAME_COUNT, "suite frame count drift")
    selected_pose = solution.get("selected_camera_pose")
    _require(isinstance(selected_pose, Mapping), "selected camera pose is missing")
    selected_yaw = float(selected_pose["yaw_deg"])
    expected_trajectory_id = f"{suite['scenarios'][0]['scenario_id']}__sensor_rig"
    _require(
        rig.get("trajectory_id") == expected_trajectory_id,
        "selected sensor rig trajectory_id differs from scenario",
    )
    first_position = list(rig["frames"][0]["world_from_rig"]["translation_m"])
    _require(
        first_position == list(selected_pose["position_m"])
        and all(
            frame["world_from_rig"] == rig["frames"][0]["world_from_rig"]
            for frame in rig["frames"]
        ),
        "selected sensor rig is not the selected canonical HOLD pose",
    )
    for suite_frame, rig_frame in zip(plan_frames, rig["frames"], strict=True):
        frame_index = int(suite_frame["frame_index"])
        _require(
            rig_frame["frame_index"] == frame_index
            and rig_frame["pts_ticks"] == suite_frame["pts_ticks"],
            "suite and selected rig clocks differ",
        )
        suite_frame["camera_state"] = {
            "frame_index": frame_index,
            "pts_ticks": rig_frame["pts_ticks"],
            "pose_hash": rig_frame["pose_hash"],
            "world_from_rig": deepcopy(rig_frame["world_from_rig"]),
            "habitat_position_m": list(rig_frame["world_from_rig"]["translation_m"]),
            "habitat_yaw_deg": selected_yaw,
            "ue_position_cm": _habitat_to_ue_cm(
                rig_frame["world_from_rig"]["translation_m"]
            ),
            "ue_yaw_deg": -90.0 - selected_yaw,
        }
    plan["camera"] = {
        **deepcopy(plan["camera"]),
        "dynamic": False,
        "habitat_position_m": first_position,
        "habitat_yaw_deg": selected_yaw,
        "ue_position_cm": _habitat_to_ue_cm(first_position),
        "ue_yaw_deg": -90.0 - selected_yaw,
        "sensor_rig_trajectory_id": rig["trajectory_id"],
        "actor_orientation_policy": (
            "frozen_suite_actor_states_not_retargeted_to_selected_camera"
        ),
    }
    for frame, rig_frame in zip(plan_frames, rig["frames"], strict=True):
        state = frame["camera_state"]
        _require(
            state["world_from_rig"] == rig_frame["world_from_rig"]
            and state["pose_hash"] == rig_frame["pose_hash"],
            "selected rig did not close suite camera state",
        )
    return result, rig


def _build_canonical_rir_plan(
    suite: Mapping[str, Any], rig: Mapping[str, Any], *, stride_frames: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    scenario = suite["scenarios"][0]
    plan = scenario["plan"]
    episode_id = scenario["scenario_id"]
    declarations = {actor["actor_id"]: actor for actor in plan["actors"]}
    roots = {"source1": [], "source2": []}
    centers = {"source1": [], "source2": []}
    actor_to_slot = {"source1_actor": "source1", "source2_actor": "source2"}
    for frame in plan["frames"]:
        for state in frame["actor_states"]:
            actor_id = state["actor_id"]
            slot = actor_to_slot[actor_id]
            root = _vector3(state["translation_m"], owner=f"{slot} root")
            offset = _vector3(
                declarations[actor_id]["emitter_offset_m"],
                owner=f"{slot} emitter offset",
            )
            roots[slot].append(root)
            centers[slot].append(_add3(root, offset))
    episode = TrajectoryEpisode(
        episode_id=episode_id,
        motion_case="strict_two_human_static_mp3d",
        source_root_paths_m={slot: np.asarray(path) for slot, path in roots.items()},
        source_center_paths_m={
            slot: np.asarray(path) for slot, path in centers.items()
        },
        statistics={
            "target_source_slot_id": "source1",
            "distractor_source_slot_id": "source2",
            "native_recapture_required": True,
        },
    )
    bank = TrajectoryBank(
        episodes=(episode,), frame_count=FRAME_COUNT, frame_rate_hz=FPS, seed=20260812
    )
    listener_positions = [
        frame["world_from_rig"]["translation_m"] for frame in rig["frames"]
    ]
    listener_orientations = [
        [rotation[3], rotation[0], rotation[1], rotation[2]]
        for rotation in (
            frame["world_from_rig"]["rotation_xyzw"] for frame in rig["frames"]
        )
    ]
    rir = build_rir_job_plan(
        bank,
        listener_positions_m_by_episode={episode_id: listener_positions},
        listener_orientations_wxyz_by_episode={episode_id: listener_orientations},
        stride_frames=stride_frames,
    )
    expected_frames = list(range(0, FRAME_COUNT, stride_frames))
    _require(
        rir["requested_pair_state_count"] == 2 * len(expected_frames),
        "canonical RIR request count drift",
    )
    uses = [use for job in rir["jobs"] for use in job["uses"]]
    _require(
        {(use["source_slot_id"], use["frame_index"]) for use in uses}
        == {(slot, index) for slot in roots for index in expected_frames},
        "canonical RIR uses do not close source slots and frames",
    )
    return bank.record(), rir


def _build_rir_plan(
    request: Mapping[str, Any], rig: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    episode_id = request["episode_id"]
    placement = request["placement"]
    centers = {
        "source1": _add3(
            placement["source1_root_habitat_m"],
            placement["source1_emitter_offset_m"],
        ),
        "source2": _add3(
            placement["source2_root_habitat_m"],
            placement["source2_emitter_offset_m"],
        ),
    }
    roots = {
        "source1": _vector3(placement["source1_root_habitat_m"], owner="source1 root"),
        "source2": _vector3(placement["source2_root_habitat_m"], owner="source2 root"),
    }
    trajectory_bank = {
        "schema": "avengine_room_trajectory_bank_v2",
        "seed": 20260812,
        "frame_count": FRAME_COUNT,
        "frame_rate_hz": FPS,
        "seconds_per_episode": 5.0,
        "episode_count": 1,
        "source_slots": ["source1", "source2"],
        "motion_case_counts": {
            "static_static": 1,
            "source1_moving_source2_static": 0,
            "source1_static_source2_moving": 0,
            "both_moving": 0,
        },
        "semantics": "distinct human source roots plus declared mouth offsets",
        "claim_boundary": "source-center only; no body-volume collision claim",
        "episodes": [
            {
                "episode_id": episode_id,
                "motion_case": "strict_two_human_static_mp3d",
                "source_root_paths_m": {
                    slot: [position] * FRAME_COUNT for slot, position in roots.items()
                },
                "source_center_paths_m": {
                    slot: [position] * FRAME_COUNT for slot, position in centers.items()
                },
                "statistics": {
                    "target_source_slot_id": "source1",
                    "distractor_source_slot_id": "source2",
                    "native_recapture_required": True,
                },
            }
        ],
    }
    stride = int(request["acoustics"]["rir_stride_frames"])
    _require(stride > 0 and (FRAME_COUNT - 1) % stride == 2, "RIR stride drift")
    listener = list(rig["frames"][0]["world_from_rig"]["translation_m"])
    rotation_xyzw = rig["frames"][0]["world_from_rig"]["rotation_xyzw"]
    listener_wxyz = [rotation_xyzw[3], *rotation_xyzw[:3]]
    uses_by_slot = {
        slot: [
            {
                "episode_id": episode_id,
                "frame_index": frame_index,
                "source_slot_id": slot,
            }
            for frame_index in range(0, FRAME_COUNT, stride)
        ]
        for slot in ("source1", "source2")
    }
    jobs = []
    for index, slot in enumerate(("source1", "source2")):
        state = {
            "source_position_m": centers[slot],
            "listener_position_m": listener,
            "listener_orientation_wxyz": listener_wxyz,
        }
        digest = _authoritative_rir_acoustic_state_sha256(
            state["source_position_m"],
            state["listener_position_m"],
            state["listener_orientation_wxyz"],
        )
        jobs.append(
            {
                "job_id": f"rir_{index:06d}_{digest[:16]}",
                "acoustic_state_sha256": digest,
                **state,
                "uses": uses_by_slot[slot],
            }
        )
    requested_states = sum(len(value) for value in uses_by_slot.values())
    plan = {
        "schema": "avengine_room_rir_job_plan_v2",
        "status": "planned_not_run",
        "producer_backend": "RLR Audio Propagation",
        "source_acoustic_profile": "omnidirectional_point_source_v1",
        "listener_pose_mode": "per_episode_frame",
        "cache_artifact": "room impulse response (RIR)",
        "cache_key_fields": [
            "source_position_m",
            "listener_position_m",
            "listener_orientation_wxyz",
        ],
        "slot_identity_affects_cache_key": False,
        "dry_audio_independent": True,
        "stride_frames": stride,
        "requested_pair_state_count": requested_states,
        "unique_listener_pose_count": 1,
        "unique_rir_job_count": len(jobs),
        "acoustic_state_sha256_authority": (
            "avengine.m6x.room_feasibility.rir_acoustic_state_sha256"
        ),
        "cache_reuse_count": requested_states - len(jobs),
        "claim_boundary": (
            "exact MP3D source/listener CPU RLR execution plan; native RLR has not run"
        ),
        "jobs": jobs,
    }
    _require(
        len(jobs) == 2, "static strict-two-human plan must deduplicate to 2 RIR jobs"
    )
    return trajectory_bank, plan


def _execution_plan(
    request: Mapping[str, Any],
    output: Path,
    *,
    rig: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    remote_root = REMOTE_REPOSITORY
    is_v2 = request.get("schema") == REQUEST_SCHEMA_V2
    if is_v2:
        preflight_output = output.resolve()
        remote_output = preflight_output.parent
        fresh_package = remote_output / "fresh_soundspaces2_package_v2"
        rir_cache = remote_output / "exact_rir_cache_v1"
        sparse_capture = remote_output / "native_sparse_f15_v1"
        full_capture = remote_output / "native_full75_v1"
        _require(
            isinstance(rig, Mapping)
            and isinstance(rig.get("frames"), list)
            and len(rig["frames"]) == FRAME_COUNT,
            "v2 execution plan requires the selected 75-frame sensor rig",
        )
        probe_origin = _vector3(
            rig["frames"][0]["world_from_rig"]["translation_m"],
            owner="selected RIR probe origin",
        )
        for owner, target in {
            "fresh acoustic package": fresh_package,
            "exact RIR cache": rir_cache,
            "sparse capture": sparse_capture,
            "full75 capture": full_capture,
        }.items():
            _require(not target.exists(), f"{owner} target already exists: {target}")
    else:
        remote_output = remote_root / "tmp/lead_a_mp3d_strict_two_human_room_atom_v1"
        fresh_package = remote_output / "fresh_soundspaces2_package_v1"
        preflight_output = remote_output / "cpu_preflight_v5"
        rir_cache = remote_output / "exact_rir_cache_v4"
        sparse_capture = remote_output / "native_sparse_f15_v1"
        full_capture = remote_output / "native_full75_v1"
        probe_origin = [-4.1499128342, 1.572447, -1.2454376221]
    runtime_probe = preflight_output / "rir_runtime_probe.json"
    suite = preflight_output / "suite_execution_plan.json"
    room_adapter = preflight_output / "room_adapter.json"
    rir_plan = preflight_output / "rir_job_plan.json"
    capture = request["capture"]
    acoustics = request["acoustics"]
    common_capture = [
        str(SPEAR_CAPTURE_PYTHON),
        str(
            remote_root
            / "tools/qa/capture_spear_imported_glb_strict_two_human_episode.py"
        ),
        "--suite-plan",
        str(suite),
        "--scenario-id",
        request["episode_id"],
        "--room-adapter",
        str(room_adapter),
        "--spear-root",
        capture["spear_root"],
        "--graphics-adapter",
        str(capture["graphics_adapter"]),
        "--rpc-port",
        str(capture["rpc_port"]),
    ]
    compile_argv = [
        str(HABITAT_PYTHON),
        "-m",
        "avengine.cli",
        "m3",
        "compile-mp3d-rlr-materials",
        "--room",
        str(remote_root / "examples/m1/rooms/habitat_mp3d_example/room_manifest.json"),
        "--materials",
        "/data/jzy/code/sound-spaces/data/mp3d_material_config.json",
        "--database-id",
        "soundspaces2_mp3d_public_materials_v1",
        "--version",
        "soundspaces_287184fd_rlr_4fd446b4",
        "--source-description",
        "SoundSpaces 2 public MP3D RLR materials",
        "--runtime-root",
        HABITAT_RUNTIME_ROOT,
        "--probe-origin",
        *(str(component) for component in probe_origin),
        "--output",
        str(fresh_package),
        "--package-id",
        (
            "habitat_mp3d_example_17DRP5sb8fy_soundspaces2_strict_two_human_v2"
            if is_v2
            else "habitat_mp3d_example_17DRP5sb8fy_soundspaces2_strict_two_human_v1"
        ),
    ]
    semantic_rir = (
        is_v2 and acoustics.get("rir_execution_mode") == SEMANTIC_RIR_EXECUTION_MODE
    )
    rir_argv = [
        str(HABITAT_PYTHON),
        str(remote_root / "tools/m6x/render_rir_cache.py"),
        "--rir-job-plan",
        str(rir_plan),
    ]
    if semantic_rir:
        rir_argv.extend(
            [
                "--semantic-no-file-evidence",
                "--acoustic-package-manifest",
                str(fresh_package / "manifest.json"),
                "--simulation-request",
                acoustics["simulation_request"],
                "--hrtf",
                acoustics["hrtf"],
            ]
        )
    else:
        rir_argv.extend(
            [
                "--room-id",
                request["room"]["room_id"],
                "--room-revision",
                request["room"]["room_revision"],
                "--room-registry",
                acoustics["room_registry"],
                "--acoustic-profile-registry",
                acoustics["acoustic_profile_registry"],
                "--simulation-profile",
                acoustics["simulation_profile"],
            ]
        )
    rir_argv.extend(
        [
            "--output",
            str(rir_cache),
            "--layout",
            "binaural",
            "--batch-size",
            "2",
            "--thread-count",
            str(acoustics["thread_count"]),
        ]
    )
    execution_environment = {
        "AVENGINE_HABITAT_RUNTIME_ROOT": HABITAT_RUNTIME_ROOT,
        "AVENGINE_SOUNDSPACES_ROOT": SOUNDSPACES_ROOT,
        "AVENGINE_MP3D_SOUNDSPACES2_PACKAGE_ROOT": str(fresh_package),
        "PATH": HABITAT_PATH,
        "PYTHONPATH": str(remote_root / "src"),
        "SKBUILD_EDITABLE_SKIP": HABITAT_EDITABLE_BUILD,
        "NUMBA_DISABLE_JIT": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    validate_rir_runtime_binding(HABITAT_PYTHON, execution_environment)
    return {
        "schema": (
            "avengine_native_strict_two_human_mp3d_execution_plan_v2"
            if is_v2
            else "avengine_native_strict_two_human_mp3d_execution_plan_v1"
        ),
        "status": "planned_not_run",
        "local_staging_output": str(output.resolve()),
        "remote_target_root": str(remote_output),
        "cpu_steps": [
            {
                "step_id": "probe_authoritative_habitat_rir_runtime",
                "working_directory": str(remote_root),
                "environment": execution_environment,
                "argv": [
                    str(HABITAT_PYTHON),
                    str(
                        remote_root
                        / "tools/qa/build_strict_two_human_mp3d_room_preflight.py"
                    ),
                    "--runtime-probe-output",
                    str(runtime_probe),
                ],
                "expected": {
                    "receipt": str(runtime_probe),
                    "status": "pass",
                    "python": "3.12.13",
                    "numpy": "2.3.5",
                    "quaternion": "2024.0.13",
                    "habitat_sim": "0.3.3",
                    "avengine_source": str(remote_root / "src/avengine/__init__.py"),
                    "compute_device": "CPU",
                    "cuda_initialized": False,
                    "qualification_claim": False,
                },
            },
            {
                "step_id": "fresh_compile_mp3d_rlr_materials",
                "working_directory": str(remote_root),
                "environment": execution_environment,
                "argv": compile_argv,
                "expected": {
                    "manifest": str(fresh_package / "manifest.json"),
                    "semantic_material_coverage": str(
                        fresh_package / "semantic_material_coverage.json"
                    ),
                    "package_mode": "research_candidate",
                    "cli_status": "research_candidate",
                    "qualification_claim": False,
                },
            },
            {
                "step_id": "render_two_exact_rirs",
                "attempt_id": rir_cache.name,
                "supersedes_failed_attempts": (
                    []
                    if is_v2
                    else [
                        "exact_rir_cache_v1",
                        "exact_rir_cache_v2",
                        "exact_rir_cache_v3",
                    ]
                ),
                "working_directory": str(remote_root),
                "environment": execution_environment,
                "argv": rir_argv,
                "expected": {
                    "compute_device": "CPU",
                    "selected_job_count": 2,
                    "full_plan_complete": True,
                    "layout": "binaural",
                    "receipt": str(rir_cache / "receipt.json"),
                    "index": str(rir_cache / "index.json"),
                },
            },
        ],
        "gpu_steps": [
            {
                "step_id": "sparse_f15_probe",
                "working_directory": str(remote_root),
                "request_status": "materialized_pending_live_bbox_and_mouth_review",
                "requires_physical_gpu1_idle": True,
                "cpu_preconditions": {
                    "fresh_navmesh_each_root_clearance_at_least_0_5m": True,
                    "fresh_navmesh_horizontal_separation_at_least_1_3m": True,
                },
                "argv": common_capture
                + [
                    "--output",
                    str(sparse_capture),
                    "--frame-index",
                    "15",
                ],
            },
            {
                "step_id": "full75_episode",
                "working_directory": str(remote_root),
                "requires_physical_gpu1_idle": True,
                "blocked_until": [
                    "fresh acoustic compile passes",
                    "2/2 exact CPU RIRs pass",
                    "sparse f15 room/actor/pixel review passes",
                ],
                "argv": common_capture
                + [
                    "--output",
                    str(full_capture),
                ],
            },
        ],
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }


def build(
    args: argparse.Namespace,
    *,
    runtime_provider_factory: Any = _open_mp3d_camera_runtime,
) -> Path:
    request = load_json_object(args.request.resolve(), owner="atom request")
    _validate_request(request)
    template_path = (args.template_suite or Path(request["template_suite"])).resolve()
    import_path = (
        args.ue_import_manifest or Path(request["room"]["ue_import_manifest"])
    ).resolve()
    runtime_path = (
        args.ue_runtime_evidence or Path(request["room"]["ue_runtime_evidence"])
    ).resolve()
    nav_probe_path = (
        args.fresh_navmesh_probe or Path(request["room"]["fresh_navmesh_probe"])
    ).resolve()
    acoustic_path = (
        args.acoustic_manifest or Path(request["acoustics"]["package_manifest"])
    ).resolve()
    room_registry_path = (
        args.room_registry or Path(request["acoustics"]["room_registry"])
    ).resolve()
    profiles_path = (
        args.acoustic_profiles
        or Path(request["acoustics"]["acoustic_profile_registry"])
    ).resolve()
    inputs = {
        "request": args.request.resolve(),
        "template_suite": template_path,
        "ue_import_manifest": import_path,
        "ue_runtime_evidence": runtime_path,
        "fresh_navmesh_probe": nav_probe_path,
        "acoustic_manifest": acoustic_path,
        "room_registry": room_registry_path,
        "acoustic_profiles": profiles_path,
    }
    is_v2 = request["schema"] == REQUEST_SCHEMA_V2
    if is_v2:
        profile_registry_path = Path(
            request["actor_framing"]["runtime_profile_registry"]
        ).resolve()
        inputs["runtime_profile_registry"] = profile_registry_path
    for owner, path in inputs.items():
        _require(path.is_file(), f"{owner} is missing: {path}")
    template = load_json_object(template_path, owner="strict template suite")
    import_manifest = load_json_object(import_path, owner="MP3D UE import")
    runtime_evidence = load_json_object(runtime_path, owner="MP3D UE evidence")
    fresh_navmesh_probe = load_json_object(
        nav_probe_path, owner="fresh MP3D navmesh probe"
    )
    if is_v2:
        _require(
            float(request["camera_runtime"]["agent_radius_m"])
            == float(fresh_navmesh_probe["agent_radius_m"]),
            "v2 camera runtime agent radius differs from fresh navmesh probe",
        )
    acoustic_manifest = load_json_object(acoustic_path, owner="MP3D acoustic package")
    room_registry = load_json_object(room_registry_path, owner="room registry")
    profiles = load_json_object(profiles_path, owner="acoustic profile registry")
    runtime_profiles = (
        load_json_object(
            inputs["runtime_profile_registry"], owner="source runtime profile registry"
        )
        if is_v2
        else None
    )
    room_adapter = build_room_adapter_record(
        import_manifest,
        execution_manifest_path=request["room"]["ue_import_manifest"],
    )
    validate_room_adapter(room_adapter)
    navigation = _validate_navigation(runtime_evidence, fresh_navmesh_probe, request)
    acoustic = _validate_acoustic_registration(
        acoustic_manifest, room_registry, profiles, request
    )
    suite, rig = _build_suite(request, template, room_adapter)
    actor_framing = None
    camera_framing = None
    runtime_camera_gates = None
    if is_v2:
        assert runtime_profiles is not None
        suite = _bind_v2_actor_revisions(suite, request, runtime_profiles)
        with runtime_provider_factory(request) as runtime_provider:
            actor_framing, camera_framing, runtime_camera_gates = (
                _runtime_gate_and_solve_full75_actor_framing(
                    request, suite, runtime_provider=runtime_provider
                )
            )
        suite, rig = _apply_selected_sensor_rig(suite, camera_framing)
        trajectory_bank, rir_plan = _build_canonical_rir_plan(
            suite, rig, stride_frames=int(request["acoustics"]["rir_stride_frames"])
        )
        selected_request = deepcopy(request)
        selected_request["camera_listener"] = deepcopy(
            suite["scenarios"][0]["plan"]["camera"]
        )
        projection = _project_mouth_proxies(selected_request)
    else:
        trajectory_bank, rir_plan = _build_rir_plan(request, rig)
        projection = _project_mouth_proxies(request)
    execution = _execution_plan(request, args.output, rig=rig)
    _require(not args.output.exists(), f"refusing to replace output: {args.output}")
    args.output.mkdir(parents=True)
    artifacts = {
        "room_adapter.json": room_adapter,
        "suite_execution_plan.json": suite,
        "sensor_rig_trajectory.json": rig,
        "trajectory_bank.json": trajectory_bank,
        "rir_job_plan.json": rir_plan,
        "execution_plan.json": execution,
    }
    if is_v2:
        artifacts.update(
            {
                "actor_framing.json": actor_framing,
                "camera_framing.json": camera_framing,
                "runtime_camera_gates.json": {
                    "schema": "avengine_mp3d_runtime_camera_gate_batch_v1",
                    "results": runtime_camera_gates,
                    "qualification_claim": False,
                    "formal_dataset_count": 0,
                },
            }
        )
    for name, value in artifacts.items():
        _write_json(args.output / name, value)
    adult_pair_ready = navigation["adult_static_pair_gate"]["status"] == "pass"
    preflight = {
        "schema": PREFLIGHT_SCHEMA,
        "status": (
            "pending_remaining_evidence"
            if is_v2 and adult_pair_ready
            else ("pass" if adult_pair_ready else "blocked")
        ),
        "cpu_planning_status": "pass" if adult_pair_ready else "blocked",
        "episode_ready": False,
        "capture_ready": False,
        "formal_ready": False,
        "request_id": request["request_id"],
        "episode_id": request["episode_id"],
        "inputs": {name: _file_record(path) for name, path in inputs.items()},
        "ue_import": {
            "status": "pass",
            "scene_id": import_manifest["scene_id"],
            "reload_verification": import_manifest["reload_verification"],
            "declared_static_mesh_count": 71,
            "fresh_cooked_load_readback_status": "planned_not_run",
            "coordinate_contract": {
                "source_axis_description": import_manifest["coordinate_contract"][
                    "source_axis_description"
                ],
                "canonical_axis_description": import_manifest["coordinate_contract"][
                    "canonical_axis_description"
                ],
                "source_to_canonical": import_manifest["coordinate_contract"][
                    "source_to_canonical"
                ],
                "canonical_habitat_to_ue": "U_cm=(100*H.x,100*H.z,100*H.y)",
            },
        },
        "navigation": navigation,
        "planned_projection": projection,
        "runtime_camera_framing": (
            {
                "status": camera_framing["status"],
                "selected_candidate_id": camera_framing["selected_candidate_id"],
                "actor_orientation_policy": actor_framing["actor_orientation_policy"],
                "native_pixel_validation_status": "pending",
                "qualification_claim": False,
            }
            if is_v2
            else {"status": "legacy_v1_not_runtime_gated"}
        ),
        "acoustic_registration": acoustic,
        "episode_contract": {
            "frame_count": FRAME_COUNT,
            "frame_rate_hz": FPS,
            "duration_seconds": 5.0,
            "static_distinct_human_pair": [
                suite["scenarios"][0]["plan"]["actors"][0]["asset_id"],
                suite["scenarios"][0]["plan"]["actors"][1]["asset_id"],
            ],
            "sparse_probe_frame_indices": [15],
            "normal_rgb_metric_depth_and_two_target_only_passes": True,
            "shared_bp_camera_sensor": True,
        },
        "rir": {
            "status": "planned_not_run",
            "unique_rir_job_count": rir_plan["unique_rir_job_count"],
            "requested_pair_state_count": rir_plan["requested_pair_state_count"],
            "source_positions_m": {
                job["uses"][0]["source_slot_id"]: job["source_position_m"]
                for job in rir_plan["jobs"]
            },
            "listener_position_m": rir_plan["jobs"][0]["listener_position_m"],
            "compute_device": "CPU",
        },
        "gpu_f15_request_materialized": True,
        "gpu_f15_request_ready": False,
        "gpu_f15_request_status": (
            "pending_live_male_female_bbox_and_mouth_review"
            if adult_pair_ready
            else "blocked_adult_nav_clearance_and_separation"
        ),
        "gpu_started": False,
        "formal_dataset_count": 0,
        "qualification_claim": False,
        "blockers": [
            *(
                []
                if adult_pair_ready
                else [
                    "current roots are only 0.9m apart; adult-pair gate requires 1.3m",
                    "source2 root clearance is 0.3566m; adult-pair gate requires 0.5m",
                    "fresh navmesh pair search must replace the current candidate",
                ]
            ),
            "fresh CPU acoustic compile has not run",
            "2 exact CPU RIR jobs have not run",
            "fresh packaged-SPEAR 71-mesh load/readback has not run",
            "f15 normal/target-only pixel review has not run",
            "live M/F bbox, mouth location, full-body clearance and visual review remain pending",
            "MP3D material semantics remain an unqualified research placeholder",
        ],
    }
    _write_json(args.output / "preflight.json", preflight)
    print(
        "STRICT_TWO_HUMAN_MP3D_PREFLIGHT_BUILT "
        f"status={preflight['status']} output={args.output} "
        "meshes=71 rirs=2 gpu_started=false",
        flush=True,
    )
    return args.output / "preflight.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--template-suite", type=Path)
    parser.add_argument("--ue-import-manifest", type=Path)
    parser.add_argument("--ue-runtime-evidence", type=Path)
    parser.add_argument("--fresh-navmesh-probe", type=Path)
    parser.add_argument("--acoustic-manifest", type=Path)
    parser.add_argument("--room-registry", type=Path)
    parser.add_argument("--acoustic-profiles", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runtime-probe-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.runtime_probe_output is not None:
        _require(
            args.request is None and args.output is None, "runtime probe is isolated"
        )
        probe_rir_runtime(args.runtime_probe_output.resolve())
        return 0
    _require(args.request is not None, "--request is required for preflight build")
    _require(args.output is not None, "--output is required for preflight build")
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
