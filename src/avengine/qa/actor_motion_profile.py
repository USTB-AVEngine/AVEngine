"""CPU-only, source-bound actor motion profiles.

The profile is intentionally data driven: it binds a proposed candidate, the
selected row that preceded it, and the materialized base suite without knowing
anything about a particular room or mechanism name.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from avengine.camera_pose import yaw_rotation_xyzw
from avengine.contracts.json_io import canonical_json_sha256, load_json
from avengine.routes.room_feasibility import (
    TrajectoryBank,
    TrajectoryEpisode,
    build_rir_job_plan,
)
from avengine.optional_backends.spear_visual import actor_ue_yaw_degrees

PROFILE_SCHEMA = "avengine_actor_motion_profile_v1"
PLANNING_PROFILE_SCHEMA = "avengine_actor_motion_profile_v2"
FRAME_SCHEMA = "avengine_actor_motion_profile_frame_v1"

_HUMAN_UE_IMPORT_IDENTITIES: dict[tuple[str, str], dict[str, str]] = {
    (
        "rocketbox_human_male_adult_01_m5_1_candidate",
        "native_runtime_ue_v3",
    ): {
        "schema": "rocketbox_native_ue_import_v3",
        "tag": "rocketbox_male_adult_01_original_ue_v3",
        "import_asset_id": "rocketbox_male_adult_01",
    },
    (
        "lead_b_rocketbox_adults_female_adult_01_original_v1",
        "native_runtime_ue_v1",
    ): {
        "schema": "rocketbox_batch_native_ue_import_v1",
        "tag": "rocketbox_adults_female_adult_01_original_ue_v1",
        "import_asset_id": "rocketbox_female_adult_01",
        "base_avatar_id": "rocketbox_adults_female_adult_01",
    },
    (
        "lead_b_rocketbox_professions_construction_male_01_original_v1",
        "native_runtime_ue_v1",
    ): {
        "schema": "rocketbox_batch_native_ue_import_v1",
        "tag": "rocketbox_professions_construction_male_01_original_ue_v1",
        "import_asset_id": "rocketbox_construction_male_01",
        "base_avatar_id": "rocketbox_professions_construction_male_01",
    },
}


class ActorMotionProfileError(ValueError):
    """A profile or one of its immutable authorities is inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ActorMotionProfileError(message)


def _validated_ue_import_manifest_ref(
    *, record: Mapping[str, Any], spear: Mapping[str, Any], actor_scale: float
) -> dict[str, Any]:
    """Close human runtime declarations against decoded UE import semantics.

    File digests and byte sizes in legacy import receipts are intentionally not
    read.  The selected regular path, declared identity, UE object paths, unit
    armature transform, and runtime scale are the authority used here.
    """

    identity = _HUMAN_UE_IMPORT_IDENTITIES.get(
        (str(record.get("asset_id")), str(record.get("revision")))
    )
    _require(identity is not None, "runtime asset lacks audited UE import identity")
    raw_ref = spear.get("ue_import_manifest_ref")
    _require(isinstance(raw_ref, Mapping), "runtime asset lacks UE import manifest ref")
    expected_fields = {"path", *identity}
    _require(
        set(raw_ref) == expected_fields
        and all(raw_ref.get(key) == value for key, value in identity.items()),
        "runtime UE import manifest ref identity drift",
    )
    raw_path = raw_ref.get("path")
    _require(
        isinstance(raw_path, str) and raw_path and Path(raw_path).is_absolute(),
        "runtime UE import manifest ref path is invalid",
    )
    path = Path(raw_path)
    _require(
        not path.is_symlink() and path.is_file(),
        f"runtime UE import manifest is missing or not regular: {path}",
    )
    manifest = load_json(path)
    _require(isinstance(manifest, Mapping), "runtime UE import manifest is invalid")
    runtime = manifest.get("runtime_contract")
    bounds = runtime.get("bounds") if isinstance(runtime, Mapping) else None
    glb = manifest.get("glb_contract")
    content = manifest.get("content")
    animations = content.get("animations") if isinstance(content, Mapping) else None
    source_glb = manifest.get("source_glb")
    source_path = Path(source_glb) if isinstance(source_glb, str) else Path()
    expected_runtime_root = (
        "rocketbox_native_runtime_ue_v3"
        if identity["schema"] == "rocketbox_native_ue_import_v3"
        else "rocketbox_batch_native_runtime_ue_v1"
    )
    base_avatar_ok = (
        "base_avatar_id" not in identity
        or manifest.get("base_avatar_id") == identity["base_avatar_id"]
    )
    _require(
        manifest.get("schema") == identity["schema"]
        and manifest.get("tag") == identity["tag"]
        and manifest.get("asset_id") == identity["import_asset_id"]
        and base_avatar_ok
        and manifest.get("usage_scope") == "research_candidate"
        and manifest.get("formal_registration_authorized") is False
        and isinstance(manifest.get("reload_verification"), Mapping)
        and manifest["reload_verification"].get("status") == "passed"
        and source_path.name == "runtime.glb"
        and source_path.parent.name == identity["tag"]
        and source_path.parent.parent.name == expected_runtime_root
        and isinstance(runtime, Mapping)
        and runtime.get("actor_scale") == actor_scale
        and runtime.get("bone_count") == 80
        and isinstance(bounds, Mapping)
        and bounds.get("height_passed") is True
        and bounds.get("ground_passed") is True
        and isinstance(glb, Mapping)
        and glb.get("armature_scale") == [1.0, 1.0, 1.0]
        and glb.get("armature_translation") == [0.0, 0.0, 0.0]
        and glb.get("animation_names") == ["Standing_Idle", "Walking"]
        and glb.get("joint_count") == 80
        and glb.get("skin_count") == 1
        and glb.get("mesh_count") == 1
        and glb.get("mesh_is_scene_root") is True,
        "runtime UE import manifest semantic contract drift",
    )
    _require(
        isinstance(content, Mapping)
        and isinstance(animations, Mapping)
        and set(animations) == {"Standing_Idle", "Walking"}
        and animations.get("Standing_Idle") == spear.get("idle_animation")
        and animations.get("Walking") == spear.get("walking_animation"),
        "runtime UE import animation binding drift",
    )
    blueprint = content.get("blueprint")
    _require(isinstance(blueprint, str) and blueprint, "runtime UE blueprint missing")
    blueprint_leaf = blueprint.rsplit("/", 1)[-1]
    mesh_directory = str(spear["idle_animation"]).rsplit("/", 1)[0]
    _require(
        spear.get("blueprint_class_path") == f"{blueprint}.{blueprint_leaf}_C"
        and content.get("skeletal_mesh") == f"{mesh_directory}/runtime.runtime"
        and content.get("skeleton")
        == f"{mesh_directory}/runtime_Skeleton.runtime_Skeleton",
        "runtime UE object binding drift",
    )
    return deepcopy(dict(raw_ref))


def bind_planning_episode(
    *,
    planning_manifest_path: str | Path,
    episode_id: str,
) -> dict[str, Any]:
    """Bind one planning row by absolute regular path and unique selector."""

    path = Path(planning_manifest_path).resolve()
    _require(path.is_file(), f"planning manifest is not a file: {path}")
    document = load_json(path)
    _require(isinstance(document, Mapping), "planning manifest is not an object")
    episodes = document.get("episodes")
    _require(isinstance(episodes, list), "planning manifest episodes are missing")
    matches = [
        (index, value)
        for index, value in enumerate(episodes)
        if isinstance(value, Mapping) and value.get("episode_id") == episode_id
    ]
    _require(
        len(matches) == 1,
        f"planning episode selector must resolve exactly once: {episode_id!r}",
    )
    index, row = matches[0]
    return {
        "path": str(path),
        "json_pointer": f"/episodes/{index}",
        "value": deepcopy(dict(row)),
    }


def _planning_runtime_declaration(
    *, role_name: str, role: Mapping[str, Any]
) -> tuple[dict[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    authority = _as_mapping(
        role.get("motion_profile_authority"),
        f"planning {role_name} motion profile authority missing",
    )
    _require(
        authority.get("schema")
        == "avengine_global100_role_motion_profile_authority_v1",
        f"planning {role_name} motion profile authority schema drift",
    )
    source = _as_mapping(
        authority.get("source_path"),
        f"planning {role_name} source-path authority missing",
    )
    runtime = _as_mapping(
        authority.get("runtime"),
        f"planning {role_name} runtime declaration missing",
    )
    _require(
        runtime.get("schema") == "avengine_global100_runtime_motion_declaration_v1"
        and runtime.get("asset_id") == role.get("runtime_asset_id")
        and runtime.get("asset_revision") == role.get("runtime_revision"),
        f"planning {role_name} runtime identity drift",
    )
    _require(
        source.get("source_suite")
        and source.get("native_source_scenario_id")
        and source.get("source_actor_id") == role.get("source_actor_id")
        and source.get("frame_index_map") == role.get("frame_index_map"),
        f"planning {role_name} source provenance drift",
    )
    idle = runtime.get("idle_action_id")
    walk = runtime.get("walking_action_id")
    animations = _as_mapping(
        runtime.get("animation_paths_by_action_id"),
        f"planning {role_name} runtime animations missing",
    )
    period = runtime.get("walk_phase_period_frames")
    forward = runtime.get("local_anatomical_forward_axis")
    emitter = runtime.get("emitter_offset_m")
    actor_scale = runtime.get("actor_scale")
    component_delta = runtime.get("ue_component_frame_delta")
    _require(
        isinstance(idle, str)
        and idle
        and isinstance(walk, str)
        and walk
        and idle != walk
        and set(animations) == {idle, walk}
        and all(isinstance(value, str) and value for value in animations.values())
        and type(period) is int
        and period > 1,
        f"planning {role_name} lacks honest action/period authority",
    )
    _require(
        isinstance(forward, list)
        and len(forward) == 3
        and all(
            type(value) in {int, float} and math.isfinite(float(value))
            for value in forward
        )
        and math.sqrt(sum(float(value) ** 2 for value in forward)) > 1.0e-9
        and isinstance(emitter, list)
        and len(emitter) == 3
        and all(
            type(value) in {int, float} and math.isfinite(float(value))
            for value in emitter
        )
        and runtime.get("emitter_offset_space") == "final_scaled_asset_root"
        and type(actor_scale) in {int, float}
        and math.isfinite(float(actor_scale))
        and float(actor_scale) > 0.0
        and isinstance(runtime.get("blueprint_class_path"), str)
        and runtime.get("blueprint_class_path")
        and runtime.get("skeletal_mesh_binding")
        in {"blueprint_component", "explicit_path"}
        and (
            runtime.get("skeletal_mesh_binding") != "explicit_path"
            or isinstance(runtime.get("skeletal_mesh_path"), str)
            and runtime.get("skeletal_mesh_path")
        )
        and type(runtime.get("ue_anatomical_forward_yaw_deg")) in {int, float}
        and math.isfinite(float(runtime["ue_anatomical_forward_yaw_deg"]))
        and isinstance(component_delta, Mapping)
        and set(component_delta)
        == {"schema", "composition", "reason", "rotation_deg", "translation_cm"}
        and component_delta.get("schema") == "avengine_spear_component_frame_delta_v1"
        and component_delta.get("composition")
        == "add_relative_preserving_blueprint_transform"
        and isinstance(component_delta.get("reason"), str)
        and component_delta.get("reason")
        and all(
            isinstance(component_delta.get(field), list)
            and len(component_delta[field]) == 3
            and all(
                type(value) in {int, float} and math.isfinite(float(value))
                for value in component_delta[field]
            )
            for field in ("rotation_deg", "translation_cm")
        )
        and type(runtime.get("floor_contact_gate")) is bool
        and isinstance(runtime.get("source_mesh_uri"), str)
        and runtime.get("source_mesh_uri")
        and runtime.get("admission_state") in {"formal", "research"},
        f"planning {role_name} runtime/emitter declaration is incomplete",
    )
    declaration = {
        "actor_id": f"{role['source_slot_id']}_actor",
        "asset_id": runtime["asset_id"],
        "asset_revision": runtime["asset_revision"],
        "actor_scale": float(actor_scale),
        "animation_paths_by_action_id": deepcopy(dict(animations)),
        "blueprint_class_path": runtime.get("blueprint_class_path"),
        "body_plan_id": runtime.get("body_plan_id"),
        "emitter_anchor_id": runtime.get("emitter_anchor_id"),
        "emitter_offset_m": deepcopy(runtime.get("emitter_offset_m")),
        "emitter_offset_space": runtime.get("emitter_offset_space"),
        "floor_contact_gate": runtime.get("floor_contact_gate"),
        "habitat_local_anatomical_forward_axis": deepcopy(
            runtime.get("local_anatomical_forward_axis")
        ),
        "idle_animation": animations[idle],
        "skeletal_mesh_binding": runtime.get("skeletal_mesh_binding"),
        "skeletal_mesh_path": runtime.get("skeletal_mesh_path"),
        "template_id": runtime.get("template_id"),
        "ue_anatomical_forward_yaw_deg": runtime.get("ue_anatomical_forward_yaw_deg"),
        "ue_component_frame_delta": deepcopy(runtime.get("ue_component_frame_delta")),
        "walking_animation": animations[walk],
        "runtime_asset_expectation": {
            "schema": "avengine_generic_human_runtime_asset_expectation_v1",
            "source_slot_id": role["source_slot_id"],
            "asset_id": runtime["asset_id"],
            "asset_revision": runtime["asset_revision"],
            "source_mesh_uri": runtime.get("source_mesh_uri"),
            "admission_state": runtime.get("admission_state"),
        },
    }
    return declaration, source, runtime


def _runtime_projection_from_registry(
    *, registry_path: Path, registry: Mapping[str, Any], record: Mapping[str, Any]
) -> dict[str, Any]:
    timeline = _as_mapping(record.get("timeline"), "registry timeline missing")
    backends = _as_mapping(record.get("runtime_backends"), "registry backends missing")
    spear = _as_mapping(backends.get("spear_unreal"), "registry SPEAR binding missing")
    anchors = _as_list(record.get("emitter_anchors"), "registry emitters missing")
    default_anchor = record.get("default_emitter_anchor_id")
    anchor_matches = [
        anchor
        for anchor in anchors
        if isinstance(anchor, Mapping) and anchor.get("anchor_id") == default_anchor
    ]
    _require(len(anchor_matches) == 1, "registry default emitter must resolve once")
    anchor = anchor_matches[0]
    idle = timeline.get("idle_action_id")
    walk = timeline.get("walking_action_id")
    geometry = _as_mapping(record.get("geometry"), "registry geometry missing")
    identity = _as_mapping(record.get("identity"), "registry identity missing")
    forward = timeline.get("local_anatomical_forward_axis")
    emitter = anchor.get("offset_m")
    actor_scale = spear.get("actor_scale")
    component_delta = spear.get("ue_component_frame_delta")
    _require(
        record.get("entity_class") == "articulated_human"
        and identity.get("species_id") == "human",
        "registry runtime asset is not human",
    )
    _require(
        (str(record.get("asset_id")), str(record.get("revision")))
        in _HUMAN_UE_IMPORT_IDENTITIES,
        "runtime asset lacks audited UE import identity/scale authority",
    )
    _require(
        isinstance(forward, list)
        and len(forward) == 3
        and all(
            type(value) in {int, float} and math.isfinite(float(value))
            for value in forward
        )
        and math.sqrt(sum(float(value) ** 2 for value in forward)) > 1.0e-9
        and isinstance(emitter, list)
        and len(emitter) == 3
        and all(
            type(value) in {int, float} and math.isfinite(float(value))
            for value in emitter
        )
        and anchor.get("offset_space") == "final_scaled_asset_root"
        and type(actor_scale) in {int, float}
        and math.isfinite(float(actor_scale))
        and float(actor_scale) > 0.0
        and isinstance(spear.get("blueprint_class_path"), str)
        and spear.get("blueprint_class_path")
        and spear.get("skeletal_mesh_binding")
        in {"blueprint_component", "explicit_path"}
        and (
            spear.get("skeletal_mesh_binding") != "explicit_path"
            or isinstance(spear.get("skeletal_mesh_path"), str)
            and spear.get("skeletal_mesh_path")
        )
        and type(spear.get("ue_anatomical_forward_yaw_deg")) in {int, float}
        and math.isfinite(float(spear["ue_anatomical_forward_yaw_deg"]))
        and isinstance(component_delta, Mapping)
        and set(component_delta)
        == {"schema", "composition", "reason", "rotation_deg", "translation_cm"}
        and component_delta.get("schema") == "avengine_spear_component_frame_delta_v1"
        and component_delta.get("composition")
        == "add_relative_preserving_blueprint_transform"
        and isinstance(component_delta.get("reason"), str)
        and component_delta.get("reason")
        and all(
            isinstance(component_delta.get(field), list)
            and len(component_delta[field]) == 3
            and all(
                type(value) in {int, float} and math.isfinite(float(value))
                for value in component_delta[field]
            )
            for field in ("rotation_deg", "translation_cm")
        )
        and type(spear.get("floor_contact_gate")) is bool
        and isinstance(geometry.get("source_mesh_uri"), str)
        and geometry.get("source_mesh_uri")
        and record.get("admission_state") in {"formal", "research"},
        "registry runtime/emitter declaration is incomplete",
    )
    ue_import_manifest_ref = _validated_ue_import_manifest_ref(
        record=record, spear=spear, actor_scale=float(actor_scale)
    )
    return {
        "schema": "avengine_global100_runtime_motion_declaration_v1",
        "runtime_registry": str(registry_path.resolve()),
        "registry_id": registry.get("registry_id"),
        "registry_revision": registry.get("revision"),
        "asset_id": record.get("asset_id"),
        "asset_revision": record.get("revision"),
        "body_plan_id": timeline.get("body_plan_id"),
        "template_id": timeline.get("template_id"),
        "local_anatomical_forward_axis": deepcopy(
            timeline.get("local_anatomical_forward_axis")
        ),
        "idle_action_id": idle,
        "walking_action_id": walk,
        "walk_phase_period_frames": timeline.get("walk_phase_period_frames"),
        "actor_scale": float(actor_scale),
        "ue_import_manifest_ref": ue_import_manifest_ref,
        "animation_paths_by_action_id": {
            idle: spear.get("idle_animation"),
            walk: spear.get("walking_animation"),
        },
        "blueprint_class_path": spear.get("blueprint_class_path"),
        "skeletal_mesh_binding": spear.get("skeletal_mesh_binding"),
        "skeletal_mesh_path": spear.get("skeletal_mesh_path"),
        "ue_anatomical_forward_yaw_deg": spear.get("ue_anatomical_forward_yaw_deg"),
        "ue_component_frame_delta": deepcopy(spear.get("ue_component_frame_delta")),
        "floor_contact_gate": spear.get("floor_contact_gate"),
        "source_mesh_uri": geometry.get("source_mesh_uri"),
        "emitter_anchor_id": anchor.get("anchor_id"),
        "emitter_offset_m": deepcopy(anchor.get("offset_m")),
        "emitter_offset_space": anchor.get("offset_space"),
        "admission_state": record.get("admission_state"),
    }


def _load_planning_source_states(
    *,
    role_name: str,
    role: Mapping[str, Any],
    source: Mapping[str, Any],
    suite_cache: dict[Path, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Resolve the exact selected actor state behind each planned output frame."""

    suite_path = Path(str(source["source_suite"])).resolve()
    _require(suite_path.is_file(), f"planning {role_name} source suite missing")
    if suite_path not in suite_cache:
        loaded_suite = load_json(suite_path)
        _require(
            isinstance(loaded_suite, Mapping),
            f"planning {role_name} source suite invalid",
        )
        suite_cache[suite_path] = loaded_suite
    suite = suite_cache[suite_path]
    scenarios = suite.get("scenarios")
    scenario_id = source["native_source_scenario_id"]
    matches = (
        [
            scenario
            for scenario in scenarios
            if isinstance(scenarios, list)
            and isinstance(scenario, Mapping)
            and scenario.get("scenario_id") == scenario_id
        ]
        if isinstance(scenarios, list)
        else []
    )
    _require(
        len(matches) == 1,
        f"planning {role_name} source scenario must resolve exactly once",
    )
    frames = _as_list(
        _as_mapping(
            matches[0].get("plan"), "planning source scenario plan missing"
        ).get("frames"),
        f"planning {role_name} source frames missing",
    )
    frame_map = _as_list(
        source.get("frame_index_map"), f"planning {role_name} source frame map missing"
    )
    root_path = _as_list(
        role.get("root_path_m"), f"planning {role_name} root path missing"
    )
    actor_id = source["source_actor_id"]
    selected: list[Mapping[str, Any]] = []
    _require(
        len(frame_map) == len(root_path) == 75,
        f"planning {role_name} source/root path is not full75",
    )
    for output_index, native_index in enumerate(frame_map):
        _require(
            type(native_index) is int and 0 <= native_index < len(frames),
            f"planning {role_name} source frame index out of range",
        )
        frame = _as_mapping(
            frames[native_index], f"planning {role_name} source frame invalid"
        )
        actor_states = _as_list(
            frame.get("actor_states"),
            f"planning {role_name} source actor states missing",
        )
        actor_matches = [
            state
            for state in actor_states
            if isinstance(state, Mapping) and state.get("actor_id") == actor_id
        ]
        _require(
            len(actor_matches) == 1,
            f"planning {role_name} source actor must resolve exactly once",
        )
        state = actor_matches[0]
        state_root = state.get("translation_m")
        _require(
            isinstance(state_root, Sequence)
            and len(state_root) == 3
            and [round(float(value), 9) for value in state_root]
            == [round(float(value), 9) for value in root_path[output_index]],
            f"planning {role_name} root/source state drift",
        )
        selected.append(state)
    return selected


def _planning_forward_path(
    roots: Sequence[Sequence[float]], *, camera_position_m: Sequence[float]
) -> list[list[float]]:
    """Derive a full75 tangent path, carrying endpoint directions honestly."""

    _require(len(roots) == 75, "planning actor root path is not full75")
    moving = any(list(root) != list(roots[0]) for root in roots[1:])
    if not moving:
        result = []
        for root in roots:
            dx = float(camera_position_m[0]) - float(root[0])
            dz = float(camera_position_m[2]) - float(root[2])
            norm = float(np.hypot(dx, dz))
            _require(norm > 1.0e-9, "planning static actor overlaps camera")
            result.append([dx / norm, 0.0, dz / norm])
        return result
    forwards: list[list[float]] = []
    last: list[float] | None = None
    for index, root in enumerate(roots):
        candidates = []
        if index + 1 < len(roots):
            candidates.append(roots[index + 1])
        if index > 0:
            candidates.append(roots[index - 1])
        vector: list[float] | None = None
        for neighbor in candidates:
            sign = 1.0 if neighbor is candidates[0] and index + 1 < len(roots) else -1.0
            dx = sign * (float(neighbor[0]) - float(root[0]))
            dz = sign * (float(neighbor[2]) - float(root[2]))
            norm = float(np.hypot(dx, dz))
            if norm > 1.0e-9:
                vector = [dx / norm, 0.0, dz / norm]
                break
        if vector is None:
            _require(last is not None, "planning moving root path has no tangent")
            vector = deepcopy(last)
        forwards.append(vector)
        last = vector
    return forwards


def _rotation_from_forward(forward: Sequence[float]) -> list[float]:
    x = float(forward[0])
    z = float(forward[2])
    norm = float(np.hypot(x, z))
    _require(norm > 1.0e-9, "planning actor forward vector is degenerate")
    yaw_degrees = float(np.degrees(np.arctan2(x / norm, z / norm)))
    return yaw_rotation_xyzw(yaw_degrees)


def build_actor_motion_profile_from_planning(
    *,
    planning_manifest_path: str | Path,
    episode_id: str,
    source_suite_cache: dict[Path, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Materialize full75 actor states directly from one provenance-rich plan row."""

    binding = bind_planning_episode(
        planning_manifest_path=planning_manifest_path, episode_id=episode_id
    )
    row = _as_mapping(binding["value"], "planning episode row invalid")
    _require(
        row.get("formal") is False and row.get("qualification_claim") is False,
        "planning row CPU/formal claim boundary drift",
    )
    timeline = _as_mapping(row.get("timeline"), "planning timeline missing")
    count = timeline.get("frame_count")
    rate = timeline.get("frame_rate_hz")
    _require(count == 75 and rate == 15, "planning timeline is not full75/15Hz")
    target = _as_mapping(row.get("target"), "planning target missing")
    audio_program = _as_mapping(
        row.get("audio_program"), "planning AudioProgram authority missing"
    )
    target_event = _as_mapping(
        audio_program.get("target_event"), "planning target audio event missing"
    )
    start_sample = target_event.get("start_sample")
    end_sample = target_event.get("end_sample_exclusive")
    _require(
        audio_program.get("mode") == "one_active_of_n"
        and audio_program.get("active_source_slots") == ["source1"]
        and audio_program.get("silent_source_slots") == ["source2"]
        and target_event.get("sound_asset_id") == target.get("sound_asset_id")
        and target_event.get("voice_id") == target.get("voice_id")
        and target_event.get("content_id") == target.get("content_id")
        and type(start_sample) is int
        and type(end_sample) is int
        and 0 <= start_sample < end_sample <= 80_000
        and end_sample - start_sample == target.get("speech_sample_count")
        and target_event.get("source_sample_rate_hz") == 16_000
        and target_event.get("source_sample_rate_hz")
        == target.get("speech_sample_rate_hz")
        and target_event.get("source_channel_count") == 1
        and target_event.get("source_channel_count")
        == target.get("speech_channel_count")
        and target_event.get("source_sample_count") == target.get("speech_sample_count")
        and isinstance(target_event.get("source_audio_uri"), str)
        and target_event.get("source_audio_uri") == target.get("speech_audio_uri"),
        "planning target audio event authority drift",
    )
    speech_window = _as_list(
        target.get("speech_frame_window_inclusive"),
        "planning speech window missing",
    )
    roles = {
        "source1": target,
        "source2": _as_mapping(row.get("distractor"), "planning distractor missing"),
    }
    registry_cache: dict[Path, Mapping[str, Any]] = {}
    suite_cache = source_suite_cache if source_suite_cache is not None else {}
    actor_declarations: dict[str, dict[str, Any]] = {}
    actors: dict[str, dict[str, Any]] = {}
    for slot, role in roles.items():
        _require(
            role.get("source_slot_id") == slot,
            f"planning {slot} role/slot drift",
        )
        role_name = "target" if slot == "source1" else "distractor"
        declaration, source, runtime = _planning_runtime_declaration(
            role_name=role_name, role=role
        )
        registry_path = Path(str(runtime["runtime_registry"])).resolve()
        if registry_path not in registry_cache:
            _require(
                registry_path.is_file(),
                f"planning {role_name} runtime registry missing",
            )
            loaded_registry = load_json(registry_path)
            _require(
                isinstance(loaded_registry, Mapping),
                f"planning {role_name} runtime registry invalid",
            )
            registry_cache[registry_path] = loaded_registry
        registry = registry_cache[registry_path]
        assets = registry.get("assets")
        matches = (
            [
                asset
                for asset in assets
                if isinstance(assets, list)
                and isinstance(asset, Mapping)
                and asset.get("asset_id") == runtime["asset_id"]
                and asset.get("revision") == runtime["asset_revision"]
            ]
            if isinstance(assets, list)
            else []
        )
        _require(
            len(matches) == 1,
            f"planning {role_name} runtime asset must resolve exactly once",
        )
        registry_record = matches[0]
        expected_runtime = _runtime_projection_from_registry(
            registry_path=registry_path,
            registry=registry,
            record=registry_record,
        )
        _require(
            dict(runtime) == expected_runtime,
            f"planning {role_name} copied runtime declaration drift",
        )
        actor_id = declaration["actor_id"]
        actor_declarations[actor_id] = declaration
        roots = _as_list(role.get("root_path_m"), f"planning {role_name} roots missing")
        frame_map = _as_list(
            role.get("frame_index_map"), f"planning {role_name} frame map missing"
        )
        _require(
            len(roots) == count
            and len(frame_map) == count
            and all(type(index) is int and index >= 0 for index in frame_map),
            f"planning {role_name} root/frame-map closure drift",
        )
        moving = any(root != roots[0] for root in roots[1:])
        expected_moving = row.get("mechanism") in (
            {"target_moves", "both_move"}
            if role_name == "target"
            else {"distractor_moves", "both_move"}
        )
        _require(
            moving is expected_moving,
            f"planning {role_name} mechanism/moving drift",
        )
        _require(
            frame_map == (list(range(count)) if moving else [frame_map[0]] * count),
            f"planning {role_name} frame map is not native stride-one/hold",
        )
        _load_planning_source_states(
            role_name=role_name,
            role=role,
            source=source,
            suite_cache=suite_cache,
        )
        idle = str(runtime["idle_action_id"])
        walk = str(runtime["walking_action_id"])
        period = int(runtime["walk_phase_period_frames"])
        # The retained animal suite authorizes only the selected root positions.
        # Human action semantics come from the planning mechanism plus the
        # runtime-human registry, never from an animal action label.
        action_ids = [walk if moving else idle for _ in range(count)]
        action_ticks: list[int] = []
        action_phases: list[float] = []
        active_index = 0
        for action in action_ids:
            if action == walk:
                action_ticks.append(active_index * 3200)
                action_phases.append((active_index % period) / period)
                active_index += 1
            else:
                action_ticks.append(0)
                action_phases.append(0.0)
                active_index = 0
        walk_segments: list[dict[str, Any]] = []
        segment_start: int | None = None
        for index, action in enumerate(action_ids + [idle]):
            if action == walk and segment_start is None:
                segment_start = index
            elif action != walk and segment_start is not None:
                segment_end = index - 1
                walk_segments.append(
                    {
                        "output_frame_range_inclusive": [
                            segment_start,
                            segment_end,
                        ],
                        "native_source_frame_range_inclusive": [
                            int(frame_map[segment_start]),
                            int(frame_map[segment_end]),
                        ],
                        "walk_phase_period_frames": period,
                        "output_frame_rate_hz": rate,
                        "native_frame_rate_hz": rate,
                        "time_scale": 1,
                        "global_time_stretch_applied": False,
                        "speech_window_inclusive": deepcopy(speech_window),
                        "speech_overlap_frame_count": len(
                            set(range(segment_start, segment_end + 1))
                            & set(
                                range(
                                    int(speech_window[0]),
                                    int(speech_window[1]) + 1,
                                )
                            )
                        ),
                    }
                )
                segment_start = None
        camera = _as_mapping(row.get("camera"), "planning camera missing")
        camera_position = _as_list(
            camera.get("translation_m"), "planning camera position missing"
        )
        _require(len(camera_position) == 3, "planning camera position shape drift")
        # Moving actors face the selected root tangent; static actors face the
        # camera.  Native animal anatomical-forward fields are non-authoritative.
        forward_path = _planning_forward_path(roots, camera_position_m=camera_position)
        local_forward = _as_list(
            runtime.get("local_anatomical_forward_axis"),
            f"planning {role_name} local forward axis missing",
        )
        ue_forward_yaw = runtime.get("ue_anatomical_forward_yaw_deg")
        _require(
            len(local_forward) == 3 and type(ue_forward_yaw) in {int, float},
            f"planning {role_name} runtime forward/yaw authority missing",
        )
        actor_yaws = [
            actor_ue_yaw_degrees(
                _rotation_from_forward(forward),
                local_forward,
                float(ue_forward_yaw),
            )
            for forward in forward_path
        ]
        actors[slot] = {
            "slot_id": slot,
            "actor_id": actor_id,
            "asset_id": role["runtime_asset_id"],
            "moving": moving,
            "root_path_m": deepcopy(roots),
            "translation_ue_cm_path": [
                [100.0 * float(root[0]), 100.0 * float(root[2]), 100.0 * float(root[1])]
                for root in roots
            ],
            "action_id_path": action_ids,
            "ue_animation_path": [
                runtime["animation_paths_by_action_id"][action] for action in action_ids
            ],
            "action_phase_path": action_phases,
            "action_time_ticks_path": action_ticks,
            "animation_timing_mode_path": [
                "runtime_asset_native_period_no_stretch_v1"
                if action == walk
                else "held_idle_v1"
                for action in action_ids
            ],
            "native_source_frame_index_path": deepcopy(frame_map),
            "actor_yaw_ue_deg_path": actor_yaws,
            "anatomical_forward_habitat_world_path": forward_path,
            "planning_source_path_authority": deepcopy(dict(source)),
            "runtime_motion_authority": deepcopy(dict(runtime)),
            "native_rate_active_interval": (
                deepcopy(walk_segments[0]) if len(walk_segments) == 1 else None
            ),
            "native_rate_action_segments": walk_segments,
            "trajectory_preflight": {
                "status": "pass_planning_roots_runtime_period_bound",
                "animation_ticks_per_phase_cycle": period * 3200,
                "walk_phase_period_frames": period,
                "global_time_stretch_applied": False,
                "source_suite": source["source_suite"],
                "native_source_scenario_id": source["native_source_scenario_id"],
                "source_actor_id": source["source_actor_id"],
                "claim_boundary": (
                    "runtime animation uses the declared per-asset native clip "
                    "period without global time stretch; the selected source root "
                    "trajectory does not prove human motion speed, human stride, "
                    "foot contact, absence of foot sliding, or visual acceptance; "
                    "all remain pending native capture and review"
                ),
            },
        }
    target_slot = str(target["source_slot_id"])
    distractor_slot = str(roles["source2"]["source_slot_id"])
    frames = []
    for index in range(count):
        states = []
        for slot, actor in actors.items():
            states.append(
                {
                    "actor_id": actor["actor_id"],
                    "slot_id": slot,
                    "translation_m": actor["root_path_m"][index],
                    "translation_ue_cm": actor["translation_ue_cm_path"][index],
                    "action_id": actor["action_id_path"][index],
                    "ue_animation": actor["ue_animation_path"][index],
                    "action_phase": actor["action_phase_path"][index],
                    "action_time_ticks": actor["action_time_ticks_path"][index],
                    "animation_timing_mode": actor["animation_timing_mode_path"][index],
                    "native_source_frame_index": actor[
                        "native_source_frame_index_path"
                    ][index],
                    "actor_yaw_ue_deg": actor["actor_yaw_ue_deg_path"][index],
                }
            )
        frames.append(
            {
                "frame_index": index,
                "pts_ticks": index * 3200,
                "frame_coverage_end_ticks": (index + 1) * 3200,
                "actor_states": states,
            }
        )
    candidate = {
        "schema": "avengine_planning_row_actor_motion_candidate_v1",
        "candidate_episode_id": episode_id,
        "legacy_episode_id": episode_id,
        "mechanism": row["mechanism"],
        "target_side": target.get("side"),
        "target_slot": target_slot,
        "distractor_slot": distractor_slot,
        "frame_count": count,
        "frame_rate_hz": rate,
        "timeline_ticks_per_second": 48_000,
        "frame_ticks": 3200,
        "qualification_claim": False,
        "formal_episode_count": 0,
        "gpu_launch_authorized": False,
        "camera": deepcopy(row.get("camera")),
        "audio_program": deepcopy(audio_program),
        "actor_declarations": actor_declarations,
        "actors": actors,
        "frames": frames,
        "mechanism_preflight": {
            "expected_moving_slots": [
                slot for slot, actor in actors.items() if actor["moving"]
            ],
            "observed_moving_slots": [
                slot for slot, actor in actors.items() if actor["moving"]
            ],
            "mechanism_speech_overlap_frame_count": len(
                set(range(int(speech_window[0]), int(speech_window[1]) + 1))
            ),
        },
    }
    core: dict[str, Any] = {
        "schema": PLANNING_PROFILE_SCHEMA,
        "status": "pass_cpu_planning_row_actor_motion_profile",
        "qualification_claim": False,
        "formal_episode_count": 0,
        "authorities": {
            "planning_episode": deepcopy(binding),
            "candidate": {"json_pointer": "", "value": candidate},
        },
    }
    core["frames"] = materialize_profile_frames(core)
    core["rir_expectation"] = _rir_expectation(core)
    return {**core, "profile_content_sha256": canonical_json_sha256(core)}


def _source_binding(
    path: str | Path,
    value: Mapping[str, Any],
    *,
    json_pointer: str,
) -> dict[str, Any]:
    source_path = Path(path).resolve()
    _require(source_path.is_file(), f"authority is not a file: {source_path}")
    return {
        "path": str(source_path),
        "json_pointer": json_pointer,
        "canonical_value_sha256": canonical_json_sha256(value),
        "value": deepcopy(dict(value)),
    }


def _candidate(profile: Mapping[str, Any]) -> Mapping[str, Any]:
    authorities = profile.get("authorities")
    _require(isinstance(authorities, Mapping), "authorities are missing")
    binding = authorities.get("candidate")
    _require(isinstance(binding, Mapping), "candidate authority is missing")
    value = binding.get("value")
    _require(isinstance(value, Mapping), "candidate value is missing")
    return value


def is_planning_actor_motion_profile(profile: Mapping[str, Any]) -> bool:
    """Return whether ``profile`` is the direct planning-row profile variant."""

    return profile.get("schema") == PLANNING_PROFILE_SCHEMA


def _as_mapping(value: object, message: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), message)
    return value


def _as_list(value: object, message: str) -> list[Any]:
    _require(isinstance(value, list), message)
    return value


def _actor_paths(
    actor: Mapping[str, Any], slot: str, count: int
) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = {}
    for key in (
        "root_path_m",
        "translation_ue_cm_path",
        "action_id_path",
        "ue_animation_path",
        "action_phase_path",
        "action_time_ticks_path",
        "animation_timing_mode_path",
        "native_source_frame_index_path",
        "actor_yaw_ue_deg_path",
    ):
        values = _as_list(actor.get(key), f"{key} missing for {slot!r}")
        _require(len(values) == count, f"{key} length drift for {slot!r}")
        result[key] = values
    return result


def _validate_actor_semantics(
    actor: Mapping[str, Any],
    declaration: Mapping[str, Any],
    *,
    slot: str,
    count: int,
    rate: int,
    frame_ticks: int,
    speech_window: list[int],
) -> None:
    _require(
        actor.get("slot_id") == slot
        and actor.get("actor_id") == declaration.get("actor_id")
        and actor.get("asset_id") == declaration.get("asset_id"),
        f"actor declaration drift for {slot!r}",
    )
    runtime = _as_mapping(
        declaration.get("runtime_asset_expectation"),
        f"runtime declaration missing for {slot!r}",
    )
    _require(
        runtime.get("source_slot_id") == slot
        and runtime.get("asset_id") == declaration.get("asset_id")
        and runtime.get("asset_revision") == declaration.get("asset_revision"),
        f"runtime declaration drift for {slot!r}",
    )
    animations = _as_mapping(
        declaration.get("animation_paths_by_action_id"),
        f"animation declaration missing for {slot!r}",
    )
    paths = _actor_paths(actor, slot, count)
    for action, animation in zip(
        paths["action_id_path"], paths["ue_animation_path"], strict=True
    ):
        _require(
            action in animations and animation == animations[action],
            f"action/animation declaration drift for {slot!r}",
        )
    moving = actor.get("moving")
    _require(type(moving) is bool, f"moving flag missing for {slot!r}")
    if not moving:
        _require(
            actor.get("native_rate_active_interval") is None
            and all(root == paths["root_path_m"][0] for root in paths["root_path_m"])
            and set(paths["action_id_path"]) == {"idle"}
            and set(paths["action_phase_path"]) == {0}
            and set(paths["action_time_ticks_path"]) == {0}
            and set(paths["native_source_frame_index_path"]) == {None}
            and len(set(paths["animation_timing_mode_path"])) == 1,
            f"static actor is not held Idle for {slot!r}",
        )
        return

    interval = _as_mapping(
        actor.get("native_rate_active_interval"),
        f"active interval missing for {slot!r}",
    )
    output_range = _as_list(
        interval.get("output_frame_range_inclusive"),
        f"output range missing for {slot!r}",
    )
    native_range = _as_list(
        interval.get("native_source_frame_range_inclusive"),
        f"native range missing for {slot!r}",
    )
    _require(
        len(output_range) == 2
        and len(native_range) == 2
        and all(type(value) is int for value in output_range + native_range),
        f"active range shape drift for {slot!r}",
    )
    start, end = output_range
    native_start, native_end = native_range
    intervals = end - start
    outside_action = interval.get("outside_action_id")
    _require(
        0 <= start < end < count
        and native_start >= 0
        and native_end - native_start == intervals
        and interval.get("output_interval_count") == intervals
        and interval.get("native_interval_count") == intervals
        and interval.get("output_sample_count") == intervals + 1
        and interval.get("native_sample_count") == intervals + 1
        and interval.get("output_frame_rate_hz") == rate
        and interval.get("native_frame_rate_hz") == rate
        and interval.get("time_scale") == 1
        and interval.get("global_time_stretch_applied") is False
        and interval.get("outside_root_policy") == "hold_nearest_boundary_root",
        f"native-rate active interval drift for {slot!r}",
    )
    actions = paths["action_id_path"]
    active_action = actions[start]
    _require(
        active_action != outside_action
        and actions[:start] == [outside_action] * start
        and actions[start : end + 1] == [active_action] * (intervals + 1)
        and actions[end + 1 :] == [outside_action] * (count - end - 1),
        f"active/outside action drift for {slot!r}",
    )
    roots = paths["root_path_m"]
    _require(
        all(root == roots[start] for root in roots[:start])
        and all(root == roots[end] for root in roots[end + 1 :]),
        f"outside roots do not hold active boundaries for {slot!r}",
    )
    native_indices = paths["native_source_frame_index_path"]
    _require(
        native_indices[:start] == [native_start] * start
        and native_indices[start : end + 1] == list(range(native_start, native_end + 1))
        and native_indices[end + 1 :] == [native_end] * (count - end - 1),
        f"native source frame mapping drift for {slot!r}",
    )
    ticks = paths["action_time_ticks_path"]
    active_ticks = ticks[start : end + 1]
    _require(
        all(type(value) is int for value in active_ticks)
        and all(
            current - previous == frame_ticks
            for previous, current in pairwise(active_ticks)
        )
        and set(ticks[:start] + ticks[end + 1 :]) <= {0},
        f"native-rate animation tick drift for {slot!r}",
    )
    trajectory = _as_mapping(
        actor.get("trajectory_preflight"),
        f"trajectory preflight missing for {slot!r}",
    )
    cycle_ticks = trajectory.get("animation_ticks_per_phase_cycle")
    _require(
        type(cycle_ticks) is int
        and cycle_ticks > 0
        and all(
            np.isclose(
                paths["action_phase_path"][index],
                (ticks[index] / cycle_ticks) % 1.0,
            )
            for index in range(start, end + 1)
        )
        and set(
            paths["action_phase_path"][:start] + paths["action_phase_path"][end + 1 :]
        )
        <= {0},
        f"animation phase/tick drift for {slot!r}",
    )
    overlap = len(
        set(range(start, end + 1)) & set(range(speech_window[0], speech_window[1] + 1))
    )
    _require(
        interval.get("speech_window_inclusive") == speech_window
        and interval.get("speech_overlap_frame_count") == overlap
        and overlap > 0,
        f"active interval lost speech overlap for {slot!r}",
    )


def validate_actor_motion_authorities(
    candidate: Mapping[str, Any],
    selected_old_row: Mapping[str, Any],
    base_suite: Mapping[str, Any],
) -> None:
    """Fail closed on motion semantics and three-way authority drift."""

    _require(
        candidate.get("qualification_claim") is False
        and candidate.get("formal_episode_count") == 0
        and candidate.get("gpu_launch_authorized") is False,
        "candidate CPU/formal claim boundary drift",
    )
    legacy_id = candidate.get("legacy_episode_id")
    mechanism = candidate.get("mechanism")
    _require(
        selected_old_row.get("episode_id") == legacy_id
        and selected_old_row.get("mechanism") == mechanism
        and selected_old_row.get("target_side") == candidate.get("target_side"),
        "candidate/old-row identity drift",
    )
    scenarios = _as_list(base_suite.get("scenarios"), "base-suite scenarios missing")
    _require(len(scenarios) == 1, "base suite must contain exactly one scenario")
    scenario = _as_mapping(scenarios[0], "base-suite scenario is invalid")
    plan = _as_mapping(scenario.get("plan"), "base-suite plan missing")
    _require(
        scenario.get("scenario_id") == legacy_id
        and scenario.get("variant_id") == mechanism,
        "candidate/base-suite identity drift",
    )
    actors = _as_mapping(candidate.get("actors"), "candidate actors missing")
    declarations = _as_mapping(
        candidate.get("actor_declarations"), "candidate declarations missing"
    )
    plan_actors = _as_list(plan.get("actors"), "base-suite actors missing")
    plan_declarations = {
        declaration["actor_id"]: declaration
        for declaration in (
            _as_mapping(value, "base-suite actor declaration invalid")
            for value in plan_actors
        )
    }
    _require(
        len(plan_declarations) == len(plan_actors)
        and declarations == plan_declarations,
        "candidate/base-suite actor declaration drift",
    )
    for role in ("target", "distractor"):
        slot = candidate.get(f"{role}_slot")
        old_role = _as_mapping(
            selected_old_row.get(role), f"old {role} authority missing"
        )
        _require(
            isinstance(slot, str)
            and slot in actors
            and old_role.get("source_slot_id") == slot,
            f"{role} slot cross-authority drift",
        )
        actor = _as_mapping(actors[slot], f"candidate {role} actor invalid")
        declaration = _as_mapping(
            declarations.get(actor.get("actor_id")),
            f"candidate {role} declaration missing",
        )
        _require(
            actor.get("asset_id") == old_role.get("runtime_asset_id")
            and declaration.get("asset_revision") == old_role.get("runtime_revision"),
            f"{role} asset/revision cross-authority drift",
        )

    camera = _as_mapping(candidate.get("camera"), "candidate camera missing")
    old_camera = _as_mapping(selected_old_row.get("camera"), "old camera missing")
    old_yaws = _as_list(old_camera.get("yaw_path_deg"), "old camera yaw path missing")
    _require(
        plan.get("camera") == camera
        and old_camera.get("translation_m") == camera.get("habitat_position_m")
        and old_camera.get("horizontal_fov_deg") == camera.get("horizontal_fov_deg")
        and old_yaws
        and all(yaw == camera.get("habitat_yaw_deg") for yaw in old_yaws),
        "camera cross-authority drift",
    )
    activation = _as_mapping(
        candidate.get("source_activation_contract"),
        "source activation contract missing",
    )
    source_logic = _as_mapping(
        activation.get("source_logic"), "candidate source logic missing"
    )
    _require(
        activation.get("modified") is False
        and source_logic == plan.get("source_logic"),
        "source activation/base-suite drift",
    )
    source_rows = _as_list(source_logic.get("sources"), "source rows missing")
    target_slot = str(candidate.get("target_slot"))
    distractor_slot = str(candidate.get("distractor_slot"))
    expected_activation = {
        actors[target_slot]["actor_id"]: "active",
        actors[distractor_slot]["actor_id"]: "silent",
    }
    observed_activation = {
        row["entity_actor_id"]: row.get("activation")
        for row in (
            _as_mapping(value, "source activation row invalid") for value in source_rows
        )
    }
    _require(
        len(observed_activation) == len(source_rows)
        and observed_activation == expected_activation,
        "source activation actor/role drift",
    )

    old_target = _as_mapping(selected_old_row.get("target"), "old target missing")
    old_distractor = _as_mapping(
        selected_old_row.get("distractor"), "old distractor missing"
    )
    contract = _as_mapping(
        candidate.get("audio_event_contract"), "audio event contract missing"
    )
    audio = _as_mapping(contract.get("audio_program"), "audio program missing")
    validation = _as_mapping(audio.get("validation"), "audio validation missing")
    sample_rate = audio.get("sample_rate_hz")
    sample_count = audio.get("sample_count")
    start_sample = audio.get("target_speech_start_sample")
    active_samples = old_target.get("speech_sample_count")
    _require(
        all(
            type(value) is int
            for value in (sample_rate, sample_count, start_sample, active_samples)
        )
        and sample_rate > 0
        and sample_count > 0
        and active_samples > 0
        and 0 <= start_sample < start_sample + active_samples <= sample_count,
        "audio sample authority drift",
    )
    rate = candidate.get("frame_rate_hz")
    _require(type(rate) is int and rate > 0, "candidate frame rate drift")
    speech_window = [
        start_sample * rate // sample_rate,
        ((start_sample + active_samples) * rate + sample_rate - 1) // sample_rate - 1,
    ]
    _require(
        contract.get("sound_event_content_and_timing_modified") is False
        and contract.get("source_activation_modified") is False
        and contract.get("existing_exact_rir_reuse_authorized") is False
        and contract.get("fresh_exact_rir_required") is True
        and audio.get("target_source_slot") == target_slot
        and audio.get("distractor_source_slot") == distractor_slot
        and audio.get("target_event_count") == 1
        and audio.get("distractor_event_count") == 0
        and contract.get("target_speech_start_sample") == start_sample
        and contract.get("speech_frame_window_inclusive") == speech_window
        and validation.get("speech_frame_window_inclusive") == speech_window
        and old_target.get("speech_frame_window_inclusive") == speech_window
        and old_target.get("voice_policy") == "speaking"
        and old_distractor.get("voice_policy") == "silent",
        "audio role/timing cross-authority drift",
    )
    if "target_sound_asset_id" in audio:
        _require(
            audio.get("target_sound_asset_id") == old_target.get("sound_asset_id"),
            "audio sound asset cross-authority drift",
        )
    if "target_active_sample_count" in validation:
        _require(
            validation.get("target_active_sample_count") == active_samples,
            "audio active-sample cross-authority drift",
        )

    count = candidate.get("frame_count")
    tick_rate = candidate.get("timeline_ticks_per_second")
    _require(
        type(count) is int
        and count > 0
        and type(tick_rate) is int
        and tick_rate > 0
        and tick_rate % rate == 0,
        "candidate timeline authority drift",
    )
    frame_ticks = tick_rate // rate
    _require(
        candidate.get("frame_ticks") == frame_ticks and len(old_yaws) == count,
        "candidate frame-tick/camera closure drift",
    )
    frames = _as_list(candidate.get("frames"), "candidate frames missing")
    _require(len(frames) == count, "candidate frame count drift")
    actor_order = list(actors)
    for index, frame_value in enumerate(frames):
        frame = _as_mapping(frame_value, f"candidate frame {index} invalid")
        states = _as_list(
            frame.get("actor_states"), f"actor states missing at f{index}"
        )
        _require(
            frame.get("frame_index") == index
            and frame.get("pts_ticks") == index * frame_ticks
            and frame.get("frame_coverage_end_ticks") == (index + 1) * frame_ticks
            and [state.get("slot_id") for state in states] == actor_order,
            f"frame timeline/actor order drift at f{index}",
        )
        for state, slot in zip(states, actor_order, strict=True):
            actor = actors[slot]
            expected = {
                "actor_id": actor["actor_id"],
                "slot_id": slot,
                "translation_m": actor["root_path_m"][index],
                "translation_ue_cm": actor["translation_ue_cm_path"][index],
                "action_id": actor["action_id_path"][index],
                "ue_animation": actor["ue_animation_path"][index],
                "action_phase": actor["action_phase_path"][index],
                "action_time_ticks": actor["action_time_ticks_path"][index],
                "animation_timing_mode": actor["animation_timing_mode_path"][index],
                "native_source_frame_index": actor["native_source_frame_index_path"][
                    index
                ],
                "actor_yaw_ue_deg": actor["actor_yaw_ue_deg_path"][index],
            }
            _require(
                dict(state) == expected, f"frame/actor path drift at f{index} {slot}"
            )

    moving_slots: list[str] = []
    for slot, actor_value in actors.items():
        actor = _as_mapping(actor_value, f"candidate actor {slot!r} invalid")
        actor_id = actor.get("actor_id")
        declaration = _as_mapping(
            declarations.get(actor_id), f"candidate declaration {actor_id!r} missing"
        )
        _validate_actor_semantics(
            actor,
            declaration,
            slot=str(slot),
            count=count,
            rate=rate,
            frame_ticks=frame_ticks,
            speech_window=speech_window,
        )
        if actor.get("moving") is True:
            moving_slots.append(str(slot))
    mechanism_preflight = _as_mapping(
        candidate.get("mechanism_preflight"), "mechanism preflight missing"
    )
    _require(
        mechanism_preflight.get("expected_moving_slots") == moving_slots
        and mechanism_preflight.get("observed_moving_slots") == moving_slots
        and mechanism_preflight.get("mechanism_speech_overlap_frame_count", 0) > 0,
        "mechanism/actor motion semantics drift",
    )


def materialize_profile_frames(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return canonical, hash-bound frames from a validated profile."""

    candidate = _candidate(profile)
    frames = candidate.get("frames")
    _require(isinstance(frames, list) and bool(frames), "candidate frames are missing")
    result: list[dict[str, Any]] = []
    for frame_index, source in enumerate(frames):
        _require(isinstance(source, Mapping), f"frame {frame_index} is not an object")
        _require(
            source.get("frame_index") == frame_index, "frame indices are not exact"
        )
        core = {
            "schema": FRAME_SCHEMA,
            "frame_index": frame_index,
            "pts_ticks": source.get("pts_ticks"),
            "actor_states": deepcopy(source.get("actor_states")),
        }
        result.append({**core, "canonical_frame_sha256": canonical_json_sha256(core)})
    return result


def source_center_paths(profile: Mapping[str, Any]) -> dict[str, list[list[float]]]:
    """Derive emitter centers by adding declared offsets to candidate roots."""

    candidate = _candidate(profile)
    declarations = candidate.get("actor_declarations")
    actors = candidate.get("actors")
    _require(isinstance(declarations, Mapping), "actor declarations are missing")
    _require(isinstance(actors, Mapping) and bool(actors), "actors are missing")
    result: dict[str, list[list[float]]] = {}
    for slot, actor in sorted(actors.items()):
        _require(isinstance(actor, Mapping), f"actor {slot!r} is invalid")
        actor_id = actor.get("actor_id")
        declaration = declarations.get(actor_id)
        _require(
            isinstance(declaration, Mapping), f"declaration for {actor_id!r} is missing"
        )
        offset = declaration.get("emitter_offset_m")
        roots = actor.get("root_path_m")
        _require(
            isinstance(offset, Sequence)
            and len(offset) == 3
            and isinstance(roots, Sequence),
            f"source-center inputs for {slot!r} are invalid",
        )
        result[str(slot)] = [
            [float(root[axis]) + float(offset[axis]) for axis in range(3)]
            for root in roots
        ]
    return result


def _rir_expectation(profile: Mapping[str, Any]) -> dict[str, Any]:
    candidate = _candidate(profile)
    frame_count = int(candidate["frame_count"])
    episode_id = str(candidate["candidate_episode_id"])
    centers = source_center_paths(profile)
    roots = {
        str(slot): np.asarray(actor["root_path_m"], dtype=np.float64)
        for slot, actor in candidate["actors"].items()
    }
    episode = TrajectoryEpisode(
        episode_id=episode_id,
        motion_case=str(candidate["mechanism"]),
        source_root_paths_m=roots,
        source_center_paths_m={
            slot: np.asarray(path, dtype=np.float64) for slot, path in centers.items()
        },
        statistics={},
    )
    bank = TrajectoryBank(
        episodes=(episode,),
        frame_count=frame_count,
        frame_rate_hz=int(candidate["frame_rate_hz"]),
        seed=0,
    )
    authorities = profile["authorities"]
    if is_planning_actor_motion_profile(profile):
        row = authorities["planning_episode"]["value"]
        camera = row["camera"]
        positions = [deepcopy(camera["translation_m"]) for _ in range(frame_count)]
        orientations = []
        for yaw_deg in camera["yaw_path_deg"]:
            xyzw = yaw_rotation_xyzw(float(yaw_deg))
            orientations.append([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
    else:
        base_suite = authorities["base_suite"]["value"]
        scenario = base_suite["scenarios"][0]
        frames = scenario["plan"]["frames"]
        positions = [frame["camera_state"]["habitat_position_m"] for frame in frames]
        orientations = []
        for frame in frames:
            xyzw = frame["camera_state"]["world_from_rig"]["rotation_xyzw"]
            orientations.append([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
    plan = build_rir_job_plan(
        bank,
        listener_positions_m_by_episode={episode_id: positions},
        listener_orientations_wxyz_by_episode={episode_id: orientations},
        stride_frames=1,
    )
    return {
        "builder": "avengine.m6x.room_feasibility.build_rir_job_plan",
        "stride_frames": plan["stride_frames"],
        "requested_pair_state_count": plan["requested_pair_state_count"],
        "unique_rir_job_count": plan["unique_rir_job_count"],
        "canonical_plan_sha256": canonical_json_sha256(plan),
    }


def build_actor_motion_profile(
    *,
    candidate_path: str | Path,
    candidate: Mapping[str, Any],
    old_preflight_path: str | Path,
    selected_old_row: Mapping[str, Any],
    base_suite_path: str | Path,
    base_suite: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a generic immutable profile from three supplied authorities."""

    validate_actor_motion_authorities(candidate, selected_old_row, base_suite)

    core: dict[str, Any] = {
        "schema": PROFILE_SCHEMA,
        "status": "pass_cpu_bound_actor_motion_profile",
        "qualification_claim": False,
        "formal_episode_count": 0,
        "authorities": {
            "candidate": _source_binding(candidate_path, candidate, json_pointer=""),
            "selected_old_row": _source_binding(
                old_preflight_path, selected_old_row, json_pointer="/canaries/0"
            ),
            "base_suite": _source_binding(base_suite_path, base_suite, json_pointer=""),
        },
    }
    core["frames"] = materialize_profile_frames(core)
    core["rir_expectation"] = _rir_expectation(core)
    profile = {**core, "profile_content_sha256": canonical_json_sha256(core)}
    validate_actor_motion_profile(profile)
    return profile


def validate_actor_motion_profile(profile: Mapping[str, Any]) -> None:
    """Fail closed on source drift and basic frame/profile inconsistency."""

    schema = profile.get("schema")
    _require(
        schema in {PROFILE_SCHEMA, PLANNING_PROFILE_SCHEMA},
        "profile schema is invalid",
    )
    _require(
        profile.get("qualification_claim") is False, "qualification claim is forbidden"
    )
    _require(
        profile.get("formal_episode_count") == 0, "formal episode count must be zero"
    )
    core = dict(profile)
    declared_hash = core.pop("profile_content_sha256", None)
    _require(
        declared_hash == canonical_json_sha256(core), "profile content hash mismatch"
    )
    authorities = profile.get("authorities")
    _require(isinstance(authorities, Mapping), "authorities are missing")
    if schema == PLANNING_PROFILE_SCHEMA:
        planning = authorities.get("planning_episode")
        candidate_binding = authorities.get("candidate")
        _require(
            isinstance(planning, Mapping)
            and isinstance(candidate_binding, Mapping)
            and isinstance(candidate_binding.get("value"), Mapping),
            "planning profile authorities are missing",
        )
        path = Path(str(planning.get("path", ""))).resolve()
        _require(path.is_file(), "planning episode authority file is missing")
        document = load_json(path)
        episodes = document.get("episodes") if isinstance(document, Mapping) else None
        pointer = planning.get("json_pointer")
        _require(
            isinstance(episodes, list)
            and isinstance(pointer, str)
            and pointer.startswith("/episodes/"),
            "planning episode authority pointer is invalid",
        )
        try:
            index = int(pointer.removeprefix("/episodes/"))
        except ValueError as error:
            raise ActorMotionProfileError(
                "planning episode authority pointer is invalid"
            ) from error
        _require(
            0 <= index < len(episodes) and planning.get("value") == episodes[index],
            "planning episode bound value drift",
        )
        rebound = build_actor_motion_profile_from_planning(
            planning_manifest_path=path,
            episode_id=str(planning["value"]["episode_id"]),
        )
        _require(profile == rebound, "planning actor motion profile drift")
        return
    for name, pointer in (
        ("candidate", ""),
        ("selected_old_row", "/canaries/0"),
        ("base_suite", ""),
    ):
        binding = authorities.get(name)
        _require(isinstance(binding, Mapping), f"{name} authority is missing")
        path = Path(str(binding.get("path", "")))
        _require(path.is_file(), f"{name} authority file is missing")
        document = load_json(path)
        actual = document if pointer == "" else document["canaries"][0]
        _require(binding.get("json_pointer") == pointer, f"{name} JSON pointer drift")
        _require(binding.get("value") == actual, f"{name} bound value drift")
        _require(
            binding.get("canonical_value_sha256") == canonical_json_sha256(actual),
            f"{name} value hash drift",
        )
    validate_actor_motion_authorities(
        authorities["candidate"]["value"],
        authorities["selected_old_row"]["value"],
        authorities["base_suite"]["value"],
    )
    _require(
        profile.get("frames") == materialize_profile_frames(profile),
        "frame materialization drift",
    )
    _require(
        profile.get("rir_expectation") == _rir_expectation(profile),
        "RIR expectation drift",
    )
