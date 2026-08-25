"""Compile the legacy Rocketbox human into the pinned Habitat skin runtime.

The Rocketbox ``male adult 01`` GLB has a valid 80-joint skin, but its
animated pelvis is the first skin joint.  The M2 local-TR runtime deliberately
owns its skin root from the articulated-object transform and therefore cannot
discard that pelvis motion.  This module adds two *zero-weight* ancestors to
the skin joint list:

``AVEngine Human Root -> Bip01 -> Bip01 Pelvis``

The new root is static.  ``Bip01`` and the pelvis remain ordinary runtime
links, so the authored Walking rotation and vertical pelvis motion survive.
The source JOINTS_0 ordinals do not change because both ancestors are appended
to the skin rather than prepended.  A new inverse-bind accessor closes the
expanded skin before the existing local-TR rebase/bake code is used.

This is a bounded M5.1 research compiler, not a general humanoid importer and
not an asset-admission decision.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from avengine.contracts.json_io import canonical_json_sha256, sha256_file
from avengine.assets.glb import (
    GlbDocument,
    GlbError,
    extract_actions,
    extract_skins,
    load_glb,
    parse_glb,
)
from avengine.assets.glb_write import build_glb
from avengine.assets.habitat import build_habitat_ao_config_data
from avengine.assets.local_tr_actions import (
    LocalTRActionSet,
    TICKS_PER_SAMPLE,
    TIME_BASE_HZ,
    bake_local_tr_actions,
    read_local_tr_actions_npz,
    write_local_tr_actions_npz,
)
from avengine.assets.local_tr_habitat import (
    LocalTRHabitatMapping,
    build_local_tr_habitat_mapping,
)
from avengine.assets.rebase import (
    RebaseError,
    _global_matrix,
    _matrix_values,
    _parents,
    rebase_skin_root_preserving_local_tr,
)


SYNTHETIC_ROOT_NAME = "AVEngine Human Root"
ARMATURE_NODE_NAME = "Bip01"
PELVIS_NODE_NAME = "Bip01 Pelvis"
HEAD_LINK_NAME = "Bip01 Head"
MOUTH_LINK_NAME = "Bip01 MJaw"
DEFAULT_SEMANTIC_ID = 220
SOURCE_ACTION_NAMES = ("Walking", "Standing_Idle")
RUNTIME_ACTION_ALIASES = {"Standing_Idle": "Idle"}
HUMAN_ANATOMICAL_FORWARD_AXIS_LABEL = "+Z"
HUMAN_ANATOMICAL_FORWARD_SOURCE = (
    "rocketbox_male_adult_01_rest_pose_head_to_mjaw_horizontal_axis"
)
_PROFILE_RETIMED_WALKING_SAMPLE_COUNT = 19
_PROFILE_RETIMED_WALKING_SOURCE_KEY_COUNTS = (2, 38)
_PROFILE_RETIMED_WALKING_SOURCE_DURATION_TICKS = 59_200
_TICK_ROUNDING_TOLERANCE = 1.0e-2


class HumanRuntimeError(ValueError):
    """The Rocketbox source or derived runtime violates the bounded contract."""


@dataclass(frozen=True)
class HumanRuntimePackage:
    """Prepared files and in-memory state needed by mixed Habitat capture."""

    source_glb: Path
    promoted_glb: Path
    visual_glb: Path
    rebase_report: Path
    actions_npz: Path
    habitat_urdf: Path
    habitat_ao_config: Path
    habitat_joint_mapping: Path
    package_manifest: Path
    document: GlbDocument
    actions: LocalTRActionSet
    mapping: LocalTRHabitatMapping
    actor_from_skin_root: tuple[tuple[float, float, float, float], ...]


def _record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "byte_size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _append_float_accessor(
    document: dict[str, Any],
    binary: bytearray,
    values: np.ndarray,
    *,
    element_type: str,
) -> int:
    array = np.asarray(values, dtype=np.dtype("<f4"))
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise HumanRuntimeError("new GLB accessor must be a finite matrix")
    component_count = {"SCALAR": 1, "VEC4": 4, "MAT4": 16}.get(element_type)
    if component_count is None or array.shape[1] != component_count:
        raise HumanRuntimeError("new GLB accessor has an unsupported shape")
    while len(binary) % 4:
        binary.append(0)
    offset = len(binary)
    payload = np.ascontiguousarray(array).tobytes(order="C")
    binary.extend(payload)
    views = document.setdefault("bufferViews", [])
    accessors = document.setdefault("accessors", [])
    if not isinstance(views, list) or not isinstance(accessors, list):
        raise HumanRuntimeError("GLB bufferViews/accessors must be arrays")
    view_index = len(views)
    views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(payload)})
    accessor_index = len(accessors)
    accessor: dict[str, Any] = {
        "bufferView": view_index,
        "componentType": 5126,
        "count": int(array.shape[0]),
        "type": element_type,
    }
    if element_type == "SCALAR":
        accessor["min"] = [float(np.min(array))]
        accessor["max"] = [float(np.max(array))]
    accessors.append(accessor)
    return accessor_index


def _walking_profile_sample_count(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HumanRuntimeError(
            "walking_profile_sample_count must be a positive integer or None"
        )
    if value != _PROFILE_RETIMED_WALKING_SAMPLE_COUNT:
        raise HumanRuntimeError(
            "only the audited 19-sample Rocketbox Walking profile may retime"
        )
    return value


def _action_span_ticks(action: Any) -> tuple[float, float, int]:
    timestamps = tuple(
        timestamp
        for channel in action.channels
        for timestamp in channel.timestamps_seconds
    )
    if not timestamps:
        raise HumanRuntimeError(f"Rocketbox action {action.name!r} has no timestamps")
    start = float(min(timestamps))
    end = float(max(timestamps))
    exact_ticks = (end - start) * TIME_BASE_HZ
    rounded_ticks = int(round(exact_ticks))
    if abs(exact_ticks - rounded_ticks) > _TICK_ROUNDING_TOLERANCE:
        raise HumanRuntimeError(
            f"Rocketbox action {action.name!r} span is not on the 48 kHz clock"
        )
    return start, end, rounded_ticks


def _retime_walking_loop_to_profile(
    promoted_payload: bytes,
    *,
    walking_profile_sample_count: int | None,
) -> tuple[bytes, dict[str, Any] | None]:
    """Affine-retime only the audited female Walking timeline to 19 samples.

    The general Local-TR baker deliberately remains strict.  This bounded
    repair preserves every animation value and key order while changing only
    the Walking input time accessor from 59,200 to 60,800 ticks.
    """

    expected_count = _walking_profile_sample_count(walking_profile_sample_count)
    if expected_count is None:
        return promoted_payload, None
    try:
        document = parse_glb(promoted_payload)
        actions = extract_actions(document)
    except GlbError as exc:
        raise HumanRuntimeError(
            f"promoted Rocketbox actions are invalid: {exc}"
        ) from exc
    walking = next((action for action in actions if action.name == "Walking"), None)
    idle = next((action for action in actions if action.name == "Idle"), None)
    if walking is None or idle is None or len(actions) != 2:
        raise HumanRuntimeError("promoted Rocketbox must contain Walking and Idle")
    source_start, source_end, source_ticks = _action_span_ticks(walking)
    timelines_by_accessor: dict[int, tuple[float, ...]] = {}
    for channel in walking.channels:
        retained = timelines_by_accessor.setdefault(
            channel.input_accessor_index, channel.timestamps_seconds
        )
        if retained != channel.timestamps_seconds:
            raise HumanRuntimeError(
                "Walking channels disagree for one shared input accessor"
            )
    key_counts = {len(channel.timestamps_seconds) for channel in walking.channels}
    timeline_spans = {
        int(round((max(timeline) - min(timeline)) * TIME_BASE_HZ))
        for timeline in timelines_by_accessor.values()
    }
    timeline_bounds = {
        (float(min(timeline)), float(max(timeline)))
        for timeline in timelines_by_accessor.values()
    }
    if (
        source_ticks != _PROFILE_RETIMED_WALKING_SOURCE_DURATION_TICKS
        or key_counts != set(_PROFILE_RETIMED_WALKING_SOURCE_KEY_COUNTS)
        or timeline_spans != {_PROFILE_RETIMED_WALKING_SOURCE_DURATION_TICKS}
        or timeline_bounds != {(source_start, source_end)}
    ):
        raise HumanRuntimeError(
            "19-sample retime requires 2/38-key Walking timelines sharing a 59,200-tick span"
        )
    target_ticks = expected_count * TICKS_PER_SAMPLE
    scale = target_ticks / source_ticks

    root = document.json
    binary = bytearray(document.binary)
    animations = root.get("animations")
    buffers = root.get("buffers")
    if (
        not isinstance(animations, list)
        or not isinstance(buffers, list)
        or len(buffers) != 1
        or not isinstance(buffers[0], dict)
    ):
        raise HumanRuntimeError("promoted Rocketbox animation buffers are malformed")
    raw_walking = animations[walking.animation_index]
    if not isinstance(raw_walking, dict) or not isinstance(
        raw_walking.get("samplers"), list
    ):
        raise HumanRuntimeError("promoted Rocketbox Walking samplers are malformed")
    new_input_accessors: dict[int, int] = {}
    for input_accessor, raw_times in sorted(timelines_by_accessor.items()):
        source_times = np.asarray(raw_times, dtype=np.float64)
        retimed_times = np.asarray(
            source_start + (source_times - source_start) * scale,
            dtype=np.dtype("<f4"),
        ).reshape(-1, 1)
        new_input_accessors[input_accessor] = _append_float_accessor(
            root,
            binary,
            retimed_times,
            element_type="SCALAR",
        )
    for channel in walking.channels:
        sampler_index = channel.sampler_index
        sampler = raw_walking["samplers"][sampler_index]
        if not isinstance(sampler, dict):
            raise HumanRuntimeError("promoted Rocketbox Walking sampler is malformed")
        sampler["input"] = new_input_accessors[channel.input_accessor_index]
    buffers[0]["byteLength"] = len(binary)
    retimed_payload = build_glb(root, binary)
    try:
        readback_actions = extract_actions(parse_glb(retimed_payload))
    except GlbError as exc:
        raise HumanRuntimeError(
            f"retimed Rocketbox GLB failed readback: {exc}"
        ) from exc
    walking_after = next(
        action for action in readback_actions if action.name == "Walking"
    )
    idle_after = next(action for action in readback_actions if action.name == "Idle")
    _, _, readback_ticks = _action_span_ticks(walking_after)
    if readback_ticks != target_ticks:
        raise HumanRuntimeError("retimed Walking duration differs on readback")
    if idle_after != idle:
        raise HumanRuntimeError("Walking retime changed the Idle action")
    if len(walking_after.channels) != len(walking.channels) or any(
        after.values != before.values
        or len(after.timestamps_seconds) != len(before.timestamps_seconds)
        for before, after in zip(walking.channels, walking_after.channels, strict=True)
    ):
        raise HumanRuntimeError("Walking retime changed key values or key counts")
    if {
        int(
            round(
                (max(channel.timestamps_seconds) - min(channel.timestamps_seconds))
                * TIME_BASE_HZ
            )
        )
        for channel in walking_after.channels
    } != {target_ticks}:
        raise HumanRuntimeError("not every Walking sampler received the retimed span")
    return retimed_payload, {
        "strategy": "affine_retime_to_profile_sample_count",
        "source_duration_ticks": source_ticks,
        "target_duration_ticks": target_ticks,
        "profile_sample_count": expected_count,
        "source_key_counts": list(_PROFILE_RETIMED_WALKING_SOURCE_KEY_COUNTS),
        "time_scale": scale,
    }


def _source_structure(document: GlbDocument) -> tuple[Any, int, int]:
    try:
        skins = extract_skins(document)
        actions = extract_actions(document)
    except GlbError as exc:
        raise HumanRuntimeError(f"invalid Rocketbox GLB: {exc}") from exc
    if len(skins) != 1 or len(skins[0].joints) != 80:
        raise HumanRuntimeError("Rocketbox source must contain one 80-joint skin")
    skin = skins[0]
    names = tuple(joint.name for joint in skin.joints)
    required = {PELVIS_NODE_NAME, HEAD_LINK_NAME, MOUTH_LINK_NAME}
    if not required.issubset(names):
        raise HumanRuntimeError(
            f"Rocketbox skin lacks required pelvis/head/mouth links: {sorted(required - set(names))}"
        )
    roots = [joint for joint in skin.joints if joint.parent_joint_node_index is None]
    if len(roots) != 1 or roots[0].name != PELVIS_NODE_NAME:
        raise HumanRuntimeError("Rocketbox skin root must be exactly 'Bip01 Pelvis'")
    if tuple(action.name for action in actions) != SOURCE_ACTION_NAMES:
        raise HumanRuntimeError(
            "Rocketbox actions must be exactly Walking, Standing_Idle in source order"
        )
    nodes = document.json.get("nodes")
    if not isinstance(nodes, list):
        raise HumanRuntimeError("Rocketbox GLB nodes must be an array")
    parents = _parents(nodes)
    armature_index = parents[roots[0].node_index]
    if (
        armature_index is None
        or not isinstance(nodes[armature_index], Mapping)
        or nodes[armature_index].get("name") != ARMATURE_NODE_NAME
        or parents[armature_index] is not None
    ):
        raise HumanRuntimeError(
            "Rocketbox pelvis must have one root ancestor named 'Bip01'"
        )
    return skin, int(armature_index), int(roots[0].node_index)


def promote_rocketbox_skin_ancestors(document: GlbDocument) -> bytes:
    """Return a valid GLB with a static zero-weight root above the animated rig."""

    if not isinstance(document, GlbDocument):
        raise HumanRuntimeError("document must be a GlbDocument")
    skin, armature_index, _pelvis_index = _source_structure(document)
    if skin.inverse_bind_matrices is None:
        raise HumanRuntimeError("Rocketbox skin must declare inverse-bind matrices")

    root = deepcopy(document.json)
    binary = bytearray(document.binary)
    nodes = root.get("nodes")
    skins = root.get("skins")
    scenes = root.get("scenes")
    animations = root.get("animations")
    buffers = root.get("buffers")
    if not all(isinstance(value, list) for value in (nodes, skins, scenes, animations)):
        raise HumanRuntimeError("Rocketbox GLB arrays are malformed")
    if not isinstance(buffers, list) or len(buffers) != 1:
        raise HumanRuntimeError("Rocketbox GLB must use exactly one embedded buffer")
    assert isinstance(nodes, list)
    assert isinstance(skins, list)
    assert isinstance(scenes, list)
    assert isinstance(animations, list)

    synthetic_index = len(nodes)
    nodes.append({"name": SYNTHETIC_ROOT_NAME, "children": [armature_index]})
    scene_replacements = 0
    for scene in scenes:
        if not isinstance(scene, dict) or not isinstance(scene.get("nodes"), list):
            raise HumanRuntimeError("Rocketbox scenes must declare root node arrays")
        if armature_index in scene["nodes"]:
            scene["nodes"] = [
                synthetic_index if value == armature_index else value
                for value in scene["nodes"]
            ]
            scene_replacements += 1
    if scene_replacements == 0:
        raise HumanRuntimeError("Rocketbox armature is not reachable as a scene root")

    parents = _parents(nodes)
    source_ibms = [
        np.asarray(value, dtype=np.float64).reshape(4, 4).T
        for value in skin.inverse_bind_matrices
    ]
    first_global = _global_matrix(skin.joints[0].node_index, nodes, parents)
    source_bind_frame = first_global @ source_ibms[0]
    expanded_ibms = list(source_ibms)
    for node_index in (armature_index, synthetic_index):
        global_bind = _global_matrix(node_index, nodes, parents)
        expanded_ibms.append(np.linalg.inv(global_bind) @ source_bind_frame)
    ibm_values = np.vstack([_matrix_values(value) for value in expanded_ibms])
    ibm_accessor = _append_float_accessor(root, binary, ibm_values, element_type="MAT4")

    raw_skin = skins[0]
    if not isinstance(raw_skin, dict) or not isinstance(raw_skin.get("joints"), list):
        raise HumanRuntimeError("Rocketbox skin JSON is malformed")
    raw_skin["joints"].extend([armature_index, synthetic_index])
    raw_skin["skeleton"] = synthetic_index
    raw_skin["inverseBindMatrices"] = ibm_accessor

    for animation in animations:
        if not isinstance(animation, dict):
            raise HumanRuntimeError("Rocketbox animation entries must be objects")
        source_name = animation.get("name")
        animation["name"] = RUNTIME_ACTION_ALIASES.get(source_name, source_name)
        samplers = animation.get("samplers")
        channels = animation.get("channels")
        if not isinstance(samplers, list) or not samplers:
            raise HumanRuntimeError("Rocketbox animation has no samplers")
        if not isinstance(channels, list):
            raise HumanRuntimeError("Rocketbox animation channels must be an array")
        first_sampler = samplers[0]
        if not isinstance(first_sampler, Mapping):
            raise HumanRuntimeError("Rocketbox animation sampler is malformed")
        time_accessor = first_sampler.get("input")
        accessors = root.get("accessors")
        if (
            isinstance(time_accessor, bool)
            or not isinstance(time_accessor, int)
            or not isinstance(accessors, list)
            or time_accessor < 0
            or time_accessor >= len(accessors)
            or not isinstance(accessors[time_accessor], Mapping)
        ):
            raise HumanRuntimeError("Rocketbox animation time accessor is invalid")
        sample_count = accessors[time_accessor].get("count")
        if (
            isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count < 2
        ):
            raise HumanRuntimeError("Rocketbox animation has an invalid sample count")
        identity_rotations = np.tile(
            np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            (sample_count, 1),
        )
        rotation_accessor = _append_float_accessor(
            root, binary, identity_rotations, element_type="VEC4"
        )
        sampler_index = len(samplers)
        samplers.append(
            {
                "input": time_accessor,
                "output": rotation_accessor,
                "interpolation": "LINEAR",
            }
        )
        channels.append(
            {
                "sampler": sampler_index,
                "target": {"node": synthetic_index, "path": "rotation"},
            }
        )

    buffers[0]["byteLength"] = len(binary)
    payload = build_glb(root, binary)
    try:
        promoted = parse_glb(payload)
        promoted_skin = extract_skins(promoted)
        promoted_actions = extract_actions(promoted)
    except GlbError as exc:
        raise HumanRuntimeError(
            f"promoted Rocketbox GLB failed readback: {exc}"
        ) from exc
    if (
        len(promoted_skin) != 1
        or len(promoted_skin[0].joints) != 82
        or promoted_skin[0].joints[-2].name != ARMATURE_NODE_NAME
        or promoted_skin[0].joints[-1].name != SYNTHETIC_ROOT_NAME
        or tuple(action.name for action in promoted_actions) != ("Walking", "Idle")
    ):
        raise HumanRuntimeError("promoted Rocketbox structure differs on readback")
    return payload


def _matrix4(
    value: Any, *, owner: str
) -> tuple[tuple[float, float, float, float], ...]:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise HumanRuntimeError(f"{owner} must be a finite 4x4 matrix")
    return tuple(tuple(float(item) for item in row) for row in matrix)


def prepare_rocketbox_habitat_runtime(
    source_glb: str | Path,
    output_dir: str | Path,
    *,
    package_stem: str = "human",
    walking_profile_sample_count: int | None = None,
    anatomical_forward_source: str = HUMAN_ANATOMICAL_FORWARD_SOURCE,
) -> HumanRuntimePackage:
    """Create and strictly read back one fresh Rocketbox Habitat package."""

    source = Path(source_glb).resolve()
    output = Path(output_dir).resolve()
    expected_walk_count = (
        _walking_profile_sample_count(walking_profile_sample_count) or 16
    )
    if not isinstance(package_stem, str) or not package_stem.isidentifier():
        raise HumanRuntimeError("package_stem must be a non-empty identifier")
    if not isinstance(anatomical_forward_source, str) or not anatomical_forward_source:
        raise HumanRuntimeError("anatomical_forward_source must be non-empty text")
    if not source.is_file() or source.is_symlink():
        raise HumanRuntimeError(f"source_glb must be a regular file: {source}")
    if output.exists() or output.is_symlink():
        raise HumanRuntimeError(f"refusing to replace output directory: {output}")
    output.mkdir(parents=True)
    promoted_path = output / "promoted_source.glb"
    visual_path = output / "visual.glb"
    rebase_path = output / "rebase_report.json"
    actions_path = output / "walking_actions.npz"
    urdf_path = output / f"{package_stem}.urdf"
    ao_config_path = output / f"{package_stem}.ao_config.json"
    mapping_path = output / "joint_mapping.json"
    manifest_path = output / "human_runtime_manifest.json"
    try:
        source_document = load_glb(source)
        promoted_payload = promote_rocketbox_skin_ancestors(source_document)
        promoted_payload, walking_retime = _retime_walking_loop_to_profile(
            promoted_payload,
            walking_profile_sample_count=walking_profile_sample_count,
        )
        promoted_path.write_bytes(promoted_payload)
        rebase = rebase_skin_root_preserving_local_tr(promoted_path, visual_path)
        rebase_path.write_text(
            json.dumps(rebase, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        document = load_glb(visual_path)
        actions = bake_local_tr_actions(document)
        write_local_tr_actions_npz(actions, actions_path)
        walk = next(
            (clip for clip in actions.actions if clip.semantic_action_id == "walk"),
            None,
        )
        if (
            walk is None
            or walk.source_action_name != "Walking"
            or walk.sample_count != expected_walk_count
        ):
            raise HumanRuntimeError(
                "Rocketbox Walking differs from the declared endpoint-exclusive profile"
            )
        if actions.translation_driven_joint_ids != (PELVIS_NODE_NAME,):
            raise HumanRuntimeError(
                "Rocketbox runtime must retain only the pelvis as translation-driven"
            )
        actor_from_skin_root = _matrix4(
            rebase.get("skin", {}).get("actor_from_canonical_root"),
            owner="rebase actor_from_canonical_root",
        )
        mapping = build_local_tr_habitat_mapping(
            document,
            actions,
            actor_from_skin_root=actor_from_skin_root,
            actor_from_skin_root_source=(
                "avengine_m2_skin_root_rebase_local_tr_v2."
                "skin.actor_from_canonical_root"
            ),
        )
        urdf_path.write_text(
            mapping.render_urdf(robot_name=f"avengine_m5_1_rocketbox_{package_stem}"),
            encoding="utf-8",
            newline="\n",
        )
        ao_config_path.write_text(
            json.dumps(
                build_habitat_ao_config_data(
                    render_asset=visual_path.name,
                    urdf_filepath=urdf_path.name,
                    semantic_id=DEFAULT_SEMANTIC_ID,
                    shader_type="pbr",
                ),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        mapping_path.write_text(
            json.dumps(
                mapping.joint_mapping_data(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        manifest: dict[str, Any] = {
            "schema": "avengine_m5_1_rocketbox_human_runtime_v1",
            "status": "pass",
            "research_only": True,
            "qualification_claim": False,
            "source": _record(source),
            "derived": {
                "promoted_glb": _record(promoted_path),
                "visual_glb": _record(visual_path),
                "rebase_report": _record(rebase_path),
                "actions_npz": _record(actions_path),
                "habitat_urdf": _record(urdf_path),
                "habitat_ao_config": _record(ao_config_path),
                "habitat_joint_mapping": _record(mapping_path),
            },
            "skin_contract": {
                "source_joint_count": 80,
                "runtime_joint_count": 82,
                "synthetic_root": SYNTHETIC_ROOT_NAME,
                "zero_weight_ancestor_links": [
                    ARMATURE_NODE_NAME,
                    SYNTHETIC_ROOT_NAME,
                ],
                "runtime_joint_state_count": len(actions.runtime_joint_order),
                "translation_driven_joint_ids": list(
                    actions.translation_driven_joint_ids
                ),
            },
            "action_contract": {
                "source_action_aliases": RUNTIME_ACTION_ALIASES,
                "walking_source_action": "Walking",
                "sample_rate_hz": actions.sample_rate_hz,
                "walking_loop_sample_count": walk.sample_count,
                "walking_loop_duration_ticks": walk.loop_duration_ticks,
                "playback": "explicit_fixed_state_modulo_loop",
            },
            "anchors": {
                "head_link": HEAD_LINK_NAME,
                "mouth_emitter_link": MOUTH_LINK_NAME,
            },
            "anatomical_frame": {
                "actor_up_axis": "+Y",
                "actor_forward_axis": HUMAN_ANATOMICAL_FORWARD_AXIS_LABEL,
                "source": anatomical_forward_source,
            },
            "actor_from_skin_root": [list(row) for row in actor_from_skin_root],
            "notes": [
                "No static-human sliding fallback is allowed.",
                "The source skin ordinals stay unchanged; appended ancestors carry zero vertex weight.",
            ],
        }
        if walking_retime is not None:
            manifest["action_contract"]["walking_timeline_retime"] = walking_retime
        manifest["manifest_content_sha256"] = canonical_json_sha256(manifest)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if read_local_tr_actions_npz(actions_path) != actions:
            raise HumanRuntimeError("actions NPZ differs on strict readback")
        return HumanRuntimePackage(
            source_glb=source,
            promoted_glb=promoted_path,
            visual_glb=visual_path,
            rebase_report=rebase_path,
            actions_npz=actions_path,
            habitat_urdf=urdf_path,
            habitat_ao_config=ao_config_path,
            habitat_joint_mapping=mapping_path,
            package_manifest=manifest_path,
            document=document,
            actions=actions,
            mapping=mapping,
            actor_from_skin_root=actor_from_skin_root,
        )
    except (GlbError, RebaseError, OSError, ValueError) as exc:
        if isinstance(exc, HumanRuntimeError):
            raise
        raise HumanRuntimeError(str(exc)) from exc


__all__ = [
    "ARMATURE_NODE_NAME",
    "DEFAULT_SEMANTIC_ID",
    "HEAD_LINK_NAME",
    "HUMAN_ANATOMICAL_FORWARD_AXIS_LABEL",
    "HUMAN_ANATOMICAL_FORWARD_SOURCE",
    "HumanRuntimeError",
    "HumanRuntimePackage",
    "MOUTH_LINK_NAME",
    "PELVIS_NODE_NAME",
    "SYNTHETIC_ROOT_NAME",
    "prepare_rocketbox_habitat_runtime",
    "promote_rocketbox_skin_ancestors",
]
