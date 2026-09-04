#!/usr/bin/env python3
"""Integrated qa-v3 batch: generic scene solver + per-type audio + facts.

把三样东西接进同一条链:场景无关几何求解器(scene_sampler)、题型专用
音频调度(audio_profiles)、以及从**最终时间线**重算的题目事实。

链路:
    场景输入(导航路线库 + 相机基准请求)
      → 逐格分配答案(方位带 / 角色绑定 / 首叫角色)
      → 几何求解(解 yaw 落带,不枚举)
      → 相机与听者同一姿态结果(apply_camera_listener_pose_ue)
      → 题型 AudioProgram(语义角色 → 槽位绑定)
      → 时间线创作(相机与折线来自求解结果)
      → **在最终相机姿态下重算真值**,与分配的答案格逐条核对
      → 事实记录(MCQ 与 Open 引用同一条事实)

边界:没有像素证据之前一律标 geometry_candidate,不是题目准入。
房间 ID、固定路线名、固定 yaw 一概不出现;场景由 --scene-config 给。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))

import audio_profiles as AP  # noqa: E402
import scene_sampler as SS  # noqa: E402
from build_qa_v3_programs import (  # noqa: E402
    build_program,
    dry_canvas_fields,
    program_request_fields,
    validate_m6_audio_program,
)
from build_qa_v3_n_actor_canary import build_endpoint_registry  # noqa: E402
from avengine.assets.sound_pool import clip_source_from_params  # noqa: E402
from qa_v3_request import answer_forms_from_params, write_requested_questions
from qa_v3_asset_policy import (  # noqa: E402
    AssetPolicyError,
    load_asset_policy,
    resolve_asset_policy,
    slot_context,
)
import qa_v3_arc as AR
import qa_v3_azimuth as AZ  # noqa: E402
from score_open_answers import angle_credit_radius, resolve_angle_policy, score_angle
from qa_v3_pixel_thresholds import card1_pixel_acceptance_block  # noqa: E402
import visibility_prediction as VP  # noqa: E402
# 选角文档的结构(蓝图/网格/动画的物理来源、UE 绑定)已在既有装配器里
# 验证过,直接复用它的构造函数,不在这里重写一份容易走样的副本。
from qa_v3_actor_selection import _selection_doc  # noqa: E402
# 静→走用与旧管线**同一个**变换:创作函数按弧长把整条路线铺满请求帧数,
# 那是"压缩式";求解器用的是保速的"平移式"。两者不一致会让中途帧的
# 位置对不上 —— 集成冒烟里正是反向题(查询帧在中途)先露馅。
from make_idle_then_walk_timeline import (  # noqa: E402
    transform_idle_then_walk,
    transform_to_solved_routes,
)
from avengine.camera_pose import apply_camera_listener_pose_ue  # noqa: E402
from avengine.dataset.apartment_dynamic_audio import (  # noqa: E402
    apartment_ue_point_to_world_m,
)
from avengine.timeline.current_apartment_visual import (  # noqa: E402
    build_current_n_actor_visual_timeline,
)

COAT_WORDS = {
    "generated_border_collie_black_white_medium_standard_adult_research_v1":
        "black-and-white",
    "generated_labrador_yellow_medium_standard_adult_research_v1": "yellow",
}
COAT_OF = dict(COAT_WORDS)
ASSET_OF = {coat: asset for asset, coat in COAT_WORDS.items()}
OTHER_COAT = {"black-and-white": "yellow", "yellow": "black-and-white"}
EP_MAP = {
    "generated_border_collie_black_white_medium_standard_adult_research_v1":
        ("qa_v2_dog_1_collie_muzzle", "qa_v2_dog_2_collie_muzzle"),
    "generated_labrador_yellow_medium_standard_adult_research_v1":
        ("qa_v2_dog_1_labrador_muzzle", "qa_v2_dog_2_labrador_muzzle"),
}
ASSET_PAIR_KIND = {asset_id: "dog" for asset_id in COAT_WORDS}


class GenerationConstraintError(ValueError):
    """A candidate is well-formed but fails a declared question constraint."""


def _require_param(params, key):
    if key not in params:
        raise ValueError(f"params missing {key}")
    return params[key]


def _sample_rate_hz(params) -> int:
    rate = int(_require_param(params, "SAMPLE_RATE_HZ"))
    if rate <= 0:
        raise ValueError("SAMPLE_RATE_HZ must be positive")
    return rate


def _pair_kind(params) -> str:
    value = str(_require_param(params, "PAIR_KIND"))
    if not value:
        raise ValueError("PAIR_KIND is empty")
    return value


def assert_assets_match_pair_kind(
    params,
    assets,
    asset_context=None,
) -> None:
    """Check the audio pair kind without conflating it with visual class."""
    pair_kind = _pair_kind(params)
    if asset_context is not None:
        if pair_kind != asset_context["pair_kind"]:
            raise ValueError(
                f"PAIR_KIND={pair_kind!r} differs from asset policy "
                f"{asset_context['pair_kind']!r}"
            )
        if set(assets) != set(asset_context["asset_ids"]):
            raise ValueError(
                "selected assets differ from the resolved asset policy pair"
            )
        return
    # Legacy direct callers retain the old dog-pair check until they opt into
    # an explicit asset policy context.
    for asset in assets:
        mapped = ASSET_PAIR_KIND.get(asset)
        if mapped is None:
            raise ValueError(f"asset {asset!r} has no pair_kind mapping")
        if mapped != pair_kind:
            raise ValueError(
                f"PAIR_KIND={pair_kind!r} does not match asset {asset!r} "
                f"(mapped {mapped!r})")


def answer_depends_on_source_displacement(profile) -> bool:
    explicit = profile.get("answer_depends_on_source_displacement")
    if explicit is not None:
        return bool(explicit)
    kind = profile.get("answer_kind", "azimuth_band")
    return kind in {"distance_change", "motion_state"} or (
        kind == "azimuth_band"
        and profile.get("temporal") in {"forward", "backward"}
    )


class PredictedVisibilityRejection(GenerationConstraintError):
    """A profile declared a minimum predicted visibility and the plan misses it."""


def _pair_label_context(pair_assets, asset_context=None):
    """Return labels and the two-way label-to-asset mapping for one request."""
    if asset_context is None:
        labels = [COAT_OF[asset] for asset in pair_assets]
        asset_by_label = {label: asset for label, asset in zip(labels, pair_assets)}
        other_by_label = {
            labels[0]: labels[1],
            labels[1]: labels[0],
        }
        return labels, asset_by_label, other_by_label
    if set(pair_assets) != set(asset_context["asset_ids"]):
        raise ValueError("pair assets differ from the asset policy context")
    labels = [
        asset_context["asset_specs"][asset]["label"] for asset in pair_assets
    ]
    if len(set(labels)) != 2:
        raise ValueError("asset policy pair referent labels must be distinct")
    asset_by_label = {
        asset_context["asset_specs"][asset]["label"]: asset
        for asset in pair_assets
    }
    other_by_label = {
        labels[0]: labels[1],
        labels[1]: labels[0],
    }
    return labels, asset_by_label, other_by_label


def sha_rng(*parts) -> np.random.Generator:
    tag = "|".join(str(p) for p in parts).encode()
    return np.random.default_rng(
        int.from_bytes(hashlib.sha256(tag).digest()[:8], "big") % 2**32)


def endpoint_ids_by_slot(selection, endpoint_records):
    """Join endpoint records by instance identity, never by sorted list order."""
    endpoint_by_instance = {
        endpoint["binding"]["entity_instance_id"]: endpoint
        for endpoint in endpoint_records
    }
    result = {}
    for actor in selection["actors"]:
        instance_id = (
            actor.get("entity_instance_id")
            or actor.get("legacy_timeline_actor_id")
        )
        endpoint = endpoint_by_instance.get(instance_id)
        if endpoint is None:
            raise GenerationConstraintError(
                f"{actor['source_slot_id']}: endpoint registry has no binding "
                f"for instance {instance_id!r}"
            )
        result[actor["source_slot_id"]] = endpoint["source_endpoint_id"]
    return result


def apply_asset_facing_policy(
    timeline,
    *,
    camera_position_ue_cm,
    assets_by_slot,
    asset_context,
) -> dict[str, str]:
    """Apply declared static-mesh facing to the authored timeline."""
    facing_by_slot = {}
    camera = [float(value) for value in camera_position_ue_cm]
    for slot, asset_id in assets_by_slot.items():
        facing = asset_context.get("facing_by_asset", {}).get(asset_id)
        if facing is None:
            continue
        facing_by_slot[slot] = facing
        spec = asset_context["asset_specs"][asset_id]
        forward_yaw = float(spec.get("ue_static_forward_yaw_deg") or 0.0)
        for frame in timeline["frames"]:
            state = next(
                item
                for item in frame["actor_states"]
                if item["source_slot_id"] == slot
            )
            if facing == "keep_mesh_forward":
                state["yaw_ue_deg"] = 0.0
                continue
            position = state["translation_ue_cm"]
            dx = camera[0] - float(position[0])
            dy = camera[1] - float(position[1])
            if math.hypot(dx, dy) <= 1.0e-9:
                yaw = 0.0
            else:
                desired = math.degrees(math.atan2(dy, dx))
                yaw = (desired - forward_yaw + 180.0) % 360.0 - 180.0
            state["yaw_ue_deg"] = 0.0 if yaw == 0.0 else yaw
    return facing_by_slot


def balanced(values, n, *seed_parts):
    offset = int(sha_rng(*seed_parts, "value-offset").integers(len(values)))
    values = list(values[offset:]) + list(values[:offset])
    reps = -(-n // len(values))
    pool = (list(values) * reps)[:n]
    order = sha_rng(*seed_parts).permutation(n)
    return [pool[int(i)] for i in order]


def balanced_binary_joint(left_values, right_values, n, *seed_parts):
    """Balance a 2x2 joint assignment, not only its two marginals.

    Repeating all four cells guarantees that each left value occurs with both
    right values.  A deterministic permutation changes row order without
    changing the joint counts.
    """
    if len(left_values) != 2 or len(right_values) != 2:
        raise ValueError("balanced_binary_joint requires two values per axis")
    diagonal = int(sha_rng(*seed_parts, "joint-diagonal").integers(2))
    if diagonal == 0:
        cycle = [
            (left_values[0], right_values[0]),
            (left_values[1], right_values[1]),
            (left_values[0], right_values[1]),
            (left_values[1], right_values[0]),
        ]
    else:
        cycle = [
            (left_values[0], right_values[1]),
            (left_values[1], right_values[0]),
            (left_values[0], right_values[0]),
            (left_values[1], right_values[1]),
        ]
    pool = (cycle * -(-n // len(cycle)))[:n]
    order = sha_rng(*seed_parts).permutation(n)
    return [pool[int(i)] for i in order]


def git_worktree_state(repo=REPO):
    revision = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, text=True, capture_output=True).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--short"],
        check=True, text=True, capture_output=True).stdout.splitlines()
    return {"revision": revision, "dirty": bool(status), "status": status}


def content_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


WORLD_TRANSFORMS = {
    "ue_xyz_cm_to_xzy_m_v1": apartment_ue_point_to_world_m,
}


def resolve_scene_render_context(scene):
    """Resolve render facts from the scene input; never fall back apartment."""
    render = scene.render_config
    required = ("native_map", "room_profile_id", "world_transform",
                "ground_z_ue_cm")
    missing = [key for key in required
               if key not in render or render[key] is None]
    if missing:
        raise ValueError(
            f"{scene.scene_id}: integration rendering requires scene.render "
            f"keys {missing}; no apartment fallback is allowed")
    native_map = str(render["native_map"])
    if not native_map.startswith("/Game/"):
        raise ValueError(
            f"{scene.scene_id}: native_map must be a /Game package path")
    transform_id = str(render["world_transform"])
    transform = WORLD_TRANSFORMS.get(transform_id)
    if transform is None:
        raise ValueError(
            f"{scene.scene_id}: unsupported world_transform {transform_id!r}")
    ground_z = float(render["ground_z_ue_cm"])
    if not np.isfinite(ground_z):
        raise ValueError(f"{scene.scene_id}: ground_z_ue_cm must be finite")
    floor = (getattr(scene, "provenance", None) or {}).get("floor_reference")
    if not floor or floor.get("status") != "measured":
        raise ValueError(
            f"{scene.scene_id}: render facts require a measured floor reference; "
            f"ground_z_ue_cm={ground_z} is unverified")
    return {
        "native_map": native_map,
        "room_profile_id": str(render["room_profile_id"]),
        "world_transform": transform,
        "world_transform_id": transform_id,
        "ground_z_ue_cm": ground_z,
        "floor_reference": floor,
    }


def recompute_azimuth(timeline, slot, frame):
    """Compute the bearing from an actor root in the final timeline."""
    record = timeline["frames"][frame]
    camera = record["camera"]
    cam_xy = (float(camera["translation_ue_cm"][0]),
              float(camera["translation_ue_cm"][1]))
    yaw = float(camera["yaw_ue_deg"])
    for state in record["actor_states"]:
        if state["source_slot_id"] == slot:
            xy = (float(state["translation_ue_cm"][0]),
                  float(state["translation_ue_cm"][1]))
            return SS.relative_azimuth_deg(cam_xy, yaw, xy)
    raise KeyError(f"slot {slot} missing at frame {frame}")


def recompute_emitter_azimuth(timeline, slot, frame,
                              emitter_paths_by_slot=None):
    """Compute audio bearing from the runtime-bound emitter child if present."""
    if emitter_paths_by_slot is None or slot not in emitter_paths_by_slot:
        return recompute_azimuth(timeline, slot, frame)
    record = timeline["frames"][frame]
    camera = record["camera"]
    cam_xy = (float(camera["translation_ue_cm"][0]),
              float(camera["translation_ue_cm"][1]))
    yaw = float(camera["yaw_ue_deg"])
    point = emitter_paths_by_slot[slot][int(frame)]
    return SS.relative_azimuth_deg(
        cam_xy, yaw, (float(point[0]), float(point[1])))


def materialize_emitter_paths(timeline, asset_registry, slot_assets):
    """Materialize registry emitter-child paths in timeline UE cm.

    Registry offsets use final_scaled_asset_root AVEngine basis
    (X-forward, Y-up, Z-right). The Apartment runtime maps them to UE
    (X-forward, Y-right, Z-up) and attaches them below the yawed timeline root.
    Actor scale comes from the authored timeline when explicitly applied;
    articulated runtime otherwise keeps the blueprint default scale.
    """
    if not isinstance(asset_registry, dict):
        raise ValueError("asset registry must be an object")
    assets = {
        str(asset["asset_id"]): asset
        for asset in asset_registry.get("assets", [])
        if isinstance(asset, dict) and asset.get("asset_id")
    }
    actors = {
        str(actor["source_slot_id"]): actor
        for actor in timeline.get("actors", [])
        if isinstance(actor, dict) and actor.get("source_slot_id")
    }
    paths = {}
    metadata = {}
    for slot, asset_id in slot_assets.items():
        asset = assets.get(str(asset_id))
        if asset is None:
            raise GenerationConstraintError(
                f"{slot}: asset {asset_id!r} is absent from the source registry")
        anchor_id = asset.get("default_emitter_anchor_id")
        anchors = [
            anchor for anchor in asset.get("emitter_anchors", [])
            if isinstance(anchor, dict) and anchor.get("anchor_id") == anchor_id
        ]
        if len(anchors) != 1:
            raise GenerationConstraintError(
                f"{slot}: registry has no unique default emitter anchor")
        anchor = anchors[0]
        if anchor.get("offset_space") != "final_scaled_asset_root":
            raise GenerationConstraintError(
                f"{slot}: unsupported emitter offset space "
                f"{anchor.get('offset_space')!r}")
        try:
            offset = np.asarray(anchor["offset_m"], dtype=float)
        except (TypeError, ValueError) as exc:
            raise GenerationConstraintError(
                f"{slot}: emitter offset must be numeric") from exc
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise GenerationConstraintError(
                f"{slot}: emitter offset must be a finite 3-vector")
        actor_scale = 1.0
        actor = actors.get(slot, {})
        if actor.get("actor_scale") is not None:
            actor_scale = float(actor["actor_scale"])
            if not math.isfinite(actor_scale) or actor_scale <= 0.0:
                raise GenerationConstraintError(
                    f"{slot}: authored actor scale is invalid")
        local_ue_cm = np.asarray(
            [100.0 * offset[0], 100.0 * offset[2], 100.0 * offset[1]],
            dtype=float,
        ) * actor_scale
        path = []
        for frame in timeline.get("frames", []):
            state = next(
                state for state in frame["actor_states"]
                if state["source_slot_id"] == slot
            )
            root = np.asarray(state["translation_ue_cm"], dtype=float)
            yaw = math.radians(float(state["yaw_ue_deg"]))
            rotated = np.asarray([
                math.cos(yaw) * local_ue_cm[0] - math.sin(yaw) * local_ue_cm[1],
                math.sin(yaw) * local_ue_cm[0] + math.cos(yaw) * local_ue_cm[1],
                local_ue_cm[2],
            ])
            point = root + rotated
            if not np.all(np.isfinite(point)):
                raise GenerationConstraintError(
                    f"{slot}: emitter path contains non-finite coordinates")
            path.append([float(value) for value in point])
        if len(path) != len(timeline.get("frames", [])):
            raise GenerationConstraintError(
                f"{slot}: emitter path has the wrong frame count")
        runtime = asset.get("runtime_backends", {}).get("spear_unreal", {})
        paths[slot] = path
        metadata[slot] = {
            "asset_id": str(asset_id),
            "revision": asset.get("revision"),
            "emitter_anchor_id": str(anchor_id),
            "emitter_offset_m": [float(value) for value in offset],
            "offset_space": anchor["offset_space"],
            "applied_actor_scale": actor_scale,
            "registry_actor_scale": runtime.get("actor_scale"),
            "local_anatomical_forward_axis": (
                asset.get("timeline", {}).get("local_anatomical_forward_axis")
            ),
            "ue_anatomical_forward_yaw_deg": runtime.get(
                "ue_anatomical_forward_yaw_deg"),
            "scale_source": (
                "timeline_actor_declaration"
                if actor.get("actor_scale") is not None
                else "articulated_runtime_blueprint_default"),
            "local_offset_ue_cm": [float(value) for value in local_ue_cm],
            "frame_count": len(path),
            "frame_rate_hz": timeline.get("render", {}).get("frame_rate_hz"),
            "ticks_per_frame": timeline.get("render", {}).get("ticks_per_frame"),
        }
    return paths, metadata

QUERY_WINDOW_S = 0.5


def query_window_seconds(
    query_frame: int, video_fps: float, *, window_seconds: float = QUERY_WINDOW_S,
) -> tuple[float, float]:
    """Return a tidy configured window containing the query frame.

    Half a second remains the historical default. Fraction arithmetic avoids
    assigning a decimal boundary such as 0.6/0.2 to the preceding window.
    """
    if isinstance(query_frame, bool) or not isinstance(query_frame, int) or query_frame < 0:
        raise ValueError("query_frame must be a nonnegative integer")
    rate = float(video_fps)
    width = float(window_seconds)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("video_fps must be finite and positive")
    if not math.isfinite(width) or width <= 0:
        raise ValueError("query window duration must be finite and positive")
    seconds = Fraction(query_frame) / Fraction(str(rate))
    duration = Fraction(str(width))
    lo = (seconds // duration) * duration
    return float(lo), float(lo + duration)


def profile_query_window(profile, params, query_frame, video_fps):
    duration = profile.get("query_window_seconds", params.get("QUERY_WINDOW_SECONDS", QUERY_WINDOW_S))
    return query_window_seconds(query_frame, video_fps, window_seconds=duration)


def query_window_frame_bounds(window_s, video_fps, frame_count):
    """Map the declared human time window to inclusive frame indices."""
    lo_s, hi_s = (float(value) for value in window_s)
    fps = float(video_fps)
    count = int(frame_count)
    if not math.isfinite(fps) or fps <= 0.0 or count < 1:
        raise GenerationConstraintError(
            "query window needs a positive frame rate and non-empty timeline")
    lo_f = max(0, math.ceil(Fraction(str(lo_s)) * Fraction(str(fps))))
    hi_f = min(count - 1, math.floor(Fraction(str(hi_s)) * Fraction(str(fps))))
    if hi_f < lo_f:
        raise GenerationConstraintError(
            f"query window {window_s} covers no frame of {count}")
    return lo_f, hi_f


def azimuth_sweep_engine_arc(timeline, slot, window_s, video_fps,
                              emitter_paths_by_slot=None):
    """Return the directed Arc swept over a query window and its frame bounds."""
    frame_count = len(timeline["frames"])
    lo_f, hi_f = query_window_frame_bounds(
        window_s, video_fps, frame_count)
    values = [recompute_emitter_azimuth(
                  timeline, slot, f, emitter_paths_by_slot)
              for f in range(lo_f, hi_f + 1)]
    return AR.Arc.from_samples(values), (lo_f, hi_f)


def azimuth_sweep_engine_frame(timeline, slot, window_s, video_fps,
                                emitter_paths_by_slot=None):
    """Compatibility numeric view of a directed query-window Arc.

    The returned ordered endpoints may have hi < lo at the seam. Callers
    that need exact direction or a sweep wider than 180 degrees should use
    azimuth_sweep_engine_arc and persist its Arc dictionary.
    """
    arc, frames = azimuth_sweep_engine_arc(
        timeline, slot, window_s, video_fps, emitter_paths_by_slot)
    if arc.width_deg >= 360.0:
        return -180.0, 180.0, frames
    return arc.start_deg, arc.end_deg, frames

def band_of(value, edges):
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return i
    return None


def audit_gatea_answers(profile, main_answer, gatea_answer, params):
    """Check question and gold relations without making audio-render claims."""
    gold_relation = profile.get("gatea_gold_relation", "flip")
    if gold_relation not in ("flip", "preserve"):
        raise GenerationConstraintError(
            f"unknown Gate A gold relation {gold_relation!r}")

    structure = {
        "mcq_stem_same": main_answer["mcq"]["stem"] == gatea_answer["mcq"]["stem"],
        "mcq_options_same": main_answer["mcq"]["options_space"] == gatea_answer["mcq"]["options_space"],
        "open_stem_same": main_answer["open"]["stem"] == gatea_answer["open"]["stem"],
    }
    mcq_flipped = (main_answer["mcq"]["truth_option"]
                   != gatea_answer["mcq"]["truth_option"])
    mcq_preserved = not mcq_flipped
    mcq_relation_satisfied = (
        mcq_flipped if gold_relation == "flip" else mcq_preserved)
    main_open = main_answer["open"]
    gatea_open = gatea_answer["open"]
    scoring = main_open["scoring"]
    if scoring != gatea_open["scoring"]:
        raise GenerationConstraintError(
            "Gate A changed the Open scoring protocol")
    if "truth_interval_deg" in main_open:
        open_preserved = (main_open["truth_interval_deg"]
                          == gatea_open.get("truth_interval_deg"))
    else:
        open_preserved = main_open["truth_value"] == gatea_open["truth_value"]
    if scoring in ("circular_deg", "circular_deg_interval"):
        angle_policy = resolve_angle_policy(params, main_open.get("certification_policy"))
        if resolve_angle_policy(params, gatea_open.get("certification_policy")) != angle_policy:
            raise GenerationConstraintError("Gate A changed the angle scoring policy")
        credit_radius = angle_credit_radius(params, angle_policy)
        credit_key = "THETA_FULL" if angle_policy == "strict_full_credit_only" else "THETA_HALF"
    if scoring == "circular_deg":
        separation = SS.circular_gap_deg(
            float(main_open["truth_value"]),
            float(gatea_open["truth_value"]))
        threshold = 2.0 * credit_radius
        open_separated = separation > threshold
        open_rule = f"circular_distance > 2*{credit_key}"
    elif scoring == "circular_deg_interval":
        # 真值是窗口内扫过的区间,所以"宽信区域不相交"要按区间算:两个区间各自向外扩
        # THETA_HALF 之后不许相交。
        #
        # 这里原来是端点相减,而且 1f3ecd5 给它加的守卫**没有盖住它自己注释举的例子**
        # (claude-d3 2026-09-04 在实跑中指出并复现):歧义不是某一个区间的属性,是这一对
        # 相对于 ±180 那条缝的位置。[172,178] 与 [-178,-172] 各自都是正序的,谁都不 wrap,
        # 所以 hi < lo 的守卫不响;线性相减照旧算出 344 度、越过 2*THETA_HALF=60、判为
        # 已分离,而圆上真实间隙是 4 度。那是认证的假通过:两个金标叠在一起,一个不听音频
        # 的模型两边都报同一个角度就全中。
        #
        # 分离是圆上的集合性质,所以按集合算,而且记录的间隙也换成环形的——原来那个 344
        # 连证据本身都是错的。
        def answer_interval_arc(open_block):
            record = open_block.get("truth_interval_arc")
            if isinstance(record, dict):
                return AR.Arc.from_dict(record)
            lo, hi = (float(v) for v in open_block["truth_interval_deg"])
            return AR.Arc.from_forward_bounds(lo, hi)

        main_arc = answer_interval_arc(main_open)
        gate_arc = answer_interval_arc(gatea_open)
        threshold = 2.0 * credit_radius
        separation = AR.circular_gap_deg(main_arc, gate_arc)
        open_separated = AR.wide_credit_regions_disjoint(
            main_arc, gate_arc, credit_radius)
        open_rule = ("dilated arcs disjoint on the circle "
                     f"(gap > 2*{credit_key} for non-wrapping pairs)")
    elif scoring == "absolute_time":
        separation = abs(float(main_open["truth_value"])
                         - float(gatea_open["truth_value"]))
        # Card8 Open is certified strictly (full credit only within T_FULL),
        # so the two golds must be separated by the same derived minimum the
        # scheduler enforces: strictly more than max(T_HALF, 2*T_FULL).
        time_scoring = AP.card8_scoring_params(params)
        threshold = time_scoring["min_first_call_separation_s"]
        open_separated = separation > threshold
        open_rule = "absolute_time_difference > max(T_HALF, 2*T_FULL)"
    elif scoring == "closed_set":
        separation = None
        threshold = None
        open_separated = (main_open["truth_value"]
                          != gatea_open["truth_value"])
        open_rule = "closed_set_gold_changed"
    elif scoring == "count_single":
        separation = abs(int(main_open["truth_value"])
                         - int(gatea_open["truth_value"]))
        threshold = 0
        open_separated = separation > 0
        open_rule = "integer_gold_changed"
    else:
        raise GenerationConstraintError(
            f"no Gate A Open audit for scoring {scoring!r}")

    open_relation_satisfied = (
        open_separated if gold_relation == "flip" else open_preserved)
    if gold_relation == "preserve":
        open_rule = f"{scoring}_gold_preserved"
    checks = {
        **structure,
        "mcq_gold_flipped": mcq_flipped,
        "open_gold_separated": open_separated,
        "gatea_gold_relation": gold_relation,
        "mcq_gold_preserved": mcq_preserved,
        "mcq_gold_relation_satisfied": mcq_relation_satisfied,
        "open_gold_preserved": open_preserved,
        "open_gold_relation_satisfied": open_relation_satisfied,
        "open_separation": separation,
        "open_min_separation": threshold,
        "open_rule": open_rule,
        "profile_id": profile["id"],
    }
    failed = [k for k, value in structure.items() if not value]
    if not mcq_relation_satisfied:
        failed.append("mcq_gold_flipped" if gold_relation == "flip"
                      else "mcq_gold_preserved")
    if not open_relation_satisfied:
        failed.append("open_gold_separated" if gold_relation == "flip"
                      else "open_gold_preserved")
    if failed:
        raise GenerationConstraintError(
            f"{profile['id']} Gate A failed generation checks: {failed}; "
            f"open separation={separation}, threshold={threshold}")
    return checks


def audit_gatea_pair(profile, main_program, gatea_program, main_answer,
                     gatea_answer, params):
    """Check both audio intervention structure and the shared answer rules."""
    checks = audit_gatea_answers(profile, main_answer, gatea_answer, params)
    main_events = main_program["events"]
    gatea_events = gatea_program["events"]

    def without_slot(event):
        return {k: v for k, v in event.items()
                if k not in ("event_id", "source_endpoint_id")}

    structure = {
        "event_count_same": len(main_events) == len(gatea_events),
        "candidate_endpoints_same": main_program["candidate_source_endpoint_ids"] == gatea_program["candidate_source_endpoint_ids"],
        "non_slot_event_fields_same": [without_slot(e) for e in main_events] == [without_slot(e) for e in gatea_events],
        "slot_sequence_changed": [e["source_endpoint_id"] for e in main_events] != [e["source_endpoint_id"] for e in gatea_events],
    }
    failed = [key for key, value in structure.items() if not value]
    if failed:
        raise GenerationConstraintError(
            f"{profile['id']} Gate A failed generation checks: {failed}")
    return {**checks, **structure}


def frame_geometry(timeline, slot, frame):
    """Distance, depression and whether the actor's base point projects inside
    the frame, from the final timeline's camera pose and render contract."""
    record = timeline["frames"][frame]
    camera = record["camera"]
    cam = [float(v) for v in camera["translation_ue_cm"]]
    state = next(s for s in record["actor_states"]
                 if s["source_slot_id"] == slot)
    actor = [float(v) for v in state["translation_ue_cm"]]
    render = timeline.get("render") or {}
    hfov = float(render.get("hfov_degrees", 105.0))
    height_px, width_px = render.get("resolution_hw", [720, 1280])
    aspect = float(width_px) / float(height_px)
    half_v = math.degrees(math.atan(math.tan(math.radians(hfov / 2.0)) / aspect))
    distance_cm = math.hypot(actor[0] - cam[0], actor[1] - cam[1])
    azimuth = recompute_azimuth(timeline, slot, frame)
    depression = math.degrees(math.atan2(cam[2] - actor[2], distance_cm))
    projected = (math.tan(math.radians(depression))
                 / max(1e-9, math.cos(math.radians(azimuth))))
    return {
        "camera_height_cm": cam[2] - actor[2],
        "distance_cm": distance_cm,
        "depression_deg": depression,
        "half_vertical_fov_deg": half_v,
        "base_projects_inside_frame": (
            abs(azimuth) <= hfov / 2.0
            and abs(projected) <= math.tan(math.radians(half_v))),
        "role": "informational_not_a_gate",
    }


REALIZED_CARD1_GATES = (
    "realized_anchor_in_allocated_band",
    "realized_query_in_answer_band",
    "realized_anchor_answer_scores_zero",
    "mcq_gold_flipped",
    "open_gold_regions_disjoint",
)


def realized_cross_time_checks(timeline, *, profile, cell, target_slot,
                               other_slot, anchor_frame, query_frame, params,
                               plan_checks=None, query_window_s=None,
                               emitter_paths_by_slot=None):
    """Fail-closed card1 acceptance on the **final** timeline.

    The solver plans angles on the pre-authoring route; idle-then-walk
    authoring and camera-pose application can move the realized anchor by a
    fraction of a frame (Kujiale card1F_002: planning value 9.415 deg, realized
    8.451 deg).  A borderline plan can therefore pass at plan time and fail on
    the clip that is actually rendered, so every acceptance gate is recomputed
    here from the timeline: allocated anchor band, answer band, zero Open
    credit for the audible anchor angle, Gate A MCQ flip and Gate A Open
    separation.  Plan values are reported only as planning values.
    """
    theta_full = float(params["THETA_FULL"])
    theta_half = float(params["THETA_HALF"])
    angle_policy = resolve_angle_policy(params)
    credit_radius = angle_credit_radius(params)
    bands = list(profile["answer_bands_deg"])

    def band_index(value):
        return next((index for index, band in enumerate(bands)
                     if SS.band_contains(value, band)), None)

    def side(slot):
        anchor = recompute_emitter_azimuth(
            timeline, slot, anchor_frame, emitter_paths_by_slot)
        query = recompute_emitter_azimuth(
            timeline, slot, query_frame, emitter_paths_by_slot)
        gap = SS.circular_gap_deg(anchor, query)
        interval_arc = None
        if query_window_s is not None:
            interval_arc, _ = azimuth_sweep_engine_arc(
                timeline, slot, query_window_s,
                _timeline_video_fps(timeline, params),
                emitter_paths_by_slot)
        scored = score_angle(
            f"{anchor} deg", query, theta_full, theta_half,
            certification_policy=angle_policy, convention="engine_right_positive",
            truth_interval_deg=(
                interval_arc.as_dict() if interval_arc is not None else None))
        if scored["status"] != "scored":
            raise GenerationConstraintError(f"cannot score realized anchor angle: {scored}")
        return {
            "slot": slot,
            "anchor_azimuth_deg_engine_frame": anchor,
            "query_azimuth_deg_engine_frame": query,
            "query_interval_engine_frame": (
                interval_arc.as_dict()
                if interval_arc is not None
                else AR.Arc(start_deg=query, sweep_deg=0.0).as_dict()),
            "anchor_query_gap_deg": gap,
            "anchor_band_index": band_index(anchor),
            "query_band_index": band_index(query),
            "anchor_angle_open_score_as_answer": scored["score"],
            "anchor_frame_geometry": frame_geometry(timeline, slot, anchor_frame),
            "query_frame_geometry": frame_geometry(timeline, slot, query_frame),
        }

    main = side(target_slot)
    gatea = side(other_slot)
    main_arc = AR.Arc.from_dict(main["query_interval_engine_frame"])
    gatea_arc = AR.Arc.from_dict(gatea["query_interval_engine_frame"])
    allocated = cell.get("anchor_band")
    answer_band = cell["answer_band"]
    plan_checks = plan_checks or {}
    resolved_geometry = _resolved_query_geometry(profile, cell)
    planned_anchor = plan_checks.get("az_anchor_deg")
    planned_query = plan_checks.get("az_end_deg", plan_checks.get("az_query_deg"))
    checks = {
        "provenance": "final_timeline_recompute_after_camera_pose",
        "anchor_frame": int(anchor_frame),
        "query_frame": int(query_frame),
        "theta_full_deg": theta_full,
        "theta_half_deg": theta_half,
        "angle_certification_policy": angle_policy,
        "credited_radius_deg": credit_radius,
        "answer_domain": profile.get("answer_domain"),
        "query_domain": resolved_geometry.get("query_domain"),
        "query_requires_visibility": resolved_geometry.get(
            "query_requires_visibility"),
        "convention": _profile_convention(profile),
        "query_window_seconds": query_window_s,
        "main": main,
        "gatea": gatea,
        "allocated_anchor_band": (
            list(SS.band_to_bounds(allocated)) if allocated is not None
            else None),
        "answer_band": list(SS.band_to_bounds(answer_band)),
        "realized_anchor_in_allocated_band": (
            allocated is None
            or SS.band_contains(
                main["anchor_azimuth_deg_engine_frame"], allocated)),
        "realized_query_in_answer_band": SS.band_contains(
            main["query_azimuth_deg_engine_frame"], answer_band),
        "realized_anchor_answer_scores_zero": main["anchor_angle_open_score_as_answer"] == 0.0,
        "mcq_gold_flipped": (
            main["query_band_index"] is not None
            and gatea["query_band_index"] is not None
            and main["query_band_index"] != gatea["query_band_index"]),
        "open_gold_separation_deg": AR.circular_gap_deg(main_arc, gatea_arc),
        "open_gold_min_separation_deg": 2.0 * credit_radius,
        "open_gold_regions_disjoint": AR.wide_credit_regions_disjoint(
            main_arc, gatea_arc, credit_radius),
        "planned_vs_realized": {
            "planned_anchor_azimuth_deg_planning_value_only": planned_anchor,
            "planned_query_azimuth_deg_planning_value_only": planned_query,
            "anchor_deviation_deg": (
                SS.circular_gap_deg(float(planned_anchor),
                                    main["anchor_azimuth_deg_engine_frame"])
                if planned_anchor is not None else None),
            "query_deviation_deg": (
                SS.circular_gap_deg(float(planned_query),
                                    main["query_azimuth_deg_engine_frame"])
                if planned_query is not None else None),
        },
        "gates": list(REALIZED_CARD1_GATES),
        # 本字典里所有方位都是引擎帧（右为正），与发布出去的
        # DCASE 左为正符号相反。
        "azimuth_frame": "engine_right_positive",
    }
    failed = [name for name in REALIZED_CARD1_GATES if not checks[name]]
    checks["failed"] = failed
    checks["passed"] = not failed
    if failed:
        raise GenerationConstraintError(
            f"{profile['id']} realized timeline failed {failed}: main anchor "
            f"{main['anchor_azimuth_deg_engine_frame']:.3f} deg, main query "
            f"{main['query_azimuth_deg_engine_frame']:.3f} deg, gap "
            f"{main['anchor_query_gap_deg']:.3f} deg, Gate A query "
            f"{gatea['query_azimuth_deg_engine_frame']:.3f} deg; planning value for the "
            f"anchor was {planned_anchor}")
    return checks


def validate_anchor_binding(profile, schedule, slot_events, *, target_slot,
                            query_frame, answer):
    """Cross-check a profile's declared audio selector against bound events."""
    binding = profile.get("anchor_binding")
    if binding == "none":
        return {"binding": binding, "selected_slot": None}
    if binding == "target":
        actual = slot_events[schedule.anchor_index][0]
        if actual != target_slot:
            raise GenerationConstraintError(
                f"{profile['id']}: identity anchor bound to {actual}, "
                f"expected target {target_slot}")
        return {"binding": binding, "selected_slot": actual}
    if binding == "query_caller":
        calling = [
            slot for (slot, _), event in zip(slot_events, schedule.events)
            if event.frame_span()[0] <= query_frame < event.frame_span()[1]
        ]
        if calling != [target_slot]:
            raise GenerationConstraintError(
                f"{profile['id']}: query caller {calling}, expected "
                f"[{target_slot}]")
        return {"binding": binding, "selected_slot": calling[0]}
    if binding == "first_caller":
        first_slot = min(slot_events, key=lambda item: item[1])[0]
        expected = answer.get("first_caller_slot")
        if expected is None or first_slot != expected:
            raise GenerationConstraintError(
                f"{profile['id']}: first caller {first_slot}, fact says "
                f"{expected}")
        return {"binding": binding, "selected_slot": first_slot}
    raise GenerationConstraintError(
        f"{profile['id']}: unknown anchor_binding {binding!r}")


def solver_target_moves_more(profile, cell):
    """Return the motion-rank constraint only when the profile declares it."""
    if profile.get("enforce_target_moves_more") is False:
        return None
    return cell.get("target_moves_more")


def profile_query_geometry(profile, cell):
    """Resolve query geometry from a profile, optionally per front/back answer."""
    geometry = {
        "query_domain": profile.get("query_domain"),
        "answer_domain": profile.get("answer_domain"),
        "query_bound_deg": profile.get("query_bound_deg"),
        "query_requires_visibility": profile.get("query_requires_visibility"),
        "secondary_anchor_bound_deg": profile.get("secondary_anchor_bound_deg"),
        "secondary_query_bound_deg": profile.get("secondary_query_bound_deg"),
    }
    if profile.get("answer_kind") != "front_back":
        return geometry
    labels = SS.front_back_labels(profile)
    bands = profile.get("answer_bands_deg")
    if not isinstance(bands, list):
        raise GenerationConstraintError(
            f"{profile.get('id')}: front/back profile has no materialized answer bands")
    index = next((i for i, band in enumerate(bands)
                  if SS.bands_equivalent(band, cell["answer_band"])), None)
    if index is None or index >= len(labels):
        raise GenerationConstraintError(
            f"{profile.get('id')}: allocated front/back answer band is invalid")
    overrides = profile.get("query_geometry_by_answer") or {}
    if not isinstance(overrides, dict):
        raise ValueError(
            f"{profile.get('id')}: query_geometry_by_answer must be an object")
    override = overrides.get(labels[index])
    if override is None:
        return geometry
    if not isinstance(override, dict):
        raise ValueError(
            f"{profile.get('id')}: query geometry for {labels[index]!r} must be an object")
    geometry = dict(geometry)
    if "query_domain" in override or "domain" in override:
        geometry["query_domain"] = override.get(
            "query_domain", override.get("domain"))
        # A per-answer query domain is intentionally distinct from the
        # front_back answer space; resolve_query_geometry must not treat it as
        # a conflicting alias.
        geometry["answer_domain"] = None
    for key in ("query_bound_deg", "query_requires_visibility",
                "secondary_anchor_bound_deg", "secondary_query_bound_deg"):
        if key in override:
            geometry[key] = override[key]
    return geometry

def solve_for_profile(profile, cell, scene, params, rng, ledger, *, candidate_validator=None):
    """Dispatch a declared temporal/question shape to the generic solver."""
    temporal = profile["temporal"]
    kind = profile.get("answer_kind", "azimuth_band")
    geometry = profile_query_geometry(profile, cell)
    # Preserve the per-cell resolution for the authored fact. In front/back
    # profiles the front and back answers can intentionally use different
    # query domains and visibility rules.
    cell["_resolved_query_geometry"] = dict(geometry)

    if (temporal == "instant"
            and kind in ("instant_azimuth_band", "first_sound_side",
                         "front_back")):
        return SS.solve_instant_azimuth(
            scene, params, answer_band=cell["answer_band"],
            answer_bands=[tuple(b) for b in profile["answer_bands_deg"]],
            query_frame=profile["binding_frames"][0],
            profile_id=profile["id"],
            idle_choices=profile["idle_choices"], rng=rng, ledger=ledger,
            target_moves_more=solver_target_moves_more(profile, cell),
            max_attempts=profile.get("max_attempts", 3000),
            open_half_width_deg=profile.get("open_half_width_deg"),
            candidate_validator=candidate_validator,
            **geometry)

    if temporal == "instant" and kind == "distance_at_query":
        return SS.solve_instant_distance_order(
            scene, params, query_frame=profile["binding_frames"][0],
            profile_id=profile["id"],
            idle_choices=profile["idle_choices"], rng=rng, ledger=ledger,
            target_moves_more=solver_target_moves_more(profile, cell),
            min_distance_gap_cm=profile.get("min_distance_gap_cm", 50.0),
            max_attempts=profile.get("max_attempts", 3000))

    if kind == "distance_change":
        start_frame, end_frame = [
            int(value) for value in profile["relation_frames"]]
        return SS.solve_distance_change_pair(
            scene, params, start_frame=start_frame, end_frame=end_frame,
            target_relation=str(cell["answer_value"]),
            profile_id=profile["id"],
            idle_choices=profile["idle_choices"], rng=rng, ledger=ledger,
            target_moves_more=solver_target_moves_more(profile, cell),
            min_change_cm=profile.get("min_distance_change_cm", 50.0),
            max_attempts=profile.get("max_attempts", 3000))

    if kind == "motion_state":
        start_frame, end_frame = [
            int(value) for value in profile["motion_frames"]]
        return SS.solve_motion_state_pair(
            scene, params, start_frame=start_frame, end_frame=end_frame,
            target_state=str(cell["answer_value"]),
            profile_id=profile["id"],
            idle_choices=profile["idle_choices"], rng=rng, ledger=ledger,
            min_motion_cm=profile.get("min_motion_cm", 10.0),
            max_attempts=profile.get("max_attempts", 3000))

    if temporal == "forward":
        return SS.solve_forward_cross_time(
            scene, params, answer_band=cell["answer_band"],
            answer_bands=[tuple(b) for b in profile["answer_bands_deg"]],
            anchor_frame=profile["anchor_frame"],
            idle_choices=profile["idle_choices"], rng=rng, ledger=ledger,
            anchor_band=cell.get("anchor_band"),
            target_moves_more=solver_target_moves_more(profile, cell),
            max_attempts=profile.get("max_attempts", 3000),
            **geometry)
    if temporal == "backward":
        return SS.solve_backward_cross_time(
            scene, params, answer_band=cell["answer_band"],
            answer_bands=[tuple(b) for b in profile["answer_bands_deg"]],
            anchor_frame=profile["anchor_frame"],
            query_frame=profile["query_frame"],
            idle_choices=profile["idle_choices"], rng=rng, ledger=ledger,
            anchor_band=cell.get("anchor_band"),
            target_moves_more=solver_target_moves_more(profile, cell),
            max_attempts=profile.get("max_attempts", 3000),
            **geometry)
    if temporal == "instant":
        return SS.solve_instant_binding(
            scene, params, instants=profile["binding_frames"],
            profile_id=profile["id"], idle_choices=profile["idle_choices"],
            rng=rng, ledger=ledger,
            target_moves_more=solver_target_moves_more(profile, cell),
            max_attempts=profile.get("max_attempts", 3000))
    raise ValueError(f"unknown temporal relation {temporal!r}")

def validate_profiles(profiles):
    """Reject configuration mistakes before creating any batch output."""
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("profiles must be a non-empty list")
    ids = [profile.get("id") for profile in profiles]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"profile ids must be non-empty and unique: {ids}")
    valid_temporal = {"forward", "backward", "instant"}
    valid_binding = {"target", "query_caller", "first_caller", "none"}
    valid_answer = {
        "azimuth_band", "instant_azimuth_band", "first_sound_side",
        "front_back",
        "coat_at_query", "distance_at_query", "time_band",
        "first_caller_coat", "event_count", "distance_change", "motion_state",
    }
    for profile in profiles:
        pid = profile["id"]
        if profile.get("temporal") not in valid_temporal:
            raise ValueError(
                f"{pid}: invalid temporal relation {profile.get('temporal')!r}")
        if profile.get("anchor_binding") not in valid_binding:
            raise ValueError(
                f"{pid}: invalid anchor_binding "
                f"{profile.get('anchor_binding')!r}")
        if profile.get("answer_kind", "azimuth_band") not in valid_answer:
            raise ValueError(
                f"{pid}: invalid answer_kind {profile.get('answer_kind')!r}")
        required = {"idle_choices"}
        kind = profile.get("answer_kind", "azimuth_band")
        if profile["temporal"] == "forward":
            required.add("anchor_frame")
            if kind == "azimuth_band":
                required.add("answer_bands_deg")
            if kind == "distance_change":
                required |= {"relation_frames", "answer_values"}
        elif profile["temporal"] == "backward":
            required |= {"anchor_frame", "query_frame", "answer_bands_deg"}
        else:
            required |= {"binding_frames"}
            if kind in ("instant_azimuth_band", "first_sound_side", "front_back"):
                required.add("answer_bands_deg")
            if kind == "event_count":
                required.add("answer_values")
        if kind == "distance_change":
            required |= {"relation_frames", "answer_values"}
        if kind == "motion_state":
            required |= {"motion_frames", "answer_values"}
        if "answer_bands_deg" in required and profile.get("answer_domain") is not None:
            required.remove("answer_bands_deg")
            required.add("answer_shape")
            if profile["answer_domain"] not in SS.ANSWER_DOMAINS:
                raise ValueError(f"{pid}: unsupported answer_domain {profile['answer_domain']!r}")
        if profile.get("query_domain") is not None:
            if profile["query_domain"] not in SS.ANSWER_DOMAINS:
                raise ValueError(f"{pid}: unsupported query_domain {profile['query_domain']!r}")
        if kind == "front_back":
            if profile.get("answer_domain") != "front_back":
                raise ValueError(f"{pid}: front_back answer_kind requires answer_domain 'front_back'")
            labels = SS.front_back_labels(profile)
            vocab_key = profile.get("vocab_key")
            if not isinstance(vocab_key, str) or not vocab_key.strip():
                raise ValueError(
                    f"{pid}: front_back requires a configured vocab_key for closed_set scoring")
            overrides = profile.get("query_geometry_by_answer")
            if not isinstance(overrides, dict) or any(label not in overrides for label in labels):
                raise ValueError(
                    f"{pid}: front_back requires query_geometry_by_answer for both labels")
            for label in labels:
                override = overrides[label]
                if not isinstance(override, dict):
                    raise ValueError(
                        f"{pid}: query geometry for {label!r} must be an object")
                domain = override.get("query_domain", override.get("domain"))
                if domain is not None and domain not in SS.ANSWER_DOMAINS:
                    raise ValueError(
                        f"{pid}: unsupported query domain {domain!r} for {label!r}")
        if profile.get("convention") is not None:
            AZ.canonical_convention(profile["convention"])
        missing = sorted(key for key in required if key not in profile)
        if missing:
            raise ValueError(f"{pid}: missing required profile fields {missing}")


def build_cell_plan(cells, profiles, pair_assets, params, seed, asset_context=None):
    """先分配答案与角色,再求解 —— 不是先采样后看落在哪。"""
    labels, asset_by_label, other_by_label = _pair_label_context(
        pair_assets, asset_context
    )
    plan = []
    per_profile = {}
    for profile in profiles:
        # 答案格按题型的答案空间分配:方位带题分带,时间带题分有序带对,
        # 外观题分目标外观 —— 一律**先分配再求解**。
        kind = profile.get("answer_kind", "azimuth_band")
        if kind in ("azimuth_band", "instant_azimuth_band", "first_sound_side", "front_back"):
            cellsets = [tuple(b) for b in profile["answer_bands_deg"]]
        elif kind in ("event_count", "distance_change", "motion_state"):
            cellsets = list(profile["answer_values"])
            if kind == "event_count":
                cellsets = [int(value) for value in cellsets]
        elif kind in ("time_band", "first_caller_coat"):
            # 直接分配**目标的答案带**(而不是带对):带对的第一分量
            # 决定目标答案带时,分布会被带对结构绑死。伙伴带随后选一个
            # 不同的带,谁先叫由 target_first 决定。
            n_bands = len(AP.card8_band_edges(params)) - 1
            cellsets = list(range(n_bands))
        else:
            cellsets = list(profile.get("answer_labels",
                                        ["black-and-white", "yellow"]))
        n = cells
        band_alloc = balanced(cellsets, n, seed, profile["id"], "band")
        anchor_alloc = [None] * n
        first_alloc = balanced([True, False], n, seed, profile["id"], "first")
        if kind in ("time_band", "first_caller_coat"):
            # 结构耦合(不是可选设计):事件按时间先后,**最早的带只能
            # 属于先叫者、最晚的带只能属于后叫者** —— 否则另一只要更早
            # 却又落在更早的带里,自相矛盾。两个因子独立抽样会造出不可能
            # 的组合(冒烟里正是它让 4 个单元失败)。所以极端带上的
            # target_first 由结构定死,只有中间带还能自由均衡。
            n_bands = len(cellsets)
            first_alloc = [
                True if band == 0 else False if band == n_bands - 1 else flag
                for band, flag in zip(band_alloc, first_alloc)]
        slots = ["source1", "source2"]
        coats = list(labels)
        if kind in ("azimuth_band", "instant_azimuth_band", "first_sound_side"):
            # Six cells cover the full slot x answer-band table once.  Coat
            # is then balanced within each slot so motion/audio slot cannot
            # deterministically recover appearance.
            if profile.get("family_id", profile["id"]) in {"card1F", "card1B"}:
                # Card1 exposes the audible anchor azimuth to A-only systems.
                # Allocate slot x anchor-band x query-band jointly before
                # search so every feasible conditional row can be reported
                # and balanced rather than discovered after sampling.
                slot_anchor_answer = balanced(
                    [(slot, anchor, answer)
                     for slot in slots
                     for anchor in cellsets
                     for answer in cellsets],
                    n, seed, profile["id"], "slot-anchor-answer")
                target_slots = [item[0] for item in slot_anchor_answer]
                anchor_alloc = [item[1] for item in slot_anchor_answer]
                band_alloc = [item[2] for item in slot_anchor_answer]
            else:
                slot_band = balanced(
                    [(slot, band) for slot in slots for band in cellsets], n,
                    seed, profile["id"], "slot-band")
                target_slots = [item[0] for item in slot_band]
                band_alloc = [item[1] for item in slot_band]
            target_coats = [None] * n
            for slot in slots:
                indices = [i for i, value in enumerate(target_slots)
                           if value == slot]
                allocated = balanced(coats, len(indices), seed,
                                     profile["id"], "coat-within-slot", slot)
                for index, coat in zip(indices, allocated):
                    target_coats[index] = coat
            answer_coats = list(target_coats)
        elif kind == "first_caller_coat":
            # The relevant shortcut table is first_slot x answer_color, not
            # two independently balanced marginals.
            first_answer = balanced_binary_joint(
                slots, coats, n, seed, profile["id"], "first-slot-answer")
            answer_coats = [item[1] for item in first_answer]
            target_slots, target_coats = [], []
            for (first_slot, answer_coat), target_first in zip(
                    first_answer, first_alloc):
                target_slot = (first_slot if target_first else
                               ("source2" if first_slot == "source1"
                                else "source1"))
                target_slots.append(target_slot)
                target_coats.append(
                    answer_coat if target_first else other_by_label[answer_coat])
        else:
            target_joint = balanced_binary_joint(
                slots, coats, n, seed, profile["id"], "slot-coat")
            target_slots = [item[0] for item in target_joint]
            target_coats = [item[1] for item in target_joint]
            answer_coats = list(target_coats)
        per_profile[profile["id"]] = {
            "band": band_alloc,
            "anchor_band": anchor_alloc,
            "target_first": first_alloc,
            "answer_coat": answer_coats,
            "target_coat": target_coats,
            "target_slot": target_slots,
            "target_moves_more": balanced(
                [True, False], n, seed, profile["id"], "motion-rank"),
        }
    for profile in profiles:
        alloc = per_profile[profile["id"]]
        for index in range(cells):
            entry = {
                "profile": profile,
                "cell_index": index,
                "answer_band": alloc["band"][index],
                "anchor_band": alloc["anchor_band"][index],
                "target_band": alloc["band"][index],
                "answer_value": alloc["band"][index],
                "target_first": bool(alloc["target_first"][index]),
                "answer_coat": alloc["answer_coat"][index],
                "target_slot": alloc["target_slot"][index],
                "target_moves_more": alloc["target_moves_more"][index],
            }
            target_coat = alloc["target_coat"][index]
            entry["target_coat"] = target_coat
            target_asset = asset_by_label[target_coat]
            other_asset = asset_by_label[other_by_label[target_coat]]
            entry["pair_assets"] = ((target_asset, other_asset)
                                    if entry["target_slot"] == "source1"
                                    else (other_asset, target_asset))
            plan.append(entry)
    return plan


FIRST_CALL_ANSWER_KINDS = ("time_band", "first_caller_coat")


def materialize_derived_params(params, profiles=None):
    """Replace stale card8 input text with the interval derived by this run.

    Card8's production path has used card8_band_edges since run02, but early
    external parameter files still carried run01's dead three-band field.
    Manifests must describe what execution actually used.  The derivation
    needs the explicit card8 scoring chain (T_FULL / T_HALF); a batch that
    contains a first-call profile fails closed without it, while a batch
    without any first-call profile records that no derivation happened.
    """
    effective = copy.deepcopy(params)
    needs_first_call = profiles is None or any(
        profile.get("answer_kind") in FIRST_CALL_ANSWER_KINDS
        for profile in profiles)
    try:
        effective["BANDS_CARD8"] = AP.card8_band_edges(effective)
    except AP.AudioProfileError as exc:
        if needs_first_call:
            raise
        effective["BANDS_CARD8_note"] = (
            "Not derived in this batch: no first-call profile requested and "
            f"the card8 scoring chain is incomplete ({exc}). The input "
            "BANDS_CARD8 text is left untouched and is not used.")
        return effective
    effective["BANDS_CARD8_note"] = (
        "Derived before generation by audio_profiles.card8_band_edges from "
        "clip/event/gap/first-min constraints; not an independent input. "
        "First calls must be strictly more than max(T_HALF, 2*T_FULL) apart; "
        "T_FULL follows the declared scoring parameters; the scoring block "
        "records its status and answer granularity.")
    effective["CARD8_FIRST_CALL_SCORING"] = AP.card8_scoring_params(effective)
    return effective


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-config", required=True, type=Path)
    parser.add_argument(
        "--historical-reproduction", action="store_true",
        help="compatibility flag for replaying a recorded historical batch")
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument(
        "--asset-policy",
        type=Path,
        default=REPO / "examples/qa/qa_v3_asset_policy_v1.json",
        help="per-request visual asset, sound-class and motion policy",
    )
    parser.add_argument("--pair-id", help="pair ID from --asset-policy")
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--cells", type=int, default=6)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--snapshot-content", default=(
        "/data/avengine_external/ue-package-stages/"
        "qa_v3_controlled_humans_20260904_v1/SpearSim/Content"))
    args = parser.parse_args(argv)
    # Repository tmp may be a declared symlink to external output storage.
    args.out_root = args.out_root.resolve()

    if args.out_root.exists():
        print(f"refusing to overwrite: {args.out_root}", file=sys.stderr)
        return 2
    scene_cfg = SS.read_scene_config(args.scene_config)
    profiles = json.loads(args.profiles.read_text())
    from qa_v3_request import read_qa_params
    params = read_qa_params(args.params)
    validate_profiles(profiles)
    clock = SS.validate_frame_clock(params, require_clip_seconds=True)
    for profile in profiles:
        kind = profile.get("answer_kind", "azimuth_band")
        if ((kind in {"instant_azimuth_band", "front_back"} and profile["temporal"] == "instant")
                or (kind == "azimuth_band" and profile["temporal"] == "backward")):
            query = (profile["query_frame"] if profile["temporal"] == "backward"
                     else profile["binding_frames"][0])
            SS.require_frame(query, clock["frame_count"], name="query_frame")
            window = profile_query_window(profile, params, query, clock["frame_rate_hz"])
            if window[1] > clock["clip_seconds"] + 1.0e-9:
                raise ValueError(f"{profile['id']}: query window {window} extends beyond the requested clip")
    scene = SS.load_scene(
        scene_cfg, frame_count=clock["frame_count"],
        frame_rate_hz=clock["frame_rate_hz"])
    # Validate all render facts before creating the fresh output directory.
    # Missing ground/map/transform is configuration failure, not a partially
    # realised candidate that should poison the no-clobber path.
    resolve_scene_render_context(scene)
    params = materialize_derived_params(params, profiles)
    SS.require_camera_clearance(scene, params)
    SS.require_route_synthesis(scene, params)
    profiles, answer_band_audit = SS.materialize_answer_domains(scene, params, profiles)
    # Every program policy value, the gain ceiling included, is read here so a
    # bad params file fails before the output directory exists.  Found on
    # 2026-09-03 by the review session's positive control: the gain check lived
    # inside realise_point, so an out-of-range value only surfaced after a full
    # geometry search and left a half-written candidate behind, and because the
    # output root now existed the obvious retry at the same path was refused.
    # The value is discarded; realise_point still reads it per candidate.
    program_request_fields(params)
    # Recorded scene inputs use the same completeness checks as the catalog.
    scene_config_path = SS.resolve_production_scene_config(
        args.scene_config,
        historical_reproduction=bool(
            getattr(args, "historical_reproduction", False)))
    SS.require_production_scene_config(scene_cfg, scene_config_path)
    base_request = json.loads(Path(scene_cfg["camera_base_request"]).read_text())
    registry = json.loads(
        (REPO / "examples/runtime/source_asset_runtime_profiles.json").read_text())
    by_id = {a["asset_id"]: a for a in registry["assets"]}
    try:
        asset_policy = load_asset_policy(args.asset_policy)
        asset_context = resolve_asset_policy(
            asset_policy,
            registry=registry,
            pair_id=args.pair_id,
            profiles=profiles,
        )
    except AssetPolicyError as error:
        raise ValueError(str(error)) from error
    pair = tuple(asset_context["asset_ids"])

    args.out_root.mkdir(parents=True)
    programs_dir = args.out_root / "programs"
    programs_dir.mkdir()
    ledger = SS.RejectionLedger()
    # 逐 profile 一本台账:总表看不出哪条约束在影响哪个题型
    per_profile_ledger = {p["id"]: SS.RejectionLedger() for p in profiles}
    cells = build_cell_plan(
        args.cells,
        profiles,
        pair,
        params,
        args.seed,
        asset_context=asset_context,
    )

    made, rejected, records = [], [], []
    for cell in cells:
        profile = cell["profile"]
        pid = f"{profile['id']}_{cell['cell_index'] + 1:03d}"
        rng = sha_rng(args.seed, pid)
        validator = instant_direction_candidate_validator(
            profile, cell, scene, params, by_id, args, asset_context)
        outcome = solve_for_profile(
            profile, cell, scene, params, rng, per_profile_ledger[profile["id"]],
            candidate_validator=validator)
        if isinstance(outcome, SS.Rejection):
            rejected.append({"point_id": pid, "reason": outcome.reason,
                             "detail": outcome.detail,
                             "cell": cell_allocation(cell)})
            continue
        try:
            record = realise_point(
                pid,
                cell,
                outcome,
                scene,
                base_request,
                params,
                by_id,
                args,
                programs_dir,
                rng,
                asset_context=asset_context,
            )
        except PredictedVisibilityRejection as exc:
            rejected.append({"point_id": pid,
                             "reason": "predicted_visibility_below_declared_minimum",
                             "detail": str(exc)[:240],
                             "cell": cell_allocation(cell)})
            continue
        except GenerationConstraintError as exc:
            rejected.append({"point_id": pid,
                             "reason": "generation_constraint_failed",
                             "detail": str(exc)[:240],
                             "cell": cell_allocation(cell)})
            continue
        made.append(pid)
        records.append(record)

    for sub in per_profile_ledger.values():
        ledger.absorb(sub)
    write_outputs(
        args,
        scene,
        scene_cfg,
        profiles,
        params,
        ledger,
        made,
        rejected,
        records,
        per_profile_ledger,
        cells=cells,
        answer_band_audit=answer_band_audit,
        asset_context=asset_context,
    )
    print(json.dumps({"out": str(args.out_root), "scene": scene.scene_id,
                      "geometry_candidates": len(made),
                      "cells_requested": len(cells),
                      "rejected": len(rejected),
                      "combinations_evaluated":
                          ledger.summary()["combinations_evaluated"],
                      "evidence_class": "geometry_candidate"},
                     ensure_ascii=False))
    return 0


def predicted_visibility_block(scene, params, profile, timeline, *, target_slot,
                               other_slot, camera_height_m, instants):
    """Predict, from the scene's clearance table, how visible each actor is
    along the final timeline, and evaluate the profile's declared visual
    requirements against the prediction.

    Returns (block, failures).  The block goes into the fact as evidence of
    what the solver expected; failures are the declarations in mode "reject"
    that the prediction misses.  Without a table nothing is predicted and the
    block says so.  Pixel truth remains the acceptance authority."""
    declarations = list(profile.get("visual_requirements") or [])
    if scene.clearance is None:
        return ({"status": "not_predicted",
                 "reason": "scene config declares no camera_clearance_table",
                 "declarations": declarations}, [])
    frames = sorted(timeline["frames"], key=lambda f: int(f["frame_index"]))
    camera = frames[0]["camera"]
    camera_xy = (float(camera["translation_ue_cm"][0]), float(camera["translation_ue_cm"][1]))
    yaw = float(camera["yaw_ue_deg"])
    ground_z = resolve_scene_render_context(scene)["ground_z_ue_cm"]
    roles = {"target": target_slot, "other": other_slot}
    routes = {}
    for slot in roles.values():
        routes[slot] = []
        for frame in frames:
            state = next(st for st in frame["actor_states"] if st["source_slot_id"] == slot)
            routes[slot].append((float(state["translation_ue_cm"][0]),
                                 float(state["translation_ue_cm"][1])))
    body = VP.body_from_params(params)
    edges = tuple(float(v) for v in params.get("PIXEL_TIER_VISIBLE_FRACTION_EDGES",
                                                 VP.TIER_EDGES_DEFAULT))
    prediction = VP.predict_timeline(
        scene.clearance, camera_xy_cm=camera_xy, camera_height_m=float(camera_height_m),
        camera_yaw_deg=yaw, hfov_deg=scene.hfov_deg, ground_z_cm=ground_z,
        routes_by_slot=routes, bodies_by_slot={slot: body for slot in routes}, edges=edges)
    rows_by_role = {role: prediction["slots"][slot]["per_frame"] for role, slot in roles.items()}
    by_frame = {role: {row["frame"]: row for row in rows} for role, rows in rows_by_role.items()}
    at_instants = {role: {name: {"frame": int(frame),
                                 "tier": by_frame[role][int(frame)]["tier"],
                                 "predicted_visible_fraction":
                                     by_frame[role][int(frame)]["predicted_visible_fraction"],
                                 "in_fov": by_frame[role][int(frame)]["in_fov"]}
                          for name, frame in instants.items()}
                   for role in roles}
    statistics = {role: VP.timeline_statistics(rows, instants=instants)
                  for role, rows in rows_by_role.items()}
    results, failures = [], []
    for declaration in declarations:
        role = declaration["referent"]
        if role not in roles:
            raise ValueError(f"visual requirement referent {role!r} is not target/other")
        minimum = float(declaration.get("min_predicted_visible_fraction", 0.0))
        mode = str(declaration.get("mode", "tier"))
        if mode not in ("tier", "reject"):
            raise ValueError(f"visual requirement mode {mode!r} must be tier or reject")
        for frame_ref in declaration["frames"]:
            frame = int(instants[frame_ref]) if isinstance(frame_ref, str) else int(frame_ref)
            row = by_frame[role][frame]
            fraction = row["predicted_visible_fraction"]
            satisfied = bool(row["in_fov"] and fraction is not None and fraction >= minimum)
            result = {"referent": role, "frame": frame, "frame_ref": frame_ref,
                      "min_predicted_visible_fraction": minimum, "mode": mode,
                      "predicted_visible_fraction": fraction, "in_fov": row["in_fov"],
                      "tier": row["tier"], "satisfied": satisfied}
            results.append(result)
            if mode == "reject" and not satisfied:
                failures.append(result)
    block = {
        "status": "predicted",
        "authority": "prediction_from_camera_clearance_table_not_pixel_truth",
        "table": scene.clearance.identity,
        "body": body,
        "tier_edges": list(edges),
        "camera_height_m": float(camera_height_m),
        "instants": {name: int(frame) for name, frame in instants.items()},
        "at_instants": at_instants,
        "statistics": statistics,
        "declarations": results,
        "reject_failures": len(failures),
        "per_frame": {role: [[row["frame"], int(row["in_fov"]),
                              row["predicted_visible_fraction"], row["tier"]]
                             for row in rows]
                      for role, rows in rows_by_role.items()},
    }
    return block, failures


def predicted_tier_distribution(records):
    """Predicted tier counts per profile at the named instants, plus how many
    candidates predict an actor that never shows in the clip."""
    out = {}
    for record in records:
        block = record.get("predicted_visibility") or {}
        if block.get("status") != "predicted":
            continue
        bucket = out.setdefault(record["profile_id"], {"at_instants": {}, "never_visible": {}})
        for role, per_instant in block["at_instants"].items():
            for name, row in per_instant.items():
                key = f"{role}@{name}"
                tiers = bucket["at_instants"].setdefault(key, {})
                tiers[row["tier"]] = tiers.get(row["tier"], 0) + 1
        for role, stats in block["statistics"].items():
            if stats.get("never_visible"):
                bucket["never_visible"][role] = bucket["never_visible"].get(role, 0) + 1
    return out


def build_point_timeline_geometry(
    pid, profile, plan, scene, params, by_id, selection_path,
    slot_asset, slot_context_data, target_slot, *, asset_context=None,
):
    """Use the runtime timeline builder for both candidate search and authoring."""
    other_slot = "source2" if target_slot == "source1" else "source1"
    clock = SS.validate_frame_clock(params, require_clip_seconds=True)
    render_context = resolve_scene_render_context(scene)
    camera_height_m = float(plan.camera_height_m
                            if plan.camera_height_m is not None
                            else scene.camera_height_m)
    camera_ue_cm = [plan.camera_xy[0], plan.camera_xy[1],
                    render_context["ground_z_ue_cm"] + camera_height_m * 100.0]
    # 时间线:相机与折线都来自求解结果
    if (
        answer_depends_on_source_displacement(profile)
        and by_id[slot_asset[target_slot]].get("entity_class") == "rigid_object"
    ):
        raise GenerationConstraintError(
            f"{pid}: a displacement-dependent answer cannot use a rigid target "
            f"asset {slot_asset[target_slot]!r}"
        )
    direct_routes = bool(profile.get("use_solved_routes_directly", False))
    base_route = (
        plan.target_route.samples_xy if direct_routes
        else plan.base_route.samples_xy)
    other_route = plan.other_route.samples_xy
    z = render_context["ground_z_ue_cm"]
    routes = {
        target_slot: (base_route[0], base_route[-1], base_route),
        other_slot: (other_route[0], other_route[-1], other_route),
    }
    route_samples_by_slot = {}
    for slot, route in routes.items():
        samples = [
            [float(point[0]), float(point[1]), float(z)]
            for point in route[2]
        ]
        if slot_context_data["motion_by_slot"].get(slot) == "must_be_still":
            samples = [list(samples[0]) for _ in samples]
        route_samples_by_slot[slot] = samples
    timeline = build_current_n_actor_visual_timeline(
        actor_selection_path=selection_path,
        source_asset_registry_path=(
            REPO / "examples/runtime/source_asset_runtime_profiles.json"),
        camera_position_ue_cm=camera_ue_cm,
        camera_yaw_deg=plan.camera_ue_yaw_deg,
        routes_by_slot_ue_cm=route_samples_by_slot,
        native_map=render_context["native_map"],
        room_profile_id=render_context["room_profile_id"],
        hfov_degrees=scene.hfov_deg,
        frame_count=clock["frame_count"],
        frame_rate_hz=clock["frame_rate_hz"],
        ticks_per_frame=clock.get("ticks_per_frame"),
    )
    authored = copy.deepcopy(timeline)
    moving_routes = {
        slot: [(point[0], point[1]) for point in samples]
        for slot, samples in route_samples_by_slot.items()
        if slot_context_data["motion_by_slot"].get(slot) != "must_be_still"
    }
    facing_by_slot = (
        apply_asset_facing_policy(
            timeline,
            camera_position_ue_cm=camera_ue_cm,
            assets_by_slot=slot_asset,
            asset_context=asset_context,
        )
        if asset_context is not None
        else {}
    )
    if direct_routes and moving_routes:
        timeline = transform_to_solved_routes(timeline, moving_routes)
    elif (
        plan.idle_frames
        and slot_context_data["motion_by_slot"].get(target_slot)
        != "must_be_still"
    ):
        timeline = transform_idle_then_walk(timeline, target_slot,
                                            plan.idle_frames)
    # Audio truth follows the registry-bound emitter child, while the actor
    # root remains the visual/navigation coordinate. The helper uses the
    # authored timeline yaw and scale rather than breed-specific constants.
    emitter_paths_by_slot, emitter_metadata = materialize_emitter_paths(
        timeline,
        {"assets": list(by_id.values())},
        slot_asset,
    )
    render_clock = timeline.get("render", {})
    timeline_rate = float(render_clock.get("frame_rate_hz"))
    parameter_rate = float(params["VIDEO_FPS"])
    if not math.isclose(timeline_rate, parameter_rate, rel_tol=0.0, abs_tol=1.0e-9):
        raise GenerationConstraintError(
            f"{pid}: timeline frame rate {timeline_rate} differs from VIDEO_FPS "
            f"{parameter_rate}")
    timeline["emitter_bindings"] = {
        "authority": "source_asset_registry_default_emitter_anchor",
        "coordinate_space": "timeline_ue_cm",
        "root_for_visual_navigation_only": True,
        "frame_clock": {
            "frame_count": len(timeline["frames"]),
            "frame_rate_hz": timeline_rate,
            "ticks_per_frame": render_clock.get("ticks_per_frame"),
        },
        "by_slot": emitter_metadata,
    }
    return timeline, authored, facing_by_slot, emitter_paths_by_slot, emitter_metadata



def instant_direction_candidate_validator(
    profile, cell, scene, params, by_id, args, asset_context,
):
    """Validate candidate emitters inside the solver's existing attempt budget."""
    kind = profile.get("answer_kind")
    if profile.get("temporal") != "instant" or kind not in {
            "instant_azimuth_band", "front_back"}:
        return None
    pid = f"{profile['id']}_{cell['cell_index'] + 1:03d}"
    selection = _selection_doc(*cell["pair_assets"], by_id, args.snapshot_content)
    input_dir = args.out_root / "candidate_inputs" / pid
    input_dir.mkdir(parents=True)
    selection_path = input_dir / "actor_selection.json"
    with selection_path.open("x", encoding="utf-8") as stream:
        json.dump(selection, stream, ensure_ascii=False, indent=2)
    slots = {actor["source_slot_id"]: actor["asset_id"] for actor in selection["actors"]}
    target = cell["target_slot"]
    other = "source2" if target == "source1" else "source1"
    context = slot_context(asset_context, assets_by_slot=slots, target_slot=target)
    labels = context["labels_by_slot"]

    def validate(plan):
        try:
            timeline, _, _, emitters, _ = build_point_timeline_geometry(
                pid, profile, plan, scene, params, by_id, selection_path,
                slots, context, target, asset_context=asset_context)
            truth = recompute_emitter_azimuth(timeline, target, plan.query_frame, emitters)
            answer = build_answer(
                kind, profile, cell, timeline, None, [], target, other, labels,
                truth, plan.query_frame, params, asset_context=asset_context,
                slot_context_data=context, emitter_paths_by_slot=emitters)
            _, _, ga_answer, _, _ = build_gatea_answer(
                kind, profile, cell, timeline, None, [], target, other, labels,
                plan.query_frame, params, asset_context=asset_context,
                slot_context_data=context, emitter_paths_by_slot=emitters)
            audit_gatea_answers(profile, answer, ga_answer, params)
        except GenerationConstraintError as error:
            return SS.Rejection("emitter_query_geometry_failed", str(error))
        return None

    return validate


def realise_point(
    pid,
    cell,
    plan,
    scene,
    base_request,
    params,
    by_id,
    args,
    programs_dir,
    rng,
    asset_context=None,
):
    profile = cell["profile"]
    clock = SS.validate_frame_clock(params, require_clip_seconds=True)
    family_id = profile.get("family_id", profile["id"])
    pdir = args.out_root / pid
    pdir.mkdir()
    target_slot = cell["target_slot"]
    other_slot = "source2" if target_slot == "source1" else "source1"
    assets = cell["pair_assets"]
    selection = _selection_doc(assets[0], assets[1], by_id,
                               args.snapshot_content)
    (pdir / "actor_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2))
    slot_asset = {
        a["source_slot_id"]: a["asset_id"] for a in selection["actors"]
    }
    if asset_context is None:
        slot_coat = {s: COAT_WORDS[a] for s, a in slot_asset.items()}
        slot_context_data = {
            "labels_by_slot": dict(slot_coat),
            "referent_phrases_by_slot": {
                slot: f"the {label} dog"
                for slot, label in slot_coat.items()
            },
            "sound_action_phrases_by_slot": {
                slot: "bark" for slot in slot_asset
            },
            "allowed_sound_class_ids_by_slot": {
                slot: ["animal_vocalization"] for slot in slot_asset
            },
            "motion_by_slot": {
                slot: "must_move" for slot in slot_asset
            },
        }
    else:
        slot_context_data = slot_context(
            asset_context,
            assets_by_slot=slot_asset,
            target_slot=target_slot,
        )
        slot_coat = dict(slot_context_data["labels_by_slot"])
    pair_kind = _pair_kind(params)
    assert_assets_match_pair_kind(params, assets, asset_context)
    clip_source = clip_source_from_params(params, rng, pair_kind=pair_kind)
    if clip_source is not None:
        clip_source = clip_source.bind_distinct_roles((AP.TARGET, AP.OTHER))

    # 相机与听者:同一份姿态结果
    render_context = resolve_scene_render_context(scene)
    camera_height_m = float(plan.camera_height_m
                            if plan.camera_height_m is not None
                            else scene.camera_height_m)
    camera_ue_cm = [plan.camera_xy[0], plan.camera_xy[1],
                    render_context["ground_z_ue_cm"]
                    + camera_height_m * 100.0]
    camera_world_m = render_context["world_transform"](camera_ue_cm)
    m1_request = apply_camera_listener_pose_ue(
        base_request, request_id=f"qa_v3_{pid}", position_m=camera_world_m,
        ue_yaw_degrees=plan.camera_ue_yaw_deg,
        horizontal_fov_deg=scene.hfov_deg)
    (pdir / "m1_capture_request.json").write_text(
        json.dumps(m1_request, ensure_ascii=False, indent=2))

    # 题型专用音频调度:语义角色 → 槽位
    if family_id == "card1F":
        schedule = AP.schedule_forward_anchor(
            rng, params=params, anchor_frame=plan.anchor_frame, clip_source=clip_source)
    elif family_id == "card1B":
        schedule = AP.schedule_backward_anchor(
            rng, params=params, anchor_frame=plan.anchor_frame,
            query_frame=plan.query_frame, clip_source=clip_source)
    elif family_id == "card5R":
        schedule = AP.schedule_forward_anchor(
            rng, params=params, anchor_frame=plan.anchor_frame, clip_source=clip_source)
    elif family_id == "card5":
        schedule = AP.schedule_first_sound_at_frame(
            rng, params=params, query_frame=plan.anchor_frame, clip_source=clip_source)
    elif family_id in ("card6", "card6R"):
        schedule = AP.schedule_second_sound_at_frame(
            rng, params=params,
            query_frame=int(profile["second_sound_frame"]), clip_source=clip_source)
    elif family_id == "card10":
        schedule = AP.schedule_first_sound_at_frame(
            rng, params=params, query_frame=plan.anchor_frame, clip_source=clip_source)
    elif profile.get("answer_kind") == "event_count":
        schedule = AP.schedule_event_count(
            rng, params=params, event_count=int(cell["answer_value"]), clip_source=clip_source)
    elif profile.get("answer_kind") == "distance_at_query":
        schedule = AP.schedule_event_count(
            rng, params=params,
            event_count=int(profile["audio_event_count"]), clip_source=clip_source)
    elif profile.get("answer_kind") == "first_sound_side":
        schedule = AP.schedule_first_sound_at_frame(
            rng, params=params, query_frame=plan.query_frame, clip_source=clip_source)
    elif profile.get("answer_kind") in ("time_band", "first_caller_coat"):
        # ⑧⑨ 都要"两只都有首叫、且可分辨";目标的答案带先定,伙伴带
        # 由 target_first 决定在它之前还是之后,再交给调度器。
        edges = AP.card8_band_edges(params)
        n_bands = len(edges) - 1
        target_band = int(cell["target_band"])
        if cell["target_first"]:
            options = [b for b in range(target_band + 1, n_bands)]
            pair = (target_band, options[int(rng.integers(len(options)))]) \
                if options else None
        else:
            options = [b for b in range(0, target_band)]
            pair = (options[int(rng.integers(len(options)))], target_band) \
                if options else None
        if pair is None:
            raise ValueError(
                f"target band {target_band} cannot be "
                f"{'first' if cell['target_first'] else 'second'} caller: "
                "no partner band on that side")
        schedule = AP.schedule_first_call_bands(
            rng, params=params, target_bands=pair, band_edges=edges,
            first_caller_role=(AP.TARGET if cell["target_first"] else AP.OTHER), clip_source=clip_source)
    else:
        schedule = AP.schedule_exactly_one_calling(
            rng, params=params, query_frame=plan.query_frame, clip_source=clip_source)
    main_role_to_slot = {AP.TARGET: target_slot, AP.OTHER: other_slot}
    gatea_role_to_slot = {AP.TARGET: other_slot, AP.OTHER: target_slot}
    slot_events = schedule.bind(main_role_to_slot)
    gatea_slot_events = schedule.bind(gatea_role_to_slot)
    endpoint_registry_path = pdir / "source_endpoints.json"
    _, endpoint_records = build_endpoint_registry(
        selection,
        by_id,
        endpoint_registry_path,
        allowed_sound_classes_by_slot=slot_context_data[
            "allowed_sound_class_ids_by_slot"
        ],
        selection_path=pdir / "actor_selection.json",
    )
    slot_endpoints = endpoint_ids_by_slot(selection, endpoint_records)
    request = {
        "pair_kind": pair_kind,
        "point_id": pid,
        "slot_endpoints": slot_endpoints,
        **program_request_fields(params),
    }
    if clip_source is None:
        request.update(dry_canvas_fields(params))
        program_events = slot_events
        gatea_program_events = gatea_slot_events
    else:
        program_events = schedule.program_events(main_role_to_slot)
        gatea_program_events = schedule.program_events(gatea_role_to_slot)
    program = build_program(request, program_events, revision="v1")
    gatea_program = build_program(
        request, gatea_program_events, revision="gateA_v1")
    validate_m6_audio_program(program)
    validate_m6_audio_program(gatea_program)
    (programs_dir / f"{program['program_id']}.json").write_text(
        json.dumps(program, ensure_ascii=False, indent=1))
    (programs_dir / f"{gatea_program['program_id']}.json").write_text(
        json.dumps(gatea_program, ensure_ascii=False, indent=1))

    timeline, authored, facing_by_slot, emitter_paths_by_slot, emitter_metadata = (
        build_point_timeline_geometry(
            pid, profile, plan, scene, params, by_id,
            pdir / "actor_selection.json", slot_asset, slot_context_data,
            target_slot, asset_context=asset_context))
    (pdir / "timeline_authored.json").write_text(
        json.dumps(authored, indent=2, sort_keys=True) + "\n")
    (pdir / "timeline.json").write_text(
        json.dumps(timeline, ensure_ascii=False, indent=1))

    # Recompute audio truth from emitter paths after final timeline authoring.
    query_frame = plan.query_frame
    answer_kind = profile.get("answer_kind", "azimuth_band")
    root_truth_deg = recompute_azimuth(timeline, target_slot, query_frame)
    root_other_deg = recompute_azimuth(timeline, other_slot, query_frame)
    truth_deg = recompute_emitter_azimuth(
        timeline, target_slot, query_frame, emitter_paths_by_slot)
    other_deg = recompute_emitter_azimuth(
        timeline, other_slot, query_frame, emitter_paths_by_slot)
    answer = build_answer(
        answer_kind,
        profile,
        cell,
        timeline,
        schedule,
        slot_events,
        target_slot,
        other_slot,
        slot_coat,
        truth_deg,
        query_frame,
        params,
        asset_context=asset_context,
        slot_context_data=slot_context_data,
        emitter_paths_by_slot=emitter_paths_by_slot,
    )

    def displacement_cm(slot):
        states = []
        for frame in timeline["frames"]:
            states.append(next(state for state in frame["actor_states"]
                               if state["source_slot_id"] == slot))
        start, end = states[0]["translation_ue_cm"], states[-1]["translation_ue_cm"]
        return float(np.linalg.norm(np.asarray(end) - np.asarray(start)))

    motion = {
        "source1_route_id": (plan.target_route.route_id
                             if target_slot == "source1"
                             else plan.other_route.route_id),
        "source2_route_id": (plan.target_route.route_id
                             if target_slot == "source2"
                             else plan.other_route.route_id),
        "source1_displacement_cm": displacement_cm("source1"),
        "source2_displacement_cm": displacement_cm("source2"),
        # 路线来源:库路线记 route_id,合成路线记完整设计(机位、关键帧方位与
        # 距离、速度、边距、最小净空)。来源不改变任何约束,只让事实可追溯。
        "source1_route_provenance": (plan.base_route.source_record
                                     if target_slot == "source1"
                                     else plan.other_route.source_record),
        "source2_route_provenance": (plan.base_route.source_record
                                     if target_slot == "source2"
                                     else plan.other_route.source_record),
        "route_sources": plan.checks.get("route_sources"),
    }
    motion["both_roles_move"] = (
        motion["source1_displacement_cm"] > 0.0
        and motion["source2_displacement_cm"] > 0.0)
    motion_constraints = {
        "target": slot_context_data["motion_by_slot"].get(target_slot, "any"),
        "other": slot_context_data["motion_by_slot"].get(other_slot, "any"),
    }
    motion["constraints"] = motion_constraints
    for role, slot in (("target", target_slot), ("other", other_slot)):
        displacement = motion[f"{slot}_displacement_cm"]
        constraint = motion_constraints[role]
        if constraint == "must_move" and displacement <= 1.0e-6:
            raise GenerationConstraintError(
                f"{pid}: {role} asset is required to move but displacement is "
                f"{displacement:.3f} cm")
        if constraint == "must_be_still" and displacement > 1.0e-6:
            raise GenerationConstraintError(
                f"{pid}: {role} asset is required to stay still but displacement is "
                f"{displacement:.3f} cm")
    observed_target_moves_more = (
        motion[f"{target_slot}_displacement_cm"]
        > motion[f"{other_slot}_displacement_cm"])
    motion["target_moves_more"] = observed_target_moves_more
    default_motion_rank = (
        True
        if asset_context is None
        else bool(asset_context["answer_depends_on_source_displacement"])
    )
    enforce_motion_rank = bool(
        profile.get("enforce_target_moves_more", default_motion_rank))
    motion["allocated_target_moves_more"] = (
        bool(cell["target_moves_more"]) if enforce_motion_rank else None)
    if asset_context is None and not motion["both_roles_move"]:
        raise GenerationConstraintError(
            f"{pid}: dual-motion profile produced a static role: {motion}")
    if (enforce_motion_rank
            and observed_target_moves_more != bool(cell["target_moves_more"])):
        raise GenerationConstraintError(
            f"{pid}: realised motion rank disagrees with allocation: {motion}")

    fact = {
        "schema": "qa_v3_fact_record_v2",
        "variant": "main",
        "point_id": pid, "scene_id": scene.scene_id,
        "profile_id": profile["id"],
        "evidence_class": "geometry_candidate",
        "temporal_relation": (
            "simultaneous_binding" if profile["temporal"] == "instant"
            else "anchor_before_query" if profile["temporal"] == "forward"
            else "anchor_after_query"),
        "anchor_frame": plan.anchor_frame, "query_frame": query_frame,
        "target_slot": target_slot, "target_coat": slot_coat[target_slot],
        "target_referent_label": slot_coat[target_slot],
        "slot_coat": slot_coat,
        "slot_assets": slot_asset,
        "slot_referent_phrases": slot_context_data[
            "referent_phrases_by_slot"
        ],
        "facing_by_slot": facing_by_slot,
        "asset_policy": (
            {
                "policy_id": asset_context["policy_id"],
                "pair_id": asset_context["pair_id"],
                "family_rule": asset_context["family_rule"],
                "pair_kind": asset_context["pair_kind"],
                "motion_by_slot": slot_context_data["motion_by_slot"],
                "answer_depends_on_source_displacement": asset_context[
                    "answer_depends_on_source_displacement"
                ],
            }
            if asset_context is not None
            else None
        ),
        "frame_clock": clock,
        "camera": {"ue_cm": camera_ue_cm,
                   "ue_yaw_deg": plan.camera_ue_yaw_deg,
                   "height_m": camera_height_m,
                   "scene_camera_height_m": float(scene.camera_height_m),
                   "clearance": plan.camera_clearance,
                   "listener_from_same_pose_result": True},
        "emitter": {
            "authority": "source_asset_registry_default_emitter_anchor",
            "root_role": "visual_navigation_only",
            "audio_role": "RLR_source_position_and_azimuth_truth",
            "frame_clock": timeline["emitter_bindings"]["frame_clock"],
            "query_frame": int(query_frame),
            "by_slot": emitter_metadata,
        },
        "room": {
            "native_map": timeline["room"]["map_path"],
            "room_profile_id": timeline["room"]["room_profile_id"],
            "world_transform": render_context["world_transform_id"],
            "ground_z_ue_cm": render_context["ground_z_ue_cm"],
            "floor_reference": render_context["floor_reference"],
        },
        "motion": motion,
        "answer_kind": answer_kind,
        "answer_domain": profile.get("answer_domain"),
        "query_domain": _resolved_query_geometry(profile, cell).get("query_domain"),
        "query_requires_visibility": _resolved_query_geometry(profile, cell).get(
            "query_requires_visibility"),
        "secondary_anchor_bound_deg": _resolved_query_geometry(
            profile, cell).get("secondary_anchor_bound_deg"),
        "secondary_query_bound_deg": _resolved_query_geometry(
            profile, cell).get("secondary_query_bound_deg"),
        "answer_shape": copy.deepcopy(profile.get("answer_shape")),
        "convention": (
            answer.get("open", {}).get("convention")
            or answer.get("truth", {}).get("convention")
        ),
        "direction_evidence": profile.get("direction_evidence"),
        "vocab_key": answer.get("open", {}).get("vocab_key"),
        "vocab_path": answer.get("open", {}).get("vocab_path"),
        "truth": dict(answer["truth"],
                      query_azimuth_deg_engine_frame=round(truth_deg, 3),
                      other_slot_azimuth_deg_engine_frame=round(
                          other_deg, 3),
                      query_azimuth_deg_engine_frame_actor_root=round(
                          root_truth_deg, 3),
                      other_slot_azimuth_deg_engine_frame_actor_root=round(
                          root_other_deg, 3),
                      azimuth_truth_source="runtime_emitter_anchor",
                      recomputed_after_camera_pose=True),
        "answer_forms": answer_forms_from_params(params),
        "mcq": answer["mcq"],
        "open": answer["open"],
        "source_endpoint_registry": "source_endpoints.json",
        "audio": {"program_id": program["program_id"],
                  "source_endpoint_registry": "source_endpoints.json",
                  "slot_endpoints": slot_endpoints,
                  "anchor_role": schedule.anchor.role,
                  "anchor_slot": main_role_to_slot[schedule.anchor.role],
                  "declared": schedule.declared,
                  "events": [{"role": e.role, "slot": slot,
                              "purpose": e.purpose,
                              "start_sample": e.start_sample}
                             for e, (slot, _) in zip(
                                 schedule.events, slot_events)]},
        "target_first": (bool(cell["target_first"])
                         if answer_kind in ("time_band",
                                            "first_caller_coat") else None),
        "first_caller_slot": answer.get("first_caller_slot"),
        "generation_checks": copy.deepcopy(plan.checks),
        "search_attempts": plan.checks.get("search_attempts"),
        "line_of_sight_screened": plan.checks.get("line_of_sight_screened"),
        "status": "research_candidate", "qualification_claim": False,
    }
    # 事实与 program 逐条一致性:锚定角色必须绑到题目目标槽位
    # 锚定与题目目标的关系**因题型而异**,按 profile 声明检查:
    #   错时族的锚就是身份锚,必须绑到目标;
    #   ⑦ 的目标是查询时刻的发声者;⑨ 的目标由"谁先叫"决定。
    # 早先这里把错时族的假设硬套到 ⑦⑨ 上,把合格点全判失败了。
    if slot_coat[target_slot] != cell["target_coat"]:
        raise ValueError(
            f"target coat {slot_coat[target_slot]} does not match the "
            f"allocated {cell['target_coat']}: the slot-asset binding drifted")
    main_binding_check = validate_anchor_binding(
        profile, schedule, slot_events, target_slot=target_slot,
        query_frame=plan.query_frame, answer=answer)
    gatea_target_slot, gatea_other_slot, gatea_answer, gatea_truth_deg, \
        gatea_other_deg = build_gatea_answer(
            answer_kind, profile, cell, timeline, schedule,
            gatea_slot_events,
            target_slot,
            other_slot,
            slot_coat,
            query_frame,
            params,
            asset_context=asset_context,
            slot_context_data=slot_context_data,
            emitter_paths_by_slot=emitter_paths_by_slot,
        )
    gatea_checks = audit_gatea_pair(
        profile, program, gatea_program, answer, gatea_answer, params)
    gatea_binding_check = validate_anchor_binding(
        profile, schedule, gatea_slot_events,
        target_slot=gatea_target_slot, query_frame=plan.query_frame,
        answer=gatea_answer)
    # 最终时间线才是验收权威:错时方位题在这里重算主题与 Gate A 指代者的
    # 锚角/查询角并 fail-closed;像素可答性阈值随事实记录一起显式声明。
    if answer_kind == "azimuth_band" and profile["temporal"] in (
            "forward", "backward"):
        fact["planned_generation_checks"] = dict(
            copy.deepcopy(plan.checks),
            provenance="solver_plan_before_timeline_authoring",
            role="planning_values_only_not_acceptance")
        fact["realized_generation_checks"] = realized_cross_time_checks(
            timeline, profile=profile, cell=cell, target_slot=target_slot,
            other_slot=other_slot, anchor_frame=plan.anchor_frame,
            query_frame=query_frame, params=params, plan_checks=plan.checks,
            query_window_s=answer.get("truth", {}).get("query_window_seconds"),
            emitter_paths_by_slot=emitter_paths_by_slot)
        fact["acceptance_authority"] = "realized_generation_checks"
        if family_id in ("card1F", "card1B"):
            fact["pixel_acceptance"] = card1_pixel_acceptance_block(
                params, target_slot=target_slot, other_slot=other_slot,
                anchor_frame=plan.anchor_frame, query_frame=query_frame)
    predicted, visibility_failures = predicted_visibility_block(
        scene, params, profile, timeline, target_slot=target_slot,
        other_slot=other_slot, camera_height_m=camera_height_m,
        instants={"anchor": int(plan.anchor_frame), "query": int(query_frame)})
    fact["predicted_visibility"] = predicted
    if visibility_failures:
        raise PredictedVisibilityRejection("; ".join(
            f"{f['referent']}@{f['frame']}: predicted "
            f"{f['predicted_visible_fraction']} < {f['min_predicted_visible_fraction']}"
            for f in visibility_failures))
    gatea_fact = copy.deepcopy(fact)
    gatea_fact.update({
        "variant": "gateA",
        "gatea_of": pid,
        "target_slot": gatea_target_slot,
        "target_coat": slot_coat[gatea_target_slot],
        "query_domain": gatea_answer.get("open", {}).get("query_domain"),
        "query_requires_visibility": gatea_answer.get("open", {}).get(
            "query_requires_visibility"),
        "truth": dict(
            gatea_answer["truth"],
            query_azimuth_deg_engine_frame=round(gatea_truth_deg, 3),
            other_slot_azimuth_deg_engine_frame=round(gatea_other_deg, 3),
            recomputed_after_camera_pose=True),
        "mcq": gatea_answer["mcq"],
        "open": gatea_answer["open"],
        "target_first": (
            not bool(cell["target_first"])
            if answer_kind in ("time_band", "first_caller_coat")
            else None),
        "first_caller_slot": gatea_answer.get("first_caller_slot"),
        "audio": {
            "program_id": gatea_program["program_id"],
            "anchor_role": schedule.anchor.role,
            "anchor_slot": gatea_role_to_slot[schedule.anchor.role],
            "declared": schedule.declared,
            "events": [
                {"role": e.role, "slot": slot, "purpose": e.purpose,
                 "start_sample": e.start_sample}
                for e, (slot, _) in zip(schedule.events, gatea_slot_events)],
        },
        "gatea_checks": gatea_checks,
        "anchor_binding_check": gatea_binding_check,
    })
    fact["anchor_binding_check"] = main_binding_check
    fact["gatea"] = {
        "program_id": gatea_program["program_id"],
        "fact_record": "fact_record_gateA.json",
        "checks": gatea_checks,
    }
    (pdir / "fact_record_gateA.json").write_text(
        json.dumps(gatea_fact, ensure_ascii=False, indent=2))
    (pdir / "fact_record.json").write_text(
        json.dumps(fact, ensure_ascii=False, indent=2))
    return fact


def build_gatea_answer(
    kind,
    profile,
    cell,
    timeline,
    schedule,
    gatea_slot_events,
    target_slot,
    other_slot,
    slot_coat,
    query_frame,
    params,
    *,
    asset_context=None,
    slot_context_data=None,
    emitter_paths_by_slot=None,
):
    """Derive Gate A gold from the final timeline and bound emitters.

    Cross-time and query-caller cards select a different visual actor after
    the audio slot swap.  Card8 keeps the text-selected visual actor but
    inherits the other role's onset.  Card9 has no fixed visual target; the
    first-caller relation simply reverses.
    """
    gatea_cell = dict(cell)
    if kind in ("azimuth_band", "instant_azimuth_band", "first_sound_side", "front_back"):
        gatea_target_slot, gatea_other_slot = other_slot, target_slot
        gatea_truth_deg = recompute_emitter_azimuth(
            timeline, gatea_target_slot, query_frame, emitter_paths_by_slot)
        bands = list(profile["answer_bands_deg"])
        gatea_band = next((band for band in bands
                           if SS.band_contains(gatea_truth_deg, band)), None)
        if gatea_band is None:
            raise GenerationConstraintError(
                f"Gate A angular truth {gatea_truth_deg:.2f} is outside all "
                "declared MCQ bands")
        gatea_cell["answer_band"] = gatea_band
    elif kind == "coat_at_query":
        gatea_target_slot, gatea_other_slot = other_slot, target_slot
        gatea_truth_deg = recompute_emitter_azimuth(
            timeline, gatea_target_slot, query_frame, emitter_paths_by_slot)
    elif kind == "time_band":
        gatea_target_slot, gatea_other_slot = target_slot, other_slot
        gatea_truth_deg = recompute_emitter_azimuth(
            timeline, gatea_target_slot, query_frame, emitter_paths_by_slot)
        firsts = {}
        for slot, start in gatea_slot_events:
            firsts.setdefault(slot, start / _sample_rate_hz(params))
        gatea_band = band_of(firsts[gatea_target_slot],
                             AP.card8_band_edges(params))
        if gatea_band is None:
            raise ValueError("Gate A card8 onset is outside the declared bands")
        gatea_cell["target_band"] = gatea_band
    elif kind == "first_caller_coat":
        gatea_target_slot, gatea_other_slot = target_slot, other_slot
        gatea_truth_deg = recompute_emitter_azimuth(
            timeline, gatea_target_slot, query_frame, emitter_paths_by_slot)
        gatea_cell["target_first"] = not bool(cell["target_first"])
    elif kind == "motion_state":
        gatea_target_slot, gatea_other_slot = other_slot, target_slot
        gatea_truth_deg = recompute_emitter_azimuth(
            timeline, gatea_target_slot, query_frame, emitter_paths_by_slot)
        gatea_cell["answer_value"] = (
            "still" if cell["answer_value"] == "moving" else "moving")
    elif kind == "distance_change":
        gatea_target_slot, gatea_other_slot = other_slot, target_slot
        gatea_truth_deg = recompute_emitter_azimuth(
            timeline, gatea_target_slot, query_frame, emitter_paths_by_slot)
        gatea_cell["answer_value"] = (
            "farther" if cell["answer_value"] == "closer" else "closer")
    elif kind in ("event_count", "distance_at_query"):
        gatea_target_slot, gatea_other_slot = target_slot, other_slot
        gatea_truth_deg = recompute_emitter_azimuth(
            timeline, gatea_target_slot, query_frame, emitter_paths_by_slot)
    else:
        raise ValueError(f"no Gate A answer derivation for {kind!r}")
    if kind == "front_back":
        gatea_cell["_resolved_query_geometry"] = profile_query_geometry(
            profile, gatea_cell)
    gatea_other_deg = recompute_emitter_azimuth(
        timeline, gatea_other_slot, query_frame, emitter_paths_by_slot)
    gatea_answer = build_answer(
        kind,
        profile,
        gatea_cell,
        timeline,
        schedule,
        gatea_slot_events,
        gatea_target_slot,
        gatea_other_slot,
        slot_coat,
        gatea_truth_deg,
        query_frame,
        params,
        asset_context=asset_context,
        slot_context_data=slot_context_data,
        emitter_paths_by_slot=emitter_paths_by_slot,
    )
    return (gatea_target_slot, gatea_other_slot, gatea_answer,
            gatea_truth_deg, gatea_other_deg)


def _resolved_query_geometry(profile, cell) -> dict:
    value = cell.get("_resolved_query_geometry")
    if isinstance(value, dict):
        return dict(value)
    return {
        "query_domain": profile.get("query_domain"),
        "answer_domain": profile.get("answer_domain"),
        "query_bound_deg": profile.get("query_bound_deg"),
        "query_requires_visibility": profile.get("query_requires_visibility"),
        "secondary_query_bound_deg": profile.get("secondary_query_bound_deg"),
    }


def _profile_convention(profile) -> str:
    return AZ.canonical_convention(profile.get("convention", AZ.CONVENTION))


def _display_angle(value) -> float:
    """Fold an endpoint while retaining a signed 180-degree boundary."""
    value = float(value)
    folded = (value + 180.0) % 360.0 - 180.0
    if math.isclose(folded, -180.0, abs_tol=1e-9):
        return -180.0 if value < 0.0 else 180.0
    return folded


def _arc_option_label(engine_band, convention) -> str:
    engine_arc = SS.band_to_arc(engine_band)
    converted = AZ.to_convention_arc(engine_arc, convention)
    if converted.width_deg >= 360.0:
        return "full circle"
    if converted.wraps:
        return (f"{converted.start_deg:g} -> {converted.end_deg:g} "
                f"(sweep {converted.sweep_deg:g} deg)")
    if isinstance(engine_band, (list, tuple)) and len(engine_band) == 2:
        raw_engine_start, raw_engine_end = (
            float(engine_band[0]), float(engine_band[1]))
    else:
        raw_engine_start = engine_arc.start_deg
        raw_engine_end = engine_arc.start_deg + engine_arc.sweep_deg
    raw_start = (-raw_engine_start
                 if convention == AZ.CONVENTION else raw_engine_start)
    raw_end = (-raw_engine_end
               if convention == AZ.CONVENTION else raw_engine_end)
    lo, hi = sorted((_display_angle(raw_start), _display_angle(raw_end)))
    return f"[{lo:g}, {hi:g})"


def _arc_option_record(engine_band, convention) -> dict:
    engine_arc = SS.band_to_arc(engine_band)
    published_arc = AZ.to_convention_arc(engine_arc, convention)
    return {
        "engine_frame": engine_arc.as_dict(),
        "published": published_arc.as_dict(),
        "convention": convention,
        "convention_note": AZ.convention_note(convention),
    }


def _arc_midpoint(arc) -> float:
    return AR.normalize_deg(arc.start_deg + arc.sweep_deg / 2.0)


def _timeline_video_fps(timeline, params) -> float:
    """Use and verify the final timeline clock for human time windows."""
    try:
        rate = float(timeline["render"]["frame_rate_hz"])
        configured = float(params["VIDEO_FPS"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GenerationConstraintError(
            "timeline and params must declare a numeric VIDEO_FPS") from exc
    if not math.isfinite(rate) or rate <= 0.0:
        raise GenerationConstraintError(
            "timeline frame rate must be finite and positive")
    if not math.isclose(rate, configured, rel_tol=0.0, abs_tol=1.0e-9):
        raise GenerationConstraintError(
            f"timeline frame rate {rate} differs from VIDEO_FPS {configured}")
    return rate


def _arc_interval_bounds(arc) -> tuple[float, float]:
    """Return numeric interval bounds without losing a directed seam Arc."""
    if arc.width_deg >= 360.0:
        return -180.0, 180.0
    if arc.wraps:
        return arc.start_deg, arc.end_deg
    return tuple(sorted((arc.start_deg, arc.end_deg)))


def query_sweep_inside_band(sweep: AR.Arc, band) -> bool:
    """The whole circular sweep must stay in the half-open answer band."""
    arc = SS.band_to_arc(band)
    return (arc.contains(sweep.start_deg) and arc.contains(sweep.end_deg)
            and math.isclose(AR.arc_overlap_width_deg(sweep, arc), sweep.width_deg,
                             rel_tol=0.0, abs_tol=1.0e-9))


def build_answer(
    kind,
    profile,
    cell,
    timeline,
    schedule,
    slot_events,
    target_slot,
    other_slot,
    slot_coat,
    truth_deg,
    query_frame,
    params,
    *,
    asset_context=None,
    slot_context_data=None,
    emitter_paths_by_slot=None,
):
    """按题型的答案空间造真值;MCQ 与 Open 引用**同一条**事实。"""
    family_id = profile.get("family_id", profile["id"])
    coat = slot_coat[target_slot]
    target_phrase = (
        slot_context_data["referent_phrases_by_slot"][target_slot]
        if slot_context_data is not None
        else f"the {coat} dog"
    )
    target_action = (
        slot_context_data["sound_action_phrases_by_slot"][target_slot]
        if slot_context_data is not None
        else "barks"
    )
    if kind == "event_count":
        actual = len(schedule.events)
        allocated = int(cell["answer_value"])
        if actual != allocated:
            raise GenerationConstraintError(
                f"event count {actual} != allocated {allocated}")
        options = [int(value) for value in profile.get(
            "mcq_options", [2, 3, 4, 5])]
        if actual not in options:
            raise GenerationConstraintError(
                f"event count {actual} is outside MCQ options {options}")
        return {
            "truth": {"event_count": actual},
            "mcq": {
                "stem": "How many sounds are heard in total?",
                "options_space": options,
                "truth_option": actual,
            },
            "open": {
                "stem": "How many sounds are heard in total?",
                "truth_value": actual,
                "scoring": "count_single",
            },
        }
    if kind == "distance_at_query":
        frame = timeline["frames"][query_frame]
        camera_xy = np.asarray(
            frame["camera"]["translation_ue_cm"][:2], dtype=float)
        positions = {
            state["source_slot_id"]: np.asarray(
                state["translation_ue_cm"][:2], dtype=float)
            for state in frame["actor_states"]
        }
        target_distance = float(np.linalg.norm(
            positions[target_slot] - camera_xy))
        other_distance = float(np.linalg.norm(
            positions[other_slot] - camera_xy))
        gap = other_distance - target_distance
        minimum = float(profile.get("min_distance_gap_cm", 50.0))
        if gap < minimum:
            raise GenerationConstraintError(
                f"distance gap {gap:.2f} cm is below {minimum:.2f} cm")
        truth = slot_coat[target_slot]
        options = ["black-and-white", "yellow"]
        fps = _timeline_video_fps(timeline, params)
        moment = (f"At zero-based video frame index {query_frame} "
                  f"({query_frame}/{fps:g} seconds)")
        return {
            "truth": {
                "closer_coat": truth,
                "target_distance_cm": round(target_distance, 3),
                "other_distance_cm": round(other_distance, 3),
                "distance_gap_cm": round(gap, 3),
            },
            "mcq": {
                "stem": f"{moment}, which dog is closer to you?",
                "options_space": options,
                "truth_option": truth,
            },
            "open": {
                "stem": f"{moment}, which dog is closer to you?",
                "truth_value": truth,
                "scoring": "closed_set",
            },
        }
    if kind == "distance_change":
        start_frame, end_frame = [
            int(value) for value in profile["relation_frames"]]
        camera_xy = np.asarray(
            timeline["frames"][start_frame]["camera"][
                "translation_ue_cm"][:2], dtype=float)

        def actor_xy(frame_index, slot):
            return np.asarray(next(
                state["translation_ue_cm"][:2]
                for state in timeline["frames"][frame_index]["actor_states"]
                if state["source_slot_id"] == slot), dtype=float)

        start_distance = float(np.linalg.norm(
            actor_xy(start_frame, target_slot) - camera_xy))
        end_distance = float(np.linalg.norm(
            actor_xy(end_frame, target_slot) - camera_xy))
        delta = end_distance - start_distance
        minimum = float(profile.get("min_distance_change_cm", 50.0))
        relation = (
            "closer" if delta <= -minimum
            else "farther" if delta >= minimum else None)
        allocated = str(cell["answer_value"])
        if relation != allocated:
            raise GenerationConstraintError(
                f"distance relation {relation} from delta {delta:.2f} cm "
                f"does not match allocated {allocated}")
        if family_id == "card5R":
            stem = (
                "After the dog that barked last stopped making sound, was it "
                "closer to you or farther from you at the end of the video?")
        else:
            stem = (
                "While the dog made the first sound, was it getting closer "
                "to you or farther from you?")
        return {
            "first_caller_slot": (
                min(slot_events, key=lambda item: item[1])[0]),
            "truth": {
                "distance_relation": relation,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_distance_cm": round(start_distance, 3),
                "end_distance_cm": round(end_distance, 3),
                "distance_delta_cm": round(delta, 3),
            },
            "mcq": {
                "stem": stem,
                "options_space": ["closer", "farther"],
                "truth_option": relation,
            },
            "open": {
                "stem": stem,
                "truth_value": relation,
                "scoring": "closed_set",
            },
        }
    if kind == "motion_state":
        start_frame, end_frame = [
            int(value) for value in profile["motion_frames"]]

        def actor_xy(frame_index, slot):
            return np.asarray(next(
                state["translation_ue_cm"][:2]
                for state in timeline["frames"][frame_index]["actor_states"]
                if state["source_slot_id"] == slot), dtype=float)

        displacement = float(np.linalg.norm(
            actor_xy(end_frame, target_slot)
            - actor_xy(start_frame, target_slot)))
        minimum = float(profile.get("min_motion_cm", 10.0))
        state = (
            "moving" if displacement >= minimum
            else "still" if displacement <= 1.0e-6 else None)
        allocated = str(cell["answer_value"])
        if state != allocated:
            raise GenerationConstraintError(
                f"motion state {state} from displacement {displacement:.2f} "
                f"does not match allocated {allocated}")
        if family_id == "card6R":
            stem = (
                "After the second sound ended, did the dog that made it move "
                "during the remaining silent part of the video?")
        elif family_id == "card6":
            stem = (
                "Was the dog that made the second sound moving while that "
                "sound was heard?")
        else:
            stem = (
                "Was the dog that made the first sound moving while that "
                "sound was heard?")
        return {
            "first_caller_slot": (
                min(slot_events, key=lambda item: item[1])[0]),
            "truth": {
                "motion_state": state,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "window_displacement_cm": round(displacement, 3),
            },
            "mcq": {
                "stem": stem,
                "options_space": ["moving", "still"],
                "truth_option": state,
            },
            "open": {
                "stem": stem,
                "truth_value": state,
                "scoring": "closed_set",
            },
        }
    if kind == "first_sound_side":
        firsts = {}
        for slot, start in slot_events:
            firsts.setdefault(slot, start)
        first_slot = min(firsts, key=firsts.get)
        if first_slot != target_slot:
            raise GenerationConstraintError(
                f"first caller {first_slot} does not match target {target_slot}")
        bands = list(profile["answer_bands_deg"])
        got = next((index for index, band in enumerate(bands)
                    if SS.band_contains(truth_deg, band)), None)
        want = next((index for index, band in enumerate(bands)
                     if SS.bands_equivalent(band, cell["answer_band"])), None)
        if got != want:
            raise GenerationConstraintError(
                f"first-sound azimuth {truth_deg:.2f} lands in band {got}, "
                f"not allocated band {want}")
        # This answer is published, so left and right follow the
        # published convention rather than the engine frame.
        convention = _profile_convention(profile)
        published_deg = AZ.to_convention_deg(truth_deg, convention)
        side = AZ.side_word(published_deg)
        fps = _timeline_video_fps(timeline, params)
        moment = (f"At zero-based video frame index {query_frame} "
                  f"({query_frame}/{fps:g} seconds)")
        return {
            "first_caller_slot": first_slot,
            "truth": {
                "first_sound_side": side,
                "first_sound_azimuth_deg_engine_frame": round(
                    truth_deg, 3),
                **AZ.convention_block(truth_deg, convention),
            },
            "mcq": {
                "stem": (f"{moment}, did the first sound come from the left "
                         "or right side relative to your facing direction?"),
                "options_space": ["left", "right"],
                "truth_option": side,
                "convention": convention,
                "convention_note": AZ.convention_note(convention),
            },
            "open": {
                "stem": (f"{moment}, which side did the first sound come "
                         "from: left or right?"),
                "truth_value": side,
                "scoring": "closed_set",
                "convention": convention,
                "convention_note": AZ.convention_note(convention),
            },
        }
    if kind == "front_back":
        bands = list(profile["answer_bands_deg"])
        labels = list(SS.front_back_labels(profile))
        if len(bands) != 2:
            raise GenerationConstraintError(
                f"{profile.get('id')}: front/back answer needs exactly two bands")
        got = next((i for i, band in enumerate(bands)
                    if SS.band_contains(truth_deg, band)), None)
        want = next((i for i, band in enumerate(bands)
                     if SS.bands_equivalent(band, cell["answer_band"])), None)
        if got is None or want is None or got != want:
            raise GenerationConstraintError(
                f"front/back truth {truth_deg:.2f} deg lands in band {got}, "
                f"not allocated band {want}")
        convention = _profile_convention(profile)
        resolved_geometry = _resolved_query_geometry(profile, cell)
        engine_arc = SS.band_to_arc(bands[got])
        published_arc = AZ.to_convention_arc(engine_arc, convention)
        fps = _timeline_video_fps(timeline, params)
        window = profile_query_window(profile, params, query_frame, fps)
        query_sweep, window_frames = azimuth_sweep_engine_arc(
            timeline, target_slot, window, fps, emitter_paths_by_slot)
        if not query_sweep_inside_band(query_sweep, bands[got]):
            raise GenerationConstraintError(
                f"front/back query sweep {query_sweep.as_dict()} crosses "
                f"band {got} during query window {window}")
        moment = f"Between {window[0]:g} and {window[1]:g} seconds"
        stem = (f"{moment}, was the queried sound source in front of you "
                f"or behind you? Choose {labels[0]} or {labels[1]}.")
        truth = {
            "answer_label": labels[got],
            "answer_band_index": got,
            "answer_domain": "front_back",
            "query_domain": resolved_geometry.get("query_domain"),
            "query_requires_visibility": resolved_geometry.get("query_requires_visibility"),
            "query_window_seconds": list(window),
            "query_window_frame_bounds": list(window_frames),
            "azimuth_deg_engine_frame": round(truth_deg, 3),
            "arc_engine_frame": engine_arc.as_dict(),
            "arc_published": published_arc.as_dict(),
            **AZ.convention_block(truth_deg, convention),
        }
        return {
            "truth": truth,
            "mcq": {
                "stem": stem,
                "options_space": labels,
                "truth_option": labels[got],
                "convention": convention,
                "convention_note": AZ.convention_note(convention),
                "direction_evidence": profile.get("direction_evidence"),
                "option_arcs": [
                    _arc_option_record(band, convention) for band in bands
                ],
            },
            "open": {
                "stem": stem,
                "truth_value": labels[got],
                "scoring": "closed_set",
                "vocab_key": profile["vocab_key"],
                "vocab_path": profile.get("vocab_path"),
                "answer_domain": "front_back",
                "query_domain": resolved_geometry.get("query_domain"),
                "query_requires_visibility": resolved_geometry.get("query_requires_visibility"),
                "convention": convention,
                "convention_note": AZ.convention_note(convention),
                "direction_evidence": profile.get("direction_evidence"),
                "truth_arc": published_arc.as_dict(),
                "truth_arc_engine_frame": engine_arc.as_dict(),
            },
        }
    if kind in ("azimuth_band", "instant_azimuth_band"):
        # Band matching stays in the engine frame: cell["answer_band"] is what
        # the solver allocated, so the frames have to agree.
        bands = list(profile["answer_bands_deg"])
        got = next((i for i, band in enumerate(bands)
                    if SS.band_contains(truth_deg, band)), None)
        want = next((i for i, band in enumerate(bands)
                     if SS.bands_equivalent(band, cell["answer_band"])), None)
        if got != want:
            raise GenerationConstraintError(
                f"recomputed truth {truth_deg:.2f} deg lands in band {got}, "
                f"not the assigned {want}: the final camera pose disagrees "
                "with the solver's geometry")
        # The camera edge comes from its calibration, not the answer domain:
        # a safety margin or a full-circle answer space does not change the lens.
        convention = _profile_convention(profile)
        resolved_geometry = _resolved_query_geometry(profile, cell)
        labels = [_arc_option_label(band, convention) for band in bands]
        option_arcs = [_arc_option_record(band, convention) for band in bands]
        try:
            frame_edge = float(timeline["render"]["hfov_degrees"]) / 2.0
        except (KeyError, TypeError, ValueError) as exc:
            raise GenerationConstraintError("angle questions need the timeline camera HFOV") from exc
        convention_text = AZ.landmark_sentence(frame_edge, convention)
        video_fps = _timeline_video_fps(timeline, params)
        if profile["temporal"] == "forward":
            # 片尾是人能对齐的时刻，本来就不用窗口。
            moment = "At the end of the video"
            referent = "the dog that barked last"
            sweep_lo = sweep_hi = truth_deg
            sweep_arc = AR.Arc(start_deg=truth_deg, sweep_deg=0.0)
            window = None
            window_frames = None
        elif profile["temporal"] in ("backward", "instant"):
            window = profile_query_window(profile, params, query_frame, video_fps)
            moment = f"Between {window[0]:g} and {window[1]:g} seconds"
            referent = ("the dog that barked last"
                        if profile["temporal"] == "backward"
                        else "the dog barking in that window")
            sweep_arc, window_frames = azimuth_sweep_engine_arc(
                timeline, target_slot, window, video_fps,
                emitter_paths_by_slot)
            # The whole query sweep must stay in one band for a unique answer.
            if not query_sweep_inside_band(sweep_arc, bands[got]):
                raise GenerationConstraintError(
                    f"azimuth sweep {sweep_arc.as_dict()} crosses band "
                    f"{got} edges {bands[got]} during query window {window}: "
                    "the banded answer would not be unique")
        else:
            raise ValueError("azimuth-band profile must declare a time direction")
        if profile["temporal"] == "forward":
            # sweep_arc was initialized above; keep the branch explicit for
            # older callers that provide a forward point without a window.
            sweep_arc = AR.Arc(start_deg=truth_deg, sweep_deg=0.0)
        published_arc = AZ.to_convention_arc(sweep_arc, convention)
        published_interval = _arc_interval_bounds(published_arc)
        published_interval = [round(v, 3) for v in published_interval]
        # A numeric pair is safe for a short non-wrapping set. Wrapped or
        # longer sweeps must stay as the existing Arc object for the scorer.
        interval_for_scorer = (
            published_arc.as_dict()
            if published_arc.wraps or published_arc.width_deg > 180.0
            else published_interval
        )
        truth_block = {
            "band_index": got,
            **AZ.convention_block(truth_deg, convention),
            "azimuth_deg_engine_frame": round(truth_deg, 3),
            "engine_frame_note": AZ.ENGINE_FRAME_NOTE,
            "query_window_seconds": list(window) if window else None,
            "query_window_frame_bounds": (
                list(window_frames) if window_frames is not None else None),
            "azimuth_interval_deg": published_interval,
            "azimuth_interval_engine_frame": list(_arc_interval_bounds(sweep_arc)),
            "azimuth_interval_arc_engine_frame": sweep_arc.as_dict(),
            "azimuth_interval_arc": published_arc.as_dict(),
            "answer_domain": profile.get("answer_domain"),
            "query_domain": resolved_geometry.get("query_domain"),
            "query_requires_visibility": resolved_geometry.get("query_requires_visibility"),
        }
        if published_arc.width_deg >= 360.0:
            midpoint = 0.0
        else:
            midpoint = _arc_midpoint(published_arc)
        return {"truth": truth_block,
                "mcq": {"stem": (f"{convention_text} {moment}, which azimuth band "
                                 f"contains {referent}?"),
                        "options_space": labels, "truth_option": labels[got],
                        "convention": convention,
                        "convention_note": AZ.convention_note(convention),
                        "direction_evidence": profile.get("direction_evidence"),
                        "option_arcs": option_arcs},
                "open": {"stem": (f"{convention_text} {moment}, roughly what is the "
                                  f"azimuth of {referent}? Report a numeric "
                                  "estimate in degrees rather than a category."),
                         "truth_interval_deg": interval_for_scorer,
                         "truth_value": round(midpoint, 3),
                         "truth_value_note": (
                             "midpoint of truth_interval_deg; the interval is "
                             "authoritative"),
                         "truth_interval_arc": published_arc.as_dict(),
                         "truth_interval_arc_engine_frame": sweep_arc.as_dict(),
                         "unit": "deg", "scoring": "circular_deg_interval",
                         "certification_policy": resolve_angle_policy(params),
                         "wide_tolerance_role": ("diagnostic_only" if resolve_angle_policy(params)
                                                 == "strict_full_credit_only" else "graded"),
                         "convention": convention,
                         "convention_note": AZ.convention_note(convention),
                         "direction_evidence": profile.get("direction_evidence"),
                         "answer_domain": profile.get("answer_domain"),
                         "query_domain": profile.get("query_domain"),
                         "query_requires_visibility": profile.get("query_requires_visibility")}}
    if kind == "coat_at_query":
        calling = [slot for slot, event in zip(
            [s for s, _ in slot_events], schedule.events)
            if event.frame_span()[0] <= query_frame < event.frame_span()[1]]
        if len(calling) != 1:
            raise ValueError(f"{len(calling)} actors sound at the query frame")
        truth = slot_coat[calling[0]]
        options = ["black-and-white", "yellow", "both", "neither"]
        seconds = query_frame / float(_require_param(params, "VIDEO_FPS"))
        moment = (f"At zero-based video frame index {query_frame} "
                  f"({query_frame}/{int(params['VIDEO_FPS'])} seconds)")
        return {"truth": {"calling_at_query": truth,
                          "query_frame": query_frame,
                          "query_second": round(seconds, 4)},
                "mcq": {"stem": f"{moment}, which dog is barking?",
                        "options_space": options, "truth_option": truth},
                "open": {"stem": (f"{moment}, which dog, if any, is "
                                  "barking?"),
                         "truth_value": truth, "scoring": "closed_set"}}
    if kind == "time_band":
        edges = AP.card8_band_edges(params)
        scoring_chain = AP.card8_scoring_params(params)
        firsts = {}
        first_samples = {}
        for (slot, start), event in zip(slot_events, schedule.events):
            firsts.setdefault(slot, start / _sample_rate_hz(params))
            first_samples.setdefault(slot, int(start))
        onset = firsts[target_slot]
        got = next((i for i in range(len(edges) - 1)
                    if edges[i] <= onset < edges[i + 1]), None)
        want = int(cell["target_band"])
        if got != want:
            raise ValueError(f"first call landed in band {got}, assigned {want}")
        labels = [f"[{edges[i]:g}, {edges[i + 1]:g})"
                  for i in range(len(edges) - 1)]
        clip_seconds = float(_require_param(params, "CLIP_SECONDS"))
        other_onset = firsts.get(other_slot)
        if other_onset is None:
            raise GenerationConstraintError(
                "card8 needs a first call from both slots")
        # 与调度器同一条规则,在绑定后的事件上再核一次(样本域整数比较):
        # 正式 Open 按 strict T_FULL 判分,两只首叫必须严格相隔超过
        # max(T_HALF, 2*T_FULL),否则"报两声中点"在认证分上拿满分。
        separation_samples = abs(first_samples[target_slot]
                                 - first_samples[other_slot])
        if separation_samples <= scoring_chain[
                "min_first_call_separation_samples"]:
            raise GenerationConstraintError(
                f"card8 first-call separation "
                f"{separation_samples / _sample_rate_hz(params):.4f}s is not strictly "
                "above max(T_HALF, 2*T_FULL)="
                f"{scoring_chain['min_first_call_separation_s']:.4f}s")
        return {"first_caller_slot": min(firsts, key=firsts.get),
                "truth": {"first_onset_s": round(onset, 4), "band_index": got,
                          "non_target_first_onset_s": round(other_onset, 4),
                          "first_call_separation_s": round(
                              separation_samples / _sample_rate_hz(params), 4),
                          "first_call_separation_above_minimum": True},
                "mcq": {"stem": (
                    f"The clip is {clip_seconds:g} seconds long. In which "
                    f"one-second interval does {target_phrase} {target_action} "
                    "for the FIRST time?"),
                        "options_space": labels, "truth_option": labels[got]},
                "open": {"stem": (
                    f"The clip is {clip_seconds:g} seconds long. At how many "
                    f"seconds does {target_phrase} {target_action} for the "
                    "first time?"),
                         "truth_value": round(onset, 4), "unit": "s",
                         "scoring": "absolute_time",
                         "certification_policy": scoring_chain[
                             "certification_policy"],
                         "wide_tolerance_role": scoring_chain[
                             "wide_tolerance_role"],
                         "T_FULL": scoring_chain["T_FULL"],
                         "T_HALF": scoring_chain["T_HALF"],
                         # 让读 fact 的人看见 T_FULL 是从答案粒度锁出来的，
                         # 而不是一个等人类校准的占位值。
                         "answer_granularity_seconds": scoring_chain[
                             "answer_granularity_seconds"],
                         "answer_granularity_rule": scoring_chain[
                             "answer_granularity_rule"],
                         "T_FULL_status": scoring_chain["T_FULL_status"],
                         "min_first_call_separation_s": scoring_chain[
                             "min_first_call_separation_s"],
                         "min_first_call_separation_rule": scoring_chain[
                             "min_first_call_separation_rule"]}}
    if kind == "first_caller_coat":
        firsts = {}
        for slot, start in slot_events:
            firsts.setdefault(slot, start)
        first_slot = min(firsts, key=firsts.get)
        truth = slot_coat[first_slot]
        expect_first = target_slot if cell["target_first"] else other_slot
        if first_slot != expect_first:
            raise ValueError(
                f"the first caller is {first_slot}, not the assigned "
                f"{expect_first}: the schedule disagrees with the cell plan")
        return {"first_caller_slot": first_slot,
                "truth": {"first_to_bark": truth,
                          "onset_gap_s": round(
                              abs(firsts[target_slot] - firsts[other_slot])
                              / _sample_rate_hz(params), 4)},
                "mcq": {"stem": ("Which dog barked first, the black-and-white "
                                 "one or the yellow one?"),
                        "options_space": ["black-and-white", "yellow"],
                        "truth_option": truth},
                "open": {"stem": "Which dog made the first sound?",
                         "truth_value": truth, "scoring": "closed_set"}}
    raise ValueError(f"unknown answer kind {kind!r}")


def cell_allocation(cell):
    """The allocation a cell was given before search (what the budget is spent on)."""
    def band(value):
        if isinstance(value, (list, tuple)):
            return f"{float(value[0]):g},{float(value[1]):g}"
        return None if value is None else str(value)
    return {"profile_id": cell["profile"]["id"],
            "target_slot": cell["target_slot"],
            "anchor_band": band(cell.get("anchor_band")),
            "answer": band(cell.get("answer_band")),
            "target_moves_more": cell.get("target_moves_more"),
            "target_coat": cell.get("target_coat")}


def route_source_counts(records):
    """How many realised candidates drew each role from the bank or from a
    designed route (see scene_sampler.route_synthesizer)."""
    counts = {"target": {}, "other": {}, "candidates_with_synthesized_route": 0}
    for record in records:
        sources = (record.get("motion") or {}).get("route_sources") or {}
        for role in ("target", "other"):
            source = sources.get(role, "unknown")
            counts[role][source] = counts[role].get(source, 0) + 1
        if "synthesized" in sources.values():
            counts["candidates_with_synthesized_route"] += 1
    return counts


def cell_budget_report(cells, made, rejected):
    """Joint allocation table per profile: for every allocated
    slot x anchor band x answer x motion-rank key, how many cells were
    requested, how many were filled, and why the rest were exhausted.

    This is the room's own denominator.  A room that cannot fill a key
    reports it here as a shortfall; nothing is borrowed from another room."""
    made_set = set(made)
    reasons = {row["point_id"]: row["reason"] for row in rejected}
    report = {}
    for cell in cells:
        profile = cell["profile"]
        pid = f"{profile['id']}_{cell['cell_index'] + 1:03d}"
        alloc = cell_allocation(cell)
        key = "|".join(f"{name}={alloc[name]}" for name in
                       ("target_slot", "anchor_band", "answer", "target_moves_more"))
        bucket = report.setdefault(profile["id"], {"cells": {}, "totals": {
            "requested": 0, "filled": 0, "exhausted": 0, "keys": 0,
            "keys_unfilled": 0}})
        row = bucket["cells"].setdefault(key, {"requested": 0, "filled": 0,
                                               "exhausted_by_reason": {}})
        row["requested"] += 1
        bucket["totals"]["requested"] += 1
        if pid in made_set:
            row["filled"] += 1
            bucket["totals"]["filled"] += 1
        else:
            reason = reasons.get(pid, "unknown")
            row["exhausted_by_reason"][reason] = (
                row["exhausted_by_reason"].get(reason, 0) + 1)
            bucket["totals"]["exhausted"] += 1
    for bucket in report.values():
        bucket["totals"]["keys"] = len(bucket["cells"])
        bucket["totals"]["keys_unfilled"] = sum(
            1 for row in bucket["cells"].values() if row["filled"] == 0)
        bucket["boundary"] = ("allocation before search; shortfalls stay with "
                              "this room and are not backfilled from others")
    return report


def conditional_balance(records):
    """逐 profile 的条件分布 —— 总表 50:50 掩盖不了 profile 内部可预测。"""
    out = {}
    for record in records:
        pid = record["profile_id"]
        bucket = out.setdefault(pid, {})
        truth = record["truth"]

        def bump(field, value):
            bucket.setdefault(field, {})
            key = str(value)
            bucket[field][key] = bucket[field].get(key, 0) + 1

        bump("target_slot", record["target_slot"])
        bump("target_coat", record["target_coat"])
        bump("source1_coat", record["slot_coat"]["source1"])
        for field in ("band_index", "calling_at_query", "first_to_bark"):
            if field in truth:
                bump(f"answer_{field}", truth[field])
        if record.get("target_first") is not None:
            bump("target_first_caller", record["target_first"])
        if record.get("first_caller_slot"):
            bump("first_caller_slot", record["first_caller_slot"])
        motion = record.get("motion") or {}
        if "target_moves_more" in motion:
            bump("target_moves_more", motion["target_moves_more"])
        if motion.get("both_roles_move") is not None:
            bump("both_roles_move", motion["both_roles_move"])
    return out


def card8_diagnostics(params, records):
    """⑧ 的时间域逐项列明,证明它没有继承 ①F 的片尾静默,并记录实际执行的
    首叫评分参数链(T_FULL / T_HALF / 推导出的最小首叫间隔 / 认证政策)。"""
    rows = [r for r in records if r.get("answer_kind") == "time_band"]
    try:
        scoring_chain = AP.card8_scoring_params(params)
        edges = AP.card8_band_edges(params)
    except AP.AudioProfileError as exc:
        if rows:
            raise
        return {"status": "not_derived",
                "reason": f"no card8 rows and incomplete scoring chain: {exc}"}
    event_seconds, event_source = AP.card8_event_length_seconds(params)
    lo, hi = AP.card8_feasible_interval(params, event_seconds=event_seconds)
    target_onsets = [r["truth"]["first_onset_s"] for r in rows
                     if "first_onset_s" in r["truth"]]
    other_onsets = [r["truth"].get("non_target_first_onset_s") for r in rows]
    other_onsets = [o for o in other_onsets if o is not None]
    counts = {}
    firsts = {}
    for record in rows:
        band = str(record["truth"].get("band_index"))
        counts[band] = counts.get(band, 0) + 1
        key = str(record.get("target_first"))
        firsts[key] = firsts.get(key, 0) + 1
    separations = [r["truth"].get("first_call_separation_s") for r in rows]
    separations = [s for s in separations if s is not None]
    return {
        "clip_duration_s": float(_require_param(params, "CLIP_SECONDS")),
        "event_seconds": event_seconds,
        "event_seconds_source": event_source,
        "card8_feasible_interval_s": [lo, hi],
        "card8_mcq_band_edges_s": edges,
        "derivation": ("clip - event - (min_events - 2) * (event + gap); "
                       "no tail silence is applied to card8; the two first "
                       "calls must be strictly more than "
                       "max(T_HALF, 2*T_FULL) apart"),
        "first_call_scoring": scoring_chain,
        "realized_first_call_separation_min_s": (
            min(separations) if separations else None),
        "realized_first_call_separation_max_s": (
            max(separations) if separations else None),
        "target_first_onset_min_s": min(target_onsets) if target_onsets else None,
        "target_first_onset_max_s": max(target_onsets) if target_onsets else None,
        "non_target_first_onset_min_s": min(other_onsets) if other_onsets else None,
        "non_target_first_onset_max_s": max(other_onsets) if other_onsets else None,
        "target_band_counts": counts,
        "target_first_counts": firsts,
        "empty_bands": [str(i) for i in range(len(edges) - 1)
                        if str(i) not in counts],
    }


def write_outputs(
    args,
    scene,
    scene_cfg,
    profiles,
    params,
    ledger,
    made,
    rejected,
    records,
    per_profile_ledger=None,
    cells=None,
    answer_band_audit=None,
    asset_context=None,
):
    by_profile = Counter(r["profile_id"] for r in records)
    coat_of_slot1 = Counter(r["slot_coat"]["source1"] for r in records)
    target_slots = Counter(r["target_slot"] for r in records)
    # 答案分布按题型的答案空间统计:方位带/时间带看带号,外观题看标签
    def answer_key(record):
        truth = record["truth"]
        for field in ("band_index", "calling_at_query", "first_to_bark"):
            if field in truth:
                return f"{record['profile_id']}:{truth[field]}"
        return f"{record['profile_id']}:unknown"

    bands = Counter(answer_key(r) for r in records)
    gatea_pairs = [r.get("gatea") for r in records if r.get("gatea")]
    gatea_by_profile = Counter(r["profile_id"] for r in records
                               if r.get("gatea"))
    manifest = {
        "schema": "qa_v3_scene_batch_manifest_v1",
        # 每个批次都把答案带跟本场景相机的对账结果带上:声明多少度、其中多少度
        # 根本到不了。没有它,外侧带 5.0 度的死区就只存在于某个人某天的记忆里。
        "answer_band_audit": answer_band_audit,
        "code": git_worktree_state(),
        "inputs": {
            "scene_config": str(args.scene_config.resolve()),
            "scene_config_content": scene_cfg,
            "route_bank_content_sha256": content_sha256(
                scene_cfg["route_bank"]),
            "camera_base_request_content_sha256": content_sha256(
                scene_cfg["camera_base_request"]),
            "profiles": str(args.profiles.resolve()),
            "profiles_content": profiles,
            "params": str(args.params.resolve()),
            "profile_ids": [profile["id"] for profile in profiles],
            "seed": args.seed,
            "cells_per_profile": args.cells,
            "asset_policy": str(args.asset_policy.resolve()),
            "asset_policy_pair_id": (
                asset_context["pair_id"] if asset_context is not None else None
            ),
            "asset_policy_content": (
                asset_context["raw"] if asset_context is not None else None
            ),
        },
        "scene": {"scene_id": scene.scene_id, "backend": scene.backend,
                  "scene_asset_id": scene_cfg.get("scene_asset_id",
                                                  scene.scene_id),
                  "route_domain": scene_cfg.get("route_domain"),
                  "bank_adapter": scene.provenance.get("bank_adapter"),
                  "routes_loaded": scene.provenance.get("routes_loaded"),
                  "line_of_sight_screened": scene.line_of_sight_screened,
                  "route_pool": SS.route_pool_report(scene, params),
                  "camera_clearance_screened": scene.camera_clearance_screened,
                  "camera_clearance_table": scene.provenance.get(
                      "camera_clearance_table"),
                  "camera_height_fallback_used": sum(
                      1 for r in records
                      if ((r.get("camera") or {}).get("clearance") or {}).get(
                          "fallback_used")),
                  "walkable_grid": scene.provenance.get("walkable_grid"),
                  "floor_reference": scene.provenance.get("floor_reference"),
                  "route_synthesis": dict(
                      SS.route_synthesis_report(scene, params),
                      realised=route_source_counts(records))},
        "evidence_class": "geometry_candidate",
        "boundary": ("no pixel or line-of-sight evidence yet; these are "
                     "pre-render candidates, not admitted questions"),
        "counts": {"cells_requested": args.cells * len(profiles),
                   "geometry_candidates": len(made),
                   "main_facts": len(records),
                   "gatea_facts": len(gatea_pairs),
                   "rejected": len(rejected),
                   "by_profile": dict(by_profile)},
        "search": {k: v for k, v in ledger.summary().items()
                   if k != "first_example"},
        "search_note": ("pass rates observed while filling the quota; "
                        "candidate order and the stopping rule affect them, "
                        "so they are not scene-wide admission probabilities"),
        "evaluated_until_quota_filled": True,
        "balance": {"source1_coat": dict(coat_of_slot1),
                    "target_slot": dict(target_slots),
                    "answer_by_profile": dict(sorted(bands.items()))},
        "per_profile_conditional_balance": conditional_balance(records),
        "cell_budget": (cell_budget_report(cells, made, rejected)
                        if cells is not None else None),
        "predicted_tier_distribution": predicted_tier_distribution(records),
        "per_profile_rejections": {
            pid: {k: v for k, v in led.summary().items()
                  if k != "first_example"}
            for pid, led in (per_profile_ledger or {}).items()},
        "gatea": {
            "pairs": len(gatea_pairs),
            "by_profile": dict(gatea_by_profile),
            "gold_relation_by_profile": {
                profile_id: next(
                    g["checks"]["gatea_gold_relation"]
                    for g in gatea_pairs
                    if g["checks"]["profile_id"] == profile_id)
                for profile_id in gatea_by_profile},
            "mcq_gold_relation_satisfied": sum(
                bool(g["checks"]["mcq_gold_relation_satisfied"])
                for g in gatea_pairs),
            "open_gold_relation_satisfied": sum(
                bool(g["checks"]["open_gold_relation_satisfied"])
                for g in gatea_pairs),
            "mcq_gold_flipped": sum(
                bool(g["checks"]["mcq_gold_flipped"])
                for g in gatea_pairs),
            "open_gold_separated": sum(
                bool(g["checks"]["open_gold_separated"])
                for g in gatea_pairs),
            "structure_preserved": sum(
                all(g["checks"][key] for key in (
                    "event_count_same", "candidate_endpoints_same",
                    "non_slot_event_fields_same", "slot_sequence_changed"))
                for g in gatea_pairs),
            "boundary": ("generation-time structural audit; rendered waveform "
                         "and pixel evidence are not established here"),
        },
        "card8_time_domain": card8_diagnostics(params, records),
        "rejections": rejected,
        "params": params,
        "status": "research_candidate", "qualification_claim": False,
    }
    request_result = write_requested_questions(
        args.out_root,
        (args.out_root / record["point_id"] / "fact_record.json" for record in records),
        params,
    )
    manifest["question_request"] = request_result
    manifest["counts"]["designed_questions"] = request_result["designed_question_count"]
    manifest["counts"]["counterfactual_questions"] = request_result["counterfactual_question_count"]
    (args.out_root / "batch_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1))
    with open(args.out_root / "facts.jsonl", "w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            gatea_path = (args.out_root / record["point_id"] /
                          record["gatea"]["fact_record"])
            gatea_record = json.loads(gatea_path.read_text())
            handle.write(json.dumps(gatea_record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
