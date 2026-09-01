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
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))

import audio_profiles as AP  # noqa: E402
import scene_sampler as SS  # noqa: E402
from build_qa_v3_programs import build_program  # noqa: E402
# 选角文档的结构(蓝图/网格/动画的物理来源、UE 绑定)已在既有装配器里
# 验证过,直接复用它的构造函数,不在这里重写一份容易走样的副本。
from design_qa_v3_pilot_batch import _selection_doc  # noqa: E402
# 静→走用与旧管线**同一个**变换:创作函数按弧长把整条路线铺满 75 帧,
# 那是"压缩式";求解器用的是保速的"平移式"。两者不一致会让中途帧的
# 位置对不上 —— 集成冒烟里正是反向题(查询帧在中途)先露馅。
from make_idle_then_walk_timeline import transform_idle_then_walk  # noqa: E402
from avengine.camera_pose import apply_camera_listener_pose_ue  # noqa: E402
from avengine.dataset.apartment_dynamic_audio import (  # noqa: E402
    apartment_ue_point_to_world_m,
)
from avengine.timeline.current_apartment_visual import (  # noqa: E402
    author_current_apartment_visual_timeline,
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


class GenerationConstraintError(ValueError):
    """A candidate is well-formed but fails a declared question constraint."""


def sha_rng(*parts) -> np.random.Generator:
    tag = "|".join(str(p) for p in parts).encode()
    return np.random.default_rng(
        int.from_bytes(hashlib.sha256(tag).digest()[:8], "big") % 2**32)


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
    return {
        "native_map": native_map,
        "room_profile_id": str(render["room_profile_id"]),
        "world_transform": transform,
        "world_transform_id": transform_id,
        "ground_z_ue_cm": ground_z,
    }


def recompute_azimuth(timeline, slot, frame):
    """从**最终时间线**重算方位:相机姿态已经应用,不能沿用旋转前的角度。"""
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


def band_of(value, edges):
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return i
    return None


def audit_gatea_pair(profile, main_program, gatea_program, main_answer,
                     gatea_answer, params):
    """Verify a profile-specific Gate A under both answer forms.

    A waveform difference is not enough.  The intervention must preserve
    event timing and all non-slot audio fields, change the slot sequence,
    flip the MCQ gold, and separate the Open golds under that form's actual
    scorer.
    """
    main_events = main_program["events"]
    gatea_events = gatea_program["events"]
    gold_relation = profile.get("gatea_gold_relation", "flip")
    if gold_relation not in ("flip", "preserve"):
        raise GenerationConstraintError(
            f"unknown Gate A gold relation {gold_relation!r}")

    def without_slot(event):
        return {k: v for k, v in event.items()
                if k not in ("event_id", "source_endpoint_id")}

    structure = {
        "event_count_same": len(main_events) == len(gatea_events),
        "candidate_endpoints_same": (
            main_program["candidate_source_endpoint_ids"]
            == gatea_program["candidate_source_endpoint_ids"]),
        "non_slot_event_fields_same": (
            [without_slot(e) for e in main_events]
            == [without_slot(e) for e in gatea_events]),
        "slot_sequence_changed": (
            [e["source_endpoint_id"] for e in main_events]
            != [e["source_endpoint_id"] for e in gatea_events]),
        "mcq_stem_same": (main_answer["mcq"]["stem"]
                           == gatea_answer["mcq"]["stem"]),
        "mcq_options_same": (main_answer["mcq"]["options_space"]
                              == gatea_answer["mcq"]["options_space"]),
        "open_stem_same": (main_answer["open"]["stem"]
                            == gatea_answer["open"]["stem"]),
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
    open_preserved = main_open["truth_value"] == gatea_open["truth_value"]
    if scoring == "circular_deg":
        separation = SS.circular_gap_deg(
            float(main_open["truth_value"]),
            float(gatea_open["truth_value"]))
        threshold = 2.0 * float(params["THETA_HALF"])
        open_separated = separation > threshold
        open_rule = "circular_distance > 2*THETA_HALF"
    elif scoring == "absolute_time":
        separation = abs(float(main_open["truth_value"])
                         - float(gatea_open["truth_value"]))
        threshold = float(params["T_HALF"])
        open_separated = separation > threshold
        open_rule = "absolute_time_difference > T_HALF"
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


def solve_for_profile(profile, cell, scene, params, rng, ledger):
    """时间关系决定用哪个求解器 —— 题型只声明关系,不写房间分支。"""
    temporal = profile["temporal"]
    kind = profile.get("answer_kind", "azimuth_band")
    if (temporal == "instant"
            and kind in ("instant_azimuth_band", "first_sound_side")):
        return SS.solve_instant_azimuth(
            scene, params, answer_band=cell["answer_band"],
            answer_bands=[tuple(b) for b in profile["answer_bands_deg"]],
            query_frame=profile["binding_frames"][0],
            profile_id=profile["id"],
            idle_choices=profile["idle_choices"], rng=rng, ledger=ledger,
            target_moves_more=cell["target_moves_more"],
            max_attempts=profile.get("max_attempts", 3000),
            open_half_width_deg=profile.get("open_half_width_deg"))

    if temporal == "forward":
        return SS.solve_forward_cross_time(
            scene, params, answer_band=cell["answer_band"],
            answer_bands=[tuple(b) for b in profile["answer_bands_deg"]],
            anchor_frame=profile["anchor_frame"],
            idle_choices=profile["idle_choices"], rng=rng, ledger=ledger,
            target_moves_more=cell["target_moves_more"],
            max_attempts=profile.get("max_attempts", 3000))
    if temporal == "backward":
        return SS.solve_backward_cross_time(
            scene, params, answer_band=cell["answer_band"],
            answer_bands=[tuple(b) for b in profile["answer_bands_deg"]],
            anchor_frame=profile["anchor_frame"],
            query_frame=profile["query_frame"],
            idle_choices=profile["idle_choices"], rng=rng, ledger=ledger,
            target_moves_more=cell["target_moves_more"],
            max_attempts=profile.get("max_attempts", 3000))
    if temporal == "instant":
        return SS.solve_instant_binding(
            scene, params, instants=profile["binding_frames"],
            profile_id=profile["id"], idle_choices=profile["idle_choices"],
            rng=rng, ledger=ledger,
            target_moves_more=cell["target_moves_more"],
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
        "coat_at_query", "time_band", "first_caller_coat", "event_count",
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
        if profile["temporal"] == "forward":
            required |= {"anchor_frame", "answer_bands_deg"}
        elif profile["temporal"] == "backward":
            required |= {"anchor_frame", "query_frame", "answer_bands_deg"}
        else:
            required |= {"binding_frames"}
            if profile.get("answer_kind") in ("instant_azimuth_band", "first_sound_side"):
                required.add("answer_bands_deg")
            if profile.get("answer_kind") == "event_count":
                required.add("answer_values")
        missing = sorted(key for key in required if key not in profile)
        if missing:
            raise ValueError(f"{pid}: missing required profile fields {missing}")


def build_cell_plan(cells, profiles, pair_assets, params, seed):
    """先分配答案与角色,再求解 —— 不是先采样后看落在哪。"""
    a1, a2 = pair_assets
    plan = []
    per_profile = {}
    for profile in profiles:
        # 答案格按题型的答案空间分配:方位带题分带,时间带题分有序带对,
        # 外观题分目标外观 —— 一律**先分配再求解**。
        kind = profile.get("answer_kind", "azimuth_band")
        if kind in ("azimuth_band", "instant_azimuth_band", "first_sound_side"):
            cellsets = [tuple(b) for b in profile["answer_bands_deg"]]
        elif kind == "event_count":
            cellsets = [int(value) for value in profile["answer_values"]]
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
        coats = [COAT_OF[a1], COAT_OF[a2]]
        if kind in ("azimuth_band", "instant_azimuth_band", "first_sound_side"):
            # Six cells cover the full slot x answer-band table once.  Coat
            # is then balanced within each slot so motion/audio slot cannot
            # deterministically recover appearance.
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
                    answer_coat if target_first else OTHER_COAT[answer_coat])
        else:
            target_joint = balanced_binary_joint(
                slots, coats, n, seed, profile["id"], "slot-coat")
            target_slots = [item[0] for item in target_joint]
            target_coats = [item[1] for item in target_joint]
            answer_coats = list(target_coats)
        per_profile[profile["id"]] = {
            "band": band_alloc,
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
                "target_band": alloc["band"][index],
                "answer_value": alloc["band"][index],
                "target_first": bool(alloc["target_first"][index]),
                "answer_coat": alloc["answer_coat"][index],
                "target_slot": alloc["target_slot"][index],
                "target_moves_more": alloc["target_moves_more"][index],
            }
            target_coat = alloc["target_coat"][index]
            entry["target_coat"] = target_coat
            target_asset = ASSET_OF[target_coat]
            other_asset = ASSET_OF[OTHER_COAT[target_coat]]
            entry["pair_assets"] = ((target_asset, other_asset)
                                    if entry["target_slot"] == "source1"
                                    else (other_asset, target_asset))
            plan.append(entry)
    return plan


def materialize_derived_params(params):
    """Replace stale card8 input text with the interval derived by this run.

    Card8's production path has used card8_band_edges since run02, but early
    external parameter files still carried run01's dead three-band field.
    Manifests must describe what execution actually used.
    """
    effective = copy.deepcopy(params)
    effective["BANDS_CARD8"] = AP.card8_band_edges(effective)
    effective["BANDS_CARD8_note"] = (
        "Derived before generation by audio_profiles.card8_band_edges from "
        "clip/event/gap/first-min constraints; not an independent input.")
    return effective


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-config", required=True, type=Path)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--cells", type=int, default=6)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--snapshot-content", default=(
        "/data/avengine_external/ue-assets/"
        "actor_content_registry_v9_20260823T033709Z/cpp/unreal_projects/"
        "SpearSim/Content"))
    args = parser.parse_args(argv)

    if args.out_root.exists():
        print(f"refusing to overwrite: {args.out_root}", file=sys.stderr)
        return 2
    scene_cfg = json.loads(args.scene_config.read_text())
    profiles = json.loads(args.profiles.read_text())
    params = json.loads(args.params.read_text())
    validate_profiles(profiles)
    scene = SS.load_scene(scene_cfg)
    # Validate all render facts before creating the fresh output directory.
    # Missing ground/map/transform is configuration failure, not a partially
    # realised candidate that should poison the no-clobber path.
    resolve_scene_render_context(scene)
    params = materialize_derived_params(params)
    base_request = json.loads(Path(scene_cfg["camera_base_request"]).read_text())
    registry = json.loads(
        (REPO / "examples/runtime/source_asset_runtime_profiles.json").read_text())
    by_id = {a["asset_id"]: a for a in registry["assets"]}
    pair = (list(COAT_WORDS)[0], list(COAT_WORDS)[1])

    args.out_root.mkdir(parents=True)
    programs_dir = args.out_root / "programs"
    programs_dir.mkdir()
    ledger = SS.RejectionLedger()
    # 逐 profile 一本台账:总表看不出哪条约束在影响哪个题型
    per_profile_ledger = {p["id"]: SS.RejectionLedger() for p in profiles}
    cells = build_cell_plan(args.cells, profiles, pair, params, args.seed)

    made, rejected, records = [], [], []
    for cell in cells:
        profile = cell["profile"]
        pid = f"{profile['id']}_{cell['cell_index'] + 1:03d}"
        rng = sha_rng(args.seed, pid)
        outcome = solve_for_profile(profile, cell, scene, params, rng,
                                    per_profile_ledger[profile["id"]])
        if isinstance(outcome, SS.Rejection):
            rejected.append({"point_id": pid, "reason": outcome.reason,
                             "detail": outcome.detail})
            continue
        try:
            record = realise_point(pid, cell, outcome, scene, base_request,
                                   params, by_id, args, programs_dir, rng)
        except GenerationConstraintError as exc:
            rejected.append({"point_id": pid,
                             "reason": "generation_constraint_failed",
                             "detail": str(exc)[:240]})
            continue
        made.append(pid)
        records.append(record)

    for sub in per_profile_ledger.values():
        for reason, count in sub.counts.items():
            ledger.counts[reason] = ledger.counts.get(reason, 0) + count
        ledger.combinations_evaluated += sub.combinations_evaluated
        ledger.stand_points_evaluated += sub.stand_points_evaluated
        ledger.budget_exhausted += sub.budget_exhausted
    write_outputs(args, scene, scene_cfg, profiles, params, ledger, made,
                  rejected, records, per_profile_ledger)
    print(json.dumps({"out": str(args.out_root), "scene": scene.scene_id,
                      "geometry_candidates": len(made),
                      "cells_requested": len(cells),
                      "rejected": len(rejected),
                      "combinations_evaluated":
                          ledger.summary()["combinations_evaluated"],
                      "evidence_class": "geometry_candidate"},
                     ensure_ascii=False))
    return 0


def realise_point(pid, cell, plan, scene, base_request, params, by_id, args,
                  programs_dir, rng):
    profile = cell["profile"]
    pdir = args.out_root / pid
    pdir.mkdir()
    target_slot = cell["target_slot"]
    other_slot = "source2" if target_slot == "source1" else "source1"
    assets = cell["pair_assets"]
    selection = _selection_doc(assets[0], assets[1], by_id,
                               args.snapshot_content)
    (pdir / "actor_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2))
    slot_asset = {a["source_slot_id"]: a["asset_id"] for a in selection["actors"]}
    slot_coat = {s: COAT_WORDS[a] for s, a in slot_asset.items()}

    # 相机与听者:同一份姿态结果
    render_context = resolve_scene_render_context(scene)
    camera_ue_cm = [plan.camera_xy[0], plan.camera_xy[1],
                    render_context["ground_z_ue_cm"]
                    + scene.camera_height_m * 100.0]
    camera_world_m = render_context["world_transform"](camera_ue_cm)
    m1_request = apply_camera_listener_pose_ue(
        base_request, request_id=f"qa_v3_{pid}", position_m=camera_world_m,
        ue_yaw_degrees=plan.camera_ue_yaw_deg,
        horizontal_fov_deg=scene.hfov_deg)
    (pdir / "m1_capture_request.json").write_text(
        json.dumps(m1_request, ensure_ascii=False, indent=2))

    # 题型专用音频调度:语义角色 → 槽位
    if profile["id"] == "card1F":
        schedule = AP.schedule_forward_anchor(
            rng, params=params, anchor_frame=plan.anchor_frame)
    elif profile["id"] == "card1B":
        schedule = AP.schedule_backward_anchor(
            rng, params=params, anchor_frame=plan.anchor_frame,
            query_frame=plan.query_frame)
    elif profile.get("answer_kind") == "event_count":
        schedule = AP.schedule_event_count(
            rng, params=params, event_count=int(cell["answer_value"]))
    elif profile.get("answer_kind") == "first_sound_side":
        schedule = AP.schedule_first_sound_at_frame(
            rng, params=params, query_frame=plan.query_frame)
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
            first_caller_role=(AP.TARGET if cell["target_first"] else AP.OTHER))
    else:
        schedule = AP.schedule_exactly_one_calling(
            rng, params=params, query_frame=plan.query_frame)
    main_role_to_slot = {AP.TARGET: target_slot, AP.OTHER: other_slot}
    gatea_role_to_slot = {AP.TARGET: other_slot, AP.OTHER: target_slot}
    slot_events = schedule.bind(main_role_to_slot)
    gatea_slot_events = schedule.bind(gatea_role_to_slot)

    request = {"pair_kind": "dog", "point_id": pid,
               "endpoint_1": EP_MAP[assets[0]][0],
               "endpoint_2": EP_MAP[assets[1]][1],
               "sound_asset_id": params["SOUND_ASSET"]}
    program = build_program(request, slot_events)
    gatea_program = build_program(
        request, gatea_slot_events, revision="gateA_v1")
    (programs_dir / f"{program['program_id']}.json").write_text(
        json.dumps(program, ensure_ascii=False, indent=1))
    (programs_dir / f"{gatea_program['program_id']}.json").write_text(
        json.dumps(gatea_program, ensure_ascii=False, indent=1))

    # 时间线:相机与折线都来自求解结果
    base_route = plan.base_route.samples_xy      # 未平移:创作用原路线
    other_route = plan.other_route.samples_xy
    z = render_context["ground_z_ue_cm"]
    routes = {target_slot: (base_route[0], base_route[-1], base_route),
              other_slot: (other_route[0], other_route[-1], other_route)}
    s1, s2 = routes["source1"], routes["source2"]
    timeline = author_current_apartment_visual_timeline(
        actor_selection_path=pdir / "actor_selection.json",
        source_asset_registry_path=(
            REPO / "examples/runtime/source_asset_runtime_profiles.json"),
        output_path=pdir / "timeline_authored.json",
        camera_position_ue_cm=camera_ue_cm,
        camera_yaw_deg=plan.camera_ue_yaw_deg,
        human_start_ue_cm=[s1[0][0], s1[0][1], z],
        human_end_ue_cm=[s1[1][0], s1[1][1], z],
        beagle_start_ue_cm=[s2[0][0], s2[0][1], z],
        beagle_end_ue_cm=[s2[1][0], s2[1][1], z],
        human_waypoints_ue_cm=([[p[0], p[1], z] for p in s1[2]]
                               if s1[2] else None),
        beagle_waypoints_ue_cm=([[p[0], p[1], z] for p in s2[2]]
                                if s2[2] else None),
        native_map=render_context["native_map"],
        room_profile_id=render_context["room_profile_id"],
        hfov_degrees=scene.hfov_deg,
    )
    if plan.idle_frames:
        timeline = transform_idle_then_walk(timeline, target_slot,
                                            plan.idle_frames)
    (pdir / "timeline.json").write_text(
        json.dumps(timeline, ensure_ascii=False, indent=1))

    # 在最终相机姿态下重算真值,并与分配的答案格核对
    query_frame = plan.query_frame
    answer_kind = profile.get("answer_kind", "azimuth_band")
    # 真值一律在**最终时间线**上重算(相机姿态已应用),不沿用求解器角度
    truth_deg = recompute_azimuth(timeline, target_slot, query_frame)
    other_deg = recompute_azimuth(timeline, other_slot, query_frame)
    answer = build_answer(answer_kind, profile, cell, timeline, schedule,
                          slot_events, target_slot, other_slot, slot_coat,
                          truth_deg, query_frame, params)

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
    }
    motion["both_roles_move"] = (
        motion["source1_displacement_cm"] > 0.0
        and motion["source2_displacement_cm"] > 0.0)
    observed_target_moves_more = (
        motion[f"{target_slot}_displacement_cm"]
        > motion[f"{other_slot}_displacement_cm"])
    motion["target_moves_more"] = observed_target_moves_more
    motion["allocated_target_moves_more"] = bool(cell["target_moves_more"])
    if not motion["both_roles_move"]:
        raise GenerationConstraintError(
            f"{pid}: dual-motion profile produced a static role: {motion}")
    if observed_target_moves_more != bool(cell["target_moves_more"]):
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
        "slot_coat": slot_coat,
        "camera": {"ue_cm": camera_ue_cm,
                   "ue_yaw_deg": plan.camera_ue_yaw_deg,
                   "listener_from_same_pose_result": True},
        "room": {
            "native_map": timeline["room"]["map_path"],
            "room_profile_id": timeline["room"]["room_profile_id"],
            "world_transform": render_context["world_transform_id"],
            "ground_z_ue_cm": render_context["ground_z_ue_cm"],
        },
        "motion": motion,
        "answer_kind": answer_kind,
        "truth": dict(answer["truth"],
                      query_azimuth_deg=round(truth_deg, 3),
                      other_slot_azimuth_deg=round(other_deg, 3),
                      recomputed_after_camera_pose=True),
        "mcq": answer["mcq"],
        "open": answer["open"],
        "audio": {"program_id": program["program_id"],
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
            gatea_slot_events, target_slot, other_slot, slot_coat,
            query_frame, params)
    gatea_checks = audit_gatea_pair(
        profile, program, gatea_program, answer, gatea_answer, params)
    gatea_binding_check = validate_anchor_binding(
        profile, schedule, gatea_slot_events,
        target_slot=gatea_target_slot, query_frame=plan.query_frame,
        answer=gatea_answer)
    gatea_fact = copy.deepcopy(fact)
    gatea_fact.update({
        "variant": "gateA",
        "gatea_of": pid,
        "target_slot": gatea_target_slot,
        "target_coat": slot_coat[gatea_target_slot],
        "truth": dict(
            gatea_answer["truth"],
            query_azimuth_deg=round(gatea_truth_deg, 3),
            other_slot_azimuth_deg=round(gatea_other_deg, 3),
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


def build_gatea_answer(kind, profile, cell, timeline, schedule,
                       gatea_slot_events, target_slot, other_slot, slot_coat,
                       query_frame, params):
    """Derive Gate A gold from the unchanged visual timeline.

    Cross-time and query-caller cards select a different visual actor after
    the audio slot swap.  Card8 keeps the text-selected visual actor but
    inherits the other role's onset.  Card9 has no fixed visual target; the
    first-caller relation simply reverses.
    """
    gatea_cell = dict(cell)
    if kind in ("azimuth_band", "instant_azimuth_band", "first_sound_side"):
        gatea_target_slot, gatea_other_slot = other_slot, target_slot
        gatea_truth_deg = recompute_azimuth(
            timeline, gatea_target_slot, query_frame)
        bands = [tuple(b) for b in profile["answer_bands_deg"]]
        gatea_band = next((band for band in bands
                           if band[0] <= gatea_truth_deg < band[1]), None)
        if gatea_band is None:
            raise GenerationConstraintError(
                f"Gate A angular truth {gatea_truth_deg:.2f} is outside all "
                "declared MCQ bands")
        gatea_cell["answer_band"] = gatea_band
    elif kind == "coat_at_query":
        gatea_target_slot, gatea_other_slot = other_slot, target_slot
        gatea_truth_deg = recompute_azimuth(
            timeline, gatea_target_slot, query_frame)
    elif kind == "time_band":
        gatea_target_slot, gatea_other_slot = target_slot, other_slot
        gatea_truth_deg = recompute_azimuth(
            timeline, gatea_target_slot, query_frame)
        firsts = {}
        for slot, start in gatea_slot_events:
            firsts.setdefault(slot, start / AP.SAMPLE_RATE)
        gatea_band = band_of(firsts[gatea_target_slot],
                             AP.card8_band_edges(params))
        if gatea_band is None:
            raise ValueError("Gate A card8 onset is outside the declared bands")
        gatea_cell["target_band"] = gatea_band
    elif kind == "first_caller_coat":
        gatea_target_slot, gatea_other_slot = target_slot, other_slot
        gatea_truth_deg = recompute_azimuth(
            timeline, gatea_target_slot, query_frame)
        gatea_cell["target_first"] = not bool(cell["target_first"])
    elif kind == "event_count":
        gatea_target_slot, gatea_other_slot = target_slot, other_slot
        gatea_truth_deg = recompute_azimuth(
            timeline, gatea_target_slot, query_frame)
    else:
        raise ValueError(f"no Gate A answer derivation for {kind!r}")
    gatea_other_deg = recompute_azimuth(
        timeline, gatea_other_slot, query_frame)
    gatea_answer = build_answer(
        kind, profile, gatea_cell, timeline, schedule, gatea_slot_events,
        gatea_target_slot, gatea_other_slot, slot_coat, gatea_truth_deg,
        query_frame, params)
    return (gatea_target_slot, gatea_other_slot, gatea_answer,
            gatea_truth_deg, gatea_other_deg)


def build_answer(kind, profile, cell, timeline, schedule, slot_events,
                 target_slot, other_slot, slot_coat, truth_deg, query_frame,
                 params):
    """按题型的答案空间造真值;MCQ 与 Open 引用**同一条**事实。"""
    coat = slot_coat[target_slot]
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
    if kind == "first_sound_side":
        firsts = {}
        for slot, start in slot_events:
            firsts.setdefault(slot, start)
        first_slot = min(firsts, key=firsts.get)
        if first_slot != target_slot:
            raise GenerationConstraintError(
                f"first caller {first_slot} does not match target {target_slot}")
        bands = [tuple(band) for band in profile["answer_bands_deg"]]
        got = next((index for index, (lo, hi) in enumerate(bands)
                    if lo <= truth_deg < hi), None)
        want = bands.index(tuple(cell["answer_band"]))
        if got != want:
            raise GenerationConstraintError(
                f"first-sound azimuth {truth_deg:.2f} lands in band {got}, "
                f"not allocated band {want}")
        side = "left" if truth_deg < 0.0 else "right"
        moment = (f"At zero-based video frame index {query_frame} "
                  f"({query_frame}/15 seconds)")
        return {
            "first_caller_slot": first_slot,
            "truth": {
                "first_sound_side": side,
                "first_sound_azimuth_deg": round(truth_deg, 3),
            },
            "mcq": {
                "stem": (f"{moment}, did the first sound come from the left "
                         "or right side relative to your facing direction?"),
                "options_space": ["left", "right"],
                "truth_option": side,
            },
            "open": {
                "stem": (f"{moment}, which side did the first sound come "
                         "from: left or right?"),
                "truth_value": side,
                "scoring": "closed_set",
            },
        }
    if kind in ("azimuth_band", "instant_azimuth_band"):
        bands = [tuple(b) for b in profile["answer_bands_deg"]]
        labels = [f"[{lo:g}, {hi:g})" for lo, hi in bands]
        got = next((i for i, (lo, hi) in enumerate(bands)
                    if lo <= truth_deg < hi), None)
        want = bands.index(tuple(cell["answer_band"]))
        if got != want:
            raise GenerationConstraintError(
                f"recomputed truth {truth_deg:.2f} deg lands in band {got}, "
                f"not the assigned {want}: the final camera pose disagrees "
                "with the solver's geometry")
        if profile["temporal"] == "forward":
            moment = "At the end of the video"
            referent = "the dog that barked last"
        elif profile["temporal"] == "backward":
            moment = (f"At zero-based video frame index {query_frame} "
                      f"({query_frame}/15 seconds)")
            referent = "the dog that barked last"
        elif profile["temporal"] == "instant":
            moment = (f"At zero-based video frame index {query_frame} "
                      f"({query_frame}/15 seconds)")
            referent = "the dog barking at that frame"
        else:
            raise ValueError("azimuth-band profile must declare a time direction")
        return {"truth": {"band_index": got},
                "mcq": {"stem": (f"{moment}, which azimuth band relative to "
                                 f"your facing direction contains {referent}? "
                                 "Right is positive."),
                        "options_space": labels, "truth_option": labels[got]},
                "open": {"stem": (f"{moment}, roughly how many degrees from "
                                  f"your facing direction is {referent}? "
                                  "Right is positive."),
                         "truth_value": round(truth_deg, 3), "unit": "deg",
                         "scoring": "circular_deg"}}
    if kind == "coat_at_query":
        calling = [slot for slot, event in zip(
            [s for s, _ in slot_events], schedule.events)
            if event.frame_span()[0] <= query_frame < event.frame_span()[1]]
        if len(calling) != 1:
            raise ValueError(f"{len(calling)} actors sound at the query frame")
        truth = slot_coat[calling[0]]
        options = ["black-and-white", "yellow", "both", "neither"]
        seconds = query_frame / 15.0
        moment = (f"At zero-based video frame index {query_frame} "
                  f"({query_frame}/15 seconds)")
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
        firsts = {}
        for (slot, start), event in zip(slot_events, schedule.events):
            firsts.setdefault(slot, start / AP.SAMPLE_RATE)
        onset = firsts[target_slot]
        got = next((i for i in range(len(edges) - 1)
                    if edges[i] <= onset < edges[i + 1]), None)
        want = int(cell["target_band"])
        if got != want:
            raise ValueError(f"first call landed in band {got}, assigned {want}")
        labels = [f"[{edges[i]:g}, {edges[i + 1]:g})"
                  for i in range(len(edges) - 1)]
        other_onset = firsts.get(other_slot)
        return {"first_caller_slot": min(firsts, key=firsts.get),
                "truth": {"first_onset_s": round(onset, 4), "band_index": got,
                          "non_target_first_onset_s": (round(other_onset, 4)
                                                       if other_onset else None)},
                "mcq": {"stem": (f"When does the {coat} dog bark for the FIRST "
                                 "time? Pick the time range in seconds."),
                        "options_space": labels, "truth_option": labels[got]},
                "open": {"stem": (f"At how many seconds does the {coat} dog "
                                  "bark for the first time?"),
                         "truth_value": round(onset, 4), "unit": "s",
                         "scoring": "absolute_time"}}
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
                              / AP.SAMPLE_RATE, 4)},
                "mcq": {"stem": ("Which dog barked first, the black-and-white "
                                 "one or the yellow one?"),
                        "options_space": ["black-and-white", "yellow"],
                        "truth_option": truth},
                "open": {"stem": "Which dog made the first sound?",
                         "truth_value": truth, "scoring": "closed_set"}}
    raise ValueError(f"unknown answer kind {kind!r}")


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
    """⑧ 的时间域逐项列明,证明它没有继承 ①F 的片尾静默。"""
    lo, hi = AP.card8_feasible_interval(params)
    edges = AP.card8_band_edges(params)
    rows = [r for r in records if r.get("answer_kind") == "time_band"]
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
    return {
        "clip_duration_s": float(params.get("CLIP_SECONDS", AP.CLIP_SECONDS)),
        "event_seconds": float(params.get("EVENT_SECONDS", AP.EVENT_SECONDS)),
        "card8_feasible_interval_s": [lo, hi],
        "card8_mcq_band_edges_s": edges,
        "derivation": ("clip - event - (min_events - 2) * (event + gap); "
                       "no tail silence is applied to card8"),
        "target_first_onset_min_s": min(target_onsets) if target_onsets else None,
        "target_first_onset_max_s": max(target_onsets) if target_onsets else None,
        "non_target_first_onset_min_s": min(other_onsets) if other_onsets else None,
        "non_target_first_onset_max_s": max(other_onsets) if other_onsets else None,
        "target_band_counts": counts,
        "target_first_counts": firsts,
        "empty_bands": [str(i) for i in range(len(edges) - 1)
                        if str(i) not in counts],
    }


def write_outputs(args, scene, scene_cfg, profiles, params, ledger, made,
                  rejected, records, per_profile_ledger=None):
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
        },
        "scene": {"scene_id": scene.scene_id, "backend": scene.backend,
                  "scene_asset_id": scene_cfg.get("scene_asset_id",
                                                  scene.scene_id),
                  "route_domain": scene_cfg.get("route_domain"),
                  "bank_adapter": scene.provenance.get("bank_adapter"),
                  "routes_loaded": scene.provenance.get("routes_loaded"),
                  "line_of_sight_screened": scene.line_of_sight_screened},
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
