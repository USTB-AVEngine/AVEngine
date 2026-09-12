"""Preallocate independent QA requests and split connected Episode groups.

Counters exist only while preparing a batch. Executing any saved request needs
no batch state and cannot replace a failed profile with an easier one.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from itertools import product
import json
import math
import random
from typing import Any, Mapping, Sequence

from avengine.rooms.conditioned_sampler import (
    CLIP_SPAN_FIT_POLICY,
    histogram_separation_5deg,
    neutral_source_declaration,
    resolve_condition_profile,
    sound_matches,
)

from avengine.dataset.production_spec import (
    CORE_TASK_FAMILIES,
    CoreGroupRequest,
    ProductionSpecError,
    StageResult,
    deep_merge_mappings,
    group_blockers,
    group_stage_units,
    initial_group_work_items,
    initial_stage_work_items,
    next_group_work_items,
    next_stage_work_items,
    parse_production_config,
    production_request_from_legacy,
    recipe_for_task_family,
    stage_protocol_summary,
)
from avengine.qa.failure_accounting import classify_failure

SOURCE_CLASSES = ("articulated_human", "articulated_animal", "rigid_static_object")
from avengine.qa.unified_catalog import QA_IDS
SOURCE_CLASS_LABEL = {
    "articulated_human": "human",
    "articulated_animal": "animal",
    "rigid_static_object": "device",
}
CLASS_PAIRS = (
    ("articulated_human", "articulated_human"),
    ("articulated_human", "articulated_animal"),
    ("articulated_human", "rigid_static_object"),
    ("articulated_animal", "articulated_animal"),
    ("articulated_animal", "rigid_static_object"),
    ("rigid_static_object", "rigid_static_object"),
)
CONDITION_GROUPS = (
    "identity_binding",
    "audio_event_relations",
    "visibility_occlusion",
    "motion_distance",
    "post_sound_state",
)
COMMON_PROFILE = {
    "separation_floor_deg": 15,
    "anchor_visibility": "in_fov",
    "competitor_visibility": "in_fov",
    "anchor_line_of_sight": "clear",
    "competitor_set": "all_other_entities_including_offscreen",
    "separation_window": "whole_audible_window_of_anchor",
    "min_gap_between_audible_windows_s": 0.5,
    "reserve_tail_s": 3.0,
    "minimum_overlap_s": 0.3,
    "retry_budget_within_profile": 200,
    "anchor_count": 1,
    "separation_bin_deg": [30, 60],
    "speech_motion": "all_still",
    "event_relation": "sequential",
    "distance_range_m": [1.5, 4.5],
    "separation_target_policy": "any_legal_in_bin",
}
# The one condition group each core task family exercises. A config may name
# condition_group itself; this table is only the default for a member that does
# not, and every row records which of the two it used.
CONDITION_GROUP_BY_TASK_FAMILY = {
    "visible_binding": "identity_binding",
    "visual_conditioned_relation": "audio_event_relations",
    "cross_event_identity": "identity_binding",
    "cross_time_state": "post_sound_state",
}
# What configs written before config.qa existed asked for. Kept so an old
# config still preallocates the same rows; a V1 config states its own quota.
LEGACY_QA_QUOTA_BY_QA = {qa: (3 if qa == "QA-25" else 1) for qa in QA_IDS}
LEGACY_ITEMS_PER_TYPE = 1
GROUP_PROFILE = {
    "identity_binding": {},
    "audio_event_relations": {"event_relation": "overlap", "anchor_count": 2},
    "visibility_occlusion": {"anchor_line_of_sight": "occluded", "separation_bin_deg": [15, 30]},
    "motion_distance": {"speech_motion": "speaker_moving"},
    "post_sound_state": {"separation_bin_deg": [60, 90]},
}




def _config_distance_range_m(config: Mapping[str, Any]) -> Sequence[float] | None:
    """Read distance_range_m from the config root or scaleup block."""
    value = config.get("distance_range_m")
    if value is None and isinstance(config.get("scaleup"), Mapping):
        value = config["scaleup"].get("distance_range_m")
    return value


def _profile_for_match(profile: Mapping[str, Any] | None) -> Any:
    """Fill portrait keys omitted by old manifests with sampler defaults."""
    if not isinstance(profile, Mapping):
        return profile
    filled = deepcopy(dict(profile))
    for key, default in (
        ("competitor_visibility", COMMON_PROFILE["competitor_visibility"]),
        ("distance_range_m", list(COMMON_PROFILE["distance_range_m"])),
        ("separation_target_policy", COMMON_PROFILE["separation_target_policy"]),
    ):
        filled.setdefault(key, deepcopy(default))
    distance = filled.get("distance_range_m")
    if isinstance(distance, (list, tuple)) and len(distance) == 2:
        filled["distance_range_m"] = [float(distance[0]), float(distance[1])]
    return filled


def class_pair_label(classes: Sequence[str]) -> str:
    labels = sorted(
        SOURCE_CLASS_LABEL[value]
        for value in classes
        if value in SOURCE_CLASS_LABEL
    )
    return "-".join(labels)


def legal_condition_groups(classes: Sequence[str], *, silent_count: int = 0) -> list[str]:
    groups = list(CONDITION_GROUPS)
    speaking = len(classes) - int(silent_count)
    if not any(value != "rigid_static_object" for value in classes):
        groups = [group for group in groups if group != "motion_distance"]
    if speaking < 2:
        groups = [group for group in groups if group != "audio_event_relations"]
    return groups


def profile_for_condition_group(
    group: str,
    classes: Sequence[str],
    *,
    silent_count: int = 0,
    event_relation: str | None = None,
    off_screen: str | None = None,
    distance_range_m: Sequence[float] | None = None,
) -> dict[str, Any]:
    if group not in GROUP_PROFILE:
        raise ValueError(f"unknown condition_group: {group}")
    profile = deepcopy(COMMON_PROFILE)
    profile.update(deepcopy(GROUP_PROFILE[group]))
    speaking = len(classes) - int(silent_count)
    if int(profile.get("anchor_count", 1)) > max(1, speaking):
        profile["anchor_count"] = max(1, speaking)
    if event_relation is not None:
        profile["event_relation"] = event_relation
    if group == "identity_binding" and class_pair_label(classes) == "device-device" and event_relation is None:
        profile["event_relation"] = "repeat"
    if off_screen == "anchor":
        profile["anchor_visibility"] = "off_screen"
    elif off_screen == "competitor":
        profile["competitor_visibility"] = "off_screen"
    if distance_range_m is not None:
        profile["distance_range_m"] = [float(distance_range_m[0]), float(distance_range_m[1])]
    return profile


def format_class_pair_condition_group_crosstab(table: Mapping[str, Any]) -> str:
    """Printable class-pair x condition-group counts for dry-run reports."""
    groups = list(table.get("condition_groups") or CONDITION_GROUPS)
    pairs = list(table.get("class_pairs") or [])
    counts = table.get("counts") or {}
    distinct = table.get("distinct_condition_groups_per_class_pair") or {}
    header = ["class_pair", *groups, "n_groups"]
    lines = ["\t".join(header)]
    for pair in pairs:
        row = [pair]
        for group in groups:
            row.append(str(int((counts.get(pair) or {}).get(group, 0))))
        row.append(str(int(distinct.get(pair, 0))))
        lines.append("\t".join(row))
    lines.append(
        "meets_acceptance=%s min_distinct=%s required=%s"
        % (table.get("meets_acceptance"), table.get("min_distinct_groups_per_class_pair"),
           table.get("acceptance_min_distinct_groups"))
    )
    return "\n".join(lines)


def class_pair_condition_group_crosstab(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    table = defaultdict(Counter)
    for row in rows:
        classes = row.get("requested_source_classes") or row.get("source_classes") or []
        pair = class_pair_label(classes)
        table[pair][row["condition_group"]] += 1
    pairs = sorted(table)
    distinct = {pair: sum(1 for group in CONDITION_GROUPS if table[pair][group]) for pair in pairs}
    return {
        "class_pairs": pairs,
        "condition_groups": list(CONDITION_GROUPS),
        "counts": {pair: {group: int(table[pair][group]) for group in CONDITION_GROUPS} for pair in pairs},
        "distinct_condition_groups_per_class_pair": distinct,
        "min_distinct_groups_per_class_pair": min(distinct.values()) if distinct else 0,
        "acceptance_min_distinct_groups": 3,
        "meets_acceptance": bool(distinct) and all(value >= 3 for value in distinct.values()),
    }


def scatter_condition_groups(
    slots: Sequence[Mapping[str, Any]],
    *,
    rooms_by_id: Mapping[str, Mapping[str, Any]],
    seed: int,
    min_groups_per_class_pair: int = 3,
    distance_range_m: Sequence[float] | None = None,
    keep_existing_repeat: bool = False,
) -> list[dict[str, Any]]:
    """Assign condition groups by seed so class pairs are not collinear with groups."""
    rng = random.Random(seed)
    slots = [deepcopy(dict(slot)) for slot in slots]
    items = []
    for index, slot in enumerate(slots):
        classes = list(slot["source_classes"])
        items.append({
            "index": index,
            "pair": class_pair_label(classes),
            "family": rooms_by_id[slot["room_id"]]["family"],
            "legal": legal_condition_groups(classes, silent_count=int(slot.get("silent_count", 0))),
            "keep_repeat": bool(keep_existing_repeat and (slot.get("profile") or {}).get("event_relation") == "repeat"),
        })
    assigned = [None] * len(slots)
    by_pair = defaultdict(list)
    for item in items:
        by_pair[item["pair"]].append(item["index"])
    for pair, indexes in by_pair.items():
        legal = items[indexes[0]]["legal"]
        need = min(min_groups_per_class_pair, len(legal), len(indexes))
        groups = list(legal)
        rng.shuffle(groups)
        chosen = groups[:need]
        order = list(indexes)
        rng.shuffle(order)
        for group, index in zip(chosen, order):
            assigned[index] = group
    families = sorted({item["family"] for item in items})
    for family in families:
        family_indexes = [item["index"] for item in items if item["family"] == family]
        have = {assigned[index] for index in family_indexes if assigned[index]}
        missing = [group for group in CONDITION_GROUPS if group not in have]
        rng.shuffle(family_indexes)
        for group in missing:
            candidates = [index for index in family_indexes
                          if group in items[index]["legal"] and assigned[index] is None]
            if not candidates:
                candidates = [index for index in family_indexes if group in items[index]["legal"]]
            if candidates:
                assigned[candidates[0]] = group
    for item in items:
        if assigned[item["index"]] is None:
            assigned[item["index"]] = rng.choice(item["legal"])
    for pair, indexes in by_pair.items():
        legal = items[indexes[0]]["legal"]
        used = {assigned[index] for index in indexes}
        while len(used) < min(min_groups_per_class_pair, len(legal), len(indexes)):
            missing = [group for group in legal if group not in used]
            counts = Counter(assigned[index] for index in indexes)
            donors = [index for index in indexes if counts[assigned[index]] > 1]
            if not missing or not donors:
                break
            assigned[donors[0]] = missing[0]
            used = {assigned[index] for index in indexes}
    for slot, item, group in zip(slots, items, assigned):
        event_relation = "repeat" if item["keep_repeat"] else None
        slot_distance = distance_range_m
        if slot_distance is None:
            slot_distance = (slot.get("profile") or {}).get("distance_range_m")
        slot["condition_group"] = group
        slot["profile"] = profile_for_condition_group(
            group, slot["source_classes"], silent_count=int(slot.get("silent_count", 0)),
            event_relation=event_relation, off_screen=slot.get("off_screen"),
            distance_range_m=slot_distance)
        slot["class_pair"] = item["pair"]
    return slots


def build_scaleup_slots(
    rooms: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    episodes_per_room: int = 50,
    include_off_screen: bool = True,
    distance_range_m: Sequence[float] = (1.5, 6.0),
    batch_id: str = "qa_scaleup",
) -> list[dict[str, Any]]:
    if episodes_per_room < len(CLASS_PAIRS) + 1:
        raise ValueError("episodes_per_room must fit every class pair plus a silent cell")
    rng = random.Random(seed)
    slots = []
    for room in rooms:
        room_slots = []
        for classes in CLASS_PAIRS:
            room_slots.append({"room_id": room["room_id"], "source_classes": list(classes), "silent_count": 0})
        room_slots.append({"room_id": room["room_id"],
                           "source_classes": ["articulated_human", "articulated_human"], "silent_count": 1})
        while len(room_slots) < episodes_per_room:
            room_slots.append({"room_id": room["room_id"], "source_classes": list(rng.choice(CLASS_PAIRS)),
                               "silent_count": 0})
        if include_off_screen:
            room_slots[0]["off_screen"] = "anchor"
            competitor = next((slot for slot in room_slots
                               if class_pair_label(slot["source_classes"]) != "device-device"
                               and slot.get("silent_count", 0) == 0 and slot is not room_slots[0]), room_slots[1])
            competitor["off_screen"] = "competitor"
        slots.extend(room_slots[:episodes_per_room])
    rooms_by_id = {room["room_id"]: room for room in rooms}
    slots = scatter_condition_groups(slots, rooms_by_id=rooms_by_id, seed=seed,
                                     distance_range_m=distance_range_m)
    for index, slot in enumerate(slots):
        pair = slot.get("class_pair") or class_pair_label(slot["source_classes"])
        family = rooms_by_id[slot["room_id"]]["family"]
        slot["episode_id"] = f"{batch_id}_{index + 1:03d}_{family}_{pair.replace('-', '_')}"
        slot["seed"] = seed + index
        if slot.get("off_screen"):
            key = "anchor_visibility" if slot["off_screen"] == "anchor" else "competitor_visibility"
            slot["profile"][key] = "off_screen"
            slot["profile"]["distance_range_m"] = [float(distance_range_m[0]), float(distance_range_m[1])]
    return slots


def build_scaleup_batch_config(
    template: Mapping[str, Any],
    catalog: Mapping[str, Any],
    *,
    seed: int = 20260907,
    episodes_per_room: int = 50,
    batch_id: str | None = None,
    include_off_screen: bool = True,
    distance_range_m: Sequence[float] = (1.5, 6.0),
) -> dict[str, Any]:
    """Deterministic 7-room scale-up config. Does not execute GPU episodes."""
    config = deepcopy(dict(template))
    batch_id = batch_id or f"qa_scaleup_7x{episodes_per_room}_{seed}"
    rooms = catalog["rooms"]
    config["batch_id"] = batch_id
    config["seed"] = int(seed)
    config["scatter_condition_groups"] = True
    config["slots"] = build_scaleup_slots(
        rooms, seed=int(seed), episodes_per_room=int(episodes_per_room),
        include_off_screen=include_off_screen, distance_range_m=distance_range_m, batch_id=batch_id)
    config["scaleup"] = {
        "episodes_per_room": int(episodes_per_room),
        "room_count": len(rooms),
        "include_off_screen_portraits": include_off_screen,
        "distance_range_m": [float(distance_range_m[0]), float(distance_range_m[1])],
        "repeat_feasibility_formula": "2 * duration(repeat_sound) + duration(other_source_sound) + 2 * gap_s <= available_s",
        "gpu_execution": False,
    }
    sound_selection = config.setdefault("base_request", {}).setdefault("sound_selection", {})
    sound_selection.setdefault("clip_span_fit_policy", CLIP_SPAN_FIT_POLICY)
    sound_selection.setdefault("max_clip_s", 5.0)
    return config


def prepare_scaleup_dry_run(
    template: Mapping[str, Any],
    registry: Mapping[str, Any],
    catalog: Mapping[str, Any],
    sounds: Sequence[Mapping[str, Any]],
    *,
    seed: int = 20260907,
    episodes_per_room: int = 50,
    batch_id: str | None = None,
) -> dict[str, Any]:
    """Generate a scattered scale-up manifest without native execution."""
    config = build_scaleup_batch_config(
        template, catalog, seed=seed, episodes_per_room=episodes_per_room, batch_id=batch_id)
    manifest = prepare_batch_manifest(config, registry, catalog, sounds)
    return {"config": config, "manifest": manifest,
            "repeat_deficit_count": int(manifest.get("preallocation_gap_counts", {}).get(
                "fixed_sound_identities_exceed_profile_clip_budget", 0)),
            "class_pair_condition_group_crosstab": manifest["class_pair_condition_group_crosstab"]}


def _clip_duration_s(sound: Mapping[str, Any]) -> float:
    return float(sound["sample_count"]) / float(sound["sample_rate_hz"])


def program_seconds_for_durations(
    durations: Sequence[float], *, gap_s: float, relation: str, repeat_index: int | None = None,
) -> float:
    """Owner rule 1: 2 * repeat + other + 2 * gap for a two-source repeat program."""
    values = [float(value) for value in durations]
    if not values:
        return 0.0
    if relation == "overlap":
        return max(values)
    total = sum(values) + gap_s * max(0, len(values) - 1)
    if relation == "repeat":
        if repeat_index is None:
            raise ValueError("repeat_index is required for a repeat program")
        total += values[repeat_index] + gap_s
    return total


def available_program_seconds(request: Mapping[str, Any], profile: Mapping[str, Any]) -> float:
    return (float(request.get("frame_count", 240)) / float(request.get("frame_rate_hz", 15))) - float(
        profile["reserve_tail_s"])


def iter_achieved_separation_deg(row: Mapping[str, Any]):
    """Yield measured separation angles; never the requested bin label."""
    achieved = row.get("achieved_conditions") or {}
    if not isinstance(achieved, Mapping):
        return
    measurements = achieved.get("anchor_event_measurements")
    if isinstance(measurements, list):
        for item in measurements:
            sep = (item or {}).get("separation") or {}
            if sep.get("status") == "measured" and sep.get("min") is not None:
                yield float(sep["min"])
        return
    sep = achieved.get("separation")
    if isinstance(sep, Mapping) and sep.get("min") is not None:
        yield float(sep["min"])
        return
    planned = row.get("planned_conditions") or achieved.get("planned_conditions") or {}
    value = planned.get("planned_anchor_nearest_competitor_separation_deg") if isinstance(planned, Mapping) else None
    if value is not None:
        yield float(value)


def _declared_preallocation_by_actor(
    request: Mapping[str, Any], instances: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]] | None:
    """Validate and return the request's explicit actor sound allowlists."""
    selection = request.get("sound_selection")
    if not isinstance(selection, Mapping):
        return None
    declared = selection.get("preallocated_sound_asset_ids_by_actor")
    if declared is None:
        return None
    if not isinstance(declared, Mapping):
        raise ValueError(
            "explicit sound preallocation must be an actor-to-sound-ID mapping"
        )
    speaking = {
        str(instance.get("instance_id"))
        for instance in instances
        if instance.get("speaking") is not False
    }
    missing = sorted(speaking - {str(key) for key in declared})
    if missing:
        raise ValueError(
            "explicit sound preallocation misses speaking actors: "
            f"{missing}"
        )
    result: dict[str, list[str]] = {}
    for actor_id in sorted(speaking):
        values = declared.get(actor_id)
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise ValueError(
                "explicit sound preallocation values must be nonempty string lists"
            )
        result[actor_id] = [str(value) for value in values]
    return result


def _bind_identity(assignment, identity, groups):
    entries = sorted(groups[identity], key=lambda sound: sound["sound_asset_id"])
    assignment.update(
        sound_status="preallocated", sound_identity_id=identity,
        sound_asset_ids=[sound["sound_asset_id"] for sound in entries],
        sound_origins=sorted({str(sound.get("source_pcm_path") or sound.get("source_origin")
                                  or sound.get("original_source_uri")) for sound in entries
                              if sound.get("source_pcm_path") or sound.get("source_origin")
                              or sound.get("original_source_uri")}))
    return assignment["sound_asset_ids"]


def _text(value: Any, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner} must be nonempty text")
    return value


def source_class(record: Mapping[str, Any]) -> str:
    value = record["entity_class"]
    return "rigid_static_object" if value in {"rigid_object", "rigid_static_object"} else value


def appearance_assignment(record: Mapping[str, Any]) -> dict[str, Any]:
    """Use observed registry fields only; missing colors stay unknown."""
    attrs = record.get("realized_attributes", {})
    if source_class(record) == "articulated_human":
        field, value = "top_color", attrs.get("top_color")
    elif source_class(record) == "articulated_animal":
        field, value = "coat_profile.value", attrs.get("coat_profile", {}).get("value")
    else:
        field = "finish" if attrs.get("finish") is not None else "body_color"
        value = attrs.get(field)
    return {"field": field, "value": value, "status": "registered" if value is not None else "unknown",
            "source": "source_asset_runtime_registry.realized_attributes"}


def sound_identity(sound: Mapping[str, Any]) -> str | None:
    """Prefer the original speaker/file identity over a crop or event identifier."""
    if sound.get("sound_identity_id"):
        return _text(sound["sound_identity_id"], "sound_identity_id")
    if sound.get("speaker_id"):
        return "speaker:" + _text(sound["speaker_id"], "speaker_id")
    for field in ("source_pcm_path", "source_origin", "original_source_uri"):
        if sound.get(field):
            return "source:" + _text(sound[field], field)
    # A caller must resolve event/crop lineage instead of treating a new crop ID
    # as an independent source for training/evaluation splitting.
    return None


def _minimum_choice(values: Sequence[Any], score, rng: random.Random):
    if not values:
        raise ValueError("cannot allocate from an empty candidate set")
    minimum = min(score(value) for value in values)
    return rng.choice([value for value in values if score(value) == minimum])


def _formal_static_placement_state(
    request: Mapping[str, Any] | None, asset_id: str,
    *, support_placed: bool = True,
) -> tuple[str, list[str]] | None:
    """Classify a formal support-placement input before legacy asset guards.

    A production request with a catalog, bounded config and an asset-specific
    support request is consumed by conditioned_sampler/source_placement. It is
    therefore not an interface gap merely because the old floor materializer
    cannot realize it. Malformed or asset-missing formal input remains an
    evidence gap and is never silently accepted.

    ``support_placed`` says whether this particular asset is one that a support
    surface carries. An articulated actor stands on the navmesh and never gets a
    support request, so demanding one for it reports a gap that nothing can ever
    close. The spec-level checks (catalog, config, a well-formed request list)
    still run for every asset.
    """
    if not isinstance(request, Mapping):
        return None
    spec = request.get("static_source_placement") or request.get(
        "static_source_placements"
    )
    if spec is None:
        return None
    if not isinstance(spec, Mapping):
        return "invalid", ["static_source_placement"]
    missing: list[str] = []
    catalog = spec.get("catalog_path") or spec.get("support_surface_catalog")
    if not isinstance(catalog, str) or not catalog.strip():
        missing.append("catalog_path")
    if not isinstance(spec.get("config") or spec.get("placement_config"), Mapping):
        missing.append("config")
    raw_requests = spec.get("requests")
    if raw_requests is not None:
        if isinstance(raw_requests, (str, bytes)) or not isinstance(raw_requests, Sequence):
            missing.append("requests")
        elif support_placed:
            matches = [
                row for row in raw_requests
                if isinstance(row, Mapping) and str(row.get("asset_id")) == str(asset_id)
            ]
            if not matches:
                missing.append(f"requests[{asset_id}]")
            elif any(
                not isinstance(row.get("support_surface_id"), str)
                or not row.get("support_surface_id").strip()
                for row in matches
            ):
                missing.append(f"support_surface_id[{asset_id}]")
    elif not (
        isinstance(spec.get("qualification_config"), str)
        and spec.get("qualification_config")
        and spec.get("qualification_episode_id")
    ):
        missing.extend(["requests", "qualification_config", "qualification_episode_id"])
    return ("valid", []) if not missing else ("invalid", missing)


def _asset_interface_gaps(
    record: Mapping[str, Any], renderer: str,
    request: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    backend = "spear_unreal" if renderer == "ue_spear" else "habitat"
    bindings = record.get("runtime_backends", {})
    if not bindings.get(backend):
        return [{"state": "interface_not_implemented", "code": "renderer_binding_missing",
                 "asset_id": record["asset_id"], "renderer": renderer,
                 "file": "examples/runtime/source_asset_runtime_profiles.json"}]
    formal = _formal_static_placement_state(
        request, str(record["asset_id"]),
        support_placed=source_class(record) == "rigid_static_object",
    )
    if formal is not None:
        status, missing = formal
        if status == "valid":
            return []
        return [{
            "state": "evidence_missing_or_unsampled",
            "code": "static_placement_input_missing",
            "asset_id": record["asset_id"],
            "renderer": renderer,
            "missing_fields": ",".join(missing),
            "file": "request.static_source_placement",
        }]
    if source_class(record) == "rigid_static_object":
        pose = bindings.get("habitat", {}).get("resting_pose", {})
        if pose.get("attachment_surface") != "floor" or pose.get("base_plane_offset_m") is None:
            return [{"state": "interface_not_implemented", "code": "floor_only_placement_interface",
                     "asset_id": record["asset_id"], "renderer": renderer,
                     "file": "src/avengine/rooms/qa_episode.py"}]
        if abs(float(pose["base_plane_offset_m"])) > 1e-9:
            return [{"state": "interface_not_implemented", "code": "nonzero_resting_base_offset",
                     "asset_id": record["asset_id"], "renderer": renderer,
                     "file": "src/avengine/rooms/qa_episode.py"}]
    return []


def merge_request_overrides(
    base: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """Recursively apply request overrides without mutating the base config.

    Nested mappings merge by key. Lists and scalar values replace the base
    value as a whole, preserving request-level configuration semantics.
    """
    if not isinstance(base, Mapping):
        raise ValueError("base_request must be a mapping")
    if not isinstance(overrides, Mapping):
        raise ValueError("slot.request_overrides must be a mapping")
    return deep_merge_mappings(base, overrides)


def resolve_qa_plan(config: Mapping[str, Any], slot: Mapping[str, Any]) -> dict[str, Any]:
    """Read the QA selection, per-type item count and quota from configuration.

    A config `qa` block applies to the whole batch and a slot `qa` block
    overrides it. Nothing here decides how many items a QA type is worth: when
    neither level says, the row records that it fell back to the pre-config
    default instead of silently owning that number.
    """
    batch_block = config.get("qa")
    slot_block = slot.get("qa")
    for owner, block in (("config.qa", batch_block), ("slot.qa", slot_block)):
        if block is not None and not isinstance(block, Mapping):
            raise ValueError(f"{owner} must be a mapping")
    block = {**dict(batch_block or {}), **dict(slot_block or {})}
    declared_ids = block.get("qa_ids")
    if declared_ids is None:
        qa_ids = list(QA_IDS)
        qa_ids_source = "unified_catalog_all"
    else:
        if not isinstance(declared_ids, list) or not declared_ids:
            raise ValueError("qa.qa_ids must be a nonempty list")
        unknown = [value for value in declared_ids if value not in QA_IDS]
        if unknown:
            raise ValueError(f"qa.qa_ids are not in the unified catalog: {unknown}")
        if len(set(declared_ids)) != len(declared_ids):
            raise ValueError("qa.qa_ids must be distinct")
        qa_ids = list(declared_ids)
        qa_ids_source = "config"
    items_per_type = block.get("items_per_type")
    if items_per_type is None:
        items_per_type, items_source = LEGACY_ITEMS_PER_TYPE, "legacy_default"
    else:
        if isinstance(items_per_type, bool) or not isinstance(items_per_type, int) or items_per_type < 1:
            raise ValueError("qa.items_per_type must be a positive integer")
        items_source = "config"
    quota = block.get("quota_by_qa")
    if quota is None:
        quota_by_qa = {qa: LEGACY_QA_QUOTA_BY_QA[qa] for qa in qa_ids}
        quota_source = "legacy_default"
    else:
        if not isinstance(quota, Mapping) or not quota:
            raise ValueError("qa.quota_by_qa must be a nonempty mapping")
        outside = sorted(set(quota) - set(qa_ids))
        if outside:
            raise ValueError(f"qa.quota_by_qa names QA types outside qa_ids: {outside}")
        for qa, value in quota.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"qa.quota_by_qa[{qa}] must be a positive integer")
        quota_by_qa = {qa: int(quota.get(qa, LEGACY_QA_QUOTA_BY_QA[qa])) for qa in qa_ids}
        quota_source = "config" if set(quota) == set(qa_ids) else "config_partial_legacy_default"
    return {"qa_ids": qa_ids, "qa_ids_source": qa_ids_source,
            "items_per_type": int(items_per_type), "items_per_type_source": items_source,
            "quota_by_qa": quota_by_qa, "quota_source": quota_source,
            "qa_targets": deepcopy(block.get("qa_targets"))}


def resolve_qa_targets(
    qa_plan: Mapping[str, Any],
    *,
    instances: Sequence[Mapping[str, Any]],
    condition: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Name the target entity instances and the event selection for each QA type.

    A declared target is used as written. A derived target reads the resolved
    anchor entities and asks about their own audible window: `anchor_indices`
    says which entity anchors the question, never that the answer is the first
    sound of the program.
    """
    instance_ids = [instance["instance_id"] for instance in instances]
    declared = qa_plan.get("qa_targets")
    if declared is not None:
        if not isinstance(declared, list) or not declared:
            raise ValueError("qa.qa_targets must be a nonempty list")
        targets = []
        for index, raw in enumerate(declared):
            if not isinstance(raw, Mapping):
                raise ValueError("qa.qa_targets entries must be mappings")
            qa_id = _text(raw.get("qa_id"), f"qa_targets[{index}].qa_id")
            if qa_id not in qa_plan["qa_ids"]:
                raise ValueError(f"qa_targets[{index}].qa_id is outside qa_ids: {qa_id}")
            named = raw.get("target_instance_ids")
            if not isinstance(named, list) or not named:
                raise ValueError(f"qa_targets[{index}].target_instance_ids must name an instance")
            unknown = [value for value in named if value not in instance_ids]
            if unknown:
                raise ValueError(
                    f"qa_targets[{index}].target_instance_ids are not in this episode: {unknown}")
            event = raw.get("event")
            if not isinstance(event, Mapping) or not event.get("kind"):
                raise ValueError(f"qa_targets[{index}].event must state its kind")
            targets.append({**deepcopy(dict(raw)),
                            "target_source": raw.get("target_source", "config")})
        return targets
    if condition is not None:
        anchors = [instance_ids[index] for index in condition["anchor_indices"]]
        speaking = [instance_ids[index] for index in condition["speaking_indices"]]
        source = "resolved_anchor_entities"
    else:
        anchors = [instance["instance_id"] for instance in instances]
        speaking = list(anchors)
        source = "unresolved_all_instances"
    return [{"qa_id": qa_id,
             "target_instance_ids": list(anchors),
             "competitor_instance_ids": [value for value in speaking if value not in anchors],
             "event": {"kind": "target_audible_window"},
             "items": int(qa_plan["quota_by_qa"][qa_id]),
             "target_source": source}
            for qa_id in qa_plan["qa_ids"]]


def declared_silent_count(instances: Sequence[Mapping[str, Any]]) -> int | None:
    """How many instances declared themselves silent, or None if none declared.

    A count is only meaningful when the instances state their own flags; an
    episode that states nothing keeps the upstream count.
    """
    if not any(isinstance(row, Mapping) and row.get("speaking") is not None
               for row in instances):
        return None
    return sum(1 for row in instances
               if isinstance(row, Mapping) and row.get("speaking") is False)


def resolve_silent_count(
    instances: Sequence[Mapping[str, Any]], stated: Any, *, owner: str,
) -> int:
    """Reconcile a stated silent_count with the per-instance speaking flags.

    The flags win, because they say *which* instance is silent and a count only
    says how many. A stated count that disagrees is a real contradiction in the
    request and is reported, never quietly replaced.
    """
    declared = declared_silent_count(instances)
    if declared is None:
        return 0 if stated is None else int(stated)
    if stated is not None and int(stated) != declared:
        silent = [row.get("instance_id") for row in instances
                  if isinstance(row, Mapping) and row.get("speaking") is False]
        raise ValueError(
            f"{owner}: entities.silent_count is {int(stated)} but the declared "
            f"instance speaking flags make {declared} instance(s) silent "
            f"({silent}); state one of them, not both")
    if not 0 <= declared < len(instances):
        raise ValueError(
            f"{owner}: the declared speaking flags leave no speaking instance")
    return declared


def entity_instances_for_slot(
    slot: Mapping[str, Any], classes: Sequence[str], assets: Sequence[str] | None
) -> list[dict[str, Any]]:
    """Name one identity per entity instance, not one per registry entry.

    Two instances may resolve the same asset. The instance is the identity, so
    `source1` and `source2` stay separate rows even when they share `asset_id`.
    """
    declared = slot.get("entity_instances")
    if declared is not None:
        if not isinstance(declared, list) or len(declared) != len(classes):
            raise ValueError("slot.entity_instances must supply one entry per source class")
        instances = []
        for index, raw in enumerate(declared):
            if not isinstance(raw, Mapping):
                raise ValueError("slot.entity_instances entries must be mappings")
            instance = {"instance_id": _text(raw.get("instance_id", f"source{index + 1}"),
                                             "entity_instance.instance_id"),
                        "source_class": classes[index]}
            declared_class = raw.get("source_class")
            if declared_class is not None and declared_class != classes[index]:
                raise ValueError(
                    f"entity_instances[{index}].source_class disagrees with source_classes: "
                    f"{declared_class} vs {classes[index]}")
            asset_id = raw.get("asset_id", assets[index] if assets is not None else None)
            if asset_id is not None:
                instance["asset_id"] = _text(asset_id, "entity_instance.asset_id")
            if raw.get("role") is not None:
                instance["role"] = _text(raw["role"], "entity_instance.role")
            speaking = raw.get("speaking")
            if speaking is not None:
                if not isinstance(speaking, bool):
                    raise ValueError(
                        f"entity_instances[{index}].speaking must be true or false, "
                        f"got {speaking!r}")
                # The declared flag is the identity of this instance, not a count.
                # Dropping it here is what made the sampler draw the silent actor
                # at random and silence whichever instance the draw happened to hit.
                instance["speaking"] = speaking
            instances.append(instance)
    else:
        instances = [{"instance_id": f"source{index + 1}", "source_class": kind,
                      **({"asset_id": assets[index]} if assets is not None else {})}
                     for index, kind in enumerate(classes)]
    ids = [instance["instance_id"] for instance in instances]
    if len(set(ids)) != len(ids):
        raise ValueError(f"entity instance_id values must be distinct: {ids}")
    return instances


def shared_asset_instance_gap(instances: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Report the exact place a repeated asset is still refused downstream."""
    bound = [instance.get("asset_id") for instance in instances if instance.get("asset_id")]
    if len(set(bound)) == len(bound):
        return None
    repeated = sorted({value for value in bound if bound.count(value) > 1})
    return {
        "state": "interface_not_implemented",
        "code": "repeated_asset_across_entity_instances",
        "asset_ids": repeated,
        "instance_ids": [instance["instance_id"] for instance in instances],
        "file": "src/avengine/rooms/conditioned_sampler.py",
        "functions": ["resolve_condition_profile", "select_entities"],
        "detail": ("both require a distinct asset_id per source, so two instances of one "
                   "asset cannot be planned or sampled until that rule is relaxed"),
    }


def _member_stage_scope(row: Mapping[str, Any], slot: Mapping[str, Any]) -> dict[str, Any]:
    """Where a core member's stages actually live, and which units deliver it."""
    recipe = recipe_for_task_family(row["task_family"])
    index = slot.get("member_index")
    delivering = None if index is None else recipe.member_unit_ids[int(index)]
    unit = None if delivering is None else recipe.unit(delivering)
    return {
        "kind": "core_group",
        "group_id": row["group_id"],
        "task_family": row["task_family"],
        "member_index": index,
        "delivering_unit_id": delivering,
        "consumes_visual_unit_id": None if unit is None else unit.visual_unit_id,
        "entry_point": "avengine.qa.batch_manifest.stage_work_items_for_group",
        "reason": "one visual and one audio column are shared, so stages are group scoped",
    }


def _condition_group_for_slot(slot: Mapping[str, Any]) -> tuple[str, str]:
    declared = slot.get("condition_group")
    if declared is not None:
        return _text(declared, "condition_group"), "config"
    family = slot.get("task_family")
    if family in CONDITION_GROUP_BY_TASK_FAMILY:
        return CONDITION_GROUP_BY_TASK_FAMILY[family], "task_family_default"
    raise ValueError("condition_group must be nonempty text")


def _slot_qa_block(request: Any) -> dict[str, Any]:
    """Carry only what the production request was actually told.

    A quota the spec filled in per unit, or a target it derived from the
    speaking instances, is not a declaration. Passing those on would shadow a
    batch-level `config.qa` and would replace the anchor-resolved targets this
    module can compute once the condition profile exists.
    """
    block: dict[str, Any] = {"qa_ids": list(request.qa_ids),
                             "items_per_type": request.items_per_type}
    if request.quota_source != "unit_default":
        block["quota_by_qa"] = dict(request.quota_by_qa)
    if request.qa_targets_declared:
        block["qa_targets"] = [target.to_dict() for target in request.qa_targets]
    return block


def _preallocation_by_asset(
    instances: Sequence[Mapping[str, Any]],
    allowlists: Mapping[str, Any],
) -> dict[str, list[str]]:
    """Canonicalize actor-keyed candidate pools by the bound physical asset."""
    result: dict[str, list[str]] = {}
    for instance in instances:
        actor_id = str(instance.get("instance_id") or "")
        if actor_id not in allowlists:
            continue
        values = allowlists[actor_id]
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) or not value for value in values)
        ):
            raise ValueError(
                "sound preallocation must map every speaking actor to sound IDs"
            )
        asset_id = instance.get("asset_id")
        key = str(asset_id) if asset_id is not None else f"actor:{actor_id}"
        copied = [str(value) for value in values]
        if key in result and result[key] != copied:
            raise ValueError(
                f"one physical asset has conflicting sound preallocation: {key}"
            )
        result[key] = copied
    return result


def _preallocation_for_instances(
    instances: Sequence[Mapping[str, Any]],
    by_asset: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    """Project one shared physical-asset pool onto a member's actor slots."""
    result: dict[str, list[str]] = {}
    for instance in instances:
        actor_id = str(instance.get("instance_id") or "")
        asset_id = instance.get("asset_id")
        key = str(asset_id) if asset_id is not None else f"actor:{actor_id}"
        if key in by_asset:
            result[actor_id] = [str(value) for value in by_asset[key]]
    return result


def _intersect_preallocations(
    left: Mapping[str, Sequence[str]],
    right: Mapping[str, Sequence[str]],
    *,
    required_assets: Sequence[str],
) -> dict[str, list[str]]:
    """Keep only candidate sounds legal for both members of one audio column."""
    result: dict[str, list[str]] = {}
    empty = []
    for asset_id in sorted(set(left) | set(right)):
        left_values = [str(value) for value in left.get(asset_id, ())]
        right_values = [str(value) for value in right.get(asset_id, ())]
        right_set = set(right_values)
        common = [value for value in left_values if value in right_set]
        if asset_id in required_assets and not common:
            empty.append(asset_id)
        if common:
            result[asset_id] = common
    if empty:
        raise ValueError(
            "shared audio column has no common legal sound candidate for asset(s): "
            f"{empty}"
        )
    return result


def _request_preallocation_by_asset(request: Any) -> dict[str, list[str]] | None:
    """Read an explicitly declared actor-keyed candidate pool from a request."""
    selection = getattr(request, "sound_selection", None)
    if not isinstance(selection, Mapping):
        return None
    declared = selection.get("preallocated_sound_asset_ids_by_actor")
    if declared is None:
        return None
    if not isinstance(declared, Mapping):
        raise ProductionSpecError(
            "explicit sound preallocation must be an actor-to-sound-ID mapping"
        )
    instances = [instance.to_dict() for instance in request.instances]
    speaking = {
        str(instance["instance_id"])
        for instance in instances
        if instance.get("speaking") is not False
    }
    missing = sorted(speaking - set(str(key) for key in declared))
    if missing:
        raise ProductionSpecError(
            f"explicit sound preallocation misses speaking actors: {missing}"
        )
    return _preallocation_by_asset(instances, declared)


def production_config_slots(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Turn one production-spec config into batch slots plus its own summary.

    The same configuration therefore drives ordinary Episodes and the four core
    group tasks through the existing preallocation path.
    """
    parsed = parse_production_config(config)
    # The parser keeps per-instance speaking flags and drops an episode-level
    # entities.silent_count once instances are declared. Compare them here, while
    # the raw block is still readable, so a contradiction is reported rather than
    # silently resolved in favour of the flags.
    by_request_id = {request.request_id: request for request in parsed.all_requests()}
    for raw in list(config.get("episodes") or ()) + [
        member
        for group in (config.get("core_groups") or ())
        if isinstance(group, Mapping)
        for member in (group.get("members") or ())
    ]:
        if not isinstance(raw, Mapping):
            continue
        request_id = raw.get("request_id") or raw.get("episode_id")
        request = by_request_id.get(str(request_id))
        entities = raw.get("entities")
        if request is None or not isinstance(entities, Mapping):
            continue
        if entities.get("silent_count") is None or not raw.get("instances"):
            continue
        resolve_silent_count(
            [instance.to_dict() for instance in request.instances],
            entities["silent_count"],
            owner=f"episode {request_id}")
    for group in parsed.core_groups:
        member_by_id = {member.request_id: member for member in group.members}
        columns: dict[str, list[str]] = {}
        for unit in group_stage_units(group):
            if unit["unit_kind"] != "audio":
                continue
            unit_id = str(unit["unit_id"])
            suffix = unit_id.rsplit("_a", 1)[-1] if "_a" in unit_id else ""
            column = f"a{suffix}" if suffix.isdigit() else unit_id
            columns.setdefault(column, []).extend(
                str(value) for value in unit.get("member_request_ids") or ()
            )
        pairs = group.shared_audio_member_ids or tuple(
            tuple(values) for values in columns.values() if len(values) == 2
        )
        for pair in pairs:
            if len(pair) != 2 or pair[0] not in member_by_id or pair[1] not in member_by_id:
                continue
            left = _request_preallocation_by_asset(member_by_id[pair[0]])
            right = _request_preallocation_by_asset(member_by_id[pair[1]])
            if left is not None and right is not None:
                _intersect_preallocations(
                    left,
                    right,
                    required_assets=sorted(set(left) | set(right)),
                )
    member_index_by_request_id = {
        member.request_id: index
        for group in parsed.core_groups
        for index, member in enumerate(group.members)
    }
    member_audio_column_by_request_id: dict[str, tuple[str, str]] = {}
    for group in parsed.core_groups:
        for unit in group_stage_units(group):
            if unit["unit_kind"] != "audio":
                continue
            unit_id = str(unit["unit_id"])
            suffix = unit_id.rsplit("_a", 1)[-1] if "_a" in unit_id else ""
            column = f"a{suffix}" if suffix.isdigit() else unit_id
            for member_id in unit.get("member_request_ids") or ():
                member_audio_column_by_request_id[str(member_id)] = (
                    group.group_id,
                    column,
                )
    slots: list[dict[str, Any]] = []
    for request in parsed.all_requests():
        legacy = request.to_legacy_request()
        bound = [instance.asset_id for instance in request.instances]
        slot: dict[str, Any] = {
            "episode_id": request.request_id,
            "room_id": request.room_id,
            "source_classes": [instance.source_class for instance in request.instances],
            "silent_count": request.silent_count,
            "seed": request.seed,
            "profile": deepcopy(request.profile),
            "entity_instances": [instance.to_dict() for instance in request.instances],
            "request_overrides": legacy,
            "qa": _slot_qa_block(request),
            "production_request": request.to_dict(),
        }
        if request.task_family is not None:
            slot["task_family"] = request.task_family
        if request.group_id is not None:
            slot["group_id"] = request.group_id
        if request.member_role is not None:
            slot["member_role"] = request.member_role
        if request.group_id is not None:
            slot["member_index"] = member_index_by_request_id[request.request_id]
            audio_key = member_audio_column_by_request_id.get(request.request_id)
            if audio_key is not None:
                slot["_production_audio_key"] = f"{audio_key[0]}:{audio_key[1]}"
        if request.condition_group is not None:
            slot["condition_group"] = request.condition_group
        if all(value is not None for value in bound):
            slot["source_asset_ids"] = list(bound)
        slots.append(slot)
    summary = {
        "schema": parsed.schema,
        "batch_id": parsed.batch_id,
        "episode_count": len(parsed.episodes),
        "core_group_count": len(parsed.core_groups),
        "core_member_count": sum(len(group.members) for group in parsed.core_groups),
        "core_groups": [
            {**group.to_dict(),
             "initial_work_items": [item.to_dict() for item in initial_group_work_items(group)]}
            for group in parsed.core_groups
        ],
        "shared_unit_count": sum(len(group_stage_units(group)) for group in parsed.core_groups),
        "coverage_quota": deepcopy(parsed.coverage_quota),
        "task_families": list(CORE_TASK_FAMILIES),
    }
    return slots, summary


def stage_work_items_for_row(
    row: Mapping[str, Any], *, results: Sequence[Mapping[str, Any]] = ()
) -> list[dict[str, Any]]:
    """Describe the stages one ordinary Episode row can run next.

    With no results this is the planning stage only; the schedule grows from
    actual results. A core group member has no row-scoped schedule because its
    visual and audio units are shared, so this refuses one and names the group
    entry point instead of quietly planning a fourth independent world.
    """
    if row.get("group_id") or row.get("task_family"):
        raise ValueError(
            f"{row['episode_id']} is a core group member of {row.get('group_id')}; "
            "use stage_work_items_for_group so one visual is not captured twice"
        )
    request = production_request_from_legacy(
        row["request"], request_id=row["episode_id"], kind="episode")
    parsed_results = [StageResult.from_mapping(value, owner="stage_result") for value in results]
    items = (initial_stage_work_items(request) if not parsed_results
             else next_stage_work_items(request, parsed_results))
    return [item.to_dict() for item in items]


def core_group_from_manifest(manifest: Mapping[str, Any], group_id: str) -> CoreGroupRequest:
    """Rebuild one saved group so its shared units are described exactly once."""
    rows = [row for row in manifest["episodes"] if row.get("group_id") == group_id]
    if not rows:
        raise ValueError(f"manifest has no rows for group {group_id!r}")
    declared = next(
        (entry for entry in ((manifest.get("production") or {}).get("core_groups") or [])
         if entry.get("group_id") == group_id),
        {},
    )
    order = declared.get("member_request_ids")
    if order:
        position = {value: index for index, value in enumerate(order)}
        missing = [row["episode_id"] for row in rows if row["episode_id"] not in position]
        if missing:
            raise ValueError(f"group {group_id} rows are not in its member list: {missing}")
        rows = sorted(rows, key=lambda row: position[row["episode_id"]])
    families = {row.get("task_family") for row in rows}
    if len(families) != 1 or None in families:
        raise ValueError(f"group {group_id} rows disagree on task_family: {sorted(families)}")
    room_ids = {row["room_id"] for row in rows}
    if len(room_ids) != 1:
        raise ValueError(f"group {group_id} rows span rooms {sorted(room_ids)}")
    members = tuple(
        production_request_from_legacy(row["request"], request_id=row["episode_id"],
                                       kind="core_group_member")
        for row in rows
    )
    return CoreGroupRequest(
        group_id=group_id,
        task_family=next(iter(families)),
        room_id=next(iter(room_ids)),
        members=members,
        shared_audio_member_ids=tuple(
            tuple(pair) for pair in declared.get("shared_audio_member_ids") or ()),
        shared_visual_member_ids=tuple(
            tuple(pair) for pair in declared.get("shared_visual_member_ids") or ()),
    )


def stage_work_items_for_group(
    manifest: Mapping[str, Any], group_id: str, *, results: Sequence[Mapping[str, Any]] = ()
) -> list[dict[str, Any]]:
    """Every shared unit of one group whose real dependencies have passed."""
    group = core_group_from_manifest(manifest, group_id)
    parsed = [StageResult.from_mapping(value, owner="stage_result") for value in results]
    items = (initial_group_work_items(group) if not parsed
             else next_group_work_items(group, parsed))
    return [item.to_dict() for item in items]


def group_blockers_for_group(
    manifest: Mapping[str, Any], group_id: str, *, results: Sequence[Mapping[str, Any]] = ()
) -> list[dict[str, Any]]:
    """Why a group cannot advance: a failed round, or no legal movement time."""
    group = core_group_from_manifest(manifest, group_id)
    parsed = [StageResult.from_mapping(value, owner="stage_result") for value in results]
    return group_blockers(group, parsed)


def prepare_batch_manifest(
    config: Mapping[str, Any], registry: Mapping[str, Any],
    room_catalog: Mapping[str, Any], sounds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Assign assets, observed appearance, profiles and sound identities offline.

    config.slots is the quota list. Failed or unallocatable slots remain rows.
    Per-actor sound-ID allowlists are consumed by the existing conditioned
    sampler; it still filters clip duration inside each fixed identity.
    """
    # These controls belong to the production runner, not individual Episodes.
    # Preserve them through the public prepare CLI instead of silently dropping
    # resource limits or coverage-driven request generation at this boundary.
    runner_controls = {}
    for key in ("p19_coverage", "resource_policy"):
        if key in config:
            if not isinstance(config[key], Mapping):
                raise ValueError(f"config.{key} must be a mapping")
            runner_controls[key] = deepcopy(dict(config[key]))
    production_summary = None
    production_audio_scope_by_group: dict[str, str] = {}
    production_audio_selection_by_key: dict[str, dict[str, list[str]]] = {}
    production_audio_rows_by_key: dict[str, list[dict[str, Any]]] = {}
    if config.get("production") is not None:
        if config.get("slots") is not None:
            raise ValueError("config declares both production and slots; keep one source of slots")
        production_block = config["production"]
        if not isinstance(production_block, Mapping):
            raise ValueError("config.production must be a mapping")
        derived_slots, production_summary = production_config_slots(production_block)
        config = {**dict(config), "slots": derived_slots,
                  "batch_id": config.get("batch_id", production_summary["batch_id"]),
                  "seed": config.get("seed", production_block.get("seed", 0))}
        production_audio_scope_by_group = {
            str(entry["group_id"]): str(
                (entry.get("recipe") or {}).get(
                    "audio_content_scope", "shared_audio_pairs"
                )
            )
            for entry in production_summary.get("core_groups") or ()
        }
    batch_id = _text(config.get("batch_id"), "batch_id")
    seed = int(config.get("seed", 0))
    slots = config.get("slots")
    if not isinstance(slots, list) or not slots:
        raise ValueError("batch slots must be a nonempty quota list")
    assets = {record["asset_id"]: record for record in registry["assets"]}
    rooms = {record["room_id"]: record for record in room_catalog["rooms"]}
    if len(assets) != len(registry["assets"]) or len(rooms) != len(room_catalog["rooms"]):
        raise ValueError("asset and room IDs must be unique")
    candidate_ids = config.get("candidate_asset_ids_by_class")
    if candidate_ids is not None:
        if not isinstance(candidate_ids, Mapping) or set(candidate_ids) != set(SOURCE_CLASSES):
            raise ValueError("candidate_asset_ids_by_class must cover all three source classes")
        for kind, values in candidate_ids.items():
            if not isinstance(values, list) or len(set(values)) != len(values):
                raise ValueError("candidate asset IDs must be distinct lists")
            if any(value not in assets or source_class(assets[value]) != kind for value in values):
                raise ValueError("candidate asset is absent or disagrees with its class")
    by_class = {kind: sorted((record for record in assets.values() if source_class(record) == kind
                             and (candidate_ids is None or record["asset_id"] in candidate_ids[kind])),
                             key=lambda record: record["asset_id"]) for kind in SOURCE_CLASSES}
    sound_ids = [sound.get("sound_asset_id") for sound in sounds]
    if any(not isinstance(value, str) or not value for value in sound_ids) or len(set(sound_ids)) != len(sound_ids):
        raise ValueError("preallocation sound_asset_id values must be nonempty and unique")
    appearance_counts, asset_counts, identity_counts, cross_counts = Counter(), Counter(), Counter(), Counter()
    rng = random.Random(seed)
    if config.get("scatter_condition_groups"):
        slots = scatter_condition_groups(
            slots, rooms_by_id=rooms, seed=seed,
            distance_range_m=_config_distance_range_m(config),
            keep_existing_repeat=bool(config.get("keep_repeat_on_device_device", False)))
    rows, ids = [], set()
    for index, slot in enumerate(slots):
        episode_id = _text(slot.get("episode_id", f"{batch_id}_{index + 1:03d}"), "episode_id")
        if episode_id in ids:
            raise ValueError(f"duplicate episode_id: {episode_id}")
        ids.add(episode_id)
        room_id = _text(slot.get("room_id"), "slot.room_id")
        if room_id not in rooms:
            raise ValueError(f"room is absent from supplied catalog: {room_id}")
        room = rooms[room_id]
        classes = slot.get("source_classes")
        if not isinstance(classes, list) or not 2 <= len(classes) <= 4 or any(c not in SOURCE_CLASSES for c in classes):
            raise ValueError("slot source_classes must specify 2..4 supported sources")
        request = merge_request_overrides(
            config.get("base_request", {}), slot.get("request_overrides", {})
        )
        request.update(episode_id=episode_id, room_id=room_id,
                       seed=int(slot.get("seed", seed + index)), sampling_policy="conditioned_static_v2")
        request["camera"] = {**request.get("camera", {}), "motion": "static"}
        qa_plan = resolve_qa_plan(config, slot)
        request["qa_ids"] = list(qa_plan["qa_ids"])
        request["qa_sampling"] = {**request.get("qa_sampling", {}),
                                  "items_per_type": qa_plan["items_per_type"]}
        slot_instances = slot.get("entity_instances") or []
        silent_count = resolve_silent_count(
            slot_instances,
            slot.get("silent_count"),
            owner=f"episode {episode_id}")
        request["entities"] = {**request.get("entities", {}), "total_count": len(classes),
                               "silent_count": silent_count, "min_articulated_count": 0}
        request["profile"] = {**request.get("profile", {}), **deepcopy(slot.get("profile", {}))}
        explicit = slot.get("source_asset_ids")
        if explicit is not None and len(explicit) != len(classes):
            raise ValueError("explicit slot assets must match source count")
        instances = entity_instances_for_slot(slot, classes, explicit)
        request["entity_instances"] = deepcopy(instances)
        # instance_requests() prefers entities.instances, so a stale block left by
        # the base request would outrank the rows this slot just resolved.
        request["entities"]["instances"] = deepcopy(instances)
        selected, assignments, gaps = [], [], []
        shared_gap = shared_asset_instance_gap(instances)
        if shared_gap is not None:
            gaps.append(shared_gap)
        for actor_index, kind in enumerate(classes):
            if explicit is not None:
                if explicit[actor_index] not in assets or source_class(assets[explicit[actor_index]]) != kind:
                    raise ValueError("explicit asset is absent or disagrees with source class")
                record = assets[explicit[actor_index]]
            else:
                candidates = [record for record in by_class[kind] if record["asset_id"] not in selected]
                if not candidates:
                    gaps.append({"state": "evidence_missing_or_unsampled", "code": "insufficient_distinct_assets",
                                 "source_class": kind})
                    continue
                def score(record):
                    appearance = appearance_assignment(record)
                    appearance_key = json.dumps([appearance["field"], appearance["value"]], sort_keys=True)
                    return (appearance_counts[(room["family"], kind, appearance_key)],
                            asset_counts[record["asset_id"]])
                record = _minimum_choice(candidates, score, rng)
            selected.append(record["asset_id"])
            appearance = appearance_assignment(record)
            appearance_key = json.dumps([appearance["field"], appearance["value"]], sort_keys=True)
            appearance_counts[(room["family"], kind, appearance_key)] += 1
            asset_counts[record["asset_id"]] += 1
            assignment = {"actor_id": instances[actor_index]["instance_id"],
                          "instance_id": instances[actor_index]["instance_id"],
                          "asset_id": record["asset_id"],
                          "asset_revision": record["revision"], "source_class": kind,
                          "appearance": appearance, "sound_identity_id": None, "sound_asset_ids": []}
            if instances[actor_index].get("role") is not None:
                assignment["role"] = instances[actor_index]["role"]
            assignments.append(assignment)
            gaps.extend(_asset_interface_gaps(record, room["renderer"], request=request))
        request["source_asset_ids"] = selected
        for instance, assignment in zip(instances, assignments):
            instance["asset_id"] = assignment["asset_id"]
        request["entity_instances"] = deepcopy(instances)
        request["entities"]["instances"] = deepcopy(instances)
        condition = None
        if len(selected) == len(classes) and shared_gap is None:
            condition = resolve_condition_profile(request, registry)
            allowlists = {}
            declared_sound_ids_by_actor = _declared_preallocation_by_actor(
                request, instances
            )
            selection_config = request.get("sound_selection")
            selection_config = (
                selection_config if isinstance(selection_config, Mapping) else {}
            )
            sample_rate_hz = int(request.get("sample_rate_hz", 16000))
            max_clip_s = selection_config.get("max_clip_s")
            max_clip_samples = None
            if max_clip_s is not None:
                if isinstance(max_clip_s, bool) or not isinstance(max_clip_s, (int, float)):
                    raise ValueError("sound_selection.max_clip_s must be finite")
                if not math.isfinite(float(max_clip_s)) or float(max_clip_s) <= 0.0:
                    raise ValueError("sound_selection.max_clip_s must be positive")
                max_clip_samples = int(round(float(max_clip_s) * sample_rate_hz))
            speaker_rows = []
            for actor_index, assignment in enumerate(assignments):
                actor_id = assignment["actor_id"]
                assignment["speaking"] = actor_index in condition["speaking_indices"]
                if not assignment["speaking"]:
                    assignment["sound_status"] = "silent_by_request"
                    continue
                actor = neutral_source_declaration(assets[assignment["asset_id"]], actor_id)
                groups = defaultdict(list)
                declared_ids = (
                    None
                    if declared_sound_ids_by_actor is None
                    else set(declared_sound_ids_by_actor[actor_id])
                )
                compatible_sound_count = 0
                for sound in sounds:
                    allowed = sound.get("compatible_asset_ids")
                    if allowed is not None and assignment["asset_id"] not in allowed:
                        continue
                    if declared_ids is not None and sound.get("sound_asset_id") not in declared_ids:
                        continue
                    if max_clip_samples is not None and int(sound.get("sample_count", 0)) > max_clip_samples:
                        continue
                    if int(sound.get("sample_rate_hz", sample_rate_hz)) != sample_rate_hz:
                        continue
                    if not sound_matches(actor, sound):
                        continue
                    compatible_sound_count += 1
                    identity = sound_identity(sound)
                    if identity is not None:
                        groups[identity].append(sound)
                if declared_ids is not None:
                    assignment["declared_sound_asset_ids"] = sorted(declared_ids)
                speaker_rows.append({"assignment": assignment, "groups": groups,
                                     "compatible_sound_count": compatible_sound_count,
                                     "declared_sound_asset_ids": (
                                         sorted(declared_ids) if declared_ids is not None else None
                                     ),
                                     "appearance_key": json.dumps(assignment["appearance"], sort_keys=True),
                                     "kind": assignment["source_class"]})
            used_identities = set()
            for row in speaker_rows:
                assignment = row["assignment"]
                actor_id = assignment["actor_id"]
                row["distinct"] = sorted(row["groups"])
                if not row["distinct"]:
                    assignment["sound_status"] = "evidence_missing_or_unsampled"
                    allowlists[actor_id] = []
                    if int(row.get("compatible_sound_count") or 0) <= 0:
                        gaps.append({"state": "evidence_missing_or_unsampled",
                                     "code": "no_compatible_sounds",
                                     "asset_id": assignment["asset_id"], "actor_id": actor_id,
                                     "compatible_sound_count": 0})
                    else:
                        gaps.append({"state": "evidence_missing_or_unsampled",
                                     "code": "no_distinct_compatible_sound_identity",
                                     "asset_id": assignment["asset_id"], "actor_id": actor_id})
            ready = [row for row in speaker_rows if row["assignment"].get("sound_status") != "evidence_missing_or_unsampled"]
            # For a production shared audio column, the selector's full
            # compatible pools are the common legal universe.  Each generated
            # row remains a real selector input; explicit allowlists remain
            # restrictive and are intersected below.
            common_sound_ids_by_asset: dict[str, list[str]] = {}
            for speaker in speaker_rows:
                assignment = speaker["assignment"]
                asset_id = str(assignment["asset_id"])
                candidate_ids = sorted({
                    str(sound["sound_asset_id"])
                    for sounds_by_identity in speaker["groups"].values()
                    for sound in sounds_by_identity
                    if sound.get("sound_asset_id")
                })
                if candidate_ids:
                    common_sound_ids_by_asset[asset_id] = candidate_ids
            substitution = False
            substitution_applied = False
            repeat_actor_id = None
            all_speaking_have_groups = len(ready) == len(speaker_rows) and bool(ready)
            if ready:
                relation = condition["event_relation"]
                gap = condition["min_gap_between_audible_windows_s"]
                available = available_program_seconds(request, condition)

                def usage(row, identity):
                    return (cross_counts[(row["kind"], row["appearance_key"], identity)],
                            identity_counts[(row["kind"], identity)])

                greedy = []
                taken = set()
                unique_ok = True
                for row in ready:
                    candidates = [identity for identity in row["distinct"] if identity not in taken]
                    if not candidates:
                        row["assignment"]["sound_status"] = "evidence_missing_or_unsampled"
                        allowlists[row["assignment"]["actor_id"]] = []
                        gaps.append({"state": "evidence_missing_or_unsampled",
                                     "code": "no_distinct_compatible_sound_identity",
                                     "asset_id": row["assignment"]["asset_id"],
                                     "actor_id": row["assignment"]["actor_id"]})
                        unique_ok = False
                        break
                    identity = _minimum_choice(candidates, lambda value, row=row: usage(row, value), rng)
                    greedy.append(identity)
                    taken.add(identity)
                if (not unique_ok) or (not all_speaking_have_groups):
                    for row, identity in zip(ready, greedy):
                        identity_counts[(row["kind"], identity)] += 1
                        cross_counts[(row["kind"], row["appearance_key"], identity)] += 1
                        allowlists[row["assignment"]["actor_id"]] = _bind_identity(
                            row["assignment"], identity, row["groups"])
                        used_identities.add(identity)
                if unique_ok and all_speaking_have_groups:
                    greedy_durs = [min(_clip_duration_s(sound) for sound in row["groups"][identity])
                                   for row, identity in zip(ready, greedy)]
                    chosen = list(greedy)
                    repeat_index = None
                    if relation == "repeat":
                        legal_repeats = [index for index in range(len(chosen))
                                         if program_seconds_for_durations(
                                             greedy_durs, gap_s=gap, relation=relation, repeat_index=index)
                                         <= available + 1e-9]
                        if legal_repeats:
                            shortest = min(greedy_durs[index] for index in legal_repeats)
                            repeat_index = rng.choice([index for index in legal_repeats if greedy_durs[index] == shortest])
                        else:
                            substitution = True
                    elif relation == "sequential" and program_seconds_for_durations(
                        greedy_durs, gap_s=gap, relation=relation
                    ) > available + 1e-9:
                        substitution = True
                    if substitution:
                        legal = []
                        lists = [row["distinct"] for row in ready]
                        if math.prod(max(1, len(values)) for values in lists) <= 40000:
                            for combo in product(*lists):
                                if len(set(combo)) != len(combo):
                                    continue
                                durs = [min(_clip_duration_s(sound) for sound in row["groups"][identity])
                                        for row, identity in zip(ready, combo)]
                                score = tuple(usage(row, identity) for row, identity in zip(ready, combo))
                                repeat_indices = range(len(combo)) if relation == "repeat" else (None,)
                                for index in repeat_indices:
                                    seconds = program_seconds_for_durations(
                                        durs, gap_s=gap, relation=relation, repeat_index=index)
                                    if seconds <= available + 1e-9:
                                        legal.append((score, combo, index, seconds))
                        if legal:
                            minimum = min(item[0] for item in legal)
                            score, combo, repeat_index, _seconds = rng.choice(
                                [item for item in legal if item[0] == minimum])
                            chosen = list(combo)
                            substitution_applied = True
                        else:
                            chosen = None
                    elif relation != "repeat" and program_seconds_for_durations(
                        greedy_durs, gap_s=gap, relation=relation
                    ) > available + 1e-9:
                        chosen = None
                    if chosen is None:
                        for row, identity in zip(ready, greedy):
                            allowlists[row["assignment"]["actor_id"]] = _bind_identity(
                                row["assignment"], identity, row["groups"])
                            used_identities.add(identity)
                        seconds = program_seconds_for_durations(
                            greedy_durs, gap_s=gap, relation=relation,
                            repeat_index=0 if relation == "repeat" else None)
                        gaps.append({
                            "state": "evidence_missing_or_unsampled",
                            "code": "fixed_sound_identities_exceed_profile_clip_budget",
                            "minimum_program_seconds_under_sampler_clip_budget": seconds,
                            "available_seconds_before_reserved_tail": available,
                            "identity_substitution_applied": False,
                            "identity_substitution_attempted": bool(substitution),
                            "repeat_feasibility_formula": "2 * duration(repeat_sound) + duration(other_source_sound) + 2 * gap_s",
                        })
                    else:
                        for row, identity in zip(ready, chosen):
                            used_identities.add(identity)
                            identity_counts[(row["kind"], identity)] += 1
                            cross_counts[(row["kind"], row["appearance_key"], identity)] += 1
                            allowlists[row["assignment"]["actor_id"]] = _bind_identity(
                                row["assignment"], identity, row["groups"])
                        if relation == "repeat" and repeat_index is not None:
                            repeat_actor_id = ready[repeat_index]["assignment"]["actor_id"]
            selection = {**request.get("sound_selection", {}),
                         "preallocated_sound_asset_ids_by_actor": allowlists,
                         "clip_span_fit_policy": request.get("sound_selection", {}).get(
                             "clip_span_fit_policy", CLIP_SPAN_FIT_POLICY)}
            production_audio_key = slot.get("_production_audio_key")
            production_group_id = slot.get("group_id")
            if (
                isinstance(production_audio_key, str)
                and production_audio_scope_by_group.get(str(production_group_id))
                == "shared_audio_pairs"
            ):
                declared_selection = request.get("sound_selection")
                declared_selection = (
                    declared_selection
                    if isinstance(declared_selection, Mapping)
                    else {}
                )
                declared_preallocation = declared_selection.get(
                    "preallocated_sound_asset_ids_by_actor"
                )
                speaking_assets = [
                    str(assignment["asset_id"])
                    for assignment in assignments
                    if assignment.get("speaking")
                ]
                if declared_preallocation is not None:
                    if not isinstance(declared_preallocation, Mapping):
                        raise ValueError(
                            "explicit sound preallocation must be an actor-to-sound-ID mapping"
                        )
                    speaking_actor_ids = {
                        str(assignment["actor_id"])
                        for assignment in assignments
                        if assignment.get("speaking")
                    }
                    missing = sorted(speaking_actor_ids - set(declared_preallocation))
                    if missing:
                        raise ValueError(
                            "explicit sound preallocation misses speaking actors: "
                            f"{missing}"
                        )
                    effective_by_asset = _preallocation_by_asset(
                        instances, declared_preallocation
                    )
                else:
                    effective_by_asset = dict(common_sound_ids_by_asset)
                canonical = production_audio_selection_by_key.get(production_audio_key)
                if canonical is None:
                    canonical = effective_by_asset
                else:
                    canonical = _intersect_preallocations(
                        canonical,
                        effective_by_asset,
                        required_assets=speaking_assets,
                    )
                if not canonical:
                    raise ValueError(
                        f"shared audio column {production_audio_key} has no legal candidates"
                    )
                production_audio_selection_by_key[production_audio_key] = canonical
                allowlists = _preallocation_for_instances(instances, canonical)
                for assignment in assignments:
                    assignment["sound_asset_ids"] = deepcopy(
                        allowlists.get(str(assignment["actor_id"]), [])
                    )
                for previous in production_audio_rows_by_key.get(
                    production_audio_key, ()
                ):
                    previous_instances = previous.get("entity_instances") or ()
                    previous_allowlists = _preallocation_for_instances(
                        previous_instances, canonical
                    )
                    previous["request"]["sound_selection"] = {
                        **dict(previous["request"].get("sound_selection") or {}),
                        "preallocated_sound_asset_ids_by_actor": previous_allowlists,
                    }
                    for assignment in previous.get("source_assignments") or ():
                        assignment["sound_asset_ids"] = deepcopy(
                            previous_allowlists.get(
                                str(assignment["actor_id"]), []
                            )
                        )
                selection["preallocated_sound_asset_ids_by_actor"] = allowlists
            if repeat_actor_id is not None:
                selection["repeat_actor_id"] = repeat_actor_id
                selection["identity_substitution_applied"] = substitution
            elif substitution_applied:
                selection["identity_substitution_applied"] = True
            request["sound_selection"] = selection
        condition_group, condition_group_source = _condition_group_for_slot(slot)
        # Catalog target metadata does not itself opt into conditioned
        # sampling. Only an explicit target declaration belongs in the
        # execution request; branch/drive/profile controls retain their own
        # existing execution semantics.
        target_plan = dict(qa_plan)
        if target_plan.get("qa_targets") is None and request.get("qa_targets"):
            target_plan["qa_targets"] = deepcopy(request["qa_targets"])
        qa_targets = resolve_qa_targets(target_plan, instances=instances, condition=condition)
        if target_plan.get("qa_targets") is not None:
            request["qa_targets"] = deepcopy(qa_targets)
        else:
            request.pop("qa_targets", None)
        row = {"episode_id": episode_id, "room_id": room_id, "room_family": room["family"],
               "renderer": room["renderer"], "condition_group": condition_group,
               "condition_group_source": condition_group_source,
               "class_pair": class_pair_label(classes),
               "requested_source_classes": deepcopy(classes), "requested_profile": deepcopy(condition),
               "entity_instances": deepcopy(instances),
               "entity_instance_count": len(instances),
               "distinct_asset_count": len({instance["asset_id"] for instance in instances
                                            if instance.get("asset_id")}),
               "requested_qa_ids": list(qa_plan["qa_ids"]),
               "qa_ids_source": qa_plan["qa_ids_source"],
               "items_per_type": qa_plan["items_per_type"],
               "items_per_type_source": qa_plan["items_per_type_source"],
               "qa_targets": deepcopy(qa_targets),
               "requested_quota_by_qa": dict(qa_plan["quota_by_qa"]),
               "requested_quota_source": qa_plan["quota_source"],
               "source_assignments": assignments,
               "preallocation_gaps": gaps, "request": request, "execution_status": "not_run",
               "achieved_conditions": None}
        for key in ("task_family", "group_id", "member_role"):
            if slot.get(key) is not None:
                row[key] = _text(slot[key], "slot." + key)
        if slot.get("production_request") is not None:
            row["production_request"] = deepcopy(slot["production_request"])
        if row.get("group_id"):
            row["stage_scope"] = _member_stage_scope(row, slot)
            row["stage_work_items"] = []
        else:
            row["stage_scope"] = {"kind": "episode", "scope_id": episode_id}
            row["stage_work_items"] = stage_work_items_for_row(row)
        production_audio_key = slot.get("_production_audio_key")
        if (
            isinstance(production_audio_key, str)
            and "sound_selection" in request
            and production_audio_scope_by_group.get(str(slot.get("group_id")))
            == "shared_audio_pairs"
        ):
            production_audio_rows_by_key.setdefault(production_audio_key, []).append(row)
        rows.append(row)
    group_quota = Counter((row["room_family"], row["room_id"], row["condition_group"]) for row in rows)
    crosstab = class_pair_condition_group_crosstab(rows)
    if production_summary is not None:
        production_summary = {**production_summary, "derived_slot_count": len(rows)}
    return {"schema": "avengine_qa_batch_manifest_v1", "batch_id": batch_id, "seed": seed,
            **runner_controls,
            "stage_protocol": stage_protocol_summary(),
            "production": production_summary,
            "claim_boundary": "Preallocated requests only; no native execution, achieved quota or admission claim.",
            "allocation_policy": "offline_least_used_appearance_asset_and_sound_identity_with_repeat_feasibility_substitution",
            "class_pair_condition_group_crosstab": crosstab,
            "achieved_separation_histogram_5deg": histogram_separation_5deg([]),
            "separation_coverage_unit": "achieved_angle_5deg_bins",
            "runtime_shared_counters": False, "asset_inventory": sorted(assets),
            "candidate_asset_ids_by_class": {kind: [record["asset_id"] for record in by_class[kind]]
                                             for kind in SOURCE_CLASSES},
            "candidate_scope": deepcopy(config.get("candidate_scope")),
            "assets_outside_candidate_scope": sorted(set(assets) - {
                record["asset_id"] for values in by_class.values() for record in values}),
            "room_inventory": deepcopy(room_catalog["rooms"]), "episodes": rows,
            "requested_episode_count": len(rows), "executed_episode_count": 0,
            "quota_by_condition_group": [
                {"room_family": family, "room_id": room_id, "condition_group": group,
                 "requested": count, "achieved": 0, "unmet": count}
                for (family, room_id, group), count in sorted(group_quota.items())],
            "preallocation_gap_counts": dict(Counter(gap["code"] for row in rows for gap in row["preallocation_gaps"]))}


def _classify_outcome_failure(outcome: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(outcome, Mapping) or outcome.get("status") == "delivered":
        return None
    stage = outcome.get("failure_stage")
    declared_gap = outcome.get("gap_state")
    status = outcome.get("status")
    reason_code = outcome.get("failure_code") or outcome.get("reason_code")
    if reason_code == "unclassified_failure":
        declared_gap = None
    reason = outcome.get("failure_reason") or outcome.get("reason") or ""
    histogram = outcome.get("failure_histogram")
    return classify_failure(
        failure_stage=stage if isinstance(stage, str) else None,
        reason=str(reason),
        reason_code=reason_code if isinstance(reason_code, str) else None,
        histogram=histogram if isinstance(histogram, Mapping) else None,
        status=status if isinstance(status, str) else None,
        declared_gap_state=declared_gap if isinstance(declared_gap, str) else None,
    )


def _outcome_failure_fields(outcome: Mapping[str, Any] | None) -> tuple[str | None, str | None]:
    """Fill failure fields through the shared classifier used by the runner."""
    classified = _classify_outcome_failure(outcome)
    if classified is None:
        return None, None
    return classified["failure_stage"], classified["gap_state"]


def collect_batch_outcomes(manifest: Mapping[str, Any], outcomes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Left-join outcomes to every requested slot, retaining failures and deficits."""
    declared = {row["episode_id"]: row for row in manifest["episodes"]}
    by_id = {}
    for outcome in outcomes:
        episode_id = _text(outcome.get("episode_id"), "outcome.episode_id")
        if episode_id not in declared or episode_id in by_id:
            raise ValueError("outcome episode is unknown or duplicated")
        status = outcome.get("status")
        if status not in {"preallocation_blocked", "resource_failed", "planning_failed", "capture_failed", "audio_failed", "delivery_failed", "review_failed", "delivered"}:
            raise ValueError(f"unsupported outcome status: {status}")
        if status == "delivered":
            if not outcome.get("facts_path") or not outcome.get("questions_path"):
                raise ValueError("delivered outcome requires facts and questions paths")
        by_id[episode_id] = deepcopy(outcome)
    rows = []
    for requested in manifest["episodes"]:
        outcome = by_id.get(requested["episode_id"])
        observed_profile = outcome.get("condition_profile") if outcome else None
        mismatch = (
            observed_profile is not None
            and _profile_for_match(observed_profile) != _profile_for_match(requested.get("requested_profile"))
        )
        achieved = outcome.get("achieved_conditions") if outcome else None
        # Copy only a native/PCM evidence result supplied by the caller. Never
        # promote planned_conditions to achieved_conditions.
        if achieved is not None and not outcome.get("achieved_conditions_source"):
            raise ValueError("achieved_conditions require an actual evidence source")
        requested_classes = requested.get("requested_source_classes")
        if requested_classes is None:
            requested_classes = [
                actor.get("source_class") for actor in requested.get("source_assignments") or []
            ]
        classified = _classify_outcome_failure(outcome)
        failure_stage = classified["failure_stage"] if classified is not None else None
        gap_state = classified["gap_state"] if classified is not None else None
        if isinstance(outcome, dict) and classified is not None:
            if failure_stage and not outcome.get("failure_stage"):
                outcome["failure_stage"] = failure_stage
            if gap_state and not outcome.get("gap_state"):
                outcome["gap_state"] = gap_state
            existing_diagnostic = outcome.get("diagnostic")
            outcome["diagnostic"] = {
                **deepcopy(classified["diagnostic"]),
                **(deepcopy(existing_diagnostic) if isinstance(existing_diagnostic, Mapping) else {}),
            }
            classified_code = classified.get("reason_code")
            if classified_code and not outcome.get("reason_code"):
                outcome["reason_code"] = classified_code
            if classified["failure_reason"] and not outcome.get("failure_reason"):
                outcome["failure_reason"] = classified["failure_reason"]
            if classified["failure_reason"] and not outcome.get("reason"):
                outcome["reason"] = classified["failure_reason"]
        rows.append({"episode_id": requested["episode_id"], "room_id": requested["room_id"],
                     "room_family": requested["room_family"], "condition_group": requested["condition_group"],
                     "requested_source_classes": deepcopy(requested_classes),
                     "requested_profile": deepcopy(requested["requested_profile"]),
                     "requested_source_assignments": deepcopy(requested["source_assignments"]),
                     "outcome": outcome, "status": "not_run" if outcome is None else outcome["status"],
                     "failure_stage": failure_stage, "gap_state": gap_state,
                     "profile_matches_request": None if observed_profile is None else not mismatch,
                     "achieved_conditions": deepcopy(achieved),
                     "achieved_conditions_source": outcome.get("achieved_conditions_source") if outcome else None,
                     "requested_quota_by_qa": deepcopy(requested["requested_quota_by_qa"]),
                     "unmet_quota_by_qa": {
                         qa: max(0, quota - int(outcome.get("produced_count_by_qa", {}).get(qa, 0)))
                         if outcome and not mismatch else quota
                         for qa, quota in requested["requested_quota_by_qa"].items()},
                     "preallocation_gaps": deepcopy(requested["preallocation_gaps"])})
    groups = defaultdict(list)
    for row in rows:
        groups[(row["room_family"], row["room_id"], row["condition_group"])].append(row)
    angles = [angle for row in rows for angle in iter_achieved_separation_deg(row)]
    return {"schema": "avengine_qa_batch_outcomes_v1", "batch_id": manifest["batch_id"],
            "episode_denominator": len(rows), "episodes": rows,
            "outcome_counts": dict(Counter(row["status"] for row in rows)),
            "class_pair_condition_group_crosstab": class_pair_condition_group_crosstab(rows),
            "achieved_separation_histogram_5deg": histogram_separation_5deg(angles),
            "separation_coverage_unit": "achieved_angle_5deg_bins",
            "quota_by_condition_group": [
                {"room_family": family, "room_id": room, "condition_group": group,
                 "requested": len(values),
                 "delivered": sum(row["status"] == "delivered" and row["profile_matches_request"] is True for row in values),
                 "unmet": sum(row["status"] != "delivered" or row["profile_matches_request"] is not True for row in values)}
                for (family, room, group), values in sorted(groups.items())]}


def grouped_splits(records: Sequence[Mapping[str, Any]], *, ratios: Mapping[str, float],
                   seed: int = 0) -> dict[str, Any]:
    """Keep the transitive union of Episode, visual, room, route and sound groups."""
    if not ratios or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                         or not math.isfinite(float(value)) or value <= 0 for value in ratios.values()):
        raise ValueError("split ratios must be finite and positive")
    names = sorted(ratios)
    total_ratio = sum(ratios.values())
    parent = list(range(len(records)))
    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left
    owners, keys_by_index, missing = {}, [], {}
    record_ids = []
    for index, record in enumerate(records):
        record_id = _text(record.get("record_id"), "record_id")
        if record_id in record_ids:
            raise ValueError(f"duplicate split record_id: {record_id}")
        record_ids.append(record_id)
        keys, errors = [], []
        for field in ("episode_id", "visual_episode_id", "room_id"):
            value = record.get(field)
            if value is None and field == "visual_episode_id":
                value = record.get("episode_id")
            if value is None:
                errors.append("missing_" + field)
            else:
                keys.append((field, _text(value, field)))
        # Episode and visual identifiers share a namespace so an audio variant
        # referring to another row's episode_id is connected in either form.
        episode = record.get("episode_id")
        visual = record.get("visual_episode_id", episode)
        for value in (episode, visual):
            if value is not None:
                keys.append(("visual_episode", _text(value, "visual Episode identifier")))
        routes = record.get("route_ids", [])
        sounds = record.get("sound_identity_ids")
        if not isinstance(routes, list) or any(not isinstance(value, str) or not value for value in routes):
            raise ValueError("route_ids must be a list of nonempty original route identities")
        if sounds is None or (record.get("audio_present", True) and not sounds):
            errors.append("missing_sound_identity_ids")
            sounds = []
        if not isinstance(sounds, list) or any(not isinstance(value, str) or not value for value in sounds):
            raise ValueError("sound_identity_ids must be a list of nonempty original identities")
        keys.extend(("route_id", value) for value in routes)
        keys.extend(("sound_identity_id", value) for value in sounds)
        for key in keys:
            if key in owners:
                union(index, owners[key])
            else:
                owners[key] = index
        keys_by_index.append(keys)
        missing[index] = errors
    members = defaultdict(list)
    for index in range(len(records)):
        members[find(index)].append(index)
    components = sorted(members.values(), key=lambda values: min(record_ids[index] for index in values))
    rng = random.Random(seed)
    scheduling = list(range(len(components)))
    rng.shuffle(scheduling)
    scheduling.sort(key=lambda index: -len(components[index]))
    assigned, totals = {}, Counter()
    targets = {name: len(records) * ratios[name] / total_ratio for name in names}
    for component_index in scheduling:
        values = components[component_index]
        if any(missing[index] for index in values):
            assigned[component_index] = None
            continue
        split = max(names, key=lambda name: targets[name] - totals[name])
        assigned[component_index] = split
        totals[split] += len(values)
    groups, rows = [], []
    for component_index, values in enumerate(components):
        group_id = f"group_{component_index + 1:04d}"
        split = assigned[component_index]
        group_keys = sorted({key for index in values for key in keys_by_index[index]})
        reasons = sorted({error for index in values for error in missing[index]})
        groups.append({"group_id": group_id, "split": split,
                       "record_ids": sorted(record_ids[index] for index in values),
                       "group_keys": [list(key) for key in group_keys], "missing_group_metadata": reasons})
        for index in values:
            rows.append({"record_id": record_ids[index], "episode_id": records[index].get("episode_id"),
                         "group_id": group_id, "split": split,
                         "status": "assigned" if split else "unassigned_missing_group_metadata"})
    # Room grouping is deliberately broader than every individual route in that
    # room. Missing route IDs do not split a room into separate sets.
    return {"schema": "avengine_qa_grouped_splits_v1", "seed": seed,
            "grouping_fields": ["episode_id", "visual_episode_id", "room_id", "route_ids", "sound_identity_ids"],
            "missing_route_policy": "whole_room_grouping_is_more_conservative",
            "record_denominator": len(records), "groups": groups,
            "records": sorted(rows, key=lambda row: row["record_id"]),
            "requested_counts": targets, "actual_counts": {name: totals[name] for name in names},
            "unassigned_count": sum(row["split"] is None for row in rows),
            "claim_boundary": "Connected groups are indivisible; requested proportions may be unattainable."}
