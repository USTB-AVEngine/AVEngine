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

SOURCE_CLASSES = ("articulated_human", "articulated_animal", "rigid_static_object")
QA_IDS = tuple(f"QA-{index:02d}" for index in range(1, 25))
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
GROUP_PROFILE = {
    "identity_binding": {},
    "audio_event_relations": {"event_relation": "overlap", "anchor_count": 2},
    "visibility_occlusion": {"anchor_line_of_sight": "occluded", "separation_bin_deg": [15, 30]},
    "motion_distance": {"speech_motion": "speaker_moving"},
    "post_sound_state": {"separation_bin_deg": [60, 90]},
}



def class_pair_label(classes: Sequence[str]) -> str:
    labels = sorted(SOURCE_CLASS_LABEL[value] for value in classes)
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
        slot["condition_group"] = group
        slot["profile"] = profile_for_condition_group(
            group, slot["source_classes"], silent_count=int(slot.get("silent_count", 0)),
            event_relation=event_relation, off_screen=slot.get("off_screen"),
            distance_range_m=distance_range_m)
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


def _asset_interface_gaps(record: Mapping[str, Any], renderer: str) -> list[dict[str, str]]:
    backend = "spear_unreal" if renderer == "ue_spear" else "habitat"
    bindings = record.get("runtime_backends", {})
    if not bindings.get(backend):
        return [{"state": "interface_not_implemented", "code": "renderer_binding_missing",
                 "asset_id": record["asset_id"], "renderer": renderer,
                 "file": "examples/runtime/source_asset_runtime_profiles.json"}]
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


def prepare_batch_manifest(
    config: Mapping[str, Any], registry: Mapping[str, Any],
    room_catalog: Mapping[str, Any], sounds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Assign assets, observed appearance, profiles and sound identities offline.

    config.slots is the quota list. Failed or unallocatable slots remain rows.
    Per-actor sound-ID allowlists are consumed by the existing conditioned
    sampler; it still filters clip duration inside each fixed identity.
    """
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
            distance_range_m=config.get("distance_range_m"),
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
        request = deepcopy(config.get("base_request", {}))
        request.update(deepcopy(slot.get("request_overrides", {})))
        request.update(episode_id=episode_id, room_id=room_id,
                       seed=int(slot.get("seed", seed + index)), sampling_policy="conditioned_static_v2")
        request["camera"] = {**request.get("camera", {}), "motion": "static"}
        request["qa_ids"] = list(QA_IDS)
        request["qa_sampling"] = {**request.get("qa_sampling", {}), "items_per_type": 1}
        request["entities"] = {**request.get("entities", {}), "total_count": len(classes),
                               "silent_count": int(slot.get("silent_count", 0)), "min_articulated_count": 0}
        request["profile"] = {**request.get("profile", {}), **deepcopy(slot.get("profile", {}))}
        explicit = slot.get("source_asset_ids")
        if explicit is not None and (len(explicit) != len(classes) or len(set(explicit)) != len(explicit)):
            raise ValueError("explicit slot assets must be distinct and match source count")
        selected, assignments, gaps = [], [], []
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
            assignment = {"actor_id": f"source{actor_index + 1}", "asset_id": record["asset_id"],
                          "asset_revision": record["revision"], "source_class": kind,
                          "appearance": appearance, "sound_identity_id": None, "sound_asset_ids": []}
            assignments.append(assignment)
            gaps.extend(_asset_interface_gaps(record, room["renderer"]))
        request["source_asset_ids"] = selected
        condition = None
        if len(selected) == len(classes):
            condition = resolve_condition_profile(request, registry)
            allowlists = {}
            speaker_rows = []
            for actor_index, assignment in enumerate(assignments):
                actor_id = assignment["actor_id"]
                assignment["speaking"] = actor_index in condition["speaking_indices"]
                if not assignment["speaking"]:
                    assignment["sound_status"] = "silent_by_request"
                    continue
                actor = neutral_source_declaration(assets[assignment["asset_id"]], actor_id)
                groups = defaultdict(list)
                for sound in sounds:
                    allowed = sound.get("compatible_asset_ids")
                    if allowed is not None and assignment["asset_id"] not in allowed:
                        continue
                    if not sound_matches(actor, sound):
                        continue
                    identity = sound_identity(sound)
                    if identity is not None:
                        groups[identity].append(sound)
                speaker_rows.append({"assignment": assignment, "groups": groups,
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
                    gaps.append({"state": "evidence_missing_or_unsampled",
                                 "code": "no_distinct_compatible_sound_identity",
                                 "asset_id": assignment["asset_id"], "actor_id": actor_id})
            ready = [row for row in speaker_rows if row["assignment"].get("sound_status") != "evidence_missing_or_unsampled"]
            substitution = False
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
                            legal = []
                            lists = [row["distinct"] for row in ready]
                            if math.prod(max(1, len(values)) for values in lists) <= 40000:
                                for combo in product(*lists):
                                    if len(set(combo)) != len(combo):
                                        continue
                                    durs = [min(_clip_duration_s(sound) for sound in row["groups"][identity])
                                            for row, identity in zip(ready, combo)]
                                    score = tuple(usage(row, identity) for row, identity in zip(ready, combo))
                                    for index in range(len(combo)):
                                        seconds = program_seconds_for_durations(
                                            durs, gap_s=gap, relation=relation, repeat_index=index)
                                        if seconds <= available + 1e-9:
                                            legal.append((score, combo, index, seconds))
                            if legal:
                                minimum = min(item[0] for item in legal)
                                score, combo, repeat_index, _seconds = rng.choice(
                                    [item for item in legal if item[0] == minimum])
                                chosen = list(combo)
                            else:
                                chosen = None
                    elif program_seconds_for_durations(greedy_durs, gap_s=gap, relation=relation) > available + 1e-9:
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
            if repeat_actor_id is not None:
                selection["repeat_actor_id"] = repeat_actor_id
                selection["identity_substitution_applied"] = substitution
            request["sound_selection"] = selection
        rows.append({"episode_id": episode_id, "room_id": room_id, "room_family": room["family"],
                     "renderer": room["renderer"], "condition_group": _text(slot.get("condition_group"), "condition_group"),
                     "class_pair": class_pair_label(classes),
                     "requested_source_classes": deepcopy(classes), "requested_profile": deepcopy(condition),
                     "requested_quota_by_qa": {qa: 1 for qa in QA_IDS}, "source_assignments": assignments,
                     "preallocation_gaps": gaps, "request": request, "execution_status": "not_run",
                     "achieved_conditions": None})
    group_quota = Counter((row["room_family"], row["room_id"], row["condition_group"]) for row in rows)
    crosstab = class_pair_condition_group_crosstab(rows)
    return {"schema": "avengine_qa_batch_manifest_v1", "batch_id": batch_id, "seed": seed,
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


def _outcome_failure_fields(outcome: Mapping[str, Any] | None) -> tuple[str | None, str | None]:
    """Fill failure_stage / gap_state from the outcome, using status only when those keys are absent."""
    if not isinstance(outcome, Mapping) or outcome.get("status") == "delivered":
        return None, None
    stage = outcome.get("failure_stage")
    gap = outcome.get("gap_state")
    status = outcome.get("status")
    code = outcome.get("failure_code") or outcome.get("reason_code")
    reason = str(outcome.get("failure_reason") or outcome.get("reason") or "")
    histogram = outcome.get("failure_histogram")
    if not isinstance(stage, str) or not stage:
        stage = {
            "preallocation_blocked": "planning",
            "planning_failed": "planning",
            "capture_failed": "capture",
            "audio_failed": "audio",
            "delivery_failed": "finalize",
            "review_failed": "finalize",
            "resource_failed": "launch",
        }.get(status)
        if code == "preallocation_gap":
            stage = "planning"
    if not isinstance(gap, str) or not gap:
        exhausted = (
            isinstance(histogram, Mapping) and bool(histogram)
            or "fixed condition profile exhausted" in reason.lower()
            or "conditionedplanningfailure" in reason.lower()
        )
        if status == "preallocation_blocked" or code == "preallocation_gap" or status == "planning_failed" or exhausted:
            gap = "evidence_missing_or_unsampled"
        elif status in {"audio_failed", "resource_failed"}:
            gap = "interface_not_implemented"
    if not isinstance(stage, str) or not stage:
        stage = None
    if not isinstance(gap, str) or not gap:
        gap = None
    return stage, gap


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
        mismatch = observed_profile is not None and observed_profile != requested["requested_profile"]
        achieved = outcome.get("achieved_conditions") if outcome else None
        # Copy only a native/PCM evidence result supplied by the caller. Never
        # promote planned_conditions to achieved_conditions.
        if achieved is not None and not outcome.get("achieved_conditions_source"):
            raise ValueError("achieved_conditions require an actual evidence source")
        failure_stage, gap_state = _outcome_failure_fields(outcome)
        if isinstance(outcome, dict):
            if failure_stage and not outcome.get("failure_stage"):
                outcome["failure_stage"] = failure_stage
            if gap_state and not outcome.get("gap_state"):
                outcome["gap_state"] = gap_state
        rows.append({"episode_id": requested["episode_id"], "room_id": requested["room_id"],
                     "room_family": requested["room_family"], "condition_group": requested["condition_group"],
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
