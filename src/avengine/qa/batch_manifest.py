"""Preallocate independent QA requests and split connected Episode groups.

Counters exist only while preparing a batch. Executing any saved request needs
no batch state and cannot replace a failed profile with an easier one.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
import random
from typing import Any, Mapping, Sequence

from avengine.rooms.conditioned_sampler import (
    neutral_source_declaration,
    resolve_condition_profile,
    sound_matches,
)

SOURCE_CLASSES = ("articulated_human", "articulated_animal", "rigid_static_object")
QA_IDS = tuple(f"QA-{index:02d}" for index in range(1, 25))


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
            used_identities = set()
            for actor_index, assignment in enumerate(assignments):
                actor_id = assignment["actor_id"]
                assignment["speaking"] = actor_index in condition["speaking_indices"]
                if not assignment["speaking"]:
                    assignment["sound_status"] = "silent_by_request"
                    continue
                actor = neutral_source_declaration(assets[assignment["asset_id"]], actor_id)
                groups = defaultdict(list)
                for sound in sounds:
                    # Respect an explicit asset allowlist for every source class.
                    allowed = sound.get("compatible_asset_ids")
                    if allowed is not None and assignment["asset_id"] not in allowed:
                        continue
                    if not sound_matches(actor, sound):
                        continue
                    identity = sound_identity(sound)
                    if identity is not None:
                        groups[identity].append(sound)
                distinct = [identity for identity in sorted(groups) if identity not in used_identities]
                if not distinct:
                    assignment["sound_status"] = "evidence_missing_or_unsampled"
                    allowlists[actor_id] = []
                    gaps.append({"state": "evidence_missing_or_unsampled",
                                 "code": "no_distinct_compatible_sound_identity",
                                 "asset_id": assignment["asset_id"], "actor_id": actor_id})
                    continue
                appearance_key = json.dumps(assignment["appearance"], sort_keys=True)
                kind = assignment["source_class"]
                identity = _minimum_choice(distinct,
                    lambda value: (cross_counts[(kind, appearance_key, value)], identity_counts[(kind, value)]), rng)
                used_identities.add(identity)
                identity_counts[(kind, identity)] += 1
                cross_counts[(kind, appearance_key, identity)] += 1
                entries = sorted(groups[identity], key=lambda sound: sound["sound_asset_id"])
                allowed_ids = [sound["sound_asset_id"] for sound in entries]
                assignment.update(sound_status="preallocated", sound_identity_id=identity,
                                  sound_asset_ids=allowed_ids,
                                  sound_origins=sorted({str(sound.get("source_pcm_path") or sound.get("source_origin")
                                                         or sound.get("original_source_uri")) for sound in entries
                                                        if sound.get("source_pcm_path") or sound.get("source_origin")
                                                        or sound.get("original_source_uri")}))
                allowlists[actor_id] = allowed_ids
            request["sound_selection"] = {**request.get("sound_selection", {}),
                                          "preallocated_sound_asset_ids_by_actor": allowlists}
            if all(assignment.get("sound_status") == "preallocated" for assignment in assignments
                   if assignment.get("speaking")):
                by_sound_id = {sound["sound_asset_id"]: sound for sound in sounds}
                minimum_seconds = [
                    min(by_sound_id[value]["sample_count"] / by_sound_id[value]["sample_rate_hz"]
                        for value in assignment["sound_asset_ids"])
                    for assignment in assignments if assignment.get("speaking")]
                relation = condition["event_relation"]
                gap = condition["min_gap_between_audible_windows_s"]
                if relation == "overlap":
                    lower_bound = max(minimum_seconds)
                else:
                    lower_bound = sum(minimum_seconds) + gap * (len(minimum_seconds) - 1)
                    if relation == "repeat":
                        lower_bound += min(minimum_seconds) + gap
                available = (float(request.get("frame_count", 240)) /
                             float(request.get("frame_rate_hz", 15))) - condition["reserve_tail_s"]
                if lower_bound > available + 1e-9:
                    gaps.append({
                        "state": "evidence_missing_or_unsampled",
                        "code": "fixed_sound_identities_exceed_profile_clip_budget",
                        "minimum_program_seconds_under_sampler_clip_budget": lower_bound,
                        "available_seconds_before_reserved_tail": available,
                        "identity_substitution_applied": False})
        rows.append({"episode_id": episode_id, "room_id": room_id, "room_family": room["family"],
                     "renderer": room["renderer"], "condition_group": _text(slot.get("condition_group"), "condition_group"),
                     "requested_source_classes": deepcopy(classes), "requested_profile": deepcopy(condition),
                     "requested_quota_by_qa": {qa: 1 for qa in QA_IDS}, "source_assignments": assignments,
                     "preallocation_gaps": gaps, "request": request, "execution_status": "not_run",
                     "achieved_conditions": None})
    group_quota = Counter((row["room_family"], row["room_id"], row["condition_group"]) for row in rows)
    return {"schema": "avengine_qa_batch_manifest_v1", "batch_id": batch_id, "seed": seed,
            "claim_boundary": "Preallocated requests only; no native execution, achieved quota or admission claim.",
            "allocation_policy": "offline_least_used_appearance_asset_and_sound_identity_with_seeded_ties",
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
        rows.append({"episode_id": requested["episode_id"], "room_id": requested["room_id"],
                     "room_family": requested["room_family"], "condition_group": requested["condition_group"],
                     "requested_profile": deepcopy(requested["requested_profile"]),
                     "requested_source_assignments": deepcopy(requested["source_assignments"]),
                     "outcome": outcome, "status": "not_run" if outcome is None else outcome["status"],
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
    return {"schema": "avengine_qa_batch_outcomes_v1", "batch_id": manifest["batch_id"],
            "episode_denominator": len(rows), "episodes": rows,
            "outcome_counts": dict(Counter(row["status"] for row in rows)),
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
