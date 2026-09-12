"""Callable capability resolution for registered source assets and the final sound pool.

The QA task families need to know, per registered source asset, what the project can
actually do with it: which family it belongs to, whether it can be instanced more than
once, whether it can act, where it may be placed, where it emits, which appearance can
be named, and which pool sounds may bind to it.  Those are separate facts with separate
evidence, so they are resolved separately here and a missing measurement never collapses
into a semantically inapplicable dimension.

Five identifiers stay distinct because the cross-event identity task depends on it:

``asset_id``            registered source asset (with ``asset_revision``)
``entity_instance_id``  one physical instance placed in a scene; one asset may carry many
``sound_asset_id``      one entry of the final sound pool
``event_id``            one scheduled emission; one instance may carry many
``sound_identity_id``   recognisable content identity of the audio (speaker/original file)

A shared voiceprint or a shared object category is never instance evidence, so instance
grouping keys strictly on ``entity_instance_id`` and the shortcut is reported instead.

Layering: this module reads the runtime registry and an already-built sound pool and
imports nothing from ``avengine.qa`` or ``avengine.rooms``, so the condition samplers and
manifest builders in those packages can call into it without an import cycle.

Nothing here writes: registry documents, pool documents and PCM stay untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import wave
from typing import Any, Mapping, Sequence

from avengine.runtime_profiles import (
    load_source_asset_runtime_registry,
    resolve_source_asset_runtime_profile,
    source_asset_runtime_index,
)

SOURCE_CAPABILITY_SCHEMA = "avengine_source_capabilities_v1"

FAMILY_HUMAN = "human"
FAMILY_ANIMAL = "animal"
FAMILY_DEVICE = "device"
SOURCE_FAMILIES = (FAMILY_HUMAN, FAMILY_ANIMAL, FAMILY_DEVICE)

RIGID_ENTITY_CLASSES = frozenset({"rigid_object", "rigid_static_object"})
_CLASS_TO_FAMILY = {
    "articulated_human": FAMILY_HUMAN,
    "articulated_animal": FAMILY_ANIMAL,
    "rigid_object": FAMILY_DEVICE,
    "rigid_static_object": FAMILY_DEVICE,
}

# The positive token is capability availability rather than a produced dataset item, but
# the three gap tokens are deliberately the same strings as the coverage accounting in
# ``avengine.qa.batch_coverage.COVERAGE_STATES`` so one vocabulary spans both layers.
STATE_AVAILABLE = "available"
STATE_NOT_APPLICABLE = "not_applicable_by_definition"
STATE_NOT_IMPLEMENTED = "interface_not_implemented"
STATE_EVIDENCE_MISSING = "evidence_missing_or_unsampled"
CAPABILITY_STATES = (
    STATE_AVAILABLE,
    STATE_NOT_APPLICABLE,
    STATE_NOT_IMPLEMENTED,
    STATE_EVIDENCE_MISSING,
)
GAP_STATES = (STATE_NOT_APPLICABLE, STATE_NOT_IMPLEMENTED, STATE_EVIDENCE_MISSING)

CAPABILITY_DIMENSIONS = ("emission", "placement", "locomotion", "appearance")

SPEECH_SOUND_CLASSES = frozenset({"speech", "speech_playback"})
MUSIC_PLAYBACK_SOUND_CLASS = "music_playback"
GENERIC_PLAYBACK_SOUND_CLASS = "any_audioset_class_playback"

# Placement verdict the asset finalizer emits when a declared wall/ceiling mount has no
# measurable mounting plane.  The asset is registered and its geometry is real, but the
# placement interface cannot ground it, so it must not be presented as placeable.
NO_MOUNTING_PLANE_VERDICT = "no_mounting_plane_found"

# Set on the result of ``normalize_sound_class_config`` so a pre-resolved mapping can be
# threaded through hot loops without being rebuilt for every candidate pair.
_NORMALIZED_MARKER = "__avengine_sound_class_config_normalized__"


class SourceCapabilityError(ValueError):
    """A capability request is malformed or contradicts registered evidence."""


def _state(state: str, reason: str, **basis: Any) -> dict[str, Any]:
    if state not in CAPABILITY_STATES:
        raise SourceCapabilityError(f"unknown capability state: {state!r}")
    result = {"state": state, "reason": reason}
    if basis:
        result["basis"] = basis
    return result


# --------------------------------------------------------------------------- families


def source_family(record: Mapping[str, Any]) -> str:
    """Return ``human``/``animal``/``device`` for one registry record."""
    entity_class = record.get("entity_class")
    try:
        return _CLASS_TO_FAMILY[str(entity_class)]
    except KeyError as error:
        raise SourceCapabilityError(
            f"unknown entity_class {entity_class!r} for asset {record.get('asset_id')!r}"
        ) from error


def source_class_token(record: Mapping[str, Any]) -> str:
    """Return the request-facing source class token used by the condition profile."""
    entity_class = str(record.get("entity_class"))
    if entity_class in RIGID_ENTITY_CLASSES:
        return "rigid_static_object"
    if entity_class not in _CLASS_TO_FAMILY:
        raise SourceCapabilityError(f"unknown entity_class {entity_class!r}")
    return entity_class


def entity_combinations() -> tuple[tuple[str, str], ...]:
    """Return the six unordered two-entity family combinations, all retained."""
    return tuple(
        (SOURCE_FAMILIES[i], SOURCE_FAMILIES[j])
        for i in range(len(SOURCE_FAMILIES))
        for j in range(i, len(SOURCE_FAMILIES))
    )


def combination_key(first: str, second: str) -> str:
    """Canonical order-insensitive key for a two-entity family combination."""
    for family in (first, second):
        if family not in SOURCE_FAMILIES:
            raise SourceCapabilityError(f"unknown source family: {family!r}")
    return "+".join(sorted((first, second), key=SOURCE_FAMILIES.index))


# ----------------------------------------------------------------- sound class mapping


def normalize_sound_class_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Read the declared semantic sound mapping without inventing defaults.

    ``human_nonverbal_sound_classes`` is intentionally allowed to be absent: the current
    production configuration declares none, and an empty declaration is a real coverage
    gap rather than a licence to guess laughter/cough classes.
    """
    config = config or {}
    if config.get(_NORMALIZED_MARKER):
        # Hot loops resolve the mapping once and pass it down; re-normalizing per
        # (asset, sound) pair dominated the whole report before this check existed.
        return config if isinstance(config, dict) else dict(config)
    species = config.get("species_sound_classes") or {}
    objects = config.get("object_sound_classes") or {}
    if not isinstance(species, Mapping) or not isinstance(objects, Mapping):
        raise SourceCapabilityError("sound class mappings must be objects")
    categories = config.get("speech_playback_categories")
    if categories is None:
        categories = ["audio_playback"]
    nonverbal = config.get("human_nonverbal_sound_classes") or []
    if not isinstance(categories, Sequence) or isinstance(categories, (str, bytes)):
        raise SourceCapabilityError("speech_playback_categories must be a list")
    if not isinstance(nonverbal, Sequence) or isinstance(nonverbal, (str, bytes)):
        raise SourceCapabilityError("human_nonverbal_sound_classes must be a list")
    return {
        _NORMALIZED_MARKER: True,
        "species_sound_classes": {str(k): [str(c) for c in v] for k, v in species.items()},
        "object_sound_classes": {str(k): [str(c) for c in v] for k, v in objects.items()},
        "speech_playback_categories": [str(value) for value in categories],
        "human_nonverbal_sound_classes": [str(value) for value in nonverbal],
    }


def declared_sound_classes(
    record: Mapping[str, Any], config: Mapping[str, Any] | None = None
) -> list[str]:
    """Return the sound classes the configuration declares for this asset."""
    resolved = normalize_sound_class_config(config)
    family = source_family(record)
    identity = record.get("identity") or {}
    if family == FAMILY_ANIMAL:
        return list(resolved["species_sound_classes"].get(str(identity.get("species_id")), []))
    if family == FAMILY_DEVICE:
        declared = list(
            resolved["object_sound_classes"].get(str(identity.get("object_type")), [])
        )
        if str(identity.get("category")) in resolved["speech_playback_categories"]:
            for sound_class in ("speech_playback",):
                if sound_class not in declared:
                    declared.append(sound_class)
        return declared
    return list(resolved["human_nonverbal_sound_classes"]) + ["speech_playback"]


def sound_class_asset_index(
    registry: Mapping[str, Any], config: Mapping[str, Any] | None = None
) -> dict[str, list[str]]:
    """Invert the declared mapping once: sound class -> sorted accepting asset IDs.

    Callers that classify a whole sound library previously rescanned every asset for
    every sound; one inversion keeps that work proportional to the registry instead of
    to library size times registry size.

    Registry order is preserved rather than sorted, because the produced sound pools
    already carry their allowlists in registry order and a reordering would rewrite
    existing artifacts for no gain.
    """
    resolved = normalize_sound_class_config(config)
    index: dict[str, list[str]] = defaultdict(list)
    for record in registry.get("assets", []):
        asset_id = str(record["asset_id"])
        for sound_class in declared_sound_classes(record, resolved):
            if asset_id not in index[sound_class]:
                index[sound_class].append(asset_id)
    return dict(index)


def assets_accepting_sound_class(
    registry: Mapping[str, Any],
    sound_class: str,
    config: Mapping[str, Any] | None = None,
    *,
    index: Mapping[str, Sequence[str]] | None = None,
) -> list[str]:
    """Return the registered assets whose declared mapping accepts ``sound_class``."""
    if index is None:
        index = sound_class_asset_index(registry, config)
    return list(index.get(str(sound_class), []))


# ------------------------------------------------------------------ capability probes


def emission_capability(record: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the measured emission point(s) and the per-backend emitter binding."""
    anchors = record.get("emitter_anchors")
    default_anchor_id = record.get("default_emitter_anchor_id")
    if not isinstance(anchors, Sequence) or not anchors:
        return _state(STATE_EVIDENCE_MISSING, "asset declares no emitter anchor")
    matches = [
        anchor for anchor in anchors if anchor.get("anchor_id") == default_anchor_id
    ]
    if len(matches) != 1:
        return _state(
            STATE_EVIDENCE_MISSING,
            "default emitter anchor does not resolve to exactly one anchor",
            default_emitter_anchor_id=default_anchor_id,
            anchor_ids=[anchor.get("anchor_id") for anchor in anchors],
        )
    backends = record.get("runtime_backends") or {}
    habitat_emitter = (backends.get("habitat") or {}).get("emitter")
    result = _state(
        STATE_AVAILABLE,
        "measured emitter anchor resolved from the runtime registry",
        default_emitter_anchor_id=str(default_anchor_id),
        anchor_type=matches[0].get("anchor_type"),
        offset_m=deepcopy(list(matches[0].get("offset_m", []))),
        offset_space=matches[0].get("offset_space"),
        anchor_ids=[str(anchor.get("anchor_id")) for anchor in anchors],
    )
    basis = result["basis"]
    basis["habitat_joint_bound"] = isinstance(habitat_emitter, Mapping) and bool(
        habitat_emitter.get("joint_id")
    )
    basis["habitat_semantic_anchor_id"] = (
        habitat_emitter.get("semantic_anchor_id")
        if isinstance(habitat_emitter, Mapping)
        else None
    )
    return result


def placement_capability(record: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve where the asset may rest, from the measured resting pose only.

    The room-side floor reference stays the authority for ground height; this reports the
    asset-local measurement so no caller hand-writes a ground offset from it.
    """
    habitat = (record.get("runtime_backends") or {}).get("habitat") or {}
    pose = habitat.get("resting_pose")
    if not isinstance(pose, Mapping):
        return _state(STATE_EVIDENCE_MISSING, "asset declares no measured resting pose")
    surface = pose.get("attachment_surface")
    verdict = pose.get("verdict")
    basis = {
        "attachment_surface": surface,
        "attachment_surface_assumed": pose.get("attachment_surface_assumed"),
        "verdict": verdict,
        "base_plane_offset_m": pose.get("base_plane_offset_m"),
        "measured_from": pose.get("measured_from"),
        "floor_reference_owner": "room_runtime_profile",
        "evidence_layer": "registry_resting_pose_record",
        "native_placement_verified_backends": [],
        "claim_boundary": (
            "available means the measured resting-pose record is usable; it is not "
            "evidence that native placement passed on any backend"
        ),
    }
    if verdict == NO_MOUNTING_PLANE_VERDICT:
        return _state(
            STATE_NOT_IMPLEMENTED,
            f"declared {surface} mount has no measurable mounting plane",
            **basis,
        )
    if source_family(record) != FAMILY_DEVICE:
        # Articulated placement authority is native navigation plus the floor contact
        # gate; the finalizer verdict field applies to rigid assets only.
        basis["locomotion_placement_authority"] = "native_navigation"
        return _state(
            STATE_AVAILABLE, "native skin-rest probe measured a floor-resting pose", **basis
        )
    if surface is None:
        return _state(STATE_EVIDENCE_MISSING, "resting pose declares no attachment surface", **basis)
    if verdict is None:
        return _state(STATE_EVIDENCE_MISSING, "rigid resting pose carries no verdict", **basis)
    for key in ("height_m", "footprint_extent_m", "how_to_place", "tolerance"):
        if pose.get(key) is not None:
            basis[key] = deepcopy(pose[key])
    return _state(STATE_AVAILABLE, f"measured {surface} resting pose, verdict {verdict}", **basis)


def locomotion_capability(record: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve whether the asset can act and walk under its own registered actions.

    A device has no registered locomotion by definition rather than by missing data: the
    project rule is that static devices never walk themselves, so a device is reported
    ``not_applicable_by_definition`` and can still hold every non-locomotion role.
    """
    family = source_family(record)
    if family == FAMILY_DEVICE:
        return _state(
            STATE_NOT_APPLICABLE,
            "a registered device is a static source and is never a self-locomotion target",
            motion_model="rigid_static",
            registered_timeline=False,
        )
    timeline = record.get("timeline")
    if not isinstance(timeline, Mapping):
        return _state(
            STATE_EVIDENCE_MISSING,
            "articulated asset carries no registered Timeline action profile",
            motion_model="articulated",
        )
    missing = [
        key
        for key in ("idle_action_id", "walking_action_id", "body_plan_id",
                    "walk_phase_period_frames")
        if timeline.get(key) in (None, "")
    ]
    if missing:
        return _state(
            STATE_EVIDENCE_MISSING,
            "registered Timeline profile is incomplete",
            motion_model="articulated",
            missing_fields=missing,
        )
    spear = (record.get("runtime_backends") or {}).get("spear_unreal") or {}
    return _state(
        STATE_AVAILABLE,
        "registered idle/walk actions resolve for this body plan",
        motion_model="articulated",
        body_plan_id=timeline["body_plan_id"],
        idle_action_id=timeline["idle_action_id"],
        walking_action_id=timeline["walking_action_id"],
        walk_phase_period_frames=int(timeline["walk_phase_period_frames"]),
        floor_contact_gate=spear.get("floor_contact_gate"),
    )


def appearance_capability(record: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the observed appearance label, keeping an unknown colour visible.

    A missing colour label means the instance cannot be *named* by colour; it never means
    the instance is absent from the picture, so this stays a metadata gap.
    """
    attributes = record.get("realized_attributes") or {}
    family = source_family(record)
    if family == FAMILY_HUMAN:
        field, value = "top_color", attributes.get("top_color")
    elif family == FAMILY_ANIMAL:
        field = "coat_profile.value"
        value = (attributes.get("coat_profile") or {}).get("value")
    else:
        field = "finish" if attributes.get("finish") is not None else "body_color"
        value = attributes.get(field)
    basis = {
        "field": field,
        "value": value,
        "source": "source_asset_runtime_registry.realized_attributes",
        "display_label": record.get("display_label"),
        "pixel_visibility_authority": "native_pixel_readback",
    }
    if value is None:
        return _state(
            STATE_EVIDENCE_MISSING,
            f"registry declares no {field}; the instance may still be visible and counted",
            **basis,
        )
    return _state(STATE_AVAILABLE, f"registered {field} is nameable", **basis)


def instancing_capability(record: Mapping[str, Any]) -> dict[str, Any]:
    """Repeated instances of one asset are expressible for every registered family."""
    return _state(
        STATE_AVAILABLE,
        "one registered asset may back several distinct physical instances",
        instance_identity_field="entity_instance_id",
        distinguished_by=["entity_instance_id", "placement", "route"],
        never_distinguished_by=["sound_identity_id", "identity.category"],
    )


def resolve_source_capabilities(
    registry: Mapping[str, Any], asset_id: str, *, revision: str | None = None
) -> dict[str, Any]:
    """Resolve every capability dimension for one registered asset."""
    return _capabilities_for_record(
        resolve_source_asset_runtime_profile(registry, asset_id, revision)
    )


def _capabilities_for_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Capability core for an already-resolved registry record.

    Callers that sweep the whole registry resolve the index once and reach this
    directly; resolving each asset through the registry revalidates and re-hashes the
    whole document, which measurably dominated a full report.
    """
    capabilities = {
        "emission": emission_capability(record),
        "placement": placement_capability(record),
        "locomotion": locomotion_capability(record),
        "appearance": appearance_capability(record),
    }
    backends = record.get("runtime_backends") or {}
    return {
        "schema": SOURCE_CAPABILITY_SCHEMA,
        "asset_id": str(record["asset_id"]),
        "asset_revision": str(record["revision"]),
        "family": source_family(record),
        "source_class": source_class_token(record),
        "display_label": record.get("display_label"),
        "identity": deepcopy(dict(record.get("identity") or {})),
        "realized_attributes": deepcopy(dict(record.get("realized_attributes") or {})),
        "admission_state": record.get("admission_state"),
        "runtime_backends": sorted(str(name) for name in backends),
        "instancing": instancing_capability(record),
        "capabilities": capabilities,
        "capability_states": {name: value["state"] for name, value in capabilities.items()},
    }


# ------------------------------------------------------------- sound compatibility


def _pool_sounds(pool: Any) -> list[Mapping[str, Any]]:
    if isinstance(pool, Mapping):
        sounds = pool.get("sounds")
    else:
        sounds = pool
    if not isinstance(sounds, Sequence):
        raise SourceCapabilityError("sound pool must carry a 'sounds' list")
    return [sound for sound in sounds if isinstance(sound, Mapping)]


def _gender(value: Any) -> str | None:
    return {"m": "male", "f": "female", "male": "male", "female": "female"}.get(
        str(value).lower()
    )


def sound_identity_of(sound: Mapping[str, Any]) -> str | None:
    """Return the recognisable content identity, never a crop or event identifier."""
    if sound.get("sound_identity_id"):
        return str(sound["sound_identity_id"])
    if sound.get("speaker_id"):
        return "speaker:" + str(sound["speaker_id"])
    for field in ("source_pcm_path", "source_origin", "original_source_uri"):
        if sound.get(field):
            return "source:" + str(sound[field])
    return None


def sound_recognizability(sound: Mapping[str, Any]) -> dict[str, Any]:
    """Report the declared recognisability target without upgrading it to a certificate."""
    review = sound.get("human_review")
    review_status = review.get("status") if isinstance(review, Mapping) else None
    return {
        "sound_identity_id": sound_identity_of(sound),
        "human_review_status": review_status,
        "activity_calibration": sound.get("activity_calibration"),
        "activity_measurement": sound.get("activity_measurement"),
        "metadata_status": sound.get("metadata_status"),
        "transcript_available": sound.get("transcript") not in (None, ""),
        "certified": False,
        "claim_boundary": (
            "declared target and placeholder detector extents; no human audibility or "
            "event-count certification"
        ),
    }


def sound_compatibility(
    record: Mapping[str, Any],
    sound: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Decide whether one pool sound may bind to one registered asset, with the reason.

    Declared semantic mapping and the pool entry's own allowlist must both admit the
    pairing, and the deciding evidence is named so a rejection can be audited.
    """
    resolved = normalize_sound_class_config(config)
    family = source_family(record)
    identity = record.get("identity") or {}
    asset_id = str(record["asset_id"])
    sound_class = str(sound.get("sound_class") or "")
    declared = declared_sound_classes(record, resolved)

    allowed_assets = sound.get("compatible_asset_ids")
    if isinstance(allowed_assets, (list, tuple)):
        if allowed_assets and asset_id not in {str(value) for value in allowed_assets}:
            return {
                "compatible": False,
                "state": STATE_NOT_APPLICABLE,
                "reason": "pool_entry_asset_allowlist_excludes_asset",
                "basis": {"decided_by": "sound.compatible_asset_ids", "sound_class": sound_class},
            }
    allowed_categories = sound.get("compatible_object_categories")
    if isinstance(allowed_categories, (list, tuple)):
        if allowed_categories and family == FAMILY_DEVICE:
            if str(identity.get("category")) not in {str(v) for v in allowed_categories}:
                return {
                    "compatible": False,
                    "state": STATE_NOT_APPLICABLE,
                    "reason": "pool_entry_category_allowlist_excludes_category",
                    "basis": {
                        "decided_by": "sound.compatible_object_categories",
                        "category": identity.get("category"),
                    },
                }

    if family == FAMILY_HUMAN and sound_class in SPEECH_SOUND_CLASSES:
        asset_sex = _gender((record.get("realized_attributes") or {}).get("sex_or_gender_label"))
        sound_gender = _gender(sound.get("gender"))
        if asset_sex is None:
            return {
                "compatible": False,
                "state": STATE_EVIDENCE_MISSING,
                "reason": "asset_declares_no_usable_sex_or_gender_label",
                "basis": {"decided_by": "registry.realized_attributes.sex_or_gender_label"},
            }
        if sound_gender is None:
            return {
                "compatible": False,
                "state": STATE_EVIDENCE_MISSING,
                "reason": "speech_sound_declares_no_usable_gender_metadata",
                "basis": {"decided_by": "sound.gender", "gender": sound.get("gender")},
            }
        if asset_sex != sound_gender:
            return {
                "compatible": False,
                "state": STATE_NOT_APPLICABLE,
                "reason": "speech_gender_does_not_match_registered_appearance",
                "basis": {"asset_sex": asset_sex, "sound_gender": sound_gender},
            }
        return {
            "compatible": True,
            "state": STATE_AVAILABLE,
            "reason": "speech_gender_matches_registered_appearance",
            "basis": {
                "decided_by": "registry.realized_attributes + sound.gender",
                "asset_sex": asset_sex,
                "sound_class": sound_class,
            },
        }

    if family == FAMILY_ANIMAL:
        species = identity.get("species_id")
        declared_species = sound.get("species_id")
        if species is None:
            return {
                "compatible": False,
                "state": STATE_EVIDENCE_MISSING,
                "reason": "animal_asset_declares_no_species_id",
                "basis": {"decided_by": "registry.identity.species_id"},
            }
        if declared_species is not None and str(declared_species) == str(species):
            return {
                "compatible": True,
                "state": STATE_AVAILABLE,
                "reason": "pool_entry_declares_the_same_species",
                "basis": {"decided_by": "sound.species_id", "species_id": str(species)},
            }
        if sound_class in declared:
            return {
                "compatible": True,
                "state": STATE_AVAILABLE,
                "reason": "configured_species_sound_class_accepts_this_asset",
                "basis": {
                    "decided_by": "config.species_sound_classes",
                    "species_id": str(species),
                    "sound_class": sound_class,
                },
            }
        return {
            "compatible": False,
            "state": STATE_NOT_APPLICABLE,
            "reason": "sound_class_is_not_declared_for_this_species",
            "basis": {"species_id": str(species), "sound_class": sound_class,
                      "declared_classes": declared},
        }

    if sound_class in declared:
        decided_by = (
            "config.object_sound_classes" if family == FAMILY_DEVICE
            else "config.human_nonverbal_sound_classes"
        )
        return {
            "compatible": True,
            "state": STATE_AVAILABLE,
            "reason": "configured_semantic_mapping_accepts_this_asset",
            "basis": {"decided_by": decided_by, "sound_class": sound_class},
        }
    if family == FAMILY_HUMAN:
        return {
            "compatible": False,
            "state": (
                STATE_EVIDENCE_MISSING if not resolved["human_nonverbal_sound_classes"]
                else STATE_NOT_APPLICABLE
            ),
            "reason": (
                "no_human_nonverbal_sound_classes_are_configured"
                if not resolved["human_nonverbal_sound_classes"]
                else "sound_class_is_not_declared_for_humans"
            ),
            "basis": {"decided_by": "config.human_nonverbal_sound_classes",
                      "sound_class": sound_class, "declared_classes": declared},
        }
    return {
        "compatible": False,
        "state": STATE_NOT_APPLICABLE,
        "reason": "sound_class_is_not_declared_for_this_object_type",
        "basis": {
            "decided_by": "config.object_sound_classes",
            "object_type": identity.get("object_type"),
            "sound_class": sound_class,
            "declared_classes": declared,
        },
    }


def compatible_sounds(
    registry: Mapping[str, Any],
    asset_id: str,
    pool: Any,
    config: Mapping[str, Any] | None = None,
    *,
    revision: str | None = None,
) -> dict[str, Any]:
    """Resolve every pool sound that may bind to one asset, keeping rejection reasons."""
    return _compatible_sounds_for_record(
        resolve_source_asset_runtime_profile(registry, asset_id, revision),
        _pool_sounds(pool),
        normalize_sound_class_config(config),
    )


def _compatible_sounds_for_record(
    record: Mapping[str, Any],
    sounds: Sequence[Mapping[str, Any]],
    resolved: Mapping[str, Any],
) -> dict[str, Any]:
    """Sound-compatibility core for an already-resolved record, pool and config."""
    accepted: list[dict[str, Any]] = []
    rejections: Counter = Counter()
    for sound in sounds:
        verdict = sound_compatibility(record, sound, resolved)
        if not verdict["compatible"]:
            rejections[verdict["reason"]] += 1
            continue
        accepted.append(
            {
                "sound_asset_id": sound.get("sound_asset_id"),
                "sound_class": sound.get("sound_class"),
                "sound_identity_id": sound_identity_of(sound),
                "decided_by": verdict["basis"].get("decided_by"),
                "recognizability": sound_recognizability(sound),
            }
        )
    declared = declared_sound_classes(record, resolved)
    if accepted:
        state, reason = STATE_AVAILABLE, "declared mapping resolves usable pool sounds"
    elif not declared:
        state = STATE_EVIDENCE_MISSING
        reason = "no_sound_class_is_declared_for_this_asset"
    else:
        state = STATE_EVIDENCE_MISSING
        reason = "declared_sound_classes_have_no_usable_entry_in_this_pool"
    return {
        "asset_id": str(record["asset_id"]),
        "family": source_family(record),
        "state": state,
        "reason": reason,
        "declared_sound_classes": declared,
        "candidate_count": len(accepted),
        "candidates": accepted,
        "candidate_classes": dict(
            Counter(str(item["sound_class"]) for item in accepted)
        ),
        "rejections_by_reason": dict(rejections),
        "pool_size": len(sounds),
    }


def music_suite_capability(
    registry: Mapping[str, Any], pool: Any, config: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Report the music playback integration point and what the current pool holds.

    Music is reachable today as ``music_playback`` entries bound to playback devices.  A
    packaged multi-track music suite is a separate asset family that is not registered,
    so this names the extension point instead of implying it already exists.
    """
    resolved = normalize_sound_class_config(config)
    sounds = _pool_sounds(pool)
    index = sound_class_asset_index(registry, resolved)
    music_entries = [
        sound for sound in sounds
        if str(sound.get("sound_class")) == MUSIC_PLAYBACK_SOUND_CLASS
    ]
    accepting = assets_accepting_sound_class(
        registry, MUSIC_PLAYBACK_SOUND_CLASS, resolved, index=index
    )
    if not accepting:
        state = STATE_EVIDENCE_MISSING
        reason = "no registered asset declares music_playback"
    elif not music_entries:
        state = STATE_EVIDENCE_MISSING
        reason = "music_playback is declared but the current pool holds no entry"
    else:
        state = STATE_AVAILABLE
        reason = "music_playback entries bind to registered playback devices"
    return {
        "state": state,
        "reason": reason,
        "sound_class": MUSIC_PLAYBACK_SOUND_CLASS,
        "pool_entry_count": len(music_entries),
        "accepting_asset_ids": accepting,
        "generic_playback_entry_count": sum(
            1 for sound in sounds
            if str(sound.get("sound_class")) == GENERIC_PLAYBACK_SOUND_CLASS
        ),
        "packaged_suite": {
            "state": STATE_NOT_IMPLEMENTED,
            "reason": (
                "a packaged multi-track music suite is not a registered asset family; "
                "individual music_playback clips are the only music route today"
            ),
            "extension_point": (
                "declare the suite's classes under config.object_sound_classes for the "
                "playback object types, then admit its clips into the sound pool; no "
                "code path here needs a new branch"
            ),
        },
    }


# ------------------------------------------------------- instances, events, planning


def make_instance_id(asset_id: str, ordinal: int) -> str:
    """Build a distinct physical-instance ID for the n-th instance of one asset."""
    if not isinstance(asset_id, str) or not asset_id:
        raise SourceCapabilityError("asset_id must be a non-empty string")
    if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 1:
        raise SourceCapabilityError("instance ordinal must be a positive integer")
    return f"{asset_id}#instance{ordinal:02d}"


def make_event_id(entity_instance_id: str, ordinal: int) -> str:
    """Build a distinct event ID for the n-th emission of one physical instance."""
    if not isinstance(entity_instance_id, str) or not entity_instance_id:
        raise SourceCapabilityError("entity_instance_id must be a non-empty string")
    if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 1:
        raise SourceCapabilityError("event ordinal must be a positive integer")
    return f"{entity_instance_id}:event{ordinal:02d}"


def assert_motion_target(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the locomotion capability, refusing a device as a self-motion target."""
    capability = locomotion_capability(record)
    if capability["state"] != STATE_AVAILABLE:
        raise SourceCapabilityError(
            f"asset {record.get('asset_id')!r} cannot be a self-locomotion target: "
            f"{capability['state']}: {capability['reason']}"
        )
    return capability


def declare_instances(
    registry: Mapping[str, Any], requests: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Materialize physical instances, allowing several instances of one asset.

    Each request declares ``asset_id`` and may declare ``source_slot_id`` and
    ``motion`` (``static``/``walk``).  A repeated asset yields distinct
    ``entity_instance_id`` values, so two same-model devices or two same-model dogs are
    separate physical entities rather than one merged source.
    """
    if not isinstance(requests, Sequence) or isinstance(requests, (str, bytes)):
        raise SourceCapabilityError("instance requests must be a sequence")
    ordinals: Counter = Counter()
    instances: list[dict[str, Any]] = []
    seen_slots: set[str] = set()
    for position, request in enumerate(requests, start=1):
        if not isinstance(request, Mapping):
            raise SourceCapabilityError("each instance request must be an object")
        asset_id = str(request.get("asset_id") or "")
        record = resolve_source_asset_runtime_profile(
            registry, asset_id, request.get("asset_revision")
        )
        slot = request.get("source_slot_id") or f"source{position}"
        slot = str(slot)
        if slot in seen_slots:
            raise SourceCapabilityError(f"duplicate source slot: {slot!r}")
        seen_slots.add(slot)
        ordinals[asset_id] += 1
        instance_id = str(
            request.get("entity_instance_id") or make_instance_id(asset_id, ordinals[asset_id])
        )
        motion = str(request.get("motion") or "static")
        if motion not in {"static", "walk"}:
            raise SourceCapabilityError(f"unsupported motion request: {motion!r}")
        locomotion = (
            assert_motion_target(record) if motion == "walk"
            else locomotion_capability(record)
        )
        instances.append(
            {
                "entity_instance_id": instance_id,
                "asset_id": asset_id,
                "asset_revision": str(record["revision"]),
                "source_slot_id": slot,
                "instance_ordinal": ordinals[asset_id],
                "family": source_family(record),
                "source_class": source_class_token(record),
                "motion": motion,
                "locomotion": locomotion,
                "placement": placement_capability(record),
                "emission": emission_capability(record),
                "appearance": appearance_capability(record),
            }
        )
    identifiers = [instance["entity_instance_id"] for instance in instances]
    if len(set(identifiers)) != len(identifiers):
        raise SourceCapabilityError("physical instance IDs must be distinct")
    return instances


def declare_events(
    registry: Mapping[str, Any],
    instance: Mapping[str, Any],
    sounds: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Bind one or more emissions to one physical instance.

    Several events per instance are expressible: one entity owning two events is a normal
    case, never a general one-event-per-entity restriction.  Every event carries its own
    ``event_id`` and repeats the owning ``entity_instance_id``, so identity is never
    inferred from the audio content.
    """
    record = resolve_source_asset_runtime_profile(
        registry, str(instance.get("asset_id")), instance.get("asset_revision")
    )
    resolved = normalize_sound_class_config(config)
    if not isinstance(sounds, Sequence) or isinstance(sounds, (str, bytes)):
        raise SourceCapabilityError("sounds must be a sequence")
    events: list[dict[str, Any]] = []
    for ordinal, sound in enumerate(sounds, start=1):
        if not isinstance(sound, Mapping):
            raise SourceCapabilityError("each sound must be an object")
        verdict = sound_compatibility(record, sound, resolved)
        if not verdict["compatible"]:
            raise SourceCapabilityError(
                f"sound {sound.get('sound_asset_id')!r} cannot bind to instance "
                f"{instance.get('entity_instance_id')!r}: {verdict['reason']}"
            )
        events.append(
            {
                "event_id": make_event_id(str(instance["entity_instance_id"]), ordinal),
                "entity_instance_id": str(instance["entity_instance_id"]),
                "asset_id": str(record["asset_id"]),
                "asset_revision": str(record["revision"]),
                "event_ordinal": ordinal,
                "sound_asset_id": sound.get("sound_asset_id"),
                "sound_class": sound.get("sound_class"),
                "sound_identity_id": sound_identity_of(sound),
                "compatibility": verdict,
                "recognizability": sound_recognizability(sound),
            }
        )
    return events


def instance_event_identity_report(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Group events by physical instance and expose sound-identity shortcuts.

    Grouping keys only on ``entity_instance_id``.  Sound identities reused across
    distinct instances, and instances whose events all share one sound identity, are
    reported so no caller treats a shared voiceprint as identity evidence.
    """
    by_instance: dict[str, list[str]] = defaultdict(list)
    identity_to_instances: dict[str, set[str]] = defaultdict(set)
    identities_per_instance: dict[str, set[Any]] = defaultdict(set)
    for event in events:
        instance_id = event.get("entity_instance_id")
        if not isinstance(instance_id, str) or not instance_id:
            raise SourceCapabilityError("every event must name its entity_instance_id")
        by_instance[instance_id].append(str(event.get("event_id")))
        identity = event.get("sound_identity_id")
        identities_per_instance[instance_id].add(identity)
        if identity:
            identity_to_instances[str(identity)].add(instance_id)
    shared = {
        identity: sorted(instances)
        for identity, instances in identity_to_instances.items()
        if len(instances) > 1
    }
    single_identity_instances = sorted(
        instance_id
        for instance_id, identities in identities_per_instance.items()
        if len(identities) == 1 and len(by_instance[instance_id]) > 1
    )
    return {
        "grouping_key": "entity_instance_id",
        "instance_count": len(by_instance),
        "event_count": sum(len(values) for values in by_instance.values()),
        "events_by_instance": {key: sorted(value) for key, value in by_instance.items()},
        "multi_event_instances": sorted(
            key for key, value in by_instance.items() if len(value) > 1
        ),
        "sound_identities_shared_across_instances": shared,
        "instances_whose_events_share_one_sound_identity": single_identity_instances,
        "claim_boundary": (
            "a shared sound identity is not instance evidence; physical identity comes "
            "from entity_instance_id, placement and route"
        ),
    }


def plan_source_binding(
    registry: Mapping[str, Any],
    pool: Any,
    slots: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve one concrete multi-slot binding: instances, events and identity report.

    This is the entry a condition sampler calls once it has chosen assets and sounds.
    Each slot declares ``asset_id``, optional ``source_slot_id``/``motion`` and either
    ``sound_asset_ids`` or ``event_count``; sounds resolve out of the supplied pool.
    """
    resolved = normalize_sound_class_config(config)
    sounds_by_id = {
        str(sound.get("sound_asset_id")): sound for sound in _pool_sounds(pool)
    }
    instances = declare_instances(registry, slots)
    events: list[dict[str, Any]] = []
    for slot_request, instance in zip(slots, instances):
        requested = slot_request.get("sound_asset_ids")
        if requested is None:
            continue
        if isinstance(requested, (str, bytes)) or not isinstance(requested, Sequence):
            raise SourceCapabilityError("sound_asset_ids must be a list")
        chosen = []
        for sound_asset_id in requested:
            try:
                chosen.append(sounds_by_id[str(sound_asset_id)])
            except KeyError as error:
                raise SourceCapabilityError(
                    f"sound {sound_asset_id!r} is absent from the supplied pool"
                ) from error
        events.extend(declare_events(registry, instance, chosen, resolved))
    families = [instance["family"] for instance in instances]
    result = {
        "schema": SOURCE_CAPABILITY_SCHEMA,
        "instances": instances,
        "events": events,
        "identity": instance_event_identity_report(events) if events else {
            "grouping_key": "entity_instance_id", "instance_count": len(instances),
            "event_count": 0, "events_by_instance": {},
            "claim_boundary": "no event was declared for these instances",
        },
        "families": families,
        "repeated_assets": sorted(
            asset_id
            for asset_id, count in Counter(
                instance["asset_id"] for instance in instances
            ).items()
            if count > 1
        ),
        "motion_targets": [
            instance["entity_instance_id"]
            for instance in instances
            if instance["motion"] == "walk"
        ],
    }
    if len(families) == 2:
        result["combination"] = combination_key(*families)
    return result


# --------------------------------------------------------------------- reporting


def _asset_evaluations(
    index: Mapping[str, Mapping[str, Any]],
    sounds: Sequence[Mapping[str, Any]],
    resolved: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Resolve capabilities and sound bindings once per asset for a whole sweep."""
    return {
        asset_id: {
            "capabilities": _capabilities_for_record(record),
            "sound_binding": _compatible_sounds_for_record(record, sounds, resolved),
        }
        for asset_id, record in index.items()
    }


def combination_candidates(
    registry: Mapping[str, Any],
    pool: Any,
    combination: Sequence[str],
    config: Mapping[str, Any] | None = None,
    *,
    require_motion_target: bool = False,
    evaluations: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve per-family candidates for one two-entity combination.

    Slot states stay separate.  A mixed combination where one family cannot hold a role
    keeps the other family's real candidates, so a pair is never excluded as a whole
    class because one of its two slots is inapplicable.

    ``evaluations`` optionally supplies the per-asset resolution from
    ``_asset_evaluations`` so a caller sweeping all six combinations pays for it once.
    """
    if len(combination) != 2:
        raise SourceCapabilityError("a combination must name exactly two families")
    resolved = normalize_sound_class_config(config)
    key = combination_key(*combination)
    index = source_asset_runtime_index(registry)
    if evaluations is None:
        evaluations = _asset_evaluations(index, _pool_sounds(pool), resolved)
    slots = []
    for family in combination:
        entries = []
        for asset_id, record in sorted(index.items()):
            if source_family(record) != family:
                continue
            evaluation = evaluations[asset_id]
            sounds = evaluation["sound_binding"]
            capabilities = evaluation["capabilities"]
            motion = capabilities["capabilities"]["locomotion"]
            eligible = (
                sounds["state"] == STATE_AVAILABLE
                and capabilities["capability_states"]["emission"] == STATE_AVAILABLE
                and capabilities["capability_states"]["placement"] == STATE_AVAILABLE
            )
            if require_motion_target and motion["state"] != STATE_AVAILABLE:
                eligible = False
            entries.append(
                {
                    "asset_id": asset_id,
                    "eligible": eligible,
                    "sound_state": sounds["state"],
                    "sound_candidate_count": sounds["candidate_count"],
                    "capability_states": capabilities["capability_states"],
                    "blocking_states": {
                        name: value
                        for name, value in capabilities["capability_states"].items()
                        if value != STATE_AVAILABLE and name != "locomotion"
                    },
                    "motion_state": motion["state"],
                    "motion_reason": motion["reason"],
                }
            )
        eligible_ids = [entry["asset_id"] for entry in entries if entry["eligible"]]
        if eligible_ids:
            state, reason = STATE_AVAILABLE, "family has usable registered candidates"
        elif require_motion_target and all(
            entry["motion_state"] == STATE_NOT_APPLICABLE for entry in entries
        ):
            state = STATE_NOT_APPLICABLE
            reason = "no member of this family is a self-locomotion target by definition"
        else:
            state = STATE_EVIDENCE_MISSING
            reason = "registered members exist but none currently resolves a usable binding"
        slots.append(
            {
                "family": family,
                "state": state,
                "reason": reason,
                "registered_count": len(entries),
                "eligible_count": len(eligible_ids),
                "eligible_asset_ids": eligible_ids,
                "assets": entries,
            }
        )
    eligible_slots = [slot for slot in slots if slot["state"] == STATE_AVAILABLE]
    if len(eligible_slots) == 2:
        state = STATE_AVAILABLE
        reason = "both slots resolve registered candidates with real metadata"
    elif eligible_slots:
        state = STATE_EVIDENCE_MISSING
        reason = (
            "one slot resolves candidates; the other is reported separately and this "
            "combination is not excluded as a class"
        )
    else:
        state = STATE_EVIDENCE_MISSING
        reason = "neither slot currently resolves a usable candidate"
    return {
        "combination": key,
        "families": list(combination),
        "state": state,
        "reason": reason,
        "require_motion_target": bool(require_motion_target),
        "slots": slots,
        "usable_motion_target_families": [
            slot["family"] for slot in slots
            if any(
                entry["motion_state"] == STATE_AVAILABLE and entry["eligible"]
                for entry in slot["assets"]
            )
        ],
    }


def capability_report(
    registry: Mapping[str, Any], pool: Any = None, config: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Resolve every registered asset and every combination, denominator preserved.

    All registered assets stay in the denominator.  An asset without a usable placement,
    action or sound is reported with its exact gap state rather than dropped or presented
    as implemented.
    """
    resolved = normalize_sound_class_config(config)
    index = source_asset_runtime_index(registry)
    sounds = _pool_sounds(pool) if pool is not None else []
    evaluations = _asset_evaluations(index, sounds, resolved)
    assets = []
    for asset_id in sorted(index):
        entry = deepcopy(evaluations[asset_id]["capabilities"])
        if pool is not None:
            entry["sound_binding"] = evaluations[asset_id]["sound_binding"]
            entry["capability_states"]["sound_binding"] = entry["sound_binding"]["state"]
        assets.append(entry)
    combinations = (
        [
            combination_candidates(
                registry, pool, combination, resolved, evaluations=evaluations
            )
            for combination in entity_combinations()
        ]
        if pool is not None
        else []
    )
    motion_combinations = (
        [
            combination_candidates(
                registry, pool, combination, resolved,
                require_motion_target=True, evaluations=evaluations,
            )
            for combination in entity_combinations()
        ]
        if pool is not None
        else []
    )
    dimensions = list(CAPABILITY_DIMENSIONS) + (["sound_binding"] if pool else [])
    summary = {}
    for dimension in dimensions:
        summary[dimension] = dict(
            Counter(entry["capability_states"].get(dimension) for entry in assets)
        )
    return {
        "schema": SOURCE_CAPABILITY_SCHEMA,
        "registry_id": registry.get("registry_id"),
        "registry_revision": registry.get("revision"),
        "registered_asset_count": len(assets),
        "family_counts": dict(Counter(entry["family"] for entry in assets)),
        "assets": assets,
        "capability_state_counts": summary,
        "combinations": combinations,
        "motion_target_combinations": motion_combinations,
        "music_suite": music_suite_capability(registry, pool, resolved) if pool else None,
        "gap_assets": [
            {
                "asset_id": entry["asset_id"],
                "family": entry["family"],
                "gaps": {
                    name: state
                    for name, state in entry["capability_states"].items()
                    if state in GAP_STATES and name != "locomotion"
                },
            }
            for entry in assets
            if any(
                state in GAP_STATES and name != "locomotion"
                for name, state in entry["capability_states"].items()
            )
        ],
        "claim_boundary": (
            "registry and pool metadata resolution only; no native execution, human "
            "audibility review or formal admission is claimed here"
        ),
    }


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _library_config_from_args(args: Any) -> dict[str, Any]:
    """Merge a declared library config file with any --sound-library-root flags."""
    config: dict[str, Any] = {}
    if args.library_config:
        loaded = _load_json(args.library_config)
        if isinstance(loaded, Mapping):
            config.update(loaded)
    roots = list(config.get("sound_library_roots") or [])
    for index, raw in enumerate(args.sound_library_root or []):
        parts = str(raw).split(":")
        entry: dict[str, Any] = {"root": parts[0]}
        if len(parts) > 1 and parts[1]:
            entry["priority"] = int(parts[1])
        else:
            entry["priority"] = 100 - index
        if len(parts) > 2 and parts[2]:
            entry["role"] = parts[2]
        roots.append(entry)
    if roots:
        config["sound_library_roots"] = roots
    return config


def _run_sound_library_mode(args: Any) -> None:
    """Report the library overlay, same-source identity and candidate accounting."""
    config = _library_config_from_args(args)
    overlay = load_sound_library_overlay(config)
    print(f"library clips: {overlay['clip_count']}")
    for root, summary in overlay["root_summary"].items():
        print(f"  {summary['role']:<14} priority={summary['priority']:<4} "
              f"{summary['status']:<8} clips={summary['clip_count']}  {root}")
    print(f"  shadowed by a higher-priority root: {len(overlay['shadowed'])}")
    admission = Counter(
        clip["admission"]["state"] for clip in overlay["clips"].values())
    print(f"  admission: {dict(admission)}")

    registered = {"sound_asset_ids_by_relative": {}}
    if args.event_manifest:
        registered = registered_source_index(_load_json(args.event_manifest))
        print(f"  registered library relatives: "
              f"{registered['registered_relative_count']}")
    identity = resolve_library_identity(overlay, registered)
    print(f"  identity: {dict(identity['counts'])}")
    print(f"  registered sound assets reused rather than re-minted: "
          f"{identity['registered_sound_asset_ids_reused']}")

    selections = None
    if args.segment_selections:
        selections = _load_json(args.segment_selections)
        if isinstance(selections, Mapping) and "selections" in selections:
            selections = selections["selections"]
    budget = segment_budget(
        episode_s=args.episode_s,
        reserve_tail_s=args.reserve_tail_s,
        earliest_start_s=args.earliest_start_s,
    )
    report = library_candidate_report(
        overlay, identity, budget=budget, config=config,
        selections=selections if isinstance(selections, Mapping) else None,
    )
    print(f"  one dry program has {budget['max_single_program_span_s']:.3f} s "
          f"(episode {budget['episode_s']:.3f} s, reserve "
          f"{budget['reserve_tail_s']:.3f} s, start {budget['earliest_start_s']:.3f} s)")
    for name, value in report["counts"].items():
        print(f"    {name:<28} {value}")
    if report["blocked_by_state"]:
        print(f"    blocked_by_state: {report['blocked_by_state']}")
    if args.library_report:
        Path(args.library_report).write_text(
            json.dumps({"overlay_root_summary": overlay["root_summary"],
                        "shadowed_count": len(overlay["shadowed"]),
                        "identity": identity,
                        "candidates": report},
                       indent=1, sort_keys=True, ensure_ascii=False),
            encoding="utf-8")
        print(f"  library report: {args.library_report}")


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve and print the capability report for a registry and optional sound pool."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-registry")
    parser.add_argument("--sound-pool")
    parser.add_argument("--sound-class-config",
                        help="JSON file holding the sound_sources mapping block")
    parser.add_argument("--config-key", default="sound_sources")
    parser.add_argument("--output", help="write the full JSON report here")
    parser.add_argument("--sound-library-root", action="append", default=[],
                        metavar="ROOT[:PRIORITY[:ROLE]]",
                        help="declare a library root; repeat for an overlay, highest "
                             "priority wins for a shared relative path")
    parser.add_argument("--library-config",
                        help="JSON file declaring sound_library_roots and "
                             "segment_activity_requirements")
    parser.add_argument("--event-manifest",
                        help="event manifest whose clips carry the registered lineage")
    parser.add_argument("--segment-selections",
                        help="JSON file of segment selections returned by P25")
    parser.add_argument("--episode-s", type=float, default=10.0)
    parser.add_argument("--reserve-tail-s", type=float, default=3.0)
    parser.add_argument("--earliest-start-s", type=float, default=0.0)
    parser.add_argument("--library-report",
                        help="write the sound library candidate report here")
    args = parser.parse_args(argv)

    if not args.source_registry and not (args.sound_library_root or args.library_config):
        parser.error("declare --source-registry, --sound-library-root or --library-config")

    if args.sound_library_root or args.library_config:
        _run_sound_library_mode(args)
        if not args.source_registry:
            return 0

    registry = load_source_asset_runtime_registry(args.source_registry)
    pool = _load_json(args.sound_pool) if args.sound_pool else None
    config = None
    if args.sound_class_config:
        raw = _load_json(args.sound_class_config)
        config = raw.get(args.config_key, raw) if isinstance(raw, Mapping) else raw
    report = capability_report(registry, pool, config)

    print(f"registered assets: {report['registered_asset_count']}")
    print(f"families: {report['family_counts']}")
    for dimension, counts in report["capability_state_counts"].items():
        print(f"  {dimension}: {counts}")
    for combination in report["combinations"]:
        slots = " | ".join(
            f"{slot['family']}={slot['eligible_count']}/{slot['registered_count']}"
            for slot in combination["slots"]
        )
        print(f"  {combination['combination']:16s} {combination['state']:32s} {slots}")
    if report["gap_assets"]:
        print(f"assets with a capability gap: {len(report['gap_assets'])}")
        for entry in report["gap_assets"]:
            print(f"    {entry['asset_id']}: {entry['gaps']}")
    if args.output:
        Path(args.output).write_text(
            json.dumps(report, indent=1, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"report: {args.output}")
    return 0



# ------------------------------------------------------- sound library overlay

SOUND_LIBRARY_OVERLAY_SCHEMA = "avengine_sound_library_overlay_v1"
# P25 (``avengine.dataset.sound_segments``) owns activity measurement, the contiguous
# choice, the fresh PCM and the read-back.  These are its field names, not a parallel
# scheme: an authorized crop is marked by ``selection_authorized`` plus a named
# ``crop_authorization``, and its bounds are ``source_crop_*`` in the original
# recording's coordinates.
SEGMENT_SELECTION_REQUEST_SCHEMA = "avengine_sound_segment_selection_request_v1"
SEGMENT_AUTHORIZATION_FIELD = "selection_authorized"
SEGMENT_AUTHORIZATION_REFERENCE_FIELD = "crop_authorization"
SEGMENT_BOUND_FIELDS = ("source_crop_start_sample", "source_crop_end_sample_exclusive")

CLIP_AUDIO_NAME = "clip.wav"
CLIP_SIDECAR_NAME = "clip.json"
CLIP_QC_SIDECAR_NAME = "clip.qc.json"

# One library clip is identified by its root-relative path.  The same recording is
# copied byte-for-byte into successive library roots, and the event library resamples
# it into a third root, so the absolute path and the content hash both differ per root
# while the relative path stays stable.  Identity therefore keys on the relative path.
IDENTITY_SAME_SOURCE = "same_source_as_registered"
IDENTITY_NEW_CANDIDATE = "new_candidate"
IDENTITY_ORPHANED_REGISTRATION = "registered_source_absent_from_library"
# A later library root may carry different audio at a path the registered lineage
# already uses.  That is a different recording, so it must not inherit the existing
# sound asset IDs just because the path matches.
IDENTITY_SAME_PATH_DIFFERENT_SOURCE = "same_relative_path_different_source"

ADMISSION_CANDIDATE = "candidate"
ADMISSION_BLOCKED_MACHINE_FAIL = "blocked_machine_qc_fail"
ADMISSION_BLOCKED_NO_DECLARED_CLASS = "blocked_no_declared_event_class"
ADMISSION_BLOCKED_MISSING_SIDECAR = "blocked_missing_or_unreadable_sidecar"
ADMISSION_STATES = (
    ADMISSION_CANDIDATE,
    ADMISSION_BLOCKED_MACHINE_FAIL,
    ADMISSION_BLOCKED_NO_DECLARED_CLASS,
    ADMISSION_BLOCKED_MISSING_SIDECAR,
)


def normalize_library_roots(config: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Read the declared library roots, highest priority first.

    Roots come from configuration rather than a constant, so a later library drop is a
    config edit.  A higher ``priority`` wins for a relative path present in more than
    one root; ties keep declaration order.
    """
    config = config or {}
    declared = config.get("sound_library_roots")
    if declared is None:
        return []
    if isinstance(declared, (str, bytes)) or not isinstance(declared, Sequence):
        raise SourceCapabilityError("sound_library_roots must be a list")
    roots: list[dict[str, Any]] = []
    for index, entry in enumerate(declared):
        if isinstance(entry, (str, bytes)):
            entry = {"root": str(entry)}
        if not isinstance(entry, Mapping):
            raise SourceCapabilityError("each sound library root must be a path or object")
        root = entry.get("root")
        if not root:
            raise SourceCapabilityError("each sound library root must declare 'root'")
        roots.append({
            "root": str(root),
            "priority": int(entry.get("priority", 0)),
            "role": str(entry.get("role") or "unspecified"),
            "declaration_index": index,
        })
    roots.sort(key=lambda item: (-item["priority"], item["declaration_index"]))
    return roots


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def clip_admission(record: Mapping[str, Any]) -> dict[str, Any]:
    """Decide whether one library clip may enter the candidate pool.

    Machine QC and human listening are separate records and are both carried forward.
    A machine ``fail`` blocks the clip even when the library stamps a blanket human
    ``pass`` on it: the new library marks every inherited clip ``pass``, including the
    ones QC failed for severe clipping or DC offset, so trusting the human field alone
    would silently readmit known-broken audio.
    """
    qc = record.get("qc") or {}
    human = record.get("human_review") or {}
    verdict = qc.get("verdict")
    human_status = human.get("status")
    findings = [
        finding for finding in (qc.get("findings") or [])
        if isinstance(finding, Mapping)
    ]
    blocking = [
        finding for finding in findings
        if str(finding.get("severity")) == "fail"
    ]
    basis = {
        "machine_qc_verdict": verdict,
        "machine_qc_blocking_findings": [
            {"name": finding.get("name"), "severity": finding.get("severity")}
            for finding in blocking
        ],
        "machine_qc_finding_count": len(findings),
        "human_review_status": human_status,
        "human_review_author": human.get("author") or (
            (human.get("event_classes") or {}).get(
                next(iter(human.get("event_classes") or {}), ""), {}
            ).get("author") if isinstance(human.get("event_classes"), Mapping) else None
        ),
        "precedence": "machine_fail_is_never_overridden_by_a_human_pass",
        "human_scope": "original_recording_only",
    }
    if record.get("sidecar_status") != "read":
        return {"state": ADMISSION_BLOCKED_MISSING_SIDECAR,
                "reason": "clip.json or clip.qc.json is missing or unreadable",
                "basis": basis}
    if verdict == "fail" or blocking:
        return {"state": ADMISSION_BLOCKED_MACHINE_FAIL,
                "reason": "machine QC reports a blocking defect",
                "basis": basis}
    if not record.get("declared_event_classes"):
        return {"state": ADMISSION_BLOCKED_NO_DECLARED_CLASS,
                "reason": "clip.json declares no event_classes",
                "basis": basis}
    return {"state": ADMISSION_CANDIDATE,
            "reason": "machine QC carries no blocking defect and a class is declared",
            "basis": basis}


def load_sound_library_overlay(
    config: Mapping[str, Any] | None = None, *, roots: Sequence[Mapping[str, Any]] | None = None
) -> dict[str, Any]:
    """Scan the declared library roots into one overlay keyed by relative path.

    Only the small JSON sidecars are read; no PCM is opened and nothing is written, so
    an inventory pass never touches the recordings.  Duration, rate and channel count
    come from the QC sidecar the library already carries.
    """
    resolved_roots = list(roots) if roots is not None else normalize_library_roots(config)
    if not resolved_roots:
        raise SourceCapabilityError("no sound library root is declared")
    clips: dict[str, dict[str, Any]] = {}
    shadowed: list[dict[str, Any]] = []
    per_root: dict[str, dict[str, Any]] = {}
    for entry in resolved_roots:
        root = Path(entry["root"])
        found = 0
        if not root.is_dir():
            per_root[str(root)] = {"role": entry["role"], "priority": entry["priority"],
                                   "status": "missing", "clip_count": 0}
            continue
        for audio in sorted(root.rglob(CLIP_AUDIO_NAME)):
            relative = audio.relative_to(root).as_posix()
            found += 1
            if relative in clips:
                shadowed.append({
                    "relative": relative, "shadowed_root": str(root),
                    "winning_root": clips[relative]["root"],
                    "shadowed_byte_size": audio.stat().st_size,
                    "winning_byte_size": clips[relative].get("byte_size"),
                })
                continue
            sidecar = audio.with_name(CLIP_SIDECAR_NAME)
            qc_sidecar = audio.with_name(CLIP_QC_SIDECAR_NAME)
            record: dict[str, Any] = {
                "relative": relative,
                "root": str(root),
                "root_role": entry["role"],
                "root_priority": entry["priority"],
                "audio_path": str(audio),
                "sidecar_status": "read",
                "byte_size": audio.stat().st_size,
            }
            try:
                declared = _read_json(sidecar)
                qc = _read_json(qc_sidecar)
            except (OSError, ValueError) as error:
                record.update(sidecar_status="unreadable", sidecar_error=str(error),
                             declared_event_classes=[], qc={}, human_review={})
            else:
                measured = qc.get("measured") or {}
                record.update(
                    declared_event_classes=[
                        str(value) for value in (declared.get("event_classes") or [])
                    ],
                    source_note=declared.get("source"),
                    license=declared.get("license"),
                    dry=declared.get("dry"),
                    human_review=deepcopy(declared.get("human_review") or {}),
                    qc={"verdict": qc.get("verdict"),
                        "findings": deepcopy(qc.get("findings") or []),
                        "schema": qc.get("schema")},
                    duration_s=measured.get("duration_s"),
                    sample_rate_hz=qc.get("sample_rate_hz"),
                    channel_count=qc.get("channel_count"),
                    active_frame_ratio=measured.get("active_frame_ratio"),
                    continuous=measured.get("continuous"),
                    peak=measured.get("peak"),
                    rms_dbfs=measured.get("rms_dbfs"),
                    noise_floor_dbfs=measured.get("noise_floor_dbfs"),
                    decay_to_minus20db_s=measured.get("decay_to_minus20db_s"),
                )
            record["admission"] = clip_admission(record)
            clips[relative] = record
        per_root[str(root)] = {"role": entry["role"], "priority": entry["priority"],
                               "status": "read", "clip_count": found}
    return {
        "schema": SOUND_LIBRARY_OVERLAY_SCHEMA,
        "roots": resolved_roots,
        "root_summary": per_root,
        "clips": clips,
        "shadowed": shadowed,
        "clip_count": len(clips),
        "identity_key": "root_relative_path",
        "claim_boundary": (
            "sidecar inventory only; no PCM was read, nothing was written, and no clip "
            "is admitted to a pool by being listed here"
        ),
    }


def registered_source_index(event_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Map library relative path -> already registered sound assets.

    The event manifest records each registered event's originating library relative
    path, so this is the existing lineage rather than a new identity scheme.
    """
    clips = event_manifest.get("clips")
    if not isinstance(clips, Sequence):
        raise SourceCapabilityError("event manifest carries no clips list")
    by_relative: dict[str, list[str]] = defaultdict(list)
    classes: dict[str, set[str]] = defaultdict(set)
    for clip in clips:
        if not isinstance(clip, Mapping):
            continue
        relative = clip.get("source")
        if not relative:
            continue
        asset_id = clip.get("sound_asset_id")
        if asset_id:
            by_relative[str(relative)].append(str(asset_id))
        if clip.get("event_class"):
            classes[str(relative)].add(str(clip["event_class"]))
    return {
        "library_root": event_manifest.get("library_root"),
        "sound_asset_ids_by_relative": {k: sorted(v) for k, v in by_relative.items()},
        "event_classes_by_relative": {k: sorted(v) for k, v in classes.items()},
        "registered_relative_count": len(by_relative),
    }


def resolve_library_identity(
    overlay: Mapping[str, Any],
    registered: Mapping[str, Any],
    *,
    lineage_root: str | None = None,
    alias_roots: Sequence[str] = (),
) -> dict[str, Any]:
    """Split the overlay into already-registered sources and genuinely new candidates.

    A clip whose relative path already backs registered sound assets keeps that
    identity, so copying the same recording under a new root does not mint a second
    sound identity or count the recording twice.

    That inheritance is only safe while the path really names the same audio.
    ``lineage_root`` declares which root the registered lineage was built from;
    ``alias_roots`` declares further roots asserted to carry the same recordings.  A
    clip winning from any other root is reported as
    ``same_relative_path_different_source`` and inherits nothing.  If the overlay saw
    both copies and their byte sizes disagree, the assertion is contradicted and the
    clip is separated even when its root was declared an alias.

    With no ``lineage_root`` declared, inheritance is by path alone; that is recorded
    in ``lineage_root_declared`` so a caller can see which basis was used.
    """
    by_relative = registered.get("sound_asset_ids_by_relative") or {}
    clips = overlay.get("clips") or {}
    trusted = {str(lineage_root)} if lineage_root else set()
    trusted.update(str(root) for root in alias_roots)
    size_conflicts = {
        str(row["relative"]): row
        for row in (overlay.get("shadowed") or [])
        if row.get("shadowed_byte_size") is not None
        and row.get("winning_byte_size") is not None
        and row["shadowed_byte_size"] != row["winning_byte_size"]
    }

    same_source: list[str] = []
    new_candidates: list[str] = []
    diverged: list[dict[str, Any]] = []
    for relative in sorted(clips):
        if relative not in by_relative:
            new_candidates.append(relative)
            continue
        clip_root = str(clips[relative].get("root"))
        if trusted and clip_root not in trusted:
            diverged.append({
                "relative": relative, "root": clip_root,
                "reason": "winning_root_is_not_the_declared_lineage_root_or_an_alias",
                "registered_sound_asset_ids": list(by_relative[relative]),
            })
            continue
        if relative in size_conflicts:
            diverged.append({
                "relative": relative, "root": clip_root,
                "reason": "declared_same_path_copies_differ_in_byte_size",
                "byte_sizes": [size_conflicts[relative]["winning_byte_size"],
                               size_conflicts[relative]["shadowed_byte_size"]],
                "registered_sound_asset_ids": list(by_relative[relative]),
            })
            continue
        same_source.append(relative)

    orphans = sorted(set(by_relative) - set(clips))
    return {
        "identity_key": "root_relative_path",
        "lineage_root": lineage_root,
        "lineage_root_declared": bool(lineage_root),
        "alias_roots": [str(root) for root in alias_roots],
        "same_source_as_registered": same_source,
        "new_candidate": new_candidates,
        "same_relative_path_different_source": diverged,
        "registered_source_absent_from_library": orphans,
        "counts": {
            IDENTITY_SAME_SOURCE: len(same_source),
            IDENTITY_NEW_CANDIDATE: len(new_candidates),
            IDENTITY_SAME_PATH_DIFFERENT_SOURCE: len(diverged),
            IDENTITY_ORPHANED_REGISTRATION: len(orphans),
        },
        "registered_sound_asset_ids_reused": sum(
            len(by_relative[relative]) for relative in same_source
        ),
        "registered_sound_asset_ids_withheld": sum(
            len(row["registered_sound_asset_ids"]) for row in diverged
        ),
        "claim_boundary": (
            "identity is the library relative path, and it is only inherited from the "
            "declared lineage root or a declared alias; a path match alone never "
            "transfers an existing sound asset ID to different audio"
        ),
    }


# --------------------------------------------- segment selection contract (P25)

# The episode is ten seconds with a reserved wet tail.  A single dry program that must
# fit on its own therefore ends before the reserve begins, but that arithmetic is a
# derived budget for one such case, never a global clip cap: sequential, overlapping
# and repeated events subdivide the same span, and each task family adds its own
# windows.  Nothing here rewrites the episode length or shortens the reserve.
def segment_budget(
    *,
    episode_s: float,
    reserve_tail_s: float,
    earliest_start_s: float = 0.0,
) -> dict[str, Any]:
    """Derive how much room one dry program has, from explicit parameters only."""
    for name, value in (("episode_s", episode_s), ("reserve_tail_s", reserve_tail_s),
                        ("earliest_start_s", earliest_start_s)):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise SourceCapabilityError(f"{name} must be a number")
        if value < 0:
            raise SourceCapabilityError(f"{name} must not be negative")
    latest_end = float(episode_s) - float(reserve_tail_s)
    span = latest_end - float(earliest_start_s)
    if span <= 0:
        raise SourceCapabilityError(
            "the reserved tail and earliest start leave no room for a dry program")
    return {
        "episode_s": float(episode_s),
        "reserve_tail_s": float(reserve_tail_s),
        "earliest_start_s": float(earliest_start_s),
        "latest_program_end_s": latest_end,
        "max_single_program_span_s": span,
        "claim_boundary": (
            "budget for one dry program that must fit alone; several events share this "
            "span, so this is not a per-clip maximum and not a new global cap"
        ),
    }


def segment_requirements(
    activity_family: str, config: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Read the declared activity requirements for one activity family.

    Coverage and internal-silence limits are configuration, not constants invented
    here.  When a family has no declared requirement the result says so and the segment
    cannot be verified, because a non-zero waveform is not evidence of sounding.
    """
    config = config or {}
    declared = config.get("segment_activity_requirements") or {}
    if not isinstance(declared, Mapping):
        raise SourceCapabilityError("segment_activity_requirements must be an object")
    entry = declared.get(str(activity_family))
    if not isinstance(entry, Mapping):
        return {
            "activity_family": str(activity_family),
            "state": STATE_EVIDENCE_MISSING,
            "reason": "no activity requirement is configured for this family",
            "minimum_activity_coverage": None,
            "maximum_internal_silence_s": None,
            "minimum_segment_s": None,
        }
    return {
        "activity_family": str(activity_family),
        "state": STATE_AVAILABLE,
        "reason": "configured activity requirement",
        "minimum_activity_coverage": entry.get("minimum_activity_coverage"),
        "maximum_internal_silence_s": entry.get("maximum_internal_silence_s"),
        "minimum_segment_s": entry.get("minimum_segment_s"),
    }


def segment_selection_request(
    clip: Mapping[str, Any],
    *,
    sound_class: str,
    budget: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the request P25 consumes for one library clip.

    P07 states which recording, which class, how much room the schedule has and which
    activity evidence must come back.  P25 owns the detection, the contiguous choice
    and the fresh PCM; the original recording stays read-only.
    """
    from avengine.assets.sound_prepare import activity_profile_for_class

    profile = activity_profile_for_class(sound_class)
    requirements = segment_requirements(profile["activity_family"], config)
    return {
        "schema": SEGMENT_SELECTION_REQUEST_SCHEMA,
        "relative": clip.get("relative"),
        "root": clip.get("root"),
        "audio_path": clip.get("audio_path"),
        "sound_class": str(sound_class),
        "source_duration_s": clip.get("duration_s"),
        "source_sample_rate_hz": clip.get("sample_rate_hz"),
        "activity_profile": profile,
        "requirements": requirements,
        "budget": dict(budget),
        "authorization": {
            "allows": ["contiguous_segment_selection", "gain", "zero_pad"],
            "forbids": [
                "time_stretch",
                "wet_tail_truncation",
                "per_sample_level_matching",
                "discontiguous_splicing",
            ],
            "original_pcm": "read_only",
        },
        "claim_boundary": (
            "a request, not a result; segment feasibility is only known after P25 "
            "measures actual sounding activity"
        ),
    }


def _segment_relative(selection: Mapping[str, Any]) -> str | None:
    """The recording a segment came from, however the producer spelled it."""
    for field in ("relative_path", "relative", "source_origin", "source_path"):
        value = selection.get(field)
        if value:
            return str(value)
    return None


# ``sound_segments.pool_row`` publishes the measurement flat; the fuller segment
# record keeps it under ``planned_activity``.  Both are the same numbers, so both are
# read, and which one was used is reported.
_ACTIVITY_CONTAINERS = ("planned_activity", "segment_activity")


def _activity_value(selection: Mapping[str, Any], field: str) -> tuple[Any, str | None]:
    """The measured value and where it came from, or (None, None) if absent."""
    if field in selection:
        return selection.get(field), field
    for container in _ACTIVITY_CONTAINERS:
        nested = selection.get(container)
        if isinstance(nested, Mapping) and field in nested:
            return nested.get(field), f"{container}.{field}"
    return None, None


def _finite_number(value: Any) -> float | None:
    """A real finite number, or None.  Bools and numeric strings are not numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _exact_int(value: Any) -> int | None:
    """An exact non-negative sample index.  A float sample index is not exact."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def _source_header_path(selection: Mapping[str, Any]) -> Any:
    """Where the original recording is, however the producer named the field.

    ``sound_segments.pool_row`` publishes it as ``source_origin`` while the fuller
    segment record keeps ``source_path``, so both are consulted.
    """
    for field in ("source_path", "source_origin"):
        value = selection.get(field)
        if value:
            return value
    aliases = selection.get("source_origin_aliases")
    if isinstance(aliases, (list, tuple)) and aliases:
        return aliases[0]
    return None


def _wav_frame_count(path: Any) -> int | None:
    """Exact frame count from the file header, never derived from a rounded duration."""
    if not path:
        return None
    try:
        with wave.open(str(path), "rb") as handle:
            return int(handle.getnframes())
    except (OSError, wave.Error, EOFError, ValueError):
        return None


def _resolve_length(
    selection: Mapping[str, Any],
    caller: Mapping[str, Any] | None,
    fields: Sequence[str],
    *,
    header_path: Any = None,
) -> tuple[int | None, str]:
    """Find an exact sample count: the row, then the caller, then the file header.

    A rounded QC duration is never turned into a sample count; when nothing exact is
    available the caller is told the bound could not be checked.
    """
    for name in fields:
        exact = _exact_int(selection.get(name))
        if exact is not None:
            return exact, f"selection.{name}"
    if caller:
        for name in fields:
            exact = _exact_int(caller.get(name))
            if exact is not None:
                return exact, f"caller.{name}"
    if header_path is not None:
        frames = _wav_frame_count(header_path)
        if frames is not None:
            return frames, "wav_header"
    return None, "unresolved"


def _validate_intervals(
    value: Any, *, bound: int | None, bound_source: str, label: str
) -> list[str]:
    """Ordered, disjoint, non-empty integer sample pairs inside their own coordinates."""
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        return [f"{label}_is_not_a_list_of_sample_pairs"]
    if not value:
        return [f"{label}_is_empty"]
    previous_end: int | None = None
    for pair in value:
        if (isinstance(pair, (str, bytes)) or not isinstance(pair, (list, tuple))
                or len(pair) != 2):
            return [f"{label}_has_a_malformed_pair"]
        first, second = _exact_int(pair[0]), _exact_int(pair[1])
        if first is None or second is None:
            return [f"{label}_has_a_non_integer_sample"]
        if second <= first:
            return [f"{label}_has_an_empty_or_reversed_pair"]
        if previous_end is not None and first < previous_end:
            return [f"{label}_pairs_are_not_ordered_and_disjoint"]
        if bound is not None and second > bound:
            return [f"{label}_exceeds_{bound_source}"]
        previous_end = second
    return []


# The same field name means different coordinates in the two shapes P25 publishes, so
# the shape is decided first and the names are read accordingly.
#
#   segment record: segment_activity_intervals_samples -> delivered segment samples
#                   source_activity_intervals_samples  -> whole recording at the
#                                                         analysis rate
#   pool_row:       source_activity_intervals_samples  -> delivered segment samples
#                   origin_activity_intervals_samples  -> whole recording at the
#                                                         analysis rate
#
# ``pool_row`` copies the record's ``activity_interval_coordinates`` map verbatim even
# though it renamed the fields, so that map is not trusted to identify them.
_RECORD_SEGMENT_FIELD = "segment_activity_intervals_samples"
_RECORD_ANALYSIS_FIELD = "source_activity_intervals_samples"
_POOL_ROW_SEGMENT_FIELD = "source_activity_intervals_samples"
_POOL_ROW_ANALYSIS_FIELD = "origin_activity_intervals_samples"


def _interval_fields(selection: Mapping[str, Any]) -> tuple[str | None, str | None, str]:
    """Return (segment field, analysis field, shape) for this row."""
    if _RECORD_SEGMENT_FIELD in selection:
        return _RECORD_SEGMENT_FIELD, _RECORD_ANALYSIS_FIELD, "segment_record"
    if _POOL_ROW_SEGMENT_FIELD in selection:
        return _POOL_ROW_SEGMENT_FIELD, _POOL_ROW_ANALYSIS_FIELD, "pool_row"
    return None, _POOL_ROW_ANALYSIS_FIELD, "unknown"


def verify_segment_selection(
    selection: Mapping[str, Any],
    *,
    clip: Mapping[str, Any] | None = None,
    requirements: Mapping[str, Any] | None = None,
    expected_processing: Mapping[str, Any] | None = None,
    read_source_header: bool = True,
    require_bounds: bool = True,
) -> dict[str, Any]:
    """Check that a returned segment is a bounded, measured, authorized crop.

    This is what separates an authorized selection from an unexplained truncation.  The
    row must say it was authorized, name the authorization, carry sample bounds, and
    carry an activity measurement whose numbers are real finite values of the right
    type.  A missing or wrongly typed field is a refusal, not a skipped check.

    Three coordinate systems are kept apart and each is bounded by its own length:
    the original recording (``source_crop_*`` against the source sample count), the
    resampled whole recording P25 measures on (``origin_activity_intervals_samples``
    against the analysis sample count) and the delivered segment
    (``segment_activity_intervals_samples``, or ``pool_row``'s renamed
    ``source_activity_intervals_samples``, against the segment sample count).  A 16 kHz
    interval is never compared with a 44.1 kHz crop position.

    Structural verification is not a PCM read-back and not a listening test; P25 owns
    both of those and reports them separately.
    """
    if not isinstance(selection, Mapping):
        raise SourceCapabilityError("segment selection must be an object")
    problems: list[str] = []
    unverified: list[str] = []

    if selection.get(SEGMENT_AUTHORIZATION_FIELD) is not True:
        problems.append("selection_is_not_marked_authorized")
    if not selection.get(SEGMENT_AUTHORIZATION_REFERENCE_FIELD):
        problems.append("missing_crop_authorization")
    if selection.get("truncated"):
        problems.append("row_claims_both_authorized_selection_and_truncation")
    relative = _segment_relative(selection)
    if relative is None:
        problems.append("missing_source_identity")

    # --- original recording coordinates -----------------------------------------
    start_field, end_field = SEGMENT_BOUND_FIELDS
    crop_start = _exact_int(selection.get(start_field))
    crop_end = _exact_int(selection.get(end_field))
    if crop_start is None:
        problems.append(f"invalid_{start_field}")
    if crop_end is None:
        problems.append(f"invalid_{end_field}")
    if crop_start is not None and crop_end is not None and crop_end <= crop_start:
        problems.append("empty_sample_range")
    source_length, source_length_from = _resolve_length(
        selection, clip, ("source_sample_count", "source_frames"),
        header_path=(_source_header_path(selection) if read_source_header else None))
    if source_length is None:
        unverified.append("source_sample_count")
        if require_bounds:
            problems.append("source_sample_count_is_not_verifiable")
    elif crop_end is not None and crop_end > source_length:
        problems.append("source_crop_end_exceeds_source_sample_count")

    # --- delivered segment coordinates ------------------------------------------
    segment_length, segment_length_from = _resolve_length(
        selection, clip, ("prepared_sample_count", "sample_count"))
    segment_field, analysis_field, shape = _interval_fields(selection)
    if segment_field is None:
        problems.append("missing_activity_intervals")
    else:
        if segment_length is None:
            unverified.append("segment_sample_count")
        problems.extend(_validate_intervals(
            selection.get(segment_field), bound=segment_length,
            bound_source="segment_sample_count", label=segment_field))

    # --- resampled whole-recording coordinates ----------------------------------
    analysis_length, analysis_length_from = _resolve_length(
        selection, clip, ("analysis_sample_count",))
    if analysis_field is not None and analysis_field in selection:
        if analysis_length is None:
            unverified.append("analysis_sample_count")
        problems.extend(_validate_intervals(
            selection.get(analysis_field), bound=analysis_length,
            bound_source="analysis_sample_count", label=analysis_field))

    # --- measured activity, typed --------------------------------------------------
    raw_coverage, coverage_from = _activity_value(selection, "activity_coverage")
    coverage = _finite_number(raw_coverage)
    if coverage_from is not None and coverage is None:
        problems.append("activity_coverage_is_not_a_finite_number")
    elif coverage is not None and not 0.0 <= coverage <= 1.0:
        problems.append("activity_coverage_is_outside_zero_to_one")
    raw_silence, silence_from = _activity_value(selection, "max_internal_silence_s")
    silence = _finite_number(raw_silence)
    if silence_from is not None and silence is None:
        problems.append("max_internal_silence_s_is_not_a_finite_number")
    elif silence is not None and silence < 0.0:
        problems.append("max_internal_silence_s_is_negative")

    # --- configured conditions, never skipped for a missing or mistyped field ----
    if requirements is not None:
        if requirements.get("state") != STATE_AVAILABLE:
            problems.append("activity_requirement_is_not_configured")
        else:
            minimum = _finite_number(requirements.get("minimum_activity_coverage"))
            if minimum is not None:
                if coverage is None:
                    problems.append("activity_coverage_is_required_but_unusable")
                elif coverage < minimum:
                    problems.append("activity_coverage_below_configured_minimum")
            allowed = _finite_number(requirements.get("maximum_internal_silence_s"))
            if allowed is not None:
                if silence is None:
                    problems.append("max_internal_silence_s_is_required_but_unusable")
                elif silence > allowed:
                    problems.append("internal_silence_exceeds_configured_maximum")
            shortest = _finite_number(requirements.get("minimum_segment_s"))
            if shortest is not None:
                duration = _finite_number(selection.get("prepared_duration_s"))
                if duration is None:
                    duration = _finite_number(_activity_value(selection, "duration_s")[0])
                if duration is None and segment_length is not None:
                    rate = _finite_number(selection.get("sample_rate_hz")) or \
                        _finite_number(selection.get("target_rate_hz"))
                    if rate:
                        duration = segment_length / rate
                if duration is None:
                    problems.append("segment_duration_is_required_but_unusable")
                elif duration < shortest:
                    problems.append("segment_shorter_than_configured_minimum")

    # --- the crop must be the one that was asked for -----------------------------
    processing_mismatch: dict[str, Any] = {}
    if expected_processing:
        for name, wanted in expected_processing.items():
            actual = selection.get(name)
            wanted_number, actual_number = _finite_number(wanted), _finite_number(actual)
            if wanted_number is not None and actual_number is not None:
                if abs(wanted_number - actual_number) > 1e-9:
                    processing_mismatch[name] = {"requested": wanted, "recorded": actual}
            elif actual != wanted:
                processing_mismatch[name] = {"requested": wanted, "recorded": actual}
        if processing_mismatch:
            problems.append("segment_processing_parameters_do_not_match_the_request")

    if clip is not None:
        expected = _segment_relative(clip) or clip.get("relative")
        if expected and relative and str(relative) != str(expected):
            if not str(relative).endswith(str(expected)):
                problems.append("selection_does_not_match_the_requested_clip")

    return {
        "verified": not problems,
        "problems": problems,
        "relative": relative,
        "source_crop_start_sample": crop_start,
        "source_crop_end_sample_exclusive": crop_end,
        "crop_authorization": selection.get(SEGMENT_AUTHORIZATION_REFERENCE_FIELD),
        "activity_coverage": coverage,
        "activity_coverage_from": coverage_from,
        "max_internal_silence_s": silence,
        "max_internal_silence_s_from": silence_from,
        "coordinate_bounds": {
            "source_sample_count": source_length,
            "source_sample_count_from": source_length_from,
            "segment_sample_count": segment_length,
            "segment_sample_count_from": segment_length_from,
            "analysis_sample_count": analysis_length,
            "analysis_sample_count_from": analysis_length_from,
            "segment_interval_field": segment_field,
            "analysis_interval_field": analysis_field,
            "row_shape": shape,
        },
        "unverified_bounds": sorted(set(unverified)),
        "processing_mismatch": processing_mismatch,
        "claim_boundary": (
            "structural, typed and configured-threshold verification of a declared "
            "selection; P25 owns the activity measurement and the PCM read-back, and "
            "neither this nor that is a human audibility judgement"
        ),
    }


def library_segment_sources(
    overlay: Mapping[str, Any],
    identity: Mapping[str, Any] | None = None,
    *,
    include_registered: bool = True,
) -> list[dict[str, Any]]:
    """Hand P25 the admitted, de-duplicated recordings in the shape it consumes.

    Blocked clips never leave here, so a machine-failed recording is not offered for
    cropping.  ``include_registered=False`` keeps only recordings that do not already
    back a registered sound asset, which is the incremental pass.
    """
    same_source = set((identity or {}).get("same_source_as_registered") or [])
    sources: list[dict[str, Any]] = []
    for relative in sorted(overlay.get("clips") or {}):
        clip = overlay["clips"][relative]
        if (clip.get("admission") or {}).get("state") != ADMISSION_CANDIDATE:
            continue
        if not include_registered and relative in same_source:
            continue
        classes = clip.get("declared_event_classes") or []
        parts = relative.split("/")
        sources.append({
            "source_path": clip["audio_path"],
            "relative_path": relative,
            "source_asset_id": "/".join(parts[:2]) if len(parts) > 1 else relative,
            "sound_class": classes[0] if classes else parts[0],
            "directory_class": parts[0],
            "declared_event_classes": list(classes),
            "library_root": clip["root"],
            "source_human_review": deepcopy(clip.get("human_review") or {}),
            "machine_qc_verdict": (clip.get("qc") or {}).get("verdict"),
            "already_registered": relative in same_source,
        })
    return sources


def library_candidate_report(
    overlay: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    budget: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    selections: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Report inventory, crop candidates, schedulable clips and qualified clips apart.

    These four are different questions and are never collapsed:

    ``inventory`` every clip the declared roots carry, whatever its state;
    ``crop_candidate`` clips whose machine QC and declared class allow a segment to be
    attempted at all;
    ``schedulable_now`` clips that either already fit the derived budget whole, or are
    longer and therefore depend on a segment selection;
    ``qualified`` clips with a verified segment selection in hand.

    ``qualified`` stays zero until P25 returns measured selections; a clip counted as a
    crop candidate has not been shown to contain usable sounding audio.
    """
    clips = overlay.get("clips") or {}
    selections = selections or {}
    same_source = set(identity.get("same_source_as_registered") or [])
    span = float(budget["max_single_program_span_s"]) if budget else None

    rows: list[dict[str, Any]] = []
    for relative in sorted(clips):
        clip = clips[relative]
        admission = clip.get("admission") or {}
        classes = clip.get("declared_event_classes") or []
        duration = clip.get("duration_s")
        is_candidate = admission.get("state") == ADMISSION_CANDIDATE
        fits_whole = (
            span is not None and isinstance(duration, (int, float))
            and float(duration) <= span
        )
        needs_segment = (
            span is not None and isinstance(duration, (int, float))
            and float(duration) > span
        )
        selection = selections.get(relative)
        verified = False
        if selection is not None:
            requirements = None
            if classes:
                from avengine.assets.sound_prepare import activity_profile_for_class
                requirements = segment_requirements(
                    activity_profile_for_class(classes[0])["activity_family"], config)
            verified = bool(
                verify_segment_selection(
                    selection, clip=clip, requirements=requirements)["verified"])

        rows.append({
            "relative": relative,
            "declared_event_classes": classes,
            "duration_s": duration,
            "admission_state": admission.get("state"),
            "identity": (IDENTITY_SAME_SOURCE if relative in same_source
                         else IDENTITY_NEW_CANDIDATE),
            "crop_candidate": is_candidate,
            "fits_budget_whole": fits_whole,
            "needs_segment_selection": needs_segment,
            "qualified": bool(is_candidate and verified),
        })

    def count(predicate) -> int:
        return sum(1 for row in rows if predicate(row))

    return {
        "schema": SOUND_LIBRARY_OVERLAY_SCHEMA,
        "budget": dict(budget) if budget else None,
        "counts": {
            "inventory": len(rows),
            "crop_candidate": count(lambda r: r["crop_candidate"]),
            "blocked": count(lambda r: not r["crop_candidate"]),
            "fits_budget_whole": count(
                lambda r: r["crop_candidate"] and r["fits_budget_whole"]),
            "needs_segment_selection": count(
                lambda r: r["crop_candidate"] and r["needs_segment_selection"]),
            "schedulable_now": count(
                lambda r: r["crop_candidate"] and (r["fits_budget_whole"] or r["qualified"])),
            "qualified": count(lambda r: r["qualified"]),
            "same_source_as_registered": count(
                lambda r: r["identity"] == IDENTITY_SAME_SOURCE),
            "new_candidate": count(lambda r: r["identity"] == IDENTITY_NEW_CANDIDATE),
        },
        "blocked_by_state": dict(Counter(
            row["admission_state"] for row in rows if not row["crop_candidate"])),
        "needs_segment_by_class": dict(Counter(
            (row["declared_event_classes"] or ["(none)"])[0]
            for row in rows
            if row["crop_candidate"] and row["needs_segment_selection"])),
        "rows": rows,
        "segment_selection_owner": "P25",
        "claim_boundary": (
            "a crop candidate is not a usable sound; qualified counts only clips whose "
            "returned segment passed verification, and no count here is an audibility "
            "or admission certificate"
        ),
    }

__all__ = [
    "CAPABILITY_DIMENSIONS",
    "CAPABILITY_STATES",
    "GAP_STATES",
    "MUSIC_PLAYBACK_SOUND_CLASS",
    "SOURCE_CAPABILITY_SCHEMA",
    "SOURCE_FAMILIES",
    "STATE_AVAILABLE",
    "STATE_EVIDENCE_MISSING",
    "STATE_NOT_APPLICABLE",
    "STATE_NOT_IMPLEMENTED",
    "ADMISSION_CANDIDATE",
    "ADMISSION_STATES",
    "CLIP_QC_SIDECAR_NAME",
    "CLIP_SIDECAR_NAME",
    "IDENTITY_NEW_CANDIDATE",
    "IDENTITY_ORPHANED_REGISTRATION",
    "IDENTITY_SAME_SOURCE",
    "SEGMENT_SELECTION_REQUEST_SCHEMA",
    "SOUND_LIBRARY_OVERLAY_SCHEMA",
    "clip_admission",
    "load_sound_library_overlay",
    "normalize_library_roots",
    "registered_source_index",
    "resolve_library_identity",
    "library_candidate_report",
    "segment_budget",
    "segment_requirements",
    "segment_selection_request",
    "verify_segment_selection",
    "SEGMENT_AUTHORIZATION_FIELD",
    "SEGMENT_AUTHORIZATION_REFERENCE_FIELD",
    "SEGMENT_BOUND_FIELDS",
    "library_segment_sources",
    "IDENTITY_SAME_PATH_DIFFERENT_SOURCE",
    "SourceCapabilityError",
    "appearance_capability",
    "assert_motion_target",
    "assets_accepting_sound_class",
    "capability_report",
    "combination_candidates",
    "combination_key",
    "compatible_sounds",
    "declare_events",
    "declare_instances",
    "declared_sound_classes",
    "emission_capability",
    "entity_combinations",
    "instance_event_identity_report",
    "instancing_capability",
    "locomotion_capability",
    "main",
    "make_event_id",
    "make_instance_id",
    "music_suite_capability",
    "normalize_sound_class_config",
    "placement_capability",
    "plan_source_binding",
    "resolve_source_capabilities",
    "sound_class_asset_index",
    "sound_compatibility",
    "sound_identity_of",
    "sound_recognizability",
    "source_class_token",
    "source_family",
]


if __name__ == "__main__":
    raise SystemExit(main())
