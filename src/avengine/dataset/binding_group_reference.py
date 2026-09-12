"""CPU planning and native-finalizer adapters for QA-05 reference groups.

A reference group keeps three speaking sources physically and acoustically bound to
their own clips/endpoints.  A fourth registered source is silent and moves between
the two visual variants.  The public query names only the candidate scope and the
pixel selector; private selector proofs and plans retain the actor-level facts.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from avengine.dataset import binding_group_native as native

REFERENCE_TASK_FAMILY = native.RELATION_TASK_FAMILY
REFERENCE_QUERY_KIND = "two_nearest_to_leftmost"
REFERENCE_NAMED_QUERY_KIND = "two_nearest_to_named_reference"
REFERENCE_QUERY_KINDS = (REFERENCE_QUERY_KIND, REFERENCE_NAMED_QUERY_KIND)
REFERENCE_SPEAKING_ACTOR_IDS = ("source1", "source2", "source3")
REFERENCE_ACTOR_IDS = (*REFERENCE_SPEAKING_ACTOR_IDS, "source4")
REFERENCE_REFERENCE_ACTOR_ID = "source4"
_DEFAULT_REFERENCE_PAIRS = {
    "v0": ("source1", "source2"),
    "v1": ("source2", "source3"),
}
class ReferenceNativeError(native.BindingNativeError):
    """A QA-05 reference group cannot be planned from the supplied inputs."""


def _load(path: str | Path) -> dict[str, Any]:
    try:
        return native._load(Path(path).expanduser().resolve())
    except native.BindingNativeError as exc:
        raise ReferenceNativeError(str(exc)) from exc


def _write(path: str | Path, value: Any) -> Path:
    try:
        return native._write(Path(path).expanduser().resolve(), value)
    except native.BindingNativeError as exc:
        raise ReferenceNativeError(str(exc)) from exc


def _file(value: Any, *, base: Path, owner: str) -> Path:
    try:
        return native._file(value, base=base, owner=owner)
    except native.BindingNativeError as exc:
        raise ReferenceNativeError(str(exc)) from exc


def _mapping(value: Any, owner: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ReferenceNativeError(f"{owner} must be an object")
    return dict(value)


def _normalise_reference_appearance(value: Any, owner: str) -> dict[str, str]:
    appearance = _mapping(value, owner)
    required = {"field", "value"}
    if set(appearance) != required:
        raise ReferenceNativeError(
            f"{owner} must contain exactly field and value"
        )
    field = appearance.get("field")
    selected_value = appearance.get("value")
    if (
        not isinstance(field, str)
        or not field.strip()
        or not isinstance(selected_value, str)
        or not selected_value.strip()
    ):
        raise ReferenceNativeError(f"{owner}.field and {owner}.value must be nonempty strings")
    field = field.strip()
    selected_value = selected_value.strip()
    if field != "top_color" or selected_value.lower() != "blue":
        raise ReferenceNativeError(
            f"{owner} must identify the reviewed blue top with field=top_color and value=blue"
        )
    return {"field": field, "value": selected_value.lower()}


def _finite_number(value: Any, owner: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReferenceNativeError(f"{owner} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ReferenceNativeError(f"{owner} must be a finite number")
    return result


def _position(value: Any, owner: str) -> list[float]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 3
    ):
        raise ReferenceNativeError(f"{owner} must be a three-number position")
    return [_finite_number(item, f"{owner}[{index}]") for index, item in enumerate(value)]


def _normalise_kind(value: Any, owner: str) -> str:
    folded = str(value or "").strip().lower()
    aliases = {
        "animal": "articulated_animal",
        "articulated_animal": "articulated_animal",
        "human": "articulated_human",
        "articulated_human": "articulated_human",
        "device": "rigid_object",
        "object": "rigid_object",
        "rigid_object": "rigid_object",
        "rigid_static_object": "rigid_object",
    }
    if folded not in aliases:
        raise ReferenceNativeError(
            f"{owner}.kind must be animal, human, or device"
        )
    return aliases[folded]


def _plan_path(value: str | Path, *, owner: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / "plan/episode_plan.json"
    if not path.is_file():
        raise ReferenceNativeError(f"{owner} is unavailable: {path}")
    return path


def _capture_path(value: str | Path, *, owner: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise ReferenceNativeError(f"{owner} must be a capture directory: {path}")
    return path


def _request_config(request: Mapping[str, Any]) -> dict[str, Any]:
    """Validate fixed QA-05 invariants and return the explicit recipe config."""
    recipe = request.get("reference_recipe")
    recipe = dict(recipe) if isinstance(recipe, Mapping) else {}

    qa_ids = request.get("qa_ids")
    if not isinstance(qa_ids, list) or "QA-05" not in qa_ids:
        raise ReferenceNativeError("reference recipe requires qa_ids containing QA-05")

    assets = request.get("source_asset_ids")
    if (
        isinstance(assets, (str, bytes))
        or not isinstance(assets, Sequence)
        or len(assets) != 4
        or len(set(assets)) != 4
        or any(not isinstance(item, str) or not item.strip() for item in assets)
    ):
        raise ReferenceNativeError(
            "reference recipe requires four distinct source_asset_ids"
        )

    entities = _mapping(request.get("entities"), "request.entities")
    if int(entities.get("total_count", -1)) != 4:
        raise ReferenceNativeError("reference recipe requires entities.total_count=4")
    if int(entities.get("silent_count", -1)) != 1:
        raise ReferenceNativeError("reference recipe requires entities.silent_count=1")

    camera = _mapping(request.get("camera"), "request.camera")
    if str(camera.get("motion") or "").lower() != "static":
        raise ReferenceNativeError("reference recipe requires a fixed camera")

    if int(request.get("frame_count", -1)) != 150:
        raise ReferenceNativeError("reference recipe requires frame_count=150")
    if not math.isclose(float(request.get("frame_rate_hz", -1)), 15.0, abs_tol=1e-9):
        raise ReferenceNativeError("reference recipe requires frame_rate_hz=15")
    if int(request.get("sample_rate_hz", -1)) != 16000:
        raise ReferenceNativeError("reference recipe requires sample_rate_hz=16000")

    profile = _mapping(request.get("profile"), "request.profile")
    reserve = _finite_number(profile.get("reserve_tail_s"), "profile.reserve_tail_s")
    if not math.isclose(reserve, 3.0, abs_tol=1e-9):
        raise ReferenceNativeError("reference recipe requires reserve_tail_s=3")
    gain = _finite_number(
        request.get("post_assembly_convolution_gain"),
        "post_assembly_convolution_gain",
    )
    if not math.isclose(gain, 0.5, abs_tol=1e-9):
        raise ReferenceNativeError(
            "reference recipe requires post_assembly_convolution_gain=0.5"
        )

    speaking = recipe.get(
        "speaking_actor_ids",
        request.get("speaking_actor_ids", list(REFERENCE_SPEAKING_ACTOR_IDS)),
    )
    if (
        isinstance(speaking, (str, bytes))
        or not isinstance(speaking, Sequence)
        or tuple(speaking) != REFERENCE_SPEAKING_ACTOR_IDS
    ):
        raise ReferenceNativeError(
            "speaking_actor_ids must be exactly source1, source2, source3"
        )
    reference_actor = recipe.get(
        "reference_actor_id",
        request.get("reference_actor_id", REFERENCE_REFERENCE_ACTOR_ID),
    )
    if reference_actor != REFERENCE_REFERENCE_ACTOR_ID:
        raise ReferenceNativeError("reference_actor_id must be source4")

    visual_selector = recipe.get(
        "visual_selector", request.get("visual_selector")
    )
    visual_selector = _mapping(visual_selector, "visual_selector")
    selector_kind = visual_selector.get("kind")
    if selector_kind not in REFERENCE_QUERY_KINDS:
        raise ReferenceNativeError(
            "visual_selector.kind must be "
            f"{REFERENCE_QUERY_KIND} or {REFERENCE_NAMED_QUERY_KIND}"
        )
    if selector_kind == REFERENCE_NAMED_QUERY_KIND:
        visual_selector["reference_appearance"] = _normalise_reference_appearance(
            visual_selector.get("reference_appearance"),
            "visual_selector.reference_appearance",
        )
    elif "reference_appearance" in visual_selector:
        raise ReferenceNativeError(
            "visual_selector.reference_appearance is only valid for "
            f"{REFERENCE_NAMED_QUERY_KIND}"
        )
    margin = _finite_number(
        visual_selector.get("minimum_margin_px"),
        "visual_selector.minimum_margin_px",
    )
    if margin < 0.0:
        raise ReferenceNativeError(
            "visual_selector.minimum_margin_px must be nonnegative and explicit"
        )
    for key in ("candidate_scope_en", "candidate_scope_zh"):
        value = visual_selector.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ReferenceNativeError(
                f"visual_selector.{key} must be nonempty"
            )
    visual_selector["minimum_margin_px"] = margin

    reference_time = recipe.get(
        "reference_time_s", request.get("reference_time_s", 0)
    )
    if isinstance(reference_time, bool) or not isinstance(reference_time, int):
        raise ReferenceNativeError("reference_time_s must be an integer")
    if reference_time != 0:
        raise ReferenceNativeError(
            "reference_time_s must be 0 for the public QA-05 reference question"
        )

    window = recipe.get("window_s", request.get("window_s"))
    if (
        isinstance(window, (str, bytes))
        or not isinstance(window, Sequence)
        or len(window) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in window)
        or int(window[0]) < 0
        or int(window[1]) <= int(window[0])
    ):
        raise ReferenceNativeError(
            "window_s must contain two increasing nonnegative integer seconds"
        )
    window_s = [int(window[0]), int(window[1])]

    scope_bindings = recipe.get(
        "candidate_scope_bindings",
        request.get("candidate_scope_bindings"),
    )
    scope_bindings = _mapping(scope_bindings, "candidate_scope_bindings")
    if set(scope_bindings) != set(REFERENCE_ACTOR_IDS):
        raise ReferenceNativeError(
            "candidate_scope_bindings must cover source1 through source4 exactly"
        )
    scope_bindings = {
        str(actor_id): _mapping(value, f"candidate_scope_bindings.{actor_id}")
        for actor_id, value in scope_bindings.items()
    }

    expected_pairs = recipe.get(
        "expected_pairs_by_variant",
        request.get("expected_pairs_by_variant", {}),
    )
    expected_pairs = _mapping(expected_pairs, "expected_pairs_by_variant")
    if set(expected_pairs) != {"v0", "v1"}:
        raise ReferenceNativeError(
            "expected_pairs_by_variant must declare v0 and v1"
        )
    normalised_pairs = {}
    pair_sets = {}
    speaking_set = set(REFERENCE_SPEAKING_ACTOR_IDS)
    for variant, expected in expected_pairs.items():
        if (
            isinstance(expected, (str, bytes))
            or not isinstance(expected, Sequence)
            or len(expected) != 2
            or any(
                not isinstance(item, str) or item not in speaking_set
                for item in expected
            )
            or len(set(expected)) != 2
        ):
            raise ReferenceNativeError(
                f"expected_pairs_by_variant.{variant} must contain two distinct speaking actor IDs"
            )
        normalised_pairs[variant] = sorted(str(item) for item in expected)
        pair_sets[variant] = set(normalised_pairs[variant])
    if (
        pair_sets["v0"] == pair_sets["v1"]
        or pair_sets["v0"] | pair_sets["v1"] != speaking_set
    ):
        raise ReferenceNativeError(
            "expected pair truth must use two distinct 2-of-3 speaking pairs whose union covers source1..source3"
        )

    reference_positions = recipe.get(
        "reference_positions_m",
        request.get("reference_positions_m"),
    )
    if reference_positions is not None:
        reference_positions = _mapping(reference_positions, "reference_positions_m")
        if set(reference_positions) != {"v0", "v1"}:
            raise ReferenceNativeError(
                "reference_positions_m must declare v0 and v1"
            )
        reference_positions = {
            variant: _position(value, f"reference_positions_m.{variant}")
            for variant, value in reference_positions.items()
        }

    audio_starts = recipe.get(
        "audio_start_times_s",
        request.get("audio_start_times_s"),
    )
    if audio_starts is not None:
        audio_starts = _mapping(audio_starts, "audio_start_times_s")
        if set(audio_starts) != {"a0", "a1"}:
            raise ReferenceNativeError(
                "audio_start_times_s must declare a0 and a1"
            )
        normalised_starts = {}
        for assignment, values in audio_starts.items():
            values = _mapping(values, f"audio_start_times_s.{assignment}")
            if set(values) != set(REFERENCE_SPEAKING_ACTOR_IDS):
                raise ReferenceNativeError(
                    f"audio_start_times_s.{assignment} must cover source1..source3"
                )
            normalised_starts[assignment] = {
                actor_id: _finite_number(
                    values[actor_id],
                    f"audio_start_times_s.{assignment}.{actor_id}",
                    nonnegative=True,
                )
                for actor_id in REFERENCE_SPEAKING_ACTOR_IDS
            }
        audio_starts = normalised_starts

    source_registry = request.get("source_registry")
    if not isinstance(source_registry, (str, Path)) or not str(source_registry).strip():
        raise ReferenceNativeError("request.source_registry is required")
    sound_pool = request.get("sound_pool")
    if not isinstance(sound_pool, (str, Path)) or not str(sound_pool).strip():
        raise ReferenceNativeError("request.sound_pool is required")
    _file(sound_pool, base=native.REPOSITORY, owner="sound pool")
    prepared_manifest = request.get("prepared_manifest")
    if not isinstance(prepared_manifest, (str, Path)) or not str(prepared_manifest).strip():
        raise ReferenceNativeError("request.prepared_manifest is required")
    _file(prepared_manifest, base=native.REPOSITORY, owner="prepared manifest")

    query = {
        "visual_selector": deepcopy(visual_selector),
        "reference_time_s": 0,
        "window_s": window_s,
    }
    return {
        "source_asset_ids": tuple(str(item) for item in assets),
        "speaking_actor_ids": REFERENCE_SPEAKING_ACTOR_IDS,
        "reference_actor_id": REFERENCE_REFERENCE_ACTOR_ID,
        "visual_selector": visual_selector,
        "reference_time_s": 0,
        "window_s": window_s,
        "candidate_scope_bindings": scope_bindings,
        "expected_pairs_by_variant": normalised_pairs,
        "reference_positions_m": reference_positions,
        "audio_start_times_s": audio_starts,
        "query": query,
        "source_registry": source_registry,
        "sound_pool": sound_pool,
        "prepared_manifest": prepared_manifest,
    }


def _registry(request: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    path = _file(
        request.get("source_registry"),
        base=native.REPOSITORY,
        owner="source registry",
    )
    value = _load(path)
    assets = value.get("assets")
    if not isinstance(assets, list):
        raise ReferenceNativeError("source registry.assets must be a list")
    return {
        str(row["asset_id"]): row
        for row in assets
        if isinstance(row, Mapping) and isinstance(row.get("asset_id"), str)
    }


def _plan_actors(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    visual = plan.get("visual_plan")
    if not isinstance(visual, Mapping) or not isinstance(visual.get("actors"), list):
        raise ReferenceNativeError("plan.visual_plan.actors must be a list")
    actors = {}
    for actor in visual["actors"]:
        if not isinstance(actor, Mapping):
            continue
        actor_id = actor.get("actor_id")
        if not isinstance(actor_id, str) or not actor_id.strip():
            raise ReferenceNativeError("every plan actor needs a nonempty actor_id")
        if actor_id in actors:
            raise ReferenceNativeError(f"duplicate plan actor_id: {actor_id}")
        actors[actor_id] = actor
    if set(actors) != set(REFERENCE_ACTOR_IDS):
        raise ReferenceNativeError(
            "reference plan must contain source1 through source4 exactly"
        )
    return actors


def _asset_kind(record: Mapping[str, Any]) -> str:
    entity_class = str(record.get("entity_class") or "").strip()
    return _normalise_kind(entity_class, "registry asset")


def _record_appearance_value(
    record: Mapping[str, Any],
    *,
    field: str,
    owner: str,
) -> str | None:
    if field == "top_color":
        realised = record.get("realized_attributes")
        if isinstance(realised, Mapping):
            value = realised.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
        return None
    raise ReferenceNativeError(f"{owner} uses unsupported appearance field: {field}")


def _validate_named_reference_scope(
    config: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Require one private source4 blue human and non-blue active humans."""
    selector = config["visual_selector"]
    appearance = _normalise_reference_appearance(
        selector.get("reference_appearance"),
        "visual_selector.reference_appearance",
    )
    field = appearance["field"]
    expected_value = appearance["value"]
    reference_actor_id = str(config["reference_actor_id"])
    matches = []
    observed = {}
    for actor_id in REFERENCE_ACTOR_IDS:
        asset_id = str(
            config["source_asset_ids"][REFERENCE_ACTOR_IDS.index(actor_id)]
        )
        record = records.get(asset_id)
        if not isinstance(record, Mapping):
            raise ReferenceNativeError(
                f"named reference asset is absent from source registry: {asset_id}"
            )
        kind = _asset_kind(record)
        value = _record_appearance_value(
            record,
            field=field,
            owner=f"source registry asset {asset_id}",
        )
        observed[actor_id] = {
            "asset_id": asset_id,
            "kind": kind,
            "value": value,
        }
        if kind == "articulated_human" and value == expected_value:
            matches.append(actor_id)
    if matches != [reference_actor_id]:
        raise ReferenceNativeError(
            "named reference must identify exactly one blue human and it must be source4"
        )
    reference = observed[reference_actor_id]
    if reference["kind"] != "articulated_human" or reference["value"] != expected_value:
        raise ReferenceNativeError(
            "source4 must be the uniquely reviewed blue-shirted human reference"
        )
    for actor_id in REFERENCE_SPEAKING_ACTOR_IDS:
        row = observed[actor_id]
        if row["kind"] == "articulated_human" and row["value"] == expected_value:
            raise ReferenceNativeError(
                f"active human {actor_id} cannot use the blue reference appearance"
            )
        if row["kind"] == "articulated_human" and row["value"] is None:
            raise ReferenceNativeError(
                f"active human {actor_id} lacks a non-blue top_color"
            )
    return {
        "status": "pass",
        "reference_actor_id": reference_actor_id,
        "reference_appearance": deepcopy(appearance),
        "matching_actor_ids": list(matches),
        "observed": observed,
        "active_humans_nonblue": True,
    }


def validate_candidate_scope(
    request: Mapping[str, Any],
    *,
    plan: Mapping[str, Any] | None = None,
    registry: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Check structured scope bindings against the selected registered assets."""
    config = _request_config(request)
    records = dict(registry or _registry(request))
    actors = _plan_actors(plan) if plan is not None else {}
    result = []
    for actor_id in REFERENCE_ACTOR_IDS:
        binding = config["candidate_scope_bindings"][actor_id]
        asset_id = (
            str(actors[actor_id].get("asset_id"))
            if actors
            else config["source_asset_ids"][REFERENCE_ACTOR_IDS.index(actor_id)]
        )
        if asset_id not in records:
            raise ReferenceNativeError(
                f"candidate scope actor {actor_id} asset is absent from source registry: {asset_id}"
            )
        if asset_id != config["source_asset_ids"][REFERENCE_ACTOR_IDS.index(actor_id)]:
            raise ReferenceNativeError(
                f"plan actor {actor_id} asset differs from request source_asset_ids"
            )
        actual = records[asset_id]
        expected_kind = _normalise_kind(binding.get("kind"), f"candidate_scope_bindings.{actor_id}")
        actual_kind = _asset_kind(actual)
        if actual_kind != expected_kind:
            raise ReferenceNativeError(
                f"candidate scope kind mismatch for {actor_id}: {actual_kind} != {expected_kind}"
            )
        identity = actual.get("identity") if isinstance(actual.get("identity"), Mapping) else {}
        realised = actual.get("realized_attributes") if isinstance(actual.get("realized_attributes"), Mapping) else {}
        for field in ("species_id", "breed_id", "object_type", "object_category", "category", "top_color"):
            if field not in binding:
                continue
            expected = binding[field]
            if field in {"object_category", "category"}:
                observed = identity.get("category")
            elif field == "top_color":
                observed = realised.get("top_color")
            else:
                observed = identity.get(field)
            if observed != expected:
                raise ReferenceNativeError(
                    f"candidate scope {field} mismatch for {actor_id}: {observed!r} != {expected!r}"
                )
        binding_asset = binding.get("asset_id")
        if binding_asset is not None and binding_asset != asset_id:
            raise ReferenceNativeError(
                f"candidate scope asset_id mismatch for {actor_id}"
            )
        result.append({
            "actor_id": actor_id,
            "asset_id": asset_id,
            "kind": actual_kind,
            "identity": deepcopy(dict(identity)),
            "realized_attributes": deepcopy(dict(realised)),
            "scope_binding": deepcopy(binding),
        })
    selector = config["visual_selector"]
    named_reference = None
    if selector["kind"] == REFERENCE_NAMED_QUERY_KIND:
        named_reference = _validate_named_reference_scope(config, records)
    return {
        "status": "pass",
        "candidate_scope_en": selector["candidate_scope_en"],
        "candidate_scope_zh": selector["candidate_scope_zh"],
        "actors": result,
        "named_reference": named_reference,
        "claim_boundary": "registry identity and declared scope only; native visibility and appearance review remain separate",
    }


def _configured_reference_pairs(config: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    pairs = config.get("expected_pairs_by_variant")
    if not isinstance(pairs, Mapping):
        raise ReferenceNativeError("reference config lacks expected_pairs_by_variant")
    return {
        variant: tuple(str(actor_id) for actor_id in pairs[variant])
        for variant in ("v0", "v1")
    }


def _clock(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    clock = plan.get("clock")
    if not isinstance(clock, Mapping):
        raise ReferenceNativeError("plan.clock must be an object")
    required = ("frame_count", "frame_rate_hz", "sample_rate_hz", "sample_count", "time_base_hz", "ticks_per_frame")
    if any(key not in clock for key in required):
        raise ReferenceNativeError(f"plan.clock lacks one of {required}")
    return clock


def _state_map(frame: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    states = frame.get("actor_states")
    if not isinstance(states, list):
        raise ReferenceNativeError("every visual frame needs actor_states")
    result = {}
    for state in states:
        if not isinstance(state, Mapping) or not isinstance(state.get("actor_id"), str):
            raise ReferenceNativeError("every actor state needs actor_id")
        if state["actor_id"] in result:
            raise ReferenceNativeError(f"duplicate actor state: {state['actor_id']}")
        result[state["actor_id"]] = state
    if set(result) != set(REFERENCE_ACTOR_IDS):
        raise ReferenceNativeError("every frame must contain source1 through source4")
    return result


def _same(left: Any, right: Any, *, tolerance: float = 1.0e-8) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        try:
            return math.isfinite(float(left)) and math.isfinite(float(right)) and abs(float(left) - float(right)) <= tolerance
        except (TypeError, ValueError, OverflowError):
            return False
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_same(left[key], right[key], tolerance=tolerance) for key in left)
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)) and isinstance(right, Sequence) and not isinstance(right, (str, bytes)):
        return len(left) == len(right) and all(_same(a, b, tolerance=tolerance) for a, b in zip(left, right, strict=True))
    return left == right


def _delta(left: Any, right: Any) -> float:
    if isinstance(left, bool) or isinstance(right, bool):
        return 0.0 if left == right else math.inf
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        try:
            a, b = float(left), float(right)
        except (TypeError, ValueError, OverflowError):
            return math.inf
        return abs(a - b) if math.isfinite(a) and math.isfinite(b) else math.inf
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return math.inf
        return max((_delta(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)) and isinstance(right, Sequence) and not isinstance(right, (str, bytes)):
        if len(left) != len(right):
            return math.inf
        return max((_delta(a, b) for a, b in zip(left, right, strict=True)), default=0.0)
    return 0.0 if left == right else math.inf


def _static_state_projection(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "root_transform": deepcopy(state.get("root_transform")),
        "planned_emitter_m": deepcopy(state.get("planned_emitter_m")),
        "moving": deepcopy(state.get("moving")),
        "action_id": deepcopy(state.get("action_id")),
        "action_phase": deepcopy(state.get("action_phase")),
        "action_time_ticks": deepcopy(state.get("action_time_ticks")),
    }


def _validate_visual_plan(plan: Mapping[str, Any], request: Mapping[str, Any], *, label: str) -> dict[str, Mapping[str, Any]]:
    config = _request_config(request)
    clock = _clock(plan)
    if int(clock["frame_count"]) != 150 or not math.isclose(float(clock["frame_rate_hz"]), 15.0, abs_tol=1e-9) or int(clock["sample_rate_hz"]) != 16000:
        raise ReferenceNativeError(f"{label} plan clock differs from the fixed QA-05 clock")
    expected_samples = int(round(int(clock["sample_rate_hz"]) * int(clock["frame_count"]) / float(clock["frame_rate_hz"])))
    if int(clock["sample_count"]) != expected_samples:
        raise ReferenceNativeError(f"{label} plan sample_count is not 10 seconds")
    visual = _mapping(plan.get("visual_plan"), f"{label}.visual_plan")
    camera = _mapping(visual.get("camera"), f"{label}.visual_plan.camera")
    if str(camera.get("motion") or "").lower() != "static":
        raise ReferenceNativeError(f"{label} plan camera is not static")
    frames = visual.get("frames")
    if not isinstance(frames, list) or len(frames) != 150:
        raise ReferenceNativeError(f"{label} plan must contain 150 visual frames")
    actor_map = _plan_actors(plan)
    expected_assets = dict(zip(REFERENCE_ACTOR_IDS, config["source_asset_ids"], strict=True))
    for actor_id, actor in actor_map.items():
        if actor.get("asset_id") != expected_assets[actor_id]:
            raise ReferenceNativeError(f"{label} actor {actor_id} asset differs from request")
        binding = actor.get("emitter_binding")
        if not isinstance(binding, Mapping) or not isinstance(binding.get("semantic_anchor_id"), str):
            raise ReferenceNativeError(f"{label} actor {actor_id} lacks semantic emitter anchor")
    first_states = None
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise ReferenceNativeError(f"{label} frame {frame_index} is invalid")
        states = _state_map(frame)
        frame_camera = frame.get("camera_state")
        if not isinstance(frame_camera, Mapping):
            raise ReferenceNativeError(f"{label} frame {frame_index} lacks camera_state")
        if frame_camera.get("motion", "static") != "static":
            raise ReferenceNativeError(f"{label} frame {frame_index} camera moves")
        if first_states is None:
            first_states = states
        for actor_id, state in states.items():
            if state.get("moving") is not False:
                raise ReferenceNativeError(
                    f"{label} actor {actor_id} must remain static in this reference recipe"
                )
            if not isinstance(state.get("root_transform"), Mapping):
                raise ReferenceNativeError(f"{label} actor {actor_id} lacks root_transform")
            if actor_id in REFERENCE_SPEAKING_ACTOR_IDS and not isinstance(state.get("planned_emitter_m"), Sequence):
                raise ReferenceNativeError(f"{label} speaking actor {actor_id} lacks planned_emitter_m")
            if frame_index and not _same(
                _static_state_projection(state),
                _static_state_projection(first_states[actor_id]),
            ):
                raise ReferenceNativeError(
                    f"{label} actor {actor_id} is not fixed across frames"
                )
    events = [row for row in plan.get("audio_events", []) if isinstance(row, Mapping)]
    if len(events) != 3:
        raise ReferenceNativeError(f"{label} must retain exactly three speaking events")
    by_actor = {}
    event_ids = set()
    for event in events:
        event_id = event.get("event_id")
        actor_id = event.get("actor_id")
        if not isinstance(event_id, str) or not event_id.strip() or event_id in event_ids:
            raise ReferenceNativeError(f"{label} audio event IDs must be unique")
        if actor_id not in REFERENCE_SPEAKING_ACTOR_IDS or actor_id in by_actor:
            raise ReferenceNativeError(
                f"{label} audio events must contain one event for each speaking actor"
            )
        event_ids.add(event_id)
        sample_count = event.get("sample_count")
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
            raise ReferenceNativeError(f"{label} event {event_id} lacks positive sample_count")
        if int(event.get("source_start_sample", -1)) != 0 or int(event.get("source_end_sample_exclusive", -1)) != sample_count:
            raise ReferenceNativeError(
                f"{label} event {event_id} must retain its complete source clip"
            )
        endpoint = event.get("source_endpoint_id")
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ReferenceNativeError(f"{label} event {event_id} lacks source_endpoint_id")
        intervals = _activity_intervals(event)
        if not intervals:
            raise ReferenceNativeError(f"{label} event {event_id} lacks activity intervals")
        for start, end in intervals:
            if start < 0 or end <= start or end > sample_count:
                raise ReferenceNativeError(
                    f"{label} event {event_id} has activity outside its complete clip"
                )
        _clip_path(event, request)
        by_actor[str(actor_id)] = event
    if set(by_actor) != set(REFERENCE_SPEAKING_ACTOR_IDS):
        raise ReferenceNativeError(f"{label} speaking event actor coverage is incomplete")
    bindings = [row for row in plan.get("voice_bindings", []) if isinstance(row, Mapping)]
    if not bindings:
        raise ReferenceNativeError(f"{label} must retain voice_bindings")
    records = _registry(request)
    from avengine.rooms.conditioned_sampler import sound_matches
    for event in events:
        sound_id = str(event.get("sound_asset_id"))
        actor_id = str(event["actor_id"])
        binding = next(
            (
                row for row in bindings
                if str(row.get("sound_asset_id") or "") == sound_id
                and (row.get("actor_id") is None or str(row.get("actor_id")) == actor_id)
            ),
            None,
        )
        if binding is None:
            raise ReferenceNativeError(
                f"{label} lacks voice binding for {event.get('sound_asset_id')}"
            )
        actor = actor_map[actor_id]
        asset_id = str(actor.get("asset_id"))
        if asset_id not in records:
            raise ReferenceNativeError(f"{label} actor asset is absent from source registry: {asset_id}")
        binding_actor = binding.get("actor_id")
        if binding_actor is not None and str(binding_actor) != actor_id:
            raise ReferenceNativeError(f"{label} sound {sound_id} is bound to the wrong actor")
        binding_endpoint = binding.get("source_endpoint_id")
        if binding_endpoint is not None and str(binding_endpoint) != str(event["source_endpoint_id"]):
            raise ReferenceNativeError(f"{label} sound {sound_id} endpoint differs from its event")
        if not sound_matches({
            "entity_class": actor.get("entity_class") or records[asset_id].get("entity_class"),
            "asset_id": asset_id,
            "identity": deepcopy(dict(actor.get("identity") or records[asset_id].get("identity") or {})),
            "realized_attributes": deepcopy(dict(actor.get("realized_attributes") or records[asset_id].get("realized_attributes") or {})),
        }, binding):
            raise ReferenceNativeError(
                f"{label} sound {sound_id} is incompatible with actor {actor_id}"
            )
        allowed_assets = binding.get("compatible_asset_ids")
        if isinstance(allowed_assets, list) and allowed_assets and asset_id not in allowed_assets:
            raise ReferenceNativeError(
                f"{label} sound {sound_id} does not allow actor asset {asset_id}"
            )
    return actor_map


def _clip_path(event: Mapping[str, Any], request: Mapping[str, Any]) -> Path:
    pool_path = _file(request.get("sound_pool"), base=native.REPOSITORY, owner="sound pool")
    value = event.get("path") or event.get("prepared")
    return _file(value, base=pool_path.parent, owner="complete source clip")


def _activity_intervals(event: Mapping[str, Any]) -> list[tuple[int, int]]:
    raw = event.get("source_activity_intervals_samples")
    if isinstance(raw, list) and raw:
        result = []
        for item in raw:
            if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
                raise ReferenceNativeError("source activity interval must be [start,end)")
            result.append((int(item[0]), int(item[1])))
        return result
    start = event.get("audible_start_sample")
    end = event.get("audible_end_sample_exclusive")
    if isinstance(start, int) and isinstance(end, int):
        return [(start, end)]
    return []


def _clone_reference_variant(
    plan: Mapping[str, Any],
    *,
    reference_actor_id: str,
    position: Sequence[float],
    variant: str,
) -> dict[str, Any]:
    cloned = deepcopy(dict(plan))
    visual = cloned["visual_plan"]
    frames = visual["frames"]
    first = _state_map(frames[0])[reference_actor_id]
    old_root = first["root_transform"].get("translation_m")
    old_emitter = first.get("planned_emitter_m")
    if not isinstance(old_root, Sequence) or not isinstance(old_emitter, Sequence):
        raise ReferenceNativeError("reference actor needs root and planned emitter positions")
    offset = [float(old_emitter[i]) - float(old_root[i]) for i in range(3)]
    new_position = _position(position, f"reference_positions_m.{variant}")
    for frame in frames:
        states = _state_map(frame)
        state = states[reference_actor_id]
        root = deepcopy(dict(state["root_transform"]))
        root["translation_m"] = list(new_position)
        state["root_transform"] = root
        state["planned_emitter_m"] = [
            new_position[i] + offset[i] for i in range(3)
        ]
        state["moving"] = False
    cloned["reference_visual_variant"] = variant
    return cloned


def compare_reference_visual_plans(
    v0: Mapping[str, Any],
    v1: Mapping[str, Any],
    *,
    reference_actor_id: str = REFERENCE_REFERENCE_ACTOR_ID,
) -> dict[str, Any]:
    """Require camera/speakers to stay fixed while allowing only silent-reference position."""
    if reference_actor_id != REFERENCE_REFERENCE_ACTOR_ID:
        raise ReferenceNativeError("reference_actor_id must be source4")
    visual0 = _mapping(v0.get("visual_plan"), "v0.visual_plan")
    visual1 = _mapping(v1.get("visual_plan"), "v1.visual_plan")
    if not _same(visual0.get("camera"), visual1.get("camera")):
        raise ReferenceNativeError("v0/v1 camera differs")
    if not _same(v0.get("clock"), v1.get("clock")):
        raise ReferenceNativeError("v0/v1 clock differs")
    if native.room_family_from_plan(v0) != native.room_family_from_plan(v1):
        raise ReferenceNativeError("v0/v1 validated room family differs")
    resources0 = v0.get("resources") if isinstance(v0.get("resources"), Mapping) else {}
    resources1 = v1.get("resources") if isinstance(v1.get("resources"), Mapping) else {}
    room0 = (
        (resources0.get("room_package") or {}).get("room_id")
        if isinstance(resources0.get("room_package"), Mapping)
        else None
    ) or resources0.get("room_id") or (v0.get("scene") or {}).get("room_id")
    room1 = (
        (resources1.get("room_package") or {}).get("room_id")
        if isinstance(resources1.get("room_package"), Mapping)
        else None
    ) or resources1.get("room_id") or (v1.get("scene") or {}).get("room_id")
    if room0 != room1:
        raise ReferenceNativeError("v0/v1 validated room_id differs")
    frames0, frames1 = visual0.get("frames"), visual1.get("frames")
    if not isinstance(frames0, list) or not isinstance(frames1, list) or len(frames0) != len(frames1):
        raise ReferenceNativeError("v0/v1 frame count differs")
    max_speaker_delta = 0.0
    ref_deltas = []
    for frame0, frame1 in zip(frames0, frames1, strict=True):
        states0, states1 = _state_map(frame0), _state_map(frame1)
        if not _same(frame0.get("camera_state"), frame1.get("camera_state")):
            raise ReferenceNativeError("v0/v1 frame camera differs")
        for actor_id in REFERENCE_SPEAKING_ACTOR_IDS:
            delta = _delta(_static_state_projection(states0[actor_id]), _static_state_projection(states1[actor_id]))
            max_speaker_delta = max(max_speaker_delta, delta)
            if delta > 1.0e-8:
                raise ReferenceNativeError(
                    f"speaking actor {actor_id} changed between visual variants"
                )
        ref_deltas.append(_delta(
            _static_state_projection(states0[reference_actor_id]),
            _static_state_projection(states1[reference_actor_id]),
        ))
        if states0[reference_actor_id].get("moving") is not False or states1[reference_actor_id].get("moving") is not False:
            raise ReferenceNativeError("silent reference must remain static")
    if max(ref_deltas, default=0.0) <= 1.0e-8:
        raise ReferenceNativeError("v0/v1 silent-reference positions are identical")
    return {
        "status": "pass",
        "changed_actor": reference_actor_id,
        "max_speaking_state_delta": max_speaker_delta,
        "max_reference_state_delta": max(ref_deltas, default=0.0),
        "camera_and_speaking_states": "identical",
        "reference_position_only": True,
    }


def _centre_value(value: Any, owner: str) -> tuple[float, float, bool]:
    visible = True
    if isinstance(value, Mapping):
        visible = value.get("visible", value.get("visible_pixels", True)) not in (False, 0, "0")
        value = value.get("center_xy_px", value.get("centroid_xy_px", value.get("target_centroid_xy_px")))
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        raise ReferenceNativeError(f"{owner} must provide a two-number pixel center")
    x = _finite_number(value[0], f"{owner}[0]")
    y = _finite_number(value[1], f"{owner}[1]")
    return x, y, bool(visible)


def select_two_nearest_to_leftmost(
    pixel_centers_px: Mapping[str, Any],
    *,
    minimum_margin_px: float,
    expected_pair: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Select the unique leftmost candidate and two nearest pixel centroids."""
    if not isinstance(pixel_centers_px, Mapping) or set(pixel_centers_px) != set(REFERENCE_ACTOR_IDS):
        raise ReferenceNativeError(
            "pixel centers must cover source1 through source4 exactly"
        )
    margin = _finite_number(minimum_margin_px, "minimum_margin_px")
    if margin < 0.0:
        raise ReferenceNativeError("minimum_margin_px must be nonnegative")
    centers = {}
    for actor_id in REFERENCE_ACTOR_IDS:
        x, y, visible = _centre_value(pixel_centers_px[actor_id], f"pixel_centers_px.{actor_id}")
        if not visible:
            raise ReferenceNativeError(f"{actor_id} is not visibly observed at reference_time_s=0")
        centers[actor_id] = [x, y]
    ordered_x = sorted(REFERENCE_ACTOR_IDS, key=lambda actor_id: (centers[actor_id][0], actor_id))
    leftmost = ordered_x[0]
    leftmost_margin = centers[ordered_x[1]][0] - centers[leftmost][0]
    if leftmost_margin <= margin:
        raise ReferenceNativeError(
            f"leftmost uniqueness margin {leftmost_margin:.6g}px is below configured {margin:.6g}px"
        )
    distances = []
    for actor_id in REFERENCE_ACTOR_IDS:
        if actor_id == leftmost:
            continue
        dx = centers[actor_id][0] - centers[leftmost][0]
        dy = centers[actor_id][1] - centers[leftmost][1]
        distances.append((math.hypot(dx, dy), actor_id))
    distances.sort(key=lambda row: (row[0], row[1]))
    nearest = [row[1] for row in distances[:2]]
    nearest_margin = distances[2][0] - distances[1][0]
    if nearest_margin <= margin:
        raise ReferenceNativeError(
            f"nearest-selection margin {nearest_margin:.6g}px is below configured {margin:.6g}px"
        )
    if expected_pair is not None and set(nearest) != set(expected_pair):
        raise ReferenceNativeError(
            f"selected nearest pair {sorted(nearest)} differs from expected {sorted(expected_pair)}"
        )
    return {
        "status": "pass",
        "reference_actor_id": leftmost,
        "selected_pair": sorted(nearest),
        "centers_px": centers,
        "leftmost_unique_margin_px": leftmost_margin,
        "nearest_selection_margin_px": nearest_margin,
        "distances_px": [
            {"actor_id": actor_id, "distance_px": distance}
            for distance, actor_id in distances
        ],
        "minimum_margin_px": margin,
        "visibility": {actor_id: True for actor_id in REFERENCE_ACTOR_IDS},
        "authority": "pixel_centroid_at_reference_time_s",
    }


def select_two_nearest_to_named_reference(
    pixel_centers_px: Mapping[str, Any],
    *,
    reference_actor_id: str,
    reference_appearance: Mapping[str, Any],
    minimum_margin_px: float,
    expected_pair: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Select the two nearest visible centroids to a private named reference."""
    if (
        not isinstance(reference_actor_id, str)
        or reference_actor_id not in REFERENCE_ACTOR_IDS
    ):
        raise ReferenceNativeError(
            "named reference actor must be source1 through source4"
        )
    appearance = _normalise_reference_appearance(
        reference_appearance,
        "reference_appearance",
    )
    if (
        not isinstance(pixel_centers_px, Mapping)
        or set(pixel_centers_px) != set(REFERENCE_ACTOR_IDS)
    ):
        raise ReferenceNativeError(
            "pixel centers must cover source1 through source4 exactly"
        )
    margin = _finite_number(minimum_margin_px, "minimum_margin_px")
    if margin < 0.0:
        raise ReferenceNativeError("minimum_margin_px must be nonnegative")
    centers = {}
    visibility = {}
    for actor_id in REFERENCE_ACTOR_IDS:
        x, y, visible = _centre_value(
            pixel_centers_px[actor_id],
            f"pixel_centers_px.{actor_id}",
        )
        if not visible:
            raise ReferenceNativeError(
                f"{actor_id} is not visibly observed at reference_time_s=0"
            )
        centers[actor_id] = [x, y]
        visibility[actor_id] = True
    distances = []
    reference_center = centers[reference_actor_id]
    for actor_id in REFERENCE_ACTOR_IDS:
        if actor_id == reference_actor_id:
            continue
        distance = math.hypot(
            centers[actor_id][0] - reference_center[0],
            centers[actor_id][1] - reference_center[1],
        )
        distances.append((distance, actor_id))
    distances.sort(key=lambda row: (row[0], row[1]))
    nearest = [row[1] for row in distances[:2]]
    nearest_margin = distances[2][0] - distances[1][0]
    if nearest_margin <= margin:
        raise ReferenceNativeError(
            f"nearest-selection margin {nearest_margin:.6g}px is below configured {margin:.6g}px"
        )
    if expected_pair is not None and set(nearest) != set(expected_pair):
        raise ReferenceNativeError(
            f"selected nearest pair {sorted(nearest)} differs from expected {sorted(expected_pair)}"
        )
    return {
        "status": "pass",
        "reference_actor_id": reference_actor_id,
        "reference_appearance": deepcopy(appearance),
        "selected_pair": sorted(nearest),
        "centers_px": centers,
        "nearest_selection_margin_px": nearest_margin,
        "distances_px": [
            {"actor_id": actor_id, "distance_px": distance}
            for distance, actor_id in distances
        ],
        "minimum_margin_px": margin,
        "visibility": visibility,
        "authority": "pixel_centroid_at_reference_time_s_named_reference",
    }


def _pixel_centers_from_truth(
    value: str | Path | Mapping[str, Any],
    *,
    frame_index: int,
    owner: str,
) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / "pixel_visibility_truth.json"
    if not path.is_file():
        raise ReferenceNativeError(f"{owner} pixel truth is unavailable: {path}")
    payload = _load(path)
    per_instance = payload.get("per_instance")
    if not isinstance(per_instance, Mapping):
        raise ReferenceNativeError(f"{owner} pixel truth lacks per_instance")
    result = {}
    for actor_id in REFERENCE_ACTOR_IDS:
        record = per_instance.get(actor_id)
        if not isinstance(record, Mapping) or not isinstance(record.get("frames"), list):
            raise ReferenceNativeError(f"{owner} pixel truth lacks {actor_id}.frames")
        rows = [row for row in record["frames"] if isinstance(row, Mapping) and int(row.get("frame_index", -1)) == frame_index]
        if len(rows) != 1:
            raise ReferenceNativeError(f"{owner} pixel truth lacks unique frame {frame_index} for {actor_id}")
        row = rows[0]
        if int(row.get("visible_pixels", 0)) <= 0:
            raise ReferenceNativeError(f"{owner} {actor_id} is not visible at frame {frame_index}")
        result[actor_id] = {
            "center_xy_px": row.get("target_centroid_xy_px"),
            "visible_pixels": int(row["visible_pixels"]),
            "state": row.get("state"),
        }
    return result


def _audio_events_by_actor(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    events = [row for row in plan.get("audio_events", []) if isinstance(row, Mapping)]
    result = {str(row.get("actor_id")): row for row in events if row.get("actor_id")}
    if set(result) != set(REFERENCE_SPEAKING_ACTOR_IDS) or len(result) != len(events):
        raise ReferenceNativeError("reference audio plan must have one event per speaking actor")
    return result


def _shifted_activity(event: Mapping[str, Any], start_sample: int) -> list[list[int]]:
    return [
        [start_sample + start, start_sample + end]
        for start, end in _activity_intervals(event)
    ]


def _overlap_in_window(
    left: Sequence[Sequence[int]],
    right: Sequence[Sequence[int]],
    window_samples: Sequence[int],
) -> bool:
    lo, hi = int(window_samples[0]), int(window_samples[1])
    for left_start, left_end in left:
        for right_start, right_end in right:
            if max(lo, int(left_start), int(right_start)) < min(hi, int(left_end), int(right_end)):
                return True
    return False


def schedule_reference_audio_plan(
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    audio_variant: str,
    start_times_s_by_actor: Mapping[str, float],
    target_pair: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Shift only event timing while retaining each actor's complete clip/endpoint."""
    if audio_variant not in {"a0", "a1"}:
        raise ReferenceNativeError("audio_variant must be a0 or a1")
    plan = _normalise_planner_audio_metadata(plan)
    config = _request_config(request)
    if (
        isinstance(target_pair, (str, bytes))
        or not isinstance(target_pair, Sequence)
        or len(target_pair) != 2
        or any(
            not isinstance(actor_id, str)
            or actor_id not in REFERENCE_SPEAKING_ACTOR_IDS
            for actor_id in target_pair
        )
        or len(set(target_pair)) != 2
    ):
        raise ReferenceNativeError(
            "target_pair must contain two distinct speaking actor IDs"
        )
    starts = _mapping(start_times_s_by_actor, "start_times_s_by_actor")
    if set(starts) != set(REFERENCE_SPEAKING_ACTOR_IDS):
        raise ReferenceNativeError("start_times_s_by_actor must cover source1..source3")
    clock = _clock(plan)
    sr, tb, sample_count = int(clock["sample_rate_hz"]), int(clock["time_base_hz"]), int(clock["sample_count"])
    reserve_samples = int(round(3.0 * sr))
    event_by_actor = _audio_events_by_actor(plan)
    scheduled = deepcopy(dict(plan))
    events = []
    shifted_by_actor = {}
    latest_end = 0
    for actor_id in REFERENCE_SPEAKING_ACTOR_IDS:
        event = event_by_actor[actor_id]
        start_seconds = _finite_number(starts[actor_id], f"start_times_s_by_actor.{actor_id}", nonnegative=True)
        start_sample = int(round(start_seconds * sr))
        duration = int(event["sample_count"])
        end_sample = start_sample + duration
        if end_sample > sample_count - reserve_samples:
            raise ReferenceNativeError(
                f"{actor_id} complete clip leaves less than 3 seconds of tail"
            )
        shifted = deepcopy(dict(event))
        shifted.update(
            start_sample=start_sample,
            end_sample=end_sample,
            end_sample_exclusive=end_sample,
            start_tick=int(round(start_sample * tb / sr)),
            end_tick=int(round(end_sample * tb / sr)),
            end_tick_exclusive=int(round(end_sample * tb / sr)),
            start_s=start_sample / sr,
            end_s=end_sample / sr,
            scheduled_activity_intervals_samples=_shifted_activity(event, start_sample),
            source_activity_schedule_status="timing_only_from_complete_clip_activity_intervals",
        )
        shifted["planned_audible_interval_samples"] = [
            shifted["scheduled_activity_intervals_samples"][0][0],
            shifted["scheduled_activity_intervals_samples"][-1][1],
        ]
        events.append(shifted)
        shifted_by_actor[actor_id] = shifted["scheduled_activity_intervals_samples"]
        latest_end = max(latest_end, end_sample)
    window = [int(config["window_s"][0] * sr), int(config["window_s"][1] * sr)]
    pair_truth = {}
    for left, right in (
        ("source1", "source2"),
        ("source1", "source3"),
        ("source2", "source3"),
    ):
        pair_truth[f"{left}+{right}"] = _overlap_in_window(
            shifted_by_actor[left], shifted_by_actor[right], window
        )
    target = tuple(sorted(str(item) for item in target_pair))
    configured_pairs = _configured_reference_pairs(config)
    target_variant = "v0" if audio_variant == "a0" else "v1"
    opposite_variant = "v1" if audio_variant == "a0" else "v0"
    if set(target) != set(configured_pairs[target_variant]):
        raise ReferenceNativeError(
            f"{audio_variant} target pair differs from expected_pairs_by_variant.{target_variant}"
        )
    opposite = tuple(sorted(configured_pairs[opposite_variant]))
    if not pair_truth[f"{target[0]}+{target[1]}"]:
        raise ReferenceNativeError(
            f"{audio_variant} target pair has no actual activity overlap in the query window"
        )
    opposite_key = f"{opposite[0]}+{opposite[1]}"
    if pair_truth[opposite_key]:
        raise ReferenceNativeError(
            f"{audio_variant} opposite configured pair also overlaps in the query window"
        )
    available_tail = (sample_count - latest_end) / sr
    if available_tail < 3.0 - 1.0e-9:
        raise ReferenceNativeError("scheduled audio leaves less than the fixed 3-second tail")
    schedule = {
        "schema": "avengine_reference_audio_schedule_v1",
        "status": "pass",
        "audio_variant": audio_variant,
        "intervention": "timing_only_preserve_actor_clip_and_endpoint",
        "window_s": list(config["window_s"]),
        "window_samples": window,
        "target_pair": list(target),
        "opposite_pair": list(opposite),
        "pair_overlap_truth": pair_truth,
        "start_times_s_by_actor": {
            actor_id: float(starts[actor_id]) for actor_id in REFERENCE_SPEAKING_ACTOR_IDS
        },
        "events": [
            {
                "event_id": str(event["event_id"]),
                "actor_id": str(event["actor_id"]),
                "source_endpoint_id": str(event["source_endpoint_id"]),
                "sound_asset_id": str(event["sound_asset_id"]),
                "sample_count": int(event["sample_count"]),
                "start_sample": int(event["start_sample"]),
                "end_sample_exclusive": int(event["end_sample_exclusive"]),
                "scheduled_activity_intervals_samples": deepcopy(
                    event["scheduled_activity_intervals_samples"]
                ),
            }
            for event in events
        ],
        "available_tail_s": available_tail,
        "reserve_tail_s": 3.0,
        "activity_authority": "actual_prepared_clip_activity_intervals",
    }
    events.sort(key=lambda row: (int(row["start_sample"]), str(row["event_id"])))
    scheduled["audio_events"] = events
    scheduled["audio_schedule"] = deepcopy(schedule)
    scheduled_request = deepcopy(dict(request))
    scheduled_request["audio_schedule"] = deepcopy(schedule)
    scheduled_request["reference_audio_variant"] = audio_variant
    scheduled_request["reference_audio_target_pair"] = list(target)
    scheduled["request"] = deepcopy(scheduled_request)
    return scheduled, scheduled_request, schedule


def _audio_column_signature(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            key: deepcopy(event.get(key))
            for key in (
                "event_id",
                "actor_id",
                "source_endpoint_id",
                "sound_asset_id",
                "path",
                "prepared",
                "sample_count",
                "source_start_sample",
                "source_end_sample_exclusive",
                "start_sample",
                "end_sample_exclusive",
                "start_tick",
                "end_tick_exclusive",
                "scheduled_activity_intervals_samples",
            )
        }
        for event in sorted(
            (row for row in plan.get("audio_events", []) if isinstance(row, Mapping)),
            key=lambda row: str(row.get("event_id")),
        )
    ]


def build_reference_audio_assignment_plan(
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
    assignment: str,
    *,
    endpoint_by_actor: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reuse the native assignment helper without moving a clip to another actor."""
    if assignment not in {"a0", "a1"}:
        raise ReferenceNativeError("assignment must be a0 or a1")
    event_targets = {
        str(event["event_id"]): str(event["actor_id"])
        for event in plan.get("audio_events", [])
        if isinstance(event, Mapping)
        and isinstance(event.get("event_id"), str)
        and isinstance(event.get("actor_id"), str)
    }
    if len(event_targets) != 3:
        raise ReferenceNativeError(
            "reference assignment requires three uniquely identified audio events"
        )
    try:
        return native.build_audio_assignment_plan(
            plan,
            request,
            assignment,
            assignment_targets={"a0": event_targets, "a1": event_targets},
            expected_event_count=3,
            endpoint_by_actor=endpoint_by_actor,
            require_authoritative_endpoints=True,
        )
    except native.BindingNativeError as exc:
        raise ReferenceNativeError(str(exc)) from exc


def finalize_reference_audio_variant(
    visual_capture: Mapping[str, Any],
    output: str | Path,
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    member_id: str,
    endpoint_by_actor: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Materialize one member and invoke the existing independent finalizer."""
    if not isinstance(member_id, str) or member_id not in {
        "v0_a0", "v0_a1", "v1_a0", "v1_a1"
    }:
        raise ReferenceNativeError("member_id must be one of v0_a0/v0_a1/v1_a0/v1_a1")
    if endpoint_by_actor is None:
        endpoint_by_actor = native._neutral_endpoint_bindings(
            visual_capture["neutral_readback"], plan=plan
        )
    assignment = member_id.rsplit("_", 1)[1]
    assigned, rebound = build_reference_audio_assignment_plan(
        plan, request, assignment, endpoint_by_actor=endpoint_by_actor
    )
    try:
        root = native.materialize_audio_variant(
            visual_capture,
            Path(output).expanduser().resolve(),
            assigned,
            rebound,
            member_id=member_id,
        )
        # Deliberately omit audio_report: all four members get independent RLR.
        return native.finalize_audio_assignment(root, rebound)
    except (native.BindingNativeError, OSError, RuntimeError, ValueError) as exc:
        raise ReferenceNativeError(str(exc)) from exc


def _write_group_spec(
    output: Path,
    *,
    group_id: str,
    world_id: str,
    request: Mapping[str, Any],
    config: Mapping[str, Any],
    visual_plans: Mapping[str, Mapping[str, Any]],
    member_plans: Mapping[str, Path],
    capture_roots: Mapping[str, Path | None],
) -> Path:
    room_family = native.room_family_from_plan(visual_plans["v0"])
    room_id = str(
        ((visual_plans["v0"].get("resources") or {}).get("room_package") or {}).get("room_id")
        or (visual_plans["v0"].get("resources") or {}).get("room_id")
        or visual_plans["v0"].get("scene", {}).get("room_id")
        or ""
    )
    if not room_id:
        raise ReferenceNativeError("reference plan lacks room_id")
    members = []
    for member_id in ("v0_a0", "v0_a1", "v1_a0", "v1_a1"):
        visual_id, audio_id = member_id.split("_")
        row = {
            "member_id": member_id,
            "plan_path": str(member_plans[member_id].resolve()),
            "interventions": {
                "visual_variant": visual_id,
                "audio_variant": audio_id,
                "visual_intervention": "silent_reference_static_position_only",
                "audio_intervention": "timing_only_preserve_actor_clip_and_endpoint",
            },
        }
        if capture_roots.get(visual_id) is not None:
            row["native_capture_root"] = str(capture_roots[visual_id].resolve())
        members.append(row)
    group = {
        "group_id": group_id,
        "world_id": world_id,
        "task_family": REFERENCE_TASK_FAMILY,
        "room_family": room_family,
        "room_id": room_id,
        "split": "pilot",
        "query": deepcopy(config["query"]),
        "profile": {
            "source_count": 4,
            "speaking_count": 3,
            "silent_reference_count": 1,
            "camera_motion": "static",
            "reserve_tail_s": 3.0,
            "post_assembly_convolution_gain": 0.5,
            "candidate_scope_en": config["visual_selector"]["candidate_scope_en"],
            "candidate_scope_zh": config["visual_selector"]["candidate_scope_zh"],
            "minimum_margin_px": config["visual_selector"]["minimum_margin_px"],
            "appearance_review_required": True,
            "appearance_review_authority": "actual_native_appearance_review",
            "query_truth_authority": "private_selector_proof_and_actual_activity_intervals",
        },
        "members": members,
        "comparisons": [
            {
                "members": ["v0_a0", "v0_a1"],
                "shared_modality": "video",
                "answer_relation": "different",
                "kind": "necessity",
            },
            {
                "members": ["v1_a0", "v1_a1"],
                "shared_modality": "video",
                "answer_relation": "different",
                "kind": "necessity",
            },
            {
                "members": ["v0_a0", "v1_a0"],
                "shared_modality": "audio",
                "answer_relation": "different",
                "kind": "necessity",
            },
            {
                "members": ["v0_a1", "v1_a1"],
                "shared_modality": "audio",
                "answer_relation": "different",
                "kind": "necessity",
            },
        ],
    }
    return _write(
        output / "group_spec.json",
        {
            "schema": "avengine_binding_group_spec_v1",
            "planning_status": "research_candidate",
            "request": deepcopy(dict(request)),
            "groups": [group],
        },
    )


def plan_reference_group(
    *,
    base_plan_path: str | Path,
    request_path: str | Path,
    output_root: str | Path,
    second_plan_path: str | Path | None = None,
    v0_capture_root: str | Path | None = None,
    v1_capture_root: str | Path | None = None,
    pixel_centers_by_variant: Mapping[str, Mapping[str, Any]] | None = None,
    group_id: str = "visual_conditioned_reference_group_v1",
    world_id: str = "world_visual_conditioned_reference_0001",
) -> dict[str, Any]:
    """Write a CPU-only four-member reference-group plan.

    If a v1 plan is not supplied, its only visual change is the explicit
    source4 position in request.reference_positions_m. Pixel centers must come
    from actual native truth or a separately supplied CPU planning fixture.
    """
    request_file = Path(request_path).expanduser().resolve()
    base_plan_file = _plan_path(base_plan_path, owner="base plan")
    request = _load(request_file)
    config = _request_config(request)
    registry = _registry(request)
    base_plan = _normalise_planner_audio_metadata(_load(base_plan_file))
    _validate_visual_plan(base_plan, request, label="v0")
    scope = validate_candidate_scope(request, plan=base_plan, registry=registry)

    positions = config["reference_positions_m"]
    if positions is not None:
        base_reference = _state_map(base_plan["visual_plan"]["frames"][0])[
            config["reference_actor_id"]
        ]["root_transform"].get("translation_m")
        if not _same(base_reference, positions["v0"]):
            raise ReferenceNativeError(
                "base v0 reference position differs from request.reference_positions_m.v0"
            )
    if second_plan_path is not None:
        second_file = _plan_path(second_plan_path, owner="v1 plan")
        v1_plan = _normalise_planner_audio_metadata(_load(second_file))
        if positions is not None:
            v1_reference = _state_map(v1_plan["visual_plan"]["frames"][0])[
                config["reference_actor_id"]
            ]["root_transform"].get("translation_m")
            if not _same(v1_reference, positions["v1"]):
                raise ReferenceNativeError(
                    "supplied v1 reference position differs from request.reference_positions_m.v1"
                )
    else:
        if positions is None:
            raise ReferenceNativeError(
                "v1 plan is absent; request must provide explicit reference_positions_m"
            )
        v1_plan = _clone_reference_variant(
            base_plan,
            reference_actor_id=config["reference_actor_id"],
            position=positions["v1"],
            variant="v1",
        )
    _validate_visual_plan(v1_plan, request, label="v1")
    visual_relation = compare_reference_visual_plans(base_plan, v1_plan)

    centers = dict(pixel_centers_by_variant or {})
    if "v0" not in centers:
        if v0_capture_root is None:
            raise ReferenceNativeError(
                "v0 pixel centers must come from actual native truth or a CPU planning fixture"
            )
        centers["v0"] = _pixel_centers_from_truth(
            _capture_path(v0_capture_root, owner="v0 capture"),
            frame_index=config["reference_time_s"],
            owner="v0",
        )
    if "v1" not in centers:
        if v1_capture_root is None:
            raise ReferenceNativeError(
                "v1 pixel centers must come from actual native truth or a CPU planning fixture"
            )
        centers["v1"] = _pixel_centers_from_truth(
            _capture_path(v1_capture_root, owner="v1 capture"),
            frame_index=config["reference_time_s"],
            owner="v1",
        )
    selector_kind = config["visual_selector"]["kind"]
    if selector_kind == REFERENCE_NAMED_QUERY_KIND:
        selector_proofs = {
            variant: select_two_nearest_to_named_reference(
                centers[variant],
                reference_actor_id=config["reference_actor_id"],
                reference_appearance=config["visual_selector"]["reference_appearance"],
                minimum_margin_px=config["visual_selector"]["minimum_margin_px"],
                expected_pair=config["expected_pairs_by_variant"][variant],
            )
            for variant in ("v0", "v1")
        }
    else:
        selector_proofs = {
            variant: select_two_nearest_to_leftmost(
                centers[variant],
                minimum_margin_px=config["visual_selector"]["minimum_margin_px"],
                expected_pair=config["expected_pairs_by_variant"][variant],
            )
            for variant in ("v0", "v1")
        }
    if (
        selector_proofs["v0"]["reference_actor_id"] != config["reference_actor_id"]
        or selector_proofs["v1"]["reference_actor_id"] != config["reference_actor_id"]
    ):
        if selector_kind == REFERENCE_NAMED_QUERY_KIND:
            raise ReferenceNativeError(
                "named selector did not resolve the private source4 reference in both variants"
            )
        raise ReferenceNativeError(
            "source4 must be the unique leftmost visible reference in both variants"
        )

    starts = config["audio_start_times_s"]
    if starts is None:
        raise ReferenceNativeError(
            "request must provide explicit audio_start_times_s for CPU timing preflight"
        )
    scheduled = {}
    schedules = {}
    requests = {}
    configured_pairs = _configured_reference_pairs(config)
    for audio_variant, target_pair in (
        ("a0", configured_pairs["v0"]),
        ("a1", configured_pairs["v1"]),
    ):
        for visual_variant, visual_plan in (("v0", base_plan), ("v1", v1_plan)):
            plan, rebound_request, schedule = schedule_reference_audio_plan(
                visual_plan,
                request,
                audio_variant=audio_variant,
                start_times_s_by_actor=starts[audio_variant],
                target_pair=target_pair,
            )
            plan["reference_visual_variant"] = visual_variant
            plan["reference_audio_variant"] = audio_variant
            scheduled[f"{visual_variant}_{audio_variant}"] = plan
            requests[f"{visual_variant}_{audio_variant}"] = rebound_request
            schedules[f"{visual_variant}_{audio_variant}"] = schedule

    audio_columns = {}
    for audio_variant in ("a0", "a1"):
        left = scheduled[f"v0_{audio_variant}"]
        right = scheduled[f"v1_{audio_variant}"]
        same = _audio_column_signature(left) == _audio_column_signature(right)
        audio_columns[audio_variant] = {
            "status": "pass" if same else "blocked",
            "same_timing_clip_endpoint_plan": same,
            "native_pcm_equality": "not_run",
            "claim_boundary": "actual PCM equality requires four independent native finalizer runs",
        }
        if not same:
            raise ReferenceNativeError(
                f"audio column {audio_variant} differs between visual variants"
            )

    output = Path(output_root).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ReferenceNativeError(f"refusing existing output root: {output}")
    output.mkdir(parents=True)
    (output / "requests").mkdir()
    (output / "plans").mkdir()
    (output / "selector").mkdir()
    _write(output / "provenance_started.json", {
        "schema": "avengine_binding_group_reference_provenance_v1",
        "status": "running",
        "repository": str(native.REPOSITORY.resolve()),
        "base_plan": str(base_plan_file),
        "v1_plan": str(_plan_path(second_plan_path, owner="v1 plan")) if second_plan_path is not None else None,
        "request": str(request_file),
        "group_id": group_id,
        "world_id": world_id,
        "task_family": REFERENCE_TASK_FAMILY,
        "source_asset_ids": list(config["source_asset_ids"]),
        "speaking_actor_ids": list(config["speaking_actor_ids"]),
        "reference_actor_id": config["reference_actor_id"],
        "query": deepcopy(config["query"]),
        "native_execution": "not_run",
        "rlr_execution": "not_run",
        "started_at": datetime.now(timezone.utc).isoformat(),
    })
    member_paths = {}
    for member_id in ("v0_a0", "v0_a1", "v1_a0", "v1_a1"):
        visual_id, audio_id = member_id.split("_")
        member_paths[member_id] = _write(
            output / "plans" / f"{member_id}.json",
            scheduled[member_id],
        )
        _write(output / "requests" / f"{member_id}_request.json", requests[member_id])
    for variant in ("v0", "v1"):
        _write(output / "selector" / f"{variant}_proof.json", selector_proofs[variant])
    capture_roots = {
        "v0": Path(v0_capture_root).expanduser().resolve() if v0_capture_root is not None else None,
        "v1": Path(v1_capture_root).expanduser().resolve() if v1_capture_root is not None else None,
    }
    group_spec = _write_group_spec(
        output,
        group_id=group_id,
        world_id=world_id,
        request=request,
        config=config,
        visual_plans={"v0": base_plan, "v1": v1_plan},
        member_plans=member_paths,
        capture_roots=capture_roots,
    )
    summary = {
        "schema": "avengine_binding_group_reference_summary_v1",
        "status": "research_candidate",
        "group_id": group_id,
        "world_id": world_id,
        "task_family": REFERENCE_TASK_FAMILY,
        "native_execution": "not_run",
        "rlr_execution": "not_run",
        "room_family": native.room_family_from_plan(base_plan),
        "room_id": str(
            ((base_plan.get("resources") or {}).get("room_package") or {}).get("room_id")
            or (base_plan.get("resources") or {}).get("room_id")
            or (base_plan.get("scene") or {}).get("room_id")
        ),
        "request": str(request_file),
        "query": deepcopy(config["query"]),
        "candidate_scope": scope,
        "appearance_review": {
            "status": "not_run",
            "authority": "actual_native_appearance_review",
            "required_for_public_object_naming": True,
        },
        "visual_plan_relation": visual_relation,
        "selector_proofs": {
            variant: str((output / "selector" / f"{variant}_proof.json").resolve())
            for variant in ("v0", "v1")
        },
        "audio_schedules": schedules,
        "audio_columns": audio_columns,
        "member_plans": {
            member_id: str(path.resolve()) for member_id, path in member_paths.items()
        },
        "group_spec": str(group_spec.resolve()),
        "four_native_members": {
            member_id: {
                "native_visual": "not_run",
                "native_audio": "not_run",
                "independent_rlr_required": True,
            }
            for member_id in member_paths
        },
        "claim_boundary": "CPU planning candidate only; no native visual/audio, actual PCM, appearance review, QA validity, human answerability, model evaluation, or formal admission claim",
    }
    _write(output / "summary.json", summary)
    return summary


def _native_visual_descriptor(capture_root: str | Path) -> dict[str, Any]:
    capture = _capture_path(capture_root, owner="native capture")
    required = (
        "neutral_readback.json",
        "pixel_visibility_truth.json",
        "native_pixel_masks_depth_authority_v1.npz",
        "research_receipt.json",
    )
    missing = [name for name in required if not (capture / name).is_file()]
    if missing:
        raise ReferenceNativeError(
            f"native capture lacks required reference evidence {missing}: {capture}"
        )
    frame_readbacks = capture / "frame_readbacks.json"
    if not frame_readbacks.is_file():
        frame_readbacks = capture / "frame_records.json"
    if not frame_readbacks.is_file():
        raise ReferenceNativeError(
            f"native capture lacks frame readbacks: {capture}"
        )
    return {
        "output": str(capture.parent),
        "plan": str(capture.parent / "plan/episode_plan.json"),
        "capture": str(capture),
        "neutral_readback": str((capture / "neutral_readback.json").resolve()),
        "frame_readbacks": str(frame_readbacks.resolve()),
        "visual_video": (
            str((capture / "ue_visual_only.mp4").resolve())
            if (capture / "ue_visual_only.mp4").is_file()
            else None
        ),
        "reused": True,
    }


def _pcm_signature(path: str | Path) -> tuple[Any, ...]:
    """Read exact RIFF audio payloads, including native IEEE FLOAT WAVs."""
    import struct

    value = Path(path).expanduser().resolve()
    try:
        with value.open("rb") as stream:
            if stream.read(4) != b"RIFF":
                raise OSError("missing RIFF header")
            stream.read(4)  # RIFF container size is not part of PCM identity.
            if stream.read(4) != b"WAVE":
                raise OSError("missing WAVE header")
            fmt = None
            payload = None
            while True:
                header = stream.read(8)
                if not header:
                    break
                if len(header) != 8:
                    raise OSError("truncated RIFF chunk header")
                chunk_id, chunk_size = struct.unpack("<4sI", header)
                chunk = stream.read(int(chunk_size))
                if len(chunk) != int(chunk_size):
                    raise OSError("truncated RIFF chunk payload")
                if chunk_size & 1:
                    stream.read(1)
                if chunk_id == b"fmt " and len(chunk) >= 16:
                    fmt = struct.unpack("<HHIIHH", chunk[:16])
                elif chunk_id == b"data":
                    payload = bytes(chunk)
                if fmt is not None and payload is not None:
                    break
        if fmt is None or payload is None:
            raise OSError("WAV lacks fmt/data chunks")
        audio_format, channels, sample_rate, _byte_rate, block_align, bits = fmt
        if channels < 1 or sample_rate < 1 or block_align < 1 or bits < 1:
            raise OSError("WAV format is invalid")
        if block_align % channels or len(payload) % block_align:
            raise OSError("WAV data alignment is invalid")
        # Preserve the historical sample-width slot while accepting PCM
        # integer (format 1) and IEEE FLOAT (format 3) payloads.
        if audio_format not in (1, 3):
            raise OSError(f"unsupported WAV format: {audio_format}")
        sample_width = int(bits // 8)
        if sample_width <= 0:
            raise OSError("WAV sample width is invalid")
        frames = len(payload) // int(block_align)
        return (
            int(channels),
            sample_width,
            int(sample_rate),
            int(frames),
            payload,
        )
    except (OSError, struct.error) as exc:
        raise ReferenceNativeError(f"cannot read native PCM output: {value}: {exc}") from exc


def compare_reference_audio_pcm(
    left: str | Path,
    right: str | Path,
) -> dict[str, Any]:
    """Compare exact native PCM payloads after independent finalizer runs."""
    left_signature = _pcm_signature(left)
    right_signature = _pcm_signature(right)
    same = left_signature == right_signature
    return {
        "status": "pass" if same else "fail",
        "same": same,
        "left": str(Path(left).expanduser().resolve()),
        "right": str(Path(right).expanduser().resolve()),
        "channels": left_signature[0],
        "sample_width": left_signature[1],
        "sample_rate_hz": left_signature[2],
        "frame_count": left_signature[3],
        "claim_boundary": "actual PCM equality only; no perceptual or model equivalence claim",
    }


def _existing_reference_audio_result(root: Path) -> dict[str, Any] | None:
    """Read one completed member for narrow finalization resume.

    A member is reusable only when its own result and all public payload paths
    are present. Partial roots return ``None`` so the current finalizer can
    materialize that member afresh without touching completed siblings.
    """
    if root.is_symlink() or not root.is_dir():
        return None
    delivery = root / "delivery"
    result_path = delivery / "result.json"
    if not result_path.is_file():
        return None
    try:
        result = _load(result_path)
    except (OSError, TypeError, ValueError):
        return None
    if result.get("status") not in {"research_only", "pass"}:
        return None
    paths = {
        "facts": result.get("facts") or delivery / "facts.json",
        "questions": result.get("questions_path") or delivery / "questions.json",
        "audio": result.get("lossless_stereo_wav"),
        "visual_video": result.get("visual_video"),
        "preview": result.get("preview"),
    }
    resolved: dict[str, str] = {}
    for key, raw in paths.items():
        if isinstance(raw, Path):
            value = raw.expanduser().resolve()
        elif isinstance(raw, str) and raw.strip():
            value = Path(raw).expanduser().resolve()
        elif key in {"facts", "questions"}:
            value = Path(raw).expanduser().resolve()
        else:
            return None
        if not value.is_file():
            return None
        resolved[key] = str(value)
    report = delivery / "research_report.json"
    if not report.is_file():
        return None
    return {
        "result": result,
        "elapsed_s": 0.0,
        "facts": resolved["facts"],
        "questions": resolved["questions"],
        "audio": resolved["audio"],
        "audio_report": str(report.resolve()),
        "visual_video": resolved["visual_video"],
        "preview": resolved["preview"],
        "reused": True,
    }


def finalize_reference_group_from_captures(
    *,
    plan_root: str | Path,
    v0_capture_root: str | Path,
    v1_capture_root: str | Path,
    output_root: str | Path,
    group_id: str | None = None,
    world_id: str | None = None,
    resume_existing: bool = False,
) -> dict[str, Any]:
    """Finalize four independent audio members from two completed visual captures.

    This continuation consumes the CPU plan written by plan_reference_group.
    It never starts UE; callers provide completed native visual captures. Each
    member is sent through the existing materializer/finalizer independently,
    then the two visual variants in each audio column are compared by exact
    PCM. Assembly is attempted only after those actual checks pass.
    """
    source = Path(plan_root).expanduser().resolve()
    if not source.is_dir():
        raise ReferenceNativeError(f"reference CPU plan root is unavailable: {source}")
    summary_path = source / "summary.json"
    group_spec_path = source / "group_spec.json"
    if not summary_path.is_file() or not group_spec_path.is_file():
        raise ReferenceNativeError(
            "reference CPU plan root must contain summary.json and group_spec.json"
        )
    summary = _load(summary_path)
    if summary.get("status") != "research_candidate":
        raise ReferenceNativeError("reference CPU plan summary is not a candidate")
    request_path = Path(summary.get("request", "")).expanduser().resolve()
    if not request_path.is_file():
        raise ReferenceNativeError(f"reference request is unavailable: {request_path}")
    request = _load(request_path)
    config = _request_config(request)
    v0_capture = _native_visual_descriptor(v0_capture_root)
    v1_capture = _native_visual_descriptor(v1_capture_root)
    plan_paths = {
        member_id: source / "plans" / f"{member_id}.json"
        for member_id in ("v0_a0", "v0_a1", "v1_a0", "v1_a1")
    }
    request_paths = {
        member_id: source / "requests" / f"{member_id}_request.json"
        for member_id in plan_paths
    }
    if any(not path.is_file() for path in (*plan_paths.values(), *request_paths.values())):
        raise ReferenceNativeError("reference CPU plan is missing member plans or requests")
    endpoint_v0 = native._neutral_endpoint_bindings(
        v0_capture["neutral_readback"],
        plan=_load(plan_paths["v0_a0"]),
    )
    endpoint_v1 = native._neutral_endpoint_bindings(
        v1_capture["neutral_readback"],
        plan=_load(plan_paths["v1_a0"]),
    )
    if (
        set(endpoint_v0) != set(REFERENCE_ACTOR_IDS)
        or set(endpoint_v1) != set(REFERENCE_ACTOR_IDS)
        or any(
            endpoint_v0.get(actor_id) != endpoint_v1.get(actor_id)
            for actor_id in REFERENCE_ACTOR_IDS
        )
    ):
        raise ReferenceNativeError(
            "visual variants lack complete or matching native source endpoint identities"
        )
    final_output = Path(output_root).expanduser().resolve()
    if final_output.exists() or final_output.is_symlink():
        if not resume_existing or final_output.is_symlink() or not final_output.is_dir():
            raise ReferenceNativeError(f"refusing existing reference finalization root: {final_output}")
    else:
        final_output.mkdir(parents=True)
    variants_root = final_output / "variants"
    if variants_root.exists() and variants_root.is_symlink():
        raise ReferenceNativeError(f"refusing symlinked variants root: {variants_root}")
    variants_root.mkdir(exist_ok=True)
    variants = {}
    variant_roots: dict[str, Path] = {}
    reused_members = []
    pending_audio_tasks: dict[str, dict[str, Any]] = {}
    for member_id in ("v0_a0", "v0_a1", "v1_a0", "v1_a1"):
        visual_id = member_id.split("_", 1)[0]
        capture = v0_capture if visual_id == "v0" else v1_capture
        member_plan = _load(plan_paths[member_id])
        member_request = _load(request_paths[member_id])
        member_root = variants_root / member_id
        candidate_roots = [member_root]
        if resume_existing:
            candidate_roots.extend(
                sorted(
                    variants_root.glob(f"{member_id}_retry_*"),
                    key=lambda path: path.name,
                    reverse=True,
                )
            )
        reused = None
        for candidate_root in candidate_roots:
            candidate_result = (
                _existing_reference_audio_result(candidate_root)
                if resume_existing
                else None
            )
            if candidate_result is None:
                continue
            # Validate a completed member against the current source plan
            # before accepting it during a continuation.
            assigned, _ = build_reference_audio_assignment_plan(
                member_plan,
                member_request,
                member_id.rsplit("_", 1)[1],
                endpoint_by_actor=endpoint_v0,
            )
            old_plan_path = candidate_root / "plan/episode_plan.json"
            old_capture = candidate_root / "capture"
            if not old_plan_path.is_file() or not old_capture.is_symlink():
                continue
            old_plan = _load(old_plan_path)
            old_events = {
                (
                    str(row.get("event_id")),
                    str(row.get("actor_id")),
                    str(row.get("source_endpoint_id")),
                    str(row.get("sound_asset_id")),
                )
                for row in old_plan.get("audio_events", [])
                if isinstance(row, Mapping)
            }
            old_bindings_rows = [
                row for row in old_plan.get("voice_bindings", [])
                if isinstance(row, Mapping)
            ]
            old_bindings = {
                (
                    str(row.get("event_id")),
                    str(row.get("actor_id")),
                    str(row.get("source_endpoint_id")),
                    str(row.get("sound_asset_id")),
                )
                for row in old_bindings_rows
            }
            expected_events = {
                (
                    str(row.get("event_id")),
                    str(row.get("actor_id")),
                    str(row.get("source_endpoint_id")),
                    str(row.get("sound_asset_id")),
                )
                for row in assigned.get("audio_events", [])
                if isinstance(row, Mapping)
            }
            expected_capture = Path(capture["capture"]).resolve()
            current_binding_ids = [row.get("event_id") for row in old_bindings_rows]
            binding_identity_ok = (
                len(old_bindings_rows) == len(old_bindings) == len(expected_events)
                and all(isinstance(value, str) and value for value in current_binding_ids)
                and old_bindings == expected_events
            )
            if (
                old_events == expected_events
                and binding_identity_ok
                and old_capture.resolve() == expected_capture
            ):
                member_root = candidate_root
                reused = candidate_result
                break
        if reused is not None:
            variants[member_id] = reused
            variant_roots[member_id] = member_root
            reused_members.append(member_id)
            continue
        if resume_existing and (member_root.exists() or member_root.is_symlink()):
            # Preserve an incomplete or stale member root. The current
            # finalizer writes a fresh sibling and the old partial evidence
            # remains available for diagnosis.
            for index in range(1, 100):
                candidate_root = member_root.with_name(f"{member_id}_retry_{index:02d}")
                if not candidate_root.exists() and not candidate_root.is_symlink():
                    member_root = candidate_root
                    break
            else:
                raise ReferenceNativeError(
                    f"unable to allocate member retry root: {member_root}"
                )
        pending_audio_tasks[member_id] = {
            "root": member_root,
            "request": member_request,
        }
        variant_roots[member_id] = member_root
    if pending_audio_tasks:
        variants.update(
            native.finalize_audio_assignments(
                pending_audio_tasks,
                max_workers=2,
            )
        )
    pcm_by_column = {}
    for audio_variant in ("a0", "a1"):
        pcm = compare_reference_audio_pcm(
            variants[f"v0_{audio_variant}"]["audio"],
            variants[f"v1_{audio_variant}"]["audio"],
        )
        pcm_by_column[audio_variant] = pcm
        if not pcm["same"]:
            raise ReferenceNativeError(
                f"independently rendered audio column {audio_variant} differs in actual PCM"
            )
    native_spec = _load(group_spec_path)
    group = native_spec["groups"][0]
    member_rows = {row["member_id"]: row for row in group.get("members", [])}
    for member_id, result in variants.items():
        row = member_rows.get(member_id)
        if not isinstance(row, Mapping):
            raise ReferenceNativeError(f"group spec lacks member {member_id}")
        row = dict(row)
        visual_id = member_id.split("_", 1)[0]
        visual = v0_capture if visual_id == "v0" else v1_capture
        variant_root = variant_roots.get(member_id)
        if variant_root is None:
            raise ReferenceNativeError(f"missing finalized variant root for {member_id}")
        row.update(
            plan_path=str((variant_root / "plan/episode_plan.json").resolve()),
            request_path=str((variant_root / "request.json").resolve()),
            facts_path=str(Path(result["facts"]).resolve()),
            video_path=str(Path(result["visual_video"]).resolve()),
            audio_path=str(Path(result["audio"]).resolve()),
            native_capture_root=str(Path(visual["capture"]).resolve()),
        )
        member_rows[member_id] = row
    group["members"] = [member_rows[member_id] for member_id in ("v0_a0", "v0_a1", "v1_a0", "v1_a1")]
    if group_id is not None:
        group["group_id"] = str(group_id)
    if world_id is not None:
        group["world_id"] = str(world_id)
    native_spec["planning_status"] = "native_finalized"
    native_spec["groups"] = [group]
    native_spec_path = final_output / "group_spec_native.json"
    if native_spec_path.exists() or native_spec_path.is_symlink():
        if not resume_existing:
            raise ReferenceNativeError(f"refusing existing native group spec: {native_spec_path}")
        for index in range(1, 100):
            candidate = final_output / f"group_spec_native_retry_{index:02d}.json"
            if not candidate.exists() and not candidate.is_symlink():
                native_spec_path = candidate
                break
        else:
            raise ReferenceNativeError(f"unable to allocate group spec retry path: {final_output}")
    _write(native_spec_path, native_spec)
    assembled_output = final_output / "assembled"
    if resume_existing and assembled_output.exists():
        for index in range(1, 100):
            candidate = final_output / f"assembled_retry_{index:02d}"
            if not candidate.exists() and not candidate.is_symlink():
                assembled_output = candidate
                break
        else:
            raise ReferenceNativeError(f"unable to allocate fresh assembly retry path: {final_output}")
    try:
        from avengine.qa.binding_groups import assemble_binding_dataset
        assembled = assemble_binding_dataset(
            native_spec,
            input_base=final_output,
            output=assembled_output,
            seed=str(group.get("group_id") or "reference"),
            verify_media=True,
        )
    except Exception as exc:
        raise ReferenceNativeError(
            f"reference native members finalized but assembly failed: {type(exc).__name__}: {exc}"
        ) from exc
    final_summary = {
        "schema": "avengine_binding_group_reference_native_summary_v1",
        "status": "pass",
        "group_id": group.get("group_id"),
        "world_id": group.get("world_id"),
        "task_family": REFERENCE_TASK_FAMILY,
        "native_execution": "reused_completed_visual_captures",
        "rlr_execution": "four_independent_finalizer_runs",
        "request": str(request_path),
        "source_cpu_plan_root": str(source),
        "native_visual_captures": {
            "v0": v0_capture,
            "v1": v1_capture,
        },
        "endpoint_identity_equivalence": {
            "status": "pass",
            "same_by_actor": True,
            "actors": endpoint_v0,
        },
        "pcm_by_column": pcm_by_column,
        "group_spec": str(native_spec_path.resolve()),
        "assembled": str((assembled_output / "binding_groups.json").resolve()),
        "reused_completed_members": list(reused_members),
        "assembled_sample_count": assembled["sample_count"],
        "validation": assembled["validation"],
        "claim_boundary": "native visual/audio and binding evidence only; human answerability, model evaluation, and formal admission remain separate",
    }
    summary_output = final_output / "summary.json"
    if summary_output.exists() or summary_output.is_symlink():
        if not resume_existing:
            raise ReferenceNativeError(f"refusing existing native final summary: {summary_output}")
        for index in range(1, 100):
            candidate = final_output / f"summary_retry_{index:02d}.json"
            if not candidate.exists() and not candidate.is_symlink():
                summary_output = candidate
                break
        else:
            raise ReferenceNativeError(f"unable to allocate summary retry path: {final_output}")
    final_summary["summary_path"] = str(summary_output.resolve())
    _write(summary_output, final_summary)
    return final_summary


def prepare_reference_group(
    *,
    execute_native: bool = False,
    plan_root: str | Path | None = None,
    v0_capture_root: str | Path | None = None,
    v1_capture_root: str | Path | None = None,
    output_root: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Plan a group, or finalize one CPU plan from completed captures.

    Native execution is never implicit. Set execute_native=True only when the
    caller supplies an existing CPU plan root and two completed native capture
    roots; this path starts no UE process.
    """
    if not execute_native:
        if output_root is None:
            raise ReferenceNativeError("output_root is required for CPU reference planning")
        return plan_reference_group(output_root=output_root, **kwargs)
    if plan_root is None or v0_capture_root is None or v1_capture_root is None or output_root is None:
        raise ReferenceNativeError(
            "native finalization requires plan_root, v0_capture_root, v1_capture_root and output_root"
        )
    return finalize_reference_group_from_captures(
        plan_root=plan_root,
        v0_capture_root=v0_capture_root,
        v1_capture_root=v1_capture_root,
        output_root=output_root,
        group_id=kwargs.get("group_id"),
        world_id=kwargs.get("world_id"),
    )


__all__ = [
    "REFERENCE_TASK_FAMILY",
    "REFERENCE_QUERY_KIND",
    "REFERENCE_NAMED_QUERY_KIND",
    "REFERENCE_QUERY_KINDS",
    "REFERENCE_SPEAKING_ACTOR_IDS",
    "REFERENCE_ACTOR_IDS",
    "REFERENCE_REFERENCE_ACTOR_ID",
    "ReferenceNativeError",
    "validate_candidate_scope",
    "select_two_nearest_to_leftmost",
    "select_two_nearest_to_named_reference",
    "_pixel_centers_from_truth",
    "compare_reference_visual_plans",
    "schedule_reference_audio_plan",
    "build_reference_audio_assignment_plan",
    "finalize_reference_audio_variant",
    "compare_reference_audio_pcm",
    "finalize_reference_group_from_captures",
    "sample_reference_visual_positions",
    "sample_reference_audio_starts",
    "plan_reference_family_matrix",
    "plan_reference_group",
    "prepare_reference_group",
]

def _normalise_planner_audio_metadata(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Bridge planner output that omits explicit full-clip/endpoint fields.

    The conditioned planner selects a complete prepared clip but leaves
    source_start/source_end and endpoint identity for the native executor.
    This bridge records the full clip and the registered slot/anchor endpoint
    without changing audio bytes or event timing.
    """
    cloned = deepcopy(dict(plan))
    actors = _plan_actors(cloned)
    events = [
        row for row in cloned.get("audio_events", []) if isinstance(row, Mapping)
    ]
    bindings = [
        row for row in cloned.get("voice_bindings", []) if isinstance(row, Mapping)
    ]
    bindings_by_sound_actor = {}
    for row in bindings:
        if not row.get("sound_asset_id"):
            continue
        key = (str(row["sound_asset_id"]), str(row.get("actor_id") or ""))
        bindings_by_sound_actor.setdefault(key, row)
    normalised_events = []
    for raw in events:
        event = deepcopy(dict(raw))
        actor_id = str(event.get("actor_id") or "")
        if actor_id not in actors:
            raise ReferenceNativeError(
                f"planner audio event references unknown actor: {actor_id}"
            )
        sample_count = event.get("sample_count")
        if (
            isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count <= 0
        ):
            raise ReferenceNativeError(
                "planner audio event lacks positive sample_count"
            )
        event.setdefault("source_start_sample", 0)
        event.setdefault("source_end_sample_exclusive", int(sample_count))
        if event.get("source_start_sample") is None:
            event["source_start_sample"] = 0
        if event.get("source_end_sample_exclusive") is None:
            event["source_end_sample_exclusive"] = int(sample_count)
        endpoint = event.get("source_endpoint_id")
        actor_binding = actors[actor_id].get("emitter_binding")
        if not isinstance(endpoint, str) or not endpoint.strip():
            endpoint = (
                actors[actor_id].get("source_endpoint_id")
                or (
                    actor_binding.get("source_endpoint_id")
                    if isinstance(actor_binding, Mapping)
                    else None
                )
            )
        if not isinstance(endpoint, str) or not endpoint.strip():
            anchor = (
                actor_binding.get("semantic_anchor_id")
                if isinstance(actor_binding, Mapping)
                else None
            )
            if not isinstance(anchor, str) or not anchor.strip():
                raise ReferenceNativeError(
                    f"planner event {event.get('event_id')} lacks registered endpoint anchor"
                )
            endpoint = f"{actor_id}_{anchor.strip()}"
        event["source_endpoint_id"] = endpoint.strip()
        sound_id = str(event.get("sound_asset_id") or "")
        binding = bindings_by_sound_actor.get((sound_id, actor_id))
        if binding is None:
            binding = next(
                (
                    row for row in bindings
                    if str(row.get("sound_asset_id") or "") == sound_id
                ),
                None,
            )
        if binding is not None:
            binding_endpoint = binding.get("source_endpoint_id")
            if not isinstance(binding_endpoint, str) or not binding_endpoint.strip():
                binding["source_endpoint_id"] = event["source_endpoint_id"]
            if binding.get("source_start_sample") is None:
                binding["source_start_sample"] = 0
            if binding.get("source_end_sample_exclusive") is None:
                binding["source_end_sample_exclusive"] = int(sample_count)
        normalised_events.append(event)
    cloned["audio_events"] = normalised_events
    cloned["voice_bindings"] = [deepcopy(dict(row)) for row in bindings]
    cloned["planner_metadata_bridge"] = {
        "status": "pass",
        "full_clip_fields_explicit": True,
        "endpoint_fields_explicit": True,
        "authority": "registered_plan_actor_emitter_anchor",
        "audio_bytes_changed": False,
    }
    return cloned


def _reference_layout_config(request: Mapping[str, Any]) -> dict[str, Any]:
    recipe = request.get("reference_recipe")
    recipe = recipe if isinstance(recipe, Mapping) else {}
    raw = recipe.get("reference_layout", request.get("reference_layout"))
    raw = _mapping(raw, "reference_layout")
    required = (
        "minimum_entity_separation_m",
        "visibility_margin_deg",
        "same_floor_tolerance_m",
        "candidate_retry_budget",
        "minimum_variant_displacement_m",
        "visibility_probe_offset_m",
        "position_source",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise ReferenceNativeError(
            f"reference_layout lacks explicit configuration: {missing}"
        )
    value = {
        "minimum_entity_separation_m": _finite_number(
            raw["minimum_entity_separation_m"],
            "reference_layout.minimum_entity_separation_m",
            nonnegative=True,
        ),
        "visibility_margin_deg": _finite_number(
            raw["visibility_margin_deg"],
            "reference_layout.visibility_margin_deg",
            nonnegative=True,
        ),
        "same_floor_tolerance_m": _finite_number(
            raw["same_floor_tolerance_m"],
            "reference_layout.same_floor_tolerance_m",
            nonnegative=True,
        ),
        "minimum_variant_displacement_m": _finite_number(
            raw["minimum_variant_displacement_m"],
            "reference_layout.minimum_variant_displacement_m",
            nonnegative=True,
        ),
        "position_source": str(raw["position_source"]).strip(),
        "visibility_probe_offset_m": _position(
            raw["visibility_probe_offset_m"],
            "reference_layout.visibility_probe_offset_m",
        ),
    }
    budget = raw["candidate_retry_budget"]
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ReferenceNativeError(
            "reference_layout.candidate_retry_budget must be a positive integer"
        )
    value["candidate_retry_budget"] = int(budget)
    if value["position_source"] not in {
        "native_pathfinder",
        "native_route_bank_or_pathfinder",
    }:
        raise ReferenceNativeError(
            "reference_layout.position_source must name native_pathfinder or native_route_bank_or_pathfinder"
        )
    return value


def _rotate_offset_from_state(state: Mapping[str, Any]) -> list[float]:
    root = state.get("root_transform")
    emitter = state.get("planned_emitter_m")
    if not isinstance(root, Mapping) or not isinstance(root.get("translation_m"), Sequence):
        raise ReferenceNativeError("actor state lacks root translation")
    if not isinstance(emitter, Sequence) or len(emitter) != 3:
        raise ReferenceNativeError("actor state lacks planned emitter")
    return [
        float(emitter[index]) - float(root["translation_m"][index])
        for index in range(3)
    ]


def _camera_projected_center(
    camera: Mapping[str, Any],
    point: Sequence[float],
    *,
    visibility_margin_deg: float,
) -> tuple[list[float] | None, dict[str, Any]]:
    try:
        origin = [float(item) for item in camera["position_m"]]
        forward = [float(item) for item in camera["basis"]["forward"]]
        right = [float(item) for item in camera["basis"]["right"]]
        up = [float(item) for item in camera["basis"]["up"]]
        height, width = [int(item) for item in camera["resolution_hw"]]
        fov = float(camera["horizontal_fov_deg"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReferenceNativeError("static camera lacks projection fields") from exc
    delta = [float(point[index]) - origin[index] for index in range(3)]
    depth = sum(delta[index] * forward[index] for index in range(3))
    side = sum(delta[index] * right[index] for index in range(3))
    vertical = sum(delta[index] * up[index] for index in range(3))
    bearing = math.degrees(math.atan2(side, depth)) if depth > 0.0 else math.inf
    if depth <= 0.0 or abs(bearing) >= fov / 2.0 - visibility_margin_deg:
        return None, {"depth_m": depth, "bearing_deg": bearing, "visible": False}
    scale = 1.0 / (2.0 * math.tan(math.radians(fov) / 2.0))
    center = [
        width / 2.0 + side / depth * width * scale,
        height / 2.0 - vertical / depth * height * scale,
    ]
    return center, {"depth_m": depth, "bearing_deg": bearing, "visible": True}


def _candidate_positions_from_native_space(
    space: Any,
    *,
    rng: Any,
    retry_budget: int,
) -> tuple[list[Any], str]:
    import numpy as np

    route_bank = space.route_bank()
    if route_bank:
        positions = []
        for route in route_bank:
            points = route.get("points_m") if isinstance(route, Mapping) else None
            if points is None:
                continue
            points = np.asarray(points, dtype=float)
            if points.ndim != 2 or points.shape[1] != 3 or not len(points):
                continue
            for point in (points[0], points[-1]):
                if not any(float(np.linalg.norm(point - other)) <= 1.0e-8 for other in positions):
                    positions.append(point)
        rng.shuffle(positions)
        return positions, "native_route_bank_endpoint"
    bounds = space.bounds()
    positions = []
    for _ in range(int(retry_budget)):
        try:
            point = np.asarray(space.sample_navigable(rng, bounds), dtype=float)
        except (TypeError, ValueError):
            continue
        if point.shape == (3,) and not any(float(np.linalg.norm(point - other)) <= 1.0e-8 for other in positions):
            positions.append(point)
    return positions, "native_pathfinder_sample"


def sample_reference_visual_positions(
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    """Sample two silent-reference placements from the existing native solver."""
    import numpy as np

    config = _request_config(request)
    layout = _reference_layout_config(request)
    actors = _validate_visual_plan(
        _normalise_planner_audio_metadata(plan),
        request,
        label="reference-layout-base",
    )
    from avengine.capture.qa_plan_adapters import load_planning_resources
    from avengine.qa.answerability import line_of_sight

    try:
        space, mesh, _ = load_planning_resources(plan["resources"], request)
    except Exception as exc:
        raise ReferenceNativeError(
            f"cannot load native room planning resources: {type(exc).__name__}: {exc}"
        ) from exc
    rng = np.random.default_rng(int(seed if seed is not None else plan.get("seed", 0)))
    candidates, source = _candidate_positions_from_native_space(
        space, rng=rng, retry_budget=layout["candidate_retry_budget"]
    )
    if not candidates:
        raise ReferenceNativeError("native room planner yielded no candidate reference positions")
    visual = plan["visual_plan"]
    camera = visual["camera"]
    records = _registry(request)
    reference_asset_id = config["source_asset_ids"][REFERENCE_ACTOR_IDS.index(config["reference_actor_id"])]
    reference_record = records.get(reference_asset_id)
    if reference_record is None:
        raise ReferenceNativeError(
            f"silent reference asset is absent from source registry: {reference_asset_id}"
        )
    habitat_runtime = (
        reference_record.get("runtime_backends", {}).get("habitat", {})
        if isinstance(reference_record.get("runtime_backends"), Mapping)
        else {}
    )
    resting_pose = (
        habitat_runtime.get("resting_pose")
        if isinstance(habitat_runtime, Mapping)
        else {}
    )
    attachment = resting_pose.get("attachment_surface") if isinstance(resting_pose, Mapping) else None
    if attachment is not None and str(attachment).lower() not in {"floor", "ground"}:
        raise ReferenceNativeError(
            f"silent reference placement requires floor attachment; registry declares {attachment!r}"
        )
    states = _state_map(visual["frames"][0])
    speaker_roots = {
        actor_id: np.asarray(states[actor_id]["root_transform"]["translation_m"], dtype=float)
        for actor_id in REFERENCE_SPEAKING_ACTOR_IDS
    }
    reference_offset = np.asarray(_rotate_offset_from_state(states[config["reference_actor_id"]]), dtype=float)
    floor_y = float(np.median([point[1] for point in speaker_roots.values()]))
    camera_origin = np.asarray(camera["position_m"], dtype=float)
    probe_offset = np.asarray(layout["visibility_probe_offset_m"], dtype=float)
    minimum_separation = layout["minimum_entity_separation_m"]
    visibility_margin = layout["visibility_margin_deg"]

    def evaluate(point: Any, expected_pair: Sequence[str]) -> tuple[Any, dict[str, Any]] | None:
        point = np.asarray(point, dtype=float)
        if abs(float(point[1]) - floor_y) > layout["same_floor_tolerance_m"]:
            return None
        if any(float(np.linalg.norm(point - root)) < minimum_separation for root in speaker_roots.values()):
            return None
        emitter = point + reference_offset
        if not bool(space.is_navigable(point)):
            return None
        if line_of_sight(mesh, camera_origin, emitter) != "clear":
            return None
        if line_of_sight(mesh, camera_origin, point + probe_offset) != "clear":
            return None
        centers = {}
        projection = {}
        for actor_id in REFERENCE_SPEAKING_ACTOR_IDS:
            center, detail = _camera_projected_center(
                camera,
                states[actor_id]["planned_emitter_m"],
                visibility_margin_deg=visibility_margin,
            )
            if center is None:
                return None
            centers[actor_id] = center
            projection[actor_id] = detail
        center, detail = _camera_projected_center(
            camera,
            emitter,
            visibility_margin_deg=visibility_margin,
        )
        if center is None:
            return None
        centers[config["reference_actor_id"]] = center
        projection[config["reference_actor_id"]] = detail
        try:
            if config["visual_selector"]["kind"] == REFERENCE_NAMED_QUERY_KIND:
                proof = select_two_nearest_to_named_reference(
                    centers,
                    reference_actor_id=config["reference_actor_id"],
                    reference_appearance=config["visual_selector"]["reference_appearance"],
                    minimum_margin_px=config["visual_selector"]["minimum_margin_px"],
                    expected_pair=expected_pair,
                )
            else:
                proof = select_two_nearest_to_leftmost(
                    centers,
                    minimum_margin_px=config["visual_selector"]["minimum_margin_px"],
                    expected_pair=expected_pair,
                )
        except ReferenceNativeError:
            return None
        if proof["reference_actor_id"] != config["reference_actor_id"]:
            return None
        return point, {
            "position_m": [float(value) for value in point],
            "emitter_m": [float(value) for value in emitter],
            "pixel_centers_px": centers,
            "projection": projection,
            "selector_proof": proof,
            "placement_authority": source,
            "minimum_entity_separation_m": minimum_separation,
            "same_floor_tolerance_m": layout["same_floor_tolerance_m"],
            "visibility_margin_deg": visibility_margin,
        }

    v0 = None
    v1 = None
    for candidate in candidates:
        evaluated = evaluate(
            candidate,
            config["expected_pairs_by_variant"]["v0"],
        )
        if evaluated is not None:
            v0 = evaluated
            break
    if v0 is None:
        raise ReferenceNativeError(
            "native room planner yielded no legal v0 reference placement for source1/source2"
        )
    for candidate in candidates:
        if float(np.linalg.norm(np.asarray(candidate) - v0[0])) < layout["minimum_variant_displacement_m"]:
            continue
        evaluated = evaluate(
            candidate,
            config["expected_pairs_by_variant"]["v1"],
        )
        if evaluated is not None:
            v1 = evaluated
            break
    if v1 is None:
        raise ReferenceNativeError(
            "native room planner yielded no legal v1 reference placement for source1/source3"
        )
    return {
        "status": "pass",
        "reference_actor_id": config["reference_actor_id"],
        "position_source": source,
        "v0": v0[1],
        "v1": v1[1],
        "minimum_variant_displacement_m": layout["minimum_variant_displacement_m"],
        "candidate_count": len(candidates),
        "candidate_retry_budget": layout["candidate_retry_budget"],
        "visibility_authority": "CPU native pathfinder/route authority plus planned emitter/root line-of-sight; native pixel centroid remains not_run",
    }


def _planner_module() -> Any:
    import importlib.util

    path = native.REPOSITORY / "tools/studio/run_qa_episode.py"
    spec = importlib.util.spec_from_file_location("reference_run_qa_episode", path)
    if spec is None or spec.loader is None:
        raise ReferenceNativeError(f"cannot load native CPU planner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pool_sound_ids_for_actor(
    request: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, Any]],
    actor_id: str,
) -> list[str]:
    """Return the current pool's sound IDs compatible with one registered actor.

    The conditioned sampler still performs the final duration, sample-rate, and
    activity checks. This allowlist only prevents the native planner from
    drawing a biologically or object-class incompatible sound for the actor.
    """
    if actor_id not in REFERENCE_SPEAKING_ACTOR_IDS:
        raise ReferenceNativeError(
            f"sound allowlist actor must be a speaking actor: {actor_id}"
        )
    assets = request.get("source_asset_ids")
    if (
        isinstance(assets, (str, bytes))
        or not isinstance(assets, Sequence)
        or len(assets) != len(REFERENCE_ACTOR_IDS)
    ):
        raise ReferenceNativeError(
            "reference request must declare four source_asset_ids before sound selection"
        )
    actor_index = REFERENCE_ACTOR_IDS.index(actor_id)
    asset_id = assets[actor_index]
    if not isinstance(asset_id, str) or not asset_id.strip():
        raise ReferenceNativeError(
            f"reference request has no registered asset for {actor_id}"
        )
    record = registry.get(asset_id)
    if not isinstance(record, Mapping):
        raise ReferenceNativeError(
            f"reference request asset is absent from source registry: {asset_id}"
        )
    actor = {
        "actor_id": actor_id,
        "asset_id": asset_id,
        "entity_class": record.get("entity_class"),
        "identity": deepcopy(dict(record.get("identity") or {})),
        "realized_attributes": deepcopy(
            dict(record.get("realized_attributes") or {})
        ),
    }
    pool_path = _file(
        request.get("sound_pool"), base=native.REPOSITORY, owner="sound pool"
    )
    pool = _load(pool_path)
    sounds = pool.get("sounds")
    if not isinstance(sounds, list):
        raise ReferenceNativeError("sound pool.sounds must be a list")
    from avengine.rooms.conditioned_sampler import sound_matches

    selected = []
    for sound in sounds:
        if not isinstance(sound, Mapping):
            continue
        sound_id = sound.get("sound_asset_id")
        if not isinstance(sound_id, str) or not sound_id.strip():
            continue
        try:
            matches = bool(sound_matches(actor, sound))
        except (KeyError, TypeError, ValueError):
            matches = False
        if matches:
            selected.append(sound_id)
    result = sorted(set(selected))
    if not result:
        raise ReferenceNativeError(
            f"sound pool has no compatible sound for {actor_id} asset {asset_id}"
        )
    return result

def _request_for_reference_family(
    template: Mapping[str, Any],
    *,
    episode_id: str,
    room_id: str,
    seed: int,
    source_asset_ids: Sequence[str],
    scope_bindings: Mapping[str, Mapping[str, Any]],
    scope_en: str,
    scope_zh: str,
    minimum_margin_px: float,
    reference_layout: Mapping[str, Any],
    expected_pairs_by_variant: Mapping[str, Sequence[str]] | None = None,
    selector_kind: str = REFERENCE_QUERY_KIND,
    reference_appearance: Mapping[str, Any] | None = None,
    prepared_manifest: str | Path | None = None,
) -> dict[str, Any]:
    request = deepcopy(dict(template))
    request.update(
        episode_id=episode_id,
        room_id=room_id,
        seed=int(seed),
        source_asset_ids=list(source_asset_ids),
        frame_count=150,
        frame_rate_hz=15,
        sample_rate_hz=16000,
        qa_ids=["QA-05"],
        sampling_policy="conditioned_static_v2",
        source_context_policy="independent_states",
        reference_time_s=0,
        window_s=[4, 6],
        reference_actor_id="source4",
        speaking_actor_ids=list(REFERENCE_SPEAKING_ACTOR_IDS),
        expected_pairs_by_variant={
            "v0": list(
                expected_pairs_by_variant.get("v0", _DEFAULT_REFERENCE_PAIRS["v0"])
            )
            if isinstance(expected_pairs_by_variant, Mapping)
            else list(_DEFAULT_REFERENCE_PAIRS["v0"]),
            "v1": list(
                expected_pairs_by_variant.get("v1", _DEFAULT_REFERENCE_PAIRS["v1"])
            )
            if isinstance(expected_pairs_by_variant, Mapping)
            else list(_DEFAULT_REFERENCE_PAIRS["v1"]),
        },
        candidate_scope_bindings=deepcopy(dict(scope_bindings)),
        reference_layout=deepcopy(dict(reference_layout)),
    )
    request["entities"] = {
        **dict(request.get("entities") or {}),
        "total_count": 4,
        "silent_count": 1,
    }
    request["camera"] = {
        **dict(request.get("camera") or {}),
        "motion": "static",
    }
    if prepared_manifest is not None:
        request["prepared_manifest"] = str(
            _file(prepared_manifest, base=native.REPOSITORY, owner="prepared manifest")
        )
    elif not request.get("prepared_manifest"):
        pool_path = _file(
            request.get("sound_pool"),
            base=native.REPOSITORY,
            owner="sound pool",
        )
        pool = _load(pool_path)
        rows = pool.get("sounds") if isinstance(pool.get("sounds"), list) else []
        manifests = {
            str(row.get("source_metadata_manifest"))
            for row in rows
            if isinstance(row, Mapping)
            and isinstance(row.get("source_metadata_manifest"), str)
            and Path(str(row["source_metadata_manifest"])).is_file()
        }
        if len(manifests) == 1:
            request["prepared_manifest"] = next(iter(manifests))
        elif len(manifests) > 1:
            raise ReferenceNativeError(
                "multiple prepared manifests are present in current sound pool; pass prepared_manifest explicitly"
            )
    request["profile"] = {
        **dict(request.get("profile") or {}),
        "anchor_count": 1,
        "event_relation": "overlap",
        "speech_motion": "all_still",
        "reserve_tail_s": 3.0,
    }
    if selector_kind not in REFERENCE_QUERY_KINDS:
        raise ReferenceNativeError(
            f"unsupported reference selector kind: {selector_kind}"
        )
    visual_selector = {
        "kind": selector_kind,
        "minimum_margin_px": float(minimum_margin_px),
        "candidate_scope_en": str(scope_en),
        "candidate_scope_zh": str(scope_zh),
    }
    if selector_kind == REFERENCE_NAMED_QUERY_KIND:
        visual_selector["reference_appearance"] = _normalise_reference_appearance(
            reference_appearance,
            "visual_selector.reference_appearance",
        )
    elif reference_appearance is not None:
        raise ReferenceNativeError(
            "reference_appearance is only valid for the named reference selector"
        )
    request["visual_selector"] = visual_selector
    sound_selection = dict(request.get("sound_selection") or {})
    registry = _registry(request)
    sound_selection["preallocated_sound_asset_ids_by_actor"] = {
        actor_id: _pool_sound_ids_for_actor(request, registry, actor_id)
        for actor_id in REFERENCE_SPEAKING_ACTOR_IDS
    }
    request["sound_selection"] = sound_selection
    if selector_kind == REFERENCE_NAMED_QUERY_KIND:
        _validate_named_reference_scope(_request_config(request), registry)
    return request


def _reference_speaking_seeds(
    request: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    seed_start: int,
    search_limit: int,
) -> list[int]:
    from avengine.rooms.conditioned_sampler import resolve_condition_profile

    result = []
    for offset in range(int(search_limit)):
        seed = int(seed_start) + offset
        candidate = dict(request)
        candidate["seed"] = seed
        profile = resolve_condition_profile(candidate, {"assets": list(registry.values())})
        if profile.get("speaking_indices") == [0, 1, 2]:
            result.append(seed)
    return result


def plan_reference_family_matrix(
    *,
    base_request_path: str | Path,
    output_root: str | Path,
    family_matrix: Mapping[str, Mapping[str, Any]],
    room_ids: Sequence[str],
    minimum_margin_px: float,
    prepared_manifest: str | Path | None = None,
    seed_start: int = 202609110000,
    planner_seed_search_limit: int = 256,
    planner_attempt_budget: int = 8,
) -> dict[str, Any]:
    """Create bounded CPU candidate requests/plans for source-family combinations.

    The existing conditioned native room planner supplies each four-object
    layout. Only the silent reference is subsequently moved using the same
    native pathfinder/route space; no position is invented by this producer.
    """
    if (
        isinstance(room_ids, (str, bytes))
        or not isinstance(room_ids, Sequence)
        or not room_ids
        or any(not isinstance(room_id, str) or not room_id.strip() for room_id in room_ids)
    ):
        raise ReferenceNativeError("room_ids must contain one or more room IDs")
    if (
        isinstance(planner_attempt_budget, bool)
        or not isinstance(planner_attempt_budget, int)
        or planner_attempt_budget <= 0
    ):
        raise ReferenceNativeError("planner_attempt_budget must be a positive integer")
    margin = _finite_number(minimum_margin_px, "minimum_margin_px")
    if margin < 0.0:
        raise ReferenceNativeError("minimum_margin_px must be nonnegative")
    template_path = Path(base_request_path).expanduser().resolve()
    template = _load(template_path)
    records = _registry(template)
    output = Path(output_root).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ReferenceNativeError(f"refusing existing reference matrix root: {output}")
    output.mkdir(parents=True)
    rows = []

    for family_index, (family_name, raw_spec) in enumerate(family_matrix.items()):
        spec = _mapping(raw_spec, f"family_matrix.{family_name}")
        speaking_assets = spec.get("speaking_source_asset_ids")
        if (
            isinstance(speaking_assets, (str, bytes))
            or not isinstance(speaking_assets, Sequence)
            or len(speaking_assets) != 3
            or len(set(speaking_assets)) != 3
        ):
            raise ReferenceNativeError(
                f"family_matrix.{family_name}.speaking_source_asset_ids must contain three distinct assets"
            )
        reference_asset = spec.get("reference_source_asset_id")
        if not isinstance(reference_asset, str) or not reference_asset.strip():
            raise ReferenceNativeError(
                f"family_matrix.{family_name}.reference_source_asset_id is required"
            )
        assets = [str(item) for item in speaking_assets] + [reference_asset]
        scope_bindings = _mapping(
            spec.get("candidate_scope_bindings"),
            f"family_matrix.{family_name}.candidate_scope_bindings",
        )
        scope_en = spec.get("candidate_scope_en")
        scope_zh = spec.get("candidate_scope_zh")
        if (
            not isinstance(scope_en, str)
            or not scope_en.strip()
            or not isinstance(scope_zh, str)
            or not scope_zh.strip()
        ):
            raise ReferenceNativeError(
                f"family_matrix.{family_name} must provide bilingual candidate scope text"
            )
        layout = _mapping(
            spec.get("reference_layout"),
            f"family_matrix.{family_name}.reference_layout",
        )
        selector_kind = str(
            spec.get("visual_selector_kind", spec.get("selector_kind", REFERENCE_QUERY_KIND))
        )
        reference_appearance = spec.get("reference_appearance")
        expected_pairs_by_variant = spec.get(
            "expected_pairs_by_variant",
            _DEFAULT_REFERENCE_PAIRS,
        )
        if selector_kind not in REFERENCE_QUERY_KINDS:
            raise ReferenceNativeError(
                f"family_matrix.{family_name}.selector_kind is unsupported: {selector_kind}"
            )
        if selector_kind == REFERENCE_NAMED_QUERY_KIND:
            reference_appearance = _normalise_reference_appearance(
                reference_appearance,
                f"family_matrix.{family_name}.reference_appearance",
            )
        elif reference_appearance is not None:
            raise ReferenceNativeError(
                f"family_matrix.{family_name}.reference_appearance requires named selector"
            )
        for room_index, room_id in enumerate(room_ids):
            candidate_id = f"{family_name}_{room_id}"
            candidate_root = output / candidate_id
            if candidate_root.exists() or candidate_root.is_symlink():
                raise ReferenceNativeError(f"refusing existing candidate root: {candidate_root}")
            candidate_root.mkdir(parents=True)
            planner_template = {
                **template,
                "source_asset_ids": assets,
                "entities": {
                    **dict(template.get("entities") or {}),
                    "total_count": 4,
                    "silent_count": 1,
                },
                "camera": {**dict(template.get("camera") or {}), "motion": "static"},
            }
            seed_cursor = int(seed_start) + family_index * 1009 + room_index * 101
            candidate_seeds = _reference_speaking_seeds(
                planner_template,
                records,
                seed_start=seed_cursor,
                search_limit=planner_seed_search_limit,
            )[:planner_attempt_budget]
            row = {
                "candidate_id": candidate_id,
                "family_name": str(family_name),
                "room_id": str(room_id),
                "selector_kind": selector_kind,
                "reference_appearance": deepcopy(reference_appearance)
                if reference_appearance is not None
                else None,
                "expected_pairs_by_variant": deepcopy(expected_pairs_by_variant),
                "reference_layout": deepcopy(layout),
                "native_execution": "not_run",
                "rlr_execution": "not_run",
                "status": "blocked",
                "attempts": [],
            }
            if not candidate_seeds:
                row["reason"] = (
                    f"native planner did not find source4-silent seed in "
                    f"{planner_seed_search_limit} candidates"
                )
                rows.append(row)
                continue
            for attempt_index, request_seed in enumerate(candidate_seeds):
                attempt_root = candidate_root / f"attempt_{attempt_index:02d}"
                attempt_root.mkdir(parents=True)
                try:
                    request = _request_for_reference_family(
                        template,
                        episode_id=f"reference_{family_name}_{room_id}_v2_attempt_{attempt_index:02d}",
                        room_id=room_id,
                        seed=request_seed,
                        source_asset_ids=assets,
                        scope_bindings=scope_bindings,
                        scope_en=scope_en,
                        scope_zh=scope_zh,
                        minimum_margin_px=margin,
                        reference_layout=layout,
                        expected_pairs_by_variant=expected_pairs_by_variant,
                        selector_kind=selector_kind,
                        reference_appearance=reference_appearance,
                        prepared_manifest=prepared_manifest,
                    )
                    planner_request_path = attempt_root / "planner_request.json"
                    _write(planner_request_path, request)
                    raw_output = attempt_root / "native_plan"
                    command = [
                        sys.executable,
                        str(native.REPOSITORY / "tools/studio/run_qa_episode.py"),
                        "--request",
                        str(planner_request_path),
                        "--output",
                        str(raw_output),
                        "--plan-only",
                    ]
                    process = native._run(
                        command,
                        log=attempt_root / "native_plan.log",
                        label=f"{candidate_id}_attempt_{attempt_index:02d}_plan",
                    )
                    row["attempts"].append({
                        "attempt_index": attempt_index,
                        "seed": request_seed,
                        "planner_request": str(planner_request_path),
                        "native_plan_output": str(raw_output),
                        "native_plan_process": process,
                        "status": "planner_pass",
                    })
                    raw_plan_path = raw_output / "plan/episode_plan.json"
                    if not raw_plan_path.is_file():
                        raise ReferenceNativeError(
                            "existing native planner did not produce episode_plan.json"
                        )
                    raw_plan = _normalise_planner_audio_metadata(_load(raw_plan_path))
                    profile = (
                        raw_plan.get("condition_profile")
                        if isinstance(raw_plan.get("condition_profile"), Mapping)
                        else {}
                    )
                    if profile.get("speaking_indices") != [0, 1, 2]:
                        raise ReferenceNativeError(
                            "native planner returned a non-source4-silent profile"
                        )
                    positions = sample_reference_visual_positions(
                        raw_plan, request, seed=request_seed + 1009
                    )
                    v0_plan = _clone_reference_variant(
                        raw_plan,
                        reference_actor_id="source4",
                        position=positions["v0"]["position_m"],
                        variant="v0",
                    )
                    v0_plan_path = _write(attempt_root / "v0_base_plan.json", v0_plan)
                    request["reference_positions_m"] = {
                        "v0": positions["v0"]["position_m"],
                        "v1": positions["v1"]["position_m"],
                    }
                    audio_starts = sample_reference_audio_starts(
                        v0_plan,
                        request,
                        seed=request_seed + 2027,
                        retry_budget=int(layout.get("candidate_retry_budget", 200)),
                    )
                    request["audio_start_times_s"] = audio_starts["start_times_s_by_variant"]
                    request_path = attempt_root / "request.json"
                    _write(request_path, request)
                    group_output = attempt_root / "reference_group"
                    planned = plan_reference_group(
                        base_plan_path=v0_plan_path,
                        request_path=request_path,
                        output_root=group_output,
                        pixel_centers_by_variant={
                            "v0": positions["v0"]["pixel_centers_px"],
                            "v1": positions["v1"]["pixel_centers_px"],
                        },
                        group_id=f"reference_{family_name}_{room_id}_v2",
                        world_id=f"world_{family_name}_{room_id}_v2",
                    )
                    row.update({
                        "status": "pass",
                        "request": str(request_path),
                        "native_plan_output": str(raw_output),
                        "reference_group_output": str(group_output),
                        "reference_summary": str(group_output / "summary.json"),
                        "reference_positions": positions,
                        "audio_starts": audio_starts,
                        "plan_status": planned["status"],
                        "successful_attempt_index": attempt_index,
                    })
                    row.pop("reason", None)
                    row["attempts"][-1]["status"] = "pass"
                    break
                except Exception as exc:
                    row["attempts"].append({
                        "attempt_index": attempt_index,
                        "seed": request_seed,
                        "status": "blocked",
                        "reason": f"{type(exc).__name__}: {exc}",
                    })
                    row["reason"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    summary = {
        "schema": "avengine_reference_family_matrix_summary_v1",
        "status": "research_candidate",
        "native_execution": "not_run",
        "rlr_execution": "not_run",
        "base_request": str(template_path),
        "minimum_margin_px": margin,
        "planner_attempt_budget": int(planner_attempt_budget),
        "reference_position_retry_budget": sorted({
            int(row["reference_layout"]["candidate_retry_budget"])
            for row in rows
            if isinstance(row.get("reference_layout"), Mapping)
        }),
        "selector_kinds": sorted({
            str(row["selector_kind"])
            for row in rows
            if row.get("selector_kind")
        }),
        "expected_pairs_by_variant": (
            deepcopy(rows[0]["expected_pairs_by_variant"])
            if rows and isinstance(rows[0].get("expected_pairs_by_variant"), Mapping)
            else None
        ),
        "room_ids": [str(room_id) for room_id in room_ids],
        "candidate_count": len(rows),
        "passed_plan_candidates": sum(row["status"] == "pass" for row in rows),
        "blocked_plan_candidates": sum(row["status"] == "blocked" for row in rows),
        "candidates": rows,
        "claim_boundary": "CPU planning candidates only; native pixel centroids, native media, actual PCM, appearance review, QA validity and formal admission remain not_run",
    }
    _write(output / "summary.json", summary)
    return summary

def sample_reference_audio_starts(
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    seed: int,
    retry_budget: int,
) -> dict[str, Any]:
    """Sample legal timing-only columns from complete per-actor clip activity."""
    import numpy as np

    if isinstance(retry_budget, bool) or not isinstance(retry_budget, int) or retry_budget <= 0:
        raise ReferenceNativeError("audio schedule retry budget must be a positive integer")
    config = _request_config(request)
    events = _audio_events_by_actor(_normalise_planner_audio_metadata(plan))
    clock = _clock(plan)
    sr = int(clock["sample_rate_hz"])
    reserve = int(round(3.0 * sr))
    rng = np.random.default_rng(int(seed))
    starts = {}
    attempts = 0
    configured_pairs = _configured_reference_pairs(config)
    targets = {
        "a0": configured_pairs["v0"],
        "a1": configured_pairs["v1"],
    }
    max_by_actor = {
        actor_id: int(clock["sample_count"]) - reserve - int(events[actor_id]["sample_count"])
        for actor_id in REFERENCE_SPEAKING_ACTOR_IDS
    }
    if any(value < 0 for value in max_by_actor.values()):
        raise ReferenceNativeError("complete clips cannot fit the fixed 3-second tail")
    for audio_variant, target_pair in targets.items():
        found = None
        for _ in range(retry_budget):
            attempts += 1
            candidate = {
                actor_id: int(rng.integers(max_by_actor[actor_id] + 1)) / sr
                for actor_id in REFERENCE_SPEAKING_ACTOR_IDS
            }
            try:
                _, _, schedule = schedule_reference_audio_plan(
                    plan,
                    request,
                    audio_variant=audio_variant,
                    start_times_s_by_actor=candidate,
                    target_pair=target_pair,
                )
            except ReferenceNativeError:
                continue
            found = {
                "start_times_s": candidate,
                "schedule": schedule,
            }
            break
        if found is None:
            raise ReferenceNativeError(
                f"no legal {audio_variant} activity schedule in {retry_budget} attempts"
            )
        starts[audio_variant] = found
    return {
        "status": "pass",
        "seed": int(seed),
        "attempts": attempts,
        "start_times_s_by_variant": {
            variant: dict(value["start_times_s"])
            for variant, value in starts.items()
        },
        "schedules_by_variant": {
            variant: value["schedule"] for variant, value in starts.items()
        },
        "authority": "uniform_seeded_sample_over_native_clock_start_budget_then_actual_clip_activity_intersection",
    }



