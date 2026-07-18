"""Strict M5.1 source/event contracts migrated from legacy AVEngine semantics.

The legacy SPEAR spike represented clip labels as flat booleans.  M5.1 keeps
the useful 12-label definition domain while making source identity, event
activity, scope, unknown state, provenance, and evidence explicit.  This is a
research-only manifest; it neither admits assets nor makes a qualification
claim.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator

from avengine.contracts.json_io import canonical_json_sha256, load_json, sha256_file
from avengine.m5_1.orientation import habitat_basis_from_yaw_degrees


SOURCE_MANIFEST_SCHEMA = "avengine_m5_1_source_manifest_v1"
SCHEMA_FILENAME = "m5_1_source_manifest_v1.schema.json"

DEFAULT_TICKS_PER_FRAME = 3_200
DEFAULT_TICKS_PER_SAMPLE = 3

# The order is inherited from the legacy flag definition registry and is part
# of the public manifest contract.
SOURCE_FLAG_IDS = (
    "occluded_by_furniture",
    "occluded_by_wall",
    "never_occluded",
    "leaves_camera_fov",
    "stays_in_camera_fov",
    "crosses_azimuth_zero",
    "passes_close_to_mic",
    "far_from_mic_whole_clip",
    "stationary",
    "steady_walk",
    "stop_and_go",
)
PAIR_FLAG_IDS = ("sources_pass_each_other",)
ALL_FLAG_IDS = SOURCE_FLAG_IDS + PAIR_FLAG_IDS

# These reproduce the old clip-level aggregation semantics while preserving
# ``not_evaluated`` instead of silently coercing it to false.
OR_AGGREGATED_FLAGS = frozenset(
    {
        "occluded_by_furniture",
        "occluded_by_wall",
        "leaves_camera_fov",
        "crosses_azimuth_zero",
        "passes_close_to_mic",
        "stationary",
        "stop_and_go",
        "sources_pass_each_other",
    }
)
AND_AGGREGATED_FLAGS = frozenset(
    {
        "never_occluded",
        "stays_in_camera_fov",
        "far_from_mic_whole_clip",
        "steady_walk",
    }
)

_STATUS_VALUE = {
    "present": True,
    "absent": False,
    "not_evaluated": None,
}


class SourceContractError(ValueError):
    """One or more M5.1 source-manifest invariants failed."""

    def __init__(self, errors: Iterable[str]):
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))


def sample_boundary(
    frame_boundary_index: int,
    *,
    ticks_per_frame: int = DEFAULT_TICKS_PER_FRAME,
    ticks_per_sample: int = DEFAULT_TICKS_PER_SAMPLE,
) -> int:
    """Return the exact nearest-sample boundary for a frame boundary."""

    if (
        isinstance(frame_boundary_index, bool)
        or not isinstance(frame_boundary_index, int)
        or frame_boundary_index < 0
    ):
        raise ValueError("frame_boundary_index must be a non-negative integer")
    if (
        isinstance(ticks_per_frame, bool)
        or not isinstance(ticks_per_frame, int)
        or ticks_per_frame <= 0
        or isinstance(ticks_per_sample, bool)
        or not isinstance(ticks_per_sample, int)
        or ticks_per_sample <= 0
    ):
        raise ValueError("ticks_per_frame and ticks_per_sample must be positive integers")
    return (
        ticks_per_frame * frame_boundary_index + ticks_per_sample // 2
    ) // ticks_per_sample


def _schema_path() -> Path:
    source = Path(__file__).resolve().parents[3] / "schemas" / SCHEMA_FILENAME
    installed = Path(sys.prefix) / "share" / "avengine" / "schemas" / SCHEMA_FILENAME
    path = source if source.is_file() else installed
    if not path.is_file():
        raise FileNotFoundError(f"AVEngine schema is unavailable: {SCHEMA_FILENAME}")
    return path


def _schema_errors(value: Any) -> list[str]:
    schema = load_json(_schema_path())
    result: list[str] = []
    for error in sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    ):
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        result.append(f"JSON Schema {location}: {error.message}")
    return result


def _all_numbers_finite(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_all_numbers_finite(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(_all_numbers_finite(item) for item in value)
    return False


def _mapping_list(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _canonical_without(value: Mapping[str, Any], field: str) -> str | None:
    try:
        return canonical_json_sha256(
            {key: item for key, item in value.items() if key != field}
        )
    except (TypeError, ValueError):
        return None


def _stable_values(
    values: Sequence[Mapping[str, Any]],
    field: str,
    owner: str,
    errors: list[str],
) -> list[str]:
    result = [item.get(field) for item in values]
    strings = [item for item in result if isinstance(item, str)]
    if len(strings) != len(set(strings)):
        errors.append(f"{owner}.{field} values must be unique")
    return strings


def _assessment_errors(
    value: Any,
    *,
    owner: str,
    expected_scope: str,
) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{owner} must be an assessment mapping"]
    errors: list[str] = []
    if value.get("scope") != expected_scope:
        errors.append(f"{owner}.scope must be {expected_scope!r}")

    status = value.get("status")
    if status in _STATUS_VALUE and value.get("value") is not _STATUS_VALUE[status]:
        errors.append(f"{owner}.value must match status {status!r}")

    evidence = _mapping_list(value.get("evidence"))
    evidence_ids = [item.get("evidence_id") for item in evidence]
    if len(evidence_ids) != len(set(evidence_ids)):
        errors.append(f"{owner}.evidence IDs must be unique within the assessment")
    evidence_kinds = {item.get("kind") for item in evidence}
    if status == "not_evaluated" and not evidence_kinds.intersection(
        {"missing_dependency", "manual_review"}
    ):
        errors.append(
            f"{owner} not_evaluated requires missing_dependency or manual_review evidence"
        )
    if status in {"present", "absent"} and evidence_kinds == {"missing_dependency"}:
        errors.append(
            f"{owner} evaluated status cannot rely only on missing_dependency evidence"
        )
    return errors


def _status(value: Any) -> str | None:
    if isinstance(value, Mapping):
        candidate = value.get("status")
        if candidate in _STATUS_VALUE:
            return str(candidate)
    return None


def _mutual_flag_errors(flags: Mapping[str, Any], owner: str) -> list[str]:
    errors: list[str] = []

    def present(flag_id: str) -> bool:
        return _status(flags.get(flag_id)) == "present"

    def known(flag_id: str) -> bool:
        return _status(flags.get(flag_id)) in {"present", "absent"}

    if present("never_occluded") and (
        present("occluded_by_furniture") or present("occluded_by_wall")
    ):
        errors.append(
            f"{owner} never_occluded conflicts with an occlusion-present flag"
        )
    if known("leaves_camera_fov") and known("stays_in_camera_fov"):
        if present("leaves_camera_fov") == present("stays_in_camera_fov"):
            errors.append(
                f"{owner} leaves_camera_fov and stays_in_camera_fov must be inverse"
            )
    if present("stationary") and (present("steady_walk") or present("stop_and_go")):
        errors.append(f"{owner} stationary conflicts with a locomotion-present flag")
    if present("steady_walk") and present("stop_and_go"):
        errors.append(f"{owner} steady_walk conflicts with stop_and_go")
    if present("passes_close_to_mic") and present("far_from_mic_whole_clip"):
        errors.append(
            f"{owner} passes_close_to_mic conflicts with far_from_mic_whole_clip"
        )
    return errors


def _repository_uri_path(uri: Any) -> Path | None:
    if not isinstance(uri, str) or not uri.startswith("repo://"):
        return None
    root = Path(__file__).resolve().parents[3]
    candidate = (root / uri.removeprefix("repo://")).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _route_binding_errors(
    value: Any,
    keyframes: Sequence[Mapping[str, Any]],
    *,
    frame_count: int,
    owner: str,
) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{owner} must be a route-binding mapping"]
    errors: list[str] = []
    if value.get("point_count") != frame_count or len(keyframes) != frame_count:
        errors.append(f"{owner} and keyframes must bind every clip frame")
    if [item.get("frame_index") for item in keyframes] != list(range(frame_count)):
        errors.append(f"{owner} requires one canonical keyframe per clip frame")

    path = _repository_uri_path(value.get("authority_uri"))
    if path is None or not path.is_file():
        errors.append(f"{owner}.authority_uri must resolve to a repository file")
        return errors
    if value.get("authority_file_sha256") != sha256_file(path):
        errors.append(f"{owner}.authority_file_sha256 differs from the route file")
    try:
        route = load_json(path)
    except (OSError, ValueError) as exc:
        errors.append(f"{owner}.authority_uri cannot be loaded: {exc}")
        return errors
    if not isinstance(route, Mapping):
        errors.append(f"{owner}.authority_uri must contain a route manifest")
        return errors
    if value.get("authority_manifest_content_sha256") != route.get(
        "manifest_content_sha256"
    ):
        errors.append(f"{owner}.authority_manifest_content_sha256 differs")
    if value.get("authority_schema") != route.get("schema"):
        errors.append(f"{owner}.authority_schema differs")
    if value.get("authority_route_id") != route.get("route_id"):
        errors.append(f"{owner}.authority_route_id differs")

    routes = route.get("routes")
    route_key = value.get("route_key")
    record = routes.get(route_key) if isinstance(routes, Mapping) else None
    if not isinstance(record, Mapping):
        errors.append(f"{owner}.route_key does not resolve")
        return errors
    if value.get("route_trajectory_sha256") != record.get(
        "habitat_trajectory_sha256"
    ):
        errors.append(f"{owner}.route_trajectory_sha256 differs")
    points = record.get("habitat_trajectory_m")
    if not isinstance(points, list):
        errors.append(f"{owner} route trajectory is missing")
        return errors
    start = int(value.get("source_frame_start", 0))
    step = int(value.get("source_frame_step", 1))
    count = int(value.get("point_count", 0))
    selected = points[start : start + step * count : step]
    if len(selected) != count:
        errors.append(f"{owner} route frame selection escapes its authority")
        return errors
    offset = float(value.get("emitter_height_offset_m", 0.0))
    for index, (keyframe, point) in enumerate(zip(keyframes, selected)):
        position = keyframe.get("position_m")
        if not (
            isinstance(position, list)
            and len(position) == 3
            and isinstance(point, list)
            and len(point) == 3
            and all(
                math.isclose(
                    float(observed),
                    float(expected) + (offset if axis == 1 else 0.0),
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                for axis, (observed, expected) in enumerate(zip(position, point))
            )
        ):
            errors.append(
                f"{owner} keyframe {index} differs from route plus emitter offset"
            )
            break
    return errors


def _trajectory_errors(value: Any, owner: str, frame_count: int) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{owner} must be a trajectory mapping"]
    errors: list[str] = []
    declared = value.get("trajectory_content_sha256")
    actual = _canonical_without(value, "trajectory_content_sha256")
    if actual is None or declared != actual:
        errors.append(f"{owner}.trajectory_content_sha256 does not match content")

    keyframes = _mapping_list(value.get("keyframes"))
    indices = [item.get("frame_index") for item in keyframes]
    if indices and indices[0] != 0:
        errors.append(f"{owner}.keyframes must start at frame 0")
    if indices and indices[-1] != frame_count - 1:
        errors.append(
            f"{owner}.keyframes must end at clip frame {frame_count - 1}"
        )
    if len(indices) > 1 and any(
        not isinstance(left, int)
        or isinstance(left, bool)
        or not isinstance(right, int)
        or isinstance(right, bool)
        or left >= right
        for left, right in zip(indices, indices[1:])
    ):
        errors.append(f"{owner}.keyframe frame indices must be strictly increasing")
    errors.extend(
        _route_binding_errors(
            value.get("route_binding"),
            keyframes,
            frame_count=frame_count,
            owner=f"{owner}.route_binding",
        )
    )
    return errors


def _event_errors(
    source: Mapping[str, Any],
    *,
    owner: str,
    frame_count: int,
    sample_count: int,
    sample_rate_hz: int,
    ticks_per_frame: int,
    ticks_per_sample: int,
) -> tuple[list[str], list[Mapping[str, Any]]]:
    errors: list[str] = []
    source_id = source.get("source_id")
    source_class = source.get("asset_class")
    expected_event_class = "human_voice" if source_class == "human" else "animal_call"
    taxonomy = (
        source.get("voice_taxonomy")
        if source_class == "human"
        else source.get("call_taxonomy")
    )
    taxonomy_id = taxonomy.get("taxonomy_id") if isinstance(taxonomy, Mapping) else None

    provenance = source.get("provenance")
    audio_assets = (
        _mapping_list(provenance.get("audio_assets"))
        if isinstance(provenance, Mapping)
        else []
    )
    audio_by_id: dict[str, list[Mapping[str, Any]]] = {}
    for asset in audio_assets:
        asset_id = asset.get("asset_id")
        if isinstance(asset_id, str):
            audio_by_id.setdefault(asset_id, []).append(asset)
    if any(len(items) != 1 for items in audio_by_id.values()):
        errors.append(f"{owner}.provenance.audio_assets asset IDs must be unique")

    events = _mapping_list(source.get("event_windows"))
    event_ids = [event.get("event_id") for event in events]
    if len(event_ids) != len(set(event_ids)):
        errors.append(f"{owner}.event_windows event IDs must be unique")

    ordering: list[tuple[Any, Any, Any]] = []
    previous_end: int | None = None
    for index, event in enumerate(events):
        event_owner = f"{owner}.event_windows[{index}]"
        start = event.get("start_frame")
        end = event.get("end_frame_exclusive")
        ordering.append((start, end, event.get("event_id")))
        if not (
            isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
            and 0 <= start < end <= frame_count
        ):
            errors.append(
                f"{event_owner} must satisfy 0 <= start_frame < end_frame "
                f"<= {frame_count}"
            )
        else:
            if event.get("start_sample") != sample_boundary(
                start,
                ticks_per_frame=ticks_per_frame,
                ticks_per_sample=ticks_per_sample,
            ):
                errors.append(
                    f"{event_owner}.start_sample does not match exact boundary"
                )
            if event.get("end_sample_exclusive") != sample_boundary(
                end,
                ticks_per_frame=ticks_per_frame,
                ticks_per_sample=ticks_per_sample,
            ):
                errors.append(
                    f"{event_owner}.end_sample_exclusive does not match exact boundary"
                )
            if event.get("end_sample_exclusive", sample_count + 1) > sample_count:
                errors.append(f"{event_owner} exceeds clip sample_count")
            if previous_end is not None and start < previous_end:
                errors.append(f"{owner}.event_windows must not overlap")
            previous_end = end

        if event.get("source_id") != source_id:
            errors.append(f"{event_owner}.source_id does not match its owner")
        if event.get("event_class") != expected_event_class:
            errors.append(f"{event_owner}.event_class does not match asset_class")
        if event.get("taxonomy_id") != taxonomy_id:
            errors.append(f"{event_owner}.taxonomy_id does not match source taxonomy")

        dry_id = event.get("dry_audio_asset_id")
        candidates = audio_by_id.get(dry_id, []) if isinstance(dry_id, str) else []
        if len(candidates) != 1:
            errors.append(
                f"{event_owner}.dry_audio_asset_id must resolve exactly once in provenance"
            )
        elif event.get("dry_audio_asset_sha256") != candidates[0].get("sha256"):
            errors.append(
                f"{event_owner}.dry_audio_asset_sha256 does not match provenance"
            )
        if len(candidates) == 1:
            program = event.get("audio_program")
            if isinstance(program, Mapping):
                asset = candidates[0]
                source_start = int(program["source_start_sample"])
                source_end = int(program["source_end_sample_exclusive"])
                source_rate = int(program["source_sample_rate_hz"])
                render_rate = int(program["render_sample_rate_hz"])
                if not 0 <= source_start < source_end <= int(asset["sample_count"]):
                    errors.append(f"{event_owner}.audio_program source slice is invalid")
                if source_rate != asset.get("sample_rate_hz"):
                    errors.append(
                        f"{event_owner}.audio_program source rate differs from provenance"
                    )
                if render_rate != sample_rate_hz:
                    errors.append(
                        f"{event_owner}.audio_program render rate differs from clip"
                    )
                source_length = source_end - source_start
                expected_resampled = (
                    source_length * render_rate + source_rate // 2
                ) // source_rate
                if program.get("resampled_content_sample_count") != expected_resampled:
                    errors.append(
                        f"{event_owner}.audio_program resampled length differs"
                    )
                expected_policy = (
                    "none" if source_rate == render_rate else "deterministic_rate_conversion"
                )
                if program.get("resample_policy") != expected_policy:
                    errors.append(
                        f"{event_owner}.audio_program resample policy differs"
                    )
                event_samples = event.get("end_sample_exclusive", 0) - event.get(
                    "start_sample", 0
                )
                if program.get("event_sample_count") != event_samples:
                    errors.append(
                        f"{event_owner}.audio_program event length differs from window"
                    )
                if program.get("tail_padding_samples") != (
                    event_samples - expected_resampled
                ):
                    errors.append(
                        f"{event_owner}.audio_program tail padding does not close window"
                    )

    if ordering and ordering != sorted(ordering):
        errors.append(f"{owner}.event_windows must be in canonical temporal order")
    return errors, events


def _source_errors(
    source: Mapping[str, Any], index: int, clip: Mapping[str, Any]
) -> tuple[list[str], list[Mapping[str, Any]]]:
    owner = f"sources[{index}]"
    errors: list[str] = []
    source_class = source.get("asset_class")
    emitter = source.get("emitter")
    if isinstance(emitter, Mapping):
        expected_anchor = "mouth" if source_class == "human" else "muzzle"
        if emitter.get("semantic_anchor_id") != expected_anchor:
            errors.append(
                f"{owner}.emitter.semantic_anchor_id must be {expected_anchor!r}"
            )
        trajectory = source.get("trajectory")
        if isinstance(trajectory, Mapping) and emitter.get("path_sha256") != trajectory.get(
            "trajectory_content_sha256"
        ):
            errors.append(
                f"{owner}.emitter.path_sha256 must bind the declared emitter trajectory"
            )

    provenance = source.get("provenance")
    visual = provenance.get("visual_asset") if isinstance(provenance, Mapping) else None
    if isinstance(visual, Mapping) and visual.get("asset_id") != source.get("asset_id"):
        errors.append(f"{owner}.provenance.visual_asset.asset_id must match asset_id")

    frame_count = int(clip["frame_count"])
    errors.extend(
        _trajectory_errors(
            source.get("trajectory"), f"{owner}.trajectory", frame_count
        )
    )
    event_errors, events = _event_errors(
        source,
        owner=owner,
        frame_count=frame_count,
        sample_count=int(clip["sample_count"]),
        sample_rate_hz=int(clip["sample_rate_hz"]),
        ticks_per_frame=int(clip["ticks_per_frame"]),
        ticks_per_sample=int(clip["ticks_per_sample"]),
    )
    errors.extend(event_errors)

    flags = source.get("flags")
    if isinstance(flags, Mapping):
        for flag_id in SOURCE_FLAG_IDS:
            errors.extend(
                _assessment_errors(
                    flags.get(flag_id),
                    owner=f"{owner}.flags.{flag_id}",
                    expected_scope="per_source",
                )
            )
        errors.extend(_mutual_flag_errors(flags, f"{owner}.flags"))
    return errors, events


def _expected_frame_events(
    sources: Sequence[Mapping[str, Any]],
    events_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
    frame_index: int,
) -> tuple[dict[str, str | None], list[str]]:
    result: dict[str, str | None] = {}
    errors: list[str] = []
    for source in sources:
        source_id = source.get("source_id")
        if not isinstance(source_id, str):
            continue
        active = [
            event.get("event_id")
            for event in events_by_source.get(source_id, ())
            if isinstance(event.get("start_frame"), int)
            and isinstance(event.get("end_frame_exclusive"), int)
            and event["start_frame"] <= frame_index < event["end_frame_exclusive"]
        ]
        if len(active) > 1:
            errors.append(
                f"source {source_id!r} has multiple active events at frame {frame_index}"
            )
        result[source_id] = active[0] if len(active) == 1 else None
    return result, errors


def _frame_state_errors(
    value: Any,
    sources: Sequence[Mapping[str, Any]],
    events_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
    clip: Mapping[str, Any],
) -> list[str]:
    states = _mapping_list(value)
    errors: list[str] = []
    frame_count = int(clip["frame_count"])
    ticks_per_frame = int(clip["ticks_per_frame"])
    ticks_per_sample = int(clip["ticks_per_sample"])
    if len(states) != frame_count:
        errors.append("frame_event_state length must equal clip.frame_count")
    for index, state in enumerate(states):
        owner = f"frame_event_state[{index}]"
        if state.get("frame_index") != index:
            errors.append(f"{owner}.frame_index must equal its array index")
        if state.get("pts_ticks") != index * ticks_per_frame:
            errors.append(f"{owner}.pts_ticks does not match frame index")
        if state.get("sample_start") != sample_boundary(
            index,
            ticks_per_frame=ticks_per_frame,
            ticks_per_sample=ticks_per_sample,
        ):
            errors.append(f"{owner}.sample_start does not match exact boundary")
        if state.get("sample_end") != sample_boundary(
            index + 1,
            ticks_per_frame=ticks_per_frame,
            ticks_per_sample=ticks_per_sample,
        ):
            errors.append(f"{owner}.sample_end does not match exact boundary")

        expected, active_errors = _expected_frame_events(
            sources, events_by_source, index
        )
        errors.extend(active_errors)
        current = state.get("current_event_by_source")
        if not isinstance(current, Mapping) or dict(current) != expected:
            errors.append(
                f"{owner}.current_event_by_source does not match event windows"
            )
    return errors


def _expected_overlaps(
    source_ids: tuple[str, str],
    events_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for left in events_by_source.get(source_ids[0], ()):
        for right in events_by_source.get(source_ids[1], ()):
            left_start = left.get("start_frame")
            right_start = right.get("start_frame")
            left_end = left.get("end_frame_exclusive")
            right_end = right.get("end_frame_exclusive")
            if not all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in (left_start, right_start, left_end, right_end)
            ):
                continue
            start = max(left_start, right_start)
            end = min(left_end, right_end)
            if start < end:
                result.append(
                    {
                        "event_ids": [left.get("event_id"), right.get("event_id")],
                        "start_frame": start,
                        "end_frame_exclusive": end,
                    }
                )
    return sorted(
        result,
        key=lambda item: (
            item["start_frame"],
            item["end_frame_exclusive"],
            tuple(str(value) for value in item["event_ids"]),
        ),
    )


def _relationship_errors(
    value: Any,
    source_ids: Sequence[str],
    events_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[str], list[Mapping[str, Any]]]:
    relationships = _mapping_list(value)
    errors: list[str] = []
    expected_pairs = {tuple(pair) for pair in combinations(source_ids, 2)}
    observed_pairs: list[tuple[str, str]] = []
    relationship_ids = [item.get("relationship_id") for item in relationships]
    if len(relationship_ids) != len(set(relationship_ids)):
        errors.append("relationships.relationship_id values must be unique")

    for index, relationship in enumerate(relationships):
        owner = f"relationships[{index}]"
        raw_ids = relationship.get("source_ids")
        pair = tuple(raw_ids) if isinstance(raw_ids, list) else ()
        if len(pair) != 2 or not all(isinstance(item, str) for item in pair):
            errors.append(f"{owner}.source_ids must contain two source IDs")
            continue
        if pair != tuple(sorted(pair)):
            errors.append(f"{owner}.source_ids must use canonical bytewise order")
        typed_pair = (str(pair[0]), str(pair[1]))
        observed_pairs.append(typed_pair)
        if typed_pair not in expected_pairs:
            errors.append(
                f"{owner}.source_ids do not resolve to one declared source pair"
            )
        expected_overlaps = _expected_overlaps(typed_pair, events_by_source)
        actual_overlaps = relationship.get("event_overlap_windows")
        if actual_overlaps != expected_overlaps:
            errors.append(f"{owner}.event_overlap_windows do not match source events")

        flags = relationship.get("flags")
        if isinstance(flags, Mapping):
            errors.extend(
                _assessment_errors(
                    flags.get("sources_pass_each_other"),
                    owner=f"{owner}.flags.sources_pass_each_other",
                    expected_scope="pairwise",
                )
            )

    if set(observed_pairs) != expected_pairs or len(observed_pairs) != len(
        expected_pairs
    ):
        errors.append(
            "relationships must contain exactly one entry for every source pair"
        )
    return errors, relationships


def _aggregate_status(statuses: Sequence[str | None], mode: str) -> str | None:
    if not statuses or any(status not in _STATUS_VALUE for status in statuses):
        return None
    if mode == "or":
        if "present" in statuses:
            return "present"
        if all(status == "absent" for status in statuses):
            return "absent"
        return "not_evaluated"
    if "absent" in statuses:
        return "absent"
    if all(status == "present" for status in statuses):
        return "present"
    return "not_evaluated"


def _clip_flag_errors(
    value: Any,
    sources: Sequence[Mapping[str, Any]],
    relationships: Sequence[Mapping[str, Any]],
) -> list[str]:
    if not isinstance(value, Mapping):
        return ["clip_flags must be a mapping"]
    errors: list[str] = []
    for flag_id in ALL_FLAG_IDS:
        assessment = value.get(flag_id)
        owner = f"clip_flags.{flag_id}"
        errors.extend(
            _assessment_errors(
                assessment,
                owner=owner,
                expected_scope="clip",
            )
        )
        if flag_id in PAIR_FLAG_IDS:
            statuses = [
                _status(
                    relationship.get("flags", {}).get(flag_id)
                    if isinstance(relationship.get("flags"), Mapping)
                    else None
                )
                for relationship in relationships
            ]
        else:
            statuses = [
                _status(
                    source.get("flags", {}).get(flag_id)
                    if isinstance(source.get("flags"), Mapping)
                    else None
                )
                for source in sources
            ]
        mode = "or" if flag_id in OR_AGGREGATED_FLAGS else "and"
        expected = _aggregate_status(statuses, mode)
        if expected is not None and isinstance(assessment, Mapping):
            if assessment.get("status") != expected:
                errors.append(
                    f"{owner}.status must equal the legacy {mode.upper()} aggregate"
                )
            if assessment.get("value") is not _STATUS_VALUE[expected]:
                errors.append(f"{owner}.value must equal the aggregate status value")
    return errors


def _clip_timing_errors(clip: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    frame_count = int(clip["frame_count"])
    ticks_per_frame = int(clip["ticks_per_frame"])
    duration_ticks = int(clip["duration_ticks"])
    time_base_hz = int(clip["time_base_hz"])
    fps_num = int(clip["fps_num"])
    fps_den = int(clip["fps_den"])
    sample_rate_hz = int(clip["sample_rate_hz"])
    ticks_per_sample = int(clip["ticks_per_sample"])
    sample_count = int(clip["sample_count"])
    if duration_ticks != frame_count * ticks_per_frame:
        errors.append("clip.duration_ticks must equal frame_count * ticks_per_frame")
    if time_base_hz * fps_den != ticks_per_frame * fps_num:
        errors.append("clip frame rate is inconsistent with the tick timebase")
    if time_base_hz != sample_rate_hz * ticks_per_sample:
        errors.append("clip sample rate is inconsistent with the tick timebase")
    if duration_ticks != sample_count * ticks_per_sample:
        errors.append("clip.sample_count must span the exact clip duration")
    if sample_boundary(
        frame_count,
        ticks_per_frame=ticks_per_frame,
        ticks_per_sample=ticks_per_sample,
    ) != sample_count:
        errors.append("clip final frame/sample boundaries must coincide")
    return errors


def _observer_errors(observer: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    path = _repository_uri_path(observer.get("authority_uri"))
    if path is None or not path.is_file():
        return ["observer.authority_uri must resolve to a repository file"]
    if observer.get("authority_file_sha256") != sha256_file(path):
        errors.append("observer.authority_file_sha256 differs from the route file")
    route = load_json(path)
    camera = route.get("camera") if isinstance(route, Mapping) else None
    if not isinstance(camera, Mapping):
        return errors + ["observer authority has no camera record"]
    if observer.get("position_m") != camera.get("habitat_position_m"):
        errors.append("observer.position_m differs from route camera")
    if observer.get("yaw_deg") != camera.get("habitat_yaw_deg"):
        errors.append("observer.yaw_deg differs from route camera")
    if observer.get("horizontal_fov_deg") != camera.get("horizontal_fov_deg"):
        errors.append("observer.horizontal_fov_deg differs from route camera")
    return errors


def _expected_assessment_error(
    assessment: Any, *, expected: bool, owner: str
) -> list[str]:
    if not isinstance(assessment, Mapping):
        return [f"{owner} must be an assessment mapping"]
    expected_status = "present" if expected else "absent"
    errors: list[str] = []
    if assessment.get("status") != expected_status:
        errors.append(f"{owner}.status differs from trajectory recomputation")
    if assessment.get("value") is not expected:
        errors.append(f"{owner}.value differs from trajectory recomputation")
    evidence = _mapping_list(assessment.get("evidence"))
    if not any(isinstance(item.get("measurement"), Mapping) for item in evidence):
        errors.append(f"{owner} requires quantitative measurement evidence")
    return errors


def _trajectory_flag_errors(
    sources: Sequence[Mapping[str, Any]],
    relationships: Sequence[Mapping[str, Any]],
    observer: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    *,
    fps: float,
) -> list[str]:
    errors: list[str] = []
    observer_position = tuple(float(item) for item in observer["position_m"])
    right_xz = habitat_basis_from_yaw_degrees(observer["yaw_deg"]).right_xz
    positions_by_source: dict[str, list[tuple[float, float, float]]] = {}
    for index, source in enumerate(sources):
        positions = [
            tuple(float(item) for item in keyframe["position_m"])
            for keyframe in source["trajectory"]["keyframes"]
        ]
        source_id = str(source["source_id"])
        positions_by_source[source_id] = positions
        distances = [math.dist(position, observer_position) for position in positions]
        speeds = [
            math.dist(left, right) * fps for left, right in zip(positions, positions[1:])
        ]
        mean_speed = sum(speeds) / len(speeds)
        speed_std = math.sqrt(
            sum((speed - mean_speed) ** 2 for speed in speeds) / len(speeds)
        )
        speed_cv = speed_std / max(mean_speed, 1.0e-6)
        lateral = [
            (position[0] - observer_position[0]) * right_xz[0]
            + (position[2] - observer_position[2]) * right_xz[1]
            for position in positions
        ]
        stop_threshold = float(thresholds["stop_and_go_stop_speed_mps"])
        stopped = sum(speed < stop_threshold for speed in speeds)
        moving = sum(speed >= stop_threshold for speed in speeds)
        expected = {
            "crosses_azimuth_zero": min(lateral) < 0.0 < max(lateral),
            "passes_close_to_mic": min(distances)
            < float(thresholds["passes_close_to_mic_m"]),
            "far_from_mic_whole_clip": min(distances)
            > float(thresholds["far_from_mic_whole_clip_m"]),
            "stationary": mean_speed
            < float(thresholds["stationary_mean_speed_mps"]),
            "steady_walk": len(speeds) >= 3
            and mean_speed >= float(thresholds["steady_walk_min_mean_speed_mps"])
            and speed_cv < float(thresholds["steady_walk_max_speed_cv"]),
            "stop_and_go": stopped
            >= int(thresholds["stop_and_go_min_stopped_frames"])
            and moving >= int(thresholds["stop_and_go_min_moving_frames"]),
        }
        flags = source["flags"]
        for flag_id, expected_value in expected.items():
            errors.extend(
                _expected_assessment_error(
                    flags.get(flag_id),
                    expected=expected_value,
                    owner=f"sources[{index}].flags.{flag_id}",
                )
            )

    for index, relationship in enumerate(relationships):
        left_id, right_id = relationship["source_ids"]
        left = positions_by_source[left_id]
        right = positions_by_source[right_id]
        minimum = min(
            math.hypot(a[0] - b[0], a[2] - b[2]) for a, b in zip(left, right)
        )
        expected = minimum < float(thresholds["sources_pass_each_other_m"])
        errors.extend(
            _expected_assessment_error(
                relationship["flags"].get("sources_pass_each_other"),
                expected=expected,
                owner=f"relationships[{index}].flags.sources_pass_each_other",
            )
        )
    return errors


def validate_source_manifest(value: Any) -> list[str]:
    """Return all schema and cross-field errors without mutating ``value``."""

    errors = _schema_errors(value)
    if not isinstance(value, Mapping):
        return list(dict.fromkeys(errors + ["manifest must be a mapping"]))
    if not _all_numbers_finite(value):
        errors.append("manifest numbers must all be finite")
    # Cross-field checks assume the structural types guaranteed by the JSON
    # Schema.  Fail closed here instead of risking secondary exceptions or
    # misleading semantic errors for a malformed document.
    if errors:
        return list(dict.fromkeys(errors))

    declared = value.get("manifest_content_sha256")
    actual = _canonical_without(value, "manifest_content_sha256")
    if actual is None or declared != actual:
        errors.append("manifest_content_sha256 does not match canonical content")

    clip = value["clip"]
    errors.extend(_clip_timing_errors(clip))
    observer = value["observer"]
    errors.extend(_observer_errors(observer))
    sources = _mapping_list(value.get("sources"))
    source_ids = _stable_values(sources, "source_id", "sources", errors)
    _stable_values(sources, "actor_id", "sources", errors)
    _stable_values(sources, "asset_id", "sources", errors)
    emitters = [
        source.get("emitter")
        for source in sources
        if isinstance(source.get("emitter"), Mapping)
    ]
    _stable_values(emitters, "emitter_id", "sources.emitter", errors)
    if source_ids != sorted(source_ids):
        errors.append("sources must use canonical bytewise source_id order")

    events_by_source: dict[str, list[Mapping[str, Any]]] = {}
    all_event_ids: list[Any] = []
    for index, source in enumerate(sources):
        source_errors, events = _source_errors(source, index, clip)
        errors.extend(source_errors)
        source_id = source.get("source_id")
        if isinstance(source_id, str):
            events_by_source[source_id] = events
        all_event_ids.extend(event.get("event_id") for event in events)
    if len(all_event_ids) != len(set(all_event_ids)):
        errors.append("event_id values must be globally unique")

    errors.extend(
        _frame_state_errors(
            value.get("frame_event_state"), sources, events_by_source, clip
        )
    )
    relationship_errors, relationships = _relationship_errors(
        value.get("relationships"), source_ids, events_by_source
    )
    errors.extend(relationship_errors)
    errors.extend(
        _trajectory_flag_errors(
            sources,
            relationships,
            observer,
            value["legacy_flag_thresholds"],
            fps=float(clip["fps_num"]) / float(clip["fps_den"]),
        )
    )
    errors.extend(_clip_flag_errors(value.get("clip_flags"), sources, relationships))
    return list(dict.fromkeys(errors))


def bind_source_manifest_hashes(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached manifest with trajectory and outer hashes rebound."""

    result = deepcopy(dict(value))
    sources = result.get("sources")
    if isinstance(sources, list):
        for source in sources:
            if not isinstance(source, dict):
                continue
            trajectory = source.get("trajectory")
            if isinstance(trajectory, dict):
                trajectory.pop("trajectory_content_sha256", None)
                trajectory["trajectory_content_sha256"] = canonical_json_sha256(
                    trajectory
                )
                emitter = source.get("emitter")
                if isinstance(emitter, dict):
                    emitter["path_sha256"] = trajectory["trajectory_content_sha256"]
    result.pop("manifest_content_sha256", None)
    result["manifest_content_sha256"] = canonical_json_sha256(result)
    return result


def load_source_manifest(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate an M5.1 source manifest, failing closed."""

    value = load_json(Path(path))
    errors = validate_source_manifest(value)
    if errors:
        raise SourceContractError(errors)
    return value
